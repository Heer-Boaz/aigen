from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

from aigen.canny_control import CannyControl, CannyControlError, render_canny_control
from aigen.character_conditioning_models import CharacterConditioningPlanError, CharacterConditioningPlanSpec
from aigen.character_conditioning_planner import CharacterConditioningPlanner
from aigen.character_edit_plan_models import (
    CHARACTER_EDIT_PLAN_KIND,
    CharacterEditPlanError,
    CharacterEditPlanSpec,
    load_character_edit_plan,
)
from aigen.character_instruction_models import (
    CharacterInstructionError,
    CharacterInstructionPlanSpec,
    InstructionEnvelopeSpec,
)
from aigen.character_instruction_parser import (
    CharacterInstructionParser,
    CharacterInstructionParserConfig,
)
from aigen.character_reference_models import (
    CharacterReferenceError,
    CharacterReferencePackSpec,
    load_completed_character_reference_pack,
)
from aigen.character_reference_selector import (
    ReferenceSelection,
    select_reference_subset,
)
from aigen.character_task_route_models import CharacterTaskRoutePlanSpec
from aigen.character_task_router import CharacterTaskRouter
from aigen.depth_v2_control import DepthV2Control, DepthV2ControlError, render_depth_v2_control
from aigen.dwpose_control import DWPoseControl, DWPoseControlError, render_dwpose_control
from aigen.generation.qwen_image_edit_identity import (
    QWEN_IDENTITY_CASES,
    QwenControlImage,
    QwenIdentityCase,
    QwenImageEditIdentityProfile,
    run_qwen_image_edit_cases,
)
from aigen.image_assets import image_asset_json
from aigen.manifest_io import read_json, resolve_existing_path, write_json
from aigen.progress import StatusReporter
from aigen.text_llm import TextLlmError
from aigen.vlm_qwen import QwenVlm, QwenVlmConfig


class QwenCharacterEditError(RuntimeError):
    pass


QWEN_POSE_CONTROL_NAME = "pose_keypoint"
QWEN_DEPTH_CONTROL_NAME = "depth"
QWEN_EDGE_CONTROL_NAME = "edge"
QWEN_STRUCTURE_CONTROL_NAMES = ("depth", "edge")
QWEN_STRUCTURE_MODE_BY_CONTROL = {
    "depth": "depth",
    "edge": "edge_or_sketch",
}
QWEN_CONTROL_NAME_BY_MODE = {
    "pose_keypoint": QWEN_POSE_CONTROL_NAME,
    "depth": QWEN_DEPTH_CONTROL_NAME,
    "edge_or_sketch": QWEN_EDGE_CONTROL_NAME,
}


@dataclass(frozen=True)
class QwenCharacterEditCaseTemplate:
    name: str


@dataclass(frozen=True)
class ParsedQwenCharacterEditCase:
    template: QwenCharacterEditCaseTemplate
    source_instruction: str | None
    instruction_request: str
    instruction_plan: CharacterInstructionPlanSpec
    task_route_plan: CharacterTaskRoutePlanSpec


@dataclass(frozen=True)
class PlannedQwenCharacterEdit:
    context: QwenCharacterReferenceContext
    edit_cases: tuple[QwenIdentityCase, ...]

    @property
    def reference_paths(self) -> dict[str, Path]:
        return self.context.reference_paths


@dataclass(frozen=True)
class QwenCharacterReferenceContext:
    pack_path: Path
    pack: CharacterReferencePackSpec
    reference_paths: dict[str, Path]


QWEN_CHARACTER_EDIT_DEFAULT_CASES = ("front", "side", "right_profile", "back", "three_quarter", "portrait")

# Cases name requested outputs only. Qwen receives the reference pack plus the
# raw case/user intent; output gates judge whether the candidate is usable.
QWEN_CHARACTER_EDIT_CASES: dict[str, QwenCharacterEditCaseTemplate] = {
    "front": QwenCharacterEditCaseTemplate(
        name="front",
    ),
    "side": QwenCharacterEditCaseTemplate(
        name="side",
    ),
    "right_profile": QwenCharacterEditCaseTemplate(
        name="right_profile",
    ),
    "back": QwenCharacterEditCaseTemplate(
        name="back",
    ),
    "three_quarter": QwenCharacterEditCaseTemplate(
        name="three_quarter",
    ),
    "portrait": QwenCharacterEditCaseTemplate(
        name="portrait",
    ),
}

QWEN_CHARACTER_EDIT_CASE_ALIASES = {
    "right-profile": "right_profile",
    "three-quarter": "three_quarter",
}


def qwen_character_edit_case_names() -> tuple[str, ...]:
    return tuple(QWEN_CHARACTER_EDIT_CASES) + tuple(QWEN_CHARACTER_EDIT_CASE_ALIASES)


def run_qwen_character_edit(
    *,
    pack_path: Path,
    output_dir: Path,
    profile: QwenImageEditIdentityProfile,
    instruction_parser_config: CharacterInstructionParserConfig,
    vlm_config: QwenVlmConfig,
    cases: Sequence[str],
    instruction: str | None,
    max_side: int,
    steps: int | None,
    true_cfg_scale: float | None,
    guidance_scale: float | None,
    seed: int,
    max_sequence_length: int,
    candidates_per_case: int,
    output_format: str | None,
    resolution: str | None,
    overwrite: bool,
    nunchaku_blocks_on_gpu: int | None,
    plan_path: Path | None,
    pose_source_path: Path | None = None,
    structure_source_path: Path | None = None,
    structure_control: str | None = None,
    progress: StatusReporter,
) -> dict[str, Any]:
    if (structure_source_path is None) != (structure_control is None):
        raise QwenCharacterEditError("--structure-source and --structure-control must be provided together")
    if structure_control is not None and structure_control not in QWEN_STRUCTURE_MODE_BY_CONTROL:
        allowed = ", ".join(QWEN_STRUCTURE_CONTROL_NAMES)
        raise QwenCharacterEditError(f"Unknown structure control {structure_control}; expected one of: {allowed}")
    available_modes = []
    if pose_source_path is not None:
        available_modes.append("pose_keypoint")
    if structure_control is not None:
        available_modes.append(QWEN_STRUCTURE_MODE_BY_CONTROL[structure_control])
    if plan_path is not None:
        if instruction is not None or cases:
            raise QwenCharacterEditError("--plan replaces --instruction and --case; do not combine them")
        planned = _planned_edit_from_plan_file(
            pack_path=pack_path,
            plan_path=plan_path,
            progress=progress,
        )
    else:
        planned = _build_qwen_character_edit_plan(
            pack_path=pack_path,
            instruction_parser_config=instruction_parser_config,
            vlm_config=vlm_config,
            cases=cases,
            instruction=instruction,
            candidates_per_case=candidates_per_case,
            output_format=output_format,
            resolution=resolution,
            source_image_present=False,
            pose_source_present=pose_source_path is not None,
            progress=progress,
        )
    conditioning_plans = _plan_route_conditioning(
        planned.edit_cases,
        available_modes=available_modes,
    )
    pose_conditioning = (
        _render_pose_conditioning(pose_source_path, progress)
        if pose_source_path is not None
        else None
    )
    structure_conditioning = (
        _render_structure_conditioning(structure_source_path, structure_control, progress)
        if structure_source_path is not None and structure_control is not None
        else None
    )
    edit_cases = _apply_route_conditioning(conditioning_plans)
    controls: dict[str, QwenControlImage] = {}
    if pose_conditioning is not None:
        controls[QWEN_POSE_CONTROL_NAME] = QwenControlImage(
            image=pose_conditioning[1].image,
            content_box=pose_conditioning[1].content_box,
        )
    if structure_conditioning is not None:
        controls[QWEN_CONTROL_NAME_BY_MODE[QWEN_STRUCTURE_MODE_BY_CONTROL[structure_control]]] = QwenControlImage(
            image=structure_conditioning[1].image,
        )
    output_dir = output_dir.resolve()
    result = run_qwen_image_edit_cases(
        references=planned.reference_paths,
        controls=controls,
        output_dir=output_dir,
        profile=profile,
        edit_cases=edit_cases,
        max_side=max_side,
        steps=steps,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale,
        seed=seed,
        max_sequence_length=max_sequence_length,
        candidates_per_case=candidates_per_case,
        overwrite=overwrite,
        nunchaku_blocks_on_gpu=nunchaku_blocks_on_gpu,
        output_format=output_format,
        resolution=resolution,
        result_kind="qwen-character-edit-result",
        manifest_context=None,
        progress=progress,
    )
    conditioning_result: dict[str, Any] = {}
    if pose_conditioning is not None:
        source_path, pose_control = pose_conditioning
        conditioning_result[QWEN_POSE_CONTROL_NAME] = {
            "source_image": image_asset_json(source_path),
            "control_image": result["controls"][QWEN_POSE_CONTROL_NAME],
            "preprocessor": pose_control.metadata
            | {
                "tool": "dwpose_keypoint_map",
                "device": pose_control.device,
                "det_model": pose_control.det_model.as_posix(),
                "pose_model": pose_control.pose_model.as_posix(),
            },
        }
    if structure_conditioning is not None:
        source_path, structure_control_result = structure_conditioning
        control_name = QWEN_CONTROL_NAME_BY_MODE[QWEN_STRUCTURE_MODE_BY_CONTROL[structure_control]]
        preprocessor = structure_control_result.metadata | {
            "tool": "depth_anything_v2_map" if structure_control == "depth" else "canny_edge_map",
        }
        if isinstance(structure_control_result, DepthV2Control):
            preprocessor |= {
                "device": structure_control_result.device,
                "model": structure_control_result.model.as_posix(),
            }
        conditioning_result[control_name] = {
            "source_image": image_asset_json(source_path),
            "control_image": result["controls"][control_name],
            "preprocessor": preprocessor,
        }
    if conditioning_result:
        result["conditioning"] = conditioning_result
    write_json(output_dir / "result.json", result)
    return result


def _build_qwen_character_edit_plan(
    *,
    pack_path: Path,
    instruction_parser_config: CharacterInstructionParserConfig,
    vlm_config: QwenVlmConfig,
    cases: Sequence[str],
    instruction: str | None,
    candidates_per_case: int,
    output_format: str | None,
    resolution: str | None,
    source_image_present: bool,
    pose_source_present: bool,
    progress: StatusReporter,
) -> PlannedQwenCharacterEdit:
    if candidates_per_case < 1:
        raise QwenCharacterEditError("candidates_per_case must be at least 1")
    if (output_format is None) != (resolution is None):
        raise QwenCharacterEditError("output_format and resolution must be provided together")
    context = load_qwen_character_reference_context(
        pack_path=pack_path,
        progress=progress,
        phase="load qwen character edit reference pack",
    )
    templates = _selected_templates(cases)
    progress.phase("parse qwen character user instruction")
    instruction_parser = CharacterInstructionParser(instruction_parser_config)
    parsed_cases = tuple(
        _parsed_case(
            template=template,
            context=context,
            instruction=instruction,
            instruction_parser=instruction_parser,
            output_format=output_format,
            resolution=resolution,
            source_image_present=source_image_present,
            pose_source_present=pose_source_present,
        )
        for template in templates
    )
    needs_selection = _needs_reference_selection(context.pack, parsed_cases)
    if needs_selection:
        progress.phase("select qwen character edit references with VLM")
        with closing(QwenVlm(vlm_config)) as runner:
            edit_cases = tuple(
                _planned_case(parsed_case=parsed_case, context=context, runner=runner)
                for parsed_case in parsed_cases
            )
    else:
        progress.phase("select qwen character edit references")
        edit_cases = tuple(
            _planned_case(parsed_case=parsed_case, context=context, runner=None)
            for parsed_case in parsed_cases
        )
    _validate_planned_references(context.pack, edit_cases)
    return PlannedQwenCharacterEdit(
        context=context,
        edit_cases=edit_cases,
    )


def plan_qwen_character_edit(
    *,
    pack_path: Path,
    instruction_parser_config: CharacterInstructionParserConfig,
    vlm_config: QwenVlmConfig,
    cases: Sequence[str],
    instruction: str | None,
    output_path: Path,
    overwrite: bool,
    progress: StatusReporter,
) -> dict[str, Any]:
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise QwenCharacterEditError(f"Edit plan exists and overwrite=false: {output_path.as_posix()}")
    planned = _build_qwen_character_edit_plan(
        pack_path=pack_path,
        instruction_parser_config=instruction_parser_config,
        vlm_config=vlm_config,
        cases=cases,
        instruction=instruction,
        candidates_per_case=1,
        output_format=None,
        resolution=None,
        source_image_present=False,
        pose_source_present=False,
        progress=progress,
    )
    pack = planned.context.pack
    payload = {
        "status": "completed",
        "kind": CHARACTER_EDIT_PLAN_KIND,
        "character_id": pack.character_id,
        "reference_pack": planned.context.pack_path.as_posix(),
        "reference_sha256": {name: asset.sha256 for name, asset in pack.references.items()},
        "source_instruction": instruction,
        "cases": [
            {
                "name": case.name,
                "references": list(case.references),
                "prompt": case.prompt,
                "prompt_source": case.edit_context["prompt_source"],
                "portrait_canvas": case.portrait_canvas,
                "reference_selector": case.edit_context["reference_selector"],
                "route_kind": case.edit_context["route_kind"],
            }
            for case in planned.edit_cases
        ],
    }
    load_character_edit_plan(payload, path_label=output_path.as_posix())
    write_json(output_path, payload)
    return payload


def _planned_edit_from_plan_file(
    *,
    pack_path: Path,
    plan_path: Path,
    progress: StatusReporter,
) -> PlannedQwenCharacterEdit:
    plan_path = plan_path.resolve()
    context = load_qwen_character_reference_context(
        pack_path=pack_path,
        progress=progress,
        phase="load qwen character edit reference pack",
    )
    progress.phase("load qwen character edit plan")
    try:
        plan = load_character_edit_plan(
            read_json(plan_path, label="character edit plan"),
            path_label=plan_path.as_posix(),
        )
    except CharacterEditPlanError as error:
        raise QwenCharacterEditError(str(error)) from error
    _validate_plan_against_pack(plan, context=context, plan_path=plan_path)
    edit_cases = tuple(
        QwenIdentityCase(
            name=case.name,
            references=tuple(case.references),
            prompt=case.prompt,
            portrait_canvas=case.portrait_canvas,
            edit_context={
                "task": "identity_edit",
                "case": case.name,
                "reference_selector": case.reference_selector,
                "refs_used": list(case.references),
                "prompt_source": case.prompt_source,
                "route_kind": case.route_kind,
                "edit_plan": plan_path.as_posix(),
            },
        )
        for case in plan.cases
    )
    return PlannedQwenCharacterEdit(context=context, edit_cases=edit_cases)


def _validate_plan_against_pack(
    plan: CharacterEditPlanSpec,
    *,
    context: QwenCharacterReferenceContext,
    plan_path: Path,
) -> None:
    plan_label = plan_path.as_posix()
    pack_names = set(context.pack.references)
    plan_names = set(plan.reference_sha256)
    if pack_names != plan_names:
        raise QwenCharacterEditError(
            f"Edit plan {plan_label} is stale: pack references {sorted(pack_names)} do not match plan references "
            f"{sorted(plan_names)}; re-run qwen-edit-plan"
        )
    for name, planned_sha in plan.reference_sha256.items():
        current_sha = image_asset_json(context.reference_paths[name])["sha256"]
        if current_sha != planned_sha:
            raise QwenCharacterEditError(
                f"Edit plan {plan_label} is stale: reference {name!r} changed since planning; re-run qwen-edit-plan"
            )


def load_qwen_character_reference_context(
    *,
    pack_path: Path,
    progress: StatusReporter,
    phase: str,
) -> QwenCharacterReferenceContext:
    progress.phase(phase)
    resolved_pack_path = pack_path.resolve()
    pack = _load_reference_pack(resolved_pack_path)
    return QwenCharacterReferenceContext(
        pack_path=resolved_pack_path,
        pack=pack,
        reference_paths=_reference_paths(pack, resolved_pack_path),
    )


def _load_reference_pack(pack_path: Path) -> CharacterReferencePackSpec:
    try:
        return load_completed_character_reference_pack(
            read_json(pack_path, label="character reference pack"),
            path_label=pack_path.as_posix(),
        )
    except CharacterReferenceError as error:
        raise QwenCharacterEditError(str(error)) from error


def _validate_planned_references(pack: CharacterReferencePackSpec, edit_cases: Sequence[QwenIdentityCase]) -> None:
    missing = sorted(
        {
            reference_name
            for edit_case in edit_cases
            for reference_name in edit_case.references
            if reference_name not in pack.references
        }
    )
    if missing:
        raise QwenCharacterEditError(f"Reference pack is missing required reference(s): {', '.join(missing)}")


def _selected_templates(case_names: Sequence[str]) -> tuple[QwenCharacterEditCaseTemplate, ...]:
    if not case_names:
        case_names = QWEN_CHARACTER_EDIT_DEFAULT_CASES
    selected = []
    for case_name in case_names:
        normalized_case_name = QWEN_CHARACTER_EDIT_CASE_ALIASES.get(case_name, case_name)
        try:
            selected.append(QWEN_CHARACTER_EDIT_CASES[normalized_case_name])
        except KeyError as error:
            allowed = ", ".join(qwen_character_edit_case_names())
            raise QwenCharacterEditError(f"Unknown Qwen edit case {case_name}; expected one of: {allowed}") from error
    return tuple(selected)


def _parsed_case(
    *,
    template: QwenCharacterEditCaseTemplate,
    context: QwenCharacterReferenceContext,
    instruction: str | None,
    instruction_parser: CharacterInstructionParser,
    output_format: str | None,
    resolution: str | None,
    source_image_present: bool,
    pose_source_present: bool,
) -> ParsedQwenCharacterEditCase:
    instruction_request = _instruction_request(template, instruction)
    try:
        instruction_plan = instruction_parser.parse(
            _instruction_envelope(
                template=template,
                context=context,
                instruction_request=instruction_request,
                output_format=output_format,
                resolution=resolution,
                source_image_present=source_image_present,
            )
        )
    except (CharacterInstructionError, TextLlmError) as error:
        raise QwenCharacterEditError(str(error)) from error
    task_route_plan = CharacterTaskRouter().route(
        instruction_plan,
        pose_source_present=pose_source_present,
    )
    return ParsedQwenCharacterEditCase(
        template=template,
        source_instruction=instruction,
        instruction_request=instruction_request,
        instruction_plan=instruction_plan,
        task_route_plan=task_route_plan,
    )


def _needs_reference_selection(
    pack: CharacterReferencePackSpec,
    parsed_cases: Sequence[ParsedQwenCharacterEditCase],
) -> bool:
    reference_count = len(pack.references)
    return any(
        reference_count > parsed_case.task_route_plan.final_editor_constraints.reference_budget.max
        for parsed_case in parsed_cases
    )


def _planned_case(
    *,
    parsed_case: ParsedQwenCharacterEditCase,
    context: QwenCharacterReferenceContext,
    runner: QwenVlm | None,
) -> QwenIdentityCase:
    try:
        selection = select_reference_subset(
            runner=runner,
            pack=context.pack,
            reference_paths=context.reference_paths,
            instruction=parsed_case.task_route_plan.source_instruction,
            route_kind=parsed_case.task_route_plan.route_kind,
            output_mode=parsed_case.task_route_plan.output_mode,
            budget=parsed_case.task_route_plan.final_editor_constraints.reference_budget.max,
            path_label=f"{context.pack_path.as_posix()}#{parsed_case.template.name}",
        )
    except CharacterReferenceError as error:
        raise QwenCharacterEditError(str(error)) from error
    prompt, portrait_canvas = _thin_prompt(parsed_case)
    edit_context = _edit_context(
        template=parsed_case.template,
        selection=selection,
        source_instruction=parsed_case.source_instruction,
        task_route_plan=parsed_case.task_route_plan,
    )
    return QwenIdentityCase(
        name=parsed_case.template.name,
        references=selection.selected_refs,
        prompt=prompt,
        portrait_canvas=portrait_canvas,
        edit_context=edit_context,
    )


def _plan_route_conditioning(
    edit_cases: Sequence[QwenIdentityCase],
    *,
    available_modes: Sequence[str],
) -> tuple[tuple[QwenIdentityCase, CharacterConditioningPlanSpec], ...]:
    planner = CharacterConditioningPlanner()

    planned_cases = []
    for case in edit_cases:
        route_kind = _case_route_kind(case)
        try:
            conditioning_plan = planner.plan(
                route_kind=route_kind,
                available_modes=available_modes,
            )
        except CharacterConditioningPlanError as error:
            if route_kind == "pose_transfer":
                raise QwenCharacterEditError(
                    f"Qwen edit case {case.name} uses pose_transfer and requires --pose-source with a source image"
                ) from error
            if route_kind == "local_repair_or_inpaint":
                raise QwenCharacterEditError(
                    f"Qwen edit case {case.name} uses local_repair_or_inpaint and requires a source image and region mask"
                ) from error
            raise QwenCharacterEditError(f"Qwen edit case {case.name}: {error}") from error
        planned_cases.append((case, conditioning_plan))
    return tuple(planned_cases)


def _apply_route_conditioning(
    planned_cases: Sequence[tuple[QwenIdentityCase, CharacterConditioningPlanSpec]],
) -> tuple[QwenIdentityCase, ...]:
    conditioned_cases = []
    for case, conditioning_plan in planned_cases:
        modes = list(conditioning_plan.conditioning_modes)
        case_controls = tuple(QWEN_CONTROL_NAME_BY_MODE[mode] for mode in modes)
        if case.edit_context is None:
            raise QwenCharacterEditError(f"Character edit case {case.name} has no route context")
        conditioned_cases.append(
            replace(
                case,
                controls=case_controls,
                edit_context=case.edit_context
                | {
                    "conditioning_modes": modes,
                    "conditioning_tools": list(conditioning_plan.planned_tools),
                    "controls_used": list(case_controls),
                },
            )
        )
    return tuple(conditioned_cases)


def _render_pose_conditioning(
    pose_source_path: Path,
    progress: StatusReporter,
) -> tuple[Path, DWPoseControl]:
    source_path = resolve_existing_path(pose_source_path.as_posix(), Path.cwd())
    progress.phase("build DWPose keypoint control")
    try:
        with Image.open(source_path) as source_image:
            control = render_dwpose_control(
                source_image,
                source_label=source_path.as_posix(),
            )
    except (OSError, DWPoseControlError) as error:
        raise QwenCharacterEditError(str(error)) from error
    return source_path, control


def _render_structure_conditioning(
    structure_source_path: Path,
    structure_control: str,
    progress: StatusReporter,
) -> tuple[Path, DepthV2Control | CannyControl]:
    source_path = resolve_existing_path(structure_source_path.as_posix(), Path.cwd())
    progress.phase(f"build {structure_control} scene control")
    try:
        with Image.open(source_path) as source_image:
            if structure_control == "depth":
                control = render_depth_v2_control(
                    source_image,
                    source_label=source_path.as_posix(),
                )
            elif structure_control == "edge":
                control = render_canny_control(
                    source_image,
                    source_label=source_path.as_posix(),
                )
            else:
                raise QwenCharacterEditError(f"Unknown structure control {structure_control}")
    except (OSError, DepthV2ControlError, CannyControlError) as error:
        raise QwenCharacterEditError(str(error)) from error
    return source_path, control


def _case_route_kind(case: QwenIdentityCase) -> str:
    if case.edit_context is None:
        raise QwenCharacterEditError(f"Character edit case {case.name} has no route context")
    return str(case.edit_context["route_kind"])


def _thin_prompt(parsed_case: ParsedQwenCharacterEditCase) -> tuple[str, bool]:
    fallback = QWEN_IDENTITY_CASES[parsed_case.template.name]
    source_instruction = parsed_case.source_instruction
    if source_instruction is not None and source_instruction.strip():
        return source_instruction.strip(), fallback.portrait_canvas
    return fallback.prompt, fallback.portrait_canvas


def _instruction_request(template: QwenCharacterEditCaseTemplate, instruction: str | None) -> str:
    if instruction is not None and instruction.strip():
        return instruction.strip()
    # Without a user instruction the generic case prompt IS the instruction;
    # feeding the parser the bare case name breaks routing and ref selection.
    return QWEN_IDENTITY_CASES[template.name].prompt


def _instruction_envelope(
    *,
    template: QwenCharacterEditCaseTemplate,
    context: QwenCharacterReferenceContext,
    instruction_request: str,
    output_format: str | None,
    resolution: str | None,
    source_image_present: bool,
) -> InstructionEnvelopeSpec:
    generation_panel_settings: dict[str, Any] = {"case": template.name}
    if output_format is not None:
        generation_panel_settings["output_format"] = output_format
    if resolution is not None:
        generation_panel_settings["resolution"] = resolution
    return InstructionEnvelopeSpec(
        raw_instruction=instruction_request,
        ui_mode="reference_conditioned_generation",
        reference_count=len(context.pack.references),
        source_image_present=source_image_present,
        mask_present=False,
        region_plan_present=False,
        generation_panel_settings=generation_panel_settings,
        requested_model_family="qwen-image-edit",
        negative_prompt_present=False,
        aspect_ratio_setting=output_format,
        seed_setting=None,
    )


def _edit_context(
    *,
    template: QwenCharacterEditCaseTemplate,
    selection: ReferenceSelection,
    source_instruction: str | None,
    task_route_plan: CharacterTaskRoutePlanSpec,
) -> dict[str, Any]:
    used_user_instruction = bool(source_instruction and source_instruction.strip())
    return {
        "task": "identity_edit",
        "case": template.name,
        "reference_selector": selection.selector,
        "selected_ref_indices": list(selection.selected_ref_indices),
        "refs_used": list(selection.selected_refs),
        "prompt_source": "user_instruction" if used_user_instruction else "generic_case_prompt",
        "route_kind": task_route_plan.route_kind,
        "output_mode": task_route_plan.output_mode,
        "editor_route": task_route_plan.editor_route,
    }


def _reference_paths(pack: CharacterReferencePackSpec, pack_path: Path) -> dict[str, Path]:
    return {
        name: resolve_existing_path(asset.path, pack_path.parent)
        for name, asset in pack.references.items()
    }
