from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.character_conditioning_models import CharacterConditioningPlanSpec
from aigen.character_conditioning_planner import CharacterConditioningPlanner
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
from aigen.character_reference_pack import (
    ReferenceSelection,
    select_reference_subset,
)
from aigen.character_task_route_models import CharacterTaskRoutePlanSpec
from aigen.character_task_router import CharacterTaskRouter
from aigen.generation.qwen_image_edit_identity import (
    QWEN_IDENTITY_CASES,
    QwenIdentityCase,
    QwenImageEditIdentityProfile,
    run_qwen_image_edit_cases,
)
from aigen.manifest_io import read_json, resolve_existing_path, write_json
from aigen.progress import StatusReporter
from aigen.text_llm import TextLlmError
from aigen.vlm_qwen import QwenVlm, QwenVlmConfig


class QwenCharacterEditError(RuntimeError):
    pass


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
    reference_paths: dict[str, Path]
    edit_cases: tuple[QwenIdentityCase, ...]


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
    output_format: str,
    resolution: str,
    overwrite: bool,
    nunchaku_blocks_on_gpu: int | None,
    progress: StatusReporter,
) -> dict[str, Any]:
    planned = _build_qwen_character_edit_plan(
        pack_path=pack_path,
        instruction_parser_config=instruction_parser_config,
        vlm_config=vlm_config,
        cases=cases,
        instruction=instruction,
        candidates_per_case=candidates_per_case,
        output_format=output_format,
        resolution=resolution,
        progress=progress,
    )
    output_dir = output_dir.resolve()
    result = run_qwen_image_edit_cases(
        references=planned.reference_paths,
        output_dir=output_dir,
        profile=profile,
        edit_cases=planned.edit_cases,
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
        reference_paths=context.reference_paths,
        edit_cases=edit_cases,
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
            )
        )
    except (CharacterInstructionError, TextLlmError) as error:
        raise QwenCharacterEditError(str(error)) from error
    task_route_plan = CharacterTaskRouter().route(instruction_plan)
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
        reference_count > parsed_case.task_route_plan.capability_registry.max_qwen_edit_refs
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
            route_plan=parsed_case.task_route_plan,
            path_label=f"{context.pack_path.as_posix()}#{parsed_case.template.name}",
        )
    except CharacterReferenceError as error:
        raise QwenCharacterEditError(str(error)) from error
    conditioning_plan = CharacterConditioningPlanner().plan(
        instruction_plan=parsed_case.instruction_plan,
        task_route_plan=parsed_case.task_route_plan,
    )
    prompt, portrait_canvas = _thin_prompt(parsed_case)
    edit_context = _edit_context(
        template=parsed_case.template,
        selection=selection,
        source_instruction=parsed_case.source_instruction,
        task_route_plan=parsed_case.task_route_plan,
        conditioning_plan=conditioning_plan,
    )
    return QwenIdentityCase(
        name=parsed_case.template.name,
        references=selection.selected_refs,
        prompt=prompt,
        portrait_canvas=portrait_canvas,
        edit_context=edit_context,
    )


def _thin_prompt(parsed_case: ParsedQwenCharacterEditCase) -> tuple[str, bool]:
    fallback = QWEN_IDENTITY_CASES[parsed_case.template.name]
    source_instruction = parsed_case.source_instruction
    if source_instruction is not None and source_instruction.strip():
        return source_instruction.strip(), fallback.portrait_canvas
    return fallback.prompt, fallback.portrait_canvas


def _instruction_request(template: QwenCharacterEditCaseTemplate, instruction: str | None) -> str:
    if instruction is not None and instruction.strip():
        return instruction.strip()
    return template.name


def _instruction_envelope(
    *,
    template: QwenCharacterEditCaseTemplate,
    context: QwenCharacterReferenceContext,
    instruction_request: str,
    output_format: str | None,
    resolution: str | None,
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
        source_image_present=False,
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
    conditioning_plan: CharacterConditioningPlanSpec,
) -> dict[str, Any]:
    used_user_instruction = bool(source_instruction and source_instruction.strip())
    return {
        "task": "identity_edit",
        "case": template.name,
        "reference_selector": selection.selector,
        "selected_planner_refs": list(selection.selected_planner_refs),
        "refs_used": list(selection.selected_refs),
        "prompt_source": "user_instruction" if used_user_instruction else "generic_case_prompt",
        "route_kind": task_route_plan.route_kind,
        "output_mode": task_route_plan.output_mode,
        "editor_route": task_route_plan.editor_route,
        "conditioning_status": conditioning_plan.status,
        "conditioning_modes": list(conditioning_plan.conditioning_modes),
        "conditioning_deferred_to": conditioning_plan.deferred_to,
        "identity_profile_used": False,
    }


def _reference_paths(pack: CharacterReferencePackSpec, pack_path: Path) -> dict[str, Path]:
    return {
        name: resolve_existing_path(asset.path, pack_path.parent)
        for name, asset in pack.references.items()
    }
