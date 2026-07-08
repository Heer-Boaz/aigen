from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    character_instruction_parser_config_json,
)
from aigen.character_reference_models import (
    CharacterReferenceError,
    CharacterReferencePackSpec,
    load_completed_character_reference_pack,
)
from aigen.character_reference_pack import parse_character_edit_plan
from aigen.character_task_route_models import CharacterTaskRoutePlanSpec
from aigen.character_task_router import CharacterTaskRouter, compact_vlm_planner_context
from aigen.generation.qwen_image_edit_identity import (
    QwenIdentityCase,
    QwenImageEditIdentityProfile,
    run_qwen_image_edit_cases,
)
from aigen.manifest_io import read_json, resolve_existing_path, write_json
from aigen.progress import StatusReporter
from aigen.text_llm import TextLlmError
from aigen.vlm_qwen import QwenVlm, QwenVlmConfig, qwen_vlm_config_json


class QwenCharacterEditError(RuntimeError):
    pass


@dataclass(frozen=True)
class QwenCharacterEditCaseTemplate:
    name: str


@dataclass(frozen=True)
class PlannedQwenCharacterEdit:
    reference_paths: dict[str, Path]
    edit_cases: tuple[QwenIdentityCase, ...]
    manifest: dict[str, Any]


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


def plan_qwen_character_edit(
    *,
    pack_path: Path,
    instruction_parser_config: CharacterInstructionParserConfig,
    vlm_config: QwenVlmConfig,
    cases: Sequence[str],
    instruction: str | None,
    candidates_per_case: int,
    progress: StatusReporter,
) -> dict[str, Any]:
    return {
        "status": "planned",
        **_build_qwen_character_edit_plan(
            pack_path=pack_path,
            instruction_parser_config=instruction_parser_config,
            vlm_config=vlm_config,
            cases=cases,
            instruction=instruction,
            candidates_per_case=candidates_per_case,
            progress=progress,
        ).manifest,
    }


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
        result_kind="qwen-character-edit-result",
        manifest_context=planned.manifest,
        progress=progress,
    )
    edit_plan_path = output_dir / "edit_plan.json"
    write_json(edit_plan_path, {"status": "completed", **planned.manifest})
    result["output"]["edit_plan"] = edit_plan_path.as_posix()
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
    progress: StatusReporter,
) -> PlannedQwenCharacterEdit:
    if candidates_per_case < 1:
        raise QwenCharacterEditError("candidates_per_case must be at least 1")
    context = load_qwen_character_reference_context(
        pack_path=pack_path,
        progress=progress,
        phase="load qwen character edit reference pack",
    )
    templates = _selected_templates(cases)
    progress.phase("parse qwen character user instruction")
    instruction_parser = CharacterInstructionParser(instruction_parser_config)
    progress.phase("plan qwen character edit with VLM")
    with closing(QwenVlm(vlm_config)) as runner:
        edit_cases = tuple(
            _planned_case(
                template=template,
                context=context,
                instruction=instruction,
                instruction_parser=instruction_parser,
                runner=runner,
            )
            for template in templates
        )
        edit_planner = qwen_vlm_config_json(vlm_config) | {"device_report": runner.device_report}
    _validate_planned_references(context.pack, edit_cases)
    return PlannedQwenCharacterEdit(
        reference_paths=context.reference_paths,
        edit_cases=edit_cases,
        manifest=_plan_manifest(
            pack_path=context.pack_path,
            pack=context.pack,
            edit_cases=edit_cases,
            candidates_per_case=candidates_per_case,
            instruction=instruction,
            instruction_parser=character_instruction_parser_config_json(instruction_parser.config),
            edit_planner=edit_planner,
        ),
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


def _planned_case(
    *,
    template: QwenCharacterEditCaseTemplate,
    context: QwenCharacterReferenceContext,
    instruction: str | None,
    instruction_parser: CharacterInstructionParser,
    runner: QwenVlm,
) -> QwenIdentityCase:
    instruction_request = _instruction_request(template, instruction)
    try:
        instruction_plan = instruction_parser.parse(
            _instruction_envelope(
                template=template,
                context=context,
                instruction_request=instruction_request,
            )
        )
    except (CharacterInstructionError, TextLlmError) as error:
        raise QwenCharacterEditError(str(error)) from error
    task_route_plan = CharacterTaskRouter().route(instruction_plan)
    planner_context = compact_vlm_planner_context(instruction_plan, task_route_plan)
    try:
        planned = parse_character_edit_plan(
            runner=runner,
            pack=context.pack,
            reference_paths=context.reference_paths,
            case_name=template.name,
            user_instruction=instruction_request,
            planner_context=planner_context,
            path_label=f"{context.pack_path.as_posix()}#{template.name}",
        )
    except CharacterReferenceError as error:
        raise QwenCharacterEditError(str(error)) from error
    conditioning_plan = CharacterConditioningPlanner().plan(
        instruction_plan=instruction_plan,
        task_route_plan=task_route_plan,
        visual_analysis=planned.visual_analysis,
    )
    normalized_instruction = _normalized_instruction(
        template=template,
        reference_ids=planned.selected_refs,
        source_instruction=instruction,
        instruction_request=instruction_request,
        instruction_plan=instruction_plan,
        task_route_plan=task_route_plan,
        edit_instruction=planned.edit_instruction,
        edit_planner_raw_response=planned.raw_text,
        selected_planner_refs=planned.selected_planner_refs,
        planner_reference_map=planned.planner_reference_map,
        reference_semantics=planned.reference_semantics,
        visual_analysis=planned.visual_analysis,
        planner_context=planner_context,
        conditioning_plan=conditioning_plan,
    )
    return QwenIdentityCase(
        name=template.name,
        references=planned.selected_refs,
        prompt=planned.edit_instruction,
        normalized_instruction=normalized_instruction,
    )


def _instruction_request(template: QwenCharacterEditCaseTemplate, instruction: str | None) -> str:
    if instruction is not None and instruction.strip():
        return instruction.strip()
    return template.name


def _instruction_envelope(
    *,
    template: QwenCharacterEditCaseTemplate,
    context: QwenCharacterReferenceContext,
    instruction_request: str,
) -> InstructionEnvelopeSpec:
    return InstructionEnvelopeSpec(
        raw_instruction=instruction_request,
        ui_mode="reference_conditioned_generation",
        reference_count=len(context.pack.references),
        source_image_present=False,
        mask_present=False,
        region_plan_present=False,
        generation_panel_settings={"case": template.name},
        requested_model_family="qwen-image-edit",
        negative_prompt_present=False,
        aspect_ratio_setting=None,
        seed_setting=None,
    )


def _normalized_instruction(
    *,
    template: QwenCharacterEditCaseTemplate,
    reference_ids: Sequence[str],
    source_instruction: str | None,
    instruction_request: str,
    instruction_plan: CharacterInstructionPlanSpec,
    task_route_plan: CharacterTaskRoutePlanSpec,
    edit_instruction: str,
    edit_planner_raw_response: str,
    selected_planner_refs: Sequence[str],
    planner_reference_map: Mapping[str, str],
    reference_semantics: Mapping[str, str],
    visual_analysis: Mapping[str, Any],
    planner_context: Mapping[str, Any],
    conditioning_plan: CharacterConditioningPlanSpec,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "task": "identity_edit",
        "case": template.name,
        "planner_input_refs": list(planner_reference_map),
        "selected_planner_refs": list(selected_planner_refs),
        "planner_reference_map": dict(planner_reference_map),
        "reference_semantics": dict(reference_semantics),
        "refs_used": list(reference_ids),
        "source_instruction": source_instruction,
        "instruction_request": instruction_request,
        "instruction_plan": instruction_plan.model_dump(mode="json"),
        "task_route_plan": task_route_plan.model_dump(mode="json"),
        "planner_context": dict(planner_context),
        "visual_analysis": dict(visual_analysis),
        "conditioning_plan": conditioning_plan.model_dump(mode="json"),
        "edit_instruction_source": "qwen_vlm_edit_planner",
        "edit_instruction": edit_instruction,
        "edit_planner_raw_response": edit_planner_raw_response,
        "identity_profile_used": False,
    }
    return normalized


def _reference_paths(pack: CharacterReferencePackSpec, pack_path: Path) -> dict[str, Path]:
    return {
        name: resolve_existing_path(asset.path, pack_path.parent)
        for name, asset in pack.references.items()
    }


def _plan_manifest(
    *,
    pack_path: Path,
    pack: CharacterReferencePackSpec,
    edit_cases: Sequence[QwenIdentityCase],
    candidates_per_case: int,
    instruction: str | None,
    instruction_parser: Mapping[str, Any],
    edit_planner: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "qwen-character-edit-plan",
        "character_id": pack.character_id,
        "reference_pack": pack_path.as_posix(),
        "source_instruction": instruction,
        "normalizer": "text_instruction_parser_plus_qwen_vlm_edit_planner_v1",
        "instruction_parser": dict(instruction_parser),
        "edit_planner": dict(edit_planner),
        "reference_selector": "qwen_vlm_selected_refs_v1",
        "candidates_per_case": candidates_per_case,
        "cases": [
            {
                "name": case.name,
                "references": list(case.references),
                "refs_used": list(case.references),
                "identity_profile_used": False,
                "normalized_instruction": case.normalized_instruction,
                "prompt": case.prompt,
            }
            for case in edit_cases
        ],
    }
