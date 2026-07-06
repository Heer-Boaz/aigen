from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.character_reference_models import (
    CharacterIdentityProfileSpec,
    CharacterReferenceError,
    CharacterReferencePackSpec,
    load_completed_character_identity_profile,
    load_completed_character_reference_pack,
)
from aigen.character_reference_pack import parse_character_edit_plan
from aigen.generation.qwen_image_edit_identity import (
    QwenIdentityCase,
    QwenImageEditIdentityProfile,
    run_qwen_image_edit_cases,
)
from aigen.manifest_io import read_json, resolve_existing_path, write_json
from aigen.progress import StatusReporter
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
    identity_profile_path: Path
    pack: CharacterReferencePackSpec
    identity_profile: CharacterIdentityProfileSpec
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
    identity_profile_path: Path | None,
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
            identity_profile_path=identity_profile_path,
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
    identity_profile_path: Path | None,
    output_dir: Path,
    profile: QwenImageEditIdentityProfile,
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
        identity_profile_path=identity_profile_path,
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
    identity_profile_path: Path | None,
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
        identity_profile_path=identity_profile_path,
        progress=progress,
        phase="load qwen character edit reference pack",
    )
    templates = _selected_templates(cases)
    progress.phase("plan qwen character edit with VLM")
    with closing(QwenVlm(vlm_config)) as runner:
        edit_cases = tuple(
            _planned_case(
                template=template,
                context=context,
                instruction=instruction,
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
            identity_profile_path=context.identity_profile_path,
            pack=context.pack,
            identity_profile=context.identity_profile,
            edit_cases=edit_cases,
            candidates_per_case=candidates_per_case,
            instruction=instruction,
            edit_planner=edit_planner,
        ),
    )


def load_qwen_character_reference_context(
    *,
    pack_path: Path,
    identity_profile_path: Path | None,
    progress: StatusReporter,
    phase: str,
) -> QwenCharacterReferenceContext:
    progress.phase(phase)
    resolved_pack_path = pack_path.resolve()
    pack = _load_reference_pack(resolved_pack_path)
    resolved_identity_profile_path = _identity_profile_path(resolved_pack_path, identity_profile_path)
    identity_profile = _load_identity_profile(resolved_identity_profile_path)
    _validate_identity_profile(pack, identity_profile, resolved_identity_profile_path)
    return QwenCharacterReferenceContext(
        pack_path=resolved_pack_path,
        identity_profile_path=resolved_identity_profile_path,
        pack=pack,
        identity_profile=identity_profile,
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


def _load_identity_profile(identity_profile_path: Path) -> CharacterIdentityProfileSpec:
    try:
        return load_completed_character_identity_profile(
            read_json(identity_profile_path, label="character identity profile"),
            path_label=identity_profile_path.as_posix(),
        )
    except CharacterReferenceError as error:
        raise QwenCharacterEditError(str(error)) from error


def _identity_profile_path(pack_path: Path, identity_profile_path: Path | None) -> Path:
    if identity_profile_path is not None:
        return identity_profile_path.resolve()
    return pack_path.parent / "identity_profile.json"


def _validate_identity_profile(
    pack: CharacterReferencePackSpec,
    identity_profile: CharacterIdentityProfileSpec,
    identity_profile_path: Path,
) -> None:
    if identity_profile.character_id != pack.character_id:
        raise QwenCharacterEditError(
            f"Identity profile {identity_profile_path.as_posix()} is for {identity_profile.character_id}, "
            f"expected {pack.character_id}"
        )
    missing_roles = sorted(name for name in pack.references if name not in identity_profile.reference_roles)
    unknown_roles = sorted(name for name in identity_profile.reference_roles if name not in pack.references)
    if missing_roles or unknown_roles:
        raise QwenCharacterEditError(
            f"Identity profile {identity_profile_path.as_posix()} reference_roles must match the reference pack"
        )
    unknown_evidence = sorted(
        name for name in identity_profile.body_proportion.evidence_refs if name not in pack.references
    )
    if unknown_evidence:
        raise QwenCharacterEditError(
            f"Identity profile {identity_profile_path.as_posix()} body_proportion evidence references unknown "
            f"image(s): {', '.join(unknown_evidence)}"
        )


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
    runner: QwenVlm,
) -> QwenIdentityCase:
    instruction_request = _instruction_request(template, instruction)
    try:
        planned = parse_character_edit_plan(
            runner=runner,
            pack=context.pack,
            reference_paths=context.reference_paths,
            identity_profile=context.identity_profile,
            case_name=template.name,
            user_instruction=instruction_request,
            path_label=f"{context.pack_path.as_posix()}#{template.name}",
        )
    except CharacterReferenceError as error:
        raise QwenCharacterEditError(str(error)) from error
    normalized_instruction = _normalized_instruction(
        template=template,
        planner_input_refs=context.pack.references,
        reference_ids=planned.selected_refs,
        identity_profile=context.identity_profile,
        source_instruction=instruction,
        instruction_request=instruction_request,
        edit_instruction=planned.edit_instruction,
        edit_planner_raw_response=planned.raw_text,
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


def _normalized_instruction(
    *,
    template: QwenCharacterEditCaseTemplate,
    planner_input_refs: Mapping[str, Any],
    reference_ids: Sequence[str],
    identity_profile: CharacterIdentityProfileSpec,
    source_instruction: str | None,
    instruction_request: str,
    edit_instruction: str,
    edit_planner_raw_response: str,
) -> dict[str, Any]:
    must_preserve = _identity_must_preserve(identity_profile)
    avoid = _identity_avoid(identity_profile)
    normalized: dict[str, Any] = {
        "task": "identity_edit",
        "case": template.name,
        "planner_input_refs": list(planner_input_refs),
        "refs_used": list(reference_ids),
        "source_instruction": source_instruction,
        "instruction_request": instruction_request,
        "edit_instruction_source": "qwen_vlm_edit_planner",
        "edit_instruction": edit_instruction,
        "edit_planner_raw_response": edit_planner_raw_response,
        "must_preserve": must_preserve,
        "avoid": avoid,
        "identity": dict(identity_profile.identity),
        "identity_profile_used": True,
        "body_proportion_source": identity_profile.body_proportion_source,
        "body_proportion": identity_profile.body_proportion.model_dump(mode="json"),
        "optional_missing_refs": identity_profile.optional_missing_refs,
        "reference_roles": {
            name: identity_profile.reference_roles[name]
            for name in reference_ids
        },
    }
    return normalized


def _identity_must_preserve(identity_profile: CharacterIdentityProfileSpec) -> list[str]:
    return list(dict.fromkeys(identity_profile.must_preserve + identity_profile.body_proportion.do_not_change))


def _identity_avoid(identity_profile: CharacterIdentityProfileSpec) -> list[str]:
    return list(dict.fromkeys(identity_profile.avoid))


def _reference_paths(pack: CharacterReferencePackSpec, pack_path: Path) -> dict[str, Path]:
    return {
        name: resolve_existing_path(asset.path, pack_path.parent)
        for name, asset in pack.references.items()
    }


def _plan_manifest(
    *,
    pack_path: Path,
    identity_profile_path: Path,
    pack: CharacterReferencePackSpec,
    identity_profile: CharacterIdentityProfileSpec,
    edit_cases: Sequence[QwenIdentityCase],
    candidates_per_case: int,
    instruction: str | None,
    edit_planner: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "qwen-character-edit-plan",
        "character_id": pack.character_id,
        "reference_pack": pack_path.as_posix(),
        "identity_profile": identity_profile_path.as_posix(),
        "source_instruction": instruction,
        "normalizer": "qwen_vlm_edit_planner_v1",
        "edit_planner": dict(edit_planner),
        "reference_selector": "qwen_vlm_selected_refs_v1",
        "candidates_per_case": candidates_per_case,
        "cases": [
            {
                "name": case.name,
                "references": list(case.references),
                "refs_used": list(case.references),
                "identity_profile_used": True,
                "body_proportion_source": identity_profile.body_proportion_source,
                "optional_missing_refs": identity_profile.optional_missing_refs,
                "body_proportion": identity_profile.body_proportion.model_dump(mode="json"),
                "normalized_instruction": case.normalized_instruction,
                "prompt": case.prompt,
            }
            for case in edit_cases
        ],
        "identity": identity_profile.identity,
        "body_proportion": identity_profile.body_proportion.model_dump(mode="json"),
        "body_proportion_source": identity_profile.body_proportion_source,
        "optional_missing_refs": identity_profile.optional_missing_refs,
        "must_preserve": identity_profile.must_preserve,
        "avoid": identity_profile.avoid,
    }
