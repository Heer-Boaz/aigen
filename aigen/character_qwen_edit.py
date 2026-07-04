from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from aigen.generation.qwen_image_edit_identity import (
    QwenIdentityCase,
    QwenImageEditIdentityProfile,
    run_qwen_image_edit_cases,
)
from aigen.manifest_io import read_json, resolve_existing_path, write_json
from aigen.progress import StatusReporter


class QwenCharacterEditError(RuntimeError):
    pass


@dataclass(frozen=True)
class QwenCharacterEditCaseTemplate:
    name: str
    references: tuple[str, ...]
    task: str
    requested_view: str
    requested_pose: str
    background: str
    portrait_canvas: bool = False


@dataclass(frozen=True)
class PlannedQwenCharacterEdit:
    reference_paths: dict[str, Path]
    edit_cases: tuple[QwenIdentityCase, ...]
    manifest: dict[str, Any]


QWEN_CHARACTER_EDIT_DEFAULT_CASES = ("front", "side", "right_profile", "back", "three_quarter", "portrait")

QWEN_CHARACTER_EDIT_CASES: dict[str, QwenCharacterEditCaseTemplate] = {
    "front": QwenCharacterEditCaseTemplate(
        name="front",
        references=("front", "portrait", "side"),
        task="Generate a clean full-body front view character reference image in a neutral standing pose.",
        requested_view="front",
        requested_pose="neutral standing",
        background="plain light studio background",
    ),
    "side": QwenCharacterEditCaseTemplate(
        name="side",
        references=("side", "portrait", "front"),
        task="Generate a clean full-body left side view character reference image in a neutral standing pose.",
        requested_view="left_side",
        requested_pose="neutral standing",
        background="plain light studio background",
    ),
    "right_profile": QwenCharacterEditCaseTemplate(
        name="right_profile",
        references=("side", "portrait", "front"),
        task="Generate a clean full-body right profile view character reference image in a neutral standing pose.",
        requested_view="right_profile",
        requested_pose="neutral standing",
        background="plain light studio background",
    ),
    "back": QwenCharacterEditCaseTemplate(
        name="back",
        references=("back", "portrait", "front"),
        task="Generate a clean full-body back view character reference image in a neutral standing pose.",
        requested_view="back",
        requested_pose="neutral standing",
        background="plain light studio background",
    ),
    "three_quarter": QwenCharacterEditCaseTemplate(
        name="three_quarter",
        references=("front", "side", "portrait"),
        task="Generate a clean full-body three-quarter front view character reference image in a neutral standing pose.",
        requested_view="three_quarter_front",
        requested_pose="neutral standing",
        background="plain light studio background",
    ),
    "portrait": QwenCharacterEditCaseTemplate(
        name="portrait",
        references=("portrait", "front", "side"),
        task="Generate a clean shoulders-up portrait of the same character.",
        requested_view="portrait",
        requested_pose="neutral expression",
        background="plain light studio background",
        portrait_canvas=True,
    ),
    "body_proportion": QwenCharacterEditCaseTemplate(
        name="body_proportion",
        references=("body_shape", "side", "portrait"),
        task="Generate a clean full-body neutral character reference focused on matching body proportions and silhouette.",
        requested_view="front_body_proportion",
        requested_pose="neutral standing",
        background="plain light studio background",
    ),
}

QWEN_CHARACTER_EDIT_CASE_ALIASES = {
    "right-profile": "right_profile",
    "three-quarter": "three_quarter",
    "body-proportion": "body_proportion",
}

IDENTITY_PRESERVE_FIELDS = {
    "hair": "hair",
    "eyes": "eyes",
    "neckwear": "neckwear",
    "top": "top",
    "bottom": "bottom",
    "legwear": "legwear",
    "footwear": "footwear",
    "body_shape": "body shape",
    "style": "art style",
}


def qwen_character_edit_case_names() -> tuple[str, ...]:
    return tuple(QWEN_CHARACTER_EDIT_CASES) + tuple(QWEN_CHARACTER_EDIT_CASE_ALIASES)


def plan_qwen_character_edit(
    *,
    pack_path: Path,
    identity_profile_path: Path | None,
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
    cases: Sequence[str],
    instruction: str | None,
    candidates_per_case: int,
    progress: StatusReporter,
) -> PlannedQwenCharacterEdit:
    if candidates_per_case < 1:
        raise QwenCharacterEditError("candidates_per_case must be at least 1")
    progress.phase("load qwen character edit reference pack")
    pack_path = pack_path.resolve()
    pack = _load_reference_pack(pack_path)
    resolved_identity_profile_path = _identity_profile_path(pack_path, identity_profile_path)
    identity_profile = _load_identity_profile(resolved_identity_profile_path)
    _validate_identity_profile(pack, identity_profile, resolved_identity_profile_path)
    templates = _selected_templates(cases)
    edit_cases = tuple(_planned_case(template, identity_profile, instruction) for template in templates)
    _validate_planned_references(pack, edit_cases)
    return PlannedQwenCharacterEdit(
        reference_paths=_reference_paths(pack, pack_path),
        edit_cases=edit_cases,
        manifest=_plan_manifest(
            pack_path=pack_path,
            identity_profile_path=resolved_identity_profile_path,
            pack=pack,
            identity_profile=identity_profile,
            edit_cases=edit_cases,
            candidates_per_case=candidates_per_case,
            instruction=instruction,
        ),
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
    template: QwenCharacterEditCaseTemplate,
    identity_profile: CharacterIdentityProfileSpec,
    instruction: str | None,
) -> QwenIdentityCase:
    normalized_instruction = _normalized_instruction(template, identity_profile, instruction)
    return QwenIdentityCase(
        name=template.name,
        references=template.references,
        prompt=_prompt_from_instruction(normalized_instruction),
        portrait_canvas=template.portrait_canvas,
        normalized_instruction=normalized_instruction,
    )


def _normalized_instruction(
    template: QwenCharacterEditCaseTemplate,
    identity_profile: CharacterIdentityProfileSpec,
    instruction: str | None,
) -> dict[str, Any]:
    must_preserve = _identity_must_preserve(identity_profile)
    avoid = _identity_avoid(identity_profile)
    normalized: dict[str, Any] = {
        "task": "identity_edit",
        "case": template.name,
        "task_prompt": template.task,
        "requested_view": template.requested_view,
        "requested_pose": template.requested_pose,
        "background": template.background,
        "source_instruction": instruction,
        "must_preserve": must_preserve,
        "avoid": avoid,
        "identity": dict(identity_profile.identity),
        "reference_roles": {
            name: identity_profile.reference_roles[name]
            for name in template.references
            if name in identity_profile.reference_roles
        },
    }
    return normalized


def _identity_must_preserve(identity_profile: CharacterIdentityProfileSpec) -> list[str]:
    identity_facts = [
        _identity_preserve_fact(field, label, identity_profile.identity[field])
        for field, label in IDENTITY_PRESERVE_FIELDS.items()
        if identity_profile.identity.get(field)
    ]
    return list(dict.fromkeys(identity_profile.must_preserve + identity_facts))


def _identity_avoid(identity_profile: CharacterIdentityProfileSpec) -> list[str]:
    return list(dict.fromkeys(identity_profile.avoid))


def _identity_preserve_fact(field: str, label: str, value: str) -> str:
    normalized_value = value.strip()
    if field in {"neckwear", "top", "bottom", "legwear", "footwear"}:
        return normalized_value
    if label in normalized_value.lower():
        return normalized_value
    if field == "hair":
        return f"{normalized_value} hairstyle"
    if field == "eyes":
        return f"{normalized_value} eyes"
    if field == "body_shape":
        return f"{normalized_value} body shape"
    if field == "style":
        return f"{normalized_value} art style"
    return f"{normalized_value} {label}"


def _prompt_from_instruction(instruction: Mapping[str, Any]) -> str:
    identity = instruction["identity"]
    identity_facts = "; ".join(f"{name}: {value}" for name, value in identity.items() if value)
    must_preserve = "; ".join(instruction["must_preserve"])
    avoid = "; ".join(instruction["avoid"])
    source_instruction = instruction.get("source_instruction")
    requested = f" Additional user request: {source_instruction.strip()}" if source_instruction else ""
    return (
        "Use the input images as references for the same character. "
        f"{instruction['task_prompt']} "
        f"Background: {instruction['background']}. "
        f"Stable identity facts: {identity_facts}. "
        f"Must preserve: {must_preserve}. "
        f"Avoid: {avoid}. "
        "Do not add props, text, extra characters, alternate outfits, or scene layout elements."
        f"{requested}"
    )


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
) -> dict[str, Any]:
    return {
        "kind": "qwen-character-edit-plan",
        "character_id": pack.character_id,
        "reference_pack": pack_path.as_posix(),
        "identity_profile": identity_profile_path.as_posix(),
        "source_instruction": instruction,
        "normalizer": "identity_profile_case_template_v1",
        "reference_selector": "case_reference_roles_v1",
        "candidates_per_case": candidates_per_case,
        "cases": [
            {
                "name": case.name,
                "references": list(case.references),
                "normalized_instruction": case.normalized_instruction,
                "prompt": case.prompt,
            }
            for case in edit_cases
        ],
        "identity": identity_profile.identity,
        "must_preserve": identity_profile.must_preserve,
        "avoid": identity_profile.avoid,
    }
