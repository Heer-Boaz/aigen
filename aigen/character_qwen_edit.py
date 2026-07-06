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
    view_anchor_role: str
    reference_purposes: tuple[tuple[str, str], ...]
    view_constraints: tuple[str, ...]
    portrait_canvas: bool = False


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
    role_reference_ids: dict[str, str]
    reference_paths: dict[str, Path]


QWEN_CHARACTER_EDIT_DEFAULT_CASES = ("front", "side", "right_profile", "back", "three_quarter", "portrait")

QWEN_CHARACTER_EDIT_CASES: dict[str, QwenCharacterEditCaseTemplate] = {
    "front": QwenCharacterEditCaseTemplate(
        name="front",
        references=("front", "portrait", "side"),
        task="Generate a clean full-body front view character reference image in a neutral standing pose.",
        requested_view="front",
        requested_pose="neutral standing",
        background="plain light studio background",
        view_anchor_role="front",
        reference_purposes=(
            ("front", "primary camera angle and full-body outfit anchor"),
            ("portrait", "face identity and facial proportion support"),
            ("side", "body depth and side-silhouette support"),
        ),
        view_constraints=(
            "front-facing full-body camera",
            "entire head-to-toe character remains fully visible with a small background margin",
            "torso and face face the camera",
            "do not rotate the character into a side, back, or three-quarter view",
        ),
    ),
    "side": QwenCharacterEditCaseTemplate(
        name="side",
        references=("side", "portrait", "front"),
        task="Generate a clean full-body left side view character reference image in a neutral standing pose.",
        requested_view="left_side",
        requested_pose="neutral standing",
        background="plain light studio background",
        view_anchor_role="side",
        reference_purposes=(
            ("side", "primary camera angle and side-body geometry anchor"),
            ("portrait", "face identity and facial proportion support"),
            ("front", "full-body outfit, color, and proportion support"),
        ),
        view_constraints=(
            "strict left side profile",
            "entire head-to-toe character remains fully visible with a small background margin",
            "side-on body turned 90 degrees from the camera",
            "do not produce a front-facing or three-quarter torso",
        ),
    ),
    "right_profile": QwenCharacterEditCaseTemplate(
        name="right_profile",
        references=("side", "portrait", "front"),
        task="Generate a clean full-body right profile view character reference image in a neutral standing pose.",
        requested_view="right_profile",
        requested_pose="neutral standing",
        background="plain light studio background",
        view_anchor_role="side",
        reference_purposes=(
            ("side", "primary camera angle and side-body geometry anchor"),
            ("portrait", "face identity and facial proportion support"),
            ("front", "full-body outfit, color, and proportion support"),
        ),
        view_constraints=(
            "strict right side profile",
            "entire head-to-toe character remains fully visible with a small background margin",
            "side-on body turned 90 degrees from the camera",
            "only one side of the face and body is visible",
            "do not produce a front-facing or three-quarter view",
        ),
    ),
    "back": QwenCharacterEditCaseTemplate(
        name="back",
        references=("back", "portrait", "front"),
        task="Generate a clean full-body back view character reference image in a neutral standing pose.",
        requested_view="back",
        requested_pose="neutral standing",
        background="plain light studio background",
        view_anchor_role="back",
        reference_purposes=(
            ("back", "primary camera angle and back-view outfit anchor"),
            ("portrait", "head and face identity support"),
            ("front", "full-body outfit, color, and proportion support"),
        ),
        view_constraints=(
            "back-facing full-body camera",
            "entire head-to-toe character remains fully visible with a small background margin",
            "back of the head, body, and outfit face the camera",
            "do not show the front torso or front face",
        ),
    ),
    "three_quarter": QwenCharacterEditCaseTemplate(
        name="three_quarter",
        references=("front", "side", "portrait"),
        task="Generate a clean full-body three-quarter front view character reference image in a neutral standing pose.",
        requested_view="three_quarter_front",
        requested_pose="neutral standing",
        background="plain light studio background",
        view_anchor_role="front",
        reference_purposes=(
            ("front", "primary full-body outfit and front identity anchor"),
            ("side", "body depth and turned-view geometry support"),
            ("portrait", "face identity and facial proportion support"),
        ),
        view_constraints=(
            "three-quarter front camera",
            "entire head-to-toe character remains fully visible with a small background margin",
            "front of the character remains visible while the body is slightly turned",
            "do not produce a strict front, side, or back view",
        ),
    ),
    "portrait": QwenCharacterEditCaseTemplate(
        name="portrait",
        references=("portrait", "front", "side"),
        task="Generate a clean shoulders-up portrait of the same character.",
        requested_view="portrait",
        requested_pose="neutral expression",
        background="plain light studio background",
        view_anchor_role="portrait",
        reference_purposes=(
            ("portrait", "primary face identity and portrait framing anchor"),
            ("front", "outfit collar, color, and upper-body support"),
            ("side", "head shape and side-silhouette support"),
        ),
        view_constraints=(
            "shoulders-up portrait framing",
            "face identity comes from the portrait reference",
            "do not generate a full-body reference card",
        ),
        portrait_canvas=True,
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
    context = load_qwen_character_reference_context(
        pack_path=pack_path,
        identity_profile_path=identity_profile_path,
        progress=progress,
        phase="load qwen character edit reference pack",
    )
    templates = _selected_templates(cases)
    edit_cases = tuple(
        _planned_case(template, context.identity_profile, context.role_reference_ids, instruction)
        for template in templates
    )
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
        role_reference_ids=_role_reference_ids(pack, identity_profile),
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


def _role_reference_ids(
    pack: CharacterReferencePackSpec,
    identity_profile: CharacterIdentityProfileSpec,
) -> dict[str, str]:
    role_reference_ids: dict[str, str] = {}
    for reference_id, role in identity_profile.reference_roles.items():
        if reference_id not in pack.references:
            raise QwenCharacterEditError(f"Identity profile role references unknown image: {reference_id}")
        existing_reference_id = role_reference_ids.get(role)
        if existing_reference_id is not None:
            raise QwenCharacterEditError(
                f"Identity profile assigned role {role} to both "
                f"{existing_reference_id} and {reference_id}; qwen edit needs one reference per role"
            )
        role_reference_ids[role] = reference_id
    return role_reference_ids


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
    role_reference_ids: Mapping[str, str],
    instruction: str | None,
) -> QwenIdentityCase:
    reference_ids = tuple(
        _reference_id_for_role(role_reference_ids, role, template.name)
        for role in template.references
    )
    normalized_instruction = _normalized_instruction(template, reference_ids, identity_profile, instruction)
    return QwenIdentityCase(
        name=template.name,
        references=reference_ids,
        prompt=_prompt_from_instruction(normalized_instruction),
        portrait_canvas=template.portrait_canvas,
        normalized_instruction=normalized_instruction,
    )


def _reference_id_for_role(role_reference_ids: Mapping[str, str], role: str, case_name: str) -> str:
    try:
        return role_reference_ids[role]
    except KeyError as error:
        raise QwenCharacterEditError(f"Qwen edit case {case_name} needs inferred reference role {role}") from error


def _normalized_instruction(
    template: QwenCharacterEditCaseTemplate,
    reference_ids: Sequence[str],
    identity_profile: CharacterIdentityProfileSpec,
    instruction: str | None,
) -> dict[str, Any]:
    must_preserve = _identity_must_preserve(identity_profile)
    avoid = _identity_avoid(identity_profile)
    reference_id_by_role = dict(zip(template.references, reference_ids, strict=True))
    view_anchor_ref = reference_id_by_role[template.view_anchor_role]
    normalized: dict[str, Any] = {
        "task": "identity_edit",
        "case": template.name,
        "required_roles": list(template.references),
        "refs_used": list(reference_ids),
        "view_anchor_role": template.view_anchor_role,
        "view_anchor_ref": view_anchor_ref,
        "view_anchor_input_index": template.references.index(template.view_anchor_role) + 1,
        "reference_purposes": [
            {
                "role": role,
                "reference_id": reference_id_by_role[role],
                "input_index": template.references.index(role) + 1,
                "purpose": purpose,
            }
            for role, purpose in template.reference_purposes
        ],
        "view_constraints": list(template.view_constraints),
        "task_prompt": template.task,
        "requested_view": template.requested_view,
        "requested_pose": template.requested_pose,
        "background": template.background,
        "source_instruction": instruction,
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


def _prompt_from_instruction(instruction: Mapping[str, Any]) -> str:
    identity = instruction["identity"]
    identity_facts = "; ".join(f"{name}: {value}" for name, value in identity.items() if value)
    must_preserve = "; ".join(instruction["must_preserve"])
    avoid = "; ".join(instruction["avoid"])
    body_proportion = instruction["body_proportion"]
    body_proportion_facts = _body_proportion_facts(body_proportion)
    reference_purposes = _reference_purpose_facts(instruction["reference_purposes"])
    view_constraints = "; ".join(instruction["view_constraints"])
    source_instruction = instruction.get("source_instruction")
    requested = f" Additional user request: {source_instruction.strip()}" if source_instruction else ""
    return (
        "Use the input images as references for the same character. "
        f"Reference routing: {reference_purposes}. "
        f"Camera/view anchor: input image {instruction['view_anchor_input_index']} "
        f"({instruction['view_anchor_ref']}, role {instruction['view_anchor_role']}) controls the requested "
        "camera angle and body orientation; non-anchor references preserve identity details but must not override "
        "the camera angle. "
        f"{instruction['task_prompt']} "
        f"View constraints: {view_constraints}. "
        f"Background: {instruction['background']}. "
        f"Stable identity facts: {identity_facts}. "
        f"Body proportion source: {instruction['body_proportion_source']}. "
        f"Body proportion invariants: {body_proportion_facts}. "
        f"Must preserve: {must_preserve}. "
        f"Avoid: {avoid}. "
        "Do not add props, text, extra characters, alternate outfits, or scene layout elements."
        f"{requested}"
    )


def _body_proportion_facts(body_proportion: Mapping[str, Any]) -> str:
    return "; ".join(
        f"{name}: {value}"
        for name, value in body_proportion.items()
        if isinstance(value, str) and value
    )


def _reference_purpose_facts(reference_purposes: Sequence[Mapping[str, Any]]) -> str:
    return "; ".join(
        f"input image {reference['input_index']} ({reference['reference_id']}, role {reference['role']}): "
        f"{reference['purpose']}"
        for reference in reference_purposes
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
        "normalizer": "identity_profile_case_template_v2",
        "reference_selector": "case_reference_roles_with_identity_profile_body_proportion_v1",
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
