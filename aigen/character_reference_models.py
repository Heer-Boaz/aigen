from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


CHARACTER_REFERENCE_PACK_KIND = "character-reference-pack"
CHARACTER_IDENTITY_PROFILE_KIND = "character-identity-profile"
CHARACTER_BODY_PROPORTION_SOURCE = "model_extracted_from_reference_pack"
CHARACTER_REFERENCE_ROLE_NAMES = ("front", "portrait", "side", "back", "body_shape")
CHARACTER_REFERENCE_ROLE_SET = frozenset(CHARACTER_REFERENCE_ROLE_NAMES)
CHARACTER_REFERENCE_NAMES = CHARACTER_REFERENCE_ROLE_NAMES
CHARACTER_REFERENCE_NAME_SET = CHARACTER_REFERENCE_ROLE_SET


class CharacterReferenceError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageAssetSpec(StrictModel):
    path: str
    sha256: str
    mode: str
    width: int
    height: int


class CharacterReferencePackOutputSpec(StrictModel):
    directory: str
    reference_pack: str


class CharacterReferencePackSpec(StrictModel):
    kind: Literal["character-reference-pack"]
    character_id: str
    references: dict[str, ImageAssetSpec]
    reference_hints: dict[str, str]
    output: CharacterReferencePackOutputSpec


class BodyProportionSpec(StrictModel):
    chest_size: str
    build: str
    shoulder_width: str
    waist: str
    hip_skirt_silhouette: str
    side_body_thickness: str
    leg_proportion: str
    skirt_back_shape: str
    do_not_change: list[str]
    evidence_refs: list[str]


class CharacterIdentityVlmResponseSpec(StrictModel):
    identity: dict[str, str]
    body_proportion: BodyProportionSpec
    reference_roles: dict[str, str]
    must_preserve: list[str]
    avoid: list[str]


class CharacterEditPlanVlmResponseSpec(StrictModel):
    selected_refs: list[str]
    edit_instruction: str
    reference_semantics: dict[str, str] = Field(default_factory=dict)


class CharacterIdentityProfileOutputSpec(StrictModel):
    identity_profile: str


class CharacterIdentityProfileSpec(StrictModel):
    kind: Literal["character-identity-profile"]
    character_id: str
    source_reference_pack: str
    identity: dict[str, str]
    body_proportion: BodyProportionSpec
    body_proportion_source: Literal["model_extracted_from_reference_pack"]
    reference_roles: dict[str, str]
    optional_missing_refs: list[str]
    must_preserve: list[str]
    avoid: list[str]
    parser: dict[str, Any]
    output: CharacterIdentityProfileOutputSpec


def character_reference_pack_schema() -> dict[str, Any]:
    schema = CharacterReferencePackSpec.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def character_identity_profile_schema() -> dict[str, Any]:
    schema = CharacterIdentityProfileSpec.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_character_reference_pack_payload(data: dict[str, Any], *, path_label: str) -> CharacterReferencePackSpec:
    try:
        pack = CharacterReferencePackSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterReferenceError(f"Invalid character reference pack {path_label}: {error}") from error
    _validate_reference_ids(pack.references, path_label=path_label)
    _validate_reference_ids(pack.reference_hints, path_label=path_label)
    if set(pack.reference_hints) != set(pack.references):
        raise CharacterReferenceError(
            f"Invalid character reference pack {path_label}: reference_hints must match references"
        )
    return pack


def load_completed_character_reference_pack(data: dict[str, Any], *, path_label: str) -> CharacterReferencePackSpec:
    return load_character_reference_pack_payload(_without_completed_status(data), path_label=path_label)


def load_character_identity_profile_payload(data: dict[str, Any], *, path_label: str) -> CharacterIdentityProfileSpec:
    try:
        profile = CharacterIdentityProfileSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterReferenceError(f"Invalid character identity profile {path_label}: {error}") from error
    _validate_reference_ids(profile.reference_roles, path_label=path_label)
    _validate_reference_role_values(profile.reference_roles, path_label=path_label)
    _validate_body_proportion(profile.body_proportion, path_label=path_label)
    return profile


def load_completed_character_identity_profile(data: dict[str, Any], *, path_label: str) -> CharacterIdentityProfileSpec:
    return load_character_identity_profile_payload(_without_completed_status(data), path_label=path_label)


def load_character_identity_vlm_response(
    data: dict[str, Any],
    *,
    path_label: str,
    reference_ids: set[str] | frozenset[str] | None = None,
) -> CharacterIdentityVlmResponseSpec:
    try:
        response = CharacterIdentityVlmResponseSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterReferenceError(f"Invalid character identity parser response {path_label}: {error}") from error
    if not response.identity:
        raise CharacterReferenceError(f"Invalid character identity parser response {path_label}: identity is empty")
    if not response.reference_roles:
        raise CharacterReferenceError(
            f"Invalid character identity parser response {path_label}: reference_roles is empty"
        )
    if not response.must_preserve:
        raise CharacterReferenceError(
            f"Invalid character identity parser response {path_label}: must_preserve is empty"
        )
    _validate_reference_ids(response.reference_roles, path_label=path_label)
    _validate_reference_role_values(response.reference_roles, path_label=path_label)
    _validate_body_proportion(response.body_proportion, path_label=path_label)
    if reference_ids is not None:
        missing_roles = sorted(name for name in reference_ids if name not in response.reference_roles)
        unknown_roles = sorted(name for name in response.reference_roles if name not in reference_ids)
        unknown_evidence = sorted(name for name in response.body_proportion.evidence_refs if name not in reference_ids)
        if missing_roles:
            raise CharacterReferenceError(
                f"Invalid character identity parser response {path_label}: missing role for reference "
                f"{', '.join(missing_roles)}"
            )
        if unknown_roles:
            raise CharacterReferenceError(
                f"Invalid character identity parser response {path_label}: role for unknown reference "
                f"{', '.join(unknown_roles)}"
            )
        if unknown_evidence:
            raise CharacterReferenceError(
                f"Invalid character identity parser response {path_label}: body_proportion evidence for unknown "
                f"reference {', '.join(unknown_evidence)}"
            )
    return response


def load_character_edit_plan_vlm_response(
    data: dict[str, Any],
    *,
    path_label: str,
    reference_ids: set[str] | frozenset[str],
) -> CharacterEditPlanVlmResponseSpec:
    try:
        response = CharacterEditPlanVlmResponseSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterReferenceError(f"Invalid character edit plan response {path_label}: {error}") from error
    if not response.selected_refs:
        raise CharacterReferenceError(f"Invalid character edit plan response {path_label}: selected_refs is empty")
    if len(response.selected_refs) > 3:
        raise CharacterReferenceError(
            f"Invalid character edit plan response {path_label}: selected_refs must contain at most 3 refs"
        )
    if len(set(response.selected_refs)) != len(response.selected_refs):
        raise CharacterReferenceError(f"Invalid character edit plan response {path_label}: selected_refs has duplicates")
    unknown_refs = sorted(ref for ref in response.selected_refs if ref not in reference_ids)
    if unknown_refs:
        raise CharacterReferenceError(
            f"Invalid character edit plan response {path_label}: selected_refs contains unknown reference(s) "
            f"{', '.join(unknown_refs)}"
        )
    unknown_semantics = sorted(ref for ref in response.reference_semantics if ref not in reference_ids)
    if unknown_semantics:
        raise CharacterReferenceError(
            f"Invalid character edit plan response {path_label}: reference_semantics contains unknown reference(s) "
            f"{', '.join(unknown_semantics)}"
        )
    if not response.edit_instruction.strip():
        raise CharacterReferenceError(f"Invalid character edit plan response {path_label}: edit_instruction is empty")
    return response


def _validate_reference_ids(mapping: dict[str, Any], *, path_label: str) -> None:
    invalid = sorted(name for name in mapping if not name.strip())
    if invalid:
        raise CharacterReferenceError(
            f"Invalid character reference pack {path_label}: reference ids must not be empty"
        )


def _validate_reference_role_values(mapping: dict[str, str], *, path_label: str) -> None:
    unknown = sorted(role for role in mapping.values() if role not in CHARACTER_REFERENCE_ROLE_SET)
    if unknown:
        allowed = ", ".join(CHARACTER_REFERENCE_ROLE_NAMES)
        raise CharacterReferenceError(
            f"Invalid character identity profile {path_label}: unknown reference role(s) "
            f"{', '.join(unknown)}; expected one of: {allowed}"
        )


def _validate_body_proportion(body_proportion: BodyProportionSpec, *, path_label: str) -> None:
    if not body_proportion.do_not_change:
        raise CharacterReferenceError(
            f"Invalid character identity profile {path_label}: body_proportion.do_not_change is empty"
        )
    if not body_proportion.evidence_refs:
        raise CharacterReferenceError(
            f"Invalid character identity profile {path_label}: body_proportion.evidence_refs is empty"
        )


def _without_completed_status(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("status") != "completed":
        return data
    payload = dict(data)
    del payload["status"]
    return payload
