from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError


CHARACTER_REFERENCE_PACK_KIND = "character-reference-pack"
CHARACTER_IDENTITY_PROFILE_KIND = "character-identity-profile"
CHARACTER_REFERENCE_ANALYSIS_KIND = "character-reference-analysis"
CHARACTER_REFERENCE_NAMES = ("front", "portrait", "side", "back", "body_shape")
CHARACTER_REFERENCE_NAME_SET = frozenset(CHARACTER_REFERENCE_NAMES)
CHARACTER_REFERENCE_ROLES = {
    "front": "front outfit, colors, face placement and body proportions",
    "portrait": "face, hair, eyes, neckwear and expression",
    "side": "profile silhouette, outfit thickness and body shape",
    "back": "back hair shape, jacket back, skirt back and footwear from behind",
    "body_shape": "body proportions and silhouette consistency",
}


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
    reference_roles: dict[str, str]
    output: CharacterReferencePackOutputSpec


class CharacterIdentityVlmResponseSpec(StrictModel):
    identity: dict[str, str]
    reference_roles: dict[str, str]
    must_preserve: list[str]
    avoid: list[str]


class CharacterIdentityProfileOutputSpec(StrictModel):
    identity_profile: str


class CharacterIdentityProfileSpec(StrictModel):
    kind: Literal["character-identity-profile"]
    character_id: str
    source_reference_pack: str
    identity: dict[str, str]
    reference_roles: dict[str, str]
    must_preserve: list[str]
    avoid: list[str]
    parser: dict[str, Any]
    output: CharacterIdentityProfileOutputSpec


class BodyMeasurementSpec(StrictModel):
    value: float
    unit: str
    confidence: float
    evidence_refs: list[str]
    evidence: dict[str, Any]


class BodyProfileSpec(StrictModel):
    source: Literal["measured_from_reference_pack"]
    extractors: dict[str, str]
    measurements: dict[str, BodyMeasurementSpec]
    semantic_summary: dict[str, str]
    evidence_refs: list[str]
    optional_missing_refs: list[str]
    confidence_warnings: list[str]


class ReferenceAnalysisSpec(StrictModel):
    role: str
    image: ImageAssetSpec
    extractors_used: dict[str, str]
    artifacts: dict[str, str]
    mask: dict[str, Any]
    pose: dict[str, Any]
    warnings: list[str]


class CharacterReferenceAnalysisOutputSpec(StrictModel):
    reference_analysis: str
    artifacts_directory: str


class CharacterReferenceAnalysisSpec(StrictModel):
    kind: Literal["character-reference-analysis"]
    character_id: str
    source_reference_pack: str
    references: dict[str, ReferenceAnalysisSpec]
    body_profile: BodyProfileSpec
    output: CharacterReferenceAnalysisOutputSpec


def character_reference_pack_schema() -> dict[str, Any]:
    schema = CharacterReferencePackSpec.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def character_identity_profile_schema() -> dict[str, Any]:
    schema = CharacterIdentityProfileSpec.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def character_reference_analysis_schema() -> dict[str, Any]:
    schema = CharacterReferenceAnalysisSpec.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_character_reference_pack_payload(data: dict[str, Any], *, path_label: str) -> CharacterReferencePackSpec:
    try:
        pack = CharacterReferencePackSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterReferenceError(f"Invalid character reference pack {path_label}: {error}") from error
    _validate_reference_names(pack.references, path_label=path_label)
    _validate_reference_names(pack.reference_roles, path_label=path_label)
    return pack


def load_completed_character_reference_pack(data: dict[str, Any], *, path_label: str) -> CharacterReferencePackSpec:
    return load_character_reference_pack_payload(_without_completed_status(data), path_label=path_label)


def load_character_identity_profile_payload(data: dict[str, Any], *, path_label: str) -> CharacterIdentityProfileSpec:
    try:
        profile = CharacterIdentityProfileSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterReferenceError(f"Invalid character identity profile {path_label}: {error}") from error
    _validate_reference_names(profile.reference_roles, path_label=path_label)
    return profile


def load_completed_character_identity_profile(data: dict[str, Any], *, path_label: str) -> CharacterIdentityProfileSpec:
    return load_character_identity_profile_payload(_without_completed_status(data), path_label=path_label)


def load_character_reference_analysis_payload(data: dict[str, Any], *, path_label: str) -> CharacterReferenceAnalysisSpec:
    try:
        analysis = CharacterReferenceAnalysisSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterReferenceError(f"Invalid character reference analysis {path_label}: {error}") from error
    _validate_reference_names(analysis.references, path_label=path_label)
    return analysis


def load_completed_character_reference_analysis(data: dict[str, Any], *, path_label: str) -> CharacterReferenceAnalysisSpec:
    return load_character_reference_analysis_payload(_without_completed_status(data), path_label=path_label)


def load_character_identity_vlm_response(data: dict[str, Any], *, path_label: str) -> CharacterIdentityVlmResponseSpec:
    try:
        response = CharacterIdentityVlmResponseSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterReferenceError(f"Invalid character identity parser response {path_label}: {error}") from error
    if not response.identity:
        raise CharacterReferenceError(f"Invalid character identity parser response {path_label}: identity is empty")
    if not response.must_preserve:
        raise CharacterReferenceError(
            f"Invalid character identity parser response {path_label}: must_preserve is empty"
        )
    return response


def _validate_reference_names(mapping: dict[str, Any], *, path_label: str) -> None:
    unknown = sorted(name for name in mapping if name not in CHARACTER_REFERENCE_NAME_SET)
    if unknown:
        allowed = ", ".join(CHARACTER_REFERENCE_NAMES)
        raise CharacterReferenceError(
            f"Invalid character reference pack {path_label}: unknown reference name(s) "
            f"{', '.join(unknown)}; expected one of: {allowed}"
        )


def _without_completed_status(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("status") != "completed":
        return data
    payload = dict(data)
    del payload["status"]
    return payload
