from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


CHARACTER_REFERENCE_OBSERVATION_SET_KIND = "character-reference-observation-set"


class CharacterReferenceObservationError(RuntimeError):
    pass


class ObservationModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class CharacterReferenceObservationSpec(ObservationModel):
    reference_id: str
    visual_summary: str
    visible_subjects: list[Any] = Field(default_factory=list)
    view_or_framing: str = ""
    visible_components: list[Any] = Field(default_factory=list)
    occlusion_or_quality_notes: list[Any] = Field(default_factory=list)
    text_or_symbol_notes: list[Any] = Field(default_factory=list)
    uncertainties: list[Any] = Field(default_factory=list)


class CharacterReferenceObservationSetSpec(ObservationModel):
    kind: Literal["character-reference-observation-set"]
    observations: dict[str, CharacterReferenceObservationSpec]


def load_character_reference_observation(
    data: dict[str, Any],
    *,
    path_label: str,
    reference_id: str,
) -> CharacterReferenceObservationSpec:
    try:
        observation = CharacterReferenceObservationSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterReferenceObservationError(
            f"Invalid character reference observation {path_label}: {error}"
        ) from error
    _validate_observation(observation, path_label=path_label, reference_id=reference_id)
    return observation


def load_character_reference_observation_set(
    data: dict[str, Any],
    *,
    path_label: str,
    reference_ids: tuple[str, ...],
) -> CharacterReferenceObservationSetSpec:
    try:
        observation_set = CharacterReferenceObservationSetSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterReferenceObservationError(
            f"Invalid character reference observation set {path_label}: {error}"
        ) from error
    expected = set(reference_ids)
    observed = set(observation_set.observations)
    missing = sorted(expected - observed)
    if missing:
        raise CharacterReferenceObservationError(
            f"Invalid character reference observation set {path_label}: missing observation(s) "
            f"{', '.join(missing)}"
        )
    unknown = sorted(observed - expected)
    if unknown:
        raise CharacterReferenceObservationError(
            f"Invalid character reference observation set {path_label}: unknown observation(s) "
            f"{', '.join(unknown)}"
        )
    for reference_id, observation in observation_set.observations.items():
        _validate_observation(observation, path_label=path_label, reference_id=reference_id)
    return observation_set


def _validate_observation(
    observation: CharacterReferenceObservationSpec,
    *,
    path_label: str,
    reference_id: str,
) -> None:
    if observation.reference_id != reference_id:
        raise CharacterReferenceObservationError(
            f"Invalid character reference observation {path_label}: reference_id "
            f"{observation.reference_id!r} does not match {reference_id!r}"
        )
    if not observation.visual_summary.strip():
        raise CharacterReferenceObservationError(
            f"Invalid character reference observation {path_label}: visual_summary is empty"
        )
