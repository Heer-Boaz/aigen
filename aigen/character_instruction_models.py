from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


CHARACTER_INSTRUCTION_PLAN_KIND = "character-instruction-plan"

CHARACTER_INSTRUCTION_TASK_FAMILIES = (
    "reference_character_portrait",
    "reference_character_full_body",
    "view_change",
    "pose_transfer",
    "local_repair",
    "outfit_swap",
    "style_transfer",
    "scene_insertion",
    "layout_or_sheet",
    "text_or_label_heavy",
    "unknown",
)
CHARACTER_INSTRUCTION_EDIT_SCOPES = ("global", "local", "mixed", "unknown")
CHARACTER_INSTRUCTION_SUBJECT_BINDINGS = (
    "referenced_character",
    "image_index_binding",
    "source_image_subject",
    "multiple_subjects",
    "unspecified",
)
CHARACTER_INSTRUCTION_DOWNSTREAM_REQUIREMENTS = (
    "visual_identity_analysis",
    "multi_reference_alignment",
    "region_grounding",
    "mask_generation",
    "pose_conditioning",
    "text_rendering_risk",
    "external_concept_resolution",
    "clarification_needed",
    "visual_disambiguation",
)


class CharacterInstructionError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InstructionEnvelopeSpec(StrictModel):
    raw_instruction: str
    ui_mode: str
    reference_count: int = Field(ge=0)
    source_image_present: bool
    mask_present: bool
    region_plan_present: bool
    generation_panel_settings: dict[str, Any] = Field(default_factory=dict)
    requested_model_family: str | None = None
    negative_prompt_present: bool = False
    aspect_ratio_setting: str | None = None
    seed_setting: int | None = None


class InstructionSubjectBindingSpec(StrictModel):
    kind: str
    reference_mentions: list[str] = Field(default_factory=list)
    note: str = ""


class InstructionTargetConstraintsSpec(StrictModel):
    framing: list[str] = Field(default_factory=list)
    camera_view: list[str] = Field(default_factory=list)
    pose: list[str] = Field(default_factory=list)
    gaze: list[str] = Field(default_factory=list)
    expression: list[str] = Field(default_factory=list)
    lighting: list[str] = Field(default_factory=list)
    background: list[str] = Field(default_factory=list)
    explicit_style_or_role: list[str] = Field(default_factory=list)
    scene: list[str] = Field(default_factory=list)
    action: list[str] = Field(default_factory=list)
    mood_or_personality: list[str] = Field(default_factory=list)
    composition: list[str] = Field(default_factory=list)
    text_or_logo: list[str] = Field(default_factory=list)


class CharacterInstructionModelResponseSpec(StrictModel):
    language: str
    task_family: str
    edit_scope: str
    subject_binding: InstructionSubjectBindingSpec
    target_constraints: InstructionTargetConstraintsSpec
    named_external_concepts: list[str] = Field(default_factory=list)
    downstream_requirements: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)


class CharacterInstructionPlanSpec(StrictModel):
    kind: Literal["character-instruction-plan"]
    raw_instruction: str
    normalized_instruction_text: str
    envelope: InstructionEnvelopeSpec
    language: str
    task_family: Literal[
        "reference_character_portrait",
        "reference_character_full_body",
        "view_change",
        "pose_transfer",
        "local_repair",
        "outfit_swap",
        "style_transfer",
        "scene_insertion",
        "layout_or_sheet",
        "text_or_label_heavy",
        "unknown",
    ]
    edit_scope: Literal["global", "local", "mixed", "unknown"]
    subject_binding: InstructionSubjectBindingSpec
    target_constraints: InstructionTargetConstraintsSpec
    named_external_concepts: list[str]
    downstream_requirements: list[str]
    ambiguities: list[str]
    conflicts: list[str]
    parser: dict[str, Any]
    raw_model_response: dict[str, Any]


def character_instruction_model_response_schema() -> dict[str, Any]:
    schema = CharacterInstructionModelResponseSpec.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_character_instruction_model_response(
    data: dict[str, Any],
    *,
    path_label: str,
) -> CharacterInstructionModelResponseSpec:
    try:
        response = CharacterInstructionModelResponseSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterInstructionError(f"Invalid character instruction parser response {path_label}: {error}") from error
    if not response.task_family.strip():
        raise CharacterInstructionError(
            f"Invalid character instruction parser response {path_label}: task_family is empty"
        )
    if not response.edit_scope.strip():
        raise CharacterInstructionError(
            f"Invalid character instruction parser response {path_label}: edit_scope is empty"
        )
    if not response.subject_binding.kind.strip():
        raise CharacterInstructionError(
            f"Invalid character instruction parser response {path_label}: subject_binding.kind is empty"
        )
    return response
