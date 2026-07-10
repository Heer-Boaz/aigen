from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


CHARACTER_TASK_ROUTE_PLAN_KIND = "character-task-route-plan"

CHARACTER_TASK_ROUTE_KINDS = (
    "portrait_identity_generation",
    "full_body_identity_generation",
    "view_change",
    "pose_transfer",
    "scene_insertion",
    "local_repair_or_inpaint",
    "outfit_swap",
    "style_transfer",
    "layout_or_sheet",
    "text_or_label_heavy",
    "unknown_reference_edit",
)

CHARACTER_EDITOR_ROUTES = (
    "qwen_image_edit_multi_reference",
    "qwen_image_edit_with_control_condition",
    "qwen_image_edit_inpaint_or_masked_edit",
    "layout_heavy_reference_generation",
    "qwen_image_edit_text_heavy",
    "qwen_image_edit_unknown_reference",
)

CHARACTER_OUTPUT_MODES = (
    "single_image_portrait",
    "single_image_full_body",
    "single_image_view",
    "single_image_scene",
    "single_image_pose",
    "masked_refine_candidates",
    "layout_or_sheet",
    "text_or_label_image",
    "single_image_reference_edit",
)


class CharacterTaskRouteError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReferenceBudgetSpec(StrictModel):
    min: int = Field(ge=1)
    max: int = Field(ge=1)


class FinalEditorConstraintsSpec(StrictModel):
    reference_budget: ReferenceBudgetSpec
    mask_required: bool
    pose_conditioning: bool
    text_rendering: bool
    layout_complexity: Literal["simple", "moderate", "layout_heavy"]


class ModelCapabilityRegistrySpec(StrictModel):
    max_qwen_edit_images: int = Field(default=3, ge=1)


class CharacterTaskRoutePlanSpec(StrictModel):
    kind: Literal["character-task-route-plan"]
    route_kind: Literal[
        "portrait_identity_generation",
        "full_body_identity_generation",
        "view_change",
        "pose_transfer",
        "scene_insertion",
        "local_repair_or_inpaint",
        "outfit_swap",
        "style_transfer",
        "layout_or_sheet",
        "text_or_label_heavy",
        "unknown_reference_edit",
    ]
    source_instruction: str
    editor_route: Literal[
        "qwen_image_edit_multi_reference",
        "qwen_image_edit_with_control_condition",
        "qwen_image_edit_inpaint_or_masked_edit",
        "layout_heavy_reference_generation",
        "qwen_image_edit_text_heavy",
        "qwen_image_edit_unknown_reference",
    ]
    output_mode: Literal[
        "single_image_portrait",
        "single_image_full_body",
        "single_image_view",
        "single_image_scene",
        "single_image_pose",
        "masked_refine_candidates",
        "layout_or_sheet",
        "text_or_label_image",
        "single_image_reference_edit",
    ]
    final_editor_constraints: FinalEditorConstraintsSpec
    capability_registry: ModelCapabilityRegistrySpec


def character_task_route_plan_schema() -> dict[str, Any]:
    schema = CharacterTaskRoutePlanSpec.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_character_task_route_plan(
    data: dict[str, Any],
    *,
    path_label: str,
) -> CharacterTaskRoutePlanSpec:
    try:
        return CharacterTaskRoutePlanSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterTaskRouteError(f"Invalid character task route plan {path_label}: {error}") from error
