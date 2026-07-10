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
    supports_route_now: bool


class ModelCapabilityRegistrySpec(StrictModel):
    max_qwen_edit_refs: int = Field(default=3, ge=1)
    supports_multi_image_edit: bool = True
    supports_keypoint_condition: bool = True
    supports_depth_condition: bool = True
    supports_edge_condition: bool = True
    supports_masked_refine: bool = True
    supports_text_heavy_generation: Literal["supported", "experimental", "weak_or_experimental", "route_only"] = (
        "weak_or_experimental"
    )
    supports_layout_sheet: Literal["supported", "experimental", "route_only"] = "route_only"
    supports_external_web_resolution: bool = False


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
    visual_analysis_focus: list[str]
    reference_selection_intent: list[str]
    conditioning_needs: list[str]
    final_editor_constraints: FinalEditorConstraintsSpec
    unsupported_or_deferred: list[str]
    audit_notes: list[str]
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
        route_plan = CharacterTaskRoutePlanSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterTaskRouteError(f"Invalid character task route plan {path_label}: {error}") from error
    if not route_plan.visual_analysis_focus:
        raise CharacterTaskRouteError(f"Invalid character task route plan {path_label}: visual_analysis_focus is empty")
    if not route_plan.reference_selection_intent:
        raise CharacterTaskRouteError(
            f"Invalid character task route plan {path_label}: reference_selection_intent is empty"
        )
    return route_plan
