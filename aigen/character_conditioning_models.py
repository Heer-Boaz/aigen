from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


CHARACTER_CONDITIONING_PLAN_KIND = "character-conditioning-plan"

CHARACTER_CONDITIONING_STATUSES = (
    "no_extra_conditioning",
    "deferred",
    "required_but_missing_inputs",
    "ready",
)

CHARACTER_CONDITIONING_MODES = (
    "region_mask",
    "pose_keypoint",
    "edge_or_sketch",
    "depth",
    "text_layout_risk",
    "layout_planning",
)

CHARACTER_CONDITIONING_INPUTS = (
    "source_image",
    "mask",
    "region_plan",
    "pose_source",
)

CHARACTER_CONDITIONING_TOOLS = (
    "florence2_region_grounding",
    "sam2_mask_generation",
    "dwpose_keypoint_map",
)

CHARACTER_CONDITIONING_DEFER_TARGETS = (
    "region-plan",
    "qwen-edit-refine",
    "qwen-edit-pose",
    "none",
)


class CharacterConditioningPlanError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CharacterConditioningPlanSpec(StrictModel):
    kind: Literal["character-conditioning-plan"]
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
    status: Literal["no_extra_conditioning", "deferred", "required_but_missing_inputs", "ready"]
    conditioning_modes: list[
        Literal[
            "region_mask",
            "pose_keypoint",
            "edge_or_sketch",
            "depth",
            "text_layout_risk",
            "layout_planning",
        ]
    ] = Field(default_factory=list)
    required_inputs: list[
        Literal["source_image", "mask", "region_plan", "pose_source"]
    ] = Field(default_factory=list)
    planned_tools: list[
        Literal["florence2_region_grounding", "sam2_mask_generation", "dwpose_keypoint_map"]
    ] = Field(default_factory=list)
    deferred_to: Literal["region-plan", "qwen-edit-refine", "qwen-edit-pose", "none"]
    supports_current_command: bool
    audit_notes: list[str] = Field(default_factory=list)


def character_conditioning_plan_schema() -> dict[str, Any]:
    schema = CharacterConditioningPlanSpec.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_character_conditioning_plan(
    data: dict[str, Any],
    *,
    path_label: str,
) -> CharacterConditioningPlanSpec:
    try:
        plan = CharacterConditioningPlanSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterConditioningPlanError(f"Invalid character conditioning plan {path_label}: {error}") from error
    if plan.status == "no_extra_conditioning" and plan.conditioning_modes:
        raise CharacterConditioningPlanError(
            f"Invalid character conditioning plan {path_label}: no_extra_conditioning cannot list modes"
        )
    if plan.status != "no_extra_conditioning" and not plan.conditioning_modes:
        raise CharacterConditioningPlanError(
            f"Invalid character conditioning plan {path_label}: conditioning mode is required"
        )
    return plan
