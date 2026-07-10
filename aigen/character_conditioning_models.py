from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


CHARACTER_CONDITIONING_PLAN_KIND = "character-conditioning-plan"

CHARACTER_CONDITIONING_MODES = (
    "region_mask",
    "pose_keypoint",
    "edge_or_sketch",
    "depth",
)

CHARACTER_CONDITIONING_TOOLS = (
    "florence2_region_grounding",
    "sam2_mask_generation",
    "dwpose_keypoint_map",
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
    conditioning_modes: list[
        Literal[
            "region_mask",
            "pose_keypoint",
            "edge_or_sketch",
            "depth",
        ]
    ] = Field(default_factory=list)
    planned_tools: list[
        Literal["florence2_region_grounding", "sam2_mask_generation", "dwpose_keypoint_map"]
    ] = Field(default_factory=list)


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
        return CharacterConditioningPlanSpec.model_validate(data)
    except ValidationError as error:
        raise CharacterConditioningPlanError(f"Invalid character conditioning plan {path_label}: {error}") from error
