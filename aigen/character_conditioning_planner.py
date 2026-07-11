from __future__ import annotations

from collections.abc import Collection

from aigen.character_conditioning_models import (
    CHARACTER_CONDITIONING_PLAN_KIND,
    CharacterConditioningPlanError,
    CharacterConditioningPlanSpec,
)


_ALLOWED_MODES_BY_ROUTE = {
    "pose_transfer": ("pose_keypoint",),
    "local_repair_or_inpaint": ("region_mask",),
    "scene_insertion": ("depth", "edge_or_sketch"),
    "outfit_swap": ("region_mask",),
}

_REQUIRED_MODES_BY_ROUTE = {
    "pose_transfer": frozenset(("pose_keypoint",)),
    "local_repair_or_inpaint": frozenset(("region_mask",)),
}
_NO_REQUIRED_MODES = frozenset()
_TOOLS_BY_MODE = {
    "region_mask": ("florence2_region_grounding", "sam2_mask_generation"),
    "pose_keypoint": ("dwpose_keypoint_map",),
    "depth": ("depth_anything_v2_map",),
    "edge_or_sketch": ("canny_edge_map",),
}


class CharacterConditioningPlanner:
    def plan(
        self,
        *,
        route_kind: str,
        available_modes: Collection[str],
    ) -> CharacterConditioningPlanSpec:
        allowed_modes = _ALLOWED_MODES_BY_ROUTE.get(route_kind, ())
        available_mode_set = set(available_modes)
        unexpected_modes = available_mode_set.difference(allowed_modes)
        if unexpected_modes:
            raise CharacterConditioningPlanError(
                f"Unexpected conditioning modes for {route_kind}: {sorted(unexpected_modes)}"
            )
        missing_modes = _REQUIRED_MODES_BY_ROUTE.get(route_kind, _NO_REQUIRED_MODES).difference(available_mode_set)
        if missing_modes:
            raise CharacterConditioningPlanError(
                f"Missing required conditioning modes for {route_kind}: {sorted(missing_modes)}"
            )
        return CharacterConditioningPlanSpec(
            kind=CHARACTER_CONDITIONING_PLAN_KIND,
            route_kind=route_kind,
            conditioning_modes=[mode for mode in allowed_modes if mode in available_mode_set],
            planned_tools=[
                tool
                for mode in allowed_modes
                if mode in available_mode_set and mode in _TOOLS_BY_MODE
                for tool in _TOOLS_BY_MODE[mode]
            ],
        )
