from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


POLISH_OPERATIONS = (
    "detail_restore",
    "expression_refine",
    "identity_restore",
    "lineart_sharpen",
    "color_restore",
    "shape_fix",
    "hand_fix",
    "artifact_remove",
)

POLISH_OPERATION_PROFILES: dict[str, dict[str, float | int]] = {
    "detail_restore": {
        "strength_min": 0.18,
        "strength_max": 0.40,
        "steps_min": 14,
        "steps_max": 22,
        "guidance_min": 1.8,
        "guidance_max": 2.6,
        "true_cfg_min": 1.0,
        "true_cfg_max": 1.4,
    },
    "expression_refine": {
        "strength_min": 0.28,
        "strength_max": 0.55,
        "steps_min": 16,
        "steps_max": 24,
        "guidance_min": 2.0,
        "guidance_max": 2.8,
        "true_cfg_min": 1.1,
        "true_cfg_max": 1.6,
    },
    "identity_restore": {
        "strength_min": 0.22,
        "strength_max": 0.48,
        "steps_min": 16,
        "steps_max": 24,
        "guidance_min": 2.0,
        "guidance_max": 2.8,
        "true_cfg_min": 1.1,
        "true_cfg_max": 1.6,
    },
    "lineart_sharpen": {
        "strength_min": 0.16,
        "strength_max": 0.34,
        "steps_min": 12,
        "steps_max": 20,
        "guidance_min": 1.7,
        "guidance_max": 2.5,
        "true_cfg_min": 1.0,
        "true_cfg_max": 1.35,
    },
    "color_restore": {
        "strength_min": 0.14,
        "strength_max": 0.32,
        "steps_min": 12,
        "steps_max": 20,
        "guidance_min": 1.7,
        "guidance_max": 2.4,
        "true_cfg_min": 1.0,
        "true_cfg_max": 1.35,
    },
    "shape_fix": {
        "strength_min": 0.45,
        "strength_max": 0.85,
        "steps_min": 20,
        "steps_max": 28,
        "guidance_min": 2.2,
        "guidance_max": 3.0,
        "true_cfg_min": 1.2,
        "true_cfg_max": 1.8,
    },
    "hand_fix": {
        "strength_min": 0.38,
        "strength_max": 0.75,
        "steps_min": 18,
        "steps_max": 28,
        "guidance_min": 2.1,
        "guidance_max": 3.0,
        "true_cfg_min": 1.15,
        "true_cfg_max": 1.8,
    },
    "artifact_remove": {
        "strength_min": 0.16,
        "strength_max": 0.42,
        "steps_min": 12,
        "steps_max": 22,
        "guidance_min": 1.7,
        "guidance_max": 2.5,
        "true_cfg_min": 1.0,
        "true_cfg_max": 1.4,
    },
}


class KeyframePolishError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolishPipelineSpec(StrictModel):
    profile: str


class PolishBaseSpec(StrictModel):
    run_dir: str
    candidate: str


class PolishIdentityPrimerSpec(StrictModel):
    view: Literal["front", "left_profile", "right_profile", "back"]
    path: str


class PolishCharacterSpec(StrictModel):
    id: str
    identity_primer: PolishIdentityPrimerSpec


class PolishPlanPathSpec(StrictModel):
    path: str


class PolishPlannerSpec(StrictModel):
    max_regions: int = 4


class PolishMicroSweepSpec(StrictModel):
    strength_offsets: list[float]
    seed_offsets: list[int]


class PolishOutputSpec(StrictModel):
    directory: str
    overwrite: bool
    save_debug_images: bool
    save_contact_sheet: bool


class PolishAcceptanceSpec(StrictModel):
    manual: list[str]


class KeyframePolishJobSpec(StrictModel):
    schema_path: str = Field(alias="$schema")
    schema_version: Literal[1]
    kind: Literal["keyframe-polish"]
    id: str
    pipeline: PolishPipelineSpec
    base: PolishBaseSpec
    character: PolishCharacterSpec
    plan: PolishPlanPathSpec
    planner: PolishPlannerSpec
    micro_sweep: PolishMicroSweepSpec
    output: PolishOutputSpec
    acceptance: PolishAcceptanceSpec


class PlannedPolishParameters(StrictModel):
    strength: float
    steps: int
    guidance_scale: float
    true_cfg_scale: float
    feather_px: int
    crop_padding_px: int
    crop_upsample_factor: float
    max_sequence_length: int


class PlannedPolishRegion(StrictModel):
    id: str
    label: str
    bbox: tuple[int, int, int, int]
    mask_prompt: str
    operation: Literal[
        "detail_restore",
        "expression_refine",
        "identity_restore",
        "lineart_sharpen",
        "color_restore",
        "shape_fix",
        "hand_fix",
        "artifact_remove",
    ]
    reason: str
    reference_crop_requirements: list[str]
    parameters: PlannedPolishParameters
    prompt: str
    negative_prompt: str
    must_not_change: list[str]
    acceptance_checks: list[str]

    @field_validator("reference_crop_requirements", mode="before")
    @classmethod
    def _listify_reference_crop_requirements(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value


class KeyframePolishPlan(StrictModel):
    schema_version: Literal[1]
    kind: Literal["keyframe-polish-plan"]
    job_id: str
    base_candidate: str
    needs_polish: bool
    regions: list[PlannedPolishRegion]
    summary: str


class PolishSelectionCheck(StrictModel):
    target_detail_restored: bool
    identity_preserved: bool
    outside_mask_changed: bool
    pose_changed: bool
    style_match: bool


class PolishRegionSelection(StrictModel):
    region_id: str
    best_variant: str
    passes: bool
    checks: PolishSelectionCheck
    reason: str
