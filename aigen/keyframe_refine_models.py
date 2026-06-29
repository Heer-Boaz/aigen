from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


KEYFRAME_REFINE_JOB_SCHEMA = "schemas/keyframe-refine-job.schema.json"


class KeyframeRefineError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RefinePipelineSpec(StrictModel):
    profile: str


class RefineBaseSpec(StrictModel):
    run_dir: str
    candidate: str


class PathSpec(StrictModel):
    path: str


class IdentityPrimerSpec(StrictModel):
    view: str
    path: str


class RefineCharacterSpec(StrictModel):
    id: str
    identity_primer: IdentityPrimerSpec


class RefineMaskSourceSpec(StrictModel):
    type: Literal["pose_contour_auto"]
    pose: str
    contour: str
    candidate_foreground: bool


class RefineRegionSpec(StrictModel):
    name: str
    mask_source: RefineMaskSourceSpec
    dilate_px: int
    feather_px: int
    crop_padding_px: int


class RefinePromptSpec(StrictModel):
    clip: str
    t5: str
    negative: str | None = None
    true_cfg_scale: float


class RefineSamplingSpec(StrictModel):
    steps: int
    guidance_scale: float
    strength: float
    max_sequence_length: int


class RefineVariantSpec(StrictModel):
    name: str
    seed: int


class RefineOutputSpec(StrictModel):
    directory: str
    filename: str
    overwrite: bool
    save_debug_images: bool
    save_contact_sheet: bool


class RefineAcceptanceSpec(StrictModel):
    manual: list[str]


class KeyframeRefineJobSpec(StrictModel):
    schema_path: str = Field(alias="$schema")
    kind: Literal["keyframe-refine"]
    id: str
    pipeline: RefinePipelineSpec
    base: RefineBaseSpec
    character: RefineCharacterSpec
    region: RefineRegionSpec
    prompt: RefinePromptSpec
    sampling: RefineSamplingSpec
    variants: list[RefineVariantSpec]
    output: RefineOutputSpec
    acceptance: RefineAcceptanceSpec


def keyframe_refine_job_schema() -> dict[str, Any]:
    schema = KeyframeRefineJobSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_keyframe_refine_job(path: Path) -> KeyframeRefineJobSpec:
    try:
        return KeyframeRefineJobSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise KeyframeRefineError(f"Invalid keyframe refine job {path}: {error}") from error
