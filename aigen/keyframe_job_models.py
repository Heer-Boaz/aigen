from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


KEYFRAME_JOB_SCHEMA = "schemas/keyframe-job.schema.json"
KEYFRAME_SCHEMA_VERSION = 1


class KeyframeJobError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PipelineSpec(StrictModel):
    profile: str


class PathSpec(StrictModel):
    path: str


class IdentityPrimerSpec(StrictModel):
    view: Literal["front", "left_profile", "right_profile", "back"]
    path: str


class CharacterSpec(StrictModel):
    id: str
    identity_primer: IdentityPrimerSpec


class KeyframeSpec(StrictModel):
    action: str
    phase: str
    direction: Literal["left", "right"]
    camera: Literal["orthographic-side"]


class AssetSpec(StrictModel):
    pose: PathSpec
    contour: PathSpec | None = None
    canny_lineart: PathSpec | None = None
    boundary_mask: PathSpec | None = None
    depth: PathSpec | None = None
    softedge: PathSpec | None = None
    gray: PathSpec | None = None
    filled_silhouette: PathSpec | None = None
    full_silhouette_mask: PathSpec | None = None
    arm_hand_mask: PathSpec | None = None


class PromptSpec(StrictModel):
    clip: str
    t5: str
    negative: str | None = None
    true_cfg_scale: float


class CanvasSpec(StrictModel):
    width: int
    height: int
    reference_max_area: int
    max_sequence_length: int


class SamplingSpec(StrictModel):
    steps: int
    guidance_scale: float


class ControlConditionSpec(StrictModel):
    name: str
    type: Literal["pose", "canny", "softedge", "depth", "gray"]
    image: str
    scale: float
    start: float
    end: float
    residual_mask: str | None = None


class VariantSpec(StrictModel):
    name: str
    seed: int


class OutputSpec(StrictModel):
    directory: str
    filename: str
    overwrite: bool
    save_conditions: bool
    save_contact_sheet: bool


class AcceptanceSpec(StrictModel):
    manual: list[str]
    minimum_passing_variants: int


class KeyframeJobSpec(StrictModel):
    schema_path: str = Field(alias="$schema")
    schema_version: Literal[1]
    kind: Literal["character-keyframe"]
    id: str
    pipeline: PipelineSpec
    character: CharacterSpec
    keyframe: KeyframeSpec
    assets: AssetSpec
    prompt: PromptSpec
    canvas: CanvasSpec
    sampling: SamplingSpec
    conditions: list[ControlConditionSpec]
    variants: list[VariantSpec]
    output: OutputSpec
    acceptance: AcceptanceSpec


def keyframe_job_schema() -> dict[str, Any]:
    schema = KeyframeJobSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_keyframe_job(path: Path) -> KeyframeJobSpec:
    try:
        return KeyframeJobSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise KeyframeJobError(f"Invalid keyframe job {path}: {error}") from error
