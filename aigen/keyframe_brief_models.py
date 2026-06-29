from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from aigen.keyframe_job_models import (
    CanvasSpec,
    IdentityPrimerSpec,
    PathSpec,
    PipelineSpec,
    PromptSpec,
    SamplingSpec,
)


KEYFRAME_BRIEF_SCHEMA = "schemas/keyframe-brief.schema.json"
KEYFRAME_BRIEF_PLAN_SCHEMA = "schemas/keyframe-brief-plan.schema.json"
KEYFRAME_BRIEF_KIND = "keyframe-brief"
KEYFRAME_BRIEF_PLAN_KIND = "keyframe-brief-plan"
KEYFRAME_BRIEF_SCHEMA_VERSION = 1
KEYFRAME_BRIEF_PLAN_SCHEMA_VERSION = 1


class KeyframeBriefError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BriefCharacterSpec(StrictModel):
    id: str
    view_bank: PathSpec


class BriefRequestSpec(StrictModel):
    action: str
    phase: str
    direction: Literal["left", "right"]
    camera: Literal["platformer-side-view"]
    description: str


class BriefExampleSpec(StrictModel):
    path: str
    name: str
    width: int
    height: int
    mirror_x: bool


class BriefGenerationSpec(StrictModel):
    seed_start: int
    seed_count: int
    output_directory: str
    filename: str
    overwrite: bool
    save_conditions: bool
    save_contact_sheet: bool


class BriefOutputSpec(StrictModel):
    assets_directory: str
    plan_path: str
    job_path: str


class KeyframeBriefSpec(StrictModel):
    schema_path: str = Field(alias="$schema")
    schema_version: Literal[1]
    kind: Literal["keyframe-brief"]
    id: str
    pipeline: PipelineSpec
    character: BriefCharacterSpec
    request: BriefRequestSpec
    example: BriefExampleSpec
    generation: BriefGenerationSpec
    output: BriefOutputSpec


class BriefControlPlanSpec(StrictModel):
    name: str
    type: Literal["pose", "canny", "softedge", "gray"]
    source: Literal["example_pose", "example_canny_lineart", "example_softedge", "example_gray"]
    scale: float = Field(ge=0.0, le=1.0)
    start: float = Field(ge=0.0, le=1.0)
    end: float = Field(ge=0.0, le=1.0)
    residual_mask_source: Literal["example_boundary_mask", "example_full_silhouette_mask", "example_arm_hand_mask"] | None = None

    @model_validator(mode="after")
    def active_window_is_ordered(self) -> BriefControlPlanSpec:
        if self.start >= self.end:
            raise ValueError("control start must be before end")
        if self.source == "example_pose" and self.type != "pose":
            raise ValueError("example_pose controls must use type pose")
        expected_types = {
            "example_canny_lineart": "canny",
            "example_softedge": "softedge",
            "example_gray": "gray",
        }
        if self.source in expected_types and self.type != expected_types[self.source]:
            raise ValueError(f"{self.source} controls must use type {expected_types[self.source]}")
        if self.source == "example_pose" and self.residual_mask_source is not None:
            raise ValueError("pose controls must not use residual_mask_source")
        return self


class BriefIdentityDetailsSpec(StrictModel):
    subject: str
    hair: str
    face: str
    upper_clothing: str
    neckwear: str
    waist_garment: str
    legwear: str
    footwear: str
    style: str

    @field_validator(
        "subject",
        "hair",
        "face",
        "upper_clothing",
        "neckwear",
        "waist_garment",
        "legwear",
        "footwear",
        "style",
    )
    @classmethod
    def concrete_identity_slot(cls, value: str) -> str:
        placeholders = {"", ".", "...", "unknown", "n/a"}
        if value.strip().lower() in placeholders:
            raise ValueError("identity detail slots must be concrete visual descriptions")
        return value


class BriefScoringPlanSpec(StrictModel):
    top_k: int
    priorities: list[str]
    checks: list[str]

    @model_validator(mode="after")
    def top_k_selects_candidates(self) -> BriefScoringPlanSpec:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        return self

    @field_validator("priorities", "checks")
    @classmethod
    def no_placeholder_items(cls, values: list[str]) -> list[str]:
        placeholders = {
            "",
            ".",
            "...",
            "concrete visual check",
            "condition-first visual priority",
        }
        if any(value.strip().lower() in placeholders for value in values):
            raise ValueError("scoring priorities and checks must be concrete")
        return values


class BriefPolishPlanSpec(StrictModel):
    profile: Literal["kontext-inpaint-local"]
    max_regions: int = Field(ge=1)
    strength_offsets: list[float] = Field(min_length=1)
    seed_offsets: list[int] = Field(min_length=1)


class KeyframeBriefPlanSpec(StrictModel):
    schema_path: str = Field(alias="$schema")
    schema_version: Literal[1]
    kind: Literal["keyframe-brief-plan"]
    brief_id: str
    planner_id: str
    source_brief_sha256: str
    planner_prompt_sha256: str
    identity_details: BriefIdentityDetailsSpec
    identity_description: str
    pose_description: str
    platformer_camera_description: str
    identity_primer: IdentityPrimerSpec
    prompt: PromptSpec
    canvas: CanvasSpec
    sampling: SamplingSpec
    controls: list[BriefControlPlanSpec]
    scoring: BriefScoringPlanSpec
    polish: BriefPolishPlanSpec
    rationale: list[str]

    @model_validator(mode="after")
    def negative_prompt_requires_active_true_cfg(self) -> KeyframeBriefPlanSpec:
        if self.prompt.negative is not None and self.prompt.true_cfg_scale <= 1.0:
            raise ValueError("prompt.negative requires true_cfg_scale > 1.0")
        crop_phrases = ("waist up", "waist-up", "upper body", "bust shot", "portrait crop")
        text = " ".join((self.platformer_camera_description, self.prompt.clip, self.prompt.t5)).lower()
        if any(phrase in text for phrase in crop_phrases):
            raise ValueError("platformer keyframes must stay full-body, not cropped")
        return self


def keyframe_brief_schema() -> dict[str, object]:
    schema = KeyframeBriefSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def keyframe_brief_plan_schema() -> dict[str, object]:
    schema = KeyframeBriefPlanSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_keyframe_brief(path: Path) -> KeyframeBriefSpec:
    try:
        return KeyframeBriefSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise KeyframeBriefError(f"Invalid keyframe brief {path}: {error}") from error


def load_keyframe_brief_plan(path: Path) -> KeyframeBriefPlanSpec:
    try:
        return KeyframeBriefPlanSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise KeyframeBriefError(f"Invalid keyframe brief plan {path}: {error}") from error
