from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


CHARACTER_VIEW_JOB_SCHEMA = "schemas/character-view-job.schema.json"
CHARACTER_VIEW_BANK_SCHEMA = "schemas/character-view-bank.schema.json"
VIEW_BANK_KIND = "character-view-bank"

class CharacterViewError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathSpec(StrictModel):
    path: str


class ImageAssetSpec(StrictModel):
    path: str
    sha256: str
    mode: str
    width: int
    height: int


class PipelineSpec(StrictModel):
    profile: str


class SourceCharacterSpec(StrictModel):
    id: str
    source_reference: PathSpec


class CharacterViewSpec(StrictModel):
    name: str
    camera: str
    pose: str


class ViewAssetsSpec(StrictModel):
    pose: PathSpec
    contour: PathSpec
    boundary_mask: PathSpec


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
    type: Literal["pose", "canny", "softedge", "depth"]
    image: str
    scale: float
    start: float
    end: float
    residual_mask: str | None = None


class VariantSpec(StrictModel):
    name: str
    seed: int


class ViewOutputSpec(StrictModel):
    directory: str
    filename: str
    canonical_path: str
    bank_path: str
    overwrite: bool
    save_conditions: bool
    save_contact_sheet: bool


class AcceptanceSpec(StrictModel):
    manual: list[str]
    minimum_passing_variants: int


class ViewBankCharacterSpec(StrictModel):
    id: str
    source_reference: ImageAssetSpec


class ViewBankViewSpec(StrictModel):
    name: str
    camera: str
    pose: str


class ViewBankEntryAssetsSpec(StrictModel):
    pose: ImageAssetSpec | None = None
    contour: ImageAssetSpec | None = None
    boundary_mask: ImageAssetSpec | None = None
    depth: ImageAssetSpec | None = None
    softedge: ImageAssetSpec | None = None


class ViewBankEntryAcceptanceSpec(StrictModel):
    manual: list[str]
    minimum_passing_variants: int | None = None


class ViewBankEntrySpec(StrictModel):
    view: ViewBankViewSpec
    image: ImageAssetSpec
    accepted_candidate: str
    accepted_seed: int | None = None
    assets: ViewBankEntryAssetsSpec | None = None
    acceptance: ViewBankEntryAcceptanceSpec | None = None


class CharacterViewBankSpec(StrictModel):
    kind: Literal["character-view-bank"]
    character: ViewBankCharacterSpec
    views: dict[str, ViewBankEntrySpec]


class CharacterViewJobSpec(StrictModel):
    schema_path: str = Field(alias="$schema")
    kind: Literal["character-view"]
    id: str
    pipeline: PipelineSpec
    character: SourceCharacterSpec
    view: CharacterViewSpec
    assets: ViewAssetsSpec
    prompt: PromptSpec
    canvas: CanvasSpec
    sampling: SamplingSpec
    conditions: list[ControlConditionSpec]
    variants: list[VariantSpec]
    output: ViewOutputSpec
    acceptance: AcceptanceSpec


def character_view_job_schema() -> dict[str, Any]:
    schema = CharacterViewJobSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def character_view_bank_schema() -> dict[str, Any]:
    schema = CharacterViewBankSpec.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_character_view_job(path: Path) -> CharacterViewJobSpec:
    try:
        return CharacterViewJobSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise CharacterViewError(f"Invalid character view job {path}: {error}") from error


def load_character_view_bank(path: Path) -> CharacterViewBankSpec:
    try:
        return CharacterViewBankSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise CharacterViewError(f"Invalid character view bank {path}: {error}") from error
