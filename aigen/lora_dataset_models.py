from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


LORA_DATASET_SCHEMA = "schemas/lora-dataset.schema.json"


class LoraDatasetError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoraDatasetCharacterSpec(StrictModel):
    id: str
    trigger_token: str


class ViewBankLoraSourceSpec(StrictModel):
    type: Literal["view_bank"]
    path: str
    views: list[str] | None = None
    caption: str | None = None
    tags: list[str] = Field(default_factory=list)
    split: Literal["train", "val"] | None = None


class KeyframeRunLoraSourceSpec(StrictModel):
    type: Literal["keyframe_run"]
    run_dir: str
    selection_path: str | None = None
    candidates: list[str] | None = None
    caption: str | None = None
    tags: list[str] = Field(default_factory=list)
    split: Literal["train", "val"] | None = None

    @model_validator(mode="after")
    def require_explicit_selection(self) -> KeyframeRunLoraSourceSpec:
        if not self.selection_path and not self.candidates:
            raise ValueError("keyframe_run sources require selection_path or candidates")
        return self


class ImageLoraSourceSpec(StrictModel):
    type: Literal["image"]
    path: str
    caption: str
    tags: list[str] = Field(default_factory=list)
    split: Literal["train", "val"] | None = None


LoraDatasetSourceSpec = Annotated[
    ViewBankLoraSourceSpec | KeyframeRunLoraSourceSpec | ImageLoraSourceSpec,
    Field(discriminator="type"),
]


class LoraDatasetOutputSpec(StrictModel):
    directory: str
    overwrite: bool
    validation_ratio: float = Field(ge=0.0, lt=1.0)
    save_contact_sheet: bool


class LoraDatasetSpec(StrictModel):
    schema_path: str = Field(alias="$schema")
    kind: Literal["lora-dataset"]
    id: str
    character: LoraDatasetCharacterSpec
    sources: list[LoraDatasetSourceSpec]
    output: LoraDatasetOutputSpec


def lora_dataset_schema() -> dict[str, Any]:
    schema = LoraDatasetSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_lora_dataset_spec(path: Path) -> LoraDatasetSpec:
    try:
        return LoraDatasetSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise LoraDatasetError(f"Invalid LoRA dataset spec {path}: {error}") from error
