from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


LORA_DATASET_SCHEMA = "schemas/lora-dataset.schema.json"


class LoraDatasetError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoraDatasetCharacterSpec(StrictModel):
    id: str
    trigger_token: str = Field(min_length=1)


class CanonLoraSourceSpec(StrictModel):
    type: Literal["canon"]
    path: str
    images: list[str] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    split: Literal["train", "val"] | None = None


class CandidateReviewLoraSourceSpec(StrictModel):
    type: Literal["candidate_review"]
    path: str
    tags: list[str] = Field(default_factory=list)
    split: Literal["train", "val"] | None = None


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
    sources: list[CanonLoraSourceSpec | CandidateReviewLoraSourceSpec]
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
