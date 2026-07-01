from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


LORA_CANDIDATE_BRIEF_SCHEMA = "schemas/lora-candidate-brief.schema.json"


class LoraCandidateBriefError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoraCandidateCharacterSpec(StrictModel):
    canon: str


class LoraCandidateGenerationSpec(StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    steps: int = Field(gt=0)
    seed_start: int = Field(ge=0)
    seeds_per_candidate: int = Field(gt=0)


class LoraCandidatePromptSpec(StrictModel):
    positive: str = Field(min_length=1)

    @field_validator("positive")
    @classmethod
    def concrete_prompt(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if cleaned.lower() in {"", ".", "...", "prompt", "candidate prompt"}:
            raise ValueError("candidate prompt must be a concrete visual instruction")
        lowered = cleaned.lower()
        if "full body" not in lowered and "full-body" not in lowered:
            raise ValueError("candidate prompt must explicitly request full body")
        if "background" not in lowered:
            raise ValueError("candidate prompt must explicitly describe the background")
        if "anime" not in lowered and "lineart" not in lowered:
            raise ValueError("candidate prompt must explicitly describe the visual style")
        return cleaned


class LoraCandidateTemplateSpec(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    view: str = Field(min_length=1)
    pose: str = Field(min_length=1)
    identity_primer: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    prompt: LoraCandidatePromptSpec

    @field_validator("name")
    @classmethod
    def semantic_name(cls, value: str) -> str:
        if value.rsplit("_", 1)[-1].isdigit():
            raise ValueError("candidate name must be semantic, not numbered")
        return value

    @field_validator("view", "pose")
    @classmethod
    def concrete_label(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if cleaned.lower() in {"", ".", "...", "unknown", "n/a"}:
            raise ValueError("candidate fields must be concrete")
        lowered = cleaned.lower()
        if "background" in lowered or "lineart" in lowered or "full body" in lowered or "full-body" in lowered:
            raise ValueError("view and pose must describe camera/pose only")
        return cleaned


class LoraCandidateTemplateListSpec(StrictModel):
    candidates: list[LoraCandidateTemplateSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def candidate_names_are_unique(self) -> LoraCandidateTemplateListSpec:
        names = [candidate.name for candidate in self.candidates]
        if len(set(names)) != len(names):
            raise ValueError("candidate names must be unique")
        view_pose_pairs = [
            (" ".join(candidate.view.split()).lower(), " ".join(candidate.pose.split()).lower())
            for candidate in self.candidates
        ]
        if len(set(view_pose_pairs)) != len(view_pose_pairs):
            raise ValueError("candidate view/pose pairs must be unique")
        return self


class LoraCandidateOutputSpec(StrictModel):
    directory: str
    overwrite: bool


class LoraCandidateBriefSpec(StrictModel):
    schema_path: str = Field(alias="$schema")
    kind: Literal["lora-candidate-brief"]
    id: str
    character: LoraCandidateCharacterSpec
    generation: LoraCandidateGenerationSpec
    candidates: list[LoraCandidateTemplateSpec] = Field(min_length=1)
    output: LoraCandidateOutputSpec

    @model_validator(mode="after")
    def candidate_names_are_unique(self) -> LoraCandidateBriefSpec:
        LoraCandidateTemplateListSpec(candidates=self.candidates)
        return self


def lora_candidate_brief_schema() -> dict[str, Any]:
    schema = LoraCandidateBriefSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_lora_candidate_brief(path: Path) -> LoraCandidateBriefSpec:
    try:
        return LoraCandidateBriefSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise LoraCandidateBriefError(f"Invalid LoRA candidate brief {path}: {error}") from error
