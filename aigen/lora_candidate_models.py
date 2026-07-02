from __future__ import annotations

import re
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
        return cleaned


class LoraCandidateTemplateSpec(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    view: str = Field(min_length=1)
    pose: str = Field(min_length=1)
    framing: str = Field(min_length=1)
    identity_primer: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    prompt: LoraCandidatePromptSpec

    @field_validator("name")
    @classmethod
    def semantic_name(cls, value: str) -> str:
        if value.rsplit("_", 1)[-1].isdigit():
            raise ValueError("candidate name must be semantic, not numbered")
        banned = {"crop", "close", "closeup", "close_up", "partial"}
        if any(part.lower() in banned for part in value.replace("-", "_").replace(".", "_").split("_")):
            raise ValueError("candidate name must describe an intentional view/pose/framing, not a crop or close-up")
        return value

    @field_validator("view", "pose")
    @classmethod
    def concrete_label(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if cleaned.lower() in {"", ".", "...", "unknown", "n/a"}:
            raise ValueError("candidate fields must be concrete")
        lowered = cleaned.lower()
        if any(term in lowered for term in _NON_VIEW_POSE_TERMS):
            raise ValueError("view and pose must describe camera/pose only")
        return cleaned

    @field_validator("framing")
    @classmethod
    def concrete_framing(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if cleaned.lower() in {"", ".", "...", "unknown", "n/a"}:
            raise ValueError("candidate framing must be concrete")
        if not _framing_groups(cleaned):
            raise ValueError(
                "candidate framing must describe visible body coverage, such as full body, thigh-up, waist-up or portrait"
            )
        return cleaned

    @model_validator(mode="after")
    def prompt_materializes_view_pose_and_framing(self) -> LoraCandidateTemplateSpec:
        prompt = self.prompt.positive.lower()
        view = self.view.lower()
        name_parts = _name_parts(self.name)
        for term in ("front", "left", "right", "back"):
            if term in name_parts and term not in view:
                raise ValueError(f"candidate name view term does not match view field: {term}")
        missing_view_terms = _missing_view_terms(self.view, prompt)
        if missing_view_terms:
            raise ValueError(f"candidate prompt must explicitly include view term: {missing_view_terms[0]}")
        missing_framing_terms = _missing_framing_terms(self.framing, prompt)
        if missing_framing_terms:
            raise ValueError(f"candidate prompt must explicitly include framing term: {missing_framing_terms[0]}")
        if "looking at viewer" in prompt and any(term in view for term in ("profile", "back", "rear")):
            raise ValueError("candidate prompt must not include front-view gaze language for profile or rear views")
        rear_forbidden_terms = (
            "blue eyes",
            "eye",
            "eyes",
            "smile",
            "smiling",
            "blush",
            "looking at viewer",
            "flat-chested",
            "small breasts",
        )
        if any(term in view for term in ("back", "rear")) and any(_contains_term(prompt, term) for term in rear_forbidden_terms):
            raise ValueError(
                "candidate prompt must not request front-facing facial details for rear views: "
                + ", ".join(rear_forbidden_terms)
            )
        pose_terms = _semantic_terms(self.pose)
        if pose_terms and not any(_contains_term(prompt, term) for term in pose_terms):
            raise ValueError("candidate prompt must explicitly describe the requested pose")
        return self


class LoraCandidateTemplateListSpec(StrictModel):
    candidates: list[LoraCandidateTemplateSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def candidate_contract_is_unique(self) -> LoraCandidateTemplateListSpec:
        names = [candidate.name for candidate in self.candidates]
        if len(set(names)) != len(names):
            raise ValueError("candidate names must be unique")
        view_pose_pairs = [
            (
                " ".join(candidate.view.split()).lower(),
                " ".join(candidate.pose.split()).lower(),
                " ".join(candidate.framing.split()).lower(),
            )
            for candidate in self.candidates
        ]
        if len(set(view_pose_pairs)) != len(view_pose_pairs):
            raise ValueError("candidate view/pose/framing tuples must be unique")
        prompts = [" ".join(candidate.prompt.positive.split()).lower() for candidate in self.candidates]
        if len(set(prompts)) != len(prompts):
            raise ValueError("candidate generation prompts must be unique")
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
    def candidate_contract_is_unique(self) -> LoraCandidateBriefSpec:
        LoraCandidateTemplateListSpec(candidates=self.candidates)
        return self


class LoraFreeGenBucketSpec(StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @model_validator(mode="after")
    def latent_compatible(self) -> LoraFreeGenBucketSpec:
        for label, value in (("width", self.width), ("height", self.height)):
            if value % 16:
                raise ValueError(f"bucket {label} must be divisible by 16: {value}")
        return self


class LoraFreeGenGenerationSpec(StrictModel):
    buckets: list[LoraFreeGenBucketSpec] = Field(min_length=1)
    steps: int = Field(gt=0)
    seed_start: int = Field(ge=0)
    seeds_per_bucket: int = Field(gt=0)

    @model_validator(mode="after")
    def buckets_are_unique(self) -> LoraFreeGenGenerationSpec:
        sizes = [(bucket.width, bucket.height) for bucket in self.buckets]
        if len(set(sizes)) != len(sizes):
            raise ValueError("free-generation buckets must be unique")
        return self


class LoraFreeGenBriefSpec(StrictModel):
    schema_path: str = Field(alias="$schema")
    kind: Literal["lora-freegen-brief"]
    id: str
    character: LoraCandidateCharacterSpec
    generation: LoraFreeGenGenerationSpec
    identity_primers: list[str] | None = Field(default=None, min_length=1)
    output: LoraCandidateOutputSpec


def lora_candidate_brief_schema() -> dict[str, Any]:
    schema = LoraCandidateBriefSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def lora_freegen_brief_schema() -> dict[str, Any]:
    schema = LoraFreeGenBriefSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def _missing_view_terms(view: str, prompt: str) -> list[str]:
    groups = []
    lowered = view.lower()
    if "left" in lowered:
        groups.append(("left", "left-facing"))
    if "right" in lowered:
        groups.append(("right", "right-facing"))
    if "front" in lowered:
        groups.append(("front", "front-facing"))
    if "back" in lowered or "rear" in lowered:
        groups.append(("back", "rear", "back-facing", "rear-facing"))
    if "quarter" in lowered:
        groups.append(("quarter", "three-quarter", "3/4"))
    missing = []
    for group in groups:
        if not any(_contains_term(prompt, term) for term in group):
            missing.append(group[0])
    return missing


def _missing_framing_terms(framing: str, prompt: str) -> list[str]:
    missing = []
    for group in _framing_groups(framing):
        if not any(_contains_term(prompt, term) for term in group):
            missing.append(group[0])
    return missing


def _framing_groups(framing: str) -> list[tuple[str, ...]]:
    lowered = framing.lower()
    groups: list[tuple[str, ...]] = []
    if "full" in lowered and "body" in lowered:
        groups.append(("full body", "full-body", "entire body", "entire figure", "head-to-toe", "head to toe"))
    if "thigh" in lowered:
        groups.append(("thigh-up", "thigh up", "upper thighs", "mid-thigh"))
    if "knee" in lowered:
        groups.append(("knee-up", "knee up", "knees-up", "knees up"))
    if "waist" in lowered:
        groups.append(("waist-up", "waist up", "upper body"))
    if "bust" in lowered:
        groups.append(("bust portrait", "bust-up", "bust up", "chest-up", "chest up"))
    if "head" in lowered or "shoulder" in lowered:
        groups.append(("head-and-shoulders", "head and shoulders", "shoulders-up", "shoulders up"))
    if "portrait" in lowered and not groups:
        groups.append(("portrait",))
    return groups


def _name_parts(value: str) -> set[str]:
    return {part for part in value.lower().replace("-", "_").replace(".", "_").split("_") if part}


def _contains_term(text: str, term: str) -> bool:
    if term == "3/4":
        return term in text
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def _semantic_terms(value: str) -> list[str]:
    stop_words = {"view", "pose", "mild", "simple", "neutral", "relaxed"}
    return [
        token
        for token in value.lower().replace("-", " ").replace("_", " ").split()
        if len(token) > 3 and token not in stop_words
    ]


_NON_VIEW_POSE_TERMS = (
    "background",
    "lineart",
    "full body",
    "full-body",
    "entire body",
    "entire figure",
    "head-to-toe",
    "head to toe",
    "thigh-up",
    "thigh up",
    "knee-up",
    "knee up",
    "waist-up",
    "waist up",
    "upper body",
    "bust",
    "portrait",
    "crop",
)


def load_lora_candidate_brief(path: Path) -> LoraCandidateBriefSpec:
    try:
        return LoraCandidateBriefSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise LoraCandidateBriefError(f"Invalid LoRA candidate brief {path}: {error}") from error


def load_lora_freegen_brief(path: Path) -> LoraFreeGenBriefSpec:
    try:
        return LoraFreeGenBriefSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise LoraCandidateBriefError(f"Invalid LoRA free-generation brief {path}: {error}") from error
