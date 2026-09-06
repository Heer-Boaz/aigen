from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from aigen.generation.image_edit_batch import ImageEditBatchOutput
from aigen.manifest_io import atomic_write_json, file_manifest, sha256_bytes
from aigen.vlm_json import VlmJsonError, json_object_from_vlm_response
from aigen.vlm_qwen import (
    DEFAULT_JUDGE_ID, DEFAULT_JUDGE_QUANTIZATION, DEFAULT_JUDGE_REPO_ID,
    DEFAULT_JUDGE_REVISION, DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS,
    DEFAULT_QWEN_VLM_MODEL, QwenVlm, QwenVlmConfig,
)


CHARACTER_AUDIT_REVISION = "3"


class CharacterAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class AuditContextImage:
    role: str
    path: Path


@dataclass(frozen=True)
class CharacterAuditGroup:
    id: str
    instruction: str
    route: str
    references: tuple[Path, ...]
    context_images: tuple[AuditContextImage, ...]
    candidate_ids: tuple[str, ...]


class CharacterAuditVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    passed: bool
    image_index: int = Field(ge=1)
    regions: list[str]

    @model_validator(mode="after")
    def consistent_verdict(self) -> CharacterAuditVerdict:
        if self.passed and self.regions:
            raise ValueError("a passing candidate cannot have defect regions")
        if not self.passed and not self.regions:
            raise ValueError("a failing candidate must identify its defect regions")
        if any(not region.strip() for region in self.regions):
            raise ValueError("defect region pointers cannot be empty")
        return self


@dataclass(frozen=True)
class CharacterAuditResult:
    verdict: CharacterAuditVerdict
    selected: ImageEditBatchOutput
    candidate_identity: str
    report: Path


def default_character_audit_config() -> QwenVlmConfig:
    return QwenVlmConfig(
        judge_id=DEFAULT_JUDGE_ID, model=DEFAULT_QWEN_VLM_MODEL,
        repo_id=DEFAULT_JUDGE_REPO_ID, revision=DEFAULT_JUDGE_REVISION,
        dtype="bfloat16", attention_impl="sdpa", quantization=DEFAULT_JUDGE_QUANTIZATION,
        min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS,
        max_new_tokens=256, temperature=0.0,
    )


def character_audit_prompt(group: CharacterAuditGroup, candidate_count: int) -> str:
    roles = [f"Image {index}: reference." for index in range(1, len(group.references) + 1)]
    offset = len(roles)
    roles.extend(f"Image {offset + index}: {context.role}." for index, context in enumerate(group.context_images, 1))
    offset = len(roles)
    candidate_numbers = tuple(range(offset + 1, offset + candidate_count + 1))
    roles.extend(f"Image {number}: candidate." for number in candidate_numbers)
    return (
        "Select the candidate that best fulfills the requested edit while remaining visually faithful "
        "to the character references. Judge the requested changes and their natural visual consequences "
        "as intended, including any requested change of style. Pass only if the requested visual change "
        "is visibly complete and there are no unintended visual deviations. A missing or incomplete "
        "requested change is a failure, even when the references are preserved. If every candidate has defects, "
        "select the closest one and fail it.\n"
        f"Requested edit: {group.instruction}\nRoute: {group.route}\n"
        + "\n".join(roles)
        + "\nReturn one JSON object with exactly these keys: passed (boolean for the selected candidate), "
        f"image_index (the selected candidate's image number: {', '.join(map(str, candidate_numbers))}), "
        "regions (short pointers locating its visible defects, "
        "or an empty list when passed). Return region names only, without descriptions or repair instructions."
    )


def audit_character_candidates(
    judge: QwenVlm,
    group: CharacterAuditGroup,
    candidates: Sequence[ImageEditBatchOutput],
    directory: Path,
) -> CharacterAuditResult:
    paths = (*group.references, *(context.path for context in group.context_images), *(item.path for item in candidates))
    prompt = character_audit_prompt(group, len(candidates))
    request = {
        "revision": CHARACTER_AUDIT_REVISION, "group_id": group.id,
        "prompt": prompt, "images": [file_manifest(path) for path in paths],
        "candidates": [item.model_dump(mode="json") for item in candidates],
    }
    atomic_write_json(directory / "request.json", request)
    raw = judge.judge_candidate(prompt, list(paths))
    response_path = directory / "response.json"
    atomic_write_json(response_path, {"raw_response": raw})
    try:
        verdict = CharacterAuditVerdict.model_validate(json_object_from_vlm_response(raw))
        candidate_index = verdict.image_index - len(group.references) - len(group.context_images) - 1
        if not 0 <= candidate_index < len(candidates):
            raise ValueError(f"image index {verdict.image_index} does not identify a candidate in this audit batch")
    except (VlmJsonError, ValidationError, ValueError) as error:
        raise CharacterAuditError(f"Invalid character audit result; see {response_path}: {error}") from error
    selected = candidates[candidate_index]
    selected_file = request["images"][verdict.image_index - 1]
    identity = sha256_bytes(json.dumps({
        "case_id": selected.case_id, "seed": selected.seed, "sha256": selected_file["sha256"],
    }, sort_keys=True).encode())
    report = directory / "result.json"
    atomic_write_json(report, {
        "group_id": group.id, **verdict.model_dump(mode="json"),
        "candidate_identity": identity, "selected": selected.model_dump(mode="json"),
        "request": str(directory / "request.json"), "response": str(response_path),
    })
    return CharacterAuditResult(verdict, selected, identity, report)
