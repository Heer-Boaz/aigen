from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from aigen.manifest_io import read_json, sha256_file
from aigen.pix2pix.corpus_config import SAFE_NAME_PATTERN
from aigen.pix2pix.corpus_io import (
    corpus_member,
    read_json_records,
)
from aigen.pix2pix.errors import Pix2PixError


FLUX_SOURCE_SET_FORMAT = "aigen.pix2pix.flux-source-set.v1"
FLUX_SOURCE_SET_FROZEN_FORMAT = "aigen.pix2pix.frozen-flux-source-set.v1"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
PROMPT_GUIDE_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "prompting.md"
)


class _SourceSetModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FluxSourceSetRecord(_SourceSetModel):
    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    author: str = Field(min_length=1)
    seed: int = Field(ge=0)
    target_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("prompt", "author")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("FLUX source-set text has surrounding whitespace")
        return value


class FluxPromptReviewRecord(_SourceSetModel):
    id: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    target_sha256: str = Field(pattern=SHA256_PATTERN)
    verdict: Literal["pass"]
    reviewer: str = Field(min_length=1)

    @field_validator("reviewer")
    @classmethod
    def validate_reviewer(cls, reviewer: str) -> str:
        if reviewer != reviewer.strip():
            raise ValueError("prompt reviewer has surrounding whitespace")
        return reviewer


class FluxSourceSetManifest(_SourceSetModel):
    format: Literal["aigen.pix2pix.flux-source-set.v1"]
    name: str = Field(pattern=SAFE_NAME_PATTERN)
    selection_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_guide_sha256: str = Field(pattern=SHA256_PATTERN)
    records: str = Field(min_length=1)
    records_sha256: str = Field(pattern=SHA256_PATTERN)
    record_count: int = Field(gt=0)
    prompt_reviews: str = Field(min_length=1)
    prompt_reviews_sha256: str = Field(pattern=SHA256_PATTERN)


class FrozenFluxSourceSet(_SourceSetModel):
    format: Literal["aigen.pix2pix.frozen-flux-source-set.v1"]
    name: str = Field(pattern=SAFE_NAME_PATTERN)
    selection_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_guide_sha256: str = Field(pattern=SHA256_PATTERN)
    records: tuple[FluxSourceSetRecord, ...] = Field(min_length=1)
    prompt_reviews: tuple[FluxPromptReviewRecord, ...] = Field(min_length=1)


@dataclass(frozen=True)
class LoadedFluxSourceSet:
    manifest_path: Path | None
    manifest_sha256: str | None
    name: str
    selection_sha256: str
    prompt_guide_sha256: str
    records: tuple[FluxSourceSetRecord, ...]
    prompt_reviews: tuple[FluxPromptReviewRecord, ...]
    fingerprint: str

    def frozen_payload(self) -> dict[str, object]:
        return {
            "format": FLUX_SOURCE_SET_FROZEN_FORMAT,
            "name": self.name,
            "selection_sha256": self.selection_sha256,
            "prompt_guide_sha256": self.prompt_guide_sha256,
            "records": [
                record.model_dump(mode="json")
                for record in self.records
            ],
            "prompt_reviews": [
                review.model_dump(mode="json")
                for review in self.prompt_reviews
            ],
        }


def load_flux_source_set(
    path: Path,
    *,
    selected: tuple[dict[str, Any], ...],
    selection: dict[str, Any],
) -> LoadedFluxSourceSet:
    manifest_path = path.expanduser().resolve()
    payload = read_json(manifest_path, label="FLUX source-set manifest")
    try:
        manifest = FluxSourceSetManifest.model_validate(payload)
    except ValidationError as error:
        raise Pix2PixError(f"invalid FLUX source-set manifest: {error}") from error
    if manifest.prompt_guide_sha256 != sha256_file(PROMPT_GUIDE_PATH):
        raise Pix2PixError(
            "FLUX source set targets a different image-edit prompt guide"
        )

    records_path = corpus_member(
        manifest_path.parent,
        manifest.records,
        label="FLUX source-set records",
    )
    if sha256_file(records_path) != manifest.records_sha256:
        raise Pix2PixError("FLUX source-set record checksum mismatch")
    records = _load_records(records_path)

    reviews_path = corpus_member(
        manifest_path.parent,
        manifest.prompt_reviews,
        label="FLUX prompt reviews",
    )
    if sha256_file(reviews_path) != manifest.prompt_reviews_sha256:
        raise Pix2PixError("FLUX prompt-review checksum mismatch")
    reviews = _load_reviews(reviews_path)

    if manifest.record_count != len(records):
        raise Pix2PixError("FLUX source-set record count mismatch")
    return _validated_source_set(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        name=manifest.name,
        selection_sha256=manifest.selection_sha256,
        prompt_guide_sha256=manifest.prompt_guide_sha256,
        records=records,
        reviews=reviews,
        selected=selected,
        selection=selection,
    )


def load_frozen_flux_source_set(
    payload: object,
    *,
    selected: tuple[dict[str, Any], ...],
    selection: dict[str, Any],
) -> LoadedFluxSourceSet:
    try:
        frozen = FrozenFluxSourceSet.model_validate(payload)
    except ValidationError as error:
        raise Pix2PixError(f"invalid frozen FLUX source set: {error}") from error
    return _validated_source_set(
        manifest_path=None,
        manifest_sha256=None,
        name=frozen.name,
        selection_sha256=frozen.selection_sha256,
        prompt_guide_sha256=frozen.prompt_guide_sha256,
        records=frozen.records,
        reviews=frozen.prompt_reviews,
        selected=selected,
        selection=selection,
    )


def _load_records(path: Path) -> tuple[FluxSourceSetRecord, ...]:
    records = read_json_records(path, label="FLUX source-set records")
    try:
        return tuple(
            FluxSourceSetRecord.model_validate(record)
            for record in records
        )
    except ValidationError as error:
        raise Pix2PixError(f"invalid FLUX source-set record: {error}") from error


def _load_reviews(path: Path) -> tuple[FluxPromptReviewRecord, ...]:
    records = read_json_records(path, label="FLUX prompt reviews")
    try:
        return tuple(
            FluxPromptReviewRecord.model_validate(record)
            for record in records
        )
    except ValidationError as error:
        raise Pix2PixError(f"invalid FLUX prompt-review record: {error}") from error


def _validated_source_set(
    *,
    manifest_path: Path | None,
    manifest_sha256: str | None,
    name: str,
    selection_sha256: str,
    prompt_guide_sha256: str,
    records: tuple[FluxSourceSetRecord, ...],
    reviews: tuple[FluxPromptReviewRecord, ...],
    selected: tuple[dict[str, Any], ...],
    selection: dict[str, Any],
) -> LoadedFluxSourceSet:
    if selection_sha256 != selection["selected_sha256"]:
        raise Pix2PixError("FLUX source set targets a different selection")
    expected_ids = [str(record["id"]) for record in selected]
    record_ids = [record.id for record in records]
    if record_ids != expected_ids:
        raise Pix2PixError("FLUX source-set record order differs from selection")
    if len(set(record_ids)) != len(record_ids):
        raise Pix2PixError("FLUX source set contains duplicate ids")
    for source_record, selected_record in zip(records, selected, strict=True):
        if source_record.target_sha256 != selected_record["target_sha256"]:
            raise Pix2PixError(
                f"FLUX source-set target checksum mismatch: {source_record.id}"
            )

    review_ids = [review.id for review in reviews]
    if review_ids != expected_ids:
        raise Pix2PixError("FLUX prompt-review order differs from selection")
    for source_record, review in zip(records, reviews, strict=True):
        prompt_sha256 = hashlib.sha256(
            source_record.prompt.encode("utf-8")
        ).hexdigest()
        if review.prompt_sha256 != prompt_sha256:
            raise Pix2PixError(
                f"FLUX prompt-review prompt checksum mismatch: {source_record.id}"
            )
        if review.target_sha256 != source_record.target_sha256:
            raise Pix2PixError(
                f"FLUX prompt-review target checksum mismatch: {source_record.id}"
            )
        if review.reviewer == source_record.author:
            raise Pix2PixError(
                f"FLUX prompt author reviewed their own prompt: {source_record.id}"
            )

    provisional = LoadedFluxSourceSet(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        name=name,
        selection_sha256=selection_sha256,
        prompt_guide_sha256=prompt_guide_sha256,
        records=records,
        prompt_reviews=reviews,
        fingerprint="",
    )
    frozen = provisional.frozen_payload()
    fingerprint = hashlib.sha256(
        json.dumps(
            frozen,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return LoadedFluxSourceSet(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        name=name,
        selection_sha256=selection_sha256,
        prompt_guide_sha256=prompt_guide_sha256,
        records=records,
        prompt_reviews=reviews,
        fingerprint=fingerprint,
    )
