from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from aigen.manifest_io import read_json, sha256_file
from aigen.pix2pix.corpus_io import corpus_member, read_json_records
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.flux_source_set import (
    LoadedFluxSourceSet,
    SHA256_PATTERN,
)
from aigen.pix2pix.flux_source_set_corpus import flux_source_set_layout


FLUX_OUTPUT_AUDIT_FORMAT = "aigen.pix2pix.flux-output-audit.v1"


class _AuditModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FluxOutputAuditRecord(_AuditModel):
    id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    target_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    chosen_seed: int = Field(ge=0)
    tested_seeds: tuple[int, ...] = Field(min_length=1)
    verdict: Literal["pass", "reject"]
    reviewer: str = Field(min_length=1)
    notes: str = Field(min_length=1)

    @field_validator("reviewer", "notes")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("FLUX output-audit text has surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_seeds(self) -> FluxOutputAuditRecord:
        if any(seed < 0 for seed in self.tested_seeds):
            raise ValueError("FLUX output audit contains a negative tested seed")
        if len(set(self.tested_seeds)) != len(self.tested_seeds):
            raise ValueError("FLUX output audit contains duplicate tested seeds")
        if self.chosen_seed not in self.tested_seeds:
            raise ValueError("chosen FLUX seed is absent from tested seeds")
        return self


class FluxOutputAuditManifest(_AuditModel):
    format: Literal["aigen.pix2pix.flux-output-audit.v1"]
    source_plan_sha256: str = Field(pattern=SHA256_PATTERN)
    source_result_sha256: str = Field(pattern=SHA256_PATTERN)
    records: str = Field(min_length=1)
    records_sha256: str = Field(pattern=SHA256_PATTERN)
    record_count: int = Field(gt=0)


@dataclass(frozen=True)
class LoadedFluxOutputAudit:
    manifest_path: Path
    manifest_sha256: str
    records_sha256: str
    records: tuple[FluxOutputAuditRecord, ...]
    accepted_ids: frozenset[str]


def load_flux_output_audit(
    root: Path,
    name: str,
    *,
    inventory: dict[str, Path],
    source_set: LoadedFluxSourceSet,
    selected: tuple[dict[str, Any], ...],
    source_plan_sha256: str,
    source_result_sha256: str,
) -> LoadedFluxOutputAudit:
    source_root = root / flux_source_set_layout(name).directory
    manifest_path = source_root / "output-audit.json"
    payload = read_json(manifest_path, label="FLUX output-audit manifest")
    try:
        manifest = FluxOutputAuditManifest.model_validate(payload)
    except ValidationError as error:
        raise Pix2PixError(f"invalid FLUX output-audit manifest: {error}") from error
    if manifest.source_plan_sha256 != source_plan_sha256:
        raise Pix2PixError("FLUX output audit targets a different source plan")
    if manifest.source_result_sha256 != source_result_sha256:
        raise Pix2PixError("FLUX output audit targets a different source result")

    records_path = corpus_member(
        source_root,
        manifest.records,
        label="FLUX output-audit records",
    )
    if sha256_file(records_path) != manifest.records_sha256:
        raise Pix2PixError("FLUX output-audit record checksum mismatch")
    raw_records = read_json_records(
        records_path,
        label="FLUX output-audit records",
    )
    try:
        records = tuple(
            FluxOutputAuditRecord.model_validate(record)
            for record in raw_records
        )
    except ValidationError as error:
        raise Pix2PixError(f"invalid FLUX output-audit record: {error}") from error
    if manifest.record_count != len(records):
        raise Pix2PixError("FLUX output-audit record count mismatch")

    expected_ids = [str(record["id"]) for record in selected]
    if [record.id for record in records] != expected_ids:
        raise Pix2PixError("FLUX output-audit order differs from selection")
    if set(inventory) != set(expected_ids):
        raise Pix2PixError("FLUX output-audit inventory differs from selection")
    for audit_record, source_record, selected_record in zip(
        records,
        source_set.records,
        selected,
        strict=True,
    ):
        if audit_record.source_sha256 != sha256_file(inventory[audit_record.id]):
            raise Pix2PixError(
                f"FLUX output-audit source checksum mismatch: {audit_record.id}"
            )
        if audit_record.target_sha256 != selected_record["target_sha256"]:
            raise Pix2PixError(
                f"FLUX output-audit target checksum mismatch: {audit_record.id}"
            )
        prompt_sha256 = hashlib.sha256(
            source_record.prompt.encode("utf-8")
        ).hexdigest()
        if audit_record.prompt_sha256 != prompt_sha256:
            raise Pix2PixError(
                f"FLUX output-audit prompt checksum mismatch: {audit_record.id}"
            )
        if audit_record.chosen_seed != source_record.seed:
            raise Pix2PixError(
                f"FLUX output-audit chosen seed mismatch: {audit_record.id}"
            )

    accepted_ids = frozenset(
        record.id
        for record in records
        if record.verdict == "pass"
    )
    if not accepted_ids:
        raise Pix2PixError("FLUX output audit rejected every source pair")
    return LoadedFluxOutputAudit(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        records_sha256=manifest.records_sha256,
        records=records,
        accepted_ids=accepted_ids,
    )
