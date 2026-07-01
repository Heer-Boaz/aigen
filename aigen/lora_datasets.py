from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.fft import dctn

from aigen.image_assets import image_asset_json
from aigen.keyframe_image_ops import save_contact_sheet
from aigen.lora_dataset_models import (
    CandidateReviewLoraSourceSpec,
    CanonLoraSourceSpec,
    LoraDatasetError,
    LoraDatasetSpec,
    load_lora_dataset_spec,
)
from aigen.manifest_io import (
    read_json,
    resolve_existing_path,
    resolve_output_path,
    sha256_file,
    write_json,
    write_json_line,
)
from aigen.progress import StatusReporter


PHASH_SIZE = 32
PHASH_LOW_FREQUENCY_SIZE = 8


@dataclass(frozen=True)
class LoraDatasetCandidate:
    source_kind: str
    name: str
    image_path: Path
    caption: str
    tags: list[str]
    split: str | None
    source_metadata: dict[str, Any]


def build_lora_dataset(spec_path: Path, *, progress: StatusReporter) -> dict[str, Any]:
    progress.phase("load dataset spec")
    spec = load_lora_dataset_spec(spec_path)
    base_dir = spec_path.parent
    output_dir = resolve_output_path(spec.output.directory, base_dir)
    if output_dir.exists():
        if not spec.output.overwrite:
            raise LoraDatasetError(f"Output exists and overwrite=false: {output_dir.as_posix()}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    progress.step("collect accepted images")
    candidates = _dataset_candidates(spec, base_dir)
    if not candidates:
        raise LoraDatasetError("LoRA dataset has no accepted images")
    progress.step("write training images")
    records = _write_dataset_images(
        character_id=spec.character.id,
        candidates=candidates,
        output_dir=output_dir,
        validation_ratio=spec.output.validation_ratio,
    )
    if not records:
        raise LoraDatasetError("LoRA dataset has no images after deduplication")
    progress.step("write metadata")
    _write_metadata(records, output_dir)
    if spec.output.save_contact_sheet:
        progress.phase("write contact sheet")
        save_contact_sheet(
            [
                {
                    "name": record["name"],
                    "path": (output_dir / record["file_name"]).as_posix(),
                }
                for record in records
            ],
            output_dir / "contact_sheet.png",
            thumb_width=192,
            max_label_chars=24,
        )
    report = {
        "status": "completed",
        "kind": "lora-dataset-result",
        "dataset_id": spec.id,
        "character": spec.character.model_dump(mode="json"),
        "source_count": len(spec.sources),
        "accepted_image_count": len(records),
        "split_counts": _split_counts(records),
        "output": {
            "directory": output_dir.as_posix(),
            "images": (output_dir / "images").as_posix(),
            "metadata": (output_dir / "metadata.jsonl").as_posix(),
            "captions": (output_dir / "captions.txt").as_posix(),
            "contact_sheet": (output_dir / "contact_sheet.png").as_posix()
            if spec.output.save_contact_sheet
            else None,
            "report": (output_dir / "dataset_report.json").as_posix(),
        },
        "records": records,
    }
    progress.step("write dataset report")
    write_json(output_dir / "dataset_report.json", report)
    return report

def _dataset_candidates(spec: LoraDatasetSpec, base_dir: Path) -> list[LoraDatasetCandidate]:
    candidates: list[LoraDatasetCandidate] = []
    for source in spec.sources:
        match source.type:
            case "canon":
                candidates.extend(_canon_candidates(spec, source, base_dir))
            case "candidate_review":
                candidates.extend(_candidate_review_candidates(spec, source, base_dir))
    return candidates


def _canon_candidates(
    spec: LoraDatasetSpec,
    source: CanonLoraSourceSpec,
    base_dir: Path,
) -> list[LoraDatasetCandidate]:
    canon_path = resolve_existing_path(source.path, base_dir)
    manifest_path = canon_path / "canon_manifest.json"
    manifest = read_json(manifest_path, label="LoRA canon manifest")
    if manifest.get("kind") != "lora-canon":
        raise LoraDatasetError(f"Not a LoRA canon manifest: {manifest_path.as_posix()}")
    if manifest.get("status") != "active":
        raise LoraDatasetError(f"LoRA canon is not active: {manifest_path.as_posix()}")
    character = manifest["character"]
    if character["id"] != spec.character.id:
        raise LoraDatasetError(f"Canon character {character['id']} does not match {spec.character.id}")
    if character["trigger_token"] != spec.character.trigger_token:
        raise LoraDatasetError("Canon trigger token does not match dataset character trigger token")

    by_name = {item["name"]: item for item in manifest["images"]}
    candidates = []
    for image_name in source.images:
        if image_name not in by_name:
            raise LoraDatasetError(f"Canon has no image: {image_name}")
        item = by_name[image_name]
        image_path = resolve_existing_path(item["file_name"], canon_path)
        candidates.append(
            LoraDatasetCandidate(
                source_kind="canon",
                name=image_name,
                image_path=image_path,
                caption=item["training_caption"],
                tags=source.tags,
                split=source.split,
                source_metadata={
                    "canon": canon_path.as_posix(),
                    "approval": item["approval"],
                    "source_sha256": item["source_sha256"],
                    "quality_contract": manifest["quality_contract"],
                },
            )
        )
    return candidates


def _candidate_review_candidates(
    spec: LoraDatasetSpec,
    source: CandidateReviewLoraSourceSpec,
    base_dir: Path,
) -> list[LoraDatasetCandidate]:
    accepted_path = resolve_existing_path(source.path, base_dir)
    if accepted_path.is_dir():
        accepted_path = resolve_existing_path("accepted.json", accepted_path)
    payload = read_json(accepted_path, label="LoRA candidate accepted manifest")
    if payload.get("kind") != "lora-candidate-accepted":
        raise LoraDatasetError(f"Not a LoRA candidate accepted manifest: {accepted_path.as_posix()}")
    if payload.get("status") != "completed":
        raise LoraDatasetError(f"LoRA candidate review is not completed: {accepted_path.as_posix()}")

    candidates = []
    for item in payload["items"]:
        approval = item.get("approval", {})
        if approval.get("mode") != "human_approved_lora_candidate":
            raise LoraDatasetError(f"Candidate {item['name']} has no human approval")
        image_path = resolve_existing_path(item["image"]["path"], accepted_path.parent)
        candidates.append(
            LoraDatasetCandidate(
                source_kind="candidate_review",
                name=item["name"],
                image_path=image_path,
                caption=item["training_caption"],
                tags=source.tags,
                split=source.split,
                source_metadata={
                    "accepted_manifest": accepted_path.as_posix(),
                    "candidate": item["candidate"],
                    "seed": item["seed"],
                    "approval": approval,
                    "evidence": item.get("evidence", {}),
                },
            )
        )
    return candidates

def _write_dataset_images(
    *,
    character_id: str,
    candidates: list[LoraDatasetCandidate],
    output_dir: Path,
    validation_ratio: float,
) -> list[dict[str, Any]]:
    splits = _assigned_splits(candidates, validation_ratio)
    records = []
    seen_sha256: set[str] = set()
    for index, (candidate, split) in enumerate(zip(candidates, splits, strict=True), start=1):
        source_sha256 = sha256_file(candidate.image_path)
        phash = _perceptual_hash(candidate.image_path)
        if source_sha256 in seen_sha256:
            continue
        seen_sha256.add(source_sha256)
        file_name = f"images/{split}/{index:04d}_{_slug(character_id)}_{_slug(candidate.name)}.png"
        output_path = output_dir / file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _save_training_image(candidate.image_path, output_path)
        caption_path = output_path.with_suffix(".txt")
        caption_path.write_text(candidate.caption + "\n", encoding="utf-8")
        image_asset = image_asset_json(output_path)
        records.append(
            {
                "file_name": file_name,
                "caption_file": caption_path.relative_to(output_dir).as_posix(),
                "prompt": candidate.caption,
                "name": candidate.name,
                "split": split,
                "source_kind": candidate.source_kind,
                "source_path": candidate.image_path.resolve().as_posix(),
                "source_sha256": source_sha256,
                "perceptual_hash": phash,
                "tags": candidate.tags,
                "image": image_asset,
                "source_metadata": candidate.source_metadata,
            }
        )
    return records


def _assigned_splits(candidates: list[LoraDatasetCandidate], validation_ratio: float) -> list[str]:
    automatic = [candidate for candidate in candidates if candidate.split is None]
    val_count = 0
    if len(automatic) > 1 and validation_ratio > 0.0:
        val_count = min(len(automatic) - 1, max(1, round(len(automatic) * validation_ratio)))
    auto_val_start = len(automatic) - val_count
    auto_index = 0
    splits = []
    for candidate in candidates:
        if candidate.split:
            splits.append(candidate.split)
            continue
        split = "val" if auto_index >= auto_val_start else "train"
        splits.append(split)
        auto_index += 1
    return splits


def _save_training_image(source_path: Path, output_path: Path) -> None:
    with Image.open(source_path) as image:
        rgba = image.convert("RGBA")
        side = max(rgba.size)
        background = Image.new("RGBA", (side, side), "white")
        background.alpha_composite(rgba, ((side - rgba.width) // 2, (side - rgba.height) // 2))
        background.convert("RGB").save(output_path)


def _write_metadata(records: list[dict[str, Any]], output_dir: Path) -> None:
    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata:
        for record in records:
            write_json_line(metadata, record)
    with (output_dir / "captions.txt").open("w", encoding="utf-8") as captions:
        for record in records:
            captions.write(f"{record['file_name']}\t{record['prompt']}\n")


def _perceptual_hash(path: Path) -> str:
    with Image.open(path) as image:
        gray = image.convert("L").resize((PHASH_SIZE, PHASH_SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    coefficients = dctn(pixels, norm="ortho")[:PHASH_LOW_FREQUENCY_SIZE, :PHASH_LOW_FREQUENCY_SIZE]
    values = coefficients.flatten()[1:]
    median = float(np.median(values))
    bits = coefficients.flatten() > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _split_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"train": 0, "val": 0}
    for record in records:
        counts[record["split"]] += 1
    return counts


def _slug(value: str) -> str:
    chars = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    return "".join(chars).strip("-") or "image"
