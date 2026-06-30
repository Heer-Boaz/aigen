from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.fft import dctn

from aigen.character_view_models import load_character_view_bank
from aigen.image_assets import image_asset_json
from aigen.keyframe_image_ops import save_contact_sheet
from aigen.keyframe_brief_models import load_keyframe_brief_plan
from aigen.lora_dataset_models import (
    LoraCaptionSourceSpec,
    LoraDatasetError,
    LoraDatasetSpec,
    ViewBankLoraSourceSpec,
    load_lora_dataset_spec,
)
from aigen.manifest_io import (
    resolve_existing_path,
    resolve_output_path,
    sha256_file,
    write_json,
    write_json_line,
)
from aigen.progress import StatusReporter


PHASH_SIZE = 32
PHASH_LOW_FREQUENCY_SIZE = 8
LORA_TRAINING_MINIMUM_SCORES = {
    "identity_preservation": 8.0,
    "outfit_preservation": 8.0,
    "hairstyle_preservation": 8.0,
    "anatomy_quality": 8.0,
    "background_quality": 8.0,
    "style_consistency": 8.0,
    "overall": 8.0,
}


@dataclass(frozen=True)
class LoraDatasetCandidate:
    source_kind: str
    name: str
    image_path: Path
    caption: str
    tags: list[str]
    split: str | None
    source_metadata: dict[str, Any]


@dataclass(frozen=True)
class ApprovedLoraAnchor:
    name: str
    image_path: Path
    caption_parts: list[str]
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


def build_lora_anchor_dataset(
    *,
    dataset_id: str,
    character_id: str,
    trigger_token: str,
    anchors: list[ApprovedLoraAnchor],
    output_dir: Path,
    overwrite: bool,
    validation_ratio: float,
    write_contact_sheet: bool,
    progress: StatusReporter,
) -> dict[str, Any]:
    progress.phase("prepare approved anchors")
    if output_dir.exists():
        if not overwrite:
            raise LoraDatasetError(f"Output exists and overwrite=false: {output_dir.as_posix()}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    candidates = [
        LoraDatasetCandidate(
            source_kind="approved_anchor",
            name=anchor.name,
            image_path=anchor.image_path,
            caption=_caption(trigger_token, "", anchor.caption_parts, anchor.tags),
            tags=anchor.tags,
            split=anchor.split,
            source_metadata=anchor.source_metadata,
        )
        for anchor in anchors
    ]
    if not candidates:
        raise LoraDatasetError("LoRA smoke dataset has no approved anchors")
    progress.step("write approved anchor images")
    records = _write_dataset_images(
        character_id=character_id,
        candidates=candidates,
        output_dir=output_dir,
        validation_ratio=validation_ratio,
    )
    if not records:
        raise LoraDatasetError("LoRA smoke dataset has no images after deduplication")
    progress.step("write metadata")
    _write_metadata(records, output_dir)
    if write_contact_sheet:
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
        "dataset_id": dataset_id,
        "dataset_source": "approved_anchors",
        "character": {"id": character_id, "trigger_token": trigger_token},
        "source_count": len(anchors),
        "accepted_image_count": len(records),
        "split_counts": _split_counts(records),
        "output": {
            "directory": output_dir.as_posix(),
            "images": (output_dir / "images").as_posix(),
            "metadata": (output_dir / "metadata.jsonl").as_posix(),
            "captions": (output_dir / "captions.txt").as_posix(),
            "contact_sheet": (output_dir / "contact_sheet.png").as_posix() if write_contact_sheet else None,
            "report": (output_dir / "dataset_report.json").as_posix(),
        },
        "records": records,
    }
    progress.step("write dataset report")
    write_json(output_dir / "dataset_report.json", report)
    return report


def _dataset_candidates(spec: LoraDatasetSpec, base_dir: Path) -> list[LoraDatasetCandidate]:
    candidates: list[LoraDatasetCandidate] = []
    caption_cache: dict[Path, Any] = {}
    for source in spec.sources:
        candidates.extend(_view_bank_candidates(spec, source, base_dir, caption_cache))
    return candidates


def _view_bank_candidates(
    spec: LoraDatasetSpec,
    source: ViewBankLoraSourceSpec,
    base_dir: Path,
    caption_cache: dict[Path, Any],
) -> list[LoraDatasetCandidate]:
    bank_path = resolve_existing_path(source.path, base_dir)
    bank = load_character_view_bank(bank_path)
    if bank.character.id != spec.character.id:
        raise LoraDatasetError(f"View bank character {bank.character.id} does not match {spec.character.id}")
    source_caption, caption_metadata = _source_caption(source.caption_source, base_dir, caption_cache)
    candidates = []
    for view_name in source.views:
        if view_name not in bank.views:
            raise LoraDatasetError(f"View bank has no accepted view: {view_name}")
        entry = bank.views[view_name]
        training_validation = _lora_training_validation(entry, view_name)
        manual_acceptance = entry.acceptance.manual if entry.acceptance else []
        caption = _caption(
            spec.character.trigger_token,
            source_caption,
            [
                _words(entry.view.name),
                _words(entry.view.pose),
                _words(entry.view.camera),
                *manual_acceptance,
            ],
            source.tags,
        )
        candidates.append(
            LoraDatasetCandidate(
                source_kind="view_bank",
                name=view_name,
                image_path=Path(entry.image.path),
                caption=caption,
                tags=source.tags,
                split=source.split,
                source_metadata={
                    "view": entry.view.model_dump(mode="json"),
                    "accepted_candidate": entry.accepted_candidate,
                    "accepted_seed": entry.accepted_seed,
                    "bank": bank_path.as_posix(),
                    "caption_source": caption_metadata,
                    "training_validation": training_validation,
                },
            )
        )
    return candidates


def _lora_training_validation(entry: Any, view_name: str) -> dict[str, Any]:
    validation = entry.training_validation
    if validation is None:
        raise LoraDatasetError(f"View {view_name} has no model-backed LoRA training validation")
    payload = validation.model_dump(mode="json")
    if not payload["usable_for_lora_training"]:
        raise LoraDatasetError(f"View {view_name} is not usable for LoRA training")
    hard_rejects = payload["hard_rejects"]
    if any(bool(rejected) for rejected in hard_rejects.values()):
        raise LoraDatasetError(f"View {view_name} has LoRA training hard rejects")
    scores = payload["scores"]
    low_scores = [
        name for name, minimum in LORA_TRAINING_MINIMUM_SCORES.items() if float(scores[name]) < minimum
    ]
    if low_scores:
        raise LoraDatasetError(f"View {view_name} has LoRA training scores below threshold: {', '.join(low_scores)}")
    return payload


def _source_caption(
    source: LoraCaptionSourceSpec,
    base_dir: Path,
    caption_cache: dict[Path, Any],
) -> tuple[str, dict[str, str]]:
    plan_path = resolve_existing_path(source.plan, base_dir)
    if plan_path not in caption_cache:
        caption_cache[plan_path] = load_keyframe_brief_plan(plan_path).lora_captions
    caption = getattr(caption_cache[plan_path], source.field)
    return caption, {"plan": plan_path.as_posix(), "field": source.field}


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
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        background.convert("RGB").save(output_path)


def _write_metadata(records: list[dict[str, Any]], output_dir: Path) -> None:
    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata:
        for record in records:
            write_json_line(metadata, record)
    with (output_dir / "captions.txt").open("w", encoding="utf-8") as captions:
        for record in records:
            captions.write(f"{record['file_name']}\t{record['prompt']}\n")


def _caption(trigger_token: str, primary: str, parts: list[str], tags: list[str]) -> str:
    values = [trigger_token]
    values.append(primary)
    values.extend(parts)
    values.extend(tags)
    return ", ".join(_dedupe_caption_parts(values))


def _dedupe_caption_parts(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        cleaned = " ".join(value.replace("_", " ").replace("-", " ").split())
        if cleaned and cleaned.lower() not in seen:
            result.append(cleaned)
            seen.add(cleaned.lower())
    return result


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


def _words(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


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
