from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from aigen.image_assets import image_asset_json
from aigen.keyframe_image_ops import save_contact_sheet
from aigen.lora_quality import lora_quality_contract
from aigen.lora_text import caption_contains_token, join_prompt_parts
from aigen.manifest_io import read_json, resolve_existing_path, sha256_file, write_json
from aigen.progress import StatusReporter


CANON_MANIFEST = "canon_manifest.json"


class LoraCanonError(RuntimeError):
    pass


def load_lora_canon_manifest(canon_dir: Path) -> dict[str, Any]:
    canon_dir = canon_dir.resolve()
    manifest = read_json(canon_dir / CANON_MANIFEST, label="LoRA canon manifest")
    if manifest.get("kind") != "lora-canon":
        raise LoraCanonError(f"Not a LoRA canon manifest: {(canon_dir / CANON_MANIFEST).as_posix()}")
    if manifest.get("status") != "active":
        raise LoraCanonError(f"LoRA canon is not active: {(canon_dir / CANON_MANIFEST).as_posix()}")
    return manifest


def lora_canon_images_by_name(manifest: dict[str, Any], canon_dir: Path) -> dict[str, dict[str, Any]]:
    canon_dir = canon_dir.resolve()
    by_name = {}
    for image in manifest["images"]:
        path = resolve_existing_path(image["file_name"], canon_dir)
        by_name[image["name"]] = {
            "name": image["name"],
            "path": path.as_posix(),
            "sha256": image["image"]["sha256"],
            "source_sha256": image["source_sha256"],
            "training_caption": image["training_caption"],
            "approval": image["approval"],
        }
    return by_name


def init_lora_canon(
    *,
    character_id: str,
    trigger_token: str,
    identity_prompt: str,
    anchor_specs: list[str],
    output_dir: Path | None,
    approved_by: str,
    overwrite: bool,
    progress: StatusReporter,
) -> dict[str, Any]:
    if caption_contains_token(identity_prompt, trigger_token):
        raise LoraCanonError("--identity-prompt must not include the trigger token; --trigger-token owns it")
    if not anchor_specs:
        raise LoraCanonError("Canon init requires at least one human-approved anchor")

    canon_dir = (output_dir or Path("assets") / "characters" / character_id / "canon").resolve()
    if canon_dir.exists():
        if not overwrite:
            raise LoraCanonError(f"Output exists and overwrite=false: {canon_dir.as_posix()}")
        shutil.rmtree(canon_dir)
    (canon_dir / "images").mkdir(parents=True)

    progress.phase("copy canon anchors")
    images = []
    for spec in anchor_specs:
        name, source_path = _parse_named_path(spec)
        output_path = canon_dir / "images" / f"{_slug(name)}.png"
        _copy_rgb_png(source_path, output_path)
        training_caption = _training_caption(trigger_token, identity_prompt, name)
        caption_path = output_path.with_suffix(".txt")
        caption_path.write_text(training_caption + "\n", encoding="utf-8")
        images.append(
            {
                "name": name,
                "file_name": output_path.relative_to(canon_dir).as_posix(),
                "training_caption_file": caption_path.relative_to(canon_dir).as_posix(),
                "training_caption": training_caption,
                "image": image_asset_json(output_path),
                "source_path": source_path.as_posix(),
                "source_sha256": sha256_file(source_path),
                "approval": {
                    "mode": "human_approved_canon",
                    "approved_by": approved_by,
                },
            }
        )

    progress.phase("write canon evidence")
    save_contact_sheet(
        [{"name": image["name"], "path": (canon_dir / image["file_name"]).as_posix()} for image in images],
        canon_dir / "contact_sheet.png",
        thumb_width=192,
        max_label_chars=24,
    )
    manifest = {
        "status": "active",
        "kind": "lora-canon",
        "character": {
            "id": character_id,
            "trigger_token": trigger_token,
            "identity_prompt": identity_prompt,
        },
        "quality_contract": lora_quality_contract(),
        "images": images,
        "output": {
            "directory": canon_dir.as_posix(),
            "manifest": (canon_dir / CANON_MANIFEST).as_posix(),
            "contact_sheet": (canon_dir / "contact_sheet.png").as_posix(),
        },
    }
    write_json(canon_dir / CANON_MANIFEST, manifest)
    return manifest


def audit_lora_dataset_source(
    source_dir: Path,
    *,
    output_dir: Path | None,
    overwrite: bool,
    progress: StatusReporter,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    if not source_dir.exists():
        raise LoraCanonError(f"Missing dataset source: {source_dir.as_posix()}")

    audit_dir = (output_dir or source_dir / "audit").resolve()
    if audit_dir.exists():
        if not overwrite:
            raise LoraCanonError(f"Output exists and overwrite=false: {audit_dir.as_posix()}")
        shutil.rmtree(audit_dir)
    audit_dir.mkdir(parents=True)

    progress.phase("load audit source")
    manifest_path = source_dir / CANON_MANIFEST
    if manifest_path.exists():
        source = read_json(manifest_path, label="canon manifest")
        images = _canon_images(source_dir, source)
        status = "accepted_canon"
        accepted = images
        pending: list[dict[str, Any]] = []
    else:
        images = _loose_images(source_dir)
        status = "needs_human_review"
        accepted = []
        pending = images

    if not images:
        raise LoraCanonError(f"No images found for dataset audit: {source_dir.as_posix()}")

    progress.phase("write audit evidence")
    save_contact_sheet(
        [{"name": image["name"], "path": image["path"]} for image in images],
        audit_dir / "contact_sheet.png",
        thumb_width=192,
        max_label_chars=24,
    )
    rejected: list[dict[str, Any]] = []
    report = {
        "status": status,
        "kind": "lora-dataset-audit",
        "source": source_dir.as_posix(),
        "quality_contract": lora_quality_contract(),
        "counts": {
            "images": len(images),
            "accepted": len(accepted),
            "pending": len(pending),
            "rejected": len(rejected),
        },
        "accepted": accepted,
        "pending": pending,
        "rejected": rejected,
        "output": {
            "directory": audit_dir.as_posix(),
            "accepted": (audit_dir / "accepted.json").as_posix(),
            "pending": (audit_dir / "pending.json").as_posix(),
            "rejected": (audit_dir / "rejected.json").as_posix(),
            "contact_sheet": (audit_dir / "contact_sheet.png").as_posix(),
            "report": (audit_dir / "dataset_report.json").as_posix(),
        },
    }
    write_json(audit_dir / "accepted.json", {"items": accepted})
    write_json(audit_dir / "pending.json", {"items": pending})
    write_json(audit_dir / "rejected.json", {"items": rejected})
    write_json(audit_dir / "dataset_report.json", report)
    return report


def _canon_images(source_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("kind") != "lora-canon":
        raise LoraCanonError(f"Not a LoRA canon manifest: {(source_dir / CANON_MANIFEST).as_posix()}")
    images = []
    for item in manifest["images"]:
        path = resolve_existing_path(item["file_name"], source_dir)
        images.append(
            {
                "name": item["name"],
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "training_caption": item["training_caption"],
                "approval": item["approval"],
            }
        )
    return images


def _loose_images(source_dir: Path) -> list[dict[str, Any]]:
    images = []
    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        images.append(
            {
                "name": path.stem,
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "approval": {
                    "mode": "pending_human_canon_review",
                },
            }
        )
    return images


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise LoraCanonError(f"Anchor must use NAME=PATH: {value}")
    name, raw_path = value.split("=", 1)
    cleaned_name = name.strip()
    if not cleaned_name:
        raise LoraCanonError(f"Anchor name is empty: {value}")
    return cleaned_name, resolve_existing_path(raw_path.strip(), Path.cwd())


def _copy_rgb_png(source_path: Path, output_path: Path) -> None:
    with Image.open(source_path) as source:
        source.convert("RGB").save(output_path)


def _training_caption(trigger_token: str, identity_prompt: str, name: str) -> str:
    return join_prompt_parts(trigger_token, identity_prompt, _words(name))


def _slug(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif cleaned and cleaned[-1] != "_":
            cleaned.append("_")
    return "".join(cleaned).strip("_")


def _words(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").split())
