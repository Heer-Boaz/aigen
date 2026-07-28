from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from aigen.manifest_io import read_json, sha256_file
from aigen.pix2pix.config import MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZES
from aigen.pix2pix.errors import Pix2PixError


DATASET_FORMAT = "aigen.pix2pix.paired.v2"
DATASET_MANIFEST_NAME = "dataset.json"
PAIR_SPLITS = frozenset({"train", "validation", "test"})
PAIR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class PairedImage:
    id: str
    group: str
    split: str
    source_path: Path
    target_path: Path
    source_relative: str
    target_relative: str
    source_sha256: str
    target_sha256: str


@dataclass(frozen=True)
class AuditedDataset:
    root: Path
    name: str
    image_size: int
    pair_manifest: Path
    pairs: tuple[PairedImage, ...]
    split_counts: dict[str, int]
    split_group_counts: dict[str, int]
    fingerprint: str

    def split(self, name: str) -> tuple[PairedImage, ...]:
        if name not in PAIR_SPLITS:
            raise Pix2PixError(f"unsupported dataset split: {name}")
        return tuple(pair for pair in self.pairs if pair.split == name)

    def to_json(self) -> dict[str, object]:
        return {
            "status": "ok",
            "format": DATASET_FORMAT,
            "name": self.name,
            "root": self.root.as_posix(),
            "image_size": self.image_size,
            "pair_manifest": self.pair_manifest.as_posix(),
            "pair_count": len(self.pairs),
            "split_counts": self.split_counts,
            "group_count": len({pair.group for pair in self.pairs}),
            "split_group_counts": self.split_group_counts,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class _ImageInspection:
    sha256: str


def audit_dataset(root: Path) -> AuditedDataset:
    root = root.resolve()
    manifest_path = root / DATASET_MANIFEST_NAME
    manifest = read_json(manifest_path, label="pix2pix dataset manifest")
    if not isinstance(manifest, dict):
        raise Pix2PixError("dataset manifest must be a JSON object")
    expected = {"format", "name", "image_size", "pairs"}
    _require_exact_keys(manifest, expected, "dataset manifest")
    if manifest["format"] != DATASET_FORMAT:
        raise Pix2PixError(f"unsupported pix2pix dataset format: {manifest['format']!r}")
    name = _nonempty_string(manifest, "name")
    image_size = _integer(manifest, "image_size")
    if image_size not in MODEL_IMAGE_SIZES:
        supported = ", ".join(str(size) for size in sorted(MODEL_IMAGE_SIZES))
        raise Pix2PixError(
            f"pix2pix v1 dataset image_size must be one of: {supported}"
        )
    pair_manifest = _resolve_member(root, _nonempty_string(manifest, "pairs"), "pairs manifest")
    pairs_payload = _read_pair_records(pair_manifest)
    if not pairs_payload:
        raise Pix2PixError(f"pairs manifest is empty: {pair_manifest.as_posix()}")

    image_cache: dict[Path, _ImageInspection] = {}
    ids: set[str] = set()
    pairs = []
    split_counts: Counter[str] = Counter()
    groups_by_split: dict[str, set[str]] = {
        split: set() for split in PAIR_SPLITS
    }
    split_by_group: dict[str, str] = {}
    for line_number, record in pairs_payload:
        _require_exact_keys(
            record,
            {"id", "group", "split", "source", "target"},
            f"pair line {line_number}",
        )
        pair_id = _nonempty_string(record, "id")
        if not PAIR_ID_PATTERN.fullmatch(pair_id):
            raise Pix2PixError(
                f"pair line {line_number} id must use letters, digits, dot, underscore, or hyphen"
            )
        if pair_id in ids:
            raise Pix2PixError(f"duplicate pair id: {pair_id}")
        ids.add(pair_id)
        group = _nonempty_string(record, "group")
        if not PAIR_ID_PATTERN.fullmatch(group):
            raise Pix2PixError(
                f"pair {pair_id} group must use letters, digits, dot, underscore, or hyphen"
            )
        split = _nonempty_string(record, "split")
        if split not in PAIR_SPLITS:
            raise Pix2PixError(f"pair {pair_id} has unsupported split: {split}")
        existing_split = split_by_group.setdefault(group, split)
        if existing_split != split:
            raise Pix2PixError(
                f"group {group} crosses dataset splits: {existing_split} and {split}"
            )
        source_relative = _nonempty_string(record, "source")
        target_relative = _nonempty_string(record, "target")
        source_path = _resolve_member(root, source_relative, f"source for pair {pair_id}")
        target_path = _resolve_member(root, target_relative, f"target for pair {pair_id}")
        source = _inspect_once(image_cache, source_path, image_size)
        target = _inspect_once(image_cache, target_path, image_size)
        pairs.append(
            PairedImage(
                id=pair_id,
                group=group,
                split=split,
                source_path=source_path,
                target_path=target_path,
                source_relative=source_relative,
                target_relative=target_relative,
                source_sha256=source.sha256,
                target_sha256=target.sha256,
            )
        )
        split_counts[split] += 1
        groups_by_split[split].add(group)

    for required_split in ("train", "validation"):
        if split_counts[required_split] == 0:
            raise Pix2PixError(f"dataset has no {required_split} pairs")
    normalized_counts = {split: split_counts[split] for split in sorted(PAIR_SPLITS)}
    normalized_group_counts = {
        split: len(groups_by_split[split]) for split in sorted(PAIR_SPLITS)
    }
    return AuditedDataset(
        root=root,
        name=name,
        image_size=image_size,
        pair_manifest=pair_manifest,
        pairs=tuple(pairs),
        split_counts=normalized_counts,
        split_group_counts=normalized_group_counts,
        fingerprint=_dataset_fingerprint(name, image_size, pairs),
    )


def dataset_contract() -> dict[str, object]:
    return {
        "dataset_manifest": {
            "format": DATASET_FORMAT,
            "name": "example-dataset",
            "image_size": MODEL_IMAGE_SIZE,
            "pairs": "pairs.jsonl",
        },
        "pair_record": {
            "id": "unique-pair-id",
            "group": "original-subject-or-source-sequence",
            "split": "train",
            "source": "source/example.png",
            "target": "target/example.png",
        },
        "requirements": {
            "image_mode": "RGB",
            "image_dimensions": [
                [size, size] for size in sorted(MODEL_IMAGE_SIZES)
            ],
            "required_splits": ["train", "validation"],
            "allowed_splits": sorted(PAIR_SPLITS),
            "grouping": "one group must occur in exactly one split",
            "alignment": "source and target must already share the intended composition",
        },
    }


def _read_pair_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Pix2PixError(f"cannot read pairs manifest: {path.as_posix()}") from error
    records = []
    for line_number, text in enumerate(lines, start=1):
        if not text.strip():
            raise Pix2PixError(f"blank line in pairs manifest at line {line_number}")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise Pix2PixError(
                f"invalid pairs manifest JSON at line {line_number}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise Pix2PixError(f"pair line {line_number} must be a JSON object")
        records.append((line_number, payload))
    return records


def _inspect_once(
    cache: dict[Path, _ImageInspection],
    path: Path,
    image_size: int,
) -> _ImageInspection:
    inspection = cache.get(path)
    if inspection is not None:
        return inspection
    try:
        with Image.open(path) as image:
            image.load()
            mode = image.mode
            size = image.size
    except OSError as error:
        raise Pix2PixError(f"cannot decode image {path.as_posix()}: {error}") from error
    if mode != "RGB":
        raise Pix2PixError(f"image must be RGB: {path.as_posix()} has mode {mode}")
    if size != (image_size, image_size):
        raise Pix2PixError(
            f"image must be {image_size}x{image_size}: {path.as_posix()} is {size[0]}x{size[1]}"
        )
    inspection = _ImageInspection(sha256=sha256_file(path))
    cache[path] = inspection
    return inspection


def _dataset_fingerprint(
    name: str,
    image_size: int,
    pairs: list[PairedImage],
) -> str:
    digest = hashlib.sha256()
    digest.update(DATASET_FORMAT.encode())
    digest.update(b"\0")
    digest.update(name.encode())
    digest.update(b"\0")
    digest.update(str(image_size).encode())
    for pair in pairs:
        for value in (
            pair.id,
            pair.group,
            pair.split,
            pair.source_relative,
            pair.source_sha256,
            pair.target_relative,
            pair.target_sha256,
        ):
            digest.update(b"\0")
            digest.update(value.encode())
    return digest.hexdigest()


def _resolve_member(root: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise Pix2PixError(f"{label} must be relative to the dataset root")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Pix2PixError(f"{label} escapes the dataset root: {value}") from error
    if not path.is_file():
        raise Pix2PixError(f"missing {label}: {path.as_posix()}")
    return path


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    keys = set(payload)
    if keys == expected:
        return
    missing = sorted(expected - keys)
    unexpected = sorted(keys - expected)
    details = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected {', '.join(unexpected)}")
    raise Pix2PixError(f"invalid {label}: {'; '.join(details)}")


def _nonempty_string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise Pix2PixError(f"{key} must be a non-empty string")
    return value


def _integer(payload: dict[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise Pix2PixError(f"{key} must be an integer")
    return value
