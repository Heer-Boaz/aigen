from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageChops, __version__ as PILLOW_VERSION

from aigen.manifest_io import read_json, sha256_file, write_json
from aigen.pix2pix.corpus_io import (
    read_json_records,
    require_exact_keys,
    write_json_records,
)
from aigen.pix2pix.dataset import (
    ASYMMETRIC_DATASET_FORMAT,
    AuditedDataset,
    audit_dataset,
)
from aigen.pix2pix.errors import Pix2PixError
from aigen.progress import StatusReporter


BILINEAR_CONTROL_PROVENANCE_FORMAT = (
    "aigen.pix2pix.bilinear-control-provenance.v2"
)
BILINEAR_CONTROL_KIND = "pix2pix-bilinear-downscale-control"
BILINEAR_SOURCE_SIZE = 1024
BILINEAR_TARGET_SIZE = 128
BILINEAR_OPERATION = "Pillow.Image.resize(Image.Resampling.BILINEAR)"
DIV2K_CROP_SELECTION = "sha256-modulo-lossless-crop.v1"
DIV2K_SOURCE_MANIFEST = "source-crops.jsonl"
DIV2K_SPLITS = (
    ("train", "DIV2K_train_HR", 1, 800),
    ("validation", "DIV2K_valid_HR", 801, 900),
)
DIV2K_ARCHIVE_SHA256 = {
    "DIV2K_train_HR.zip": (
        "9d0b9c463f6e35b6c62cc6a930ee2224f670b34c1df841a57670f9acf0f6c335"
    ),
    "DIV2K_valid_HR.zip": (
        "20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc"
    ),
}


@dataclass(frozen=True)
class _Div2kCrop:
    id: str
    split: str
    input_path: Path
    input_relative: str
    input_sha256: str
    input_width: int
    input_height: int
    crop_left: int
    crop_top: int

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "split": self.split,
            "input": self.input_relative,
            "input_sha256": self.input_sha256,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "crop_left": self.crop_left,
            "crop_top": self.crop_top,
            "crop_size": BILINEAR_SOURCE_SIZE,
        }


class _BoundaryCoverage:
    def __init__(self, image_size: int) -> None:
        self._image_size = image_size
        self._band_width = image_size // 16
        self._image_count = 0
        self._full_pixels = 0
        self._full_white = 0
        self._boundary_pixels = 0
        self._boundary_white = 0
        self._sum = np.zeros(3, dtype=np.float64)
        self._square_sum = np.zeros(3, dtype=np.float64)
        self._histogram = np.zeros(4096, dtype=np.int64)

    def add(self, image: Image.Image) -> None:
        pixels = np.asarray(image)
        band = self._band_width
        self._image_count += 1
        self._full_pixels += pixels.shape[0] * pixels.shape[1]
        self._full_white += int(np.all(pixels == 255, axis=2).sum())
        for region in (
            pixels[:band],
            pixels[-band:],
            pixels[band:-band, :band],
            pixels[band:-band, -band:],
        ):
            values = region.reshape(-1, 3)
            self._boundary_pixels += values.shape[0]
            self._boundary_white += int(np.all(values == 255, axis=1).sum())
            float_values = values.astype(np.float64)
            self._sum += float_values.sum(axis=0)
            self._square_sum += np.square(float_values).sum(axis=0)
            quantized = values.astype(np.uint16) >> 4
            bins = (quantized[:, 0] << 8) | (quantized[:, 1] << 4) | quantized[:, 2]
            self._histogram += np.bincount(bins, minlength=4096)

    def to_json(self) -> dict[str, object]:
        mean = self._sum / self._boundary_pixels
        variance = self._square_sum / self._boundary_pixels - np.square(mean)
        dominant = int(self._histogram.argmax())
        return {
            "image_count": self._image_count,
            "image_size": self._image_size,
            "boundary_band_width": self._band_width,
            "full_exact_white_fraction": round(
                self._full_white / self._full_pixels,
                12,
            ),
            "boundary_exact_white_fraction": round(
                self._boundary_white / self._boundary_pixels,
                12,
            ),
            "boundary_channel_stddev": [
                round(math.sqrt(max(0.0, value)), 6) for value in variance
            ],
            "dominant_boundary_rgb4_bin": [
                ((dominant >> 8) & 15) * 16,
                ((dominant >> 4) & 15) * 16,
                (dominant & 15) * 16,
            ],
            "dominant_boundary_rgb4_fraction": round(
                int(self._histogram[dominant]) / self._boundary_pixels,
                12,
            ),
        }


def prepare_bilinear_control_dataset(
    source_root: Path,
    output_dir: Path,
    *,
    progress: StatusReporter,
) -> dict[str, object]:
    source_root = source_root.expanduser().resolve()
    crops, excluded, source_fingerprint = _audit_div2k_sources(
        source_root,
        progress=progress,
    )
    crop_records = tuple(crop.to_json() for crop in crops)
    output_dir = output_dir.expanduser().resolve()
    provenance_base: dict[str, object] = {
        "format": BILINEAR_CONTROL_PROVENANCE_FORMAT,
        "source_corpus": "DIV2K",
        "source_archives": DIV2K_ARCHIVE_SHA256,
        "input_source_fingerprint": source_fingerprint,
        "source_manifest": DIV2K_SOURCE_MANIFEST,
        "source_image_size": BILINEAR_SOURCE_SIZE,
        "target_image_size": BILINEAR_TARGET_SIZE,
        "crop_selection": DIV2K_CROP_SELECTION,
        "source_resize": "none",
        "source_padding": "none",
        "operation": BILINEAR_OPERATION,
        "pillow_version": PILLOW_VERSION,
        "pair_count": len(crops),
        "split_counts": _split_counts(crops),
        "excluded_below_1024": list(excluded),
    }
    if output_dir.exists():
        dataset, oracle, coverage = _audit_control_dataset(
            output_dir,
            crops=crops,
            crop_records=crop_records,
            provenance_base=provenance_base,
            progress=progress,
        )
        return _report(
            dataset,
            oracle=oracle,
            boundary_coverage=coverage,
            excluded_count=len(excluded),
            reused=True,
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    progress.begin(len(crops), "build full-frame bilinear control pairs")
    with TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.",
        suffix=".incomplete",
    ) as temporary:
        staging = Path(temporary)
        (staging / "source").mkdir()
        (staging / "target").mkdir()
        pair_records = []
        for crop in crops:
            source_relative = f"source/{crop.id}.png"
            target_relative = f"target/{crop.id}.png"
            _write_bilinear_pair(
                crop,
                staging / source_relative,
                staging / target_relative,
            )
            pair_records.append(
                {
                    "id": crop.id,
                    "group": crop.id,
                    "split": crop.split,
                    "source": source_relative,
                    "target": target_relative,
                }
            )
            progress.step(f"prepared bilinear pair {crop.id}")

        write_json(
            staging / "dataset.json",
            {
                "format": ASYMMETRIC_DATASET_FORMAT,
                "name": "div2k-bilinear-1024-to-128-control-v1",
                "source_image_size": BILINEAR_SOURCE_SIZE,
                "target_image_size": BILINEAR_TARGET_SIZE,
                "pairs": "pairs.jsonl",
            },
        )
        write_json_records(staging / "pairs.jsonl", pair_records)
        write_json_records(staging / DIV2K_SOURCE_MANIFEST, crop_records)
        dataset = audit_dataset(staging)
        oracle, coverage = _control_evidence(dataset)
        write_json(
            staging / "provenance.json",
            {
                **provenance_base,
                "source_manifest_sha256": sha256_file(
                    staging / DIV2K_SOURCE_MANIFEST
                ),
                "dataset_fingerprint": dataset.fingerprint,
                "boundary_coverage": coverage,
            },
        )
        os.rename(staging, output_dir)

    dataset, oracle, coverage = _audit_control_dataset(
        output_dir,
        crops=crops,
        crop_records=crop_records,
        provenance_base=provenance_base,
        progress=progress,
    )
    return _report(
        dataset,
        oracle=oracle,
        boundary_coverage=coverage,
        excluded_count=len(excluded),
        reused=False,
    )


def _audit_div2k_sources(
    source_root: Path,
    *,
    progress: StatusReporter,
) -> tuple[tuple[_Div2kCrop, ...], tuple[str, ...], str]:
    progress.phase("verify official DIV2K archives")
    for archive_name, expected_sha256 in DIV2K_ARCHIVE_SHA256.items():
        archive_path = source_root / archive_name
        if not archive_path.is_file():
            raise Pix2PixError(f"missing DIV2K archive: {archive_path.as_posix()}")
        if sha256_file(archive_path) != expected_sha256:
            raise Pix2PixError(f"DIV2K archive checksum mismatch: {archive_name}")

    progress.begin(900, "audit full-frame DIV2K sources")
    digest = hashlib.sha256()
    digest.update(b"aigen.pix2pix.div2k-source-corpus.v1")
    crops = []
    excluded = []
    for split, directory_name, first_id, last_id in DIV2K_SPLITS:
        directory = source_root / directory_name
        expected_names = {
            f"{image_id:04d}.png" for image_id in range(first_id, last_id + 1)
        }
        actual_names = {path.name for path in directory.glob("*.png")}
        if actual_names != expected_names:
            raise Pix2PixError(
                f"{directory_name} does not contain the official DIV2K file set"
            )
        for image_id in range(first_id, last_id + 1):
            filename = f"{image_id:04d}.png"
            input_path = directory / filename
            input_relative = f"{directory_name}/{filename}"
            input_sha256 = sha256_file(input_path)
            try:
                with Image.open(input_path) as image:
                    image.load()
                    mode = image.mode
                    width, height = image.size
            except OSError as error:
                raise Pix2PixError(
                    f"cannot decode DIV2K source {input_relative}: {error}"
                ) from error
            if mode != "RGB":
                raise Pix2PixError(
                    f"DIV2K source must be RGB: {input_relative} has mode {mode}"
                )
            for value in (
                split,
                input_relative,
                input_sha256,
                str(width),
                str(height),
            ):
                digest.update(b"\0")
                digest.update(value.encode())
            if width < BILINEAR_SOURCE_SIZE or height < BILINEAR_SOURCE_SIZE:
                excluded.append(input_relative)
                progress.step(f"audited DIV2K source {filename}")
                continue
            crop_left, crop_top = _crop_position(
                input_relative,
                input_sha256,
                width,
                height,
            )
            crops.append(
                _Div2kCrop(
                    id=f"div2k-{image_id:04d}",
                    split=split,
                    input_path=input_path,
                    input_relative=input_relative,
                    input_sha256=input_sha256,
                    input_width=width,
                    input_height=height,
                    crop_left=crop_left,
                    crop_top=crop_top,
                )
            )
            progress.step(f"audited DIV2K source {filename}")
    return tuple(crops), tuple(excluded), digest.hexdigest()


def _crop_position(
    input_relative: str,
    input_sha256: str,
    width: int,
    height: int,
) -> tuple[int, int]:
    digest = hashlib.sha256()
    digest.update(DIV2K_CROP_SELECTION.encode())
    digest.update(b"\0")
    digest.update(input_relative.encode())
    digest.update(b"\0")
    digest.update(input_sha256.encode())
    value = digest.digest()
    left = int.from_bytes(value[:8], "big") % (width - BILINEAR_SOURCE_SIZE + 1)
    top = int.from_bytes(value[8:16], "big") % (height - BILINEAR_SOURCE_SIZE + 1)
    return left, top


def _write_bilinear_pair(
    crop: _Div2kCrop,
    source_path: Path,
    target_path: Path,
) -> None:
    try:
        with Image.open(crop.input_path) as image:
            image.load()
            source = image.crop(
                (
                    crop.crop_left,
                    crop.crop_top,
                    crop.crop_left + BILINEAR_SOURCE_SIZE,
                    crop.crop_top + BILINEAR_SOURCE_SIZE,
                )
            )
        source.save(source_path, format="PNG", optimize=False)
        source.resize(
            (BILINEAR_TARGET_SIZE, BILINEAR_TARGET_SIZE),
            Image.Resampling.BILINEAR,
        ).save(target_path, format="PNG", optimize=False)
    except OSError as error:
        raise Pix2PixError(
            f"cannot prepare bilinear control pair {crop.id}: {error}"
        ) from error


def _audit_control_dataset(
    output_dir: Path,
    *,
    crops: tuple[_Div2kCrop, ...],
    crop_records: tuple[dict[str, object], ...],
    provenance_base: dict[str, object],
    progress: StatusReporter,
) -> tuple[AuditedDataset, dict[str, object], dict[str, object]]:
    progress.phase("audit full-frame bilinear control dataset")
    dataset = audit_dataset(output_dir)
    if (
        dataset.source_image_size != BILINEAR_SOURCE_SIZE
        or dataset.target_image_size != BILINEAR_TARGET_SIZE
    ):
        raise Pix2PixError(
            "bilinear control dataset must be 1024x1024 to 128x128"
        )
    expected_partition = tuple((crop.id, crop.id, crop.split) for crop in crops)
    if _pair_partition(dataset) != expected_partition:
        raise Pix2PixError(
            "bilinear control pair identities do not match the DIV2K crops"
        )
    source_manifest_path = output_dir / DIV2K_SOURCE_MANIFEST
    if read_json_records(
        source_manifest_path,
        label="bilinear control source manifest",
    ) != crop_records:
        raise Pix2PixError("bilinear control source manifest mismatch")
    oracle, coverage = _control_evidence(dataset)
    expected = {
        **provenance_base,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "dataset_fingerprint": dataset.fingerprint,
        "boundary_coverage": coverage,
    }
    provenance = read_json(
        output_dir / "provenance.json",
        label="bilinear control provenance",
    )
    require_exact_keys(provenance, set(expected), "bilinear control provenance")
    for key, value in expected.items():
        if provenance[key] != value:
            raise Pix2PixError(f"bilinear control provenance mismatch: {key}")
    return dataset, oracle, coverage


def _control_evidence(
    dataset: AuditedDataset,
) -> tuple[dict[str, object], dict[str, object]]:
    source_coverage = _BoundaryCoverage(BILINEAR_SOURCE_SIZE)
    target_coverage = _BoundaryCoverage(BILINEAR_TARGET_SIZE)
    for pair in dataset.pairs:
        try:
            with Image.open(pair.source_path) as image:
                image.load()
                source_coverage.add(image)
                expected = image.resize(
                    (BILINEAR_TARGET_SIZE, BILINEAR_TARGET_SIZE),
                    Image.Resampling.BILINEAR,
                )
            with Image.open(pair.target_path) as image:
                image.load()
                target_coverage.add(image)
                actual = image.copy()
        except OSError as error:
            raise Pix2PixError(
                f"cannot verify bilinear control pair {pair.id}: {error}"
            ) from error
        if ImageChops.difference(expected, actual).getbbox() is not None:
            raise Pix2PixError(
                f"target is not the exact bilinear reduction of its source: {pair.id}"
            )
    return (
        {
            "operation": BILINEAR_OPERATION,
            "pair_count": len(dataset.pairs),
            "exact_pair_count": len(dataset.pairs),
        },
        {
            "source": source_coverage.to_json(),
            "target": target_coverage.to_json(),
        },
    )


def _split_counts(crops: tuple[_Div2kCrop, ...]) -> dict[str, int]:
    return {
        split: sum(crop.split == split for crop in crops)
        for split, _, _, _ in DIV2K_SPLITS
    }


def _pair_partition(dataset: AuditedDataset) -> tuple[tuple[str, str, str], ...]:
    return tuple((pair.id, pair.group, pair.split) for pair in dataset.pairs)


def _report(
    dataset: AuditedDataset,
    *,
    oracle: dict[str, object],
    boundary_coverage: dict[str, object],
    excluded_count: int,
    reused: bool,
) -> dict[str, object]:
    return {
        **dataset.to_json(),
        "kind": BILINEAR_CONTROL_KIND,
        "oracle": oracle,
        "boundary_coverage": boundary_coverage,
        "excluded_below_1024": excluded_count,
        "reused": reused,
    }
