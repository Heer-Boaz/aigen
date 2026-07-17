#!/usr/bin/env python3
"""Build the cleaned, aspect-bucketed JSEED subject-LoRA dataset.

Every real image in ``assets/lora/JSEED/masters`` is an approved observation
and is emitted exactly once. AnimeSeg is used only to protect Jillian while the
existing white, blue, or pink background is normalized to a clean solid plate.
The protected rectangular subject crop is then placed on an exact-ratio canvas
and uniformly resized to one official FLUX.2 bucket. No axis is ever stretched
independently and no foreground cutout is produced.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, REPO_ROOT.as_posix())

from aigen.keyframe_segmentation import (  # noqa: E402
    AnimeForegroundSegmenter,
    AnimeSegmentationConfig,
)
from build_jseed_dataset_v12 import BUCKETS, MASTERS as WHITE_MASTERS  # noqa: E402


MASTERS_DIR = REPO_ROOT / "assets/lora/JSEED/masters"
DEFAULT_OUTPUT = REPO_ROOT / "assets/lora/JSEED/dataset-v13"
DEFAULT_REVIEW = REPO_ROOT / "assets/lora/JSEED/review/dataset-v13"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FOREGROUND_BBOX_THRESHOLD = 0.05
SUBJECT_MARGIN = 0.05
BUCKET_STRIDE = 5


@dataclass(frozen=True)
class Master:
    file_name: str
    prompt: str
    background: str


@dataclass(frozen=True)
class Layout:
    canvas_size: tuple[int, int]
    content_position: tuple[int, int]
    subject_box: tuple[int, int, int, int]
    scale: float


WHITE_PROMPTS = dict(WHITE_MASTERS)
NEW_MASTER_VARIANTS = (
    ("three-quarter-upperbody-blue.png", "three-quarter-upperbody.png", "blue"),
    (
        "three-quarter-upperbody-cellshaded-blue.png",
        "three-quarter-upperbody-cellshaded.png",
        "blue",
    ),
    (
        "three-quarter-upperbody-lineart-blue.png",
        "three-quarter-upperbody-lineart.png",
        "blue",
    ),
    ("warm_smile_nearly_fullbody-pink.jpg", "warm_smile_nearly_fullbody.png", "pink"),
    ("warm_smile_portrait-blue.png", "warm_smile_portrait.png", "blue"),
    ("warm_smile_portrait-pink.png", "warm_smile_portrait.png", "pink"),
    (
        "warm_smile_waistup_wide-lineart-blue.png",
        "warm_smile_waistup_wide-lineart.png",
        "blue",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()

    output = args.output.resolve()
    review = args.review.resolve()
    if output.exists():
        raise FileExistsError(output)
    if review.exists():
        raise FileExistsError(review)

    masters = _masters()
    _validate_master_inventory(masters)
    background_colors = _background_colors(masters)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".jseed-dataset-v13-", dir=output.parent)
    )
    staged_output = staging_root / "dataset"
    staged_review = staging_root / "review"
    staged_output.mkdir()
    staged_review.mkdir()

    records: list[dict[str, object]] = []
    with closing(AnimeForegroundSegmenter(AnimeSegmentationConfig())) as segmenter:
        for index, master in enumerate(masters):
            source_path = MASTERS_DIR / master.file_name
            source = _load_rgb(source_path)
            source_pixels = np.asarray(source)
            alpha = np.clip(segmenter.segment_image(source_pixels), 0.0, 1.0).astype(np.float32)
            clean, core_delta = _clean_background(
                source_pixels,
                alpha,
                background_colors[master.background],
            )
            raw_subject_box = _subject_box(alpha)
            content_box = _expand_box(raw_subject_box, source.size)
            bucket_index = index * BUCKET_STRIDE % len(BUCKETS)
            bucket_height, bucket_width = BUCKETS[bucket_index]
            final, layout = _layout_in_bucket(
                clean,
                content_box,
                raw_subject_box,
                (bucket_width, bucket_height),
                background_colors[master.background],
                source.size,
            )
            output_name = f"{source_path.stem}.png"
            final.save(staged_output / output_name, compress_level=4)
            records.append(
                {
                    "file_name": output_name,
                    "prompt": master.prompt,
                    "source": source_path.relative_to(REPO_ROOT).as_posix(),
                    "background": master.background,
                    "background_rgb": list(background_colors[master.background]),
                    "bucket_index": bucket_index,
                    "bucket": [bucket_height, bucket_width],
                    "source_size": list(source.size),
                    "raw_subject_box": list(raw_subject_box),
                    "content_box": list(content_box),
                    "canvas_size": list(layout.canvas_size),
                    "content_position": list(layout.content_position),
                    "output_subject_box": list(layout.subject_box),
                    "uniform_scale": layout.scale,
                    "opaque_core_max_delta": core_delta,
                }
            )
            print(f"{index + 1:02d}/{len(masters)} {master.file_name} -> bucket {bucket_index:02d}", flush=True)

    _write_metadata(records, staged_output / "metadata.jsonl")
    _write_manifest(records, background_colors, staged_review / "manifest.json")
    _save_contact_sheet(records, staged_output, staged_review / "contact-sheet.png")
    for position in ("hair", "torso", "lower"):
        _save_detail_sheet(records, staged_output, staged_review / f"details-{position}.png", position)
    _validate_dataset(records, staged_output)

    review.parent.mkdir(parents=True, exist_ok=True)
    staged_output.rename(output)
    staged_review.rename(review)

    bucket_arg = ";".join(f"{height},{width}" for height, width in BUCKETS)
    print(f"{len(records)} beelden -> {output}")
    print(f"review -> {review}")
    print(f'--aspect_ratio_buckets "{bucket_arg}"')


def _masters() -> tuple[Master, ...]:
    white = tuple(Master(file_name, prompt, "white") for file_name, prompt in WHITE_MASTERS)
    variants = tuple(
        Master(file_name, _replace_background(WHITE_PROMPTS[source], background), background)
        for file_name, source, background in NEW_MASTER_VARIANTS
    )
    return white + variants


def _replace_background(prompt: str, background: str) -> str:
    suffix = "Plain white background."
    if not prompt.endswith(suffix):
        raise ValueError(f"Base caption does not end in {suffix!r}: {prompt}")
    return prompt.removesuffix(suffix) + f"Plain {background} background."


def _validate_master_inventory(masters: tuple[Master, ...]) -> None:
    expected = {master.file_name for master in masters}
    if len(expected) != len(masters):
        raise ValueError("Master filenames must be unique")
    actual = {
        path.name
        for path in MASTERS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"Master inventory mismatch; missing={missing}, unexpected={unexpected}")


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        if "A" in source.getbands() and source.getchannel("A").getextrema() != (255, 255):
            raise ValueError(f"Master has non-opaque alpha: {path}")
        return source.convert("RGB")


def _border_median(image: Image.Image) -> tuple[int, int, int]:
    pixels = np.asarray(image)
    width = 8
    edges = np.concatenate(
        (
            pixels[:width].reshape(-1, 3),
            pixels[-width:].reshape(-1, 3),
            pixels[:, :width].reshape(-1, 3),
            pixels[:, -width:].reshape(-1, 3),
        )
    )
    return tuple(int(round(value)) for value in np.median(edges, axis=0))


def _background_colors(masters: tuple[Master, ...]) -> dict[str, tuple[int, int, int]]:
    colors: dict[str, tuple[int, int, int]] = {"white": (255, 255, 255)}
    for label in ("blue", "pink"):
        medians = [
            _border_median(_load_rgb(MASTERS_DIR / master.file_name))
            for master in masters
            if master.background == label
        ]
        colors[label] = tuple(
            int(round(value)) for value in np.median(np.asarray(medians), axis=0)
        )
    return colors


def _clean_background(
    source: np.ndarray,
    alpha: np.ndarray,
    background: tuple[int, int, int],
) -> tuple[Image.Image, int]:
    source_float = source.astype(np.float32)
    background_float = np.asarray(background, dtype=np.float32)
    clean = source_float * alpha[..., None] + background_float * (1.0 - alpha[..., None])
    clean_u8 = np.clip(np.rint(clean), 0, 255).astype(np.uint8)
    core = alpha >= 0.99
    core_delta = int(
        np.abs(clean_u8[core].astype(np.int16) - source[core].astype(np.int16)).max()
    )
    return Image.fromarray(clean_u8, mode="RGB"), core_delta


def _subject_box(alpha: np.ndarray) -> tuple[int, int, int, int]:
    foreground = alpha >= FOREGROUND_BBOX_THRESHOLD
    labels, count = ndimage.label(foreground)
    if count == 0:
        raise ValueError("AnimeSeg returned no foreground")
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    subject = labels == int(component_sizes.argmax())
    ys, xs = np.nonzero(subject)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _expand_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    width, height = image_size
    margin_x = round((right - left) * SUBJECT_MARGIN)
    margin_y = round((bottom - top) * SUBJECT_MARGIN)
    return (
        max(0, left - margin_x),
        max(0, top - margin_y),
        min(width, right + margin_x),
        min(height, bottom + margin_y),
    )


def _layout_in_bucket(
    clean: Image.Image,
    content_box: tuple[int, int, int, int],
    raw_subject_box: tuple[int, int, int, int],
    bucket_size: tuple[int, int],
    background: tuple[int, int, int],
    source_size: tuple[int, int],
) -> tuple[Image.Image, Layout]:
    bucket_width, bucket_height = bucket_size
    divisor = math.gcd(bucket_width, bucket_height)
    ratio_width = bucket_width // divisor
    ratio_height = bucket_height // divisor
    content = clean.crop(content_box)
    multiple = max(
        math.ceil(content.width / ratio_width),
        math.ceil(content.height / ratio_height),
    )
    canvas_size = (multiple * ratio_width, multiple * ratio_height)
    canvas = Image.new("RGB", canvas_size, background)

    raw_left, raw_top, raw_right, raw_bottom = raw_subject_box
    source_width, source_height = source_size
    x = _anchored_offset(
        canvas.width - content.width,
        touches_start=raw_left == 0,
        touches_end=raw_right == source_width,
    )
    y = _anchored_offset(
        canvas.height - content.height,
        touches_start=raw_top == 0,
        touches_end=raw_bottom == source_height,
    )
    canvas.paste(content, (x, y))
    scale = bucket_width / canvas.width
    if not math.isclose(scale, bucket_height / canvas.height, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Bucket layout would require non-uniform scaling")
    final = canvas.resize(bucket_size, Image.Resampling.LANCZOS)

    content_left, content_top, _content_right, _content_bottom = content_box
    subject_box = tuple(
        round(value * scale)
        for value in (
            x + raw_left - content_left,
            y + raw_top - content_top,
            x + raw_right - content_left,
            y + raw_bottom - content_top,
        )
    )
    return final, Layout(canvas.size, (x, y), subject_box, scale)


def _anchored_offset(space: int, *, touches_start: bool, touches_end: bool) -> int:
    if touches_start and not touches_end:
        return 0
    if touches_end and not touches_start:
        return space
    return space // 2


def _write_metadata(records: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    {"file_name": record["file_name"], "prompt": record["prompt"]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_manifest(
    records: list[dict[str, object]],
    background_colors: dict[str, tuple[int, int, int]],
    path: Path,
) -> None:
    bucket_counts = {
        f"{width}x{height}": sum(record["bucket"] == [height, width] for record in records)
        for height, width in BUCKETS
    }
    background_counts = {
        label: sum(record["background"] == label for record in records)
        for label in background_colors
    }
    payload = {
        "kind": "jseed-subject-lora-dataset",
        "image_count": len(records),
        "source_directory": MASTERS_DIR.relative_to(REPO_ROOT).as_posix(),
        "foreground_protection": "skytnt-anime-seg-isnet-onnx-1024-soft-mask",
        "foreground_bbox_threshold": FOREGROUND_BBOX_THRESHOLD,
        "subject_margin": SUBJECT_MARGIN,
        "background_colors": {
            label: list(color) for label, color in background_colors.items()
        },
        "background_counts": background_counts,
        "bucket_counts": bucket_counts,
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_contact_sheet(
    records: list[dict[str, object]],
    dataset_dir: Path,
    output_path: Path,
) -> None:
    columns = 6
    cell_width = 240
    image_height = 240
    label_height = 52
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * (image_height + label_height)), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, record in enumerate(records):
        with Image.open(dataset_dir / str(record["file_name"])) as image:
            thumb = image.convert("RGB")
        thumb.thumbnail((cell_width - 10, image_height - 10), Image.Resampling.LANCZOS)
        cell_x = index % columns * cell_width
        cell_y = index // columns * (image_height + label_height)
        x = cell_x + (cell_width - thumb.width) // 2
        y = cell_y + (image_height - thumb.height) // 2
        sheet.paste(thumb, (x, y))
        label = f"{record['file_name']}\nb{record['bucket_index']:02d} {record['background']}"
        draw.multiline_text((cell_x + 4, cell_y + image_height + 3), label, fill="white", font=font, spacing=2)
    sheet.save(output_path)


def _save_detail_sheet(
    records: list[dict[str, object]],
    dataset_dir: Path,
    output_path: Path,
    position: str,
) -> None:
    crop_size = 192
    label_height = 32
    columns = 8
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * crop_size, rows * (crop_size + label_height)), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    fractions = {"hair": 0.18, "torso": 0.52, "lower": 0.84}
    for index, record in enumerate(records):
        with Image.open(dataset_dir / str(record["file_name"])) as image:
            source = image.convert("RGB")
        left, top, right, bottom = record["output_subject_box"]
        center_x = (left + right) // 2
        center_y = round(top + (bottom - top) * fractions[position])
        crop = _center_crop(source, center_x, center_y, crop_size)
        cell_x = index % columns * crop_size
        cell_y = index // columns * (crop_size + label_height)
        sheet.paste(crop, (cell_x, cell_y))
        draw.text((cell_x + 3, cell_y + crop_size + 3), str(record["file_name"])[:27], fill="white", font=font)
    sheet.save(output_path)


def _center_crop(image: Image.Image, center_x: int, center_y: int, size: int) -> Image.Image:
    left = min(max(0, center_x - size // 2), image.width - size)
    top = min(max(0, center_y - size // 2), image.height - size)
    return image.crop((left, top, left + size, top + size))


def _validate_dataset(records: list[dict[str, object]], dataset_dir: Path) -> None:
    if len(records) != len(_masters()):
        raise ValueError("Not every approved master was emitted exactly once")
    names = [str(record["file_name"]) for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Output filenames must be unique")
    bucket_counts = [sum(record["bucket"] == list(bucket) for record in records) for bucket in BUCKETS]
    if not all(bucket_counts):
        raise ValueError(f"Every FLUX.2 bucket must be represented: {bucket_counts}")
    for record in records:
        height, width = record["bucket"]
        with Image.open(dataset_dir / str(record["file_name"])) as image:
            if image.size != (width, height):
                raise ValueError(f"Wrong output size for {record['file_name']}: {image.size}")


if __name__ == "__main__":
    main()
