#!/usr/bin/env python3
"""Build the current JSEED subject-LoRA dataset preview.

Every approved master is composited without distortion into eight different
FLUX.2 aspect-ratio buckets and all eight background colors. Bucket and
background assignments rotate independently per master, preventing drawing
style, expression, view, framing, canvas shape, and background from becoming
fixed pairs.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, REPO_ROOT.as_posix())

from aigen.keyframe_segmentation import (  # noqa: E402
    AnimeForegroundSegmenter,
    AnimeSegmentationConfig,
)


TRAIN_SOURCES = REPO_ROOT / "assets/lora/JSEED/train"
NEW_SOURCES = REPO_ROOT / "assets/lora/JSEED/uncleaned"
SUBJECT_EXTENT = 0.92
FOREGROUND_ALPHA_THRESHOLD = 0.90
VARIANTS_PER_MASTER = 8

# Official approximately one-megapixel FLUX.2 bucket sequence, stored as
# (height, width), from widest to tallest.
BUCKETS = (
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
)

BACKGROUNDS = (
    ("white", (255, 255, 255)),
    ("warm-ivory", (244, 239, 231)),
    ("cool-light-gray", (224, 230, 236)),
    ("pale-blue", (210, 224, 238)),
    ("pale-sage", (218, 230, 218)),
    ("pale-rose", (238, 220, 226)),
    ("medium-gray", (163, 169, 176)),
    ("dark-slate", (50, 58, 70)),
)

WIDE_SEATED_BUCKETS = (0, 2, 4, 6, 8, 9, 10, 11)

STYLES = {
    "canon": "Black ink line art with light watercolor-like coloring.",
    "cellshaded": "Anime cel shading with clean ink outlines.",
    "lineart": "Black-and-white ink line art.",
}


@dataclass(frozen=True)
class Master:
    name: str
    path: Path
    style: str
    content: str


@dataclass(frozen=True)
class Foreground:
    image: Image.Image
    anchor_bottom: bool


def _master(
    name: str,
    path: Path,
    style: str,
    content: str,
) -> Master:
    return Master(
        name=name,
        path=path,
        style=STYLES[style],
        content=content,
    )


EXISTING_GROUPS = (
    (
        "Full-body front view with a neutral expression.",
        (
            ("front-full", "front-full.png", "canon", None),
            (
                "front-full-cellshaded",
                "front-full-cellshaded.png",
                "cellshaded",
                "Full-body front view with a faint smile.",
            ),
            ("front-full-lineart", "front-full-lineart.png", "lineart", None),
        ),
    ),
    (
        "Full-body left-facing side view with a neutral expression.",
        (
            ("side-full", "side-full.png", "canon", None),
            ("side-full-cellshaded", "side-full-cellshaded.png", "cellshaded", None),
            ("side-full-lineart", "side-full-lineart.png", "lineart", None),
        ),
    ),
    (
        "Full-body rear view.",
        (
            ("back-full", "back-full.png", "canon", None),
            ("back-full-cellshaded", "back-full-cellshaded.png", "cellshaded", None),
            ("back-full-lineart", "back-full-lineart.png", "lineart", None),
        ),
    ),
    (
        "Front upper-body portrait with a neutral expression.",
        (
            ("front-upperbody", "front-upperbody.png", "canon", None),
            (
                "front-portrait-cellshaded",
                "front-portrait-cellshaded.png",
                "cellshaded",
                None,
            ),
            (
                "front-portrait-lineart",
                "font-portrait-lineart.png",
                "lineart",
                None,
            ),
        ),
    ),
    (
        "Left-facing side-profile upper-body portrait with a neutral expression.",
        (("side-upperbody", "side-upperbody.png", "canon", None),),
    ),
    (
        "Left-facing three-quarter upper-body view with a neutral expression.",
        (
            ("three-quarter-upperbody", "three-quarter-upperbody.png", "canon", None),
            (
                "three-quarter-upperbody-cellshaded",
                "three-quarter-upperbody-cellshaded.png",
                "cellshaded",
                None,
            ),
            (
                "three-quarter-upperbody-lineart",
                "three-quarter-upperbody-lineart.png",
                "lineart",
                None,
            ),
        ),
    ),
    (
        "Upper-body portrait looking to the left with a neutral expression.",
        (
            ("portrait-lookleft", "portrait-lookleft.png", "canon", None),
            (
                "portrait-lookleft-cellshaded",
                "portrait-lookleft-cellshaded.png",
                "cellshaded",
                None,
            ),
            (
                "portrait-lookleft-lineart",
                "portrait-lookleft-lineart.png",
                "lineart",
                None,
            ),
        ),
    ),
)

NEW_CONTENT = {
    "determined_fullbody_wide": "Full-body standing view with a determined expression.",
    "determined_nearly_fullbody": "Nearly full-body standing view with a determined expression.",
    "determined_portrait": "Portrait with a determined expression.",
    "determined_waistup_wide": "Waist-up view with a determined expression.",
    "upset_fullbody_wide": "Seated full-body view, tense and sweating.",
    "upset_nearly_fullbody": "Nearly full-body view, visibly upset with tears.",
    "upset_portrait": "Portrait, visibly upset with tears.",
    "upset_waistup_wide": "Waist-up view, visibly upset with tears.",
    "warm_smile_nearly_fullbody": "Nearly full-body standing view with a warm smile.",
    "warm_smile_portrait": "Portrait with a warm smile.",
    "warm_smile_waistup_wide": "Waist-up view with a warm smile.",
}


def _new_masters() -> tuple[Master, ...]:
    masters = []
    for stem, content in NEW_CONTENT.items():
        masters.extend(
            (
                _master(stem, NEW_SOURCES / f"{stem}.png", "canon", content),
                _master(
                    f"{stem}-cellshaded",
                    NEW_SOURCES / f"{stem}-cellshaded.png",
                    "cellshaded",
                    content,
                ),
                _master(
                    f"{stem}-lineart",
                    NEW_SOURCES / f"{stem}-lineart.png",
                    "lineart",
                    content,
                ),
            )
        )
    return tuple(masters)


def _existing_masters() -> tuple[Master, ...]:
    masters = []
    for shared_content, variants in EXISTING_GROUPS:
        for name, file_name, style, content in variants:
            masters.append(
                _master(
                    name,
                    TRAIN_SOURCES / file_name,
                    style,
                    content or shared_content,
                )
            )
    return tuple(masters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    masters = _existing_masters() + _new_masters()
    _validate_masters(masters)
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True)

    records = []
    with closing(
        AnimeForegroundSegmenter(AnimeSegmentationConfig())
    ) as segmenter:
        for master_index, master in enumerate(masters):
            foreground = _load_foreground(master.path, segmenter)
            for variant, bucket_index in enumerate(
                _bucket_indices(master_index, master)
            ):
                background_index = (master_index + variant * 3) % len(BACKGROUNDS)
                target_height, target_width = BUCKETS[bucket_index]
                background_name, background_color = BACKGROUNDS[background_index]
                image, scale = _compose(
                    foreground,
                    size=(target_width, target_height),
                    background=background_color,
                )
                file_name = (
                    f"{master.name}__b{bucket_index:02d}_{target_width}x{target_height}"
                    f"__{background_name}.png"
                )
                image.save(train_dir / file_name, compress_level=4)
                prompt = (
                    f"JSEED. {master.style} {master.content} "
                    f"Plain {background_name.replace('-', ' ')} background."
                )
                records.append(
                    {
                        "file_name": file_name,
                        "prompt": prompt,
                        "master": master.name,
                        "source": master.path.relative_to(REPO_ROOT).as_posix(),
                        "style": master.style,
                        "content": master.content,
                        "background": background_name,
                        "bucket": [target_height, target_width],
                        "foreground_size": list(foreground.image.size),
                        "anchor_bottom": foreground.anchor_bottom,
                        "scale": scale,
                    }
                )

    _write_metadata(records, train_dir / "metadata.jsonl")
    _write_manifest(records, masters, output_dir / "manifest.json")
    _save_reviews(records, train_dir, output_dir)
    print(f"{len(masters)} masters x {VARIANTS_PER_MASTER} variants = {len(records)} images")
    print(f"dataset -> {train_dir}")


def _bucket_indices(master_index: int, master: Master) -> tuple[int, ...]:
    if master.name.startswith("upset_fullbody_wide"):
        return WIDE_SEATED_BUCKETS
    return tuple(
        (master_index * 5 + variant * 2) % len(BUCKETS)
        for variant in range(VARIANTS_PER_MASTER)
    )


def _validate_masters(masters: tuple[Master, ...]) -> None:
    paths = {master.path for master in masters}
    missing = [path for path in paths if not path.is_file()]
    if missing:
        joined = "\n".join(path.as_posix() for path in missing)
        raise FileNotFoundError(f"Missing approved masters:\n{joined}")
    names = [master.name for master in masters]
    if len(names) != len(set(names)):
        raise ValueError("Master names must be unique")


def _load_foreground(
    path: Path,
    segmenter: AnimeForegroundSegmenter,
) -> Foreground:
    with Image.open(path) as source:
        image = source.convert("RGB")
    pixels = np.asarray(image)
    foreground = segmenter.segment_image(pixels) >= FOREGROUND_ALPHA_THRESHOLD
    anchor_bottom = bool(foreground[-1].any())
    alpha = foreground.astype(np.uint8) * 255
    rgba = np.dstack((pixels, alpha))
    foreground = Image.fromarray(rgba, "RGBA")
    box = foreground.getchannel("A").getbbox()
    if box is None:
        raise ValueError(f"No foreground found: {path}")
    return Foreground(image=foreground.crop(box), anchor_bottom=anchor_bottom)


def _compose(
    foreground: Foreground,
    *,
    size: tuple[int, int],
    background: tuple[int, int, int],
) -> tuple[Image.Image, float]:
    width, height = size
    scale = min(
        width * SUBJECT_EXTENT / foreground.image.width,
        height * SUBJECT_EXTENT / foreground.image.height,
    )
    scaled_size = (
        max(1, round(foreground.image.width * scale)),
        max(1, round(foreground.image.height * scale)),
    )
    scaled = foreground.image.resize(scaled_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, (*background, 255))
    y = height - scaled.height if foreground.anchor_bottom else (height - scaled.height) // 2
    position = ((width - scaled.width) // 2, y)
    canvas.alpha_composite(scaled, position)
    return canvas.convert("RGB"), scale


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
    masters: tuple[Master, ...],
    path: Path,
) -> None:
    bucket_counts = {
        f"{width}x{height}": sum(record["bucket"] == [height, width] for record in records)
        for height, width in BUCKETS
    }
    background_counts = {
        name: sum(record["background"] == name for record in records)
        for name, _color in BACKGROUNDS
    }
    payload = {
        "kind": "jseed-subject-lora-dataset-preview",
        "master_count": len(masters),
        "variants_per_master": VARIANTS_PER_MASTER,
        "image_count": len(records),
        "foreground_segmentation": "skytnt-anime-seg-isnet-onnx-1024",
        "foreground_alpha_threshold": FOREGROUND_ALPHA_THRESHOLD,
        "subject_extent": SUBJECT_EXTENT,
        "buckets": [list(bucket) for bucket in BUCKETS],
        "backgrounds": [
            {"name": name, "rgb": list(color)} for name, color in BACKGROUNDS
        ],
        "bucket_counts": bucket_counts,
        "background_counts": background_counts,
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _save_reviews(
    records: list[dict[str, object]],
    train_dir: Path,
    output_dir: Path,
) -> None:
    by_master = {}
    for record in records:
        by_master.setdefault(record["master"], []).append(record)

    master_samples = [variants[0] for variants in by_master.values()]
    _save_contact_sheet(
        master_samples,
        train_dir,
        output_dir / "review_masters.png",
        columns=6,
    )

    extreme_samples = []
    for variants in by_master.values():
        foreground_width, foreground_height = variants[0]["foreground_size"]
        foreground_ratio = foreground_width / foreground_height
        extreme_samples.append(
            max(
                variants,
                key=lambda record: abs(
                    math.log(
                        (record["bucket"][1] / record["bucket"][0]) / foreground_ratio
                    )
                ),
            )
        )
    _save_contact_sheet(
        extreme_samples,
        train_dir,
        output_dir / "review_extreme_ratios.png",
        columns=6,
    )

    background_samples = []
    for background_name, _color in BACKGROUNDS:
        for style in STYLES.values():
            background_samples.append(
                next(
                    record
                    for record in records
                    if record["background"] == background_name
                    and record["style"] == style
                )
            )
    _save_contact_sheet(
        background_samples,
        train_dir,
        output_dir / "review_backgrounds.png",
        columns=3,
    )

    lineart_dark_samples = [
        record
        for record in records
        if record["style"] == STYLES["lineart"]
        and record["background"] == "dark-slate"
    ]
    _save_contact_sheet(
        lineart_dark_samples,
        train_dir,
        output_dir / "review_lineart_dark.png",
        columns=4,
    )

    all_dark_samples = [
        record for record in records if record["background"] == "dark-slate"
    ]
    _save_contact_sheet(
        all_dark_samples,
        train_dir,
        output_dir / "review_all_dark.png",
        columns=4,
    )

    bucket_samples = []
    for height, width in BUCKETS:
        candidates = [
            record for record in records if record["bucket"] == [height, width]
        ]
        bucket_samples.extend(candidates[:3])
    _save_contact_sheet(
        bucket_samples,
        train_dir,
        output_dir / "review_buckets.png",
        columns=3,
    )


def _save_contact_sheet(
    records: list[dict[str, object]],
    train_dir: Path,
    output_path: Path,
    *,
    columns: int,
) -> None:
    cell_width = 280
    cell_height = 240
    label_height = 48
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * (cell_height + label_height)),
        (38, 38, 38),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, record in enumerate(records):
        with Image.open(train_dir / record["file_name"]) as source:
            thumb = source.convert("RGB")
        thumb.thumbnail((cell_width - 12, cell_height - 12), Image.Resampling.LANCZOS)
        cell_x = index % columns * cell_width
        cell_y = index // columns * (cell_height + label_height)
        x = cell_x + (cell_width - thumb.width) // 2
        y = cell_y + (cell_height - thumb.height) // 2
        sheet.paste(thumb, (x, y))
        label = (
            f"{record['master']}\n"
            f"{record['bucket'][1]}x{record['bucket'][0]} {record['background']}"
        )
        draw.multiline_text(
            (cell_x + 4, cell_y + cell_height + 3),
            label,
            fill="white",
            font=font,
            spacing=2,
        )
    sheet.save(output_path)


if __name__ == "__main__":
    main()
