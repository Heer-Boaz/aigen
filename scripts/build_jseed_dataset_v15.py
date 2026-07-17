#!/usr/bin/env python3
"""Build JSEED dataset v15 — pad-only herstel van de v13-bucketfout.

Elke master behoudt exact zijn verhouding, uitsnede, onderwerp-grootte en
compositie. De enige bewerkingen zijn:

1. randreplicatie-padding (max ~1%) tot exact de verhouding van de
   DICHTSTBIJZIJNDE officiële FLUX.2-bucket, verankerd wég van randen waar de
   figuur is afgesneden; en
2. uniforme Lanczos-schaling naar exact die bucketmaat.

Randreplicatie kopieert uitsluitend bestaande randpixels, zodat ook de
gradiëntachtergronden van de blauw/roze masters naadloos doorlopen. Geen
segmentatie, geen compositing: geen enkel bronpixel wordt gewijzigd vóór de
schaling. Aspectvariatie komt uit de bronbeelden zelf; welke buckets gevuld
raken is een uitkomst, geen doel.

v13 (afgekeurd) verdeelde de masters modulo over alle 17 buckets, waardoor
26/59 beelden de verkeerde oriëntatie kregen en de framing systematisch werd
vernield; bewijs in assets/lora/JSEED/review/dataset-v13/manifest.json.

Eén expliciete rij per master met de volledige caption als literal.

v15 t.o.v. v14 (pixels identiek): de drie determined_waistup_wide-captions
benoemen nu de linksplaatsing van de figuur, zodat die compositie promptbaar
blijft in plaats van in "waist-up + determined" gebakken te worden
(captioning-wet).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
MASTERS_DIR = REPO_ROOT / "assets/lora/JSEED/masters"
DEFAULT_OUTPUT = REPO_ROOT / "assets/lora/JSEED/dataset-v15"
DEFAULT_REVIEW = REPO_ROOT / "assets/lora/JSEED/review/dataset-v15"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Officiële ~1MP FLUX.2-buckets als (hoogte, breedte).
BUCKETS = (
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
)

# Padding-plafond: de masters passen van nature op de bucketverhoudingen
# (gemeten maximum 0,9%). Alles daarboven is een bouwfout, geen dataeigenschap.
MAX_PAD_FRACTION = 0.02

# Framing-check: master en datasetbeeld worden onafhankelijk verkleind en
# vergeleken; sub-pixelverschillen door de twee resample-routes blijven ruim
# onder deze grens, een verkeerde bucket/plaatsing gaat er ver overheen.
FRAMING_THUMB = 256
FRAMING_MAX_MEAN_DIFF = 3.0

# (bestand in masters/, volledige caption)
MASTERS = [
    ("front-full.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Full-body front view with a neutral expression. Plain white background."),
    ("front-full-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Full-body front view with a faint smile. Plain white background."),
    ("front-full-lineart.png",
     "JSEED. Black-and-white ink line art. Full-body front view with a neutral expression. Plain white background."),
    ("side-full.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Full-body left-facing side view with a neutral expression. Plain white background."),
    ("side-full-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Full-body left-facing side view with a neutral expression. Plain white background."),
    ("side-full-lineart.png",
     "JSEED. Black-and-white ink line art. Full-body left-facing side view with a neutral expression. Plain white background."),
    ("back-full.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Full-body rear view. Plain white background."),
    ("back-full-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Full-body rear view. Plain white background."),
    ("back-full-lineart.png",
     "JSEED. Black-and-white ink line art. Full-body rear view. Plain white background."),
    ("front-upperbody.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Front upper-body portrait with a neutral expression. Plain white background."),
    ("front-portrait-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Front upper-body portrait with a neutral expression. Plain white background."),
    ("font-portrait-lineart.png",
     "JSEED. Black-and-white ink line art. Front upper-body portrait with a neutral expression. Plain white background."),
    ("side-upperbody.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Left-facing side-profile upper-body portrait with a neutral expression. Plain white background."),
    ("three-quarter-upperbody.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Left-facing three-quarter upper-body view with a neutral expression. Plain white background."),
    ("three-quarter-upperbody-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Left-facing three-quarter upper-body view with a neutral expression. Plain white background."),
    ("three-quarter-upperbody-lineart.png",
     "JSEED. Black-and-white ink line art. Left-facing three-quarter upper-body view with a neutral expression. Plain white background."),
    ("portrait-lookleft.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Upper-body portrait looking to the left with a neutral expression. Plain white background."),
    ("portrait-lookleft-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Upper-body portrait looking to the left with a neutral expression. Plain white background."),
    ("portrait-lookleft-lineart.png",
     "JSEED. Black-and-white ink line art. Upper-body portrait looking to the left with a neutral expression. Plain white background."),
    ("determined_fullbody_wide.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Full-body standing view with a determined expression. Plain white background."),
    ("determined_fullbody_wide-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Full-body standing view with a determined expression. Plain white background."),
    ("determined_fullbody_wide-lineart.png",
     "JSEED. Black-and-white ink line art. Full-body standing view with a determined expression. Plain white background."),
    ("determined_nearly_fullbody.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Nearly full-body standing view with a determined expression. Plain white background."),
    ("determined_nearly_fullbody-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Nearly full-body standing view with a determined expression. Plain white background."),
    ("determined_nearly_fullbody-lineart.png",
     "JSEED. Black-and-white ink line art. Nearly full-body standing view with a determined expression. Plain white background."),
    ("determined_portrait.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Portrait with a determined expression. Plain white background."),
    ("determined_portrait-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Portrait with a determined expression. Plain white background."),
    ("determined_portrait-lineart.png",
     "JSEED. Black-and-white ink line art. Portrait with a determined expression. Plain white background."),
    ("determined_waistup_wide.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Waist-up view with a determined expression, positioned at the left of the frame with empty space to the right. Plain white background."),
    ("determined_waistup_wide-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Waist-up view with a determined expression, positioned at the left of the frame with empty space to the right. Plain white background."),
    ("determined_waistup_wide-lineart.png",
     "JSEED. Black-and-white ink line art. Waist-up view with a determined expression, positioned at the left of the frame with empty space to the right. Plain white background."),
    ("upset_fullbody_wide.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Seated full-body view, tense and sweating. Plain white background."),
    ("upset_fullbody_wide-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Seated full-body view, tense and sweating. Plain white background."),
    ("upset_fullbody_wide-lineart.png",
     "JSEED. Black-and-white ink line art. Seated full-body view, tense and sweating. Plain white background."),
    ("upset_nearly_fullbody.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Nearly full-body view, visibly upset with tears. Plain white background."),
    ("upset_nearly_fullbody-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Nearly full-body view, visibly upset with tears. Plain white background."),
    ("upset_nearly_fullbody-lineart.png",
     "JSEED. Black-and-white ink line art. Nearly full-body view, visibly upset with tears. Plain white background."),
    ("upset_portrait.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Portrait, visibly upset with tears. Plain white background."),
    ("upset_portrait-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Portrait, visibly upset with tears. Plain white background."),
    ("upset_portrait-lineart.png",
     "JSEED. Black-and-white ink line art. Portrait, visibly upset with tears. Plain white background."),
    ("upset_waistup_wide.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Waist-up view, visibly upset with tears. Plain white background."),
    ("upset_waistup_wide-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Waist-up view, visibly upset with tears. Plain white background."),
    ("upset_waistup_wide-lineart.png",
     "JSEED. Black-and-white ink line art. Waist-up view, visibly upset with tears. Plain white background."),
    ("warm_smile_nearly_fullbody.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Nearly full-body standing view with a warm smile. Plain white background."),
    ("warm_smile_nearly_fullbody-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Nearly full-body standing view with a warm smile. Plain white background."),
    ("warm_smile_nearly_fullbody-lineart.png",
     "JSEED. Black-and-white ink line art. Nearly full-body standing view with a warm smile. Plain white background."),
    ("warm_smile_portrait.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Portrait with a warm smile. Plain white background."),
    ("warm_smile_portrait-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Portrait with a warm smile. Plain white background."),
    ("warm_smile_portrait-lineart.png",
     "JSEED. Black-and-white ink line art. Portrait with a warm smile. Plain white background."),
    ("warm_smile_waistup_wide.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Waist-up view with a warm smile. Plain white background."),
    ("warm_smile_waistup_wide-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Waist-up view with a warm smile. Plain white background."),
    ("warm_smile_waistup_wide-lineart.png",
     "JSEED. Black-and-white ink line art. Waist-up view with a warm smile. Plain white background."),
    ("three-quarter-upperbody-blue.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Left-facing three-quarter upper-body view with a neutral expression. Plain blue background."),
    ("three-quarter-upperbody-cellshaded-blue.png",
     "JSEED. Anime cel shading with clean ink outlines. Left-facing three-quarter upper-body view with a neutral expression. Plain blue background."),
    ("three-quarter-upperbody-lineart-blue.png",
     "JSEED. Black-and-white ink line art. Left-facing three-quarter upper-body view with a neutral expression. Plain blue background."),
    ("warm_smile_nearly_fullbody-pink.jpg",
     "JSEED. Black ink line art with light watercolor-like coloring. Nearly full-body standing view with a warm smile. Plain pink background."),
    ("warm_smile_portrait-blue.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Portrait with a warm smile. Plain blue background."),
    ("warm_smile_portrait-pink.png",
     "JSEED. Black ink line art with light watercolor-like coloring. Portrait with a warm smile. Plain pink background."),
    ("warm_smile_waistup_wide-lineart-blue.png",
     "JSEED. Black-and-white ink line art. Waist-up view with a warm smile. Plain blue background."),
]


def nearest_bucket(width: int, height: int) -> tuple[int, int]:
    aspect = width / height
    return min(BUCKETS, key=lambda b: abs(math.log(aspect / (b[1] / b[0]))))


def edge_touches_figure(edge_pixels: np.ndarray) -> bool:
    """Raakt de figuur deze rand? Vergeleken met de mediaan van de rand ZELF,
    zodat een gradiëntachtergrond (blauw/roze masters) niet vals alarmeert."""
    median = np.median(edge_pixels, axis=0)
    distance = np.abs(edge_pixels.astype(np.float64) - median).sum(axis=1)
    return (distance > 120).mean() > 0.02


def split_pad(total: int, *, touches_start: bool, touches_end: bool, label: str) -> tuple[int, int]:
    """Verdeel padding over beide zijden; een afgesneden rand krijgt niets."""
    if total == 0:
        return 0, 0
    if touches_start and touches_end:
        raise ValueError(f"Padding nodig maar figuur raakt beide randen: {label}")
    if touches_start:
        return 0, total
    if touches_end:
        return total, 0
    return total // 2, total - total // 2


def pad_to_bucket_ratio(
    pixels: np.ndarray,
    bucket: tuple[int, int],
    file_name: str,
) -> tuple[np.ndarray, dict[str, int], dict[str, bool]]:
    """Randreplicatie-padding tot exact de bucketverhouding; voegt alleen
    gekopieerde randpixels toe, wijzigt geen enkel bronpixel."""
    height, width = pixels.shape[:2]
    bucket_height, bucket_width = bucket
    aspect = bucket_width / bucket_height
    if width / height < aspect:
        pad_x, pad_y = int(round(height * aspect)) - width, 0
    else:
        pad_x, pad_y = 0, int(round(width / aspect)) - height
    touched = {
        "left": edge_touches_figure(pixels[:, 0]),
        "right": edge_touches_figure(pixels[:, -1]),
        "top": edge_touches_figure(pixels[0]),
        "bottom": edge_touches_figure(pixels[-1]),
    }
    left, right = split_pad(
        pad_x, touches_start=touched["left"], touches_end=touched["right"], label=f"{file_name} X"
    )
    top, bottom = split_pad(
        pad_y, touches_start=touched["top"], touches_end=touched["bottom"], label=f"{file_name} Y"
    )
    padded = np.pad(pixels, ((top, bottom), (left, right), (0, 0)), mode="edge")
    pads = {"left": left, "right": right, "top": top, "bottom": bottom}
    return padded, pads, touched


def load_master(path: Path) -> Image.Image:
    with Image.open(path) as source:
        if "A" in source.getbands() and source.getchannel("A").getextrema() != (255, 255):
            raise ValueError(f"Master heeft niet-dekkende alpha: {path}")
        return source.convert("RGB")


def check_framing(source: Image.Image, final: Image.Image, pads: dict[str, int], scale: float) -> float:
    """Onafhankelijke controle: bevat het datasetbeeld de VOLLEDIGE master,
    uniform geschaald, op de verwachte plek? Beide kanten worden apart naar
    thumbnailformaat verkleind; een bucketflip of framingfout slaat hier
    keihard op aan, resample-verschillen niet."""
    left = pads["left"] * scale
    top = pads["top"] * scale
    box = (
        int(round(left)),
        int(round(top)),
        int(round(left + source.width * scale)),
        int(round(top + source.height * scale)),
    )
    content = final.crop(box)
    thumb_size = (FRAMING_THUMB, max(1, round(FRAMING_THUMB * source.height / source.width)))
    a = np.asarray(content.resize(thumb_size, Image.Resampling.LANCZOS), dtype=np.float64)
    b = np.asarray(source.resize(thumb_size, Image.Resampling.LANCZOS), dtype=np.float64)
    return float(np.abs(a - b).mean())


def build_records(staged_output: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, (file_name, prompt) in enumerate(MASTERS):
        source_path = MASTERS_DIR / file_name
        source = load_master(source_path)
        bucket = nearest_bucket(*source.size)
        bucket_height, bucket_width = bucket
        pixels = np.asarray(source)
        padded, pads, touched = pad_to_bucket_ratio(pixels, bucket, file_name)
        pad_fraction = max(
            (pads["left"] + pads["right"]) / source.width,
            (pads["top"] + pads["bottom"]) / source.height,
        )
        final = Image.fromarray(padded).resize((bucket_width, bucket_height), Image.Resampling.LANCZOS)
        scale = bucket_width / padded.shape[1]
        framing_diff = check_framing(source, final, pads, scale)
        output_name = f"{source_path.stem}.png"
        final.save(staged_output / output_name, compress_level=4)
        records.append(
            {
                "file_name": output_name,
                "prompt": prompt,
                "source": source_path.relative_to(REPO_ROOT).as_posix(),
                "source_size": list(source.size),
                "bucket": [bucket_height, bucket_width],
                "pads": pads,
                "pad_fraction": pad_fraction,
                "edges_touched": [edge for edge, hit in touched.items() if hit],
                "uniform_scale": scale,
                "framing_mean_diff": framing_diff,
            }
        )
        print(
            f"{index + 1:02d}/{len(MASTERS)} {file_name} -> {bucket_width}x{bucket_height} "
            f"pad={pad_fraction * 100:.1f}% framing-diff={framing_diff:.2f}",
            flush=True,
        )
    return records


def validate(records: list[dict[str, object]], dataset_dir: Path) -> None:
    if len(records) != len(MASTERS):
        raise ValueError("Niet elke master is exact één keer uitgegeven")
    names = [str(record["file_name"]) for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Outputnamen moeten uniek zijn")
    for record in records:
        name = record["file_name"]
        source_width, source_height = record["source_size"]
        bucket_height, bucket_width = record["bucket"]
        if (source_width > source_height) != (bucket_width > bucket_height) and bucket_width != bucket_height:
            raise ValueError(f"Oriëntatie geflipt voor {name}")
        aspect_error = abs(math.log((source_width / source_height) / (bucket_width / bucket_height)))
        if aspect_error > MAX_PAD_FRACTION:
            raise ValueError(f"Bucket past niet bij bronverhouding voor {name}: {aspect_error:.3f}")
        if record["pad_fraction"] > MAX_PAD_FRACTION:
            raise ValueError(f"Te veel padding voor {name}: {record['pad_fraction']:.3f}")
        if record["framing_mean_diff"] > FRAMING_MAX_MEAN_DIFF:
            raise ValueError(f"Framing wijkt af van master voor {name}: {record['framing_mean_diff']:.2f}")
        with Image.open(dataset_dir / str(name)) as image:
            if image.size != (bucket_width, bucket_height):
                raise ValueError(f"Verkeerde outputmaat voor {name}: {image.size}")


def validate_inventory() -> None:
    expected = {file_name for file_name, _prompt in MASTERS}
    if len(expected) != len(MASTERS):
        raise ValueError("Masterbestandsnamen moeten uniek zijn")
    actual = {
        path.name
        for path in MASTERS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"Masterinventaris klopt niet; missing={missing}, unexpected={unexpected}")


def write_metadata(records: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    {"file_name": record["file_name"], "prompt": record["prompt"]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def used_buckets(records: list[dict[str, object]]) -> list[tuple[int, int]]:
    return sorted({tuple(record["bucket"]) for record in records})


def write_manifest(records: list[dict[str, object]], path: Path) -> None:
    buckets = used_buckets(records)
    payload = {
        "kind": "jseed-subject-lora-dataset",
        "version": "v15",
        "image_count": len(records),
        "source_directory": MASTERS_DIR.relative_to(REPO_ROOT).as_posix(),
        "method": "edge-replicate-pad-to-nearest-bucket-then-lanczos",
        "max_pad_fraction": MAX_PAD_FRACTION,
        "bucket_counts": {
            f"{width}x{height}": sum(record["bucket"] == [height, width] for record in records)
            for height, width in buckets
        },
        "aspect_ratio_buckets_arg": ";".join(f"{height},{width}" for height, width in buckets),
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_contact_sheet(records: list[dict[str, object]], dataset_dir: Path, output_path: Path) -> None:
    columns = 6
    cell_width = 240
    image_height = 240
    label_height = 40
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
        sheet.paste(thumb, (cell_x + (cell_width - thumb.width) // 2, cell_y + (image_height - thumb.height) // 2))
        bucket_height, bucket_width = record["bucket"]
        label = f"{record['file_name']}\n{bucket_width}x{bucket_height}"
        draw.multiline_text((cell_x + 4, cell_y + image_height + 3), label, fill="white", font=font, spacing=2)
    sheet.save(output_path)


def save_side_by_side_sheets(records: list[dict[str, object]], dataset_dir: Path, review_dir: Path) -> None:
    """Per master: bron en datasetbeeld naast elkaar op gelijke hoogte —
    dít is de controle die de v13-framingfout direct zichtbaar had gemaakt."""
    pairs_per_sheet = 12
    columns = 3
    pair_height = 260
    label_height = 30
    font = ImageFont.load_default()
    for sheet_index in range(0, len(records), pairs_per_sheet):
        chunk = records[sheet_index : sheet_index + pairs_per_sheet]
        cells = []
        for record in chunk:
            master = load_master(REPO_ROOT / str(record["source"]))
            with Image.open(dataset_dir / str(record["file_name"])) as image:
                output = image.convert("RGB")
            master.thumbnail((10_000, pair_height), Image.Resampling.LANCZOS)
            output.thumbnail((10_000, pair_height), Image.Resampling.LANCZOS)
            cells.append((record["file_name"], master, output))
        cell_width = max(m.width + o.width + 18 for _, m, o in cells)
        rows = math.ceil(len(cells) / columns)
        sheet = Image.new("RGB", (columns * cell_width, rows * (pair_height + label_height)), (32, 32, 32))
        draw = ImageDraw.Draw(sheet)
        for index, (name, master, output) in enumerate(cells):
            x = index % columns * cell_width
            y = index // columns * (pair_height + label_height)
            sheet.paste(master, (x + 4, y))
            sheet.paste(output, (x + master.width + 12, y))
            draw.text((x + 4, y + pair_height + 4), f"{name}  (links master, rechts dataset)", fill="white", font=font)
        sheet.save(review_dir / f"side-by-side-{sheet_index // pairs_per_sheet + 1:02d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()

    output = args.output.resolve()
    review = args.review.resolve()
    if output.exists():
        raise FileExistsError(f"{output} bestaat al — een gewijzigde dataset krijgt een nieuw pad")
    if review.exists():
        raise FileExistsError(f"{review} bestaat al — een gewijzigde dataset krijgt een nieuw pad")

    validate_inventory()

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".jseed-dataset-v15-", dir=output.parent))
    staged_output = staging_root / "dataset"
    staged_review = staging_root / "review"
    staged_output.mkdir()
    staged_review.mkdir()

    records = build_records(staged_output)
    validate(records, staged_output)
    write_metadata(records, staged_output / "metadata.jsonl")
    write_manifest(records, staged_review / "manifest.json")
    save_contact_sheet(records, staged_output, staged_review / "contact-sheet.png")
    save_side_by_side_sheets(records, staged_output, staged_review)

    review.parent.mkdir(parents=True, exist_ok=True)
    staged_output.rename(output)
    staged_review.rename(review)
    staging_root.rmdir()

    buckets = used_buckets(records)
    print(f"{len(records)} beelden -> {output}")
    print(f"review -> {review}")
    bucket_arg = ";".join(f"{height},{width}" for height, width in buckets)
    print(f'gebruikte buckets ({len(buckets)}): --aspect_ratio_buckets "{bucket_arg}"')


if __name__ == "__main__":
    main()
