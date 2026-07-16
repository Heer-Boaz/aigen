#!/usr/bin/env python3
"""Build the JSEED training dataset v12 — non-destructief, met achtergrondvarianten.

Elke master wordt UITSLUITEND gepad (met de randkleur van zijn eigen canvas)
tot de dichtstbijzijnde officiële FLUX.2-bucketverhouding en daarna met
Lanczos geschaald naar exact die bucket. Geen segmentatie, geen alpha, geen
compositing: elke haarstreng en elke witte lineart-vulling blijft behouden.
De enige pixelbewerking is uniform schalen — hetzelfde dat de trainer anders
zelf zou doen.

Eén expliciete rij per master met de volledige caption als literal.
Vervangt de afgekeurde segmentatieroute uit
scripts/prepare_jillian_subject_lora_dataset_v4.py (dataset v9, zie
assets/lora/JSEED/TRAINING_HANDOFF.md).
"""
from __future__ import annotations

import json
import math

import scipy.ndimage
from pathlib import Path

import numpy as np
from PIL import Image

MASTERS_DIR = Path("assets/lora/JSEED/masters")
OUTPUT = Path("assets/lora/JSEED/dataset-v12")
REVIEW = Path("assets/lora/JSEED/review")

# Achtergrondvarianten: naam-in-caption, RGB.
BACKGROUNDS = (
    ("white", (255, 255, 255)),
    ("warm ivory", (244, 239, 231)),
    ("cool light gray", (224, 230, 236)),
    ("pale blue", (210, 224, 238)),
    ("pale sage", (218, 230, 218)),
    ("pale rose", (238, 220, 226)),
    ("medium gray", (163, 169, 176)),
    ("dark slate", (50, 58, 70)),
)

# Officiële ~1MP FLUX.2-buckets als (hoogte, breedte).
BUCKETS = (
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
)

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
     "JSEED. Black ink line art with light watercolor-like coloring. Waist-up view with a determined expression. Plain white background."),
    ("determined_waistup_wide-cellshaded.png",
     "JSEED. Anime cel shading with clean ink outlines. Waist-up view with a determined expression. Plain white background."),
    ("determined_waistup_wide-lineart.png",
     "JSEED. Black-and-white ink line art. Waist-up view with a determined expression. Plain white background."),
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
]


def border_color(image: Image.Image) -> tuple[int, int, int]:
    """Mediaan van de randpixels: robuust tegen een figuur die de rand raakt."""
    pixels = np.asarray(image, dtype=np.float64)
    edges = np.concatenate([
        pixels[:8].reshape(-1, 3),
        pixels[-8:].reshape(-1, 3),
        pixels[:, :8].reshape(-1, 3),
        pixels[:, -8:].reshape(-1, 3),
    ])
    return tuple(int(round(v)) for v in np.median(edges, axis=0))


def edge_touches_figure(edge_pixels: np.ndarray, color: tuple[int, int, int]) -> bool:
    distance = np.abs(edge_pixels.astype(np.float64) - color).sum(axis=1)
    return (distance > 120).mean() > 0.02


def nearest_bucket(width: int, height: int) -> tuple[int, int]:
    aspect = width / height
    return min(BUCKETS, key=lambda b: abs(math.log(aspect / (b[1] / b[0]))))


def pad_to_aspect(image: Image.Image, aspect: float, color: tuple[int, int, int]) -> Image.Image:
    """Padden tot exact de gevraagde verhouding; voegt alleen randpixels toe.

    Randen waar de figuur is afgesneden (bijv. de onderkant van een portret)
    krijgen geen padding — anders ontstaat daar een zwevende band."""
    width, height = image.size
    pixels = np.asarray(image)
    if width / height < aspect:
        new_width, new_height = int(round(height * aspect)), height
    else:
        new_width, new_height = width, int(round(width / aspect))
    pad_x = new_width - width
    pad_y = new_height - height
    left = pad_x // 2
    if pad_x and edge_touches_figure(pixels[:, 0], color):
        left = 0
    elif pad_x and edge_touches_figure(pixels[:, -1], color):
        left = pad_x
    top = pad_y // 2
    if pad_y and edge_touches_figure(pixels[0], color):
        top = 0
    elif pad_y and edge_touches_figure(pixels[-1], color):
        top = pad_y
    canvas = Image.new("RGB", (new_width, new_height), color)
    canvas.paste(image, (left, top))
    return canvas


LIGHT_BACKGROUNDS = ("white", "warm ivory", "cool light gray", "pale blue", "pale sage", "pale rose")


def background_region(pixels: np.ndarray, white: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Achtergrond = licht-en-kleurloos gebied dat vanaf de canvasrand bereikbaar is.

    Doorlaatbaar zijn wit én lichte grijze slagschaduwen (licht + lage
    saturatie), zodat de fill door schaduwen heen ook ingesloten zakken
    (tussen benen, onder armen) bereikt. Inktlijnen en gekleurde pixels zijn
    barrières: wit bínnen het onderwerp (blouse, lineart-vulling, highlights)
    blijft onbereikbaar en dus onaangetast."""
    passable = (pixels.min(axis=2) >= 175) & (pixels.max(axis=2) - pixels.min(axis=2) <= 30)
    labels, _ = scipy.ndimage.label(passable)
    border_labels = np.unique(np.concatenate([
        labels[0], labels[-1], labels[:, 0], labels[:, -1],
    ]))
    border_labels = border_labels[border_labels != 0]
    region = np.isin(labels, border_labels)
    band = scipy.ndimage.binary_dilation(region, iterations=2) & ~region
    return region, band


def apply_background(
    padded: Image.Image,
    region: np.ndarray,
    band: np.ndarray,
    white: tuple[int, int, int],
    color: tuple[int, int, int],
) -> Image.Image:
    """Multiplicatieve overdracht P' = P * (kleur/wit) op achtergrond + randband.

    Wit wordt exact de nieuwe kleur, een schaduwgrijs wordt dezelfde schaduw
    op de nieuwe kleur, en anti-alias-randpixels mengen mee — geen halo's."""
    pixels = np.asarray(padded, dtype=np.float64)
    factor = np.asarray(color, dtype=np.float64) / np.asarray(white, dtype=np.float64)
    output = pixels.copy()
    zone = region | band
    output[zone] = np.clip(pixels[zone] * factor, 0, 255)
    return Image.fromarray(output.astype(np.uint8))


def tint_paper(padded: Image.Image, white: tuple[int, int, int], color: tuple[int, int, int]) -> Image.Image:
    """Lineart: het hele vel krijgt de papiertint (multiply), binnenwit incluis."""
    pixels = np.asarray(padded, dtype=np.float64)
    factor = np.asarray(color, dtype=np.float64) / np.asarray(white, dtype=np.float64)
    return Image.fromarray(np.clip(pixels * factor, 0, 255).astype(np.uint8))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} bestaat al — een gewijzigde dataset krijgt een nieuw pad "
            "(en een nieuwe cache en trainingsoutput)."
        )
    OUTPUT.mkdir(parents=True)
    REVIEW.mkdir(parents=True, exist_ok=True)

    manifest = []
    records = []
    used_buckets = []
    for file_name, prompt in MASTERS:
        if not prompt.endswith("Plain white background."):
            raise ValueError(f"Caption van {file_name} eindigt niet op de achtergrondzin")
        source = Image.open(MASTERS_DIR / file_name).convert("RGB")
        white = border_color(source)
        bucket_height, bucket_width = nearest_bucket(*source.size)
        padded = pad_to_aspect(source, bucket_width / bucket_height, white)
        if (bucket_height, bucket_width) not in used_buckets:
            used_buckets.append((bucket_height, bucket_width))
        is_lineart = "-lineart" in file_name or file_name == "font-portrait-lineart.png"
        if not is_lineart:
            region, band = background_region(np.asarray(padded), white)
        stem = Path(file_name).stem
        for label, color in BACKGROUNDS:
            if is_lineart and label not in LIGHT_BACKGROUNDS:
                continue
            if label == "white":
                variant = padded
            elif is_lineart:
                variant = tint_paper(padded, white, color)
            else:
                variant = apply_background(padded, region, band, white, color)
            final = variant.resize((bucket_width, bucket_height), Image.LANCZOS)
            variant_name = f"{stem}__{label.replace(' ', '-')}.png"
            final.save(OUTPUT / variant_name, compress_level=4)
            variant_prompt = prompt.removesuffix("Plain white background.") + f"Plain {label} background."
            records.append({"file_name": variant_name, "prompt": variant_prompt})
            manifest.append({
                "file_name": variant_name,
                "source": (MASTERS_DIR / file_name).as_posix(),
                "background": label,
                "bucket": [bucket_height, bucket_width],
                "prompt": variant_prompt,
            })
        print(f"{stem}: klaar", flush=True)

    with open(OUTPUT / "metadata.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(REVIEW / "dataset-v12-manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    for label, _color in BACKGROUNDS:
        suffix = f"__{label.replace(' ', '-')}.png"
        thumbs = []
        for record in records:
            if record["file_name"].endswith(suffix):
                thumb = Image.open(OUTPUT / record["file_name"])
                scale = 220 / thumb.height
                thumbs.append(thumb.resize((int(thumb.width * scale), 220)))
        columns = 8
        rows = (len(thumbs) + columns - 1) // columns
        row_width = max(
            sum(t.width for t in thumbs[r * columns : (r + 1) * columns]) + 10 * columns
            for r in range(rows)
        )
        sheet = Image.new("RGB", (row_width, rows * 230), "white")
        for index, thumb in enumerate(thumbs):
            row, column = divmod(index, columns)
            x = sum(t.width + 10 for t in thumbs[row * columns : row * columns + column])
            sheet.paste(thumb, (x, row * 230))
        sheet.save(REVIEW / f"dataset-v12-contact-{label.replace(' ', '-')}.png")

    used_buckets.sort()
    buckets_arg = ";".join(f"{h},{w}" for h, w in used_buckets)
    print(f"{len(records)} beelden -> {OUTPUT}")
    print(f'gebruikte buckets ({len(used_buckets)}): --aspect_ratio_buckets "{buckets_arg}"')
    print(f"manifest + contactbladen -> {REVIEW}/dataset-v12-*")


if __name__ == "__main__":
    main()
