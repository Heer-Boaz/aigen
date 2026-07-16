#!/usr/bin/env python3
"""Build the Jillian subject-LoRA dataset v4.

Design, in service of a character-only LoRA whose drawing style stays
promptable at inference:
- Bare ``JSEED`` trigger (no class noun).
- Three drawing styles (canon ink+watercolor masters from v3, plus PixAI
  cel-shaded and lineart variants), each named in every caption, so style is
  NOT absorbed into the trigger and can be swapped (e.g. for pixel art) at
  inference.
- Derived square and landscape crops per master feed multi-aspect buckets,
  so the identity is not welded to one canvas shape.

Every training image is one explicit row in IMAGES with its full caption
spelled out literally — no caption derivation logic.

NB: train/ is sinds 2026-07-16 handgecureerd (Boaz verving alle beelden voor
betere knoopjes, ook de crops). Het script overschrijft daarom nooit bestaande
beelden: het leidt alleen ontbrekende crops af en ververst metadata.jsonl en
de review sheet. Wil je een crop bewust herafleiden, verwijder dan eerst het
bestand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

TARGET = Path("assets/lora/JSEED/train")
REVIEW = Path("assets/lora/JSEED/review")

# Crop kinds (geometry relative to the figure bounding box on the source):
#   None             -> master unchanged (portrait bucket)
#   "square-waist"   -> square, side = 0.50 x figure height, top-anchored
#   "square-head"    -> square, side = min(0.72 x figure height, image width), top-anchored
#   "landscape-head" -> 3:2 band, height = 0.30 x figure height, top-anchored
IMAGES = [
    # --- canon: Black ink lineart with light watercolor shading -----------------
    ("front-full.png", TARGET / "front-full.png", None,
     "JSEED. Black ink lineart with light watercolor shading. Full-body front view, standing upright with arms relaxed at her sides, centered on a plain white background."),
    ("front-full-square-upper.png", TARGET / "front-full.png", "square-waist",
     "JSEED. Black ink lineart with light watercolor shading. Upper-body front view from the waist up, neutral expression, centered on a plain white background."),
    ("front-full-landscape-head.png", TARGET / "front-full.png", "landscape-head",
     "JSEED. Black ink lineart with light watercolor shading. Close-up of the head and shoulders, front view, neutral expression, centered on a plain white background."),
    ("side-full.png", TARGET / "side-full.png", None,
     "JSEED. Black ink lineart with light watercolor shading. Full-body left-facing side profile, standing upright with arms relaxed at her sides, centered on a plain white background."),
    ("side-full-square-upper.png", TARGET / "side-full.png", "square-waist",
     "JSEED. Black ink lineart with light watercolor shading. Upper-body left-facing side profile from the waist up, neutral expression, centered on a plain white background."),
    ("side-full-landscape-head.png", TARGET / "side-full.png", "landscape-head",
     "JSEED. Black ink lineart with light watercolor shading. Close-up of the head and shoulders, left-facing side profile, neutral expression, centered on a plain white background."),
    ("back-full.png", TARGET / "back-full.png", None,
     "JSEED. Black ink lineart with light watercolor shading. Full-body rear view, standing upright with arms relaxed at her sides, centered on a plain white background."),
    ("back-full-square-upper.png", TARGET / "back-full.png", "square-waist",
     "JSEED. Black ink lineart with light watercolor shading. Upper-body rear view from the waist up, centered on a plain white background."),
    ("front-upperbody.png", TARGET / "front-upperbody.png", None,
     "JSEED. Black ink lineart with light watercolor shading. Upper-body front view from the waist up, neutral expression, centered on a plain white background."),
    ("front-upperbody-square.png", TARGET / "front-upperbody.png", "square-head",
     "JSEED. Black ink lineart with light watercolor shading. Close-up of the head and shoulders, front view, neutral expression, centered on a plain white background."),
    ("side-upperbody.png", TARGET / "side-upperbody.png", None,
     "JSEED. Black ink lineart with light watercolor shading. Upper-body left-facing side profile from the waist up, neutral expression, centered on a plain white background."),
    ("side-upperbody-square.png", TARGET / "side-upperbody.png", "square-head",
     "JSEED. Black ink lineart with light watercolor shading. Close-up of the head and shoulders, left-facing side profile, neutral expression, centered on a plain white background."),
    ("three-quarter-upperbody.png", TARGET / "three-quarter-upperbody.png", None,
     "JSEED. Black ink lineart with light watercolor shading. Left-facing three-quarter view from the mid-thigh up, neutral expression, centered on a plain white background."),
    ("three-quarter-upperbody-square.png", TARGET / "three-quarter-upperbody.png", "square-head",
     "JSEED. Black ink lineart with light watercolor shading. Close-up of the head and shoulders, left-facing three-quarter view, neutral expression, centered on a plain white background."),
    # --- cel-shaded: Flat cel-shaded anime coloring with bold clean outlines ----
    # NB: the front master smiles faintly; its caption and both crops say so.
    ("front-full-cellshaded.png", TARGET / "front-full-cellshaded.png", None,
     "JSEED. Flat cel-shaded anime coloring with bold clean outlines. Full-body front view, standing upright with arms relaxed at her sides, a faint soft smile, centered on a plain white background."),
    ("front-full-cellshaded-square-upper.png", TARGET / "front-full-cellshaded.png", "square-waist",
     "JSEED. Flat cel-shaded anime coloring with bold clean outlines. Upper-body front view from the waist up, a faint soft smile, centered on a plain white background."),
    ("front-full-cellshaded-landscape-head.png", TARGET / "front-full-cellshaded.png", "landscape-head",
     "JSEED. Flat cel-shaded anime coloring with bold clean outlines. Close-up of the head and shoulders, front view, a faint soft smile, centered on a plain white background."),
    ("side-full-cellshaded.png", TARGET / "side-full-cellshaded.png", None,
     "JSEED. Flat cel-shaded anime coloring with bold clean outlines. Full-body left-facing side profile, standing upright with arms relaxed at her sides, centered on a plain white background."),
    ("side-full-cellshaded-square-upper.png", TARGET / "side-full-cellshaded.png", "square-waist",
     "JSEED. Flat cel-shaded anime coloring with bold clean outlines. Upper-body left-facing side profile from the waist up, neutral expression, centered on a plain white background."),
    ("side-full-cellshaded-landscape-head.png", TARGET / "side-full-cellshaded.png", "landscape-head",
     "JSEED. Flat cel-shaded anime coloring with bold clean outlines. Close-up of the head and shoulders, left-facing side profile, neutral expression, centered on a plain white background."),
    ("back-full-cellshaded.png", TARGET / "back-full-cellshaded.png", None,
     "JSEED. Flat cel-shaded anime coloring with bold clean outlines. Full-body rear view, standing upright with arms relaxed at her sides, centered on a plain white background."),
    ("back-full-cellshaded-square-upper.png", TARGET / "back-full-cellshaded.png", "square-waist",
     "JSEED. Flat cel-shaded anime coloring with bold clean outlines. Upper-body rear view from the waist up, centered on a plain white background."),
    ("front-portrait-cellshaded.png", TARGET / "front-portrait-cellshaded.png", None,
     "JSEED. Flat cel-shaded anime coloring with bold clean outlines. Upper-body front view from the chest up, neutral expression, centered on a plain white background."),
    ("front-portrait-cellshaded-square.png", TARGET / "front-portrait-cellshaded.png", "square-head",
     "JSEED. Flat cel-shaded anime coloring with bold clean outlines. Close-up of the head and shoulders, front view, neutral expression, centered on a plain white background."),
    # --- lineart: Black-and-white ink line art, uncolored, with sketch hatching -
    ("front-full-lineart.png", TARGET / "front-full-lineart.png", None,
     "JSEED. Black-and-white ink line art, uncolored, with sketch hatching. Full-body front view, standing upright with arms relaxed at her sides, centered on a plain white background."),
    ("front-full-lineart-square-upper.png", TARGET / "front-full-lineart.png", "square-waist",
     "JSEED. Black-and-white ink line art, uncolored, with sketch hatching. Upper-body front view from the waist up, neutral expression, centered on a plain white background."),
    ("front-full-lineart-landscape-head.png", TARGET / "front-full-lineart.png", "landscape-head",
     "JSEED. Black-and-white ink line art, uncolored, with sketch hatching. Close-up of the head and shoulders, front view, neutral expression, centered on a plain white background."),
    ("side-full-lineart.png", TARGET / "side-full-lineart.png", None,
     "JSEED. Black-and-white ink line art, uncolored, with sketch hatching. Full-body left-facing side profile, standing upright with arms relaxed at her sides, centered on a plain white background."),
    ("side-full-lineart-square-upper.png", TARGET / "side-full-lineart.png", "square-waist",
     "JSEED. Black-and-white ink line art, uncolored, with sketch hatching. Upper-body left-facing side profile from the waist up, neutral expression, centered on a plain white background."),
    ("side-full-lineart-landscape-head.png", TARGET / "side-full-lineart.png", "landscape-head",
     "JSEED. Black-and-white ink line art, uncolored, with sketch hatching. Close-up of the head and shoulders, left-facing side profile, neutral expression, centered on a plain white background."),
    ("back-full-lineart.png", TARGET / "back-full-lineart.png", None,
     "JSEED. Black-and-white ink line art, uncolored, with sketch hatching. Full-body rear view, standing upright with arms relaxed at her sides, centered on a plain white background."),
    ("back-full-lineart-square-upper.png", TARGET / "back-full-lineart.png", "square-waist",
     "JSEED. Black-and-white ink line art, uncolored, with sketch hatching. Upper-body rear view from the waist up, centered on a plain white background."),
    ("font-portrait-lineart.png", TARGET / "font-portrait-lineart.png", None,
     "JSEED. Black-and-white ink line art, uncolored, with sketch hatching. Upper-body front view from the chest up, neutral expression, centered on a plain white background."),
    ("font-portrait-lineart-square.png", TARGET / "font-portrait-lineart.png", "square-head",
     "JSEED. Black-and-white ink line art, uncolored, with sketch hatching. Close-up of the head and shoulders, front view, neutral expression, centered on a plain white background."),
]


def figure_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    """Bounding box of the non-background figure on a near-white canvas."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    background = pixels[2:8, 2:8].reshape(-1, 3).mean(axis=0)
    distance = np.abs(pixels - background).sum(axis=2)
    mask = distance > 40
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    return columns[0], rows[0], columns[-1], rows[-1]


def crop_image(image: Image.Image, crop: str) -> Image.Image:
    left, top, right, bottom = figure_bbox(image)
    figure_height = bottom - top
    center_x = (left + right) / 2
    anchor_top = top - 0.04 * figure_height
    if crop == "square-waist":
        width = height = 0.50 * figure_height
    elif crop == "square-head":
        width = height = min(0.72 * figure_height, image.width)
    elif crop == "landscape-head":
        height = 0.30 * figure_height
        width = height * 1.5
    else:
        raise ValueError(f"Unknown crop kind: {crop}")
    width = min(width, image.width)
    height = min(height, image.height)
    x = int(round(min(max(center_x - width / 2, 0), image.width - width)))
    y = int(round(min(max(anchor_top, 0), image.height - height)))
    return image.crop((x, y, x + int(round(width)), y + int(round(height))))


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)

    for file_name, source, crop, _prompt in IMAGES:
        destination = TARGET / file_name
        if destination.exists():
            # train/ is handgecureerd: bestaande beelden nooit overschrijven,
            # alleen ontbrekende crops afleiden
            continue
        if crop is None:
            Image.open(source).convert("RGB").save(destination)
            continue
        crop_image(Image.open(source).convert("RGB"), crop).save(destination)

    with open(TARGET / "metadata.jsonl", "w", encoding="utf-8") as handle:
        for file_name, _source, _crop, prompt in IMAGES:
            handle.write(
                json.dumps({"file_name": file_name, "prompt": prompt}, ensure_ascii=False) + "\n"
            )

    # review sheet: fixed-height thumbnails in table order
    thumb_height = 320
    thumbs = []
    for file_name, _source, _crop, _prompt in IMAGES:
        thumb = Image.open(TARGET / file_name)
        scale = thumb_height / thumb.height
        thumbs.append(thumb.resize((int(thumb.width * scale), thumb_height)))
    columns = 6
    rows = (len(thumbs) + columns - 1) // columns
    row_width = max(
        sum(t.width for t in thumbs[r * columns : (r + 1) * columns]) + 10 * columns
        for r in range(rows)
    )
    sheet = Image.new("RGB", (row_width, rows * (thumb_height + 10)), "white")
    for index, thumb in enumerate(thumbs):
        row, column = divmod(index, columns)
        x = sum(t.width + 10 for t in thumbs[row * columns : row * columns + column])
        sheet.paste(thumb, (x, row * (thumb_height + 10)))
    sheet.save(REVIEW / "train-set.png")

    for file_name, _source, _crop, prompt in IMAGES:
        print(f"{file_name}: {prompt}")
    print(f"\n{len(IMAGES)} images -> {TARGET}")
    print(f"review sheet -> {REVIEW / 'train-set.png'}")


if __name__ == "__main__":
    main()
