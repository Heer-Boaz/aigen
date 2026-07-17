#!/usr/bin/env python3
"""Materialize the reviewed JSEED v10 dataset without foreground extraction.

The review manifest owns the approved source, caption, padding color and FLUX.2
bucket for every image. Sources are only padded with their recorded canvas color
and uniformly resized; no source pixels are classified, masked or composited.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


MANIFEST = Path("assets/lora/JSEED/review/dataset-v10-manifest.json")
OUTPUT = Path("assets/lora/JSEED/dataset-v10")


def edge_touches_subject(edge_pixels: np.ndarray, canvas_color: tuple[int, int, int]) -> bool:
    distance = np.abs(edge_pixels.astype(np.float64) - canvas_color).sum(axis=1)
    return bool((distance > 120).mean() > 0.02)


def pad_to_aspect(
    image: Image.Image,
    target_aspect: float,
    canvas_color: tuple[int, int, int],
) -> Image.Image:
    width, height = image.size
    if width / height < target_aspect:
        padded_width, padded_height = round(height * target_aspect), height
    else:
        padded_width, padded_height = width, round(width / target_aspect)

    pad_x = padded_width - width
    pad_y = padded_height - height
    pixels = np.asarray(image)

    left = pad_x // 2
    if pad_x and edge_touches_subject(pixels[:, 0], canvas_color):
        left = 0
    elif pad_x and edge_touches_subject(pixels[:, -1], canvas_color):
        left = pad_x

    top = pad_y // 2
    if pad_y and edge_touches_subject(pixels[0], canvas_color):
        top = 0
    elif pad_y and edge_touches_subject(pixels[-1], canvas_color):
        top = pad_y

    padded = Image.new("RGB", (padded_width, padded_height), canvas_color)
    padded.paste(image, (left, top))
    return padded


def main() -> None:
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if OUTPUT.exists():
        if any(OUTPUT.iterdir()):
            raise FileExistsError(f"Dataset output is not empty: {OUTPUT}")
    else:
        OUTPUT.mkdir(parents=True)

    metadata = []
    for record in records:
        source_path = Path(record["source"])
        with Image.open(source_path) as source:
            image = source.convert("RGB")
        if list(image.size) != record["source_size"]:
            raise ValueError(f"Source size changed since review: {source_path}")

        target_height, target_width = record["bucket"]
        canvas_color = tuple(record["pad_color"])
        padded = pad_to_aspect(image, target_width / target_height, canvas_color)
        output = padded.resize((target_width, target_height), Image.Resampling.LANCZOS)
        output.save(OUTPUT / record["file_name"], compress_level=4)
        metadata.append({"file_name": record["file_name"], "prompt": record["prompt"]})

    with (OUTPUT / "metadata.jsonl").open("w", encoding="utf-8") as stream:
        for record in metadata:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{len(metadata)} reviewed images -> {OUTPUT}")


if __name__ == "__main__":
    main()
