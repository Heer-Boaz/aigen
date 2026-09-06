"""Actual mask inference with a recorded point annotation, separate from identity extraction."""
from pathlib import Path

import numpy as np
from PIL import Image

from aigen.image_assets import image_asset_json
from aigen.image_io import open_image
from aigen.keyframe_segmentation import (
    AnimeForegroundSegmenter, AnimeSegmentationConfig, Sam2RegionSegmenter,
    Sam2SegmentationConfig, SamForegroundSegmenter, SamSegmentationConfig,
)
from aigen.manifest_io import atomic_write_json


def run_segmentation(config, job, directory, progress):
    source = Path(config["segmentation"]["input"])
    point = tuple(config["segmentation"]["point"])
    atomic_write_json(directory / "inputs.json", {"image": image_asset_json(source), "point": point, "label": 1})
    with open_image(source) as original, original.convert("RGB") as rgb:
        pixels = np.array(rgb)
    progress.phase(f"load {job['model']}")
    kind = job["model"]
    constructor, settings = {
        "sam": (SamForegroundSegmenter, SamSegmentationConfig()),
        "sam2": (Sam2RegionSegmenter, Sam2SegmentationConfig()),
        "anime-segmentation": (AnimeForegroundSegmenter, AnimeSegmentationConfig()),
    }[kind]
    model = constructor(settings)
    try:
        progress.phase(f"infer {kind} mask")
        mask = model.segment_image(pixels) if kind == "anime-segmentation" else model.segment_image_prompt(
            pixels, points=[point], labels=[1])
    finally:
        model.close()
    assert mask.shape == pixels.shape[:2]
    assert np.isfinite(mask).all() and mask.max() > 0
    assert mask[point[1], point[0]] > 0.5
    assert mask[0, 0] < 0.5
    target = directory / "mask.png"
    Image.fromarray(np.clip(mask.astype(np.float32) * 255, 0, 255).astype(np.uint8)).save(target)
    return {"model": kind, "mask": image_asset_json(target), "point_selected": True, "background_corner_excluded": True}
