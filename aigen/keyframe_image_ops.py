from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


_PIXEL_DIFF_CONNECTIVITY = np.ones((3, 3), dtype=bool)


def mask_overlay(base: Image.Image, mask: Image.Image) -> Image.Image:
    overlay = base.convert("RGBA")
    red = Image.new("RGBA", base.size, (255, 0, 0, 120))
    overlay.alpha_composite(Image.composite(red, Image.new("RGBA", base.size, (0, 0, 0, 0)), mask))
    return overlay.convert("RGB")


def paste_refined_crop(
    base: Image.Image,
    refined_crop: Image.Image,
    feather_mask: Image.Image,
    crop_box: tuple[int, int, int, int],
) -> Image.Image:
    output = base.copy()
    output.paste(
        refined_crop.resize((crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])),
        crop_box,
        feather_mask.crop(crop_box),
    )
    return output


def outside_mask_change(base: Image.Image, refined: Image.Image, feather_mask: Image.Image) -> dict[str, Any]:
    base_array = np.asarray(base.convert("RGB"), dtype=np.int16)
    refined_array = np.asarray(refined.convert("RGB"), dtype=np.int16)
    outside = np.asarray(feather_mask, dtype=np.uint8) == 0
    delta = np.abs(base_array - refined_array).max(axis=2)
    changed = delta[outside] > 1
    changed_pixels = int(changed.sum())
    total = int(outside.sum())
    max_delta = int(delta[outside].max()) if total else 0
    return {
        "outside_feather_changed_pixels": changed_pixels,
        "outside_feather_changed_ratio": float(changed_pixels / max(total, 1)),
        "outside_feather_max_delta": max_delta,
        "hard_rejects": {
            "outside_feather_changed": bool(changed_pixels > 0 or max_delta > 1),
        },
    }


def exact_outside_mask_diff(base: Image.Image, refined: Image.Image, repaint_mask: Image.Image) -> dict[str, Any]:
    base_array = np.asarray(base.convert("RGB"), dtype=np.uint8)
    refined_array = np.asarray(refined.convert("RGB"), dtype=np.uint8)
    repaint = np.asarray(repaint_mask.convert("L"), dtype=np.uint8) > 0
    outside = ~repaint
    changed = np.any(base_array != refined_array, axis=2) & outside
    changed_pixels = int(changed.sum())
    regions: list[dict[str, Any]] = []
    if changed_pixels:
        labels, _ = ndimage.label(changed, structure=_PIXEL_DIFF_CONNECTIVITY)
        component_sizes = np.bincount(labels.ravel())
        for label, component_slice in enumerate(ndimage.find_objects(labels), start=1):
            if component_slice is None:
                continue
            y_slice, x_slice = component_slice
            regions.append(
                {
                    "box": [x_slice.start, y_slice.start, x_slice.stop, y_slice.stop],
                    "changed_pixels": int(component_sizes[label]),
                }
            )
        regions.sort(key=lambda region: region["changed_pixels"], reverse=True)
    outside_pixels = int(outside.sum())
    return {
        "passed": changed_pixels == 0,
        "outside_mask_pixels": outside_pixels,
        "outside_mask_changed_pixels": changed_pixels,
        "outside_mask_changed_ratio": float(changed_pixels / max(outside_pixels, 1)),
        "outside_change_regions": regions,
    }


def save_contact_sheet(
    outputs: list[dict[str, Any]],
    output_path: Path,
    *,
    thumb_width: int,
    label_x: int = 6,
    max_label_chars: int | None = None,
    max_columns: int = 8,
) -> None:
    with Image.open(outputs[0]["path"]) as first_image:
        thumb_height = max(1, int(thumb_width * first_image.height / first_image.width))
    label_height = 32
    columns = min(max_columns, len(outputs))
    rows = (len(outputs) + columns - 1) // columns
    cell_height = thumb_height + label_height
    sheet = Image.new("RGB", (thumb_width * columns, cell_height * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, output in enumerate(outputs):
        x = index % columns * thumb_width
        y = index // columns * cell_height
        label = output["name"]
        if max_label_chars is not None:
            label = label[:max_label_chars]
        with Image.open(output["path"]) as image:
            sheet.paste(image.convert("RGB").resize((thumb_width, thumb_height), Image.Resampling.LANCZOS), (x, y + label_height))
        draw.text((x + label_x, y + 8), label, fill="black")
    sheet.save(output_path)


def bbox_mask_array(size: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    width, height = size
    left, top, right, bottom = bbox
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def expanded_aligned_box(mask: np.ndarray, padding: int, width: int, height: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot build crop box from an empty mask")
    left = max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)
    right = min(width, int(xs.max()) + 1 + padding)
    bottom = min(height, int(ys.max()) + 1 + padding)
    right = min(width, left + _align_up(right - left, 16))
    bottom = min(height, top + _align_up(bottom - top, 16))
    left = max(0, right - _align_up(right - left, 16))
    top = max(0, bottom - _align_up(bottom - top, 16))
    return left, top, right, bottom


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
