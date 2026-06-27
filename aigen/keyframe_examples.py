from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from aigen.keyframe_pose import DEFAULT_DWPOSE_DET_MODEL, DEFAULT_DWPOSE_POSE_MODEL


EXAMPLE_EXTRACTION_SCHEMA_VERSION = 1
DEFAULT_FOREGROUND_HEIGHT_RATIO = 0.84
DEFAULT_FOREGROUND_WIDTH_RATIO = 0.88
DEFAULT_BOTTOM_MARGIN_RATIO = 0.055
DEFAULT_FOREGROUND_THRESHOLD = 34.0
DEFAULT_BOUNDARY_RADIUS_PX = 18
DEFAULT_BOUNDARY_FEATHER_PX = 7


class KeyframeExampleError(RuntimeError):
    pass


@dataclass(frozen=True)
class KeyframeExampleExtractionConfig:
    source: Path
    output_dir: Path
    name: str
    width: int
    height: int
    mirror_x: bool
    pose_device: str = "cpu"
    det_model: Path = DEFAULT_DWPOSE_DET_MODEL
    pose_model: Path = DEFAULT_DWPOSE_POSE_MODEL


def extract_keyframe_example(config: KeyframeExampleExtractionConfig) -> dict[str, Any]:
    source = config.source.resolve()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image = _load_rgba(source)
    foreground = _foreground_mask(image)
    box = _bbox(foreground)
    normalized, normalized_foreground, transform = _normalize_example(
        image,
        foreground,
        box,
        width=config.width,
        height=config.height,
        mirror_x=config.mirror_x,
    )
    pose_image, pose_metadata = _dwpose_control_image(
        normalized,
        device=config.pose_device,
        det_model=config.det_model,
        pose_model=config.pose_model,
    )
    contour = _silhouette_contour(normalized_foreground)
    boundary = _boundary_mask(normalized_foreground)

    assets = {
        "source": output_dir / f"{config.name}_source.png",
        "normalized_source": output_dir / f"{config.name}_normalized.png",
        "pose": output_dir / f"{config.name}_pose.png",
        "contour": output_dir / f"{config.name}_contour.png",
        "boundary_mask": output_dir / f"{config.name}_boundary.png",
        "metadata": output_dir / f"{config.name}_extraction.json",
    }
    Image.open(source).save(assets["source"])
    normalized.convert("RGB").save(assets["normalized_source"])
    pose_image.save(assets["pose"])
    contour.save(assets["contour"])
    boundary.save(assets["boundary_mask"])

    payload = {
        "schema_version": EXAMPLE_EXTRACTION_SCHEMA_VERSION,
        "kind": "keyframe-example-extraction",
        "source": _asset_json(source),
        "canvas": {"width": config.width, "height": config.height},
        "mirror_x": config.mirror_x,
        "foreground": {
            "source_bbox": list(box),
            "normalized_bbox": list(_bbox(normalized_foreground)),
            "coverage": float(normalized_foreground.sum() / normalized_foreground.size),
        },
        "transform": transform,
        "pose": pose_metadata,
        "assets": {
            "source": _asset_json(assets["source"]),
            "normalized_source": _asset_json(assets["normalized_source"]),
            "pose": _asset_json(assets["pose"]),
            "contour": _asset_json(assets["contour"]),
            "boundary_mask": _asset_json(assets["boundary_mask"]),
        },
    }
    _write_json(assets["metadata"], payload)
    payload["assets"]["metadata"] = _file_json(assets["metadata"])
    return payload


def _load_rgba(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGBA")
    except OSError as error:
        raise KeyframeExampleError(f"Cannot read example image {path.as_posix()}: {error}") from error


def _foreground_mask(image: Image.Image) -> np.ndarray:
    data = np.asarray(image, dtype=np.uint8)
    alpha = data[:, :, 3]
    if int((alpha < 250).sum()) > 0:
        return _largest_component(alpha > 8)

    rgb = data[:, :, :3].astype(np.float32)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.sqrt(((rgb - background) ** 2).sum(axis=2))
    foreground = distance > DEFAULT_FOREGROUND_THRESHOLD
    foreground = ndimage.binary_closing(foreground, structure=np.ones((3, 3), dtype=bool), iterations=1)
    foreground = ndimage.binary_fill_holes(foreground)
    return _largest_component(foreground)


def _normalize_example(
    image: Image.Image,
    foreground: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    mirror_x: bool,
) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    data = np.asarray(image.convert("RGB"), dtype=np.uint8)
    border = np.concatenate((data[0], data[-1], data[:, 0], data[:, -1]), axis=0)
    background = tuple(int(value) for value in np.median(border, axis=0))
    crop = image.crop(box)
    crop_mask = Image.fromarray((foreground[box[1] : box[3], box[0] : box[2]].astype(np.uint8) * 255), mode="L")

    if mirror_x:
        crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        crop_mask = crop_mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    crop_width, crop_height = crop.size
    scale = min(
        (height * DEFAULT_FOREGROUND_HEIGHT_RATIO) / crop_height,
        (width * DEFAULT_FOREGROUND_WIDTH_RATIO) / crop_width,
    )
    resized_size = (max(1, round(crop_width * scale)), max(1, round(crop_height * scale)))
    resized = crop.resize(resized_size, Image.Resampling.NEAREST)
    resized_mask = crop_mask.resize(resized_size, Image.Resampling.NEAREST)
    x = round((width - resized_size[0]) / 2)
    y = height - round(height * DEFAULT_BOTTOM_MARGIN_RATIO) - resized_size[1]

    canvas = Image.new("RGBA", (width, height), (*background, 255))
    canvas.alpha_composite(resized, (x, y))
    mask = Image.new("L", (width, height), 0)
    mask.paste(resized_mask, (x, y))
    normalized_foreground = np.asarray(mask, dtype=np.uint8) > 0
    normalized_foreground = _largest_component(ndimage.binary_fill_holes(normalized_foreground))
    return canvas, normalized_foreground, {
        "source_crop": list(box),
        "canvas_paste": [x, y, x + resized_size[0], y + resized_size[1]],
        "scale": scale,
    }


def _dwpose_control_image(
    image: Image.Image,
    *,
    device: str,
    det_model: Path,
    pose_model: Path,
) -> tuple[Image.Image, dict[str, Any]]:
    try:
        from controlnet_dwpose import DWposeDetector
        from controlnet_dwpose.util import draw_pose
    except ImportError as error:
        raise KeyframeExampleError("Example extraction requires the controlnet-dwpose package.") from error

    detector = DWposeDetector(det_model.as_posix(), pose_model.as_posix(), device=device)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    pose = detector(rgb)
    if len(pose["bodies"]["subset"]) == 0:
        raise KeyframeExampleError("DWPose found no body in the example image")

    control = draw_pose(pose, image.height, image.width)
    if control.shape[0] == 3:
        control = np.transpose(control, (1, 2, 0))
    control_image = Image.fromarray(control.astype(np.uint8), mode="RGB")
    body_scores = np.asarray(pose["bodies"]["score"], dtype=np.float32)
    return control_image, {
        "body_count": int(len(pose["bodies"]["subset"])),
        "visible_body_keypoints": int((body_scores[0, :18] > 0.30).sum()),
        "mean_body_score": float(body_scores[0, :18].mean()),
    }


def _silhouette_contour(foreground: np.ndarray) -> Image.Image:
    edge = foreground ^ ndimage.binary_erosion(foreground, structure=np.ones((3, 3), dtype=bool), iterations=1)
    edge = ndimage.binary_dilation(edge, structure=np.ones((3, 3), dtype=bool), iterations=1)
    return Image.fromarray((edge.astype(np.uint8) * 255), mode="L")


def _boundary_mask(foreground: np.ndarray) -> Image.Image:
    outer = ndimage.binary_dilation(
        foreground,
        structure=np.ones((3, 3), dtype=bool),
        iterations=DEFAULT_BOUNDARY_RADIUS_PX,
    )
    inner = ndimage.binary_erosion(
        foreground,
        structure=np.ones((3, 3), dtype=bool),
        iterations=DEFAULT_BOUNDARY_RADIUS_PX,
    )
    band = outer & ~inner
    soft = ndimage.gaussian_filter(band.astype(np.float32), sigma=DEFAULT_BOUNDARY_FEATHER_PX)
    soft = soft / max(float(soft.max()), 1e-6)
    return Image.fromarray((soft * 255.0).astype(np.uint8), mode="L")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise KeyframeExampleError("Example image contains no foreground subject")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def _asset_json(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        mode = image.mode
        width, height = image.size
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mode": mode,
        "width": width,
        "height": height,
    }


def _file_json(path: Path) -> dict[str, str]:
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
