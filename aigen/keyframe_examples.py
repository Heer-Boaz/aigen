from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from aigen.image_assets import image_asset_json
from aigen.keyframe_pose import DEFAULT_DWPOSE_DET_MODEL, DEFAULT_DWPOSE_POSE_MODEL
from aigen.keyframe_segmentation import foreground_box_mask
from aigen.manifest_io import file_manifest, write_json


EXAMPLE_EXTRACTION_SCHEMA_VERSION = 1
DEFAULT_FOREGROUND_HEIGHT_RATIO = 0.84
DEFAULT_FOREGROUND_WIDTH_RATIO = 0.88
DEFAULT_BOTTOM_MARGIN_RATIO = 0.055
DEFAULT_BOUNDARY_RADIUS_PX = 18
DEFAULT_BOUNDARY_FEATHER_PX = 7
DEFAULT_BOUNDARY_FEATHER_SUPPORT_PX = DEFAULT_BOUNDARY_FEATHER_PX * 3


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
    foreground = _source_foreground(image)
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
    foreground_image = _clean_foreground_image(normalized, normalized_foreground)
    gray = _source_gray(foreground_image, normalized_foreground)
    canny_lineart = _source_canny_lineart(foreground_image, normalized_foreground)
    softedge = _source_softedge(foreground_image, normalized_foreground)
    filled_silhouette = _filled_silhouette(normalized_foreground)
    boundary = _boundary_mask(normalized_foreground)
    arm_hand_mask = _arm_hand_mask(normalized_foreground)

    assets = {
        "source": output_dir / f"{config.name}_source.png",
        "normalized_source": output_dir / f"{config.name}_normalized.png",
        "foreground_clean": output_dir / f"{config.name}_foreground_clean.png",
        "pose": output_dir / f"{config.name}_pose.png",
        "contour": output_dir / f"{config.name}_canny_lineart.png",
        "canny_lineart": output_dir / f"{config.name}_canny_lineart.png",
        "gray": output_dir / f"{config.name}_gray.png",
        "softedge": output_dir / f"{config.name}_softedge.png",
        "filled_silhouette": output_dir / f"{config.name}_filled_silhouette.png",
        "full_silhouette_mask": output_dir / f"{config.name}_full_silhouette_mask.png",
        "boundary_mask": output_dir / f"{config.name}_boundary.png",
        "arm_hand_mask": output_dir / f"{config.name}_arm_hand_mask.png",
        "metadata": output_dir / f"{config.name}_extraction.json",
    }
    with Image.open(source) as image:
        image.save(assets["source"])
    normalized.convert("RGB").save(assets["normalized_source"])
    foreground_image.save(assets["foreground_clean"])
    pose_image.save(assets["pose"])
    canny_lineart.save(assets["contour"])
    gray.save(assets["gray"])
    softedge.save(assets["softedge"])
    filled_silhouette.save(assets["filled_silhouette"])
    filled_silhouette.save(assets["full_silhouette_mask"])
    boundary.save(assets["boundary_mask"])
    arm_hand_mask.save(assets["arm_hand_mask"])

    payload = {
        "schema_version": EXAMPLE_EXTRACTION_SCHEMA_VERSION,
        "kind": "keyframe-example-extraction",
        "source": image_asset_json(source),
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
            "source": image_asset_json(assets["source"]),
            "normalized_source": image_asset_json(assets["normalized_source"]),
            "foreground_clean": image_asset_json(assets["foreground_clean"]),
            "pose": image_asset_json(assets["pose"]),
            "contour": image_asset_json(assets["contour"]),
            "canny_lineart": image_asset_json(assets["canny_lineart"]),
            "gray": image_asset_json(assets["gray"]),
            "softedge": image_asset_json(assets["softedge"]),
            "filled_silhouette": image_asset_json(assets["filled_silhouette"]),
            "full_silhouette_mask": image_asset_json(assets["full_silhouette_mask"]),
            "boundary_mask": image_asset_json(assets["boundary_mask"]),
            "arm_hand_mask": image_asset_json(assets["arm_hand_mask"]),
        },
    }
    write_json(assets["metadata"], payload)
    payload["assets"]["metadata"] = file_manifest(assets["metadata"])
    return payload


def _load_rgba(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGBA")
    except OSError as error:
        raise KeyframeExampleError(f"Cannot read example image {path.as_posix()}: {error}") from error


def _source_foreground(image: Image.Image) -> np.ndarray:
    data = np.asarray(image, dtype=np.uint8)
    alpha = data[..., 3]
    if alpha.min() != alpha.max():
        return _largest_component(ndimage.binary_fill_holes(alpha > 0))
    return foreground_box_mask(data[..., :3])


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


def _clean_foreground_image(image: Image.Image, foreground: np.ndarray) -> Image.Image:
    data = np.asarray(image.convert("RGB"), dtype=np.uint8)
    clean = np.zeros_like(data)
    clean[foreground] = data[foreground]
    return Image.fromarray(clean, mode="RGB")


def _source_gray(foreground_image: Image.Image, foreground: np.ndarray) -> Image.Image:
    gray = np.asarray(foreground_image.convert("L"), dtype=np.uint8)
    gray = np.where(foreground, gray, 0).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def _source_canny_lineart(foreground_image: Image.Image, foreground: np.ndarray) -> Image.Image:
    luma = np.asarray(foreground_image.convert("L"), dtype=np.float32) / 255.0
    gradient_x = ndimage.sobel(luma, axis=1)
    gradient_y = ndimage.sobel(luma, axis=0)
    gradient = np.hypot(gradient_x, gradient_y)
    silhouette_edge = foreground ^ ndimage.binary_erosion(
        foreground,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    )
    edge = (gradient > 0.08) | ndimage.binary_dilation(
        silhouette_edge,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    )
    edge &= ndimage.binary_dilation(foreground, structure=np.ones((3, 3), dtype=bool), iterations=1)
    return Image.fromarray((edge.astype(np.uint8) * 255), mode="L").convert("RGB")


def _source_softedge(foreground_image: Image.Image, foreground: np.ndarray) -> Image.Image:
    lineart = np.asarray(_source_canny_lineart(foreground_image, foreground).convert("L"), dtype=np.float32) / 255.0
    silhouette = foreground.astype(np.float32)
    boundary = np.abs(ndimage.gaussian_filter(silhouette, sigma=1.2) - ndimage.gaussian_filter(silhouette, sigma=3.0))
    soft = np.maximum(ndimage.gaussian_filter(lineart, sigma=0.7), boundary)
    soft = soft / max(float(soft.max()), 1e-6)
    return Image.fromarray((soft * 255.0).astype(np.uint8), mode="L").convert("RGB")


def _filled_silhouette(foreground: np.ndarray) -> Image.Image:
    return Image.fromarray((foreground.astype(np.uint8) * 255), mode="L")


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
    support = ndimage.binary_dilation(
        band,
        structure=np.ones((3, 3), dtype=bool),
        iterations=DEFAULT_BOUNDARY_FEATHER_SUPPORT_PX,
    )
    soft = np.where(support, soft, 0.0)
    soft = soft / max(float(soft.max()), 1e-6)
    return Image.fromarray((soft * 255.0).astype(np.uint8), mode="L")


def _arm_hand_mask(foreground: np.ndarray) -> Image.Image:
    x1, y1, x2, y2 = _bbox(foreground)
    ys, xs = np.indices(foreground.shape)
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) * 0.5
    upper_body = (ys >= y1 + height * 0.08) & (ys <= y1 + height * 0.68)
    lateral_extremity = np.abs(xs - center_x) >= width * 0.27
    mask = foreground & upper_body & lateral_extremity
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=max(4, round(width * 0.035)))
    soft = ndimage.gaussian_filter(mask.astype(np.float32), sigma=5.0)
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
