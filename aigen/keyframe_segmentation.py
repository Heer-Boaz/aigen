from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


DEFAULT_MODELS_ROOT = Path(__file__).resolve().parent / "models"
DEFAULT_SAM_CHECKPOINT = (
    DEFAULT_MODELS_ROOT / "segmentation/ybelkada/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
)


class KeyframeSegmentationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SamSegmentationConfig:
    checkpoint: Path = DEFAULT_SAM_CHECKPOINT
    model_type: str = "vit_b"
    device: str = "cuda"
    prompt_threshold: float = 28.0


class SamForegroundSegmenter:
    def __init__(self, config: SamSegmentationConfig):
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as error:
            raise KeyframeSegmentationError("SAM foreground segmentation requires the segment-anything package.") from error

        if not config.checkpoint.is_file():
            raise KeyframeSegmentationError(f"Missing SAM checkpoint: {config.checkpoint.as_posix()}")

        model = sam_model_registry[config.model_type](checkpoint=config.checkpoint.as_posix())
        model.to(device=config.device)
        self._predictor = SamPredictor(model)
        self._threshold = config.prompt_threshold

    def segment(self, image_path: Path) -> np.ndarray:
        image = _load_rgb(image_path)
        box = _foreground_box(image, self._threshold)
        return self.segment_image_box(image, box)

    def segment_image_box(self, image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        self._predictor.set_image(image)
        masks, scores, _ = self._predictor.predict(
            box=np.asarray(box, dtype=np.float32),
            multimask_output=True,
        )
        mask = masks[int(np.asarray(scores).argmax())].astype(bool)
        if not mask.any():
            raise KeyframeSegmentationError(f"SAM returned an empty mask for box {box}")
        return mask


def foreground_box_mask(image: np.ndarray, threshold: float = 28.0) -> np.ndarray:
    foreground = _background_distance_mask(image, threshold)
    foreground = ndimage.binary_closing(foreground, structure=np.ones((3, 3), dtype=bool), iterations=2)
    foreground = ndimage.binary_fill_holes(foreground)
    return _largest_component(foreground)


def _foreground_box(image: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    mask = foreground_box_mask(image, threshold)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise KeyframeSegmentationError("Cannot prompt SAM: image contains no foreground subject")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _background_distance_mask(image: np.ndarray, threshold: float) -> np.ndarray:
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0).astype(np.float32)
    background = np.median(border, axis=0)
    distance = np.sqrt(((image.astype(np.float32) - background) ** 2).sum(axis=2))
    return distance > threshold


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
