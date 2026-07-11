from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


DEFAULT_CANNY_LOW_THRESHOLD = 100
DEFAULT_CANNY_HIGH_THRESHOLD = 200


class CannyControlError(RuntimeError):
    pass


@dataclass(frozen=True)
class CannyControl:
    image: Image.Image
    metadata: dict[str, int | str]


def render_canny_control(
    image: Image.Image,
    *,
    source_label: str,
    low_threshold: int = DEFAULT_CANNY_LOW_THRESHOLD,
    high_threshold: int = DEFAULT_CANNY_HIGH_THRESHOLD,
) -> CannyControl:
    try:
        import cv2
    except ImportError as error:
        raise CannyControlError("Canny control rendering requires OpenCV") from error

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(
        grayscale,
        threshold1=low_threshold,
        threshold2=high_threshold,
        L2gradient=True,
    )
    edge_pixels = int(np.count_nonzero(edges))
    if edge_pixels == 0:
        raise CannyControlError(f"Canny found no edges in {source_label}")
    return CannyControl(
        image=Image.fromarray(edges, mode="L").convert("RGB"),
        metadata={
            "edge_pixels": edge_pixels,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        },
    )
