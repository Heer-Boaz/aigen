from __future__ import annotations

from aigen.image_dimensions import pixel_area_canvas_size


FLUX2_DEV_RECOMMENDED_PIXEL_AREA = 1024 * 1024
FLUX2_DEV_DIMENSION_ALIGNMENT = 16


def flux2_dev_recommended_canvas_size(
    aspect_ratio: tuple[int, int],
) -> tuple[int, int]:
    return pixel_area_canvas_size(
        aspect_ratio,
        target_pixels=FLUX2_DEV_RECOMMENDED_PIXEL_AREA,
        alignment=FLUX2_DEV_DIMENSION_ALIGNMENT,
    )
