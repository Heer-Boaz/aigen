from __future__ import annotations

import math


FLUX_TOKEN_SIZE = 16


def fit_size_to_area(width: int, height: int, *, max_area: int, multiple_of: int) -> tuple[int, int]:
    scale = min(1.0, math.sqrt(max_area / (width * height)))
    return (
        max(multiple_of, int(width * scale) // multiple_of * multiple_of),
        max(multiple_of, int(height * scale) // multiple_of * multiple_of),
    )


def flux_token_count(width: int, height: int) -> int:
    return (width // FLUX_TOKEN_SIZE) * (height // FLUX_TOKEN_SIZE)


def canvas_for_token_budget(width: int, height: int, token_budget: int) -> dict[str, int]:
    aspect = width / height
    width_cells = max(1, int((token_budget * aspect) ** 0.5))
    height_cells = max(1, int(token_budget / width_cells))
    while width_cells * height_cells > token_budget:
        if width_cells / height_cells > aspect:
            width_cells -= 1
        else:
            height_cells -= 1
    return {
        "width": width_cells * FLUX_TOKEN_SIZE,
        "height": height_cells * FLUX_TOKEN_SIZE,
        "generated_tokens": width_cells * height_cells,
    }
