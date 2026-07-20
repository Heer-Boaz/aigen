from __future__ import annotations

from math import gcd, log
from typing import Sequence


def parse_aspect_ratio(value: str) -> tuple[int, int]:
    width_text, separator, height_text = value.partition(":")
    if not separator:
        raise ValueError("aspect ratio must use W:H")
    width = int(width_text)
    height = int(height_text)
    return normalized_aspect_ratio(width, height)


def normalized_aspect_ratio(width: int, height: int) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ValueError("aspect ratio values must be positive integers")
    divisor = gcd(width, height)
    return width // divisor, height // divisor


def closest_aspect_match(
    aspect_ratio: tuple[int, int],
    candidates: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    target = log(aspect_ratio[0] / aspect_ratio[1])
    return min(
        candidates,
        key=lambda candidate: abs(log(candidate[0] / candidate[1]) - target),
    )
