#!/usr/bin/env python
"""Framing acceptance gate for generated character views.

Fails (exit 1) when the character silhouette touches a canvas edge, i.e. the
figure is cropped. Deterministic and GPU-free, so "done" is machine-checkable
instead of an eyeball judgement.

Method:
  1. background colour = median RGB of the four 8x8 corners (the background is
     not pure white, so a fixed white threshold misreads it);
  2. content mask = pixels whose euclidean RGB distance to the background > 30;
  3. bounding box of that mask -> margin to each edge in pixels;
  4. margin <= --min-margin on any edge => cropped on that edge.

Usage:
  python tools/check_character_framing.py IMAGE [IMAGE ...] [--min-margin N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


CONTENT_DISTANCE = 30.0
CORNER = 8


def content_margins(path: Path) -> tuple[int, int, int, int, list[int]]:
    """Return (top, bottom, left, right) margins in px plus the detected bg RGB."""
    array = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    height, width = array.shape[:2]
    corners = np.concatenate(
        [
            array[:CORNER, :CORNER],
            array[:CORNER, -CORNER:],
            array[-CORNER:, :CORNER],
            array[-CORNER:, -CORNER:],
        ]
    ).reshape(-1, 3)
    background = np.median(corners, axis=0)
    content = np.sqrt(((array - background) ** 2).sum(axis=2)) > CONTENT_DISTANCE
    ys, xs = np.nonzero(content)
    if len(xs) == 0:
        raise ValueError(f"{path}: no character content found (whole image matches background)")
    top = int(ys.min())
    bottom = int(height - 1 - ys.max())
    left = int(xs.min())
    right = int(width - 1 - xs.max())
    return top, bottom, left, right, [int(c) for c in background]


def check(path: Path, min_margin: int) -> bool:
    """Print the framing verdict for one image; return True if fully in frame."""
    try:
        top, bottom, left, right, background = content_margins(path)
    except (ValueError, FileNotFoundError, OSError) as error:
        print(f"{path}: ERROR {error}")
        return False
    edges = zip(("top", "bottom", "left", "right"), (top, bottom, left, right))
    cropped = [name for name, margin in edges if margin < min_margin]
    verdict = "CROPPED @ " + "+".join(cropped) if cropped else "OK"
    print(
        f"{path}: margins t/b/l/r={top}/{bottom}/{left}/{right}px "
        f"bg={background} -> {verdict}"
    )
    return not cropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path, help="Generated view PNG(s) to check")
    parser.add_argument("--min-margin", type=int, default=4, help="Minimum edge margin in px (default 4)")
    args = parser.parse_args()
    all_ok = all([check(image, args.min_margin) for image in args.images])
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
