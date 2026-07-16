"""Whiten border-connected near-white image backgrounds.

Usage:
  python tools/clean_backgrounds.py INPUT_DIR
  python tools/clean_backgrounds.py INPUT_DIR --out OUTPUT_DIR
  python tools/clean_backgrounds.py INPUT_DIR --inplace
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        color = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return np.rint(color * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return image[:, :, :3]


def border_connected_near_white(image: np.ndarray, threshold: float) -> np.ndarray:
    distance = np.linalg.norm(255.0 - image.astype(np.float32), axis=2)
    near_white = (distance < threshold).astype(np.uint8)
    _, labels = cv2.connectedComponents(near_white, connectivity=8)
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    border_labels = border_labels[border_labels != 0]
    return np.isin(labels, border_labels)


def process_image(
    input_path: Path,
    output_path: Path,
    *,
    white_threshold: float = 40.0,
) -> None:
    image = _read_rgb(input_path)
    image[border_connected_near_white(image, white_threshold)] = 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Cannot write image: {output_path}")


def _images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--white-threshold", type=float, default=40.0)
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.folder}")
    if args.inplace and args.out is not None:
        raise SystemExit("--inplace and --out are mutually exclusive")

    output_root = (
        args.folder
        if args.inplace
        else args.out or args.folder.with_name(f"{args.folder.name}_clean")
    )
    images = _images(args.folder)
    if not images:
        raise SystemExit(f"No images found in: {args.folder}")

    for input_path in tqdm(images):
        output_path = output_root / input_path.name
        if args.inplace:
            temporary_path = output_path.with_name(f".{output_path.name}.tmp.png")
            process_image(
                input_path,
                temporary_path,
                white_threshold=args.white_threshold,
            )
            temporary_path.replace(output_path)
        else:
            process_image(
                input_path,
                output_path,
                white_threshold=args.white_threshold,
            )

    print(f"{len(images)} images -> {output_root}")


if __name__ == "__main__":
    main()
