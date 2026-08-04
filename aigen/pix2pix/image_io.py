from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from aigen.pix2pix.errors import Pix2PixError


def load_rgb_tensor(path: Path, *, image_size: int) -> Tensor:
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB":
                raise Pix2PixError(
                    f"image must be RGB: {path.as_posix()} has mode {image.mode}"
                )
            if image.size != (image_size, image_size):
                raise Pix2PixError(
                    f"image must be {image_size}x{image_size}: "
                    f"{path.as_posix()} is {image.width}x{image.height}"
                )
            pixels = np.array(image, dtype=np.uint8, copy=True)
    except OSError as error:
        raise Pix2PixError(f"cannot decode image {path.as_posix()}: {error}") from error
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).to(dtype=torch.float32)
    return tensor.mul_(2.0 / 255.0).sub_(1.0)


def tensor_to_rgb_image(tensor: Tensor) -> Image.Image:
    pixels = (
        tensor.detach()
        .to(dtype=torch.float32, device="cpu")
        .add(1.0)
        .clamp_(0.0, 2.0)
        .mul_(127.5)
        .round_()
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    return Image.fromarray(pixels, mode="RGB")


def save_tensor_png(
    tensor: Tensor,
    path: Path,
    *,
    output_size: int | None = None,
) -> None:
    if path.suffix.casefold() != ".png":
        raise Pix2PixError(f"pix2pix output must be a PNG file: {path.as_posix()}")
    if path.exists():
        raise Pix2PixError(f"pix2pix output already exists: {path.as_posix()}")
    image = tensor_to_rgb_image(tensor)
    if output_size is not None:
        if output_size < 1:
            raise Pix2PixError("output_size must be positive")
        image = image.resize((output_size, output_size), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


def save_comparison_grid(
    rows: list[tuple[Tensor, Tensor, Tensor]],
    path: Path,
) -> None:
    if not rows:
        raise Pix2PixError("cannot write an empty comparison grid")
    if path.exists():
        raise Pix2PixError(f"comparison image already exists: {path.as_posix()}")
    panel_size = max(
        tensor.shape[-1]
        for row in rows
        for tensor in row
    )
    grid = Image.new("RGB", (panel_size * 3, panel_size * len(rows)))
    for row_index, (source, target, generated) in enumerate(rows):
        y = row_index * panel_size
        for column, tensor in enumerate((source, target, generated)):
            image = tensor_to_rgb_image(tensor)
            if image.size != (panel_size, panel_size):
                image = image.resize(
                    (panel_size, panel_size),
                    Image.Resampling.NEAREST,
                )
            grid.paste(image, (panel_size * column, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path, format="PNG", optimize=False)
