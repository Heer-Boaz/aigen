from __future__ import annotations

import time
from pathlib import Path

import torch

from aigen.pix2pix.artifacts import load_generator_bundle
from aigen.pix2pix.config import ModelConfig
from aigen.pix2pix.device import (
    autocast_context,
    resolve_device,
    validate_model_precision,
    validate_precision,
)
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.image_io import load_rgb_tensor, save_tensor_png


def run_inference(
    model_dir: Path,
    input_path: Path,
    output_path: Path,
    *,
    device_name: str,
    precision: str,
    output_size: int | None,
) -> dict[str, object]:
    device = resolve_device(device_name)
    validate_precision(device, precision)
    output_path = output_path.resolve()
    if output_path.suffix.casefold() != ".png":
        raise Pix2PixError(f"pix2pix output must be a PNG file: {output_path.as_posix()}")
    if output_path.exists():
        raise Pix2PixError(f"pix2pix output already exists: {output_path.as_posix()}")
    if output_size is not None and output_size < 1:
        raise Pix2PixError("output_size must be positive")
    started = time.monotonic()
    generator, metadata = load_generator_bundle(model_dir, device=device)
    validate_model_precision(next(generator.parameters()).dtype, precision)
    config = ModelConfig.from_json(metadata["model"])
    source = load_rgb_tensor(input_path.resolve(), image_size=config.image_size)
    source = source.unsqueeze(0).to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), autocast_context(device, precision):
        generated = generator(source)
    save_tensor_png(generated[0], output_path, output_size=output_size)
    peak_vram = (
        round(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        if device.type == "cuda"
        else 0
    )
    return {
        "status": "ok",
        "model": model_dir.resolve().as_posix(),
        "model_step": metadata["step"],
        "input": input_path.resolve().as_posix(),
        "output": output_path.as_posix(),
        "model_image_size": config.image_size,
        "output_size": output_size or config.image_size,
        "device": str(device),
        "precision": precision,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_vram_mb": peak_vram,
    }
