from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from aigen.manifest_io import atomic_write_json
from aigen.pix2pix.artifacts import load_generator_bundle, prepare_empty_output_dir
from aigen.pix2pix.config import ModelConfig
from aigen.pix2pix.dataset import PAIR_SPLITS, audit_dataset
from aigen.pix2pix.device import (
    resolve_device,
    validate_model_precision,
    validate_precision,
)
from aigen.pix2pix.errors import Pix2PixError
from aigen.pix2pix.evaluation import evaluate_generator
from aigen.progress import StatusReporter


def evaluate_model(
    model_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    device_name: str,
    precision: str,
    progress: StatusReporter,
) -> dict[str, object]:
    if split not in PAIR_SPLITS:
        raise Pix2PixError(f"unsupported dataset split: {split}")
    if batch_size < 1:
        raise Pix2PixError("batch_size must be positive")
    if num_workers < 0:
        raise Pix2PixError("num_workers must be non-negative")
    progress.phase("audit evaluation dataset")
    dataset = audit_dataset(dataset_dir)
    pairs = dataset.split(split)
    if not pairs:
        raise Pix2PixError(f"dataset has no {split} pairs")
    device = resolve_device(device_name)
    validate_precision(device, precision)
    output_dir = output_dir.resolve()
    generator, metadata = load_generator_bundle(model_dir, device=device)
    validate_model_precision(next(generator.parameters()).dtype, precision)
    model_config = ModelConfig.from_json(metadata["model"])
    if model_config.image_size != dataset.image_size:
        raise Pix2PixError("model image size does not match the evaluation dataset")
    prepare_empty_output_dir(output_dir)
    progress.begin(
        math.ceil(len(pairs) / batch_size),
        f"evaluate pix2pix {split}",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    result = evaluate_generator(
        generator,
        pairs,
        image_size=model_config.image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        precision=precision,
        predictions_dir=output_dir / "predictions",
        preview_path=output_dir / "comparison.png",
        on_batch_completed=lambda completed: progress.step(
            f"evaluate pix2pix {completed}/{len(pairs)}"
        ),
    )
    report = {
        "status": "ok",
        "model": model_dir.resolve().as_posix(),
        "model_step": metadata["step"],
        "dataset": dataset.to_json(),
        "split": split,
        "metrics": result.to_json(),
        "output": output_dir.as_posix(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_vram_mb": (
            round(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
            if device.type == "cuda"
            else 0
        ),
    }
    atomic_write_json(output_dir / "evaluation.json", report)
    return report
