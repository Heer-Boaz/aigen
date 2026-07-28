from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from aigen.pix2pix.dataset import PairedImage
from aigen.pix2pix.device import autocast_context
from aigen.pix2pix.image_io import save_comparison_grid, save_tensor_png
from aigen.pix2pix.training_data import create_evaluation_loader


@dataclass(frozen=True)
class EvaluationResult:
    pair_count: int
    l1: float
    mse: float
    psnr_db: float | None
    exact_match: bool

    def to_json(self) -> dict[str, object]:
        return {
            "pair_count": self.pair_count,
            "l1": self.l1,
            "mse": self.mse,
            "psnr_db": self.psnr_db,
            "exact_match": self.exact_match,
        }


def evaluate_generator(
    generator: nn.Module,
    pairs: tuple[PairedImage, ...],
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    precision: str,
    predictions_dir: Path | None = None,
    preview_path: Path | None = None,
    on_batch_completed: Callable[[int], None] | None = None,
) -> EvaluationResult:
    loader = create_evaluation_loader(
        pairs,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    was_training = generator.training
    generator.eval()
    absolute_error = 0.0
    squared_error = 0.0
    value_count = 0
    completed_pairs = 0
    preview_rows: list[tuple[Tensor, Tensor, Tensor]] = []
    with torch.inference_mode(), autocast_context(device, precision):
        for batch in loader:
            source = _batch_tensor(batch, "source").to(device, non_blocking=True)
            target = _batch_tensor(batch, "target").to(device, non_blocking=True)
            generated = generator(source)
            generated_unit = generated.float().add(1.0).mul_(0.5).clamp_(0.0, 1.0)
            target_unit = target.float().add(1.0).mul_(0.5)
            difference = generated_unit - target_unit
            absolute_error += difference.abs().sum().item()
            squared_error += difference.square().sum().item()
            value_count += difference.numel()
            ids = batch["id"]
            assert isinstance(ids, list)
            for item_index, pair_id in enumerate(ids):
                if predictions_dir is not None:
                    save_tensor_png(
                        generated[item_index],
                        predictions_dir / f"{pair_id}.png",
                    )
                if len(preview_rows) < 4:
                    preview_rows.append(
                        (
                            source[item_index].cpu(),
                            target[item_index].cpu(),
                            generated[item_index].cpu(),
                        )
                    )
            completed_pairs += len(ids)
            if on_batch_completed is not None:
                on_batch_completed(completed_pairs)
    if was_training:
        generator.train()
    l1 = absolute_error / value_count
    mse = squared_error / value_count
    if preview_path is not None:
        save_comparison_grid(preview_rows, preview_path)
    return EvaluationResult(
        pair_count=len(pairs),
        l1=l1,
        mse=mse,
        psnr_db=None if mse == 0.0 else 10.0 * math.log10(1.0 / mse),
        exact_match=mse == 0.0,
    )


def _batch_tensor(batch: dict[str, Tensor | list[str]], key: str) -> Tensor:
    value = batch[key]
    assert isinstance(value, Tensor)
    return value
