from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch

from aigen.pix2pix.errors import Pix2PixError


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise Pix2PixError("CUDA was requested but no CUDA device is available")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    raise Pix2PixError(f"unsupported device: {device_name}")


def validate_precision(device: torch.device, precision: str) -> None:
    if precision not in {"fp32", "bf16"}:
        raise Pix2PixError("precision must be fp32 or bf16")
    if precision == "bf16" and device.type != "cuda":
        raise Pix2PixError("bf16 requires a CUDA device")


def validate_model_precision(
    parameter_dtype: torch.dtype,
    compute_precision: str,
) -> None:
    if parameter_dtype == torch.float32:
        return
    if parameter_dtype == torch.bfloat16 and compute_precision == "bf16":
        return
    if parameter_dtype == torch.bfloat16:
        raise Pix2PixError("a bf16 pix2pix model requires bf16 compute precision")
    raise Pix2PixError(
        f"unsupported pix2pix parameter dtype: {parameter_dtype}"
    )


def autocast_context(
    device: torch.device,
    precision: str,
) -> ContextManager[None]:
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()
