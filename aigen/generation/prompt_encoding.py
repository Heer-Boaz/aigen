from __future__ import annotations

import gc
from collections.abc import Iterable
from typing import Any, TypeVar


T = TypeVar("T")


def ordered_unique(values: Iterable[T]) -> list[T]:
    return list(dict.fromkeys(values))


def tensor_to_cpu(tensor: Any | None) -> Any | None:
    if tensor is None:
        return None
    return tensor.to("cpu")


def tensor_to_device(tensor: Any | None, *, device: str) -> Any | None:
    if tensor is None:
        return None
    return tensor.to(device)


def release_prompt_encoder_memory(torch: Any) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
