from __future__ import annotations

from typing import Any


def resolve_torch_dtype(torch_module: Any, dtype: str, *, auto_value: Any) -> Any:
    if dtype == "auto":
        return auto_value
    if dtype == "bfloat16":
        return torch_module.bfloat16
    if dtype == "float16":
        return torch_module.float16
    if dtype == "float32":
        return torch_module.float32
    raise ValueError(f"Unsupported torch dtype: {dtype}")
