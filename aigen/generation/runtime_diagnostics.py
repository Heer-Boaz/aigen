from __future__ import annotations

from time import perf_counter
from typing import Any


def synchronized_time(torch_module: Any) -> float:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    return perf_counter()


def elapsed_ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 3)


def cuda_memory_stats(torch_module: Any, device: str) -> dict[str, int]:
    if not torch_module.cuda.is_available():
        return {
            "max_allocated_mb": 0,
            "max_reserved_mb": 0,
            "free_after_run_mb": 0,
            "device_total_mb": 0,
        }

    free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
    return {
        "max_allocated_mb": round(torch_module.cuda.max_memory_allocated(device) / 2**20),
        "max_reserved_mb": round(torch_module.cuda.max_memory_reserved(device) / 2**20),
        "free_after_run_mb": round(free_bytes / 2**20),
        "device_total_mb": round(total_bytes / 2**20),
    }


def parameter_locations(module: Any) -> list[dict[str, Any]]:
    locations = []
    for name, parameter in module.named_parameters():
        locations.append(
            {
                "name": name,
                "class": type(parameter).__name__,
                "device": str(parameter.device),
                "dtype": str(parameter.dtype),
                "shape": list(parameter.shape),
            }
        )
        if len(locations) == 10:
            break
    return locations
