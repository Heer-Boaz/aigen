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


def module_device_report(module: Any) -> dict[str, Any]:
    parameter_tensors_by_device: dict[str, int] = {}
    parameter_elements_by_device: dict[str, int] = {}
    parameter_tensors_by_dtype: dict[str, int] = {}
    sample_parameters = []
    for name, parameter in module.named_parameters():
        device = str(parameter.device)
        dtype = str(parameter.dtype)
        parameter_tensors_by_device[device] = parameter_tensors_by_device.get(device, 0) + 1
        parameter_elements_by_device[device] = parameter_elements_by_device.get(device, 0) + parameter.numel()
        parameter_tensors_by_dtype[dtype] = parameter_tensors_by_dtype.get(dtype, 0) + 1
        if len(sample_parameters) < 12:
            sample_parameters.append(
                {
                    "name": name,
                    "class": type(parameter).__name__,
                    "device": device,
                    "dtype": dtype,
                    "shape": list(parameter.shape),
                }
            )
    return {
        "class": type(module).__qualname__,
        "hf_device_map": _jsonable(getattr(module, "hf_device_map", None)),
        "parameter_tensors_by_device": parameter_tensors_by_device,
        "parameter_elements_by_device": parameter_elements_by_device,
        "parameter_tensors_by_dtype": parameter_tensors_by_dtype,
        "sample_parameters": sample_parameters,
        "accelerate_hooks": accelerate_hook_report(module),
    }


def accelerate_hook_report(module: Any) -> list[dict[str, Any]]:
    hooks = []
    for name, submodule in module.named_modules():
        hook = getattr(submodule, "_hf_hook", None)
        if hook:
            hooks.append(
                {
                    "module": name,
                    "hook_class": type(hook).__qualname__,
                    "execution_device": str(getattr(hook, "execution_device", "")),
                    "offload": _jsonable(getattr(hook, "offload", "")),
                }
            )
        if len(hooks) == 32:
            break
    return hooks


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
