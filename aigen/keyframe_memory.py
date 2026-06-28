from __future__ import annotations

import os
import subprocess
from threading import Event, Thread
from typing import Any

from aigen.flux_geometry import canvas_for_token_budget, fit_size_to_area, flux_token_count


NVIDIA_SMI_PREFLIGHT_LIMIT_MB = 1800
NVIDIA_SMI_SAMPLE_SECONDS = 0.25
VRAM_ESTIMATE_BASELINE_FRAMEBUFFER_MB = NVIDIA_SMI_PREFLIGHT_LIMIT_MB
VRAM_ESTIMATE_SAFETY_MARGIN_MB = 256
VRAM_ESTIMATE_BASE_PEAK_MB = 15200
VRAM_ESTIMATE_GENERATED_TOKEN_NUMERATOR = 22
VRAM_ESTIMATE_REFERENCE_TOKEN_NUMERATOR = 5
VRAM_ESTIMATE_TOKEN_DENOMINATOR = 100
VRAM_ESTIMATE_TRUE_CFG_MB = 250
VRAM_ESTIMATE_HIGH_GENERATED_TOKEN_THRESHOLD = 2100


class KeyframeMemoryError(RuntimeError):
    pass


class NvidiaSmiMemorySampler:
    def __init__(self, preflight: dict[str, Any]) -> None:
        self.preflight = preflight
        self.peak_used_mb = preflight["nvidia_smi_preflight_used_mb"]
        self.device_total_mb = preflight["nvidia_smi_device_total_mb"]
        self._stop = Event()
        self._thread = Thread(target=self._sample_loop, daemon=True)

    def start(self) -> None:
        if self.device_total_mb:
            self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self._thread.is_alive():
            self._stop.set()
            self._thread.join()
        return {
            **self.preflight,
            "nvidia_smi_peak_used_mb": self.peak_used_mb,
        }

    def _sample_loop(self) -> None:
        while not self._stop.wait(NVIDIA_SMI_SAMPLE_SECONDS):
            snapshot = nvidia_smi_memory_snapshot()
            self.peak_used_mb = max(self.peak_used_mb, snapshot["nvidia_smi_used_mb"])


def planned_token_metadata(
    *,
    identity_width: int,
    identity_height: int,
    canvas_width: int,
    canvas_height: int,
    reference_max_area: int,
    max_sequence_length: int,
) -> dict[str, int]:
    reference_width, reference_height = fit_size_to_area(
        identity_width,
        identity_height,
        max_area=reference_max_area,
        multiple_of=16,
    )
    generated_tokens = flux_token_count(canvas_width, canvas_height)
    reference_tokens = flux_token_count(reference_width, reference_height)
    text_tokens = max_sequence_length
    return {
        "reference_width": reference_width,
        "reference_height": reference_height,
        "reference_tokens": reference_tokens,
        "generated_tokens": generated_tokens,
        "text_tokens": text_tokens,
        "total_tokens": generated_tokens + reference_tokens + text_tokens,
    }


def keyframe_vram_plan(
    *,
    canvas_width: int,
    canvas_height: int,
    true_cfg_scale: float,
    token_metadata: dict[str, int],
) -> dict[str, Any]:
    estimated_clean_peak_mb = _estimated_clean_peak_mb(
        generated_tokens=token_metadata["generated_tokens"],
        reference_tokens=token_metadata["reference_tokens"],
        true_cfg_scale=true_cfg_scale,
    )
    return {
        "method": "nunchaku-kontext-controlnet-local",
        "baseline_framebuffer_mb": VRAM_ESTIMATE_BASELINE_FRAMEBUFFER_MB,
        "safety_margin_mb": VRAM_ESTIMATE_SAFETY_MARGIN_MB,
        "estimated_clean_peak_mb": estimated_clean_peak_mb,
        "true_cfg_enabled": true_cfg_scale > 1.0,
        "true_cfg_extra_mb": VRAM_ESTIMATE_TRUE_CFG_MB if true_cfg_scale > 1.0 else 0,
        "high_generated_token_extra_mb": max(
            0,
            token_metadata["generated_tokens"] - VRAM_ESTIMATE_HIGH_GENERATED_TOKEN_THRESHOLD,
        ),
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "generated_tokens": token_metadata["generated_tokens"],
        "reference_tokens": token_metadata["reference_tokens"],
        "text_tokens": token_metadata["text_tokens"],
        "total_tokens": token_metadata["total_tokens"],
    }


def nvidia_smi_preflight() -> dict[str, int]:
    return nvidia_smi_preflight_limit(NVIDIA_SMI_PREFLIGHT_LIMIT_MB)


def nvidia_smi_keyframe_preflight(vram_plan: dict[str, Any]) -> dict[str, int | dict[str, int]]:
    if not cuda_available():
        return {
            "nvidia_smi_preflight_used_mb": 0,
            "nvidia_smi_device_total_mb": 0,
            "nvidia_smi_preflight_utilization_gpu": 0,
            "vram_estimated_required_mb": 0,
            "vram_estimated_headroom_mb": 0,
            "vram_clean_available_mb": 0,
            "vram_max_output_canvas": {
                "width": vram_plan["canvas_width"],
                "height": vram_plan["canvas_height"],
                "generated_tokens": vram_plan["generated_tokens"],
            },
        }

    snapshot = nvidia_smi_memory_snapshot()
    extra_framebuffer_mb = max(0, snapshot["nvidia_smi_used_mb"] - vram_plan["baseline_framebuffer_mb"])
    required_mb = vram_plan["estimated_clean_peak_mb"] + extra_framebuffer_mb + vram_plan["safety_margin_mb"]
    headroom_mb = snapshot["nvidia_smi_device_total_mb"] - required_mb
    clean_available_mb = snapshot["nvidia_smi_device_total_mb"] - extra_framebuffer_mb - vram_plan["safety_margin_mb"]
    max_generated_tokens = _max_generated_tokens_for_clean_peak(vram_plan, clean_available_mb)
    if headroom_mb < 0:
        raise KeyframeMemoryError(
            "Estimated VRAM requirement exceeds available framebuffer: "
            f"need about {required_mb} MB including margin, "
            f"GPU has {snapshot['nvidia_smi_device_total_mb']} MB, "
            f"currently used {snapshot['nvidia_smi_used_mb']} MB. "
            "Close GPU consumers or lower output/reference tokens."
        )
    return {
        "nvidia_smi_preflight_used_mb": snapshot["nvidia_smi_used_mb"],
        "nvidia_smi_device_total_mb": snapshot["nvidia_smi_device_total_mb"],
        "nvidia_smi_preflight_utilization_gpu": snapshot["nvidia_smi_utilization_gpu"],
        "vram_estimated_required_mb": required_mb,
        "vram_estimated_headroom_mb": headroom_mb,
        "vram_clean_available_mb": clean_available_mb,
        "vram_max_output_canvas": canvas_for_token_budget(
            vram_plan["canvas_width"],
            vram_plan["canvas_height"],
            max_generated_tokens,
        ),
    }


def nvidia_smi_preflight_limit(limit_mb: int) -> dict[str, int]:
    if not cuda_available():
        return {
            "nvidia_smi_preflight_used_mb": 0,
            "nvidia_smi_device_total_mb": 0,
            "nvidia_smi_preflight_utilization_gpu": 0,
        }
    snapshot = nvidia_smi_memory_snapshot()
    limit_mb = int(os.environ.get("AIGEN_NVIDIA_SMI_PREFLIGHT_LIMIT_MB", limit_mb))
    if snapshot["nvidia_smi_used_mb"] > limit_mb:
        raise KeyframeMemoryError(
            f"{snapshot['nvidia_smi_used_mb']} MB used before model load; "
            f"limit is {limit_mb} MB. Close other GPU processes first."
        )
    return {
        "nvidia_smi_preflight_used_mb": snapshot["nvidia_smi_used_mb"],
        "nvidia_smi_device_total_mb": snapshot["nvidia_smi_device_total_mb"],
        "nvidia_smi_preflight_utilization_gpu": snapshot["nvidia_smi_utilization_gpu"],
    }


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def nvidia_smi_memory_snapshot() -> dict[str, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    values = output.splitlines()[0].split(",")
    return {
        "nvidia_smi_used_mb": int(values[0].strip()),
        "nvidia_smi_device_total_mb": int(values[1].strip()),
        "nvidia_smi_utilization_gpu": int(values[2].strip()),
    }


def _estimated_clean_peak_mb(*, generated_tokens: int, reference_tokens: int, true_cfg_scale: float) -> int:
    true_cfg_extra_mb = VRAM_ESTIMATE_TRUE_CFG_MB if true_cfg_scale > 1.0 else 0
    high_generated_token_extra_mb = max(0, generated_tokens - VRAM_ESTIMATE_HIGH_GENERATED_TOKEN_THRESHOLD)
    token_mb = (
        generated_tokens * VRAM_ESTIMATE_GENERATED_TOKEN_NUMERATOR
        + reference_tokens * VRAM_ESTIMATE_REFERENCE_TOKEN_NUMERATOR
    ) // VRAM_ESTIMATE_TOKEN_DENOMINATOR
    return VRAM_ESTIMATE_BASE_PEAK_MB + token_mb + true_cfg_extra_mb + high_generated_token_extra_mb


def _max_generated_tokens_for_clean_peak(vram_plan: dict[str, Any], clean_available_mb: int) -> int:
    low = 1
    high = max(2, vram_plan["generated_tokens"])
    while (
        _estimated_clean_peak_mb(
            generated_tokens=high,
            reference_tokens=vram_plan["reference_tokens"],
            true_cfg_scale=1.1 if vram_plan["true_cfg_enabled"] else 1.0,
        )
        <= clean_available_mb
    ):
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if (
            _estimated_clean_peak_mb(
                generated_tokens=middle,
                reference_tokens=vram_plan["reference_tokens"],
                true_cfg_scale=1.1 if vram_plan["true_cfg_enabled"] else 1.0,
            )
            <= clean_available_mb
        ):
            low = middle
        else:
            high = middle
    return low
