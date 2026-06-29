from __future__ import annotations

from dataclasses import dataclass

from aigen.cpu_status import CpuUsageSampler
from aigen.gpu_status import nvidia_smi_memory_snapshot


@dataclass(frozen=True)
class SystemTelemetry:
    cpu_percent: float
    gpu_percent: int
    vram_used_mb: int
    vram_total_mb: int


class SystemTelemetrySampler:
    def __init__(self) -> None:
        self._cpu = CpuUsageSampler()

    def sample(self) -> SystemTelemetry:
        cpu_percent = self._cpu.percent()
        gpu = nvidia_smi_memory_snapshot()
        return SystemTelemetry(
            cpu_percent=cpu_percent,
            gpu_percent=gpu["nvidia_smi_utilization_gpu"],
            vram_used_mb=gpu["nvidia_smi_used_mb"],
            vram_total_mb=gpu["nvidia_smi_device_total_mb"],
        )
