from __future__ import annotations

import subprocess


class GpuStatusError(RuntimeError):
    pass


def nvidia_smi_memory_snapshot() -> dict[str, int]:
    try:
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
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError) as error:
        raise GpuStatusError("Cannot read nvidia-smi GPU telemetry") from error
