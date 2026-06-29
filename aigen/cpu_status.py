from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CpuTimes:
    idle: int
    total: int


class CpuStatusError(RuntimeError):
    pass


class CpuUsageSampler:
    def __init__(self) -> None:
        self._previous = _read_cpu_times()

    def percent(self) -> float:
        current = _read_cpu_times()
        previous = self._previous
        self._previous = current
        total_delta = current.total - previous.total
        if total_delta <= 0:
            return 0.0
        idle_delta = current.idle - previous.idle
        return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def _read_cpu_times() -> CpuTimes:
    try:
        with open("/proc/stat", "r", encoding="utf-8") as handle:
            fields = handle.readline().split()
    except OSError:
        raise CpuStatusError("Cannot read /proc/stat for CPU utilization")
    if len(fields) < 5 or fields[0] != "cpu":
        raise CpuStatusError("Unexpected /proc/stat CPU line")
    values = [int(value) for value in fields[1:]]
    idle = values[3]
    if len(values) > 4:
        idle += values[4]
    return CpuTimes(idle=idle, total=sum(values))
