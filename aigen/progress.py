from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Protocol, TextIO

from aigen.system_telemetry import SystemTelemetry, SystemTelemetrySampler


DEFAULT_PROGRESS_INTERVAL_SECONDS = 2.0
ACTIVITY_WIDTH = 18


@dataclass(frozen=True)
class RuntimeStatusSnapshot:
    phase: str
    completed: int
    total: int
    elapsed_seconds: float
    remaining_seconds: float | None
    final: bool
    telemetry: SystemTelemetry


class StatusReporter(Protocol):
    @property
    def renders_live(self) -> bool: ...

    def __enter__(self) -> StatusReporter: ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...

    def begin(self, total: int, phase: str) -> None: ...

    def phase(self, text: str) -> None: ...

    def step(self, text: str) -> None: ...

    def finish(self, status: str) -> None: ...


class StatusRenderer(Protocol):
    def render(self, snapshot: RuntimeStatusSnapshot) -> None: ...

    def close(self) -> None: ...


class RuntimeStatus:
    def __init__(
        self,
        *,
        interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
        renderer: StatusRenderer,
        telemetry: SystemTelemetrySampler,
    ) -> None:
        self._renderer = renderer
        self._interval_seconds = interval_seconds
        self._telemetry = telemetry
        self._phase = "starting"
        self._completed = 0
        self._total = 0
        self._previous_step_at: float | None = None
        self._seconds_per_step: float | None = None
        self._started_at = time.monotonic()
        self._done = Event()
        self._lock = Lock()
        self._thread = Thread(target=self._render_loop, daemon=True)

    @classmethod
    def terminal(
        cls,
        *,
        interval_seconds: float,
        stream: TextIO,
        telemetry: SystemTelemetrySampler,
        close_stream: bool = False,
    ) -> RuntimeStatus:
        return cls(
            interval_seconds=interval_seconds,
            renderer=TerminalLineRenderer(stream=stream, close_stream=close_stream),
            telemetry=telemetry,
        )

    @classmethod
    def lines(
        cls,
        *,
        interval_seconds: float,
        stream: TextIO,
        telemetry: SystemTelemetrySampler,
        close_stream: bool = False,
    ) -> RuntimeStatus:
        return cls(
            interval_seconds=interval_seconds,
            renderer=ProgressLineRenderer(stream=stream, close_stream=close_stream),
            telemetry=telemetry,
        )

    @property
    def renders_live(self) -> bool:
        return True

    def __enter__(self) -> RuntimeStatus:
        self._render()
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.finish("failed" if exc_type else "completed")

    def phase(self, text: str) -> None:
        with self._lock:
            self._phase = text

    def begin(self, total: int, phase: str) -> None:
        if total < 0:
            raise ValueError("progress total must be non-negative")
        with self._lock:
            self._total = total
            self._completed = 0
            self._phase = phase
            self._previous_step_at = None
            self._seconds_per_step = None

    def step(self, text: str) -> None:
        now = time.monotonic()
        with self._lock:
            if self._previous_step_at is not None:
                step_seconds = now - self._previous_step_at
                self._seconds_per_step = (
                    step_seconds
                    if self._seconds_per_step is None
                    else self._seconds_per_step * 0.75 + step_seconds * 0.25
                )
            self._previous_step_at = now
            self._completed += 1
            self._phase = text

    def finish(self, status: str) -> None:
        if self._done.is_set():
            return
        self._done.set()
        if self._thread.is_alive():
            self._thread.join()
        with self._lock:
            self._phase = status
        self._render(final=True)
        self._renderer.close()

    def _render_loop(self) -> None:
        while not self._done.wait(self._interval_seconds):
            self._render()

    def _render(self, *, final: bool = False) -> None:
        snapshot = self._snapshot(final=final)
        self._renderer.render(snapshot)

    def _snapshot(self, *, final: bool) -> RuntimeStatusSnapshot:
        with self._lock:
            phase = self._phase
            completed = self._completed
            total = self._total
            remaining_seconds = (
                (total - completed) * self._seconds_per_step
                if self._seconds_per_step is not None and completed < total
                else None
            )
        snapshot = RuntimeStatusSnapshot(
            phase=phase,
            completed=completed,
            total=total,
            elapsed_seconds=time.monotonic() - self._started_at,
            remaining_seconds=remaining_seconds,
            final=final,
            telemetry=self._telemetry.sample(),
        )
        return snapshot


class TerminalLineRenderer:
    def __init__(self, *, stream: TextIO, close_stream: bool = False) -> None:
        self._stream = stream
        self._close_stream = close_stream
        self._last_line_length = 0

    def render(self, snapshot: RuntimeStatusSnapshot) -> None:
        line = _fit_line(_format_line(snapshot), self._stream)
        clear = " " * max(0, self._last_line_length - len(line))
        self._stream.write(f"\r{line}{clear}")
        if snapshot.final:
            self._stream.write("\n")
        self._stream.flush()
        self._last_line_length = 0 if snapshot.final else len(line)

    def close(self) -> None:
        if self._close_stream:
            self._stream.close()


class ProgressLineRenderer:
    def __init__(self, *, stream: TextIO, close_stream: bool = False) -> None:
        self._stream = stream
        self._close_stream = close_stream

    def render(self, snapshot: RuntimeStatusSnapshot) -> None:
        self._stream.write(f"{_format_line(snapshot)}\n")
        self._stream.flush()

    def close(self) -> None:
        if self._close_stream:
            self._stream.close()


class SilentRuntimeStatus:
    @property
    def renders_live(self) -> bool:
        return False

    def __enter__(self) -> SilentRuntimeStatus:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        pass

    def phase(self, text: str) -> None:
        pass

    def begin(self, total: int, phase: str) -> None:
        pass

    def step(self, text: str) -> None:
        pass

    def finish(self, status: str) -> None:
        pass


SILENT_STATUS = SilentRuntimeStatus()


def open_cli_progress() -> StatusReporter:
    _claim_progress_ownership()
    if _progress_disabled():
        return SilentRuntimeStatus()
    if os.environ.get("AIGEN_PROGRESS") == "stderr":
        return _stderr_progress()
    stream = _open_terminal_stream()
    if stream is not None:
        return RuntimeStatus.terminal(
            interval_seconds=_progress_interval_seconds(),
            stream=stream,
            telemetry=SystemTelemetrySampler(),
            close_stream=True,
        )
    if os.environ.get("AIGEN_PROGRESS") == "tty":
        return SilentRuntimeStatus()
    return _stderr_progress()


def _stderr_progress() -> RuntimeStatus:
    return RuntimeStatus.lines(
        interval_seconds=_progress_interval_seconds(),
        stream=sys.stderr,
        telemetry=SystemTelemetrySampler(),
        close_stream=False,
    )


def _claim_progress_ownership() -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def _open_terminal_stream() -> TextIO | None:
    try:
        return open("/dev/tty", "w", encoding="utf-8", buffering=1)
    except OSError:
        return None


def _progress_disabled() -> bool:
    return os.environ.get("AIGEN_PROGRESS") == "0"


def _progress_interval_seconds() -> float:
    raw = os.environ.get("AIGEN_PROGRESS_INTERVAL_SECONDS")
    if raw is None:
        return DEFAULT_PROGRESS_INTERVAL_SECONDS
    return max(0.25, float(raw))


def _format_line(snapshot: RuntimeStatusSnapshot) -> str:
    parts = []
    if snapshot.total:
        parts.append(
            f"{_activity(snapshot, width=ACTIVITY_WIDTH)} "
            f"{min(100, snapshot.completed * 100 // snapshot.total):3d}%"
        )
    parts.append(snapshot.phase)
    if snapshot.total and not snapshot.final and snapshot.completed < snapshot.total:
        parts.append(
            "eta --:--"
            if snapshot.remaining_seconds is None
            else f"eta {_duration_text(snapshot.remaining_seconds)}"
        )
    parts.extend(
        [
            f"elapsed {_duration_text(snapshot.elapsed_seconds)}",
            _cpu_text(snapshot.telemetry),
            _gpu_text(snapshot.telemetry),
        ]
    )
    return " | ".join(parts)


def _activity(snapshot: RuntimeStatusSnapshot, *, width: int) -> str:
    ratio = min(1.0, snapshot.completed / snapshot.total)
    filled = int(width * ratio)
    chars = ["="] * filled + [" "] * (width - filled)
    if not snapshot.final and snapshot.completed < snapshot.total:
        chars[min(width - 1, filled)] = ">"
    return "[" + "".join(chars) + "]"


def _cpu_text(telemetry: SystemTelemetry) -> str:
    return f"cpu {telemetry.cpu_percent:5.1f}%"


def _gpu_text(telemetry: SystemTelemetry) -> str:
    if telemetry.gpu is None:
        return "gpu n/a | vram n/a"
    vram_percent = round(telemetry.gpu.vram_used_mb * 100 / telemetry.gpu.vram_total_mb)
    return (
        f"gpu {telemetry.gpu.gpu_percent:3d}% | "
        f"vram {telemetry.gpu.vram_used_mb}/{telemetry.gpu.vram_total_mb} MB ({vram_percent}%)"
    )


def _duration_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _fit_line(line: str, stream: TextIO) -> str:
    if not stream.isatty():
        return line
    columns = shutil.get_terminal_size().columns
    if columns <= 1:
        return line
    return line[: columns - 1]
