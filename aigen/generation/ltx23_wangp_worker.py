from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, TextIO


RUNTIME_PROFILE = "4"
# mmgp can consume roughly twice its configured budget;
# 4000 MiB leaves headroom on 16 GiB GPUs.
PRELOAD_MIB = 4000
ATTENTION_MODE = "sdpa"

TWO_STAGE_SOLVER_SETTINGS = {
    "distilled_8_steps": {
        "guidance_phases": 2,
        "guidance_scale": 1.0,
        "audio_guidance_scale": 1.0,
        "alt_guidance_scale": 1.0,
        "alt_scale": 0.0,
    },
    "res2s": {
        "guidance_phases": 2,
        "guidance_scale": 3.0,
        "audio_guidance_scale": 7.0,
        "alt_guidance_scale": 3.0,
        "alt_scale": 0.45,
        "perturbation_switch": 0,
        "apg_switch": 0,
        "cfg_star_switch": 0,
        "self_refiner_setting": 0,
    },
    "euler": {
        "guidance_phases": 2,
        "guidance_scale": 3.0,
        "audio_guidance_scale": 7.0,
        "alt_guidance_scale": 3.0,
        "alt_scale": 0.7,
        "perturbation_switch": 2,
        "perturbation_layers": [28],
        "perturbation_start_perc": 0,
        "perturbation_end_perc": 100,
        "apg_switch": 0,
        "cfg_star_switch": 0,
        "self_refiner_setting": 0,
    },
}


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("usage: ltx23_wangp_worker")
    requests = [
        json.loads(line)
        for line in sys.stdin
        if line.strip()
    ]
    if not requests:
        raise SystemExit("ltx23_wangp_worker requires at least one JSON request on stdin")
    with _open_progress_stream() as progress_stream:
        try:
            return _run_requests(requests, progress_stream)
        except Exception:
            traceback.print_exc()
            return 1


def _run_requests(requests: list[dict[str, Any]], progress_stream: TextIO) -> int:
    phase_metrics = _PhaseMetrics()
    phase_metrics.start("initialization")
    started = time.monotonic()
    _send_progress(progress_stream, "phase", text="initializing WanGP")
    import torch
    from shared.api import init

    runtime_root = Path(os.environ["AIGEN_LTX23_ROOT"]).resolve()
    source_root = runtime_root / "Wan2GP"
    session = init(
        root=source_root,
        config_path=runtime_root / "config/wgp_config.json",
        output_dir=Path(requests[0]["output"]).resolve().parent,
        cli_args=[
            "--attention",
            ATTENTION_MODE,
            "--profile",
            RUNTIME_PROFILE,
            "--preload",
            str(PRELOAD_MIB),
        ],
        console_output=False,
    )
    model_type = requests[0]["model_type"]
    schema = session.get_model_schema(model_type)
    if schema is None:
        raise RuntimeError(f"WanGP has no {model_type} model definition")
    try:
        for index, request in enumerate(requests, start=1):
            if index > 1:
                phase_metrics = _PhaseMetrics()
                started = time.monotonic()
            _send_progress(
                progress_stream,
                "phase",
                text=f"LTX-2.3 job {index}/{len(requests)}",
            )
            try:
                response = _run(
                    request,
                    progress_stream,
                    session=session,
                    default_settings=schema["default_settings"],
                    model_type=model_type,
                    torch=torch,
                    phase_metrics=phase_metrics,
                    started=started,
                )
            except Exception as error:
                traceback.print_exc()
                _send_progress(
                    progress_stream,
                    "result",
                    response={
                        "status": "error",
                        "error": error.__class__.__name__,
                        "message": str(error),
                    },
                )
                return 1
            _send_progress(progress_stream, "result", response=response)
    finally:
        session.close()
    return 0


def _run(
    request: dict[str, Any],
    progress_stream: TextIO,
    *,
    session: Any,
    default_settings: dict[str, Any],
    model_type: str,
    torch: Any,
    phase_metrics: _PhaseMetrics,
    started: float,
) -> dict[str, Any]:
    output = Path(request["output"]).resolve()
    torch.cuda.reset_peak_memory_stats()
    phase_metrics.start("generation_setup")
    settings = _build_wangp_settings(request, default_settings, model_type)
    callbacks = _ProgressCallbacks(progress_stream, phase_metrics)

    _send_progress(progress_stream, "phase", text="loading LTX-2.3 and generating video")
    try:
        result = session.run_task(settings, callbacks=callbacks)
    finally:
        callbacks.finish()
    if not result.success:
        messages = "; ".join(error.message for error in result.errors)
        raise RuntimeError(messages or "WanGP generation failed without a structured error")
    videos = [
        Path(path).resolve()
        for path in result.generated_files
        if Path(path).suffix.lower() == ".mp4"
    ]
    if len(videos) != 1:
        raise RuntimeError(f"WanGP returned {len(videos)} video files; expected exactly one")
    generated = videos[0]
    if not generated.is_file() or generated.stat().st_size == 0:
        raise RuntimeError(f"WanGP video is missing or empty: {generated}")
    if output.exists():
        raise RuntimeError(f"requested output appeared while generation was running: {output}")
    phase_metrics.start("saving_video")
    _move_video_without_audio(generated, output)
    phase_metrics.finish()
    _send_progress(progress_stream, "phase", text="saved LTX-2.3 video")
    return {
        "status": "completed",
        "output": output.as_posix(),
        "audio": False,
        "effective_settings": settings,
        "phase_metrics": phase_metrics.records,
        "environment": {
            "engine": "WanGP",
            "model_type": model_type,
            "runtime_profile": int(RUNTIME_PROFILE),
            "preload_mib": PRELOAD_MIB,
            "attention": ATTENTION_MODE,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability(0)
            ),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    }


def _build_wangp_settings(
    request: dict[str, Any],
    default_settings: dict[str, Any],
    model_type: str,
) -> dict[str, Any]:
    frames = request["frames"]
    keyframes = sorted(request["keyframes"], key=lambda keyframe: keyframe["frame"])
    start = next((keyframe for keyframe in keyframes if keyframe["frame"] == 0), None)
    end = next(
        (keyframe for keyframe in keyframes if keyframe["frame"] == frames - 1),
        None,
    )
    injected = [keyframe for keyframe in keyframes if keyframe not in (start, end)]
    image_prompt_type = ("S" if start is not None else "") + (
        "E" if end is not None else ""
    )
    settings = dict(default_settings)
    settings.update(
        model_type=model_type,
        prompt=request["prompt"],
        resolution=request["resolution"],
        image_mode=0,
        image_prompt_type=image_prompt_type,
        video_prompt_type="KFI" if injected else "",
        image_start=start["image"] if start is not None else None,
        image_end=end["image"] if end is not None else None,
        image_refs=[keyframe["image"] for keyframe in injected] or None,
        frames_positions=" ".join(str(keyframe["frame"] + 1) for keyframe in injected),
        input_video_strength=request["conditioning_strength"],
        remove_background_images_ref=0,
        prompt_enhancer="",
        audio_prompt_type="A",
        postprocess_audio="",
        temporal_upsampling="",
        spatial_upsampling="",
        video_length=frames,
        force_fps=request["fps"],
        num_inference_steps=request["steps"],
        sample_solver=request["solver"],
        batch_size=1,
        repeat_generation=1,
        seed=request["seed"],
    )
    settings.update(TWO_STAGE_SOLVER_SETTINGS.get(request["solver"], {}))
    return settings


def _move_video_without_audio(source: Path, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}-",
        suffix=destination.suffix,
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary = Path(temporary_file.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                source.as_posix(),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                "-an",
                "-movflags",
                "+faststart",
                temporary.as_posix(),
            ],
            check=True,
        )
        temporary.replace(source)
        source.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


class _ProgressCallbacks:
    def __init__(self, stream: TextIO, phase_metrics: _PhaseMetrics) -> None:
        self._stream = stream
        self._phase_metrics = phase_metrics
        self._phase = ""
        self._total_steps = 0
        self._current_step = 0

    def on_status(self, text: str) -> None:
        text = text.strip()
        if text and text != self._phase:
            self._phase = text
            self._total_steps = 0
            self._current_step = 0
            self._phase_metrics.start(text)
            _send_progress(self._stream, "phase", text=text)

    def on_progress(self, update: Any) -> None:
        phase = str(update.phase or "").strip()
        total = int(update.total_steps or 0)
        current = int(update.current_step or 0)
        if total > 0:
            new_sequence = total != self._total_steps or current < self._current_step
            if new_sequence:
                restart_phase = phase == self._phase and self._total_steps > 0
                self._total_steps = total
                self._current_step = 0
                self._phase = phase
                self._phase_metrics.start(
                    phase or "inference",
                    restart=restart_phase,
                )
                _send_progress(
                    self._stream,
                    "begin",
                    total=total,
                    text=phase or "denoising LTX-2.3 video",
                )
            while self._current_step < current:
                self._current_step += 1
                _send_progress(
                    self._stream,
                    "step",
                    text=f"{phase or 'denoising'} {self._current_step}/{total}",
                )
        elif phase and phase != self._phase:
            self._phase = phase
            self._phase_metrics.start(phase)
            _send_progress(self._stream, "phase", text=phase)

    def finish(self) -> None:
        self._phase_metrics.finish()


class _PhaseMetrics:
    def __init__(self) -> None:
        self._phase = ""
        self._started: dict[str, float | int] | None = None
        self._records: list[dict[str, float | int | str]] = []

    @property
    def records(self) -> list[dict[str, float | int | str]]:
        return list(self._records)

    def start(self, phase: str, *, restart: bool = False) -> None:
        phase = phase.strip()
        if not phase or (phase == self._phase and not restart):
            return
        self.finish()
        self._phase = phase
        self._started = _metric_snapshot()

    def finish(self) -> None:
        if self._started is None:
            return
        finished = _metric_snapshot()
        elapsed_seconds = finished["wall_seconds"] - self._started["wall_seconds"]
        cpu_seconds = finished["cpu_seconds"] - self._started["cpu_seconds"]
        page_size = os.sysconf("SC_PAGE_SIZE")
        record: dict[str, float | int | str] = {
            "index": len(self._records) + 1,
            "phase": self._phase,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "process_cpu_seconds": round(cpu_seconds, 3),
            "process_cpu_percent": round(cpu_seconds / elapsed_seconds * 100, 1),
            "disk_read_gib": round(
                (finished["read_bytes"] - self._started["read_bytes"]) / 1024**3,
                3,
            ),
            "disk_write_gib": round(
                (finished["write_bytes"] - self._started["write_bytes"]) / 1024**3,
                3,
            ),
            "system_swap_in_mib": round(
                (finished["pswpin"] - self._started["pswpin"]) * page_size / 1024**2,
                1,
            ),
            "system_swap_out_mib": round(
                (finished["pswpout"] - self._started["pswpout"]) * page_size / 1024**2,
                1,
            ),
            "mem_available_start_mib": round(self._started["mem_available_kib"] / 1024),
            "mem_available_end_mib": round(finished["mem_available_kib"] / 1024),
            "swap_free_start_mib": round(self._started["swap_free_kib"] / 1024),
            "swap_free_end_mib": round(finished["swap_free_kib"] / 1024),
        }
        self._records.append(record)
        # WanGP temporarily replaces sys.stderr while run_task is active;
        # file descriptor 2 remains the worker log owned by the parent process.
        os.write(
            2,
            (
                "[aigen-ltx23-phase] "
                + json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            ).encode("utf-8"),
        )
        self._phase = ""
        self._started = None


def _metric_snapshot() -> dict[str, float | int]:
    meminfo = _key_value_file(Path("/proc/meminfo"))
    vmstat = _key_value_file(Path("/proc/vmstat"))
    process_io = _key_value_file(Path("/proc/self/io"))
    return {
        "wall_seconds": time.monotonic(),
        "cpu_seconds": time.process_time(),
        "mem_available_kib": meminfo["MemAvailable"],
        "swap_free_kib": meminfo["SwapFree"],
        "pswpin": vmstat["pswpin"],
        "pswpout": vmstat["pswpout"],
        "read_bytes": process_io["read_bytes"],
        "write_bytes": process_io["write_bytes"],
    }


def _key_value_file(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator:
            key, _, raw_value = line.partition(" ")
        values[key] = int(raw_value.strip().split()[0])
    return values


def _open_progress_stream() -> TextIO:
    sys.stdout.flush()
    progress_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return os.fdopen(progress_fd, "w", encoding="utf-8", buffering=1)


def _send_progress(stream: TextIO, kind: str, **values: Any) -> None:
    stream.write(json.dumps({"kind": kind, **values}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
