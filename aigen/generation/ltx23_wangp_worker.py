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


MODEL_TYPE = "ltx2_22B_nvfp4"
RUNTIME_PROFILE = "4"
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
    if len(sys.argv) != 3:
        raise SystemExit("usage: ltx23_wangp_worker REQUEST RESPONSE")
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    with _open_progress_stream() as progress_stream:
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            response = _run(request, progress_stream)
        except Exception as error:
            traceback.print_exc()
            response = {
                "status": "error",
                "error": error.__class__.__name__,
                "message": str(error),
            }
            response_path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
            return 1
        response_path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
    return 0


def _run(request: dict[str, Any], progress_stream: TextIO) -> dict[str, Any]:
    _send_progress(progress_stream, "phase", text="initializing WanGP")
    import torch
    from shared.api import init

    runtime_root = Path(os.environ["AIGEN_LTX23_ROOT"]).resolve()
    source_root = runtime_root / "Wan2GP"
    output = Path(request["output"]).resolve()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    session = init(
        root=source_root,
        config_path=runtime_root / "config/wgp_config.json",
        output_dir=output.parent,
        cli_args=["--attention", ATTENTION_MODE, "--profile", RUNTIME_PROFILE],
        console_output=False,
    )
    schema = session.get_model_schema(MODEL_TYPE)
    if schema is None:
        raise RuntimeError(f"WanGP has no {MODEL_TYPE} model definition")
    settings = _build_wangp_settings(request, schema["default_settings"])
    callbacks = _ProgressCallbacks(progress_stream)

    _send_progress(progress_stream, "phase", text="loading LTX-2.3 and generating video")
    result = session.run_task(settings, callbacks=callbacks)
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
    _move_video_without_audio(generated, output)
    _send_progress(progress_stream, "phase", text="saved LTX-2.3 video")
    return {
        "status": "completed",
        "output": output.as_posix(),
        "audio": False,
        "effective_settings": settings,
        "environment": {
            "engine": "WanGP",
            "model_type": MODEL_TYPE,
            "runtime_profile": int(RUNTIME_PROFILE),
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
        model_type=MODEL_TYPE,
        prompt=request["prompt"],
        resolution=request["resolution"],
        image_mode=0,
        image_prompt_type=image_prompt_type,
        video_prompt_type="KFI" if injected else "",
        image_start=start["image"] if start is not None else None,
        image_end=end["image"] if end is not None else None,
        image_refs=[keyframe["image"] for keyframe in injected] or None,
        frames_positions=" ".join(str(keyframe["frame"] + 1) for keyframe in injected),
        input_video_strength=1.0,
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
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._phase = ""
        self._total_steps = 0
        self._current_step = 0

    def on_status(self, text: str) -> None:
        text = text.strip()
        if text and text != self._phase:
            self._phase = text
            _send_progress(self._stream, "phase", text=text)

    def on_progress(self, update: Any) -> None:
        phase = str(update.phase or "").strip()
        total = int(update.total_steps or 0)
        current = int(update.current_step or 0)
        if total > 0:
            if total != self._total_steps or current < self._current_step:
                self._total_steps = total
                self._current_step = 0
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
            _send_progress(self._stream, "phase", text=phase)


def _open_progress_stream() -> TextIO:
    sys.stdout.flush()
    progress_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return os.fdopen(progress_fd, "w", encoding="utf-8", buffering=1)


def _send_progress(stream: TextIO, kind: str, **values: Any) -> None:
    stream.write(json.dumps({"kind": kind, **values}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
