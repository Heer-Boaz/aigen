from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.progress import StatusReporter
from aigen.runtime_profiles import PROJECT_ROOT


LTX23_WANGP_REVISION = "5582327dc25e45fec6cda0f27144d4dcf7ed104b"
LTX23_MODEL_TYPE = "ltx2_22B_nvfp4"
LTX23_DEFAULT_FPS = 24
LTX23_MINIMUM_FRAMES = 17
LTX23_FRAME_STEP = 8
LTX23_SOLVERS = frozenset({"distilled_8_steps", "euler", "res2s"})


class Ltx23KeyframesError(RuntimeError):
    pass


@dataclass(frozen=True)
class Ltx23Keyframe:
    image: Path
    frame: int

    def to_json(self) -> dict[str, Any]:
        return {"image": self.image.as_posix(), "frame": self.frame}


@dataclass(frozen=True)
class Ltx23KeyframesResult:
    output: Path
    config: Path
    log: Path
    keyframes: tuple[Ltx23Keyframe, ...]
    resolution: str
    frames: int
    fps: int
    steps: int
    solver: str
    seed: int
    elapsed_seconds: float
    environment: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "completed",
            "kind": "ltx-2.3-multi-keyframe-video",
            "output": self.output.as_posix(),
            "output_bytes": self.output.stat().st_size,
            "config": self.config.as_posix(),
            "log": self.log.as_posix(),
            "runtime": "WanGP",
            "runtime_revision": LTX23_WANGP_REVISION,
            "model_type": LTX23_MODEL_TYPE,
            "keyframes": [keyframe.to_json() for keyframe in self.keyframes],
            "resolution": self.resolution,
            "frames": self.frames,
            "fps": self.fps,
            "steps": self.steps,
            "solver": self.solver,
            "seed": self.seed,
            "prompt_enhancer": False,
            "audio": False,
            "background_removal": False,
            "spatial_upsampling": False,
            "temporal_upsampling": False,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "environment": self.environment,
        }


def generate_ltx23_keyframes(
    *,
    prompt: str,
    keyframes: Sequence[Ltx23Keyframe],
    output: Path,
    resolution: str,
    frames: int,
    fps: int,
    steps: int,
    solver: str,
    seed: int,
    progress: StatusReporter,
) -> Ltx23KeyframesResult:
    normalized_keyframes = _validate_request(
        prompt=prompt,
        keyframes=keyframes,
        output=output,
        resolution=resolution,
        frames=frames,
        fps=fps,
        steps=steps,
        solver=solver,
    )
    output = output.expanduser().resolve()
    config = output.with_name(f"{output.stem}_config.json")
    log = output.with_suffix(f"{output.suffix}.log")
    existing = next((path for path in (output, config, log) if path.exists()), None)
    if existing is not None:
        raise Ltx23KeyframesError(f"output already exists: {existing}")

    runtime_root = _runtime_root()
    runtime_python = runtime_root / "venv/bin/python"
    source_root = runtime_root / "Wan2GP"
    _validate_runtime(runtime_python, source_root)
    output.parent.mkdir(parents=True, exist_ok=True)

    request = {
        "kind": "aigen-ltx23-keyframes-job",
        "prompt": prompt.strip(),
        "keyframes": [keyframe.to_json() for keyframe in normalized_keyframes],
        "output": output.as_posix(),
        "resolution": resolution,
        "frames": frames,
        "fps": fps,
        "steps": steps,
        "solver": solver,
        "seed": seed,
    }
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="aigen-ltx23-job-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        request_path = temporary_path / "request.json"
        response_path = temporary_path / "response.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")

        environment = os.environ.copy()
        python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            path
            for path in (
                PROJECT_ROOT.as_posix(),
                source_root.as_posix(),
                python_path,
            )
            if path
        )
        environment.update(
            AIGEN_LTX23_ROOT=runtime_root.as_posix(),
            PYTHONUNBUFFERED="1",
            TOKENIZERS_PARALLELISM="false",
        )

        progress.phase("starting LTX-2.3 worker")
        with log.open("w", encoding="utf-8") as worker_log:
            with subprocess.Popen(
                [
                    runtime_python.as_posix(),
                    "-m",
                    "aigen.generation.ltx23_wangp_worker",
                    request_path.as_posix(),
                    response_path.as_posix(),
                ],
                cwd=source_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=worker_log,
                text=True,
                encoding="utf-8",
                bufsize=1,
            ) as worker:
                try:
                    if worker.stdout is None:
                        raise Ltx23KeyframesError("LTX-2.3 worker has no progress stream")
                    for line in worker.stdout:
                        _apply_worker_progress(line, progress)
                    returncode = worker.wait()
                except BaseException:
                    if worker.poll() is None:
                        worker.terminate()
                    raise

        response = _read_worker_response(response_path)
        if returncode != 0 or response.get("status") != "completed":
            message = response.get("message") or _log_tail(log)
            raise Ltx23KeyframesError(f"LTX-2.3 WanGP failed: {message}")

    if not output.is_file() or output.stat().st_size == 0:
        raise Ltx23KeyframesError(f"LTX-2.3 did not create a video: {output}")

    elapsed_seconds = time.monotonic() - started
    config_payload = {
        "kind": "aigen-ltx23-keyframes-config",
        "runtime": "WanGP",
        "runtime_revision": LTX23_WANGP_REVISION,
        "request": request,
        "effective_settings": response["effective_settings"],
        "environment": response["environment"],
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    config.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    progress.phase("LTX-2.3 generation completed")
    return Ltx23KeyframesResult(
        output=output,
        config=config,
        log=log,
        keyframes=normalized_keyframes,
        resolution=resolution,
        frames=frames,
        fps=fps,
        steps=steps,
        solver=solver,
        seed=seed,
        elapsed_seconds=elapsed_seconds,
        environment=dict(response["environment"]),
    )


def _validate_request(
    *,
    prompt: str,
    keyframes: Sequence[Ltx23Keyframe],
    output: Path,
    resolution: str,
    frames: int,
    fps: int,
    steps: int,
    solver: str,
) -> tuple[Ltx23Keyframe, ...]:
    if not prompt.strip():
        raise Ltx23KeyframesError("video motion prompt must not be empty")
    if output.suffix.lower() != ".mp4":
        raise Ltx23KeyframesError("LTX-2.3 output must use the .mp4 extension")
    if frames < LTX23_MINIMUM_FRAMES or (frames - 1) % LTX23_FRAME_STEP != 0:
        raise Ltx23KeyframesError(
            f"LTX-2.3 frames must be {LTX23_MINIMUM_FRAMES} or more in increments of {LTX23_FRAME_STEP}"
        )
    if fps <= 0:
        raise Ltx23KeyframesError("frames per second must be positive")
    if steps <= 0:
        raise Ltx23KeyframesError("inference steps must be positive")
    if solver not in LTX23_SOLVERS:
        raise Ltx23KeyframesError(f"unsupported LTX-2.3 solver: {solver}")
    if solver == "distilled_8_steps" and steps != 8:
        raise Ltx23KeyframesError("distilled_8_steps requires exactly 8 inference steps")
    try:
        width_text, height_text = resolution.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except ValueError as error:
        raise Ltx23KeyframesError("resolution must use WIDTHxHEIGHT") from error
    if width <= 0 or height <= 0:
        raise Ltx23KeyframesError("resolution dimensions must be positive")
    if len(keyframes) < 2:
        raise Ltx23KeyframesError("LTX-2.3 multi-keyframe generation requires at least two keyframes")

    normalized = tuple(
        sorted(
            (
                Ltx23Keyframe(image=keyframe.image.expanduser().resolve(), frame=keyframe.frame)
                for keyframe in keyframes
            ),
            key=lambda keyframe: keyframe.frame,
        )
    )
    duplicate = next(
        (
            current.frame
            for previous, current in zip(normalized, normalized[1:])
            if previous.frame == current.frame
        ),
        None,
    )
    if duplicate is not None:
        raise Ltx23KeyframesError(f"duplicate keyframe position: {duplicate}")
    invalid_position = next(
        (keyframe.frame for keyframe in normalized if not 0 <= keyframe.frame < frames),
        None,
    )
    if invalid_position is not None:
        raise Ltx23KeyframesError(
            f"keyframe position {invalid_position} is outside video frame range 0..{frames - 1}"
        )
    missing = next((keyframe.image for keyframe in normalized if not keyframe.image.is_file()), None)
    if missing is not None:
        raise Ltx23KeyframesError(f"keyframe image does not exist: {missing}")
    return normalized


def _runtime_root() -> Path:
    configured = os.environ.get("AIGEN_LTX23_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path.home() / ".cache/aigen-wangp"
    )


def _validate_runtime(runtime_python: Path, source_root: Path) -> None:
    required = (
        runtime_python,
        source_root / "wgp.py",
        source_root / "shared/api.py",
        source_root / "defaults/ltx2_22B_nvfp4.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise Ltx23KeyframesError(
            "LTX-2.3 runtime is incomplete; run scripts/install_ltx23.sh: "
            + ", ".join(path.as_posix() for path in missing)
        )
    revision = subprocess.run(
        ["git", "-C", source_root.as_posix(), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0 or revision.stdout.strip() != LTX23_WANGP_REVISION:
        raise Ltx23KeyframesError(
            f"WanGP source must be pinned to {LTX23_WANGP_REVISION}"
        )


def _read_worker_response(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Ltx23KeyframesError(f"invalid LTX-2.3 worker response {path}: {error}") from error


def _apply_worker_progress(line: str, progress: StatusReporter) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as error:
        raise Ltx23KeyframesError("invalid LTX-2.3 worker progress event") from error
    match event["kind"]:
        case "phase":
            progress.phase(event["text"])
        case "begin":
            progress.begin(event["total"], event["text"])
        case "step":
            progress.step(event["text"])
        case kind:
            raise Ltx23KeyframesError(f"unknown LTX-2.3 worker progress event: {kind}")


def _log_tail(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as error:
        return f"unable to read worker log: {error}"
    return text[-8192:] or "worker failed without an error message"
