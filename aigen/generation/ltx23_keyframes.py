from __future__ import annotations

import json
import os
import subprocess
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
    phase_metrics: tuple[dict[str, Any], ...]
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
            "phase_metrics": list(self.phase_metrics),
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
    return generate_ltx23_keyframes_seed_sweep(
        prompt=prompt,
        keyframes=keyframes,
        output=output,
        resolution=resolution,
        frames=frames,
        fps=fps,
        steps=steps,
        solver=solver,
        seeds=(seed,),
        progress=progress,
    )[0]


def generate_ltx23_keyframes_seed_sweep(
    *,
    prompt: str,
    keyframes: Sequence[Ltx23Keyframe],
    output: Path,
    resolution: str,
    frames: int,
    fps: int,
    steps: int,
    solver: str,
    seeds: Sequence[int],
    progress: StatusReporter,
) -> tuple[Ltx23KeyframesResult, ...]:
    normalized_seeds = tuple(seeds)
    if not normalized_seeds:
        raise Ltx23KeyframesError("LTX-2.3 seed sweep requires at least one seed")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise Ltx23KeyframesError("LTX-2.3 seed sweep contains duplicate seeds")
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
    outputs = tuple(
        output
        if len(normalized_seeds) == 1
        else output.with_name(f"{output.stem}-seed{seed}{output.suffix}")
        for seed in normalized_seeds
    )
    configs = tuple(
        job_output.with_name(f"{job_output.stem}_config.json")
        for job_output in outputs
    )
    log = (
        output.with_suffix(f"{output.suffix}.log")
        if len(normalized_seeds) == 1
        else output.with_name(f"{output.stem}-seed-sweep.log")
    )
    artifacts = (log, *(path for pair in zip(outputs, configs) for path in pair))
    existing = next((path for path in artifacts if path.exists()), None)
    if existing is not None:
        raise Ltx23KeyframesError(f"output already exists: {existing}")

    runtime_root = _runtime_root()
    runtime_python = runtime_root / "venv/bin/python"
    source_root = runtime_root / "Wan2GP"
    _validate_runtime(runtime_python, source_root)
    output.parent.mkdir(parents=True, exist_ok=True)

    requests = tuple(
        {
            "kind": "aigen-ltx23-keyframes-job",
            "prompt": prompt.strip(),
            "keyframes": [keyframe.to_json() for keyframe in normalized_keyframes],
            "output": job_output.as_posix(),
            "resolution": resolution,
            "frames": frames,
            "fps": fps,
            "steps": steps,
            "solver": solver,
            "seed": seed,
        }
        for seed, job_output in zip(normalized_seeds, outputs, strict=True)
    )
    started = time.monotonic()
    responses = _run_worker_requests(
        requests=requests,
        runtime_root=runtime_root,
        runtime_python=runtime_python,
        source_root=source_root,
        log=log,
        progress=progress,
    )
    total_elapsed_seconds = time.monotonic() - started

    results = []
    for request, response, job_output, config, seed in zip(
        requests,
        responses,
        outputs,
        configs,
        normalized_seeds,
        strict=True,
    ):
        if not job_output.is_file() or job_output.stat().st_size == 0:
            raise Ltx23KeyframesError(f"LTX-2.3 did not create a video: {job_output}")
        elapsed_seconds = (
            total_elapsed_seconds
            if len(requests) == 1
            else float(response["environment"]["elapsed_seconds"])
        )
        config_payload = {
            "kind": "aigen-ltx23-keyframes-config",
            "runtime": "WanGP",
            "runtime_revision": LTX23_WANGP_REVISION,
            "request": request,
            "effective_settings": response["effective_settings"],
            "phase_metrics": response["phase_metrics"],
            "environment": response["environment"],
            "elapsed_seconds": round(elapsed_seconds, 3),
        }
        config.write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(
            Ltx23KeyframesResult(
                output=job_output,
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
                phase_metrics=tuple(response["phase_metrics"]),
                environment=dict(response["environment"]),
            )
        )
    progress.phase("LTX-2.3 generation completed")
    return tuple(results)


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


def _run_worker_requests(
    *,
    requests: Sequence[dict[str, Any]],
    runtime_root: Path,
    runtime_python: Path,
    source_root: Path,
    log: Path,
    progress: StatusReporter,
) -> tuple[dict[str, Any], ...]:
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

    responses = []
    progress.phase("starting LTX-2.3 worker")
    with log.open("w", encoding="utf-8") as worker_log:
        with subprocess.Popen(
            [
                runtime_python.as_posix(),
                "-m",
                "aigen.generation.ltx23_wangp_worker",
            ],
            cwd=source_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=worker_log,
            text=True,
            encoding="utf-8",
            bufsize=1,
        ) as worker:
            try:
                if worker.stdin is None or worker.stdout is None:
                    raise Ltx23KeyframesError("LTX-2.3 worker pipes are unavailable")
                for request in requests:
                    worker.stdin.write(
                        json.dumps(request, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                worker.stdin.close()
                for line in worker.stdout:
                    response = _apply_worker_event(line, progress)
                    if response is not None:
                        responses.append(response)
                returncode = worker.wait()
            except BaseException:
                if worker.poll() is None:
                    worker.terminate()
                raise

    failed_response = next(
        (response for response in responses if response.get("status") != "completed"),
        None,
    )
    if returncode != 0 or failed_response is not None:
        message = (
            failed_response.get("message")
            if failed_response is not None
            else _log_tail(log)
        )
        raise Ltx23KeyframesError(f"LTX-2.3 WanGP failed: {message}")
    if len(responses) != len(requests):
        raise Ltx23KeyframesError(
            f"LTX-2.3 worker returned {len(responses)} results for {len(requests)} requests"
        )
    for request, response in zip(requests, responses, strict=True):
        if response["output"] != request["output"]:
            raise Ltx23KeyframesError("LTX-2.3 worker returned results out of order")
    return tuple(responses)


def _apply_worker_event(
    line: str,
    progress: StatusReporter,
) -> dict[str, Any] | None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as error:
        raise Ltx23KeyframesError("invalid LTX-2.3 worker event") from error
    match event["kind"]:
        case "phase":
            progress.phase(event["text"])
        case "begin":
            progress.begin(event["total"], event["text"])
        case "step":
            progress.step(event["text"])
        case "result":
            return event["response"]
        case kind:
            raise Ltx23KeyframesError(f"unknown LTX-2.3 worker event: {kind}")
    return None


def _log_tail(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as error:
        return f"unable to read worker log: {error}"
    return text[-8192:] or "worker failed without an error message"
