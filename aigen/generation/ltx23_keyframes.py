from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.progress import StatusReporter
from aigen.runtime_profiles import PROJECT_ROOT
from aigen.image_io import fit_image_canvas, open_image
from aigen.generation.ltx23_runtime import LTX23_WANGP_REVISION, Ltx23Installation, resolve_ltx23_installation
from aigen.generation.ltx23_settings import (
    LTX23_MODEL_TYPES, LTX23_DEFAULT_MODEL, LTX23_DEFAULT_FPS, LTX23_DEFAULT_PHASES,
    LTX23_DEFAULT_CONDITIONING_STRENGTH, LTX23_DEFAULT_NEGATIVE_PROMPT,
    LTX23_MINIMUM_FRAMES, LTX23_FRAME_STEP, LTX23_PHASES, LTX23_SOLVERS,
    Ltx23KeyframesError, resolve_ltx23_settings, validate_ltx23_positions,
)


LTX23_IMPLEMENTATION_REVISION = "2"


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
    phases: int
    solver: str
    negative_prompt: str
    conditioning_strength: float
    model: str
    model_type: str
    seed: int
    elapsed_seconds: float
    phase_metrics: tuple[dict[str, Any], ...]
    environment: dict[str, Any]
    keyframe_fit: str = "crop"

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "completed",
            "kind": "ltx-2.3-keyframe-conditioned-video",
            "output": self.output.as_posix(),
            "output_bytes": self.output.stat().st_size,
            "config": self.config.as_posix(),
            "log": self.log.as_posix(),
            "runtime": "WanGP",
            "runtime_revision": LTX23_WANGP_REVISION,
            "model": self.model,
            "model_type": self.model_type,
            "keyframes": [keyframe.to_json() for keyframe in self.keyframes],
            "keyframe_fit": self.keyframe_fit,
            "resolution": self.resolution,
            "frames": self.frames,
            "fps": self.fps,
            "steps": self.steps,
            "phases": self.phases,
            "solver": self.solver,
            "negative_prompt": self.negative_prompt,
            "conditioning_strength": self.conditioning_strength,
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
    phases: int,
    solver: str,
    negative_prompt: str,
    conditioning_strength: float,
    model: str,
    seed: int,
    progress: StatusReporter,
    keyframe_fit: str = "crop",
    installation: Ltx23Installation | None = None,
) -> Ltx23KeyframesResult:
    return generate_ltx23_keyframes_seed_sweep(
        prompt=prompt,
        keyframes=keyframes,
        output=output,
        resolution=resolution,
        frames=frames,
        fps=fps,
        steps=steps,
        phases=phases,
        solver=solver,
        negative_prompt=negative_prompt,
        conditioning_strength=conditioning_strength,
        model=model,
        seeds=(seed,),
        progress=progress,
        keyframe_fit=keyframe_fit,
        installation=installation,
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
    phases: int,
    solver: str,
    negative_prompt: str,
    conditioning_strength: float,
    model: str,
    seeds: Sequence[int],
    progress: StatusReporter,
    keyframe_fit: str = "crop",
    installation: Ltx23Installation | None = None,
    on_output: Callable[[Ltx23KeyframesResult], None] | None = None,
) -> tuple[Ltx23KeyframesResult, ...]:
    normalized_seeds = tuple(seeds)
    if not normalized_seeds:
        raise Ltx23KeyframesError("LTX-2.3 seed sweep requires at least one seed")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise Ltx23KeyframesError("LTX-2.3 seed sweep contains duplicate seeds")
    settings = resolve_ltx23_settings(
        resolution=resolution, frames=frames, fps=fps, steps=steps, phases=phases,
        solver=solver, conditioning_strength=conditioning_strength, model=model, keyframe_fit=keyframe_fit,
    )
    requested_resolution = resolution
    resolution = settings.resolution
    normalized_keyframes = _validate_request(
        prompt=prompt,
        keyframes=keyframes,
        output=output,
        frames=frames,
        negative_prompt=negative_prompt,
    )
    model_type = LTX23_MODEL_TYPES[model]
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

    installation = resolve_ltx23_installation(settings) if installation is None else installation
    runtime_root, runtime_python, source_root = installation.root, installation.python, installation.source
    output.parent.mkdir(parents=True, exist_ok=True)

    prepared_dir = output.with_name(f"{output.stem}-inputs")
    prepared_dir.mkdir()
    prepared_keyframes = []
    for index, keyframe in enumerate(normalized_keyframes):
        destination = prepared_dir / f"keyframe-{index:04d}.png"
        with open_image(keyframe.image) as source:
            with source.convert("RGB") as rgb:
                fitted = fit_image_canvas(rgb, (settings.width, settings.height), keyframe_fit)
                fitted.save(destination)
                if fitted is not rgb:
                    fitted.close()
        prepared_keyframes.append(Ltx23Keyframe(destination, keyframe.frame))

    requests = tuple(
        {
            "kind": "aigen-ltx23-keyframes-job",
            "prompt": prompt.strip(),
            "keyframes": [keyframe.to_json() for keyframe in prepared_keyframes],
            "original_keyframes": [keyframe.to_json() for keyframe in normalized_keyframes],
            "keyframe_fit": keyframe_fit,
            "requested_resolution": requested_resolution,
            "output": job_output.as_posix(),
            "resolution": resolution,
            "frames": frames,
            "fps": fps,
            "steps": steps,
            "phases": phases,
            "solver": solver,
            "negative_prompt": negative_prompt.strip(),
            "conditioning_strength": conditioning_strength,
            "model_type": model_type,
            "seed": seed,
        }
        for seed, job_output in zip(normalized_seeds, outputs, strict=True)
    )
    started = time.monotonic()
    results = []

    def completed(index: int, response: dict[str, Any]) -> None:
        request, job_output, config, seed = requests[index], outputs[index], configs[index], normalized_seeds[index]
        if not job_output.is_file() or job_output.stat().st_size == 0:
            raise Ltx23KeyframesError(f"LTX-2.3 did not create a video: {job_output}")
        elapsed_seconds = (
            time.monotonic() - started
            if len(requests) == 1
            else float(response["environment"]["elapsed_seconds"])
        )
        config_payload = {
            "kind": "aigen-ltx23-keyframes-config",
            "runtime": "WanGP",
            "runtime_revision": LTX23_WANGP_REVISION,
            "installation_provenance": installation.provenance,
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
                phases=phases,
                solver=solver,
                negative_prompt=negative_prompt.strip(),
                conditioning_strength=conditioning_strength,
                model=model,
                model_type=model_type,
                seed=seed,
                elapsed_seconds=elapsed_seconds,
                phase_metrics=tuple(response["phase_metrics"]),
                environment=dict(response["environment"]),
                keyframe_fit=keyframe_fit,
            )
        )
        if on_output is not None:
            on_output(results[-1])

    _run_worker_requests(
        requests=requests,
        runtime_root=runtime_root,
        runtime_python=runtime_python,
        source_root=source_root,
        log=log,
        progress=progress,
        on_output=completed,
    )
    progress.phase("LTX-2.3 generation completed")
    return tuple(results)


def _validate_request(
    *, prompt: str, keyframes: Sequence[Ltx23Keyframe], output: Path,
    frames: int, negative_prompt: str,
) -> tuple[Ltx23Keyframe, ...]:
    if not prompt.strip():
        raise Ltx23KeyframesError("video motion prompt must not be empty")
    if not negative_prompt.strip():
        raise Ltx23KeyframesError("video negative prompt must not be empty")
    if output.suffix.lower() != ".mp4":
        raise Ltx23KeyframesError("LTX-2.3 output must use the .mp4 extension")
    validate_ltx23_positions(tuple(keyframe.frame for keyframe in keyframes), frames)
    normalized = tuple(sorted(
        (Ltx23Keyframe(keyframe.image.expanduser().resolve(), keyframe.frame) for keyframe in keyframes),
        key=lambda keyframe: keyframe.frame,
    ))
    missing = next((keyframe.image for keyframe in normalized if not keyframe.image.is_file()), None)
    if missing is not None:
        raise Ltx23KeyframesError(f"keyframe image does not exist: {missing}")
    return normalized


def _run_worker_requests(
    *,
    requests: Sequence[dict[str, Any]],
    runtime_root: Path,
    runtime_python: Path,
    source_root: Path,
    log: Path,
    progress: StatusReporter,
    on_output: Callable[[int, dict[str, Any]], None],
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
                        index = len(responses)
                        if response.get("status") == "completed":
                            if index >= len(requests) or response["output"] != requests[index]["output"]:
                                raise Ltx23KeyframesError("LTX-2.3 worker returned results out of order")
                            on_output(index, response)
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
