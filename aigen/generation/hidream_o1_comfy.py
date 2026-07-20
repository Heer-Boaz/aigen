from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.generation.image_generation_requests import (
    ImageGenerationCaseRequest,
    ImageGenerationOutputRequest,
)
from aigen.image_edit_defaults import (
    HIDREAM_DEFAULT_GUIDANCE,
    HIDREAM_DEFAULT_SAMPLER,
    HIDREAM_DEFAULT_SCHEDULER,
    HIDREAM_DEFAULT_STEPS,
    HIDREAM_SAMPLERS,
    HIDREAM_SCHEDULERS,
)
from aigen.image_dimensions import closest_aspect_match
from aigen.progress import StatusReporter
from aigen.runtime_profiles import MODELS_ROOT, PROJECT_ROOT


COMFY_REVISION = "26515acd23fa291a8f5ab53c5997258598de0701"
HIDREAM_CHECKPOINT_REVISION = "54d16b20496bbd1bdfa6f79ec1ad2d6f0bfd2dcc"
HIDREAM_CHECKPOINT = "hidream_o1_image_fp8_scaled.safetensors"
HIDREAM_DEFAULT_RESOLUTION = "2048x2048"
HIDREAM_NOISE_SCALE = 8.0
HIDREAM_NATIVE_CANVASES = (
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
)


class HiDreamO1Error(RuntimeError):
    pass


def hidream_o1_native_canvas_size(aspect_ratio: tuple[int, int]) -> tuple[int, int]:
    return closest_aspect_match(aspect_ratio, HIDREAM_NATIVE_CANVASES)


@dataclass(frozen=True)
class HiDreamO1Result:
    output: Path
    config: Path
    log: Path
    seed: int
    width: int
    height: int
    steps: int
    guidance: float
    sampler: str
    scheduler: str
    elapsed_seconds: float
    environment: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "completed",
            "kind": "hidream-o1-full-fp8-image-edit",
            "output": self.output.as_posix(),
            "output_bytes": self.output.stat().st_size,
            "config": self.config.as_posix(),
            "log": self.log.as_posix(),
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance": self.guidance,
            "sampler": self.sampler,
            "scheduler": self.scheduler,
            "noise_scale": HIDREAM_NOISE_SCALE,
            "runtime": "ComfyUI native HiDream-O1",
            "runtime_revision": COMFY_REVISION,
            "checkpoint_revision": HIDREAM_CHECKPOINT_REVISION,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "environment": self.environment,
        }


def generate_hidream_o1_seed_sweep(
    *,
    prompt: str,
    references: Sequence[Path],
    output: Path,
    width: int,
    height: int,
    seeds: Sequence[int],
    steps: int,
    guidance: float,
    sampler: str,
    scheduler: str,
    progress: StatusReporter,
) -> tuple[HiDreamO1Result, ...]:
    normalized_seeds = tuple(seeds)
    normalized_references = _validate_request(
        prompt=prompt,
        references=references,
        output=output,
        width=width,
        height=height,
        seeds=normalized_seeds,
        steps=steps,
        guidance=guidance,
        sampler=sampler,
        scheduler=scheduler,
    )
    output = output.expanduser().resolve()
    outputs = tuple(
        output
        if len(normalized_seeds) == 1
        else output.with_name(f"{output.stem}-seed{seed}{output.suffix}")
        for seed in normalized_seeds
    )
    configs = tuple(path.with_name(f"{path.stem}_config.json") for path in outputs)
    log = (
        output.with_suffix(f"{output.suffix}.log")
        if len(normalized_seeds) == 1
        else output.with_name(f"{output.stem}-seed-sweep.log")
    )
    artifacts = (log, *(path for pair in zip(outputs, configs) for path in pair))
    existing = next((path for path in artifacts if path.exists()), None)
    if existing is not None:
        raise HiDreamO1Error(f"output already exists: {existing}")

    runtime_root = _runtime_root()
    runtime_python = runtime_root / "venv/bin/python"
    source_root = runtime_root / "ComfyUI"
    models_root = _models_root()
    _validate_runtime(runtime_python, source_root, models_root)
    output.parent.mkdir(parents=True, exist_ok=True)

    case = ImageGenerationCaseRequest(
        name=output.stem,
        prompt=prompt.strip(),
        image_paths=normalized_references,
        width=width,
        height=height,
        outputs=tuple(
            ImageGenerationOutputRequest(name=f"seed-{seed}", seed=seed, path=path)
            for seed, path in zip(normalized_seeds, outputs, strict=True)
        ),
    )
    requests = tuple(
        {
            "kind": "aigen-hidream-o1-image-edit-job",
            "name": case.name,
            "prompt": case.prompt,
            "references": [path.as_posix() for path in case.image_paths],
            "output": generation.path.as_posix(),
            "width": case.width,
            "height": case.height,
            "steps": steps,
            "guidance": guidance,
            "sampler": sampler,
            "scheduler": scheduler,
            "noise_scale": HIDREAM_NOISE_SCALE,
            "seed": generation.seed,
        }
        for generation in case.outputs
    )

    sweep_started = time.monotonic()
    results = []
    for request, image_output, config, seed in zip(
        requests,
        outputs,
        configs,
        normalized_seeds,
        strict=True,
    ):
        response = _run_worker(
            request=request,
            runtime_root=runtime_root,
            runtime_python=runtime_python,
            source_root=source_root,
            models_root=models_root,
            log=log,
            progress=progress,
        )
        if not image_output.is_file() or image_output.stat().st_size == 0:
            raise HiDreamO1Error(f"HiDream-O1 did not create an image: {image_output}")
        config_payload = {
            "kind": "aigen-hidream-o1-full-fp8-config",
            "runtime": "ComfyUI native HiDream-O1",
            "runtime_revision": COMFY_REVISION,
            "checkpoint_revision": HIDREAM_CHECKPOINT_REVISION,
            "request": request,
            "environment": response["environment"],
            "sweep_elapsed_seconds": round(time.monotonic() - sweep_started, 3),
        }
        config.write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(
            HiDreamO1Result(
                output=image_output,
                config=config,
                log=log,
                seed=seed,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                sampler=sampler,
                scheduler=scheduler,
                elapsed_seconds=float(response["environment"]["elapsed_seconds"]),
                environment=dict(response["environment"]),
            )
        )
    progress.phase("HiDream-O1 generation completed")
    return tuple(results)


def _validate_request(
    *,
    prompt: str,
    references: Sequence[Path],
    output: Path,
    width: int,
    height: int,
    seeds: tuple[int, ...],
    steps: int,
    guidance: float,
    sampler: str,
    scheduler: str,
) -> tuple[Path, ...]:
    if not prompt.strip():
        raise HiDreamO1Error("image edit prompt must not be empty")
    if output.suffix.lower() != ".png":
        raise HiDreamO1Error("HiDream-O1 output must use the .png extension")
    if not 1 <= len(references) <= 10:
        raise HiDreamO1Error("HiDream-O1 image editing requires 1 to 10 reference images")
    if not seeds:
        raise HiDreamO1Error("HiDream-O1 seed sweep requires at least one seed")
    if len(set(seeds)) != len(seeds):
        raise HiDreamO1Error("HiDream-O1 seed sweep contains duplicate seeds")
    if not 512 <= width <= 3104 or width % 32:
        raise HiDreamO1Error("HiDream-O1 width must be a multiple of 32 from 512 to 3104")
    if not 512 <= height <= 3104 or height % 32:
        raise HiDreamO1Error("HiDream-O1 height must be a multiple of 32 from 512 to 3104")
    if steps <= 0:
        raise HiDreamO1Error("HiDream-O1 inference steps must be positive")
    if guidance < 0:
        raise HiDreamO1Error("HiDream-O1 guidance must not be negative")
    if sampler not in HIDREAM_SAMPLERS:
        raise HiDreamO1Error(f"unsupported HiDream-O1 sampler: {sampler}")
    if scheduler not in HIDREAM_SCHEDULERS:
        raise HiDreamO1Error(f"unsupported HiDream-O1 scheduler: {scheduler}")

    normalized = tuple(path.expanduser().resolve() for path in references)
    missing = next((path for path in normalized if not path.is_file()), None)
    if missing is not None:
        raise HiDreamO1Error(f"reference image does not exist: {missing}")
    return normalized


def _runtime_root() -> Path:
    configured = os.environ.get("AIGEN_COMFY_IMAGE_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path.home() / ".cache/aigen-comfy-image"
    )


def _models_root() -> Path:
    configured = os.environ.get("AIGEN_MODELS_ROOT")
    return Path(configured).expanduser().resolve() if configured else MODELS_ROOT


def _validate_runtime(runtime_python: Path, source_root: Path, models_root: Path) -> None:
    required = (
        runtime_python,
        source_root / "nodes.py",
        source_root / "comfy_extras/nodes_hidream_o1.py",
        models_root / f"comfy/checkpoints/{HIDREAM_CHECKPOINT}",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise HiDreamO1Error(
            "HiDream-O1 runtime is incomplete; run scripts/install_comfy_image_runtime.sh "
            "and scripts/download_hidream_o1.sh: "
            + ", ".join(path.as_posix() for path in missing)
        )
    revision = subprocess.run(
        ["git", "-C", source_root.as_posix(), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0 or revision.stdout.strip() != COMFY_REVISION:
        raise HiDreamO1Error(f"ComfyUI source must be pinned to {COMFY_REVISION}")


def _run_worker(
    *,
    request: dict[str, Any],
    runtime_root: Path,
    runtime_python: Path,
    source_root: Path,
    models_root: Path,
    log: Path,
    progress: StatusReporter,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    environment.pop("PYTORCH_ALLOC_CONF", None)
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (PROJECT_ROOT.as_posix(), source_root.as_posix(), python_path)
        if path
    )
    environment.update(
        AIGEN_COMFY_IMAGE_ROOT=runtime_root.as_posix(),
        AIGEN_COMFY_MODELS_ROOT=(models_root / "comfy").as_posix(),
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )

    responses = []
    progress.phase(f"starting native HiDream-O1 worker for seed {request['seed']}")
    with log.open("a", encoding="utf-8") as worker_log:
        worker_log.write(f"\n=== HiDream-O1 seed {request['seed']} ===\n")
        worker_log.flush()
        with subprocess.Popen(
            [runtime_python.as_posix(), "-m", "aigen.generation.hidream_o1_comfy_worker"],
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
                    raise HiDreamO1Error("HiDream-O1 worker pipes are unavailable")
                worker.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
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

    failed = next((response for response in responses if response.get("status") != "completed"), None)
    if returncode != 0 or failed is not None:
        message = failed.get("message") if failed is not None else _log_tail(log)
        raise HiDreamO1Error(f"native HiDream-O1 failed: {message}")
    if len(responses) != 1:
        raise HiDreamO1Error(
            f"HiDream-O1 worker returned {len(responses)} results for one request"
        )
    return responses[0]


def _apply_worker_event(line: str, progress: StatusReporter) -> dict[str, Any] | None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as error:
        raise HiDreamO1Error("invalid HiDream-O1 worker event") from error
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
            raise HiDreamO1Error(f"unknown HiDream-O1 worker event: {kind}")
    return None


def _log_tail(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as error:
        return f"unable to read worker log: {error}"
    return text[-8192:] or "worker failed without an error message"
