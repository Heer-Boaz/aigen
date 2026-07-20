from __future__ import annotations

import json
import math
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
from aigen.image_dimensions import closest_aspect_match
from aigen.progress import StatusReporter
from aigen.runtime_profiles import MODELS_ROOT, PROJECT_ROOT


BOOGU_SOURCE_REVISION = "29c040ff975d19231911753a0dbf976ae98621b1"
BOOGU_MODEL_REVISION = "5f608c1680cb91ca88af88010c91e50f12c41d28"
BOOGU_MODEL_DIRECTORY = "boogu/Boogu-Image-0.1-Edit-Turbo-fp8"
BOOGU_DEFAULT_RESOLUTION = "1024x1024"
BOOGU_DEFAULT_STEPS = 4
BOOGU_DEFAULT_GUIDANCE = 1.0
BOOGU_MAX_SIDE = 2048
BOOGU_RECOMMENDED_1K_PIXEL_AREA = 1024 * 1024
BOOGU_DIMENSION_ALIGNMENT = 16
BOOGU_SUPPORTED_ASPECT_RATIOS = (
    (1, 1),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (1, 2),
    (2, 1),
    (9, 16),
    (16, 9),
)


class BooguImageEditError(RuntimeError):
    pass


def boogu_recommended_1k_canvas_size(
    aspect_ratio: tuple[int, int],
    *,
    closest: bool,
) -> tuple[int, int]:
    selected_ratio = aspect_ratio
    if selected_ratio not in BOOGU_SUPPORTED_ASPECT_RATIOS:
        if closest:
            selected_ratio = closest_aspect_match(
                selected_ratio,
                BOOGU_SUPPORTED_ASPECT_RATIOS,
            )
        else:
            supported = ", ".join(
                f"{width}:{height}" for width, height in BOOGU_SUPPORTED_ASPECT_RATIOS
            )
            raise BooguImageEditError(
                f"Boogu-Image Edit Turbo does not support aspect ratio "
                f"{aspect_ratio[0]}:{aspect_ratio[1]}; use --width/--height or one of: "
                f"{supported}"
            )

    ratio_width, ratio_height = selected_ratio
    scale = math.sqrt(
        BOOGU_RECOMMENDED_1K_PIXEL_AREA / (ratio_width * ratio_height)
    )
    width = (
        int(ratio_width * scale)
        // BOOGU_DIMENSION_ALIGNMENT
        * BOOGU_DIMENSION_ALIGNMENT
    )
    height = (
        int(ratio_height * scale)
        // BOOGU_DIMENSION_ALIGNMENT
        * BOOGU_DIMENSION_ALIGNMENT
    )
    return width, height


@dataclass(frozen=True)
class BooguImageEditResult:
    output: Path
    config: Path
    log: Path
    seed: int
    width: int
    height: int
    elapsed_seconds: float
    environment: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "completed",
            "kind": "boogu-image-edit-turbo-fp8",
            "output": self.output.as_posix(),
            "output_bytes": self.output.stat().st_size,
            "config": self.config.as_posix(),
            "log": self.log.as_posix(),
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "steps": BOOGU_DEFAULT_STEPS,
            "text_guidance": BOOGU_DEFAULT_GUIDANCE,
            "image_guidance": BOOGU_DEFAULT_GUIDANCE,
            "runtime": "official Boogu-Image Turbo FP8",
            "runtime_revision": BOOGU_SOURCE_REVISION,
            "model_revision": BOOGU_MODEL_REVISION,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "environment": self.environment,
        }


def generate_boogu_image_edit_seed_sweep(
    *,
    prompt: str,
    references: Sequence[Path],
    output: Path,
    width: int,
    height: int,
    seeds: Sequence[int],
    steps: int,
    guidance: float,
    progress: StatusReporter,
) -> tuple[BooguImageEditResult, ...]:
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
        raise BooguImageEditError(f"output already exists: {existing}")

    runtime_root = _runtime_root()
    runtime_python = runtime_root / "venv/bin/python"
    source_root = runtime_root / "Boogu-Image"
    model_root = _models_root() / BOOGU_MODEL_DIRECTORY
    _validate_runtime(runtime_python, source_root, model_root)
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
    request = {
        "kind": "aigen-boogu-image-edit-seed-sweep",
        "name": case.name,
        "prompt": case.prompt,
        "reference": case.image_paths[0].as_posix(),
        "width": case.width,
        "height": case.height,
        "steps": BOOGU_DEFAULT_STEPS,
        "outputs": [
            {"seed": generation.seed, "path": generation.path.as_posix()}
            for generation in case.outputs
        ],
    }
    sweep_started = time.monotonic()
    response = _run_worker(
        request=request,
        runtime_root=runtime_root,
        runtime_python=runtime_python,
        source_root=source_root,
        model_root=model_root,
        log=log,
        progress=progress,
    )
    environment = dict(response["environment"])
    timings = {item["seed"]: float(item["elapsed_seconds"]) for item in response["outputs"]}
    results = []
    for image_output, config, seed in zip(outputs, configs, normalized_seeds, strict=True):
        if not image_output.is_file() or image_output.stat().st_size == 0:
            raise BooguImageEditError(f"Boogu-Image did not create an image: {image_output}")
        config_payload = {
            "kind": "aigen-boogu-image-edit-turbo-fp8-config",
            "runtime": "official Boogu-Image Turbo FP8",
            "runtime_revision": BOOGU_SOURCE_REVISION,
            "model_revision": BOOGU_MODEL_REVISION,
            "request": {**request, "outputs": [{"seed": seed, "path": image_output.as_posix()}]},
            "environment": environment,
            "seed_elapsed_seconds": round(timings[seed], 3),
            "sweep_elapsed_seconds": round(time.monotonic() - sweep_started, 3),
        }
        config.write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(
            BooguImageEditResult(
                output=image_output,
                config=config,
                log=log,
                seed=seed,
                width=width,
                height=height,
                elapsed_seconds=timings[seed],
                environment=environment,
            )
        )
    progress.phase("Boogu-Image generation completed")
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
) -> tuple[Path, ...]:
    if not prompt.strip():
        raise BooguImageEditError("image edit prompt must not be empty")
    if output.suffix.lower() != ".png":
        raise BooguImageEditError("Boogu-Image output must use the .png extension")
    if len(references) != 1:
        raise BooguImageEditError("Boogu-Image Edit Turbo requires exactly one reference image")
    if not seeds:
        raise BooguImageEditError("Boogu-Image seed sweep requires at least one seed")
    if len(set(seeds)) != len(seeds):
        raise BooguImageEditError("Boogu-Image seed sweep contains duplicate seeds")
    if width <= 0 or width % 16 or height <= 0 or height % 16:
        raise BooguImageEditError("Boogu-Image dimensions must be positive multiples of 16")
    if width > BOOGU_MAX_SIDE or height > BOOGU_MAX_SIDE:
        raise BooguImageEditError(
            f"Boogu-Image dimensions must not exceed {BOOGU_MAX_SIDE}px per side"
        )
    if steps != BOOGU_DEFAULT_STEPS:
        raise BooguImageEditError("Boogu-Image Edit Turbo uses its official 4-step DMD schedule")
    if guidance != BOOGU_DEFAULT_GUIDANCE:
        raise BooguImageEditError("Boogu-Image Edit Turbo requires guidance 1")

    normalized = tuple(path.expanduser().resolve() for path in references)
    if not normalized[0].is_file():
        raise BooguImageEditError(f"reference image does not exist: {normalized[0]}")
    return normalized


def _runtime_root() -> Path:
    configured = os.environ.get("AIGEN_BOOGU_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path.home() / ".cache/aigen-boogu"
    )


def _models_root() -> Path:
    configured = os.environ.get("AIGEN_MODELS_ROOT")
    return Path(configured).expanduser().resolve() if configured else MODELS_ROOT


def _validate_runtime(runtime_python: Path, source_root: Path, model_root: Path) -> None:
    required = (
        runtime_python,
        source_root / "boogu/pipelines/boogu/pipeline_boogu_turbo.py",
        model_root / "model_index.json",
        model_root / "mllm/model-00001-of-00002.safetensors",
        model_root / "transformer/diffusion_pytorch_model-00001-of-00002.bin",
        model_root / "vae/diffusion_pytorch_model.safetensors",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise BooguImageEditError(
            "Boogu-Image runtime is incomplete; run scripts/install_boogu_image_runtime.sh "
            "and scripts/download_boogu_image.sh: "
            + ", ".join(path.as_posix() for path in missing)
        )
    revision = subprocess.run(
        ["git", "-C", source_root.as_posix(), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0 or revision.stdout.strip() != BOOGU_SOURCE_REVISION:
        raise BooguImageEditError(f"Boogu-Image source must be pinned to {BOOGU_SOURCE_REVISION}")


def _run_worker(
    *,
    request: dict[str, Any],
    runtime_root: Path,
    runtime_python: Path,
    source_root: Path,
    model_root: Path,
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
        AIGEN_BOOGU_ROOT=runtime_root.as_posix(),
        AIGEN_BOOGU_MODEL_ROOT=model_root.as_posix(),
        TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR="1",
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        device="cuda:0",
    )

    responses = []
    progress.phase("starting official Boogu-Image seed-sweep worker")
    with log.open("w", encoding="utf-8") as worker_log:
        with subprocess.Popen(
            [runtime_python.as_posix(), "-m", "aigen.generation.boogu_image_edit_worker"],
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
                    raise BooguImageEditError("Boogu-Image worker pipes are unavailable")
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
        raise BooguImageEditError(f"official Boogu-Image failed: {message}")
    if len(responses) != 1:
        raise BooguImageEditError(f"Boogu-Image worker returned {len(responses)} results")
    return responses[0]


def _apply_worker_event(line: str, progress: StatusReporter) -> dict[str, Any] | None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as error:
        raise BooguImageEditError("invalid Boogu-Image worker event") from error
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
            raise BooguImageEditError(f"unknown Boogu-Image worker event: {kind}")
    return None


def _log_tail(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as error:
        return f"unable to read worker log: {error}"
    return text[-8192:] or "worker failed without an error message"
