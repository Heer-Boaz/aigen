from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.image_dimensions import pixel_area_canvas_size
from aigen.progress import StatusReporter
from aigen.runtime_profiles import MODELS_ROOT, PROJECT_ROOT


USO_SOURCE_REVISION = "6587514aa3adf8e8f46e5f7e804239651d30b32d"
USO_MODEL_TYPE = "flux-dev-fp8"
USO_CONTENT_REFERENCE_SIZE = 512
USO_MAX_REFERENCES = 3


class UsoFlux1Error(RuntimeError):
    pass


def uso_flux1_recommended_canvas_size(
    aspect_ratio: tuple[int, int],
) -> tuple[int, int]:
    return pixel_area_canvas_size(
        aspect_ratio,
        target_pixels=1024 * 1024,
        alignment=16,
    )


@dataclass(frozen=True)
class UsoFlux1Result:
    output: Path
    config: Path
    log: Path
    seed: int
    width: int
    height: int
    steps: int
    guidance: float
    content_reference: Path
    style_references: tuple[Path, ...]
    elapsed_seconds: float
    environment: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "output": self.output.as_posix(),
            "output_bytes": self.output.stat().st_size,
            "config": self.config.as_posix(),
            "log": self.log.as_posix(),
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance": self.guidance,
            "content_reference": self.content_reference.as_posix(),
            "style_references": [
                reference.as_posix() for reference in self.style_references
            ],
            "runtime": "official ByteDance USO",
            "runtime_revision": USO_SOURCE_REVISION,
            "model_type": USO_MODEL_TYPE,
            "quantization": "on-load FP8 E4M3",
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "environment": self.environment,
        }


def generate_uso_flux1_seed_sweep(
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
) -> tuple[UsoFlux1Result, ...]:
    references = _validate_request(
        references=references,
        output=output,
        width=width,
        height=height,
        seeds=seeds,
        steps=steps,
        guidance=guidance,
    )
    seeds = tuple(seeds)
    output = output.expanduser().resolve()
    outputs = tuple(
        output
        if len(seeds) == 1
        else output.with_name(f"{output.stem}-seed{seed}{output.suffix}")
        for seed in seeds
    )
    configs = tuple(path.with_name(f"{path.stem}_config.json") for path in outputs)
    log = output.with_name(f"{output.stem}-uso.log")
    artifacts = (log, *(path for pair in zip(outputs, configs) for path in pair))
    existing = next((path for path in artifacts if path.exists()), None)
    if existing is not None:
        raise UsoFlux1Error(f"output already exists: {existing}")

    runtime_root = _runtime_root()
    source_root = runtime_root / "USO"
    models_root = _models_root() / "uso"
    _validate_runtime(source_root, models_root)
    output.parent.mkdir(parents=True, exist_ok=True)

    requests = tuple(
        {
            "prompt": prompt.strip(),
            "references": [reference.as_posix() for reference in references],
            "output": target.as_posix(),
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
        }
        for seed, target in zip(seeds, outputs, strict=True)
    )
    sweep_started = time.monotonic()
    responses = _run_worker(
        requests=requests,
        runtime_root=runtime_root,
        source_root=source_root,
        models_root=models_root,
        log=log,
        progress=progress,
    )
    sweep_elapsed = time.monotonic() - sweep_started

    results = []
    for request, response, target, config in zip(
        requests, responses, outputs, configs, strict=True
    ):
        if not target.is_file() or target.stat().st_size == 0:
            raise UsoFlux1Error(f"USO did not create an image: {target}")
        config.write_text(
            json.dumps(
                {
                    "kind": "aigen-uso-flux1-dev-fp8-config",
                    "runtime_revision": USO_SOURCE_REVISION,
                    "request": request,
                    "environment": response["environment"],
                    "seed_elapsed_seconds": response["elapsed_seconds"],
                    "sweep_elapsed_seconds": round(sweep_elapsed, 3),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        results.append(
            UsoFlux1Result(
                output=target,
                config=config,
                log=log,
                seed=request["seed"],
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                content_reference=references[0],
                style_references=references[1:],
                elapsed_seconds=float(response["elapsed_seconds"]),
                environment=dict(response["environment"]),
            )
        )
    return tuple(results)


def _validate_request(
    *,
    references: Sequence[Path],
    output: Path,
    width: int,
    height: int,
    seeds: Sequence[int],
    steps: int,
    guidance: float,
) -> tuple[Path, ...]:
    if output.suffix.casefold() != ".png":
        raise UsoFlux1Error("USO output must use the .png extension")
    references = tuple(Path(path).expanduser().resolve() for path in references)
    if not 1 <= len(references) <= USO_MAX_REFERENCES:
        raise UsoFlux1Error(
            "USO requires one content image and accepts at most two style images"
        )
    missing = next((path for path in references if not path.is_file()), None)
    if missing is not None:
        raise UsoFlux1Error(f"reference image does not exist: {missing}")
    seeds = tuple(seeds)
    if not seeds:
        raise UsoFlux1Error("USO requires at least one seed")
    if len(set(seeds)) != len(seeds):
        raise UsoFlux1Error("USO seed sweep contains duplicate seeds")
    if (
        width < 512
        or width > 1536
        or height < 512
        or height > 1536
        or width % 16
        or height % 16
    ):
        raise UsoFlux1Error(
            "USO dimensions must be multiples of 16 between 512 and 1536"
        )
    if not 1 <= steps <= 50:
        raise UsoFlux1Error("USO steps must be between 1 and 50")
    if not 1.0 <= guidance <= 5.0:
        raise UsoFlux1Error("USO guidance must be between 1 and 5")
    return references


def _runtime_root() -> Path:
    configured = os.environ.get("AIGEN_USO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".local/share/aigen/runtimes/uso"


def _models_root() -> Path:
    configured = os.environ.get("AIGEN_MODELS_ROOT")
    return Path(configured).expanduser().resolve() if configured else MODELS_ROOT


def _validate_runtime(source_root: Path, models_root: Path) -> None:
    required = (
        source_root / "uso/flux/pipeline.py",
        models_root / "black-forest-labs/FLUX.1-dev/flux1-dev.safetensors",
        models_root / "black-forest-labs/FLUX.1-dev/ae.safetensors",
        models_root / "bytedance-research/USO/uso_flux_v1.0/dit_lora.safetensors",
        models_root / "bytedance-research/USO/uso_flux_v1.0/projector.safetensors",
        models_root / "xlabs-ai/xflux_text_encoders/model.safetensors.index.json",
        models_root / "openai/clip-vit-large-patch14/model.safetensors",
        models_root / "google/siglip-so400m-patch14-384/model.safetensors",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise UsoFlux1Error(
            "USO runtime is incomplete: " + ", ".join(path.as_posix() for path in missing)
        )
    revision = subprocess.run(
        ["git", "-C", source_root.as_posix(), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode != 0 or revision.stdout.strip() != USO_SOURCE_REVISION:
        raise UsoFlux1Error(f"USO source must be pinned to {USO_SOURCE_REVISION}")


def _run_worker(
    *,
    requests: Sequence[dict[str, Any]],
    runtime_root: Path,
    source_root: Path,
    models_root: Path,
    log: Path,
    progress: StatusReporter,
) -> tuple[dict[str, Any], ...]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (
            PROJECT_ROOT.as_posix(),
            source_root.as_posix(),
            environment.get("PYTHONPATH"),
        )
        if path
    )
    flux_root = models_root / "black-forest-labs/FLUX.1-dev"
    uso_root = models_root / "bytedance-research/USO/uso_flux_v1.0"
    environment.update(
        AIGEN_USO_ROOT=runtime_root.as_posix(),
        FLUX_DEV_FP8=(flux_root / "flux1-dev.safetensors").as_posix(),
        AE=(flux_root / "ae.safetensors").as_posix(),
        T5=(models_root / "xlabs-ai/xflux_text_encoders").as_posix(),
        CLIP=(models_root / "openai/clip-vit-large-patch14").as_posix(),
        LORA=(uso_root / "dit_lora.safetensors").as_posix(),
        PROJECTION_MODEL=(uso_root / "projector.safetensors").as_posix(),
        SIGLIP_PATH=(
            models_root / "google/siglip-so400m-patch14-384"
        ).as_posix(),
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    responses: list[dict[str, Any]] = []
    progress.phase("starting USO worker")
    with log.open("w", encoding="utf-8") as worker_log:
        with subprocess.Popen(
            [sys.executable, "-m", "aigen.generation.uso_flux1_worker"],
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
                assert worker.stdin is not None and worker.stdout is not None
                for request in requests:
                    worker.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                worker.stdin.close()
                for line in worker.stdout:
                    event = json.loads(line)
                    if event["kind"] == "phase":
                        progress.phase(event["text"])
                    elif event["kind"] == "begin":
                        progress.begin(event["total"], event["text"])
                    elif event["kind"] == "step":
                        progress.step(event["text"])
                    elif event["kind"] == "result":
                        responses.append(event["response"])
                    else:
                        raise UsoFlux1Error(
                            f"unknown USO worker event: {event['kind']}"
                        )
                returncode = worker.wait()
            except BaseException:
                if worker.poll() is None:
                    worker.terminate()
                raise
    failed = next(
        (response for response in responses if response.get("status") != "completed"),
        None,
    )
    if returncode != 0 or failed is not None:
        message = failed.get("message") if failed is not None else _log_tail(log)
        raise UsoFlux1Error(f"USO generation failed: {message}")
    if len(responses) != len(requests):
        raise UsoFlux1Error(
            f"USO worker returned {len(responses)} of {len(requests)} results"
        )
    return tuple(responses)


def _log_tail(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text[-8192:] or "worker failed without an error message"
