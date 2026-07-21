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
from aigen.lora_weights import LoraLoadSpec


FLUX2_DEV_WANGP_REVISION = "5582327dc25e45fec6cda0f27144d4dcf7ed104b"
FLUX2_DEV_MODEL_TYPE = "flux2_dev_nvfp4"


class Flux2DevError(RuntimeError):
    pass


@dataclass(frozen=True)
class Flux2DevResult:
    output: Path
    seed: int
    width: int
    height: int
    steps: int
    guidance: float
    loras: tuple[LoraLoadSpec, ...]
    elapsed_seconds: float
    environment: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "output": self.output.as_posix(),
            "output_bytes": self.output.stat().st_size,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance": self.guidance,
            "loras": [lora.to_json() for lora in self.loras],
            "runtime": "WanGP",
            "runtime_revision": FLUX2_DEV_WANGP_REVISION,
            "model_type": FLUX2_DEV_MODEL_TYPE,
            "quantization": "nvfp4-mixed",
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "environment": self.environment,
        }


def generate_flux2_dev_seed_sweep(
    *,
    prompt: str,
    references: Sequence[Path],
    output: Path,
    width: int,
    height: int,
    seeds: Sequence[int],
    steps: int,
    guidance: float,
    loras: Sequence[LoraLoadSpec],
    progress: StatusReporter,
) -> tuple[Flux2DevResult, ...]:
    prompt = prompt.strip()
    if not prompt:
        raise Flux2DevError("FLUX.2 dev requires a prompt")
    references = tuple(Path(path).expanduser().resolve() for path in references)
    missing = next((path for path in references if not path.is_file()), None)
    if missing is not None:
        raise Flux2DevError(f"reference image does not exist: {missing}")
    seeds = tuple(seeds)
    if not seeds:
        raise Flux2DevError("FLUX.2 dev requires at least one seed")

    output = output.expanduser().resolve()
    outputs = tuple(
        output if len(seeds) == 1 else output.with_name(f"{output.stem}-seed{seed}{output.suffix}")
        for seed in seeds
    )
    existing = next((path for path in outputs if path.exists()), None)
    if existing is not None:
        raise Flux2DevError(f"output already exists: {existing}")
    output.parent.mkdir(parents=True, exist_ok=True)

    runtime_root = _runtime_root()
    source_root = runtime_root / "Wan2GP"
    runtime_python = runtime_root / "venv/bin/python"
    _validate_runtime(runtime_python, source_root)
    requests = tuple(
        {
            "prompt": prompt,
            "references": [path.as_posix() for path in references],
            "output": target.as_posix(),
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "loras": [lora.to_json() for lora in loras],
        }
        for seed, target in zip(seeds, outputs, strict=True)
    )
    log = output.with_name(f"{output.stem}-wangp.log")
    started = time.monotonic()
    responses = _run_worker(
        requests=requests,
        runtime_root=runtime_root,
        runtime_python=runtime_python,
        source_root=source_root,
        log=log,
        progress=progress,
    )
    total_elapsed = time.monotonic() - started
    results = []
    for request, response, target in zip(requests, responses, outputs, strict=True):
        if not target.is_file() or target.stat().st_size == 0:
            raise Flux2DevError(f"FLUX.2 dev did not create an image: {target}")
        elapsed = total_elapsed if len(outputs) == 1 else float(response["elapsed_seconds"])
        results.append(
            Flux2DevResult(
                output=target,
                seed=request["seed"],
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                loras=tuple(loras),
                elapsed_seconds=elapsed,
                environment=response["environment"],
            )
        )
    return tuple(results)


def _runtime_root() -> Path:
    configured = os.environ.get("AIGEN_WANGP_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path.home() / ".cache/aigen-wangp"


def _validate_runtime(runtime_python: Path, source_root: Path) -> None:
    required = (
        runtime_python,
        source_root / "wgp.py",
        source_root / "shared/api.py",
        source_root / "defaults/flux2_dev_nvfp4.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise Flux2DevError("WanGP FLUX.2 dev runtime is incomplete: " + ", ".join(map(str, missing)))
    revision = subprocess.run(
        ["git", "-C", source_root.as_posix(), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode != 0 or revision.stdout.strip() != FLUX2_DEV_WANGP_REVISION:
        raise Flux2DevError(f"WanGP source must be pinned to {FLUX2_DEV_WANGP_REVISION}")


def _run_worker(
    *,
    requests: Sequence[dict[str, Any]],
    runtime_root: Path,
    runtime_python: Path,
    source_root: Path,
    log: Path,
    progress: StatusReporter,
) -> tuple[dict[str, Any], ...]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (PROJECT_ROOT.as_posix(), source_root.as_posix(), environment.get("PYTHONPATH"))
        if path
    )
    environment.update(
        AIGEN_WANGP_ROOT=runtime_root.as_posix(),
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
    )
    responses: list[dict[str, Any]] = []
    progress.phase("starting FLUX.2 dev worker")
    with log.open("w", encoding="utf-8") as worker_log:
        with subprocess.Popen(
            [runtime_python.as_posix(), "-m", "aigen.generation.flux2_dev_wangp_worker"],
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
                        raise Flux2DevError(f"unknown FLUX.2 dev worker event: {event['kind']}")
                returncode = worker.wait()
            except BaseException:
                if worker.poll() is None:
                    worker.terminate()
                raise
    failed = next((response for response in responses if response.get("status") != "completed"), None)
    if returncode != 0 or failed is not None:
        message = failed.get("message") if failed is not None else _log_tail(log)
        raise Flux2DevError(f"FLUX.2 dev WanGP failed: {message}")
    if len(responses) != len(requests):
        raise Flux2DevError(f"FLUX.2 dev worker returned {len(responses)} of {len(requests)} results")
    return tuple(responses)


def _log_tail(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text[-8192:] or "worker failed without an error message"
