from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.progress import StatusReporter
from aigen.runtime_profiles import MODELS_ROOT, PROJECT_ROOT


HUNYUANVIDEO15_SOURCE_REVISION = "60783e704160023913bee78f0b47036d393d4dfa"
HUNYUANVIDEO15_MODEL_REVISION = "9b49404b3f5df2a8f0b31df27a0c7ab872e7b038"
HUNYUANVIDEO15_TRANSFORMER = "480p_i2v_step_distilled"
HUNYUANVIDEO15_STEPS = frozenset({8, 12})
HUNYUANVIDEO15_CFG_SCALE = 1.0
HUNYUANVIDEO15_FLOW_SHIFT = 7.0
HUNYUANVIDEO15_FPS = 24
HUNYUANVIDEO15_RUNTIME_PATCH = (
    PROJECT_ROOT
    / "patches/hunyuanvideo15/0001-release-cuda-cache-after-component-offload.patch"
)


class HunyuanVideo15Error(RuntimeError):
    pass


@dataclass(frozen=True)
class HunyuanVideo15Result:
    output: Path
    config: Path
    log: Path
    image: Path
    model: Path
    steps: int
    frames: int
    seed: int
    overlap_group_offloading: bool
    elapsed_seconds: float

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "completed",
            "kind": "hunyuanvideo-1.5-480p-i2v-step-distilled",
            "output": self.output.as_posix(),
            "output_bytes": self.output.stat().st_size,
            "config": self.config.as_posix(),
            "log": self.log.as_posix(),
            "image": self.image.as_posix(),
            "source_revision": HUNYUANVIDEO15_SOURCE_REVISION,
            "model_revision": HUNYUANVIDEO15_MODEL_REVISION,
            "model": self.model.as_posix(),
            "transformer": HUNYUANVIDEO15_TRANSFORMER,
            "resolution": "480p",
            "steps": self.steps,
            "cfg_scale": HUNYUANVIDEO15_CFG_SCALE,
            "flow_shift": HUNYUANVIDEO15_FLOW_SHIFT,
            "frames": self.frames,
            "fps": HUNYUANVIDEO15_FPS,
            "seed": self.seed,
            "super_resolution": False,
            "prompt_rewriting": False,
            "model_offloading": True,
            "group_offloading": True,
            "overlap_group_offloading": self.overlap_group_offloading,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def generate_hunyuanvideo15_i2v(
    *,
    prompt: str,
    image: Path,
    output: Path,
    steps: int,
    frames: int,
    seed: int,
    overlap_group_offloading: bool,
    progress: StatusReporter,
) -> HunyuanVideo15Result:
    if steps not in HUNYUANVIDEO15_STEPS:
        raise HunyuanVideo15Error("step-distilled inference supports exactly 8 or 12 steps")
    if frames <= 0:
        raise HunyuanVideo15Error("video frame count must be positive")
    if not prompt.strip():
        raise HunyuanVideo15Error("video motion prompt must not be empty")

    image = image.expanduser().resolve()
    output = output.expanduser().resolve()
    if not image.is_file():
        raise HunyuanVideo15Error(f"input image does not exist: {image}")
    if output.suffix.lower() != ".mp4":
        raise HunyuanVideo15Error("HunyuanVideo-1.5 output must use the .mp4 extension")

    config = output.with_name(f"{output.stem}_config.json")
    log = output.with_suffix(f"{output.suffix}.log")
    existing = next((path for path in (output, config) if path.exists()), None)
    if existing is not None:
        raise HunyuanVideo15Error(f"output already exists: {existing}")

    runtime_root = _runtime_root()
    source = runtime_root / "HunyuanVideo-1.5"
    torchrun = runtime_root / "venv/bin/torchrun"
    generate_script = source / "generate.py"
    model = _model_root()
    _validate_runtime(source, torchrun, generate_script)
    _validate_model(model)

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        torchrun.as_posix(),
        "--standalone",
        "--nproc_per_node=1",
        generate_script.as_posix(),
        "--prompt",
        prompt,
        "--image_path",
        image.as_posix(),
        "--output_path",
        output.as_posix(),
        "--model_path",
        model.as_posix(),
        "--resolution",
        "480p",
        "--num_inference_steps",
        str(steps),
        "--video_length",
        str(frames),
        "--seed",
        str(seed),
        "--dtype",
        "bf16",
        "--enable_step_distill",
        "true",
        "--cfg_distilled",
        "false",
        "--sparse_attn",
        "false",
        "--offloading",
        "true",
        "--group_offloading",
        "true",
        "--overlap_group_offloading",
        str(overlap_group_offloading).lower(),
        "--sr",
        "false",
        "--save_pre_sr_video",
        "false",
        "--rewrite",
        "false",
        "--enable_cache",
        "false",
        "--use_sageattn",
        "false",
        "--enable_torch_compile",
        "false",
        "--use_fp8_gemm",
        "false",
        "--save_generation_config",
        "true",
    ]
    environment = os.environ.copy()
    environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    environment.pop("PYTORCH_ALLOC_CONF", None)
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONUNBUFFERED="1",
        TORCH_SHOW_CPP_STACKTRACES="1",
    )

    progress.phase("run official HunyuanVideo-1.5 480p I2V")
    started = time.monotonic()
    try:
        with log.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=source,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    except OSError as error:
        raise HunyuanVideo15Error(f"failed to start HunyuanVideo-1.5: {error}") from error
    elapsed_seconds = time.monotonic() - started
    if completed.returncode != 0:
        details = _log_tail(log)
        raise HunyuanVideo15Error(
            f"official HunyuanVideo-1.5 exited with status {completed.returncode}: {details}"
        )

    _validate_outputs(output, config, expected_arguments={
        "cfg_distilled": False,
        "enable_cache": False,
        "enable_step_distill": True,
        "group_offloading": True,
        "image_path": image.as_posix(),
        "num_inference_steps": steps,
        "offloading": True,
        "output_path": output.as_posix(),
        "overlap_group_offloading": overlap_group_offloading,
        "resolution": "480p",
        "rewrite": False,
        "sparse_attn": False,
        "sr": False,
        "video_length": frames,
    })
    progress.phase("HunyuanVideo-1.5 generation completed")
    return HunyuanVideo15Result(
        output=output,
        config=config,
        log=log,
        image=image,
        model=model,
        steps=steps,
        frames=frames,
        seed=seed,
        overlap_group_offloading=overlap_group_offloading,
        elapsed_seconds=elapsed_seconds,
    )


def _runtime_root() -> Path:
    configured = os.environ.get("AIGEN_HUNYUANVIDEO15_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path.home() / ".cache/aigen-hunyuanvideo15"
    )


def _model_root() -> Path:
    configured = os.environ.get("AIGEN_HUNYUANVIDEO15_MODEL_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    models_root = Path(os.environ.get("AIGEN_MODELS_ROOT", MODELS_ROOT)).expanduser().resolve()
    return models_root / "hunyuanvideo15/tencent/HunyuanVideo-1.5"


def _validate_runtime(source: Path, torchrun: Path, generate_script: Path) -> None:
    missing = [path for path in (torchrun, generate_script) if not path.is_file()]
    if missing:
        raise HunyuanVideo15Error(
            "HunyuanVideo-1.5 runtime is incomplete; run scripts/install_hunyuanvideo15.sh: "
            + ", ".join(path.as_posix() for path in missing)
        )
    revision = subprocess.run(
        ["git", "-C", source.as_posix(), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0 or revision.stdout.strip() != HUNYUANVIDEO15_SOURCE_REVISION:
        raise HunyuanVideo15Error(
            f"HunyuanVideo-1.5 source must be pinned to {HUNYUANVIDEO15_SOURCE_REVISION}"
        )
    changed = subprocess.run(
        [
            "git",
            "-C",
            source.as_posix(),
            "diff",
            "--no-ext-diff",
            "--binary",
            "--unified=0",
        ],
        check=False,
        capture_output=True,
    )
    staged = subprocess.run(
        ["git", "-C", source.as_posix(), "diff", "--cached", "--quiet"],
        check=False,
    )
    if (
        changed.returncode != 0
        or staged.returncode != 0
        or changed.stdout != HUNYUANVIDEO15_RUNTIME_PATCH.read_bytes()
    ):
        raise HunyuanVideo15Error(
            "HunyuanVideo-1.5 source does not contain the required CUDA cache-release patch"
        )


def _validate_model(model: Path) -> None:
    required = (
        model / "config.json",
        model / "scheduler/scheduler_config.json",
        model / f"transformer/{HUNYUANVIDEO15_TRANSFORMER}/config.json",
        model
        / f"transformer/{HUNYUANVIDEO15_TRANSFORMER}/diffusion_pytorch_model.safetensors",
        model / "vae/config.json",
        model / "vae/diffusion_pytorch_model.safetensors",
        model / "text_encoder/llm/model.safetensors.index.json",
        model / "text_encoder/byt5-small/pytorch_model.bin",
        model / "text_encoder/Glyph-SDXL-v2/assets/color_idx.json",
        model / "text_encoder/Glyph-SDXL-v2/assets/multilingual_10-lang_idx.json",
        model / "text_encoder/Glyph-SDXL-v2/checkpoints/byt5_model.pt",
        model / "vision_encoder/siglip/feature_extractor/preprocessor_config.json",
        model / "vision_encoder/siglip/image_encoder/model.safetensors",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise HunyuanVideo15Error(
            "HunyuanVideo-1.5 model set is incomplete; run scripts/download_hunyuanvideo15.sh: "
            + ", ".join(path.as_posix() for path in missing)
        )


def _validate_outputs(
    output: Path,
    config: Path,
    *,
    expected_arguments: dict[str, Any],
) -> None:
    if not output.is_file() or output.stat().st_size == 0:
        raise HunyuanVideo15Error(f"HunyuanVideo-1.5 did not create a video: {output}")
    if not config.is_file():
        raise HunyuanVideo15Error(f"HunyuanVideo-1.5 did not save its generation config: {config}")
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HunyuanVideo15Error(f"invalid HunyuanVideo-1.5 generation config: {error}") from error
    if payload.get("task") != "i2v":
        raise HunyuanVideo15Error(f"unexpected generation task in {config}: {payload.get('task')}")
    if payload.get("transformer_version") != HUNYUANVIDEO15_TRANSFORMER:
        raise HunyuanVideo15Error(
            f"unexpected transformer in {config}: {payload.get('transformer_version')}"
        )
    arguments = payload.get("arguments", {})
    mismatches = {
        key: {"expected": expected, "actual": arguments.get(key)}
        for key, expected in expected_arguments.items()
        if arguments.get(key) != expected
    }
    if mismatches:
        raise HunyuanVideo15Error(f"generation config does not match requested profile: {mismatches}")


def _log_tail(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as error:
        return str(error)
    return text[-8192:] or "no process output"
