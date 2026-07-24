from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.generation.runtime_diagnostics import cuda_memory_stats
from aigen.progress import StatusReporter
from aigen.runtime_profiles import MODELS_ROOT


ANIMEGEN_MODEL_REVISION = "6278659f803518d72dd312a1f522e3b34b1afd72"
ANIMEGEN_BASE_MODEL_REVISION = "596658fd9ca6b7b71d5057529bbf319ecbc61d74"
ANIMEGEN_LIGHTNING_REVISION = "18bccf8884ec0a078eed79785eb4ef13ea16ce1e"
ANIMEGEN_DEFAULT_PRECISION = "fp8"
ANIMEGEN_PRECISIONS = ("fp8", "bf16")
ANIMEGEN_DEFAULT_FRAMES = 81
ANIMEGEN_DEFAULT_FPS = 16
ANIMEGEN_MAX_AREA = 832 * 480


@dataclass(frozen=True)
class AnimeGenSamplingProfile:
    steps: int
    guidance: float
    flow_shift: float
    lightning: bool


ANIMEGEN_SAMPLING_PROFILES = {
    "lightning-4": AnimeGenSamplingProfile(
        steps=4,
        guidance=1.0,
        flow_shift=3.0,
        lightning=True,
    ),
    "lightning-8": AnimeGenSamplingProfile(
        steps=8,
        guidance=1.0,
        flow_shift=3.0,
        lightning=True,
    ),
    "lightning-16": AnimeGenSamplingProfile(
        steps=16,
        guidance=1.0,
        flow_shift=3.0,
        lightning=True,
    ),
    "full-40": AnimeGenSamplingProfile(
        steps=40,
        guidance=3.5,
        flow_shift=5.0,
        lightning=False,
    ),
}
ANIMEGEN_SAMPLINGS = tuple(ANIMEGEN_SAMPLING_PROFILES)
ANIMEGEN_DEFAULT_SAMPLING = "lightning-8"


class AnimeGenI2VError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnimeGenI2VResult:
    output: Path
    config: Path
    image: Path
    last_image: Path | None
    width: int
    height: int
    frames: int
    fps: int
    sampling: str
    steps: int
    guidance: float
    flow_shift: float
    scheduler: str
    precision: str
    seed: int
    elapsed_seconds: float
    cuda_memory: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "completed",
            "kind": f"animegen-i2v-{self.sampling}",
            "output": self.output.as_posix(),
            "output_bytes": self.output.stat().st_size,
            "config": self.config.as_posix(),
            "image": self.image.as_posix(),
            "last_image": self.last_image.as_posix() if self.last_image is not None else None,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "fps": self.fps,
            "sampling": self.sampling,
            "steps": self.steps,
            "guidance": self.guidance,
            "flow_shift": self.flow_shift,
            "scheduler": self.scheduler,
            "precision": self.precision,
            "seed": self.seed,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "cuda_memory": self.cuda_memory,
        }


def generate_animegen_i2v(
    *,
    prompt: str,
    image: Path,
    last_image: Path | None,
    output: Path,
    frames: int,
    fps: int,
    sampling: str,
    steps: int | None,
    precision: str,
    seed: int,
    progress: StatusReporter,
) -> AnimeGenI2VResult:
    return generate_animegen_i2v_seed_sweep(
        prompt=prompt,
        image=image,
        last_image=last_image,
        output=output,
        frames=frames,
        fps=fps,
        sampling=sampling,
        steps=steps,
        precision=precision,
        seeds=(seed,),
        progress=progress,
    )[0]


def generate_animegen_i2v_seed_sweep(
    *,
    prompt: str,
    image: Path,
    last_image: Path | None,
    output: Path,
    frames: int,
    fps: int,
    sampling: str,
    steps: int | None,
    precision: str,
    seeds: Sequence[int],
    progress: StatusReporter,
) -> tuple[AnimeGenI2VResult, ...]:
    image, last_image, output, normalized_seeds = _validate_request(
        prompt=prompt,
        image=image,
        last_image=last_image,
        output=output,
        frames=frames,
        fps=fps,
        sampling=sampling,
        steps=steps,
        precision=precision,
        seeds=seeds,
    )
    sampling_profile = ANIMEGEN_SAMPLING_PROFILES[sampling]
    effective_steps = sampling_profile.steps if steps is None else steps
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
    existing = next(
        (path for path in (*outputs, *configs) if path.exists()),
        None,
    )
    if existing is not None:
        raise AnimeGenI2VError(f"output already exists: {existing}")

    animegen_model = _animegen_model_root()
    base_model = _base_model_root()
    lightning_model = _lightning_model_root()
    _validate_models(
        animegen_model,
        base_model,
        lightning_model,
        lightning_required=sampling_profile.lightning,
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    progress.phase(f"load AnimeGen-I2V {precision}")
    torch, pipeline, export_to_video, load_image = _load_pipeline(
        animegen_model=animegen_model,
        base_model=base_model,
        lightning_model=lightning_model,
        sampling=sampling_profile,
        precision=precision,
        progress=progress,
    )
    start = load_image(image.as_posix())
    width, height = _canvas_size(
        start.width,
        start.height,
        max_area=ANIMEGEN_MAX_AREA,
        multiple_of=pipeline.vae_scale_factor_spatial
        * pipeline.transformer.config.patch_size[1],
    )
    start = start.resize((width, height))
    end = (
        load_image(last_image.as_posix()).resize((width, height))
        if last_image is not None
        else None
    )

    results = []
    for seed, job_output, config in zip(
        normalized_seeds,
        outputs,
        configs,
        strict=True,
    ):
        torch.cuda.reset_peak_memory_stats()
        progress.begin(
            effective_steps,
            f"AnimeGen-I2V {sampling} seed {seed}",
        )
        started = time.monotonic()
        generation_prompt = prompt.strip()
        frames_output = pipeline(
            image=start,
            last_image=end,
            prompt=generation_prompt,
            height=height,
            width=width,
            num_frames=frames,
            guidance_scale=sampling_profile.guidance,
            guidance_scale_2=sampling_profile.guidance,
            num_inference_steps=effective_steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            callback_on_step_end=_denoise_progress_callback(
                progress,
                seed=seed,
                steps=effective_steps,
            ),
        ).frames[0]
        progress.phase(f"encode AnimeGen-I2V seed {seed}")
        export_to_video(frames_output, job_output.as_posix(), fps=fps)
        elapsed_seconds = time.monotonic() - started
        memory = cuda_memory_stats(torch, "cuda")
        config_payload = {
            "kind": f"animegen-i2v-{sampling}-config",
            "model": {
                "repo_id": "aidealab/AnimeGen-I2V",
                "revision": ANIMEGEN_MODEL_REVISION,
                "path": animegen_model.as_posix(),
            },
            "base_model": {
                "repo_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "revision": ANIMEGEN_BASE_MODEL_REVISION,
                "path": base_model.as_posix(),
            },
            "request": {
                "prompt": generation_prompt,
                "image": image.as_posix(),
                "last_image": last_image.as_posix() if last_image is not None else None,
                "width": width,
                "height": height,
                "frames": frames,
                "fps": fps,
                "sampling": sampling,
                "steps": effective_steps,
                "guidance": sampling_profile.guidance,
                "flow_shift": sampling_profile.flow_shift,
                "precision": precision,
                "seed": seed,
            },
            "runtime": {
                "scheduler": type(pipeline.scheduler).__name__,
                "storage_dtype": "float8_e4m3fn" if precision == "fp8" else "bfloat16",
                "compute_dtype": "bfloat16",
                "offload": (
                    "leaf-level group offload with streamed on-demand pinning"
                    if precision == "bf16"
                    else "leaf-level group offload with CUDA stream prefetch"
                ),
                "cuda_memory": memory,
            },
            "elapsed_seconds": round(elapsed_seconds, 3),
        }
        if sampling_profile.lightning:
            config_payload["lightning"] = {
                "repo_id": "lightx2v/Wan2.2-Lightning",
                "revision": ANIMEGEN_LIGHTNING_REVISION,
                "path": lightning_model.as_posix(),
            }
        config.write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(
            AnimeGenI2VResult(
                output=job_output,
                config=config,
                image=image,
                last_image=last_image,
                width=width,
                height=height,
                frames=frames,
                fps=fps,
                sampling=sampling,
                steps=effective_steps,
                guidance=sampling_profile.guidance,
                flow_shift=sampling_profile.flow_shift,
                scheduler=type(pipeline.scheduler).__name__,
                precision=precision,
                seed=seed,
                elapsed_seconds=elapsed_seconds,
                cuda_memory=memory,
            )
        )

    progress.phase("AnimeGen-I2V generation completed")
    return tuple(results)


def _load_pipeline(
    *,
    animegen_model: Path,
    base_model: Path,
    lightning_model: Path,
    sampling: AnimeGenSamplingProfile,
    precision: str,
    progress: StatusReporter,
) -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from diffusers import (
            AutoencoderKLWan,
            FlowMatchEulerDiscreteScheduler,
            UniPCMultistepScheduler,
            WanImageToVideoPipeline,
            WanTransformer3DModel,
        )
        from diffusers.utils import export_to_video, load_image
    except ImportError as error:
        raise AnimeGenI2VError(
            "AnimeGen-I2V requires the generation dependencies from scripts/setup_venv.sh"
        ) from error

    transformer_high = WanTransformer3DModel.from_pretrained(
        animegen_model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    progress.step("loaded AnimeGen-I2V high-noise transformer")

    transformer_low = WanTransformer3DModel.from_pretrained(
        animegen_model,
        subfolder="transformer_2",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    progress.step("loaded AnimeGen-I2V low-noise transformer")

    vae = AutoencoderKLWan.from_pretrained(
        base_model,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    scheduler = (
        FlowMatchEulerDiscreteScheduler(shift=sampling.flow_shift)
        if sampling.lightning
        else UniPCMultistepScheduler.from_pretrained(
            base_model,
            subfolder="scheduler",
            flow_shift=sampling.flow_shift,
            local_files_only=True,
        )
    )
    pipeline = WanImageToVideoPipeline.from_pretrained(
        base_model,
        transformer=transformer_high,
        transformer_2=transformer_low,
        scheduler=scheduler,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    if sampling.lightning:
        lightning = (
            lightning_model
            / "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"
        )
        pipeline.load_lora_weights(
            lightning,
            weight_name="high_noise_model.safetensors",
            adapter_name="high",
            local_files_only=True,
        )
        pipeline.load_lora_weights(
            lightning,
            weight_name="low_noise_model.safetensors",
            adapter_name="low",
            load_into_transformer_2=True,
            local_files_only=True,
        )
        pipeline.set_adapters(
            ["high", "low"],
            adapter_weights=[1.0, 1.0],
        )
    if precision == "fp8":
        transformer_high.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )
        transformer_low.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )
    pipeline.set_progress_bar_config(disable=True)
    progress.phase("configure streamed AnimeGen-I2V group offload")
    pipeline.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        non_blocking=True,
        use_stream=True,
        low_cpu_mem_usage=precision == "bf16",
    )
    progress.step("configured streamed AnimeGen-I2V group offload")
    return torch, pipeline, export_to_video, load_image


def _canvas_size(
    width: int,
    height: int,
    *,
    max_area: int,
    multiple_of: int,
) -> tuple[int, int]:
    aspect_ratio = height / width
    target_height = round(math.sqrt(max_area * aspect_ratio)) // multiple_of * multiple_of
    target_width = round(math.sqrt(max_area / aspect_ratio)) // multiple_of * multiple_of
    return target_width, target_height


def _denoise_progress_callback(
    progress: StatusReporter,
    *,
    seed: int,
    steps: int,
) -> Any:
    def callback(
        _pipeline: Any,
        step: int,
        _timestep: Any,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        progress.step(f"AnimeGen-I2V seed {seed} step {step + 1}/{steps}")
        return callback_kwargs

    return callback


def _validate_request(
    *,
    prompt: str,
    image: Path,
    last_image: Path | None,
    output: Path,
    frames: int,
    fps: int,
    sampling: str,
    steps: int | None,
    precision: str,
    seeds: Sequence[int],
) -> tuple[Path, Path | None, Path, tuple[int, ...]]:
    if not prompt.strip():
        raise AnimeGenI2VError("video motion prompt must not be empty")
    image = image.expanduser().resolve()
    last_image = last_image.expanduser().resolve() if last_image is not None else None
    output = output.expanduser().resolve()
    if not image.is_file():
        raise AnimeGenI2VError(f"start image does not exist: {image}")
    if last_image is not None and not last_image.is_file():
        raise AnimeGenI2VError(f"end image does not exist: {last_image}")
    if output.suffix.lower() != ".mp4":
        raise AnimeGenI2VError("AnimeGen-I2V output must use the .mp4 extension")
    if frames < 5 or (frames - 1) % 4 != 0:
        raise AnimeGenI2VError("AnimeGen-I2V frames must be 5 or more in increments of 4")
    if fps <= 0:
        raise AnimeGenI2VError("frames per second must be positive")
    if sampling not in ANIMEGEN_SAMPLING_PROFILES:
        raise AnimeGenI2VError(f"unsupported AnimeGen-I2V sampling profile: {sampling}")
    if steps is not None and steps <= 0:
        raise AnimeGenI2VError("AnimeGen-I2V steps must be positive")
    if precision not in ANIMEGEN_PRECISIONS:
        raise AnimeGenI2VError(f"unsupported AnimeGen-I2V precision: {precision}")
    normalized_seeds = tuple(seeds)
    if not normalized_seeds:
        raise AnimeGenI2VError("AnimeGen-I2V requires at least one seed")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise AnimeGenI2VError("AnimeGen-I2V seed sweep contains duplicate seeds")
    return image, last_image, output, normalized_seeds


def _animegen_model_root() -> Path:
    return MODELS_ROOT / "animegen/aidealab/AnimeGen-I2V"


def _base_model_root() -> Path:
    return MODELS_ROOT / "animegen/Wan-AI/Wan2.2-I2V-A14B-Diffusers"


def _lightning_model_root() -> Path:
    return MODELS_ROOT / "animegen/lightx2v/Wan2.2-Lightning"


def _validate_models(
    animegen_model: Path,
    base_model: Path,
    lightning_model: Path,
    *,
    lightning_required: bool,
) -> None:
    required = [
        animegen_model / "transformer/config.json",
        animegen_model / "transformer/diffusion_pytorch_model.safetensors.index.json",
        animegen_model / "transformer_2/config.json",
        animegen_model / "transformer_2/diffusion_pytorch_model.safetensors.index.json",
        base_model / "model_index.json",
        base_model / "scheduler/scheduler_config.json",
        base_model / "text_encoder/model.safetensors.index.json",
        base_model / "tokenizer/tokenizer.json",
        base_model / "vae/diffusion_pytorch_model.safetensors",
    ]
    if lightning_required:
        required.extend(
            (
                lightning_model
                / "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
                lightning_model
                / "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
            )
        )
    missing = next((path for path in required if not path.is_file()), None)
    if missing is not None:
        raise AnimeGenI2VError(
            f"AnimeGen-I2V model set is incomplete; run scripts/download_animegen_i2v.sh: {missing}"
        )
