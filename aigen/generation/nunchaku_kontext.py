from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from PIL import Image

from aigen.generation.character_concept import (
    DEFAULT_NEGATIVE_PROMPT,
    DTYPES,
    compose_character_prompt,
)
from aigen.generation.kontext_pose_control import fit_size_to_area
from aigen.generation.runtime_diagnostics import (
    cuda_memory_stats,
    elapsed_ms,
    parameter_locations,
    synchronized_time,
)


@dataclass(frozen=True)
class NunchakuKontextResult:
    output_path: str
    base_model: str
    transformer_model: str
    reference_image: str
    prompt: str
    pipeline_prompt: str
    negative_prompt: str
    width: int
    height: int
    reference_width: int
    reference_height: int
    reference_max_area: int
    reference_tokens: int
    generated_tokens: int
    text_tokens: int
    total_tokens: int
    max_sequence_length: int
    steps: int
    step_ms: list[float]
    warm_step_median_ms: float
    timings_ms: dict[str, float]
    memory: dict[str, int]
    environment: dict[str, Any]
    parameter_locations: list[dict[str, Any]]
    guidance_scale: float
    true_cfg_scale: float
    dtype: str
    device: str
    seed: int

    def to_json(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "base_model": self.base_model,
            "transformer_model": self.transformer_model,
            "reference_image": self.reference_image,
            "prompt": self.prompt,
            "pipeline_prompt": self.pipeline_prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "reference_width": self.reference_width,
            "reference_height": self.reference_height,
            "reference_max_area": self.reference_max_area,
            "reference_tokens": self.reference_tokens,
            "generated_tokens": self.generated_tokens,
            "text_tokens": self.text_tokens,
            "total_tokens": self.total_tokens,
            "max_sequence_length": self.max_sequence_length,
            "steps": self.steps,
            "step_ms": self.step_ms,
            "warm_step_median_ms": self.warm_step_median_ms,
            "timings_ms": self.timings_ms,
            "memory": self.memory,
            "environment": self.environment,
            "parameter_locations": self.parameter_locations,
            "guidance_scale": self.guidance_scale,
            "true_cfg_scale": self.true_cfg_scale,
            "dtype": self.dtype,
            "device": self.device,
            "seed": self.seed,
        }


class NunchakuKontextError(RuntimeError):
    pass


class NunchakuKontextDependencyError(NunchakuKontextError):
    pass


def run_nunchaku_kontext(
    base_model: str,
    transformer_model: str,
    reference_image: Path,
    output_path: Path,
    prompt: str,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    steps: int = 3,
    guidance_scale: float = 2.5,
    true_cfg_scale: float = 1.0,
    width: int = 384,
    height: int = 576,
    reference_max_area: int = 384 * 768,
    max_sequence_length: int = 128,
    framing: str = "full-body",
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 1,
) -> NunchakuKontextResult:
    torch, pipeline_class, transformer_class = _load_nunchaku_kontext()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    total_start = synchronized_time(torch)
    model_load_start = synchronized_time(torch)
    transformer = transformer_class.from_pretrained(transformer_model)
    pipeline = pipeline_class.from_pretrained(
        base_model,
        transformer=transformer,
        torch_dtype=_torch_dtype(torch, dtype),
        local_files_only=True,
    ).to(device)
    pipeline.vae.disable_slicing()
    pipeline.vae.disable_tiling()
    model_load_ms = elapsed_ms(model_load_start, synchronized_time(torch))

    pipeline_prompt = compose_character_prompt(prompt, framing)
    _check_t5_budget(pipeline, pipeline_prompt, max_sequence_length, "Prompt")
    if true_cfg_scale > 1:
        _check_t5_budget(pipeline, negative_prompt, max_sequence_length, "Negative prompt")

    image, reference_width, reference_height = _load_resized_reference(
        reference_image,
        max_area=reference_max_area,
        multiple_of=pipeline.vae_scale_factor * 2,
    )
    step_ms: list[float] = []
    step_start = synchronized_time(torch)

    def record_step(_pipeline: Any, _step: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        nonlocal step_start
        now = synchronized_time(torch)
        step_ms.append(elapsed_ms(step_start, now))
        step_start = now
        return callback_kwargs

    generation_start = synchronized_time(torch)
    output = pipeline(
        image=image,
        prompt=pipeline_prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        width=width,
        height=height,
        max_area=width * height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length,
        _auto_resize=False,
        generator=torch.Generator(device=device).manual_seed(seed),
        callback_on_step_end=record_step,
    ).images[0]
    generation_ms = elapsed_ms(generation_start, synchronized_time(torch))
    total_ms = elapsed_ms(total_start, synchronized_time(torch))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)

    generated_tokens = _flux_tokens(width, height)
    reference_tokens = _flux_tokens(reference_width, reference_height)
    text_tokens = max_sequence_length
    return NunchakuKontextResult(
        output_path=output_path.resolve().as_posix(),
        base_model=base_model,
        transformer_model=transformer_model,
        reference_image=reference_image.resolve().as_posix(),
        prompt=prompt,
        pipeline_prompt=pipeline_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        reference_width=reference_width,
        reference_height=reference_height,
        reference_max_area=reference_max_area,
        reference_tokens=reference_tokens,
        generated_tokens=generated_tokens,
        text_tokens=text_tokens,
        total_tokens=reference_tokens + generated_tokens + text_tokens,
        max_sequence_length=max_sequence_length,
        steps=steps,
        step_ms=step_ms,
        warm_step_median_ms=_warm_step_median(step_ms),
        timings_ms={
            "model_load_ms": model_load_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
        },
        memory=cuda_memory_stats(torch, device),
        environment=_nunchaku_environment(torch, pipeline),
        parameter_locations=parameter_locations(pipeline.transformer),
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
        dtype=dtype,
        device=device,
        seed=seed,
    )


def _load_nunchaku_kontext() -> tuple[Any, Any, Any]:
    try:
        import torch
        from diffusers import FluxKontextPipeline
        from nunchaku import NunchakuFluxTransformer2dModel
    except ImportError as exc:
        raise NunchakuKontextDependencyError(
            "Nunchaku Kontext generation requires the nunchaku wheel matching this torch build"
        ) from exc
    return torch, FluxKontextPipeline, NunchakuFluxTransformer2dModel


def _load_resized_reference(reference_image: Path, *, max_area: int, multiple_of: int) -> tuple[Image.Image, int, int]:
    image = Image.open(reference_image).convert("RGB")
    width, height = fit_size_to_area(image.width, image.height, max_area=max_area, multiple_of=multiple_of)
    if (image.width, image.height) != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image, width, height


def _check_t5_budget(pipeline: Any, prompt: str, max_sequence_length: int, label: str) -> None:
    token_count = len(
        pipeline.tokenizer_2(
            prompt,
            padding=False,
            truncation=False,
        ).input_ids
    )
    if token_count > max_sequence_length:
        raise ValueError(f"{label} requires {token_count} T5 tokens, but max_sequence_length={max_sequence_length}")


def _flux_tokens(width: int, height: int) -> int:
    return (width // 16) * (height // 16)


def _warm_step_median(step_ms: list[float]) -> float:
    if len(step_ms) < 2:
        return step_ms[0] if step_ms else 0.0
    warm = sorted(step_ms[1:])
    return warm[len(warm) // 2]


def _nunchaku_environment(torch_module: Any, pipeline: Any) -> dict[str, Any]:
    environment = {
        "torch_version": torch_module.__version__,
        "torch_cuda_version": torch_module.version.cuda,
        "nunchaku_version": version("nunchaku"),
        "transformer_class": type(pipeline.transformer).__qualname__,
    }
    if torch_module.cuda.is_available():
        environment["gpu_name"] = torch_module.cuda.get_device_name(0)
        environment["compute_capability"] = list(torch_module.cuda.get_device_capability(0))
    return environment


def _torch_dtype(torch_module: Any, dtype: str) -> Any:
    dtype_name = DTYPES[dtype]
    if dtype == "auto":
        return None
    return getattr(torch_module, dtype_name)
