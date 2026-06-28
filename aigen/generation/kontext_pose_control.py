from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.generation.character_concept import (
    DEFAULT_NEGATIVE_PROMPT,
    DTYPES,
    compose_character_prompt,
)
from aigen.generation.runtime_diagnostics import (
    cuda_memory_stats,
    elapsed_ms,
    module_device_report,
    parameter_locations,
    synchronized_time,
)

@dataclass(frozen=True)
class CharacterKontextPoseResult:
    output_path: str
    model: str
    controlnet_model: str
    reference_image: str
    pose_image: str
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
    timings_ms: dict[str, float]
    transformer_step_ms: list[float]
    controlnet_step_ms: list[float]
    controlnet_active_steps: int
    controlnet_metadata: dict[str, Any]
    memory: dict[str, int]
    environment: dict[str, Any]
    parameter_locations: list[dict[str, Any]]
    steps: int
    guidance_scale: float
    true_cfg_scale: float
    controlnet_conditioning_scale: float
    control_guidance_start: float
    control_guidance_end: float
    dtype: str
    device: str
    pipeline_cpu_offload: bool
    nunchaku_layer_offload: bool
    vae_tiling: bool
    transformer_single_file: str | None
    nunchaku_transformer_model: str | None
    attention_impl: str | None
    seed: int

    def to_json(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "model": self.model,
            "controlnet_model": self.controlnet_model,
            "reference_image": self.reference_image,
            "pose_image": self.pose_image,
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
            "timings_ms": self.timings_ms,
            "transformer_step_ms": self.transformer_step_ms,
            "controlnet_step_ms": self.controlnet_step_ms,
            "controlnet_active_steps": self.controlnet_active_steps,
            "controlnet_metadata": self.controlnet_metadata,
            "memory": self.memory,
            "environment": self.environment,
            "parameter_locations": self.parameter_locations,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "true_cfg_scale": self.true_cfg_scale,
            "controlnet_conditioning_scale": self.controlnet_conditioning_scale,
            "control_guidance_start": self.control_guidance_start,
            "control_guidance_end": self.control_guidance_end,
            "dtype": self.dtype,
            "device": self.device,
            "pipeline_cpu_offload": self.pipeline_cpu_offload,
            "nunchaku_layer_offload": self.nunchaku_layer_offload,
            "vae_tiling": self.vae_tiling,
            "transformer_single_file": self.transformer_single_file,
            "nunchaku_transformer_model": self.nunchaku_transformer_model,
            "attention_impl": self.attention_impl,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class KontextPosePrepared:
    prompt_embeds: Any
    pooled_prompt_embeds: Any
    text_ids: Any
    negative_prompt_embeds: Any
    negative_pooled_prompt_embeds: Any
    negative_text_ids: Any
    do_true_cfg: bool
    base_latents: Any
    image_latents: Any
    generated_img_ids: Any
    combined_img_ids: Any
    control_image: Any
    controlnet_blocks_repeat: bool
    transformer_guidance: Any
    controlnet_guidance: Any
    true_cfg_scale: float
    width: int
    height: int
    batch_size: int
    num_images_per_prompt: int
    num_channels_latents: int
    dtype: Any
    device: str
    seed: int
    steps: int
    token_metadata: dict[str, int]
    timings_ms: dict[str, float]


@dataclass(frozen=True)
class KontextPoseDenoised:
    name: str
    latents: Any
    controlnet_conditioning_scale: float
    control_guidance_start: float
    control_guidance_end: float
    seed: int
    transformer_step_ms: list[float]
    controlnet_step_ms: list[float]
    controlnet_active_steps: int
    controlnet_metadata: dict[str, Any]
    timings_ms: dict[str, float]


@dataclass(frozen=True)
class KontextControlCondition:
    name: str
    control_image: Any
    conditioning_scale: float
    guidance_start: float
    guidance_end: float
    controlnet_blocks_repeat: bool
    residual_mask: Any = None


@dataclass(frozen=True)
class KontextPoseVariant:
    name: str
    seed: int
    controlnet_conditioning_scale: float
    control_guidance_start: float
    control_guidance_end: float


class CharacterKontextPoseError(RuntimeError):
    pass


class CharacterKontextPoseDependencyError(CharacterKontextPoseError):
    pass


class CharacterKontextPoseSession:
    def __init__(
        self,
        model: str,
        controlnet_model: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        transformer_single_file: Path | None = None,
        nunchaku_transformer_model: Path | None = None,
        attention_impl: str | None = None,
        pipeline_cpu_offload: bool = False,
        nunchaku_layer_offload: bool = False,
        vae_tiling: bool = False,
    ) -> None:
        torch, pipeline_class, controlnet_class, transformer_class, load_image = _load_flux_kontext_controlnet()
        self.torch = torch
        self.load_image = load_image
        self.device = device
        self.model = model
        self.controlnet_model = controlnet_model
        self.pipeline_cpu_offload = pipeline_cpu_offload
        self.nunchaku_layer_offload = nunchaku_layer_offload
        self.nunchaku_transformer_model = nunchaku_transformer_model
        self.attention_impl = attention_impl

        model_load_start = synchronized_time(torch)
        self.pipeline = _build_kontext_pose_pipeline(
            pipeline_class,
            controlnet_class,
            transformer_class,
            model,
            controlnet_model,
            transformer_single_file,
            nunchaku_transformer_model,
            attention_impl,
            _torch_dtype(torch, dtype),
            device,
            pipeline_cpu_offload,
            nunchaku_layer_offload,
            vae_tiling,
        )
        self.model_load_ms = elapsed_ms(model_load_start, synchronized_time(torch))

    def prepare(
        self,
        *,
        reference_image: Path,
        pose_image: Path,
        prompt: str,
        negative_prompt: str | None,
        true_cfg_scale: float,
        width: int,
        height: int,
        reference_max_area: int,
        max_sequence_length: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        t5_prompt: str | None = None,
    ) -> KontextPosePrepared:
        return self.prepare_images(
            reference_image=self.load_image(reference_image.resolve().as_posix()),
            pose_image=self.load_image(pose_image.resolve().as_posix()),
            prompt=prompt,
            t5_prompt=t5_prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            width=width,
            height=height,
            reference_max_area=reference_max_area,
            max_sequence_length=max_sequence_length,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

    def prepare_images(
        self,
        *,
        reference_image: Any,
        pose_image: Any,
        prompt: str,
        negative_prompt: str | None,
        true_cfg_scale: float,
        width: int,
        height: int,
        reference_max_area: int,
        max_sequence_length: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        t5_prompt: str | None = None,
    ) -> KontextPosePrepared:
        return self.pipeline.prepare_conditioning(
            image=reference_image,
            control_image=pose_image,
            prompt=prompt,
            t5_prompt=t5_prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=self.torch.Generator(device=self.device).manual_seed(seed),
            latents=None,
            max_sequence_length=max_sequence_length,
            max_area=width * height,
            reference_max_area=reference_max_area,
            seed=seed,
        )

    def prepare_control_condition(
        self,
        prepared: KontextPosePrepared,
        *,
        pose_image: Any,
        seed: int,
    ) -> tuple[Any, bool, float]:
        return self.pipeline.prepare_control_condition(
            control_image=pose_image,
            width=prepared.width,
            height=prepared.height,
            batch_size=prepared.batch_size,
            num_images_per_prompt=prepared.num_images_per_prompt,
            num_channels_latents=prepared.num_channels_latents,
            generator=self.torch.Generator(device=self.device).manual_seed(seed),
            device=prepared.device,
        )

    def prepare_residual_mask(self, prepared: KontextPosePrepared, mask_image: Any) -> Any:
        return self.pipeline.prepare_residual_mask(
            mask_image=mask_image,
            width=prepared.width,
            height=prepared.height,
            batch_size=prepared.batch_size,
            num_images_per_prompt=prepared.num_images_per_prompt,
            device=prepared.device,
            dtype=prepared.dtype,
        )

    def denoise_many(
        self,
        prepared: KontextPosePrepared,
        variants: Sequence[KontextPoseVariant],
        *,
        show_progress: bool = True,
    ) -> list[KontextPoseDenoised]:
        return [
            self.pipeline.denoise_prepared(
                prepared,
                name=variant.name,
                seed=variant.seed,
                controlnet_conditioning_scale=variant.controlnet_conditioning_scale,
                control_guidance_start=variant.control_guidance_start,
                control_guidance_end=variant.control_guidance_end,
                show_progress=show_progress,
            )
            for variant in variants
        ]

    def decode_many(
        self,
        prepared: KontextPosePrepared,
        denoised: Sequence[KontextPoseDenoised],
        *,
        chunk_size: int = 1,
    ) -> tuple[Any, float]:
        return self.pipeline.decode_latents(prepared, denoised, chunk_size=chunk_size)

    def close(self) -> None:
        self.pipeline.maybe_free_model_hooks()


def run_character_kontext_pose_control(
    model: str,
    controlnet_model: str,
    reference_image: Path,
    pose_image: Path,
    output_path: Path,
    prompt: str,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    steps: int = 28,
    guidance_scale: float = 3.5,
    true_cfg_scale: float = 1.0,
    width: int = 384,
    height: int = 576,
    reference_max_area: int = 384 * 768,
    max_sequence_length: int = 128,
    framing: str = "full-body",
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 1,
    pipeline_cpu_offload: bool = False,
    nunchaku_layer_offload: bool = False,
    controlnet_conditioning_scale: float = 0.65,
    control_guidance_start: float = 0.0,
    control_guidance_end: float = 0.65,
    transformer_single_file: Path | None = None,
    nunchaku_transformer_model: Path | None = None,
    attention_impl: str | None = None,
    vae_tiling: bool = False,
) -> CharacterKontextPoseResult:
    torch, pipeline_class, controlnet_class, transformer_class, load_image = _load_flux_kontext_controlnet()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    total_start = synchronized_time(torch)
    model_load_start = synchronized_time(torch)
    pipeline = _build_kontext_pose_pipeline(
        pipeline_class,
        controlnet_class,
        transformer_class,
        model,
        controlnet_model,
        transformer_single_file,
        nunchaku_transformer_model,
        attention_impl,
        _torch_dtype(torch, dtype),
        device,
        pipeline_cpu_offload,
        nunchaku_layer_offload,
        vae_tiling,
    )
    model_load_ms = elapsed_ms(model_load_start, synchronized_time(torch))

    pipeline_prompt = compose_character_prompt(prompt, framing)
    image = pipeline(
        image=load_image(reference_image.resolve().as_posix()),
        control_image=load_image(pose_image.resolve().as_posix()),
        prompt=pipeline_prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        width=width,
        height=height,
        max_area=width * height,
        reference_max_area=reference_max_area,
        max_sequence_length=max_sequence_length,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_start=control_guidance_start,
        control_guidance_end=control_guidance_end,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).images[0]
    total_ms = elapsed_ms(total_start, synchronized_time(torch))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    token_metadata = pipeline.last_token_metadata
    timings_ms = dict(pipeline.last_timings_ms)
    timings_ms["model_load_ms"] = model_load_ms
    timings_ms["total_ms"] = total_ms
    memory = cuda_memory_stats(torch, device)
    return CharacterKontextPoseResult(
        output_path=output_path.resolve().as_posix(),
        model=model,
        controlnet_model=controlnet_model,
        reference_image=reference_image.resolve().as_posix(),
        pose_image=pose_image.resolve().as_posix(),
        prompt=prompt,
        pipeline_prompt=pipeline_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        reference_width=token_metadata["reference_width"],
        reference_height=token_metadata["reference_height"],
        reference_max_area=reference_max_area,
        reference_tokens=token_metadata["reference_tokens"],
        generated_tokens=token_metadata["generated_tokens"],
        text_tokens=token_metadata["text_tokens"],
        total_tokens=token_metadata["total_tokens"],
        max_sequence_length=max_sequence_length,
        timings_ms=timings_ms,
        transformer_step_ms=pipeline.last_transformer_step_ms,
        controlnet_step_ms=pipeline.last_controlnet_step_ms,
        controlnet_active_steps=pipeline.last_controlnet_active_steps,
        controlnet_metadata=pipeline.last_controlnet_metadata,
        memory=memory,
        environment=_generation_environment(torch, pipeline),
        parameter_locations=parameter_locations(pipeline.transformer),
        steps=steps,
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_start=control_guidance_start,
        control_guidance_end=control_guidance_end,
        dtype=dtype,
        device=device,
        pipeline_cpu_offload=pipeline_cpu_offload,
        nunchaku_layer_offload=nunchaku_layer_offload,
        vae_tiling=vae_tiling,
        transformer_single_file=transformer_single_file.resolve().as_posix() if transformer_single_file else None,
        nunchaku_transformer_model=nunchaku_transformer_model.resolve().as_posix() if nunchaku_transformer_model else None,
        attention_impl=attention_impl,
        seed=seed,
    )


def _generation_environment(torch_module: Any, pipeline: Any) -> dict[str, Any]:
    import bitsandbytes as bnb

    environment = {
        "torch_version": torch_module.__version__,
        "torch_cuda_version": torch_module.version.cuda,
        "bitsandbytes_version": bnb.__version__,
        "transformer_class": type(pipeline.transformer).__qualname__,
        "transformer_device_map": getattr(pipeline.transformer, "hf_device_map", None),
        "transformer_quantization_config": str(getattr(pipeline.transformer.config, "quantization_config", None)),
        "device_report": _pipeline_device_report(pipeline),
    }
    if torch_module.cuda.is_available():
        environment["gpu_name"] = torch_module.cuda.get_device_name(0)
        environment["compute_capability"] = list(torch_module.cuda.get_device_capability(0))
    return environment


def _pipeline_device_report(pipeline: Any) -> dict[str, Any]:
    components = {}
    for name in ("controlnet", "transformer", "vae", "text_encoder", "text_encoder_2"):
        component = getattr(pipeline, name, None)
        if component:
            components[name] = module_device_report(component)
    return {
        "pipeline_class": type(pipeline).__qualname__,
        "model_cpu_offload_seq": getattr(pipeline, "model_cpu_offload_seq", ""),
        "components": components,
    }


def extend_control_residuals(samples: Sequence[Any], total_image_tokens: int) -> list[Any]:
    import torch

    extended = []
    for sample in samples:
        if sample.ndim != 3:
            raise ValueError(f"Expected ControlNet residual [B, N, D], got {sample.shape}")

        missing_tokens = total_image_tokens - sample.shape[1]
        if missing_tokens < 0:
            raise ValueError(f"ControlNet residual is longer than transformer image tokens: {sample.shape[1]}")

        if missing_tokens:
            sample = torch.cat(
                (
                    sample,
                    sample.new_zeros(sample.shape[0], missing_tokens, sample.shape[2]),
                ),
                dim=1,
            )
        extended.append(sample)

    return extended


def add_control_residuals(total: Sequence[Any] | None, samples: Sequence[Any] | None) -> list[Any] | None:
    if samples is None:
        return list(total) if total is not None else None
    if total is None:
        return list(samples)
    return [current + sample for current, sample in zip(total, samples, strict=True)]


def apply_control_residual_mask(samples: Sequence[Any] | None, residual_mask: Any) -> list[Any] | None:
    if samples is None:
        return None
    if residual_mask is None:
        return list(samples)
    return [sample * residual_mask.to(device=sample.device, dtype=sample.dtype) for sample in samples]


def residual_suffix_is_zero(samples: Sequence[Any], generated_tokens: int) -> bool:
    return all(sample[:, generated_tokens:].count_nonzero().item() == 0 for sample in samples)


def fit_size_to_area(width: int, height: int, *, max_area: int, multiple_of: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Reference dimensions must be positive")
    if max_area <= 0:
        raise ValueError("reference_max_area must be positive")

    scale = min(1.0, math.sqrt(max_area / (width * height)))
    return (
        max(multiple_of, int(width * scale) // multiple_of * multiple_of),
        max(multiple_of, int(height * scale) // multiple_of * multiple_of),
    )


def _build_kontext_pose_pipeline(
    pipeline_class: Any,
    controlnet_class: Any,
    transformer_class: Any,
    model: str,
    controlnet_model: str,
    transformer_single_file: Path | None,
    nunchaku_transformer_model: Path | None,
    attention_impl: str | None,
    torch_dtype: Any,
    device: str,
    pipeline_cpu_offload: bool,
    nunchaku_layer_offload: bool,
    vae_tiling: bool,
) -> Any:
    controlnet = controlnet_class.from_pretrained(
        controlnet_model,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    pipeline_kwargs: dict[str, Any] = {
        "controlnet": controlnet,
        "torch_dtype": torch_dtype,
        "local_files_only": True,
    }
    if transformer_single_file:
        pipeline_kwargs["transformer"] = transformer_class.from_single_file(
            transformer_single_file.resolve().as_posix(),
            config=model,
            subfolder="transformer",
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
    if nunchaku_transformer_model:
        from nunchaku import NunchakuFluxTransformer2dModel

        pipeline_kwargs["transformer"] = NunchakuFluxTransformer2dModel.from_pretrained(
            nunchaku_transformer_model.resolve().as_posix(),
            torch_dtype=torch_dtype,
            offload=nunchaku_layer_offload,
        )
        if attention_impl:
            pipeline_kwargs["transformer"].set_attention_impl(attention_impl)

    pipeline = pipeline_class.from_pretrained(model, **pipeline_kwargs)
    if vae_tiling:
        pipeline.vae.enable_tiling()
    else:
        pipeline.vae.disable_tiling()
    pipeline.vae.disable_slicing()
    if pipeline_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)
    return pipeline


def _load_flux_kontext_controlnet() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from diffusers import FluxControlNetModel, FluxTransformer2DModel
        from diffusers.pipelines.flux.pipeline_flux_controlnet import retrieve_latents
        from diffusers.pipelines.flux.pipeline_flux_kontext import (
            FluxKontextPipeline,
            FluxPipelineOutput,
            calculate_shift,
            retrieve_timesteps,
        )
        from diffusers.utils import load_image
    except ImportError as exc:
        raise CharacterKontextPoseDependencyError(
            "Kontext pose generation requires `pip install -e .[generation]`"
        ) from exc

    class FluxKontextControlNetPipeline(FluxKontextPipeline):
        model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->transformer->vae"
        _callback_tensor_inputs = ["latents", "prompt_embeds", "control_image"]

        def __init__(
            self,
            scheduler: Any,
            vae: Any,
            text_encoder: Any,
            tokenizer: Any,
            text_encoder_2: Any,
            tokenizer_2: Any,
            transformer: Any,
            controlnet: Any,
            image_encoder: Any = None,
            feature_extractor: Any = None,
        ) -> None:
            super().__init__(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                transformer=transformer,
                image_encoder=image_encoder,
                feature_extractor=feature_extractor,
            )
            self.register_modules(controlnet=controlnet)

        def prepare_image(
            self,
            image: Any,
            width: int,
            height: int,
            batch_size: int,
            num_images_per_prompt: int,
            device: str,
            dtype: Any,
        ) -> Any:
            if not isinstance(image, torch.Tensor):
                image = self.image_processor.preprocess(image, height=height, width=width)

            repeat_by = batch_size if image.shape[0] == 1 else num_images_per_prompt
            return image.repeat_interleave(repeat_by, dim=0).to(device=device, dtype=dtype)

        def prepare_control_condition(
            self,
            *,
            control_image: Any,
            width: int,
            height: int,
            batch_size: int,
            num_images_per_prompt: int,
            num_channels_latents: int,
            generator: Any,
            device: str,
        ) -> tuple[Any, bool, float]:
            control_vae_start = synchronized_time(torch)
            image_batch_size = batch_size * num_images_per_prompt
            control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=image_batch_size,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.vae.dtype,
            )
            controlnet_blocks_repeat = bool(self.controlnet.input_hint_block)
            if not self.controlnet.input_hint_block:
                control_image = retrieve_latents(self.vae.encode(control_image), generator=generator)
                control_image = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                height_control_image, width_control_image = control_image.shape[2:]
                control_image = self._pack_latents(
                    control_image,
                    image_batch_size,
                    num_channels_latents,
                    height_control_image,
                    width_control_image,
                )
            return control_image, controlnet_blocks_repeat, elapsed_ms(control_vae_start, synchronized_time(torch))

        def prepare_residual_mask(
            self,
            *,
            mask_image: Any,
            width: int,
            height: int,
            batch_size: int,
            num_images_per_prompt: int,
            device: str,
            dtype: Any,
        ) -> Any:
            if not isinstance(mask_image, torch.Tensor):
                mask_image = self.image_processor.preprocess(mask_image, height=height, width=width)
                mask_image = (mask_image + 1) * 0.5

            mask = mask_image.mean(dim=1, keepdim=True).clamp(0, 1)
            mask = torch.nn.functional.interpolate(
                mask,
                size=(height // (self.vae_scale_factor * 2), width // (self.vae_scale_factor * 2)),
                mode="bilinear",
                align_corners=False,
            )
            repeat_by = batch_size if mask.shape[0] == 1 else num_images_per_prompt
            mask = mask.repeat_interleave(repeat_by, dim=0)
            return mask.flatten(2).transpose(1, 2).to(device=device, dtype=dtype)

        @torch.no_grad()
        def prepare_conditioning(
            self,
            *,
            image: Any,
            control_image: Any,
            prompt: str,
            negative_prompt: str | None,
            true_cfg_scale: float,
            height: int,
            width: int,
            num_inference_steps: int,
            guidance_scale: float,
            generator: Any,
            latents: Any,
            max_sequence_length: int,
            max_area: int,
            reference_max_area: int,
            seed: int,
            t5_prompt: str | None = None,
        ) -> KontextPosePrepared:
            aspect_ratio = width / height
            width = round((max_area * aspect_ratio) ** 0.5)
            height = round((max_area / aspect_ratio) ** 0.5)

            multiple_of = self.vae_scale_factor * 2
            width = width // multiple_of * multiple_of
            height = height // multiple_of * multiple_of

            self.check_inputs(
                prompt,
                t5_prompt,
                height,
                width,
                negative_prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
            )

            self._guidance_scale = guidance_scale
            self._joint_attention_kwargs = {}
            self._current_timestep = None
            self._interrupt = False

            batch_size = 1
            num_images_per_prompt = 1
            device = self._execution_device

            prompt_encode_start = synchronized_time(torch)
            text_prompt = prompt if t5_prompt is None else t5_prompt
            prompt_token_count = len(
                self.tokenizer_2(
                    text_prompt,
                    padding=False,
                    truncation=False,
                ).input_ids
            )
            if prompt_token_count > max_sequence_length:
                raise ValueError(
                    f"Prompt requires {prompt_token_count} T5 tokens, "
                    f"but max_sequence_length={max_sequence_length}"
                )
            if true_cfg_scale > 1:
                negative_prompt_token_count = len(
                    self.tokenizer_2(
                        negative_prompt,
                        padding=False,
                        truncation=False,
                    ).input_ids
                )
                if negative_prompt_token_count > max_sequence_length:
                    raise ValueError(
                        f"Negative prompt requires {negative_prompt_token_count} T5 tokens, "
                        f"but max_sequence_length={max_sequence_length}"
                    )

            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                prompt,
                t5_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
            prompt_encode_ms = elapsed_ms(prompt_encode_start, synchronized_time(torch))

            do_true_cfg = true_cfg_scale > 1
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            negative_text_ids = None
            if do_true_cfg:
                negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = self.encode_prompt(
                    negative_prompt,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                )

            img = image[0] if isinstance(image, list) else image
            image_height, image_width = self.image_processor.get_default_height_width(img)
            image_width, image_height = fit_size_to_area(
                image_width,
                image_height,
                max_area=reference_max_area,
                multiple_of=multiple_of,
            )
            image = self.image_processor.resize(image, image_height, image_width)
            image = self.image_processor.preprocess(image, image_height, image_width)

            num_channels_latents = self.transformer.config.in_channels // 4
            reference_vae_start = synchronized_time(torch)
            base_latents, image_latents, generated_img_ids, reference_img_ids = self.prepare_latents(
                image,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            reference_vae_ms = elapsed_ms(reference_vae_start, synchronized_time(torch))
            combined_img_ids = torch.cat([generated_img_ids, reference_img_ids], dim=0)
            token_metadata = {
                "reference_width": image_width,
                "reference_height": image_height,
                "reference_tokens": image_latents.shape[1],
                "generated_tokens": base_latents.shape[1],
                "text_tokens": prompt_embeds.shape[1],
                "total_tokens": image_latents.shape[1] + base_latents.shape[1] + prompt_embeds.shape[1],
            }

            control_image, controlnet_blocks_repeat, control_vae_ms = self.prepare_control_condition(
                control_image=control_image,
                width=width,
                height=height,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_channels_latents=num_channels_latents,
                generator=generator,
                device=device,
            )

            transformer_guidance = None
            if self.transformer.config.guidance_embeds:
                transformer_guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                transformer_guidance = transformer_guidance.expand(base_latents.shape[0])

            controlnet_guidance = None
            if self.controlnet.config.guidance_embeds:
                controlnet_guidance = torch.tensor([guidance_scale], device=device)
                controlnet_guidance = controlnet_guidance.expand(base_latents.shape[0])

            return KontextPosePrepared(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                negative_text_ids=negative_text_ids,
                do_true_cfg=do_true_cfg,
                base_latents=base_latents,
                image_latents=image_latents,
                generated_img_ids=generated_img_ids,
                combined_img_ids=combined_img_ids,
                control_image=control_image,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
                transformer_guidance=transformer_guidance,
                controlnet_guidance=controlnet_guidance,
                true_cfg_scale=true_cfg_scale,
                width=width,
                height=height,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_channels_latents=num_channels_latents,
                dtype=prompt_embeds.dtype,
                device=device,
                seed=seed,
                steps=num_inference_steps,
                token_metadata=token_metadata,
                timings_ms={
                    "prompt_encode_ms": prompt_encode_ms,
                    "reference_vae_ms": reference_vae_ms,
                    "control_vae_ms": control_vae_ms,
                },
            )

        def prepare_noise(self, prepared: KontextPosePrepared, seed: int) -> Any:
            if seed == prepared.seed:
                return prepared.base_latents.clone()

            generator = torch.Generator(device=prepared.device).manual_seed(seed)
            latents, _, _, _ = self.prepare_latents(
                None,
                prepared.batch_size * prepared.num_images_per_prompt,
                prepared.num_channels_latents,
                prepared.height,
                prepared.width,
                prepared.dtype,
                prepared.device,
                generator,
                None,
            )
            return latents

        @torch.no_grad()
        def denoise_prepared(
            self,
            prepared: KontextPosePrepared,
            *,
            name: str,
            seed: int,
            controlnet_conditioning_scale: float,
            control_guidance_start: float,
            control_guidance_end: float,
            show_progress: bool = True,
            control_conditions: Sequence[KontextControlCondition] | None = None,
        ) -> KontextPoseDenoised:
            denoise_start = synchronized_time(torch)
            latents = self.prepare_noise(prepared, seed)
            sigmas = np.linspace(1.0, 1 / prepared.steps, prepared.steps)
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                prepared.steps,
                prepared.device,
                sigmas=sigmas,
                mu=mu,
            )
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)

            if control_conditions is None:
                control_conditions = [
                    KontextControlCondition(
                        "pose",
                        prepared.control_image,
                        controlnet_conditioning_scale,
                        control_guidance_start,
                        control_guidance_end,
                        prepared.controlnet_blocks_repeat,
                    )
                ]
            controlnet_keep = [
                [
                    1.0
                    - float(
                        i / len(timesteps) < condition.guidance_start
                        or (i + 1) / len(timesteps) > condition.guidance_end
                    )
                    for condition in control_conditions
                ]
                for i in range(len(timesteps))
            ]

            controlnet_ms = 0.0
            transformer_ms = 0.0
            controlnet_step_ms: list[float] = []
            transformer_step_ms: list[float] = []
            controlnet_active_steps = 0
            controlnet_condition_calls = 0
            controlnet_metadata = {
                "generated_tokens": latents.shape[1],
                "reference_tokens": prepared.image_latents.shape[1],
                "combined_image_tokens": latents.shape[1] + prepared.image_latents.shape[1],
                "double_residual_count": 0,
                "single_residual_count": 0,
                "double_residual_shapes": [],
                "single_residual_shapes": [],
                "reference_suffix_zero": True,
                "controlnet_blocks_repeat": any(condition.controlnet_blocks_repeat for condition in control_conditions),
                "conditions": [
                    {
                        "name": condition.name,
                        "scale": condition.conditioning_scale,
                        "guidance_start": condition.guidance_start,
                        "guidance_end": condition.guidance_end,
                        "masked": condition.residual_mask is not None,
                    }
                    for condition in control_conditions
                ],
                "controlnet_condition_calls": 0,
            }

            self.scheduler.set_begin_index(0)
            progress_bar_context = self.progress_bar(total=num_inference_steps)
            with progress_bar_context as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    latent_model_input = torch.cat([latents, prepared.image_latents], dim=1)
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)
                    controlnet_block_samples = None
                    controlnet_single_block_samples = None
                    step_controlnet_ms = 0.0
                    active_conditions = 0
                    for condition, keep in zip(control_conditions, controlnet_keep[i], strict=True):
                        cond_scale = condition.conditioning_scale * keep
                        if cond_scale:
                            controlnet_start = synchronized_time(torch)
                            block_samples, single_block_samples = self.controlnet(
                                hidden_states=latents,
                                controlnet_cond=condition.control_image,
                                controlnet_mode=None,
                                conditioning_scale=cond_scale,
                                timestep=timestep / 1000,
                                guidance=prepared.controlnet_guidance,
                                pooled_projections=prepared.pooled_prompt_embeds,
                                encoder_hidden_states=prepared.prompt_embeds,
                                txt_ids=prepared.text_ids,
                                img_ids=prepared.generated_img_ids,
                                joint_attention_kwargs=self.joint_attention_kwargs,
                                return_dict=False,
                            )
                            call_ms = elapsed_ms(controlnet_start, synchronized_time(torch))
                            step_controlnet_ms += call_ms
                            controlnet_ms += call_ms
                            active_conditions += 1
                            controlnet_condition_calls += 1
                            block_samples = apply_control_residual_mask(
                                block_samples,
                                condition.residual_mask,
                            )
                            single_block_samples = apply_control_residual_mask(
                                single_block_samples,
                                condition.residual_mask,
                            )
                            controlnet_block_samples = add_control_residuals(
                                controlnet_block_samples,
                                block_samples,
                            )
                            controlnet_single_block_samples = add_control_residuals(
                                controlnet_single_block_samples,
                                single_block_samples,
                            )

                    if active_conditions:
                        controlnet_active_steps += 1
                        total_image_tokens = latent_model_input.shape[1]
                        controlnet_block_samples = extend_control_residuals(
                            controlnet_block_samples,
                            total_image_tokens,
                        )
                        if not controlnet_metadata["double_residual_count"]:
                            controlnet_metadata = {
                                "generated_tokens": latents.shape[1],
                                "reference_tokens": prepared.image_latents.shape[1],
                                "combined_image_tokens": total_image_tokens,
                                "double_residual_count": len(controlnet_block_samples),
                                "single_residual_count": 0,
                                "double_residual_shapes": [list(sample.shape) for sample in controlnet_block_samples],
                                "single_residual_shapes": [],
                                "reference_suffix_zero": residual_suffix_is_zero(
                                    controlnet_block_samples,
                                    latents.shape[1],
                                ),
                                "controlnet_blocks_repeat": any(
                                    condition.controlnet_blocks_repeat for condition in control_conditions
                                ),
                                "conditions": controlnet_metadata["conditions"],
                                "controlnet_condition_calls": controlnet_condition_calls,
                            }
                        if controlnet_single_block_samples:
                            controlnet_single_block_samples = extend_control_residuals(
                                controlnet_single_block_samples,
                                total_image_tokens,
                            )
                            if not controlnet_metadata["single_residual_count"]:
                                controlnet_metadata["single_residual_count"] = len(controlnet_single_block_samples)
                                controlnet_metadata["single_residual_shapes"] = [
                                    list(sample.shape) for sample in controlnet_single_block_samples
                                ]
                                controlnet_metadata["reference_suffix_zero"] = (
                                    controlnet_metadata["reference_suffix_zero"]
                                    and residual_suffix_is_zero(controlnet_single_block_samples, latents.shape[1])
                                )
                        controlnet_metadata["controlnet_condition_calls"] = controlnet_condition_calls
                    controlnet_step_ms.append(round(step_controlnet_ms, 3))

                    transformer_start = synchronized_time(torch)
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=prepared.transformer_guidance,
                        pooled_projections=prepared.pooled_prompt_embeds,
                        encoder_hidden_states=prepared.prompt_embeds,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        txt_ids=prepared.text_ids,
                        img_ids=prepared.combined_img_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        controlnet_blocks_repeat=prepared.controlnet_blocks_repeat,
                    )[0]
                    step_transformer_ms = elapsed_ms(transformer_start, synchronized_time(torch))
                    transformer_ms += step_transformer_ms
                    noise_pred = noise_pred[:, : latents.size(1)]

                    if prepared.do_true_cfg:
                        negative_transformer_start = synchronized_time(torch)
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=prepared.transformer_guidance,
                            pooled_projections=prepared.negative_pooled_prompt_embeds,
                            encoder_hidden_states=prepared.negative_prompt_embeds,
                            controlnet_block_samples=controlnet_block_samples,
                            controlnet_single_block_samples=controlnet_single_block_samples,
                            txt_ids=prepared.negative_text_ids,
                            img_ids=prepared.combined_img_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            controlnet_blocks_repeat=prepared.controlnet_blocks_repeat,
                        )[0]
                        negative_transformer_ms = elapsed_ms(negative_transformer_start, synchronized_time(torch))
                        step_transformer_ms += negative_transformer_ms
                        transformer_ms += negative_transformer_ms
                        neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                        noise_pred = neg_noise_pred + prepared.true_cfg_scale * (noise_pred - neg_noise_pred)
                    transformer_step_ms.append(round(step_transformer_ms, 3))

                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                    if show_progress and (
                        i == len(timesteps) - 1
                        or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0)
                    ):
                        progress_bar.update()

            self._current_timestep = None
            return KontextPoseDenoised(
                name=name,
                latents=latents,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
                seed=seed,
                transformer_step_ms=transformer_step_ms,
                controlnet_step_ms=controlnet_step_ms,
                controlnet_active_steps=controlnet_active_steps,
                controlnet_metadata=controlnet_metadata,
                timings_ms={
                    "controlnet_ms": round(controlnet_ms, 3),
                    "transformer_ms": round(transformer_ms, 3),
                    "denoise_ms": elapsed_ms(denoise_start, synchronized_time(torch)),
                },
            )

        @torch.no_grad()
        def decode_latents(
            self,
            prepared: KontextPosePrepared,
            denoised: Sequence[KontextPoseDenoised],
            *,
            output_type: str = "pil",
            chunk_size: int = 1,
        ) -> tuple[Any, float]:
            vae_decode_start = synchronized_time(torch)
            outputs = []
            for start in range(0, len(denoised), chunk_size):
                latents = torch.cat([result.latents for result in denoised[start : start + chunk_size]], dim=0)
                latents = latents.to(device=prepared.device, dtype=prepared.dtype)
                latents = self._unpack_latents(latents, prepared.height, prepared.width, self.vae_scale_factor)
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                output = self.vae.decode(latents, return_dict=False)[0]
                outputs.extend(self.image_processor.postprocess(output, output_type=output_type))
            return outputs, elapsed_ms(vae_decode_start, synchronized_time(torch))

        @torch.no_grad()
        def __call__(
            self,
            *,
            image: Any,
            control_image: Any,
            prompt: str,
            negative_prompt: str | None = None,
            true_cfg_scale: float = 1.0,
            height: int,
            width: int,
            num_inference_steps: int = 28,
            guidance_scale: float = 3.5,
            control_guidance_start: float = 0.0,
            control_guidance_end: float = 1.0,
            controlnet_conditioning_scale: float = 1.0,
            generator: Any = None,
            latents: Any = None,
            output_type: str = "pil",
            return_dict: bool = True,
            max_sequence_length: int = 512,
            max_area: int = 1024**2,
            reference_max_area: int = 384 * 768,
            t5_prompt: str | None = None,
        ) -> Any:
            pipeline_start = synchronized_time(torch)
            prepared = self.prepare_conditioning(
                image=image,
                control_image=control_image,
                prompt=prompt,
                t5_prompt=t5_prompt,
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                latents=latents,
                max_sequence_length=max_sequence_length,
                max_area=max_area,
                reference_max_area=reference_max_area,
                seed=0,
            )
            denoised = self.denoise_prepared(
                prepared,
                name="image",
                seed=prepared.seed,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
            )

            if output_type == "latent":
                output = denoised.latents
                vae_decode_ms = 0.0
            else:
                output, vae_decode_ms = self.decode_latents(prepared, [denoised], output_type=output_type)

            self.maybe_free_model_hooks()
            self.last_token_metadata = prepared.token_metadata
            self.last_controlnet_metadata = denoised.controlnet_metadata
            self.last_controlnet_step_ms = denoised.controlnet_step_ms
            self.last_transformer_step_ms = denoised.transformer_step_ms
            self.last_controlnet_active_steps = denoised.controlnet_active_steps
            self.last_timings_ms = {
                **prepared.timings_ms,
                "controlnet_ms": denoised.timings_ms["controlnet_ms"],
                "transformer_ms": denoised.timings_ms["transformer_ms"],
                "vae_decode_ms": vae_decode_ms,
                "pipeline_ms": elapsed_ms(pipeline_start, synchronized_time(torch)),
            }

            if not return_dict:
                return (output,)
            return FluxPipelineOutput(images=output)

    return torch, FluxKontextControlNetPipeline, FluxControlNetModel, FluxTransformer2DModel, load_image


def _torch_dtype(torch_module: Any, dtype: str) -> Any:
    dtype_name = DTYPES[dtype]
    if dtype == "auto":
        return None
    return getattr(torch_module, dtype_name)
