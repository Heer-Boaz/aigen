from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from aigen.generation.flux2_scaled_fp8 import load_flux2_klein_scaled_fp8
from aigen.progress import StatusReporter
from aigen.runtime_profiles import MODELS_ROOT


FLUX2_KLEIN_MODEL_ROOT = MODELS_ROOT / "flux2/black-forest-labs/FLUX.2-klein-9B"
FLUX2_KLEIN_TRANSFORMER = (
    MODELS_ROOT
    / "flux2/black-forest-labs/FLUX.2-klein-9b-fp8/flux-2-klein-9b-fp8.safetensors"
)
FLUX2_KLEIN_TEXT_ENCODER = MODELS_ROOT / "flux2/Qwen/Qwen3-8B-FP8"
FLUX2_KLEIN_STEPS = 4


class Flux2KleinError(RuntimeError):
    pass


class Flux2KleinDependencyError(Flux2KleinError):
    pass


@dataclass(frozen=True)
class Flux2KleinResult:
    output: str
    width: int
    height: int
    seed: int
    reference_count: int
    elapsed_ms: float
    peak_vram_mb: int
    lora: str | None
    lora_weight: float

    def to_json(self) -> dict[str, Any]:
        payload = {
            "output": self.output,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "reference_count": self.reference_count,
            "elapsed_ms": self.elapsed_ms,
            "peak_vram_mb": self.peak_vram_mb,
        }
        if self.lora is not None:
            payload["lora"] = self.lora
            payload["lora_weight"] = self.lora_weight
        return payload


def generate_flux2_klein(
    *,
    prompt: str,
    output: Path,
    references: Sequence[Path],
    seed: int,
    lora: Path | None,
    lora_weight: float,
    progress: StatusReporter,
) -> Flux2KleinResult:
    (
        torch,
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinPipeline,
        compute_empirical_mu,
        retrieve_timesteps,
        AutoModelForCausalLM,
        AutoTokenizer,
    ) = _load_dependencies()
    if not torch.cuda.is_available():
        raise Flux2KleinError("FLUX.2 Klein 9B requires CUDA")

    reference_images = _load_reference_images(references)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    progress.phase("encode prompt")
    tokenizer = AutoTokenizer.from_pretrained(
        FLUX2_KLEIN_TEXT_ENCODER,
        local_files_only=True,
    )
    text_encoder = AutoModelForCausalLM.from_pretrained(
        FLUX2_KLEIN_TEXT_ENCODER,
        dtype="auto",
        device_map="cuda",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    try:
        with torch.no_grad():
            prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                text_encoder,
                tokenizer,
                prompt,
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            ).cpu()
    finally:
        del text_encoder
        del tokenizer
        _release_cuda(torch)

    pipeline = None
    try:
        progress.phase("load FLUX.2 Klein 9B")
        transformer = load_flux2_klein_scaled_fp8(
            FLUX2_KLEIN_TRANSFORMER,
            lora=lora,
            lora_weight=lora_weight,
        )
        vae = AutoencoderKLFlux2.from_pretrained(
            FLUX2_KLEIN_MODEL_ROOT / "vae",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            FLUX2_KLEIN_MODEL_ROOT / "scheduler",
            local_files_only=True,
        )
        pipeline = Flux2KleinPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            transformer=transformer,
            is_distilled=True,
        )
        pipeline.set_progress_bar_config(disable=True)

        generator = torch.Generator(device="cuda").manual_seed(seed)
        prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=None,
            prompt_embeds=prompt_embeds,
            device=torch.device("cpu"),
        )
        condition_images, width, height = _prepare_condition_images(pipeline, reference_images)

        progress.phase("encode references")
        vae.requires_grad_(False)
        vae.to("cuda")
        image_latents = None
        image_latent_ids = None
        if condition_images:
            with torch.no_grad():
                image_latents, image_latent_ids = pipeline.prepare_image_latents(
                    images=condition_images,
                    batch_size=1,
                    generator=generator,
                    device=torch.device("cuda"),
                    dtype=vae.dtype,
                )
        if image_latents is not None:
            image_latents = image_latents.cpu()
            image_latent_ids = image_latent_ids.cpu()
        vae.to("cpu")
        _release_cuda(torch)

        progress.phase("load transformer to CUDA")
        transformer.to("cuda")
        prompt_embeds = prompt_embeds.to("cuda")
        text_ids = text_ids.to("cuda")
        if image_latents is not None:
            image_latents = image_latents.to("cuda")
            image_latent_ids = image_latent_ids.to("cuda")

        latents, latent_ids = pipeline.prepare_latents(
            batch_size=1,
            num_latents_channels=transformer.config.in_channels // 4,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=torch.device("cuda"),
            generator=generator,
        )
        mu = compute_empirical_mu(latents.shape[1], FLUX2_KLEIN_STEPS)
        timesteps, _ = retrieve_timesteps(
            scheduler,
            FLUX2_KLEIN_STEPS,
            torch.device("cuda"),
            sigmas=np.linspace(1.0, 1 / FLUX2_KLEIN_STEPS, FLUX2_KLEIN_STEPS),
            mu=mu,
        )
        scheduler.set_begin_index(0)

        progress.begin(FLUX2_KLEIN_STEPS, "denoising 0/4")
        for step_index, timestep in enumerate(timesteps):
            timestep_batch = timestep.expand(latents.shape[0]).to(latents.dtype)
            latent_model_input = latents
            latent_model_ids = latent_ids
            if image_latents is not None:
                latent_model_input = torch.cat((latents, image_latents), dim=1)
                latent_model_ids = torch.cat((latent_ids, image_latent_ids), dim=1)
            noise_prediction = transformer(
                hidden_states=latent_model_input,
                timestep=timestep_batch / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_model_ids,
                return_dict=False,
            )[0][:, : latents.size(1)]
            latents = scheduler.step(
                noise_prediction,
                timestep,
                latents,
                return_dict=False,
            )[0]
            progress.step(f"denoising {step_index + 1}/{FLUX2_KLEIN_STEPS}")

        progress.phase("offload transformer")
        transformer.to("cpu")
        del noise_prediction
        del latent_model_input
        del prompt_embeds
        del text_ids
        del image_latents
        del image_latent_ids
        _release_cuda(torch)

        progress.phase("decode image")
        vae.to("cuda")
        latent_height = 2 * (height // (pipeline.vae_scale_factor * 2))
        latent_width = 2 * (width // (pipeline.vae_scale_factor * 2))
        latents = pipeline._unpack_latents_with_ids(
            latents,
            latent_ids,
            latent_height // 2,
            latent_width // 2,
        )
        latent_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latent_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = pipeline._unpatchify_latents(latents * latent_std + latent_mean)
        with torch.no_grad():
            decoded = vae.decode(latents, return_dict=False)[0]
        image = pipeline.image_processor.postprocess(decoded, output_type="pil")[0]
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2)
        return Flux2KleinResult(
            output=output.as_posix(),
            width=image.width,
            height=image.height,
            seed=seed,
            reference_count=len(reference_images),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            peak_vram_mb=peak_vram_mb,
            lora=lora.resolve().as_posix() if lora is not None else None,
            lora_weight=lora_weight,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise Flux2KleinError("FLUX.2 Klein 9B exceeded 16 GB VRAM") from exc
    finally:
        if pipeline is not None:
            del pipeline
        _release_cuda(torch)


def _load_reference_images(paths: Sequence[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _prepare_condition_images(
    pipeline: Any,
    images: Sequence[Image.Image],
) -> tuple[list[Any], int, int]:
    condition_images = []
    width = None
    height = None
    for image in images:
        image_width, image_height = image.size
        if image_width * image_height > 1024**2:
            image = pipeline.image_processor._resize_to_target_area(image, 1024**2)
            image_width, image_height = image.size
        multiple = pipeline.vae_scale_factor * 2
        image_width = (image_width // multiple) * multiple
        image_height = (image_height // multiple) * multiple
        condition_images.append(
            pipeline.image_processor.preprocess(
                image,
                height=image_height,
                width=image_width,
                resize_mode="crop",
            )
        )
        if width is None:
            width = image_width
            height = image_height
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    return condition_images, width, height


def _release_cuda(torch: Any) -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _load_dependencies() -> tuple[Any, ...]:
    try:
        import torch
        from diffusers import (
            AutoencoderKLFlux2,
            FlowMatchEulerDiscreteScheduler,
            Flux2KleinPipeline,
        )
        from diffusers.pipelines.flux2.pipeline_flux2_klein import (
            compute_empirical_mu,
            retrieve_timesteps,
        )
        from diffusers.utils import logging as diffusers_logging
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as transformers_logging
    except ImportError as exc:
        raise Flux2KleinDependencyError(
            "FLUX.2 Klein requires the generation dependencies"
        ) from exc

    diffusers_logging.set_verbosity_error()
    transformers_logging.set_verbosity_error()
    return (
        torch,
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinPipeline,
        compute_empirical_mu,
        retrieve_timesteps,
        AutoModelForCausalLM,
        AutoTokenizer,
    )
