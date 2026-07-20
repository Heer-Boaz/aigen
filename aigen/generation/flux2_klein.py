from __future__ import annotations

import gc
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aigen.flux_geometry import FLUX_TOKEN_SIZE
from aigen.generation.flux2_scaled_fp8 import load_flux2_klein_scaled_fp8
from aigen.generation.image_generation_requests import (
    ImageGenerationCaseRequest,
    ImageGenerationOutputRequest,
)
from aigen.generation.flow_match_sampling import euler_ancestral_step
from aigen.generation.prompt_encoding import ordered_unique
from aigen.image_edit_defaults import (
    FLUX2_KLEIN_DEFAULT_SAMPLER,
    FLUX2_KLEIN_SAMPLERS,
    FLUX2_KLEIN_STEPS,
)
from aigen.lora_weights import LoraLoadSpec
from aigen.progress import StatusReporter
from aigen.runtime_profiles import MODELS_ROOT


FLUX2_KLEIN_MODEL_ROOT = MODELS_ROOT / "flux2/black-forest-labs/FLUX.2-klein-9B"
FLUX2_KLEIN_TRANSFORMER = (
    MODELS_ROOT
    / "flux2/black-forest-labs/FLUX.2-klein-9b-fp8/flux-2-klein-9b-fp8.safetensors"
)
FLUX2_KLEIN_TEXT_ENCODER = MODELS_ROOT / "flux2/Qwen/Qwen3-8B-FP8"
FLUX2_KLEIN_RECOMMENDED_MAX_SIDE = 1024
FLUX2_KLEIN_RECOMMENDED_MIN_SIDE = 256


class Flux2KleinError(RuntimeError):
    pass


class Flux2KleinDependencyError(Flux2KleinError):
    pass


def flux2_klein_recommended_canvas_size(
    aspect_ratio: tuple[int, int],
) -> tuple[int, int]:
    ratio_width, ratio_height = aspect_ratio
    if ratio_width >= ratio_height:
        width = FLUX2_KLEIN_RECOMMENDED_MAX_SIDE
        height = round(
            FLUX2_KLEIN_RECOMMENDED_MAX_SIDE
            * ratio_height
            / ratio_width
            / FLUX_TOKEN_SIZE
        ) * FLUX_TOKEN_SIZE
    else:
        height = FLUX2_KLEIN_RECOMMENDED_MAX_SIDE
        width = round(
            FLUX2_KLEIN_RECOMMENDED_MAX_SIDE
            * ratio_width
            / ratio_height
            / FLUX_TOKEN_SIZE
        ) * FLUX_TOKEN_SIZE
    return (
        max(
            FLUX2_KLEIN_RECOMMENDED_MIN_SIDE,
            min(width, FLUX2_KLEIN_RECOMMENDED_MAX_SIDE),
        ),
        max(
            FLUX2_KLEIN_RECOMMENDED_MIN_SIDE,
            min(height, FLUX2_KLEIN_RECOMMENDED_MAX_SIDE),
        ),
    )


@dataclass(frozen=True)
class Flux2KleinPromptEmbedding:
    prompt: str
    prompt_embeds: Any


@dataclass(frozen=True)
class Flux2KleinBatchOutput:
    case: str
    name: str
    output: str
    width: int
    height: int
    seed: int
    reference_count: int
    denoise_ms: float
    decode_ms: float

    def to_json(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "name": self.name,
            "output": self.output,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "reference_count": self.reference_count,
            "denoise_ms": self.denoise_ms,
            "decode_ms": self.decode_ms,
        }


@dataclass(frozen=True)
class Flux2KleinBatchResult:
    outputs: tuple[Flux2KleinBatchOutput, ...]
    generation_ms: float
    model_load_ms: float
    peak_vram_mb: int
    loras: tuple[LoraLoadSpec, ...]
    sampler: str

    def to_json(self) -> dict[str, Any]:
        payload = {
            "outputs": [output.to_json() for output in self.outputs],
            "generation_ms": self.generation_ms,
            "model_load_ms": self.model_load_ms,
            "peak_vram_mb": self.peak_vram_mb,
            "sampler": self.sampler,
        }
        if self.loras:
            payload["loras"] = [lora.to_json() for lora in self.loras]
        return payload


@dataclass(frozen=True)
class Flux2KleinResult:
    output: str
    width: int
    height: int
    seed: int
    reference_count: int
    elapsed_ms: float
    peak_vram_mb: int
    loras: tuple[LoraLoadSpec, ...]
    sampler: str

    def to_json(self) -> dict[str, Any]:
        payload = {
            "output": self.output,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "reference_count": self.reference_count,
            "elapsed_ms": self.elapsed_ms,
            "peak_vram_mb": self.peak_vram_mb,
            "sampler": self.sampler,
        }
        if self.loras:
            payload["loras"] = [lora.to_json() for lora in self.loras]
        return payload


@dataclass(frozen=True)
class _PreparedFlux2KleinCase:
    request: ImageGenerationCaseRequest
    width: int
    height: int
    prompt_embeds: Any
    text_ids: Any
    image_latents: Any | None
    image_latent_ids: Any | None
    init_source_latent: Any | None = None


@dataclass(frozen=True)
class _DenoisedFlux2KleinOutput:
    case: _PreparedFlux2KleinCase
    request: ImageGenerationOutputRequest
    latents: Any
    latent_ids: Any
    denoise_ms: float


def encode_flux2_klein_prompts(
    *,
    prompts: Sequence[str],
    progress: StatusReporter,
) -> tuple[dict[str, Flux2KleinPromptEmbedding], float]:
    (
        torch,
        _,
        _,
        Flux2KleinPipeline,
        _,
        _,
        AutoModelForCausalLM,
        AutoTokenizer,
    ) = _load_dependencies()
    if not torch.cuda.is_available():
        raise Flux2KleinError("FLUX.2 Klein 9B requires CUDA")

    started = time.perf_counter()
    progress.phase("encode prompts")
    tokenizer = AutoTokenizer.from_pretrained(
        FLUX2_KLEIN_TEXT_ENCODER,
        local_files_only=True,
    )
    text_encoder = None
    try:
        text_encoder = AutoModelForCausalLM.from_pretrained(
            FLUX2_KLEIN_TEXT_ENCODER,
            dtype="auto",
            device_map="cuda",
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        embeddings = {}
        with torch.no_grad():
            for prompt in ordered_unique(prompts):
                embeddings[prompt] = Flux2KleinPromptEmbedding(
                    prompt=prompt,
                    prompt_embeds=Flux2KleinPipeline._get_qwen3_prompt_embeds(
                        text_encoder,
                        tokenizer,
                        prompt,
                        dtype=torch.bfloat16,
                        device=torch.device("cuda"),
                    ).cpu(),
                )
        return embeddings, (time.perf_counter() - started) * 1000
    except torch.cuda.OutOfMemoryError as exc:
        raise Flux2KleinError("FLUX.2 Klein 9B exceeded 16 GB VRAM") from exc
    finally:
        if text_encoder is not None:
            del text_encoder
        del tokenizer
        _release_cuda(torch)


class Flux2KleinSession:
    def __init__(
        self,
        *,
        loras: tuple[LoraLoadSpec, ...],
        sampler: str = FLUX2_KLEIN_DEFAULT_SAMPLER,
        strength: float | None = None,
        progress: StatusReporter,
    ) -> None:
        (
            torch,
            AutoencoderKLFlux2,
            FlowMatchEulerDiscreteScheduler,
            Flux2KleinPipeline,
            compute_empirical_mu,
            retrieve_timesteps,
            _,
            _,
        ) = _load_dependencies()
        if not torch.cuda.is_available():
            raise Flux2KleinError("FLUX.2 Klein 9B requires CUDA")

        progress.phase("load FLUX.2 Klein 9B")
        started = time.perf_counter()
        transformer = load_flux2_klein_scaled_fp8(
            FLUX2_KLEIN_TRANSFORMER,
            loras=loras,
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
        self.pipeline = Flux2KleinPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            transformer=transformer,
            is_distilled=True,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.torch = torch
        self.compute_empirical_mu = compute_empirical_mu
        self.retrieve_timesteps = retrieve_timesteps
        self.loras = loras
        self.sampler = sampler
        self.strength = strength
        self.model_load_ms = (time.perf_counter() - started) * 1000

    def generate(
        self,
        *,
        cases: Sequence[ImageGenerationCaseRequest],
        prompt_embeddings: Mapping[str, Flux2KleinPromptEmbedding],
        progress: StatusReporter,
    ) -> Flux2KleinBatchResult:
        _validate_flux2_klein_cases(cases)
        started = time.perf_counter()
        try:
            with self.torch.no_grad():
                prepared_cases = _prepare_flux2_klein_cases(
                    self.pipeline,
                    cases=cases,
                    prompt_embeddings=prompt_embeddings,
                    torch=self.torch,
                    strength=self.strength,
                    progress=progress,
                )
                denoised_outputs = _denoise_flux2_klein_cases(
                    self.pipeline,
                    prepared_cases=prepared_cases,
                    torch=self.torch,
                    compute_empirical_mu=self.compute_empirical_mu,
                    retrieve_timesteps=self.retrieve_timesteps,
                    sampler=self.sampler,
                    strength=self.strength,
                    progress=progress,
                )
                outputs = _decode_flux2_klein_outputs(
                    self.pipeline,
                    denoised_outputs=denoised_outputs,
                    torch=self.torch,
                    progress=progress,
                )
            return Flux2KleinBatchResult(
                outputs=outputs,
                generation_ms=(time.perf_counter() - started) * 1000,
                model_load_ms=self.model_load_ms,
                peak_vram_mb=round(self.torch.cuda.max_memory_allocated() / 1024**2),
                loras=self.loras,
                sampler=self.sampler,
            )
        except self.torch.cuda.OutOfMemoryError as exc:
            raise Flux2KleinError("FLUX.2 Klein 9B exceeded 16 GB VRAM") from exc

    def close(self) -> None:
        del self.pipeline
        _release_cuda(self.torch)


def generate_flux2_klein(
    *,
    prompt: str,
    output: Path,
    references: Sequence[Path],
    width: int | None,
    height: int | None,
    seed: int,
    loras: tuple[LoraLoadSpec, ...],
    sampler: str = FLUX2_KLEIN_DEFAULT_SAMPLER,
    strength: float | None = None,
    progress: StatusReporter,
) -> Flux2KleinResult:
    started = time.perf_counter()
    batch = generate_flux2_klein_seed_sweep(
        prompt=prompt,
        output=output,
        references=references,
        width=width,
        height=height,
        seeds=(seed,),
        loras=loras,
        sampler=sampler,
        strength=strength,
        progress=progress,
    )
    generated = batch.outputs[0]
    return Flux2KleinResult(
        output=generated.output,
        width=generated.width,
        height=generated.height,
        seed=generated.seed,
        reference_count=generated.reference_count,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        peak_vram_mb=batch.peak_vram_mb,
        loras=batch.loras,
        sampler=batch.sampler,
    )


def generate_flux2_klein_seed_sweep(
    *,
    prompt: str,
    output: Path,
    references: Sequence[Path],
    width: int | None,
    height: int | None,
    seeds: Sequence[int],
    loras: tuple[LoraLoadSpec, ...],
    sampler: str = FLUX2_KLEIN_DEFAULT_SAMPLER,
    strength: float | None = None,
    progress: StatusReporter,
) -> Flux2KleinBatchResult:
    if strength is not None and not (0.0 < strength <= 1.0):
        raise Flux2KleinError("--strength must be in (0, 1]")
    if sampler not in FLUX2_KLEIN_SAMPLERS:
        raise Flux2KleinError(
            f"unsupported FLUX.2 Klein sampler {sampler!r}; choose from: "
            f"{', '.join(FLUX2_KLEIN_SAMPLERS)}"
        )
    normalized_seeds = tuple(seeds)
    if not normalized_seeds:
        raise Flux2KleinError("FLUX.2 Klein seed sweep requires at least one seed")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise Flux2KleinError("FLUX.2 Klein seed sweep contains duplicate seeds")

    torch = _load_dependencies()[0]
    if not torch.cuda.is_available():
        raise Flux2KleinError("FLUX.2 Klein 9B requires CUDA")
    torch.cuda.reset_peak_memory_stats()

    output = output.expanduser().resolve()
    outputs = tuple(
        output
        if len(normalized_seeds) == 1
        else output.with_name(f"{output.stem}-seed{seed}{output.suffix}")
        for seed in normalized_seeds
    )
    case = ImageGenerationCaseRequest(
        name=output.stem,
        prompt=prompt,
        image_paths=tuple(references),
        width=width,
        height=height,
        outputs=tuple(
            ImageGenerationOutputRequest(
                name=f"seed-{seed}",
                seed=seed,
                path=seed_output,
            )
            for seed, seed_output in zip(normalized_seeds, outputs, strict=True)
        ),
    )
    _validate_flux2_klein_cases((case,))
    prompt_embeddings, _ = encode_flux2_klein_prompts(
        prompts=(prompt,),
        progress=progress,
    )
    session = Flux2KleinSession(
        loras=loras,
        sampler=sampler,
        strength=strength,
        progress=progress,
    )
    try:
        return session.generate(
            cases=(case,),
            prompt_embeddings=prompt_embeddings,
            progress=progress,
        )
    finally:
        session.close()


def _validate_flux2_klein_cases(cases: Sequence[ImageGenerationCaseRequest]) -> None:
    if not cases:
        raise Flux2KleinError("FLUX.2 Klein requires at least one generation case")
    for case in cases:
        if not case.outputs:
            raise Flux2KleinError(f"FLUX.2 Klein case {case.name} has no outputs")
        if (case.width is None) != (case.height is None):
            raise Flux2KleinError("width and height must be provided together")
        if case.width is not None and (
            case.width < FLUX_TOKEN_SIZE
            or case.height < FLUX_TOKEN_SIZE
            or case.width % FLUX_TOKEN_SIZE
            or case.height % FLUX_TOKEN_SIZE
        ):
            raise Flux2KleinError(
                f"width and height must be positive multiples of {FLUX_TOKEN_SIZE}"
            )


def _prepare_flux2_klein_cases(
    pipeline: Any,
    *,
    cases: Sequence[ImageGenerationCaseRequest],
    prompt_embeddings: Mapping[str, Flux2KleinPromptEmbedding],
    torch: Any,
    strength: float | None = None,
    progress: StatusReporter,
) -> tuple[_PreparedFlux2KleinCase, ...]:
    encoded_prompts = {}
    for prompt in ordered_unique(case.prompt for case in cases):
        embedding = prompt_embeddings[prompt]
        encoded_prompts[prompt] = pipeline.encode_prompt(
            prompt=None,
            prompt_embeds=embedding.prompt_embeds,
            device=torch.device("cpu"),
        )

    progress.phase("encode references")
    vae = pipeline.vae
    vae.requires_grad_(False)
    has_references = any(case.image_paths for case in cases)
    reference_cache = {}
    prepared_cases = []
    try:
        if has_references:
            vae.to("cuda")
        for case in cases:
            reference_key = tuple(path.resolve() for path in case.image_paths)
            reference_encoding = reference_cache.get(reference_key)
            if reference_encoding is None:
                reference_images = _load_reference_images(reference_key)
                try:
                    (
                        condition_images,
                        inferred_width,
                        inferred_height,
                    ) = _prepare_condition_images(
                        pipeline,
                        reference_images,
                    )
                finally:
                    for image in reference_images:
                        image.close()
                image_latents = None
                image_latent_ids = None
                init_source_latent = None
                if strength is not None and condition_images:
                    # img2img: the first reference is the init image; encode it (unpacked)
                    # and drop kontext context so its pose is preserved by the init, not a hint.
                    generator = torch.Generator(device="cuda").manual_seed(case.outputs[0].seed)
                    with torch.no_grad():
                        init_source_latent = pipeline._encode_vae_image(
                            condition_images[0].to(device="cuda", dtype=vae.dtype),
                            generator,
                        ).cpu()
                elif condition_images:
                    generator = torch.Generator(device="cuda").manual_seed(case.outputs[0].seed)
                    with torch.no_grad():
                        image_latents, image_latent_ids = pipeline.prepare_image_latents(
                            images=condition_images,
                            batch_size=1,
                            generator=generator,
                            device=torch.device("cuda"),
                            dtype=vae.dtype,
                        )
                    image_latents = image_latents.cpu()
                    image_latent_ids = image_latent_ids.cpu()
                reference_encoding = (
                    image_latents,
                    image_latent_ids,
                    init_source_latent,
                    inferred_width,
                    inferred_height,
                )
                reference_cache[reference_key] = reference_encoding
            (
                image_latents,
                image_latent_ids,
                init_source_latent,
                inferred_width,
                inferred_height,
            ) = reference_encoding
            prompt_embeds, text_ids = encoded_prompts[case.prompt]
            prepared_cases.append(
                _PreparedFlux2KleinCase(
                    request=case,
                    width=case.width or inferred_width,
                    height=case.height or inferred_height,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    image_latents=image_latents,
                    image_latent_ids=image_latent_ids,
                    init_source_latent=init_source_latent,
                )
            )
    finally:
        if has_references:
            vae.to("cpu")
        _release_cuda(torch)
    return tuple(prepared_cases)


def _denoise_flux2_klein_cases(
    pipeline: Any,
    *,
    prepared_cases: Sequence[_PreparedFlux2KleinCase],
    torch: Any,
    compute_empirical_mu: Any,
    retrieve_timesteps: Any,
    sampler: str,
    strength: float | None = None,
    progress: StatusReporter,
) -> tuple[_DenoisedFlux2KleinOutput, ...]:
    transformer = pipeline.transformer
    total_steps = sum(len(case.request.outputs) for case in prepared_cases) * FLUX2_KLEIN_STEPS
    progress.begin(total_steps, f"denoising 0/{total_steps}")
    denoised_outputs = []
    try:
        progress.phase("load transformer to CUDA")
        transformer.to("cuda")
        for case in prepared_cases:
            denoised_outputs.extend(
                _denoise_flux2_klein_case(
                    pipeline,
                    case=case,
                    torch=torch,
                    compute_empirical_mu=compute_empirical_mu,
                    retrieve_timesteps=retrieve_timesteps,
                    sampler=sampler,
                    strength=strength,
                    progress=progress,
                )
            )
    finally:
        progress.phase("offload transformer")
        transformer.to("cpu")
        _release_cuda(torch)
    return tuple(denoised_outputs)


def _denoise_flux2_klein_case(
    pipeline: Any,
    *,
    case: _PreparedFlux2KleinCase,
    torch: Any,
    compute_empirical_mu: Any,
    retrieve_timesteps: Any,
    sampler: str,
    strength: float | None = None,
    progress: StatusReporter,
) -> list[_DenoisedFlux2KleinOutput]:
    prompt_embeds = case.prompt_embeds.to("cuda")
    text_ids = case.text_ids.to("cuda")
    image_latents = (
        case.image_latents.to("cuda") if case.image_latents is not None else None
    )
    image_latent_ids = (
        case.image_latent_ids.to("cuda") if case.image_latent_ids is not None else None
    )
    outputs = []
    for output in case.request.outputs:
        started = time.perf_counter()
        generator = torch.Generator(device="cuda").manual_seed(output.seed)
        latents, latent_ids = pipeline.prepare_latents(
            batch_size=1,
            num_latents_channels=pipeline.transformer.config.in_channels // 4,
            height=case.height,
            width=case.width,
            dtype=prompt_embeds.dtype,
            device=torch.device("cuda"),
            generator=generator,
        )
        mu = compute_empirical_mu(latents.shape[1], FLUX2_KLEIN_STEPS)
        timesteps, _ = retrieve_timesteps(
            pipeline.scheduler,
            FLUX2_KLEIN_STEPS,
            torch.device("cuda"),
            sigmas=np.linspace(1.0, 1 / FLUX2_KLEIN_STEPS, FLUX2_KLEIN_STEPS),
            mu=mu,
        )
        begin_index = 0
        if case.init_source_latent is not None and strength is not None:
            # img2img: seed the latent from the noised source so its pose is preserved,
            # then start denoising partway through the schedule (higher strength = more change).
            begin_index = min(
                FLUX2_KLEIN_STEPS - 1,
                max(0, round((1.0 - strength) * FLUX2_KLEIN_STEPS)),
            )
            source = case.init_source_latent.to(device="cuda", dtype=latents.dtype)
            init_noise = torch.randn(
                source.shape,
                generator=generator,
                device=torch.device("cuda"),
                dtype=source.dtype,
            )
            noised = pipeline.scheduler.scale_noise(
                source, timesteps[begin_index : begin_index + 1], init_noise
            )
            latents = pipeline._pack_latents(noised)
        pipeline.scheduler.set_begin_index(begin_index)
        ancestral_generator = (
            torch.Generator(device="cuda").manual_seed(output.seed)
            if sampler == "euler-ancestral"
            else None
        )

        with torch.no_grad():
            for step_index, timestep in enumerate(timesteps):
                if step_index < begin_index:
                    continue
                timestep_batch = timestep.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = latents
                latent_model_ids = latent_ids
                if image_latents is not None:
                    latent_model_input = torch.cat((latents, image_latents), dim=1)
                    latent_model_ids = torch.cat((latent_ids, image_latent_ids), dim=1)
                noise_prediction = pipeline.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_batch / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_model_ids,
                    return_dict=False,
                )[0][:, : latents.size(1)]
                if sampler == "euler-ancestral":
                    if pipeline.scheduler.step_index is None:
                        pipeline.scheduler._init_step_index(timestep)
                    sigma_index = pipeline.scheduler.step_index
                    latents = euler_ancestral_step(
                        latents,
                        noise_prediction,
                        sigma=pipeline.scheduler.sigmas[sigma_index],
                        sigma_next=pipeline.scheduler.sigmas[sigma_index + 1],
                        final=step_index + 1 == len(timesteps),
                        generator=ancestral_generator,
                        torch=torch,
                    )
                    pipeline.scheduler._step_index += 1
                else:
                    latents = pipeline.scheduler.step(
                        noise_prediction,
                        timestep,
                        latents,
                        return_dict=False,
                    )[0]
                progress.step(
                    f"{output.name}: denoising {step_index + 1}/{FLUX2_KLEIN_STEPS}"
                )
        output_latents = latents.cpu()
        output_latent_ids = latent_ids.cpu()
        del latents
        del latent_ids
        del noise_prediction
        del latent_model_input
        del latent_model_ids
        del timestep_batch
        outputs.append(
            _DenoisedFlux2KleinOutput(
                case=case,
                request=output,
                latents=output_latents,
                latent_ids=output_latent_ids,
                denoise_ms=(time.perf_counter() - started) * 1000,
            )
        )
    return outputs


def _decode_flux2_klein_outputs(
    pipeline: Any,
    *,
    denoised_outputs: Sequence[_DenoisedFlux2KleinOutput],
    torch: Any,
    progress: StatusReporter,
) -> tuple[Flux2KleinBatchOutput, ...]:
    progress.phase("decode images")
    vae = pipeline.vae
    outputs = []
    try:
        vae.to("cuda")
        for denoised in denoised_outputs:
            outputs.append(
                _decode_flux2_klein_output(
                    pipeline,
                    denoised=denoised,
                    torch=torch,
                )
            )
    finally:
        vae.to("cpu")
        _release_cuda(torch)
    return tuple(outputs)


def _decode_flux2_klein_output(
    pipeline: Any,
    *,
    denoised: _DenoisedFlux2KleinOutput,
    torch: Any,
) -> Flux2KleinBatchOutput:
    started = time.perf_counter()
    vae = pipeline.vae
    latents = denoised.latents.to("cuda")
    latent_ids = denoised.latent_ids.to("cuda")
    latent_height = 2 * (denoised.case.height // (pipeline.vae_scale_factor * 2))
    latent_width = 2 * (denoised.case.width // (pipeline.vae_scale_factor * 2))
    latents = pipeline._unpack_latents_with_ids(
        latents,
        latent_ids,
        latent_height // 2,
        latent_width // 2,
    )
    latent_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
        latents.device,
        latents.dtype,
    )
    latent_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = pipeline._unpatchify_latents(latents * latent_std + latent_mean)
    with torch.no_grad():
        decoded = vae.decode(latents, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(decoded, output_type="pil")[0]
    output_path = denoised.request.path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return Flux2KleinBatchOutput(
        case=denoised.case.request.name,
        name=denoised.request.name,
        output=output_path.as_posix(),
        width=image.width,
        height=image.height,
        seed=denoised.request.seed,
        reference_count=len(denoised.case.request.image_paths),
        denoise_ms=denoised.denoise_ms,
        decode_ms=(time.perf_counter() - started) * 1000,
    )


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
