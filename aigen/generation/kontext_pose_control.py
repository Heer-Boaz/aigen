from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.generation.character_concept import (
    DEFAULT_NEGATIVE_PROMPT,
    DTYPES,
    compose_character_prompt,
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
    steps: int
    guidance_scale: float
    true_cfg_scale: float
    controlnet_conditioning_scale: float
    control_guidance_start: float
    control_guidance_end: float
    dtype: str
    device: str
    cpu_offload: bool
    transformer_single_file: str | None
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
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "true_cfg_scale": self.true_cfg_scale,
            "controlnet_conditioning_scale": self.controlnet_conditioning_scale,
            "control_guidance_start": self.control_guidance_start,
            "control_guidance_end": self.control_guidance_end,
            "dtype": self.dtype,
            "device": self.device,
            "cpu_offload": self.cpu_offload,
            "transformer_single_file": self.transformer_single_file,
            "seed": self.seed,
        }


class CharacterKontextPoseError(RuntimeError):
    pass


class CharacterKontextPoseDependencyError(CharacterKontextPoseError):
    pass


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
    width: int = 768,
    height: int = 1152,
    framing: str = "full-body",
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 1,
    cpu_offload: bool = False,
    controlnet_conditioning_scale: float = 0.65,
    control_guidance_start: float = 0.0,
    control_guidance_end: float = 0.65,
    transformer_single_file: Path | None = None,
) -> CharacterKontextPoseResult:
    torch, pipeline_class, controlnet_class, transformer_class, load_image = _load_flux_kontext_controlnet()
    pipeline = _build_kontext_pose_pipeline(
        pipeline_class,
        controlnet_class,
        transformer_class,
        model,
        controlnet_model,
        transformer_single_file,
        _torch_dtype(torch, dtype),
        device,
        cpu_offload,
    )

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
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_start=control_guidance_start,
        control_guidance_end=control_guidance_end,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).images[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
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
        steps=steps,
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_start=control_guidance_start,
        control_guidance_end=control_guidance_end,
        dtype=dtype,
        device=device,
        cpu_offload=cpu_offload,
        transformer_single_file=transformer_single_file.resolve().as_posix() if transformer_single_file else None,
        seed=seed,
    )


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


def _build_kontext_pose_pipeline(
    pipeline_class: Any,
    controlnet_class: Any,
    transformer_class: Any,
    model: str,
    controlnet_model: str,
    transformer_single_file: Path | None,
    torch_dtype: Any,
    device: str,
    cpu_offload: bool,
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

    pipeline = pipeline_class.from_pretrained(model, **pipeline_kwargs)
    pipeline.vae.enable_tiling()
    pipeline.vae.enable_slicing()
    if cpu_offload:
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
            PREFERRED_KONTEXT_RESOLUTIONS,
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
        model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->controlnet->transformer->vae"
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

        @torch.no_grad()
        def __call__(
            self,
            *,
            image: Any,
            control_image: Any,
            prompt: str,
            negative_prompt: str,
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
            _auto_resize: bool = True,
        ) -> Any:
            aspect_ratio = width / height
            width = round((max_area * aspect_ratio) ** 0.5)
            height = round((max_area / aspect_ratio) ** 0.5)

            multiple_of = self.vae_scale_factor * 2
            width = width // multiple_of * multiple_of
            height = height // multiple_of * multiple_of

            self.check_inputs(
                prompt,
                None,
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

            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

            do_true_cfg = true_cfg_scale > 1
            if do_true_cfg:
                negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = self.encode_prompt(
                    prompt=negative_prompt,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                )

            img = image[0] if isinstance(image, list) else image
            image_height, image_width = self.image_processor.get_default_height_width(img)
            image_aspect_ratio = image_width / image_height
            if _auto_resize:
                _, image_width, image_height = min(
                    (abs(image_aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
                )
            image_width = image_width // multiple_of * multiple_of
            image_height = image_height // multiple_of * multiple_of
            image = self.image_processor.resize(image, image_height, image_width)
            image = self.image_processor.preprocess(image, image_height, image_width)

            num_channels_latents = self.transformer.config.in_channels // 4
            latents, image_latents, generated_img_ids, reference_img_ids = self.prepare_latents(
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
            combined_img_ids = torch.cat([generated_img_ids, reference_img_ids], dim=0)

            control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.vae.dtype,
            )
            height, width = control_image.shape[-2:]
            controlnet_blocks_repeat = bool(self.controlnet.input_hint_block)
            if not self.controlnet.input_hint_block:
                control_image = retrieve_latents(self.vae.encode(control_image), generator=generator)
                control_image = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                height_control_image, width_control_image = control_image.shape[2:]
                control_image = self._pack_latents(
                    control_image,
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height_control_image,
                    width_control_image,
                )

            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
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
                num_inference_steps,
                device,
                sigmas=sigmas,
                mu=mu,
            )
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)

            controlnet_keep = [
                1.0
                - float(i / len(timesteps) < control_guidance_start or (i + 1) / len(timesteps) > control_guidance_end)
                for i in range(len(timesteps))
            ]

            transformer_guidance = None
            if self.transformer.config.guidance_embeds:
                transformer_guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                transformer_guidance = transformer_guidance.expand(latents.shape[0])

            controlnet_guidance = None
            if self.controlnet.config.guidance_embeds:
                controlnet_guidance = torch.tensor([guidance_scale], device=device)
                controlnet_guidance = controlnet_guidance.expand(latents.shape[0])

            self.scheduler.set_begin_index(0)
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)
                    cond_scale = controlnet_conditioning_scale * controlnet_keep[i]

                    controlnet_block_samples = None
                    controlnet_single_block_samples = None
                    if cond_scale:
                        controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
                            hidden_states=latents,
                            controlnet_cond=control_image,
                            controlnet_mode=None,
                            conditioning_scale=cond_scale,
                            timestep=timestep / 1000,
                            guidance=controlnet_guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=generated_img_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )
                        total_image_tokens = latent_model_input.shape[1]
                        controlnet_block_samples = extend_control_residuals(
                            controlnet_block_samples,
                            total_image_tokens,
                        )
                        if controlnet_single_block_samples:
                            controlnet_single_block_samples = extend_control_residuals(
                                controlnet_single_block_samples,
                                total_image_tokens,
                            )

                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=transformer_guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        txt_ids=text_ids,
                        img_ids=combined_img_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                    if do_true_cfg:
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=transformer_guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            controlnet_block_samples=controlnet_block_samples,
                            controlnet_single_block_samples=controlnet_single_block_samples,
                            txt_ids=negative_text_ids,
                            img_ids=combined_img_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            controlnet_blocks_repeat=controlnet_blocks_repeat,
                        )[0]
                        neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                        noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            self._current_timestep = None

            if output_type == "latent":
                output = latents
            else:
                latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                output = self.vae.decode(latents, return_dict=False)[0]
                output = self.image_processor.postprocess(output, output_type=output_type)

            self.maybe_free_model_hooks()

            if not return_dict:
                return (output,)
            return FluxPipelineOutput(images=output)

    return torch, FluxKontextControlNetPipeline, FluxControlNetModel, FluxTransformer2DModel, load_image


def _torch_dtype(torch_module: Any, dtype: str) -> Any:
    dtype_name = DTYPES[dtype]
    if dtype == "auto":
        return None
    return getattr(torch_module, dtype_name)
