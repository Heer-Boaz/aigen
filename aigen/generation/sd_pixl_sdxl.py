from __future__ import annotations

import gc
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from aigen.runtime_profiles import MODELS_ROOT


SDXL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
TAESDXL_REVISION = "b20258aaef75ef61e659c1e0f14f251cf0ad153e"
CANNY_CONTROLNET_REVISION = "4d3e948836782759082d16985b139cfcae06109d"
DEPTH_CONTROLNET_REVISION = "ba71fc551635921528fc9e2856fd464046d3448e"
DPT_REVISION = "11eaf7a1cf4bd70740697dbc216f98980c0aeb03"
SDXL_MODEL_ROOT = MODELS_ROOT / "sd_pixl/stabilityai/stable-diffusion-xl-base-1.0"
TAESDXL_MODEL_ROOT = MODELS_ROOT / "sd_pixl/madebyollin/taesdxl"
CANNY_CONTROLNET_ROOT = (
    MODELS_ROOT / "sd_pixl/diffusers/controlnet-canny-sdxl-1.0-mid"
)
DEPTH_CONTROLNET_ROOT = (
    MODELS_ROOT / "sd_pixl/diffusers/controlnet-depth-sdxl-1.0-mid"
)
DPT_MODEL_ROOT = MODELS_ROOT / "sd_pixl/Intel/dpt-hybrid-midas"
SEMANTIC_CANVAS_SIZE = 1024
CONTROLNET_CONDITIONING_SCALE = 0.15

_SDXL_REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
)
_TAESDXL_REQUIRED_FILES = (
    "config.json",
    "diffusion_pytorch_model.safetensors",
)
_CONTROLNET_REQUIRED_FILES = (
    "config.json",
    "diffusion_pytorch_model.fp16.safetensors",
)
_DPT_REQUIRED_FILES = (
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
)


class SdPixlError(RuntimeError):
    pass


def build_control_images(source: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cuda()
    _require_model_files(DPT_MODEL_ROOT, _DPT_REQUIRED_FILES)

    blurred = source.filter(ImageFilter.GaussianBlur(radius=1))
    edges = cv2.Canny(np.asarray(blurred), 100, 200)
    canny = torch.from_numpy(edges).unsqueeze(0).repeat(3, 1, 1).float().div_(255)

    from transformers import DPTForDepthEstimation, DPTImageProcessor

    device = torch.device("cuda")
    processor = DPTImageProcessor.from_pretrained(
        DPT_MODEL_ROOT,
        local_files_only=True,
    )
    estimator = DPTForDepthEstimation.from_pretrained(
        DPT_MODEL_ROOT,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    inputs = processor(images=source, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        depth = estimator(inputs).predicted_depth
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(SEMANTIC_CANVAS_SIZE, SEMANTIC_CANVAS_SIZE),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = depth.amin(dim=(1, 2, 3), keepdim=True)
        depth_max = depth.amax(dim=(1, 2, 3), keepdim=True)
        depth = (depth - depth_min) / (depth_max - depth_min)
        depth = depth.repeat(1, 3, 1, 1).squeeze(0).float().cpu()

    del inputs
    del estimator
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    return canny, depth


class SdPixlSdxlRuntime:
    def __init__(
        self,
        prompt: str,
        control_images: tuple[torch.Tensor, torch.Tensor],
        *,
        seed: int,
    ) -> None:
        _require_cuda()
        _require_model_files(SDXL_MODEL_ROOT, _SDXL_REQUIRED_FILES)
        _require_model_files(TAESDXL_MODEL_ROOT, _TAESDXL_REQUIRED_FILES)
        _require_model_files(CANNY_CONTROLNET_ROOT, _CONTROLNET_REQUIRED_FILES)
        _require_model_files(DEPTH_CONTROLNET_ROOT, _CONTROLNET_REQUIRED_FILES)

        from diffusers import (
            AutoencoderTiny,
            ControlNetModel,
            DDPMScheduler,
            StableDiffusionXLPipeline,
        )
        from diffusers.models.attention_processor import AttnProcessor2_0

        self.device = torch.device("cuda")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(self.device)

        vae = AutoencoderTiny.from_pretrained(
            TAESDXL_MODEL_ROOT,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        scheduler = DDPMScheduler.from_pretrained(
            SDXL_MODEL_ROOT,
            subfolder="scheduler",
            local_files_only=True,
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL_ROOT,
            vae=vae,
            scheduler=scheduler,
            variant="fp16",
            torch_dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            add_watermarker=False,
        )

        pipeline.text_encoder.to(self.device)
        pipeline.text_encoder_2.to(self.device)
        with torch.no_grad():
            (
                prompt_embeddings,
                negative_prompt_embeddings,
                pooled_prompt_embeddings,
                negative_pooled_prompt_embeddings,
            ) = pipeline.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt="",
            )
            time_ids = pipeline._get_add_time_ids(
                (SEMANTIC_CANVAS_SIZE, SEMANTIC_CANVAS_SIZE),
                (0, 0),
                (SEMANTIC_CANVAS_SIZE, SEMANTIC_CANVAS_SIZE),
                dtype=prompt_embeddings.dtype,
                text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim,
            ).to(self.device)

        self.prompt_embeddings = torch.cat(
            (negative_prompt_embeddings, prompt_embeddings), dim=0
        )
        self.added_conditioning = {
            "text_embeds": torch.cat(
                (negative_pooled_prompt_embeddings, pooled_prompt_embeddings), dim=0
            ),
            "time_ids": time_ids.repeat(2, 1),
        }

        unet = pipeline.unet
        pipeline.text_encoder = None
        pipeline.text_encoder_2 = None
        pipeline.tokenizer = None
        pipeline.tokenizer_2 = None
        pipeline.unet = None
        pipeline.vae = None
        pipeline.scheduler = None
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        self.unet = unet.requires_grad_(False).to(self.device, dtype=torch.float16)
        self.unet.set_attn_processor(AttnProcessor2_0())
        self.vae = vae.requires_grad_(False).to(self.device, dtype=torch.float16)
        self.controlnets = tuple(
            ControlNetModel.from_pretrained(
                root,
                variant="fp16",
                torch_dtype=torch.float16,
                local_files_only=True,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            .requires_grad_(False)
            .to(self.device)
            for root in (CANNY_CONTROLNET_ROOT, DEPTH_CONTROLNET_ROOT)
        )
        for controlnet in self.controlnets:
            controlnet.set_attn_processor(AttnProcessor2_0())
        self.control_images = tuple(
            image.unsqueeze(0).to(self.device, dtype=torch.float16)
            for image in control_images
        )
        self.scheduler = scheduler
        self.alphas_cumprod = scheduler.alphas_cumprod.to(
            self.device, dtype=torch.float32
        )
        self.num_train_timesteps = scheduler.config.num_train_timesteps

    def score_distillation_loss(
        self,
        raster: torch.Tensor,
        *,
        step: int,
        total_steps: int,
        guidance_scale: float = 40.0,
    ) -> torch.Tensor:
        raster, control_images = self._augment_with_controls(raster)
        with torch.autocast("cuda", dtype=torch.float16):
            normalized = (2.0 * raster - 1.0).clamp(-1.0, 1.0)
            latents = self.vae.encode(normalized).latents
            latents = latents * self.vae.config.scaling_factor
            latents = latents * self.scheduler.init_noise_sigma
            if latents.shape[-2:] != (128, 128):
                latents = F.interpolate(
                    latents,
                    size=(128, 128),
                    mode="bilinear",
                    align_corners=True,
                )

        min_step = int(self.num_train_timesteps * 0.02)
        bound_steps = max(1, int(total_steps * 0.5))
        progress = min(step, bound_steps) / bound_steps
        max_fraction = 0.98 - (0.98 - 0.8) * progress
        max_step = int(self.num_train_timesteps * max_fraction)
        timestep = torch.randint(
            min_step,
            max_step + 1,
            (1,),
            device=self.device,
            dtype=torch.long,
        )

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(latents, noise, timestep)
            model_input = torch.cat((noisy_latents, noisy_latents), dim=0)
            model_input = self.scheduler.scale_model_input(model_input, timestep)
            down_residuals = None
            mid_residual = None
            for controlnet, control_image in zip(
                self.controlnets,
                control_images,
                strict=True,
            ):
                current_down, current_mid = controlnet(
                    model_input,
                    timestep,
                    encoder_hidden_states=self.prompt_embeddings,
                    controlnet_cond=control_image.repeat(2, 1, 1, 1),
                    conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                    guess_mode=False,
                    added_cond_kwargs=self.added_conditioning,
                    return_dict=False,
                )
                if down_residuals is None:
                    down_residuals = list(current_down)
                    mid_residual = current_mid
                else:
                    for residual, current in zip(
                        down_residuals,
                        current_down,
                        strict=True,
                    ):
                        residual.add_(current)
                    mid_residual.add_(current_mid)
            noise_prediction = self.unet(
                model_input,
                timestep,
                encoder_hidden_states=self.prompt_embeddings,
                down_block_additional_residuals=down_residuals,
                mid_block_additional_residual=mid_residual,
                added_cond_kwargs=self.added_conditioning,
                return_dict=False,
            )[0]

        negative_prediction, positive_prediction = noise_prediction.chunk(2)
        weight = 1.0 - self.alphas_cumprod[timestep]
        gradient = weight * (
            (positive_prediction.float() - noise.float())
            + guidance_scale
            * (positive_prediction.float() - negative_prediction.float())
        )
        gradient = torch.nan_to_num(gradient)
        target = (latents.float() - gradient).detach()
        return 0.5 * F.mse_loss(latents.float(), target, reduction="sum")

    def _augment_with_controls(
        self,
        raster: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from torchvision.transforms import InterpolationMode, RandomPerspective
        from torchvision.transforms import functional as TF

        images = torch.cat((raster, *self.control_images), dim=0)
        if torch.rand(()) < 0.2:
            images = TF.rgb_to_grayscale(images, num_output_channels=3)
        if torch.rand(()) < 0.5:
            startpoints, endpoints = RandomPerspective.get_params(
                SEMANTIC_CANVAS_SIZE,
                SEMANTIC_CANVAS_SIZE,
                distortion_scale=0.3,
            )
            images = TF.perspective(
                images,
                startpoints,
                endpoints,
                interpolation=InterpolationMode.BILINEAR,
            )
        if torch.rand(()) < 0.5:
            images = TF.hflip(images)
        return images[:1], (images[1:2], images[2:3])

    @property
    def peak_vram_mb(self) -> int:
        return round(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024))

    def close(self) -> None:
        del self.unet
        del self.vae
        del self.controlnets
        del self.control_images
        del self.prompt_embeddings
        del self.added_conditioning
        del self.alphas_cumprod
        del self.scheduler
        gc.collect()
        torch.cuda.empty_cache()


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SdPixlError("SD-piXL requires an available CUDA GPU")


def _require_model_files(root: Path, relative_paths: tuple[str, ...]) -> None:
    missing = [
        relative_path
        for relative_path in relative_paths
        if not (root / relative_path).is_file()
    ]
    if missing:
        raise SdPixlError(
            f"missing SD-piXL model files under {root}; run scripts/download_sd_pixl.sh"
        )
