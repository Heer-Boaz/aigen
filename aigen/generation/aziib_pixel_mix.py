from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps


SD15_CONFIG = "stable-diffusion-v1-5/stable-diffusion-v1-5"
TILE_CONTROLNET_CONFIG = "lllyasviel/control_v11f1e_sd15_tile"


class AziibPixelMixError(RuntimeError):
    pass


@dataclass(frozen=True)
class AziibPixelMixImages:
    rendered: Image.Image
    native: Image.Image
    peak_vram_mb: int


class AziibPixelMixRuntime:
    def __init__(self, checkpoint: Path, tile_controlnet: Path) -> None:
        if not checkpoint.is_file():
            raise AziibPixelMixError(f"Aziib checkpoint does not exist: {checkpoint}")
        if not tile_controlnet.is_file():
            raise AziibPixelMixError(f"tile ControlNet does not exist: {tile_controlnet}")
        if not torch.cuda.is_available():
            raise AziibPixelMixError("AziibPixelMix requires CUDA")

        from diffusers import (
            ControlNetModel,
            EulerAncestralDiscreteScheduler,
            StableDiffusionControlNetImg2ImgPipeline,
        )
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device = torch.device("cuda")
        tokenizer = CLIPTokenizer.from_pretrained(SD15_CONFIG, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            SD15_CONFIG,
            subfolder="text_encoder",
            dtype=torch.float16,
        )
        controlnet = ControlNetModel.from_single_file(
            tile_controlnet,
            config=TILE_CONTROLNET_CONFIG,
            torch_dtype=torch.float16,
        )
        self.pipeline = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            checkpoint,
            config=SD15_CONFIG,
            controlnet=controlnet,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            torch_dtype=torch.float16,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.pipeline.to(self.device)

    @torch.inference_mode()
    def convert(
        self,
        input_path: Path,
        *,
        prompt: str,
        negative_prompt: str,
        native_size: tuple[int, int],
        render_scale: int,
        strength: float,
        controlnet_strength: float,
        steps: int,
        guidance_scale: float,
        seed: int,
    ) -> AziibPixelMixImages:
        source = _load_image(input_path)
        render_size = (
            native_size[0] * render_scale,
            native_size[1] * render_scale,
        )
        prepared = ImageOps.pad(
            source,
            render_size,
            method=Image.Resampling.LANCZOS,
            color=_corner_color(source),
        )
        if max(render_size) >= 2048:
            self.pipeline.vae.enable_tiling()
        else:
            self.pipeline.vae.disable_tiling()
        torch.cuda.reset_peak_memory_stats(self.device)
        try:
            rendered = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=prepared,
                control_image=prepared,
                width=render_size[0],
                height=render_size[1],
                strength=strength,
                controlnet_conditioning_scale=controlnet_strength,
                control_guidance_start=0.0,
                control_guidance_end=0.8,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            ).images[0]
            peak_vram_mb = round(
                torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            )
        except torch.cuda.OutOfMemoryError as error:
            raise AziibPixelMixError(
                "AziibPixelMix ran out of CUDA memory"
            ) from error
        native = rendered.resize(native_size, Image.Resampling.NEAREST)
        return AziibPixelMixImages(
            rendered=rendered,
            native=native,
            peak_vram_mb=peak_vram_mb,
        )

    def close(self) -> None:
        del self.pipeline
        gc.collect()
        torch.cuda.empty_cache()


def _load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except OSError as error:
        raise AziibPixelMixError(f"cannot read input image {path}: {error}") from error


def _corner_color(image: Image.Image) -> tuple[int, int, int]:
    width, height = image.size
    corners = (
        image.getpixel((0, 0)),
        image.getpixel((width - 1, 0)),
        image.getpixel((0, height - 1)),
        image.getpixel((width - 1, height - 1)),
    )
    return tuple(
        round(sum(pixel[channel] for pixel in corners) / 4)
        for channel in range(3)
    )
