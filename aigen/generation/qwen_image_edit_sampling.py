"""Native LightX2V Qwen sampling, including source-anchored regional edits."""
from __future__ import annotations

from typing import Any

import torch
from lightx2v.models.schedulers.qwen_image.scheduler import QwenImageScheduler

from aigen.generation.flow_match_sampling import euler_ancestral_step
from aigen.image_edit_defaults import QWEN_2511_SAMPLER


class QwenEditScheduler(QwenImageScheduler):
    def __init__(self, config: Any, *, sampler: str) -> None:
        super().__init__(config)
        self.sampler = sampler
        self.clear_mask()

    def clear_mask(self) -> None:
        self.source_latents = None
        self.repaint_mask = None
        self.source_noise = None
        self.preserved_latents = None

    def prepare(self, input_info: Any) -> None:
        super().prepare(input_info)
        if self.sampler == "euler-ancestral":
            self.ancestral_generator = torch.Generator(device=self.latents.device).manual_seed(input_info.seed)

    def prepare_i2i_denoise_strength_latents(self, input_info: Any) -> None:
        # The upstream implementation selects one conditioning image and resizes
        # its latents. Our source is explicitly encoded on the output canvas;
        # conditioning remains an independent, complete multi-reference bundle.
        if self.source_latents is None or self.repaint_mask is None:
            raise ValueError("Qwen regional strength requires explicit source latents and a repaint mask")
        if self.source_latents.shape != self.latents.shape or self.repaint_mask.shape != self.latents.shape:
            raise ValueError("Qwen source and packed mask must match the generated latent shape")
        self.source_noise = self.latents
        self.latents = self.scheduler.scale_noise(self.source_latents, self.timesteps[:1], self.source_noise)
        self.preserved_latents = torch.empty_like(self.latents)

    def step_post(self) -> None:
        if self.sampler == QWEN_2511_SAMPLER:
            super().step_post()
        else:
            timestep = self.timesteps[self.step_index]
            if self.scheduler.step_index is None:
                self.scheduler._init_step_index(timestep)
            sigma_index = self.scheduler.step_index
            self.latents = euler_ancestral_step(
                self.latents, self.noise_pred,
                sigma=self.scheduler.sigmas[sigma_index],
                sigma_next=self.scheduler.sigmas[sigma_index + 1],
                final=self.step_index + 1 == len(self.timesteps),
                generator=self.ancestral_generator, torch=torch,
            )
            self.scheduler._step_index += 1
        if self.repaint_mask is not None:
            # step() has already advanced the underlying sigma index, including
            # the strength suffix's begin_index. Keep the original seed-noise.
            sigma_next = self.scheduler.sigmas[self.scheduler.step_index]
            torch.lerp(self.source_latents, self.source_noise, sigma_next.to(self.latents.dtype),
                       out=self.preserved_latents)
            torch.lerp(self.preserved_latents, self.latents, self.repaint_mask, out=self.latents)


def pack_qwen_repaint_mask(mask: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Retain all four VAE positions inside each packed transformer token."""
    latent_height, latent_width = height // 8, width // 8
    resized = torch.nn.functional.interpolate(mask, size=(latent_height, latent_width), mode="nearest")
    return QwenImageScheduler._pack_latents(
        resized.expand(-1, 16, -1, -1), mask.shape[0], 16, latent_height, latent_width,
    )
