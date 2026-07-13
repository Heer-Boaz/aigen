from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

from aigen.generation.image_caption_qwen25 import (
    QWEN25_VL_REVISION,
    caption_image,
)
from aigen.generation.sd_pixl_sdxl import (
    CANNY_CONTROLNET_REVISION,
    DEPTH_CONTROLNET_REVISION,
    DPT_REVISION,
    SEMANTIC_CANVAS_SIZE,
    SDXL_REVISION,
    TAESDXL_REVISION,
    SdPixlError,
    SdPixlSdxlRuntime,
    build_control_images,
)
from aigen.progress import StatusReporter


@dataclass(frozen=True)
class PixelArtResult:
    input: str
    output: str
    prompt: str
    caption_model_revision: str | None
    width: int
    height: int
    colors: int
    steps: int
    seed: int
    elapsed_seconds: float
    peak_vram_mb: int

    def to_json(self) -> dict[str, Any]:
        models = {
            "sdxl_revision": SDXL_REVISION,
            "taesdxl_revision": TAESDXL_REVISION,
            "canny_controlnet_revision": CANNY_CONTROLNET_REVISION,
            "depth_controlnet_revision": DEPTH_CONTROLNET_REVISION,
            "dpt_revision": DPT_REVISION,
        }
        if self.caption_model_revision is not None:
            models["caption_model_revision"] = self.caption_model_revision
        return {
            "status": "ok",
            "input": self.input,
            "output": self.output,
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "colors": self.colors,
            "steps": self.steps,
            "seed": self.seed,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "peak_vram_mb": self.peak_vram_mb,
            "models": models,
        }


class PixelPaletteRenderer(nn.Module):
    def __init__(self, palette: torch.Tensor, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("palette", palette)
        self.logits = nn.Parameter(logits)

    def forward(self, *, tau: float) -> torch.Tensor:
        probabilities = F.gumbel_softmax(self.logits, tau=tau, hard=False, dim=-1)
        return probabilities @ self.palette

    @torch.no_grad()
    def balance(self) -> None:
        self.logits.sub_(self.logits.mean(dim=-1, keepdim=True))

    @torch.no_grad()
    def indices(self) -> np.ndarray:
        return self.logits.argmax(dim=-1).byte().cpu().numpy()


def convert_to_pixel_art(
    input_path: Path,
    output_path: Path,
    *,
    prompt: str | None,
    width: int,
    height: int,
    colors: int,
    steps: int,
    save_every: int,
    seed: int,
    reference_images: Sequence[Path],
    progress: StatusReporter,
) -> PixelArtResult:
    _validate_request(
        input_path,
        prompt=prompt,
        width=width,
        height=height,
        colors=colors,
        steps=steps,
        save_every=save_every,
    )
    started = time.monotonic()
    caption_model_revision = None
    if prompt is None or not prompt.strip():
        progress.phase("caption input")
        prompt = caption_image(input_path, reference_images=reference_images)
        caption_model_revision = QWEN25_VL_REVISION
    else:
        prompt = prompt.strip()
    progress.phase("initialize palette")
    source = _load_source(input_path, width=width, height=height)
    palette, initial_logits = _initialize_palette_logits(
        source,
        width=width,
        height=height,
        colors=colors,
        seed=seed,
    )
    progress.phase("build Canny and depth controls")
    control_images = build_control_images(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runtime: SdPixlSdxlRuntime | None = None
    try:
        progress.phase("load SDXL")
        runtime = SdPixlSdxlRuntime(prompt, control_images, seed=seed)
        renderer = PixelPaletteRenderer(palette, initial_logits).to(runtime.device)
        optimizer = torch.optim.AdamW(
            renderer.parameters(),
            lr=0.025,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        )
        from diffusers.optimization import get_scheduler

        learning_rate = get_scheduler(
            "constant_with_warmup",
            optimizer=optimizer,
            num_warmup_steps=min(250, steps),
            num_training_steps=steps,
        )
        fft_mask = _fft_mask(height, width, runtime.device)
        tau_rng = np.random.default_rng(seed)

        progress.begin(steps, "optimize native raster")
        for step in range(steps):
            tau = float(tau_rng.uniform(0.5, 1.5))
            rendered = renderer(tau=tau)
            raster = F.interpolate(
                rendered.permute(2, 0, 1).unsqueeze(0),
                size=(SEMANTIC_CANVAS_SIZE, SEMANTIC_CANVAS_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            loss = runtime.score_distillation_loss(
                raster,
                step=step,
                total_steps=steps,
            )
            loss = loss + 20.0 * _fft_loss(rendered, fft_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(renderer.parameters(), 1.0)
            optimizer.step()
            learning_rate.step()
            renderer.balance()
            completed_steps = step + 1
            if save_every and completed_steps % save_every == 0:
                _save_indexed_png(
                    _snapshot_path(output_path, completed_steps),
                    renderer.indices(),
                    palette,
                )
            progress.step(f"optimize {step + 1}/{steps}")

        torch.cuda.synchronize(runtime.device)
        peak_vram_mb = runtime.peak_vram_mb
        indices = renderer.indices()
        _save_indexed_png(output_path, indices, palette)
    except torch.cuda.OutOfMemoryError as error:
        raise SdPixlError(
            "SD-piXL ran out of CUDA memory; free GPU memory before running it"
        ) from error
    finally:
        if runtime is not None:
            runtime.close()

    return PixelArtResult(
        input=input_path.resolve().as_posix(),
        output=output_path.resolve().as_posix(),
        prompt=prompt,
        caption_model_revision=caption_model_revision,
        width=width,
        height=height,
        colors=palette.shape[0],
        steps=steps,
        seed=seed,
        elapsed_seconds=time.monotonic() - started,
        peak_vram_mb=peak_vram_mb,
    )


def _load_source(path: Path, *, width: int, height: int) -> Image.Image:
    try:
        with Image.open(path) as image:
            source = ImageOps.exif_transpose(image).convert("RGB")
            canvas_width, canvas_height = _semantic_cell_size(width, height)
            contained = ImageOps.contain(
                source,
                (canvas_width, canvas_height),
                method=Image.Resampling.BILINEAR,
            )
            left = (canvas_width - contained.width) // 2
            right = canvas_width - contained.width - left
            top = (canvas_height - contained.height) // 2
            bottom = canvas_height - contained.height - top
            cell = np.pad(
                np.asarray(contained),
                ((top, bottom), (left, right), (0, 0)),
                mode="edge",
            )
            return Image.fromarray(cell).resize(
                (SEMANTIC_CANVAS_SIZE, SEMANTIC_CANVAS_SIZE),
                Image.Resampling.BILINEAR,
            )
    except OSError as error:
        raise SdPixlError(f"cannot read input image {path}: {error}") from error


def _semantic_cell_size(width: int, height: int) -> tuple[int, int]:
    if width <= height:
        return round(SEMANTIC_CANVAS_SIZE * width / height), SEMANTIC_CANVAS_SIZE
    return SEMANTIC_CANVAS_SIZE, round(SEMANTIC_CANVAS_SIZE * height / width)


def _initialize_palette_logits(
    source: Image.Image,
    *,
    width: int,
    height: int,
    colors: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = np.asarray(source, dtype=np.uint8)
    from sklearn.cluster import KMeans

    flattened = pixels.reshape(-1, 3)
    kmeans = KMeans(
        n_clusters=colors,
        n_init=4,
        random_state=seed,
    ).fit(flattened)
    palette_u8 = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)
    palette = torch.from_numpy(palette_u8.astype(np.float32) / 255.0)
    labels = kmeans.labels_.reshape(
        SEMANTIC_CANVAS_SIZE,
        SEMANTIC_CANVAS_SIZE,
    )
    selected_indices = _majority_cell_indices(
        labels,
        width=width,
        height=height,
        colors=colors,
    )
    selected_colors = palette[torch.from_numpy(selected_indices)]
    logits = 1.0 - torch.abs(
        selected_colors.unsqueeze(-2)
        - palette.view(1, 1, colors, 3)
    ).sum(dim=-1)
    logits.mul_(math.sqrt(colors))
    logits.sub_(logits.mean(dim=-1, keepdim=True))
    return palette, logits


def _majority_cell_indices(
    labels: np.ndarray,
    *,
    width: int,
    height: int,
    colors: int,
) -> np.ndarray:
    scale_y = SEMANTIC_CANVAS_SIZE // height
    scale_x = SEMANTIC_CANVAS_SIZE // width
    top = (SEMANTIC_CANVAS_SIZE % height) // 2
    left = (SEMANTIC_CANVAS_SIZE % width) // 2
    labels = labels[
        top : top + height * scale_y,
        left : left + width * scale_x,
    ]
    cell_rows = np.repeat(np.arange(height, dtype=np.int64), scale_y)
    cell_columns = np.repeat(np.arange(width, dtype=np.int64), scale_x)
    cell_ids = (cell_rows[:, None] * width + cell_columns[None, :]).reshape(-1)
    counts = np.bincount(
        cell_ids * colors + labels.reshape(-1),
        minlength=height * width * colors,
    ).reshape(height, width, colors)
    return counts.argmax(axis=-1)


def _fft_mask(height: int, width: int, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    distance = torch.sqrt((x - width / 2.0) ** 2 + (y - height / 2.0) ** 2)
    mask = (distance >= min(height, width) / (4.0 * math.sqrt(2.0))).float()
    return mask.unsqueeze(0).expand(3, -1, -1)


def _fft_loss(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    channels = image.permute(2, 0, 1)
    magnitude = torch.abs(torch.fft.fftshift(torch.fft.fft2(channels)))
    return (magnitude * mask).sum() / (mask.sum() + 1e-8)


def _save_indexed_png(
    output_path: Path,
    indices: np.ndarray,
    palette: torch.Tensor,
) -> None:
    image = Image.fromarray(indices)
    palette_u8 = (
        palette.detach().cpu().mul(255).round().clamp(0, 255).byte().numpy()
    )
    image.putpalette(palette_u8.reshape(-1).tolist() + [0] * (768 - palette_u8.size))
    image.save(output_path)


def _snapshot_path(output_path: Path, step: int) -> Path:
    return output_path.with_name(
        f"{output_path.stem}.step-{step:06d}{output_path.suffix}"
    )


def _validate_request(
    input_path: Path,
    *,
    prompt: str | None,
    width: int,
    height: int,
    colors: int,
    steps: int,
    save_every: int,
) -> None:
    if not input_path.is_file():
        raise SdPixlError(f"input image does not exist: {input_path}")
    if (
        not 1 <= width <= SEMANTIC_CANVAS_SIZE
        or not 1 <= height <= SEMANTIC_CANVAS_SIZE
    ):
        raise SdPixlError("width and height must be between 1 and 1024 pixels")
    if not 2 <= colors <= 256:
        raise SdPixlError("colors must be between 2 and 256")
    if steps < 1:
        raise SdPixlError("steps must be at least 1")
    if save_every < 0:
        raise SdPixlError("save-every must be zero or greater")
