from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from aigen.progress import StatusReporter
from aigen.runtime_profiles import MODELS_ROOT


REALESRGAN_ANIME_X4_MODEL_NAME = "RealESRGAN_x4plus_anime_6B"
REALESRGAN_ANIME_X4_MODEL = (
    MODELS_ROOT
    / "upscale_models/amd/realesrgan-x4plus-anime-6b/RealESRGAN_x4plus_anime_6B.pth"
)
UPSCALE_MODELS: dict[str, Path] = {
    "realesrgan-anime-6b": REALESRGAN_ANIME_X4_MODEL,
    "illustrationjanai-dat2": (
        MODELS_ROOT
        / "upscale_models/halllooo/4x_illustrationJaNaiV1/4x_IllustrationJaNai_V1_DAT2_190k.pth"
    ),
    "illustrationjanai-esrgan": (
        MODELS_ROOT
        / "upscale_models/halllooo/4x_illustrationJaNaiV1/4x_IllustrationJaNai_V1_ESRGAN_135k.pth"
    ),
    "animesharp-x4": (
        MODELS_ROOT / "upscale_models/Kim2091/AnimeSharp/4x-AnimeSharp.safetensors"
    ),
}
DEFAULT_UPSCALE_MODEL = "realesrgan-anime-6b"
UPSCALE_TILE_SIZE = 512
UPSCALE_TILE_OVERLAP = 32


def upscale_model_names() -> tuple[str, ...]:
    return tuple(sorted(UPSCALE_MODELS))


def upscale_model_path(name: str) -> Path:
    if name not in UPSCALE_MODELS:
        allowed = ", ".join(sorted(UPSCALE_MODELS))
        raise ImageUpscaleError(f"Unknown upscale model {name}; expected one of: {allowed}")
    return UPSCALE_MODELS[name]


class ImageUpscaleError(RuntimeError):
    pass


class ImageUpscaleDependencyError(ImageUpscaleError):
    pass


@dataclass(frozen=True)
class UpscaledImage:
    image: Image.Image
    elapsed_ms: float
    model_name: str
    model_path: Path
    scale: float
    device: str
    source_width: int
    source_height: int
    natural_width: int
    natural_height: int
    target_width: int
    target_height: int


class RealESRGANAnimeUpscaler:
    def __init__(
        self,
        *,
        model_path: Path = REALESRGAN_ANIME_X4_MODEL,
        tile_size: int = UPSCALE_TILE_SIZE,
        tile_overlap: int = UPSCALE_TILE_OVERLAP,
    ) -> None:
        self.model_path = model_path
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.torch, self.model = _load_spandrel_image_model(model_path)
        self.scale = float(self.model.scale)

    def upscale(
        self,
        image: Image.Image,
        *,
        target_size: tuple[int, int],
        progress: StatusReporter,
    ) -> UpscaledImage:
        target_width, target_height = target_size
        if target_width < 1 or target_height < 1:
            raise ImageUpscaleError("target_size must contain positive dimensions")
        device = "cuda" if self.torch.cuda.is_available() else "cpu"
        source_width, source_height = image.size
        start = perf_counter()
        progress.phase("move anime upscaler to device")
        self.model.to(device)
        tensor = _image_to_tensor(image, torch=self.torch, device=device)
        try:
            progress.phase("upscale qwen raw image")
            natural = _upscale_tensor_tiled(
                self.model,
                tensor,
                torch=self.torch,
                scale=self.scale,
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
                progress=progress,
            )
        finally:
            del tensor
            self.model.to("cpu")
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        natural_width = int(natural.shape[-1])
        natural_height = int(natural.shape[-2])
        output = _tensor_to_image(natural, torch=self.torch)
        del natural
        if output.size != target_size:
            output = output.resize(target_size, Image.Resampling.LANCZOS)
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        gc.collect()
        return UpscaledImage(
            image=output,
            elapsed_ms=(perf_counter() - start) * 1000.0,
            model_name=self.model_path.stem,
            model_path=self.model_path,
            scale=self.scale,
            device=device,
            source_width=source_width,
            source_height=source_height,
            natural_width=natural_width,
            natural_height=natural_height,
            target_width=target_width,
            target_height=target_height,
        )


def _load_spandrel_image_model(model_path: Path) -> tuple[Any, Any]:
    if not model_path.exists():
        raise ImageUpscaleError(f"Anime upscale model is missing: {model_path.as_posix()}")
    try:
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader
    except ModuleNotFoundError as error:
        raise ImageUpscaleDependencyError("Anime upscaling requires spandrel from the generation extra") from error
    if model_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(model_path, device="cpu")
    else:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif len(state_dict) == 1:
        nested = state_dict[next(iter(state_dict))]
        if isinstance(nested, dict):
            state_dict = nested
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in state_dict:
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model = ModelLoader().load_from_state_dict(state_dict).eval()
    if not isinstance(model, ImageModelDescriptor):
        raise ImageUpscaleError(f"Upscale model is not a single-image model: {model_path.as_posix()}")
    return torch, model


def _image_to_tensor(image: Image.Image, *, torch: Any, device: str) -> Any:
    rgb = image.convert("RGB")
    array = np.array(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _tensor_to_image(tensor: Any, *, torch: Any) -> Image.Image:
    tensor = torch.clamp(tensor.squeeze(0).permute(1, 2, 0), min=0.0, max=1.0)
    array = (tensor.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _upscale_tensor_tiled(
    model: Any,
    tensor: Any,
    *,
    torch: Any,
    scale: float,
    tile_size: int,
    tile_overlap: int,
    progress: StatusReporter,
) -> Any:
    if tile_size <= tile_overlap:
        raise ImageUpscaleError("tile_size must be greater than tile_overlap")
    height = int(tensor.shape[-2])
    width = int(tensor.shape[-1])
    tiles = _tile_windows(width=width, height=height, tile_size=tile_size, overlap=tile_overlap)
    progress.begin(len(tiles), "upscale qwen raw image tiles")
    output_height = round(height * scale)
    output_width = round(width * scale)
    output = torch.zeros((1, 3, output_height, output_width), dtype=tensor.dtype, device=tensor.device)
    weights = torch.zeros_like(output)
    feather = round(tile_overlap * scale)
    with torch.inference_mode():
        for x, y, tile_width, tile_height in tiles:
            tile = tensor[:, :, y : y + tile_height, x : x + tile_width]
            tile_output = model(tile).detach()
            out_x = round(x * scale)
            out_y = round(y * scale)
            tile_output = tile_output[:, :, : output_height - out_y, : output_width - out_x]
            weight = _tile_weight(
                tile_output,
                torch=torch,
                feather=feather,
                touches_left=x == 0,
                touches_top=y == 0,
                touches_right=x + tile_width >= width,
                touches_bottom=y + tile_height >= height,
            )
            output[:, :, out_y : out_y + tile_output.shape[-2], out_x : out_x + tile_output.shape[-1]].add_(
                tile_output * weight
            )
            weights[:, :, out_y : out_y + tile_output.shape[-2], out_x : out_x + tile_output.shape[-1]].add_(weight)
            progress.step("upscaled qwen raw tile")
    return output / weights.clamp_min(1.0e-6)


def _tile_windows(*, width: int, height: int, tile_size: int, overlap: int) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (x, y, min(tile_size, width - x), min(tile_size, height - y))
        for y in _tile_positions(height, tile_size=tile_size, overlap=overlap)
        for x in _tile_positions(width, tile_size=tile_size, overlap=overlap)
    )


def _tile_positions(length: int, *, tile_size: int, overlap: int) -> tuple[int, ...]:
    if length <= tile_size:
        return (0,)
    step = tile_size - overlap
    return tuple(range(0, length - overlap, step))


def _tile_weight(
    tile_output: Any,
    *,
    torch: Any,
    feather: int,
    touches_left: bool,
    touches_top: bool,
    touches_right: bool,
    touches_bottom: bool,
) -> Any:
    weight = torch.ones_like(tile_output)
    if feather < 1:
        return weight
    horizontal_feather = min(feather, max(1, tile_output.shape[-1] // 2))
    vertical_feather = min(feather, max(1, tile_output.shape[-2] // 2))
    if not touches_left:
        for index in range(horizontal_feather):
            weight[:, :, :, index : index + 1].mul_((index + 1) / horizontal_feather)
    if not touches_right:
        for index in range(horizontal_feather):
            weight[:, :, :, -index - 1 : tile_output.shape[-1] - index].mul_((index + 1) / horizontal_feather)
    if not touches_top:
        for index in range(vertical_feather):
            weight[:, :, index : index + 1, :].mul_((index + 1) / vertical_feather)
    if not touches_bottom:
        for index in range(vertical_feather):
            weight[:, :, -index - 1 : tile_output.shape[-2] - index, :].mul_((index + 1) / vertical_feather)
    return weight
