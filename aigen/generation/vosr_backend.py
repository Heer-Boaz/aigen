from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from aigen.image_assets import image_asset_json
from aigen.progress import StatusReporter
from aigen.runtime_profiles import MODELS_ROOT
from aigen.generation.vosr_runtime import upscale_vosr_images


VOSR_SOURCE_REVISION = "25fbf8e6cb9656b8991c24474f408bdce6fcb1b1"
VOSR_MODEL_REVISION = "bf5da48b5eaa4affe063e4b53b3e5d70d532615b"
VOSR_MODEL_ROOT = MODELS_ROOT / "vosr/CSWRY/VOSR/preset/ckpts"
VOSR_CHECKPOINT = VOSR_MODEL_ROOT / "VOSR_1.4B_ms"
VOSR_VAE = VOSR_MODEL_ROOT / "Qwen-Image-vae-2d"
VOSR_DINOV2_SOURCE = VOSR_MODEL_ROOT / "torch_cache/facebookresearch_dinov2_main"
VOSR_DINOV2_WEIGHTS = VOSR_MODEL_ROOT / "torch_cache/checkpoints/dinov2_vitl14_pretrain.pth"
VOSR_TORCH_CACHE = VOSR_MODEL_ROOT / "torch_cache"
VOSR_MODEL_NAME = "VOSR-1.4B-ms"
VOSR_POSTPROCESS_NAME = "vosr-1.4b-ms-upscale"
VOSR_DEFAULT_SCALE = 2
VOSR_DEFAULT_INFER_STEPS = 25
VOSR_DEFAULT_CFG_SCALE = 1.5
VOSR_DEFAULT_WEAK_COND_STRENGTH_AELQ = 0.20
VOSR_DEFAULT_ALIGN_METHOD = "wavelet"
VOSR_DEFAULT_TILE_SIZE = 512
VOSR_DEFAULT_SEED = 42


class VosrBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class VosrUpscaleBatch:
    outputs: tuple[dict[str, Any], ...]
    elapsed_ms: float


@dataclass(frozen=True)
class _PreparedVosrFile:
    input_path: Path
    output_path: Path
    image: Image.Image
    alpha: Image.Image | None
    metadata: dict[str, Any]
    target_size: tuple[int, int]


def upscale_files_with_vosr(
    *,
    files: Sequence[tuple[Path, Path]],
    scale: int = VOSR_DEFAULT_SCALE,
    long_side: int | None = None,
    infer_steps: int = VOSR_DEFAULT_INFER_STEPS,
    cfg_scale: float = VOSR_DEFAULT_CFG_SCALE,
    weak_cond_strength_aelq: float = VOSR_DEFAULT_WEAK_COND_STRENGTH_AELQ,
    align_method: str = VOSR_DEFAULT_ALIGN_METHOD,
    tile_size: int = VOSR_DEFAULT_TILE_SIZE,
    seed: int = VOSR_DEFAULT_SEED,
    progress: StatusReporter,
) -> VosrUpscaleBatch:
    started = perf_counter()
    source_root = _vosr_source_root()
    _require_vosr_installation(source_root)
    device = _require_cuda()
    prepared = tuple(
        _prepare_vosr_file(input_path, output_path, scale, long_side)
        for input_path, output_path in files
    )

    try:
        upscaled_images = upscale_vosr_images(
            tuple((item.image, item.target_size) for item in prepared),
            source_root=source_root,
            checkpoint=VOSR_CHECKPOINT,
            vae_path=VOSR_VAE,
            torch_cache=VOSR_TORCH_CACHE,
            infer_steps=infer_steps,
            cfg_scale=cfg_scale,
            weak_cond_strength_aelq=weak_cond_strength_aelq,
            align_method=align_method,
            tile_size=tile_size,
            seed=seed,
            progress=progress,
        )
    except RuntimeError as error:
        detail = str(error)
        if "out of memory" in detail.lower():
            raise VosrBackendError("Insufficient VRAM for VOSR-1.4B-ms") from error
        if "cuda" in detail.lower():
            raise VosrBackendError(f"VOSR CUDA failure: {detail}") from error
        raise
    outputs = []
    for item, output in zip(prepared, upscaled_images, strict=True):
        if item.alpha is not None:
            output.putalpha(item.alpha.resize(output.size, Image.Resampling.LANCZOS))
        item.output_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(item.output_path, **item.metadata)
        outputs.append(
            {
                "status": "completed",
                "kind": "vosr-upscale-result",
                "input": image_asset_json(item.input_path),
                "output": image_asset_json(item.output_path),
                "backend": "aigen-vosr",
                "source_revision": VOSR_SOURCE_REVISION,
                "model_revision": VOSR_MODEL_REVISION,
                "model": VOSR_MODEL_NAME,
                "device": device,
                "scale": item.target_size[0] / item.image.width,
                "long_side": long_side,
                "target_width": item.target_size[0],
                "target_height": item.target_size[1],
                "infer_steps": infer_steps,
                "cfg_scale": cfg_scale,
                "weak_cond_strength_aelq": weak_cond_strength_aelq,
                "align_method": align_method,
                "tile_size": tile_size,
                "seed": seed,
                "alpha_preserved": item.alpha is not None,
            }
        )
    return VosrUpscaleBatch(
        outputs=tuple(outputs),
        elapsed_ms=(perf_counter() - started) * 1000.0,
    )


def _prepare_vosr_file(
    input_path: Path,
    output_path: Path,
    scale: int,
    long_side: int | None,
) -> _PreparedVosrFile:
    resolved_input = input_path.resolve(strict=True)
    resolved_output = output_path.resolve()
    with Image.open(resolved_input) as source:
        source.load()
        alpha = source.getchannel("A") if "A" in source.getbands() else None
        metadata = _image_metadata(source)
        image = source.convert("RGB")
    if alpha is not None and resolved_output.suffix.lower() in {".jpg", ".jpeg"}:
        raise VosrBackendError("JPEG cannot preserve alpha")
    if long_side is None:
        target_size = (image.width * scale, image.height * scale)
    else:
        resize_factor = long_side / max(image.size)
        target_size = (
            round(image.width * resize_factor),
            round(image.height * resize_factor),
        )
    return _PreparedVosrFile(
        input_path=resolved_input,
        output_path=resolved_output,
        image=image,
        alpha=alpha,
        metadata=metadata,
        target_size=target_size,
    )


def _vosr_source_root() -> Path:
    root = Path(os.environ.get("AIGEN_VOSR_ROOT", Path.home() / ".cache/aigen-vosr"))
    return root.expanduser().resolve() / "VOSR"


def _require_vosr_installation(source_root: Path) -> None:
    required = (
        source_root / "inference_vosr.py",
        source_root / "models/lightningdit.py",
        VOSR_CHECKPOINT / "args.json",
        VOSR_CHECKPOINT / "checkpoints/ema_model.safetensors",
        VOSR_VAE / "config.json",
        VOSR_VAE / "diffusion_pytorch_model.safetensors",
        VOSR_DINOV2_SOURCE / "hubconf.py",
        VOSR_DINOV2_WEIGHTS,
    )
    missing = tuple(path.as_posix() for path in required if not path.is_file())
    if missing:
        raise VosrBackendError("Missing VOSR assets: " + ", ".join(missing))


def _require_cuda() -> str:
    try:
        import torch
    except ImportError as error:
        raise VosrBackendError("CUDA Torch is missing") from error
    if not torch.cuda.is_available():
        raise VosrBackendError("CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    return f"cuda:0 ({properties.name})"


def _image_metadata(image: Image.Image) -> dict[str, Any]:
    metadata = {}
    for key in ("icc_profile", "exif", "dpi"):
        if key in image.info:
            metadata[key] = image.info[key]
    if "exif" not in metadata:
        exif = image.getexif()
        if exif:
            metadata["exif"] = exif.tobytes()
    return metadata
