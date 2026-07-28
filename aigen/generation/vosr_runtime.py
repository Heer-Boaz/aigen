from __future__ import annotations

from contextlib import ExitStack, contextmanager
import gc
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from PIL import Image

from aigen.progress import StatusReporter


VAE_SCALE_FACTOR = 8


class VosrRuntime:
    def __init__(
        self,
        *,
        source_root: Path,
        checkpoint: Path,
        vae_path: Path,
        torch_cache: Path,
        infer_steps: int,
        cfg_scale: float,
        weak_cond_strength_aelq: float,
        align_method: str,
        tile_size: int,
        seed: int,
        progress: StatusReporter,
    ) -> None:
        import torch
        import torch.nn.functional as functional
        from safetensors.torch import load_file
        from torchvision import transforms

        self.torch = torch
        self.functional = functional
        self.transforms = transforms
        self.official = _official_vosr(source_root, torch_cache)
        self.args = _runtime_args(
            checkpoint / "args.json",
            checkpoint=checkpoint,
            infer_steps=infer_steps,
            cfg_scale=cfg_scale,
            weak_cond_strength_aelq=weak_cond_strength_aelq,
            align_method=align_method,
            tile_size=tile_size,
            seed=seed,
        )
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        self.align_method = align_method
        self.tile_size = tile_size
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        progress.phase("load VOSR VAE")
        vae_class = importlib.import_module("models.qwenimage_vae2d").AutoencoderKLQwenImage2D
        self.vae = (
            vae_class.from_pretrained(
                vae_path,
                torch_dtype=self.dtype,
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )

        progress.phase("load VOSR DINOv2")
        self.venc = self.official.load_dinov2(self.args, "cpu").to(
            device=self.device,
            dtype=self.dtype,
        )

        progress.phase("load VOSR diffusion model")
        self.model = (
            self.official.LightningDiT(
                input_size=self.args.resolution // VAE_SCALE_FACTOR,
                patch_size=self.args.patch_size,
                in_channels=32,
                out_channels=16,
                hidden_size=self.args.dim,
                depth=self.args.depth,
                num_heads=self.args.num_heads,
                mlp_ratio=self.args.mlp_ratio,
                z_dims=self.args.enc_dim,
                encdim_ratio=self.args.encdim_ratio,
                auxiliary_time_cond=False,
                use_qknorm=self.args.use_qknorm,
                use_swiglu=self.args.use_swiglu,
                use_rope=self.args.use_rope,
                use_rmsnorm=self.args.use_rmsnorm,
                wo_shift=self.args.wo_shift,
                num_fused_layers=len(self.args.layer_dinov2b_list),
            )
            .eval()
            .requires_grad_(False)
        )
        state_dict = load_file(
            checkpoint / "checkpoints/ema_model.safetensors",
            device="cpu",
        )
        self.model.load_state_dict(state_dict, strict=False)
        del state_dict
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.forward = self.model.forward_flexible

        self.vosr = self.official.VOSR(
            time_dist=self.args.time_dist,
            cfg_ratio=self.args.cfg_ratio,
            cfg_scale=self.args.cfg_scale,
            interp_type=self.args.interp_type,
            accelerator=SimpleNamespace(device=self.device),
            t_start=getattr(self.args, "t_start", 0.0),
            t_end=getattr(self.args, "t_end", 1.0),
            args=self.args,
        )

    def __enter__(self) -> VosrRuntime:
        self.inference_context = ExitStack()
        self.inference_context.enter_context(self.torch.inference_mode())
        self.inference_context.enter_context(
            self.torch.autocast("cuda", dtype=self.dtype)
        )
        self.inference_context.enter_context(_tiled_vae(self.vae, self.tile_size))
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.inference_context.close()
        del self.vosr
        del self.model
        del self.venc
        del self.vae
        gc.collect()
        self.torch.cuda.empty_cache()

    def upscale(
        self,
        image: Image.Image,
        *,
        target_size: tuple[int, int],
    ) -> Image.Image:
        condition_image = image.resize(target_size, Image.Resampling.BICUBIC)
        lq = (
            self.transforms.ToTensor()(condition_image)
            .unsqueeze(0)
            .mul_(2.0)
            .sub_(1.0)
        )
        pad_height = (-lq.shape[-2]) % VAE_SCALE_FACTOR
        pad_width = (-lq.shape[-1]) % VAE_SCALE_FACTOR
        if pad_height or pad_width:
            lq = self.functional.pad(
                lq,
                (0, pad_width, 0, pad_height),
                mode="replicate",
            )
        sr_tensor = self.official.tiled_latent_inference(
            self.model,
            self.vosr,
            self.vae,
            self.venc,
            lq.to(self.device),
            self.args,
            device=self.device,
        )
        sr_tensor = sr_tensor[..., : target_size[1], : target_size[0]]
        pixels = sr_tensor[0].float().cpu().mul_(0.5).add_(0.5)
        del sr_tensor
        del lq
        sr_image = self.transforms.ToPILImage()(pixels)
        del pixels
        if self.align_method == "wavelet":
            sr_image = self.official.wavelet_color_fix(sr_image, condition_image)
        elif self.align_method == "adain":
            sr_image = self.official.adain_color_fix(sr_image, condition_image)
        return sr_image


def _official_vosr(source_root: Path, torch_cache: Path) -> Any:
    if source_root.as_posix() not in sys.path:
        sys.path.insert(0, source_root.as_posix())
    official = importlib.import_module("inference_vosr")
    official.torch.hub.set_dir(torch_cache.as_posix())
    return official


def _runtime_args(
    path: Path,
    *,
    checkpoint: Path,
    infer_steps: int,
    cfg_scale: float,
    weak_cond_strength_aelq: float,
    align_method: str,
    tile_size: int,
    seed: int,
) -> SimpleNamespace:
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    values.update(
        checkpoint=checkpoint.as_posix(),
        infer_steps=infer_steps,
        cfg_scale=cfg_scale,
        weak_cond_strength_aelq=weak_cond_strength_aelq,
        weak_cond_strength_aelq_list=[weak_cond_strength_aelq, weak_cond_strength_aelq],
        align_method=align_method,
        tile_size=tile_size,
        tile_overlap=4,
        seed=seed,
    )
    return SimpleNamespace(**values)


@contextmanager
def _tiled_vae(vae: Any, tile_size: int) -> Iterator[None]:
    encoder_forward = vae.encoder.forward
    decoder_forward = vae.decoder.forward
    vae.encoder.forward = lambda tensor: _tiled_forward(
        encoder_forward,
        tensor,
        tile_size=tile_size,
        output_scale=1 / VAE_SCALE_FACTOR,
    )
    vae.decoder.forward = lambda tensor: _tiled_forward(
        decoder_forward,
        tensor,
        tile_size=max(1, tile_size // VAE_SCALE_FACTOR),
        output_scale=VAE_SCALE_FACTOR,
    )
    try:
        yield
    finally:
        vae.encoder.forward = encoder_forward
        vae.decoder.forward = decoder_forward


def _tiled_forward(
    forward: Any,
    tensor: Any,
    *,
    tile_size: int,
    output_scale: float,
) -> Any:
    import torch

    height, width = tensor.shape[-2:]
    tile_height = min(tile_size, height)
    tile_width = min(tile_size, width)
    height_positions = _tile_positions(height, tile_height)
    width_positions = _tile_positions(width, tile_width)
    output = None
    contributions = None

    for row, top in enumerate(height_positions):
        for column, left in enumerate(width_positions):
            tile = forward(tensor[..., top : top + tile_height, left : left + tile_width])
            if output is None:
                output_height = round(height * output_scale)
                output_width = round(width * output_scale)
                output = torch.zeros(
                    (tensor.shape[0], tile.shape[1], output_height, output_width),
                    dtype=tile.dtype,
                    device=tile.device,
                )
                contributions = torch.zeros(
                    (1, 1, output_height, output_width),
                    dtype=tile.dtype,
                    device=tile.device,
                )
            weight = _tile_weight(
                tile,
                top_overlap=_previous_overlap(height_positions, row, tile_height),
                bottom_overlap=_next_overlap(height_positions, row, tile_height),
                left_overlap=_previous_overlap(width_positions, column, tile_width),
                right_overlap=_next_overlap(width_positions, column, tile_width),
                output_scale=output_scale,
            )
            output_top = round(top * output_scale)
            output_left = round(left * output_scale)
            output[
                ...,
                output_top : output_top + tile.shape[-2],
                output_left : output_left + tile.shape[-1],
            ].add_(tile * weight)
            contributions[
                ..., output_top : output_top + tile.shape[-2], output_left : output_left + tile.shape[-1]
            ].add_(weight)

    return output / contributions


def _tile_positions(length: int, tile_size: int) -> tuple[int, ...]:
    if length <= tile_size:
        return (0,)
    stride = tile_size // 2
    positions = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if positions[-1] != final:
        positions.append(final)
    return tuple(positions)


def _previous_overlap(positions: tuple[int, ...], index: int, tile_size: int) -> int:
    if index == 0:
        return 0
    return positions[index - 1] + tile_size - positions[index]


def _next_overlap(positions: tuple[int, ...], index: int, tile_size: int) -> int:
    if index + 1 == len(positions):
        return 0
    return positions[index] + tile_size - positions[index + 1]


def _tile_weight(
    like: Any,
    *,
    top_overlap: int,
    bottom_overlap: int,
    left_overlap: int,
    right_overlap: int,
    output_scale: float,
) -> Any:
    import torch

    height, width = like.shape[-2:]
    vertical = torch.ones(height, dtype=torch.float32, device=like.device)
    horizontal = torch.ones(width, dtype=torch.float32, device=like.device)
    _feather_axis(vertical, round(top_overlap * output_scale), leading=True)
    _feather_axis(vertical, round(bottom_overlap * output_scale), leading=False)
    _feather_axis(horizontal, round(left_overlap * output_scale), leading=True)
    _feather_axis(horizontal, round(right_overlap * output_scale), leading=False)
    return (vertical[:, None] * horizontal[None, :]).to(dtype=like.dtype)[None, None]


def _feather_axis(axis: Any, length: int, *, leading: bool) -> None:
    import torch

    if length == 0:
        return
    ramp = torch.linspace(0.0, 1.0, length, dtype=axis.dtype, device=axis.device)
    if not leading:
        ramp = ramp.flip(0)
    section = axis[:length] if leading else axis[-length:]
    section.copy_(torch.minimum(section, ramp))
