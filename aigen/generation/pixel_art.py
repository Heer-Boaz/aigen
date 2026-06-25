from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Image-to-image pixel-art stylization, decoupled from the FLUX/Kontext stack.
#
# This is strictly img2img: an input image is mapped into the pixel-art domain
# by a GAN generator. Generality over "more than characters" is structural --
# the content comes from the input image, not from a text prompt.
#
# The pipeline is SIZE-PRESERVING and never resamples: it returns exactly the
# input resolution. Feed an already low-resolution image to get genuine low-res
# pixel art out. Downscaling and palette limiting are explicitly out of scope
# and handled by the caller.

NORM_LAYERS = ("instance", "batch")

DTYPES = {
    "auto": None,
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}


@dataclass(frozen=True)
class PixelArtResult:
    output_path: str
    backend: str
    model: str
    input_image: str
    width: int
    height: int
    norm: str
    n_blocks: int
    dtype: str
    device: str

    def to_json(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "backend": self.backend,
            "model": self.model,
            "input_image": self.input_image,
            "width": self.width,
            "height": self.height,
            "norm": self.norm,
            "n_blocks": self.n_blocks,
            "dtype": self.dtype,
            "device": self.device,
        }


class PixelArtError(RuntimeError):
    pass


class PixelArtDependencyError(PixelArtError):
    pass


class PixelArtBackendError(PixelArtError):
    pass


class PixelArtBackend(Protocol):
    """An img2img pixel-art backend maps an input image to a pixel-art image.

    Implementations own model loading. They must return a PIL.Image at exactly
    the input resolution, with no upscaling, downscaling, or palette
    quantization.
    """

    def pixelize(self, image: Any) -> Any:  # PIL.Image.Image -> PIL.Image.Image
        ...


def run_pixel_art(
    model: str,
    input_image: Path,
    output_path: Path,
    *,
    backend: str = "cyclegan",
    device: str = "cuda",
    dtype: str = "float32",
    norm: str = "instance",
    n_blocks: int = 9,
) -> PixelArtResult:
    if norm not in NORM_LAYERS:
        raise PixelArtError(f"unknown norm layer: {norm!r}; available: {NORM_LAYERS}")

    image = _load_input_image(input_image)
    source_size = image.size

    pixel_backend = _load_backend(
        backend,
        model=model,
        device=device,
        dtype=dtype,
        norm=norm,
        n_blocks=n_blocks,
    )

    result_image = pixel_backend.pixelize(image)

    if result_image.size != source_size:
        raise PixelArtBackendError(
            f"backend {backend!r} returned {result_image.size}, expected {source_size}; "
            "img2img pixel-art output must preserve the input resolution with no resampling"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_image.save(output_path)

    width, height = source_size
    return PixelArtResult(
        output_path=output_path.resolve().as_posix(),
        backend=backend,
        model=model,
        input_image=input_image.resolve().as_posix(),
        width=width,
        height=height,
        norm=norm,
        n_blocks=n_blocks,
        dtype=dtype,
        device=device,
    )


def _load_input_image(input_image: Path) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise PixelArtDependencyError(
            "pixel-art generation requires `pip install -e .[generation]`"
        ) from exc
    if not input_image.exists():
        raise PixelArtError(f"input image not found: {input_image.as_posix()}")
    return Image.open(input_image.resolve().as_posix()).convert("RGB")


def _load_backend(
    backend: str,
    *,
    model: str,
    device: str,
    dtype: str,
    norm: str,
    n_blocks: int,
) -> PixelArtBackend:
    if backend == "cyclegan":
        return _CycleGanBackend(
            model=model,
            device=device,
            dtype=dtype,
            norm=norm,
            n_blocks=n_blocks,
        )
    raise PixelArtError(
        f"unknown pixel-art backend: {backend!r}; available: 'cyclegan'"
    )


class _CycleGanBackend:
    """Pure-PyTorch CycleGAN / pix2pix ResNet generator for img2img pixelization.

    This intentionally avoids Stable Diffusion / FLUX and any model that ships
    custom CUDA ops, so it runs on Blackwell / sm_120 with the existing cu128
    PyTorch and no extra compilation. It loads a standard CycleGAN generator
    checkpoint (e.g. `latest_net_G.pth` from the junyanz CycleGAN/pix2pix repo,
    or a `.safetensors` export of that state dict).

    The generator is fully convolutional, so output resolution equals input
    resolution. Inputs are reflect-padded to a multiple of 4 (two stride-2
    downsamples) and cropped back, so arbitrary low-res sizes are supported.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str,
        dtype: str,
        norm: str,
        n_blocks: int,
    ) -> None:
        torch, F = _load_torch()
        self._torch = torch
        self._F = F
        self._device = device
        self._torch_dtype = _torch_dtype(torch, dtype) or torch.float32

        checkpoint = Path(model)
        if checkpoint.is_dir():
            checkpoint = _resolve_generator_file(checkpoint)
        if not checkpoint.exists():
            raise PixelArtBackendError(
                f"missing CycleGAN generator checkpoint at {checkpoint.as_posix()}"
            )

        generator = _build_resnet_generator(
            torch,
            input_nc=3,
            output_nc=3,
            ngf=64,
            norm=norm,
            n_blocks=n_blocks,
        )
        state_dict = _load_state_dict(torch, checkpoint)
        generator.load_state_dict(state_dict)
        generator.eval()
        self._generator = generator.to(device=device, dtype=self._torch_dtype)

    def pixelize(self, image: Any) -> Any:
        torch = self._torch
        F = self._F
        from PIL import Image

        width, height = image.size
        # PIL RGB -> [1, 3, H, W] in [-1, 1].
        tensor = (
            torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            .view(height, width, 3)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self._device, dtype=self._torch_dtype)
            .div(127.5)
            .sub(1.0)
        )

        pad_h = (-height) % 4
        pad_w = (-width) % 4
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            out = self._generator(tensor)

        out = out[..., :height, :width]
        out = out.float().clamp(-1, 1).add(1).div(2)  # -> [0, 1]
        array = (out[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        return Image.fromarray(array, mode="RGB")


def _resolve_generator_file(directory: Path) -> Path:
    for candidate in ("latest_net_G.pth", "net_G.pth", "generator.safetensors", "generator.pth"):
        if (directory / candidate).exists():
            return directory / candidate
    raise PixelArtBackendError(
        f"no CycleGAN generator checkpoint found in {directory.as_posix()} "
        "(looked for latest_net_G.pth, net_G.pth, generator.safetensors, generator.pth)"
    )


def _load_state_dict(torch: Any, checkpoint: Path) -> dict[str, Any]:
    if checkpoint.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise PixelArtDependencyError(
                "loading .safetensors requires `pip install -e .[generation]`"
            ) from exc
        state_dict = load_file(checkpoint.as_posix())
    else:
        state_dict = torch.load(checkpoint.as_posix(), map_location="cpu", weights_only=True)
    # CycleGAN saves the raw generator state dict; strip any DataParallel prefix.
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def _build_resnet_generator(
    torch: Any,
    *,
    input_nc: int,
    output_nc: int,
    ngf: int,
    norm: str,
    n_blocks: int,
) -> Any:
    import functools

    import torch.nn as nn

    if norm == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        use_bias = True
    else:
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        use_bias = False

    class ResnetBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            block = [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                norm_layer(dim),
                nn.ReLU(True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                norm_layer(dim),
            ]
            self.conv_block = nn.Sequential(*block)

        def forward(self, x: Any) -> Any:
            return x + self.conv_block(x)

    class ResnetGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            model: list[Any] = [
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                norm_layer(ngf),
                nn.ReLU(True),
            ]
            n_downsampling = 2
            for i in range(n_downsampling):
                mult = 2 ** i
                model += [
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                ]
            mult = 2 ** n_downsampling
            for i in range(n_blocks):
                model += [ResnetBlock(ngf * mult)]
            for i in range(n_downsampling):
                mult = 2 ** (n_downsampling - i)
                model += [
                    nn.ConvTranspose2d(
                        ngf * mult,
                        int(ngf * mult / 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model += [nn.Tanh()]
            self.model = nn.Sequential(*model)

        def forward(self, x: Any) -> Any:
            return self.model(x)

    return ResnetGenerator()


def _load_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise PixelArtDependencyError(
            "pixel-art generation requires `pip install -e .[generation]`"
        ) from exc
    return torch, F


def _torch_dtype(torch_module: Any, dtype: str) -> Any:
    if dtype == "auto":
        return None
    return getattr(torch_module, DTYPES[dtype])
