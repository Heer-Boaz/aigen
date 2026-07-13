from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from aigen.generation.wu_pixelization_networks import (
    WuAliasNetwork,
    WuI2PNetwork,
)
from aigen.runtime_profiles import MODELS_ROOT


WU_PIXELIZATION_REVISION = "dc6d3b16f34c0329ac025f36924de6bae85d1490"
WU_PIXELIZATION_SOURCE = "https://github.com/WuZongWei6/Pixelization"
WU_PIXELIZATION_MODEL_ROOT = MODELS_ROOT / "wu_pixelization"
WU_ALIAS_WEIGHTS = WU_PIXELIZATION_MODEL_ROOT / "alias_net.pth"
WU_I2P_WEIGHTS = WU_PIXELIZATION_MODEL_ROOT / "160_net_G_A.pth"

_INTERNAL_CELL_SIZE = 4
_CELL_FEATURE = (
    233356.8125, -27387.5918, -32866.8008, 126575.0312, -181590.0156, -31543.1289, 50374.1289, 99631.4062,
    -188897.375, 138322.7031, -107266.2266, 125778.5781, 42416.1836, 139710.8594, -39614.625, -69972.6875,
    -21886.4141, 86938.4766, 31457.627, -98892.2344, -1191.5887, -61662.1719, -180121.9062, -32931.0859,
    43109.0391, 21490.1328, -153485.3281, 94259.1797, 43103.1992, -231953.8125, 52496.7422, 142697.4062,
    -34882.7852, -98740.0625, 34458.5078, -135436.3438, 11420.5488, -18895.8984, -71195.4141, 176947.2344,
    -52747.5742, 109054.6562, -28124.9473, -17736.6152, -41327.1562, 69853.3906, 79046.2656, -3923.7344,
    -5644.5229, 96586.7578, -89315.2656, -146578.0156, -61862.1484, -83956.4375, 87574.5703, -75055.0469,
    19571.8203, 79358.7891, -16501.5, -147169.2188, -97861.6797, 60442.1797, 40156.9023, 223136.3906,
    -81118.0547, -221443.6406, 54911.6914, 54735.9258, -58805.7305, -168884.4844, 40865.9609, -28627.9043,
    -18604.7227, 120274.6172, 49712.2383, 164402.7031, -53165.082, -60664.0469, -97956.1484, -121468.4062,
    -69926.1484, -4889.0151, 127367.7344, 200241.0781, -85817.7578, -143190.0625, -74049.5312, 137980.5781,
    -150788.7656, -115719.6719, -189250.125, -153069.7344, -127429.7891, -187588.25, 125264.7422, -79082.3438,
    -114144.5781, 36033.5039, -57502.2188, 80488.1562, 36501.457, -138817.5938, -22189.6523, -222146.9688,
    -73292.3984, 127717.2422, -183836.375, -105907.0859, 145422.875, 66981.2031, -9596.6699, 78099.4922,
    70226.3359, 35841.8789, -116117.6016, -150986.0156, 81622.4922, 113575.0625, 154419.4844, 53586.4141,
    118494.875, 131625.4375, -19763.1094, 75581.1172, -42750.5039, 97934.8281, 6706.7949, -101179.0078,
    83519.6172, -83054.8359, -56749.2578, -30683.6992, 54615.9492, 84061.1406, -229136.7188, -60554.0,
    8120.2622, -106468.7891, -28316.3418, -166351.3125, 47797.3984, 96013.4141, 71482.9453, -101429.9297,
    209063.3594, -3033.6882, -38952.5352, -84920.6719, -5895.1543, -18641.8105, 47884.3633, -14620.0273,
    -132898.6719, -40903.5859, 197217.375, -128599.1328, -115397.8906, -22670.7676, -78569.9688, -54559.707,
    -106855.2031, 40703.1484, 55568.3164, 60202.9844, -64757.9375, -32068.8652, 160663.3438, 72187.0703,
    -148519.5469, 162952.8906, -128048.2031, -136153.8906, -15270.373, -52766.3281, -52517.4531, 18652.1992,
    195354.2188, -136657.375, -8034.2622, -92699.6016, -129169.1406, 188479.9844, 46003.75, -93383.0781,
    -67831.6484, -66710.5469, 104338.5234, 85878.8438, -73165.2031, 95857.3203, 71213.125, 94603.1094,
    -30359.8125, -107989.2578, 99822.1719, 184626.3594, 79238.4531, -272978.9375, -137948.5781, -145245.8125,
    75359.2031, 26652.793, 50421.4141, 60784.4102, -18286.3398, -182851.9531, -87178.7969, -13131.7539,
    195674.8906, 59951.7852, 124353.7422, -36709.1758, -54575.4766, 77822.6953, 43697.4102, -64394.3438,
    113281.1797, -93987.0703, 221989.7188, 132902.5, -9538.8574, -14594.1338, 65084.9453, -12501.7227,
    130330.6875, -115123.4766, 20823.0898, 75512.4922, -75255.7422, -41936.7656, -186678.8281, -166799.9375,
    138770.625, -78969.9531, 124516.8047, -85558.5781, -69272.4375, -115539.1094, 228774.4844, -76529.3281,
    -107735.8906, -76798.8906, -194335.2812, 56530.5742, -9397.7529, 132985.8281, 163929.8438, -188517.7969,
    -141155.6406, 45071.0391, 207788.3125, -125826.1172, 8965.332, -159584.8438, 95842.4609, -76929.4688,
)


class WuPixelizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class WuPixelizationResult:
    input: str
    output: str
    cell_size: int
    width: int
    height: int
    elapsed_seconds: float
    peak_vram_mb: int

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "input": self.input,
            "output": self.output,
            "cell_size": self.cell_size,
            "width": self.width,
            "height": self.height,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "peak_vram_mb": self.peak_vram_mb,
            "model_revision": WU_PIXELIZATION_REVISION,
            "model_source": WU_PIXELIZATION_SOURCE,
        }


def pixelize_with_wu(
    input_path: Path,
    output_path: Path,
    *,
    cell_size: int,
) -> WuPixelizationResult:
    if not input_path.is_file():
        raise WuPixelizationError(f"input image does not exist: {input_path}")
    if cell_size < 2:
        raise WuPixelizationError("cell-size must be at least 2")
    missing = [
        path.as_posix()
        for path in (WU_ALIAS_WEIGHTS, WU_I2P_WEIGHTS)
        if not path.is_file()
    ]
    if missing:
        raise WuPixelizationError(
            "Wu pixelization weights are missing; run scripts/download_wu_pixelization.sh"
        )
    if not torch.cuda.is_available():
        raise WuPixelizationError("Wu pixelization requires an available CUDA GPU")

    started = time.monotonic()
    source = _load_source(input_path)
    native_width = source.width // cell_size
    native_height = source.height // cell_size
    if native_width == 0 or native_height == 0:
        raise WuPixelizationError("cell-size is larger than the prepared input image")
    prepared = source.resize(
        (
            native_width * _INTERNAL_CELL_SIZE,
            native_height * _INTERNAL_CELL_SIZE,
        ),
        Image.Resampling.BICUBIC,
    )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    i2p: WuI2PNetwork | None = None
    alias: WuAliasNetwork | None = None
    try:
        i2p = WuI2PNetwork()
        i2p_state = torch.load(
            WU_I2P_WEIGHTS,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        i2p_state = {
            name: value
            for name, value in i2p_state.items()
            if not name.startswith("PBEnc.")
        }
        i2p.load_state_dict(i2p_state, strict=True)
        del i2p_state
        alias = WuAliasNetwork()
        alias.load_state_dict(
            torch.load(
                WU_ALIAS_WEIGHTS,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            ),
            strict=True,
        )
        i2p = i2p.eval().requires_grad_(False).to(device)
        alias = alias.eval().requires_grad_(False).to(device)
        image = torch.from_numpy(
            np.asarray(prepared, dtype=np.float32).transpose(2, 0, 1).copy()
        )
        image = image.unsqueeze(0).to(device).div_(127.5).sub_(1.0)
        cell_feature = torch.tensor(
            _CELL_FEATURE,
            device=device,
            dtype=image.dtype,
        ).view(1, 256, 1, 1)
        with torch.inference_mode():
            result = alias(i2p(image, cell_feature))
        pixels = (
            result[0]
            .permute(1, 2, 0)
            .add(1.0)
            .mul(127.5)
            .clamp_(0, 255)
            .byte()
            .cpu()
            .numpy()
        )
        native = Image.fromarray(pixels).resize(
            (native_width, native_height),
            Image.Resampling.NEAREST,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        native.save(output_path)
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    except torch.cuda.OutOfMemoryError as error:
        raise WuPixelizationError(
            "Wu pixelization ran out of CUDA memory; use a larger cell-size"
        ) from error
    finally:
        del i2p
        del alias
        gc.collect()
        torch.cuda.empty_cache()

    return WuPixelizationResult(
        input=input_path.resolve().as_posix(),
        output=output_path.resolve().as_posix(),
        cell_size=cell_size,
        width=native_width,
        height=native_height,
        elapsed_seconds=time.monotonic() - started,
        peak_vram_mb=peak_vram_mb,
    )


def _load_source(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            source = ImageOps.exif_transpose(image).convert("RGB")
    except OSError as error:
        raise WuPixelizationError(f"cannot read input image {path}: {error}") from error
    while source.width > 4000 or source.height > 4000:
        source = source.resize(
            (source.width // 2, source.height // 2),
            Image.Resampling.BICUBIC,
        )
    while source.width < 128 or source.height < 128:
        source = source.resize(
            (source.width * 2, source.height * 2),
            Image.Resampling.BICUBIC,
        )
    return source
