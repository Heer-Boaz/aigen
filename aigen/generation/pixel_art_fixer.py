from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from aigen.progress import StatusReporter


PIXEL_ART_FIXER_UPSTREAM_REVISION = "ef376e57e1c272633ca2dbf5f29ec3fcf6596465"
PIXEL_ART_FIXER_SOURCE = "https://github.com/Retro-Diffusion/pixel-art-fixer"
PIXEL_ART_FIXER_MODES = frozenset({"full", "fast"})


class PixelArtFixerError(RuntimeError):
    pass


@dataclass(frozen=True)
class PixelArtFixerResult:
    input: Path
    output: Path
    mode: str
    low_memory: bool
    force_step: float | None
    cols: int
    rows: int
    step_x: float
    step_y: float
    consensus: str
    confidence: str
    detect_seconds: float
    reconstruct_seconds: float

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "kind": "pixel-art-fixer-result",
            "input": self.input.as_posix(),
            "output": self.output.as_posix(),
            "mode": self.mode,
            "low_memory": self.low_memory,
            "force_step": self.force_step,
            "cols": self.cols,
            "rows": self.rows,
            "step_x": self.step_x,
            "step_y": self.step_y,
            "consensus": self.consensus,
            "confidence": self.confidence,
            "detect_seconds": self.detect_seconds,
            "reconstruct_seconds": self.reconstruct_seconds,
            "upstream_revision": PIXEL_ART_FIXER_UPSTREAM_REVISION,
            "source": PIXEL_ART_FIXER_SOURCE,
        }


def fix_pixel_art(
    input_path: Path,
    output_path: Path,
    *,
    mode: str,
    low_memory: bool,
    force_step: float | None,
    progress: StatusReporter,
) -> PixelArtFixerResult:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise PixelArtFixerError(f"input image does not exist: {input_path}")
    if mode not in PIXEL_ART_FIXER_MODES:
        raise PixelArtFixerError(
            f"unsupported Pixel Art Fixer mode: {mode}; expected full or fast"
        )
    if force_step is not None and force_step <= 0:
        raise PixelArtFixerError("forced pixel step must be positive")

    try:
        with Image.open(input_path) as image:
            rgba = np.asarray(ImageOps.exif_transpose(image).convert("RGBA"))
    except OSError as error:
        raise PixelArtFixerError(f"cannot read input image {input_path}: {error}") from error

    from aigen.pixelfixer.api import InputError, process

    started = time.monotonic()
    progress.phase("detect pixel grid")
    try:
        result = process(
            rgba,
            mode=mode,
            low_memory=low_memory,
            force_step=force_step,
            return_png=False,
        )
    except (InputError, ValueError, RuntimeError) as error:
        raise PixelArtFixerError(str(error)) from error
    progress.phase("write native pixel art")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result["array"], mode="RGBA").save(output_path)
    elapsed = time.monotonic() - started

    return PixelArtFixerResult(
        input=input_path,
        output=output_path,
        mode=mode,
        low_memory=low_memory,
        force_step=force_step,
        cols=int(result["cols"]),
        rows=int(result["rows"]),
        step_x=float(result["step_x"]),
        step_y=float(result["step_y"]),
        consensus=str(result["consensus"]),
        confidence=str(result["confidence"]),
        detect_seconds=float(result["detect_s"]),
        reconstruct_seconds=float(result["recon_s"]),
    )
