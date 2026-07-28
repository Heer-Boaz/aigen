from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from aigen.generation.image_upscale import (
    ImageUpscaleError,
    IllustrationUpscaler,
    upscale_model_names,
    upscale_model_path,
)
from aigen.generation.vosr_backend import (
    VOSR_DEFAULT_ALIGN_METHOD,
    VOSR_DEFAULT_CFG_SCALE,
    VOSR_DEFAULT_INFER_STEPS,
    VOSR_DEFAULT_SCALE,
    VOSR_DEFAULT_SEED,
    VOSR_DEFAULT_TILE_SIZE,
    VOSR_DEFAULT_WEAK_COND_STRENGTH_AELQ,
    VOSR_POSTPROCESS_NAME,
    VosrBackendError,
    upscale_files_with_vosr,
)
from aigen.progress import StatusReporter


WU_PIXELIZATION_MODEL = "wu-pixelization"
PIXEL_ART_FIXER_MODEL = "pixel-art-fixer"
IMAGE_BATCH_DEFAULT_CELL_SIZE = 16
IMAGE_BATCH_DEFAULT_FIXER_MODE = "full"
IMAGE_BATCH_DEFAULT_LOW_MEMORY = False


class ImageBatchPostprocessError(RuntimeError):
    pass


def image_batch_postprocess_model_names() -> tuple[str, ...]:
    return (
        VOSR_POSTPROCESS_NAME,
        *upscale_model_names(),
        WU_PIXELIZATION_MODEL,
        PIXEL_ART_FIXER_MODEL,
    )


@dataclass(frozen=True)
class ImageBatchPostprocessItem:
    input: Path
    output: Path
    details: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "input": self.input.as_posix(),
            "output": self.output.as_posix(),
            "details": self.details,
        }


@dataclass(frozen=True)
class ImageBatchPostprocessResult:
    model: str
    output_dir: Path
    outputs: tuple[Path, ...]
    items: tuple[ImageBatchPostprocessItem, ...]
    elapsed_ms: float

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "completed",
            "kind": "image-batch-postprocess-result",
            "model": self.model,
            "output_dir": self.output_dir.as_posix(),
            "outputs": tuple(path.as_posix() for path in self.outputs),
            "items": tuple(item.to_json() for item in self.items),
            "elapsed_ms": self.elapsed_ms,
        }


def postprocess_image_batch(
    input_paths: Sequence[Path],
    output_dir: Path,
    *,
    model: str,
    progress: StatusReporter,
    output_names: Sequence[str] | None = None,
    long_side: int | None = None,
    scale: int = VOSR_DEFAULT_SCALE,
    infer_steps: int = VOSR_DEFAULT_INFER_STEPS,
    cfg_scale: float = VOSR_DEFAULT_CFG_SCALE,
    weak_cond_strength_aelq: float = VOSR_DEFAULT_WEAK_COND_STRENGTH_AELQ,
    align_method: str = VOSR_DEFAULT_ALIGN_METHOD,
    tile_size: int = VOSR_DEFAULT_TILE_SIZE,
    seed: int = VOSR_DEFAULT_SEED,
    cell_size: int = IMAGE_BATCH_DEFAULT_CELL_SIZE,
    mode: str = IMAGE_BATCH_DEFAULT_FIXER_MODE,
    low_memory: bool = IMAGE_BATCH_DEFAULT_LOW_MEMORY,
    force_step: float | None = None,
) -> ImageBatchPostprocessResult:
    inputs = tuple(path.expanduser().resolve(strict=True) for path in input_paths)
    if not inputs:
        raise ImageBatchPostprocessError("image batch is empty")
    names = (
        tuple(output_names)
        if output_names is not None
        else tuple(path.name for path in inputs)
    )
    if len(names) != len(inputs):
        raise ImageBatchPostprocessError(
            "output filename count does not match the image batch"
        )
    if any(not name or Path(name).name != name for name in names):
        raise ImageBatchPostprocessError(
            "output filenames must be non-empty basenames"
        )
    if len(names) != len(set(names)):
        raise ImageBatchPostprocessError(
            "image batch contains duplicate output filenames"
        )
    resolved_output_dir = output_dir.expanduser().resolve()
    if resolved_output_dir.exists():
        raise ImageBatchPostprocessError(
            f"output directory already exists: {resolved_output_dir}"
        )
    if model not in image_batch_postprocess_model_names():
        raise ImageBatchPostprocessError(f"unsupported postprocessing model: {model}")

    resolved_output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with TemporaryDirectory(
        dir=resolved_output_dir.parent,
        prefix=f".{resolved_output_dir.name}-",
    ) as temporary_dir:
        staging_dir = Path(temporary_dir)
        staged_outputs = tuple(staging_dir / name for name in names)
        details = _run_batch(
            inputs,
            staged_outputs,
            model=model,
            progress=progress,
            long_side=long_side,
            scale=scale,
            infer_steps=infer_steps,
            cfg_scale=cfg_scale,
            weak_cond_strength_aelq=weak_cond_strength_aelq,
            align_method=align_method,
            tile_size=tile_size,
            seed=seed,
            cell_size=cell_size,
            mode=mode,
            low_memory=low_memory,
            force_step=force_step,
        )
        staging_dir.replace(resolved_output_dir)

    outputs = tuple(resolved_output_dir / name for name in names)
    items = tuple(
        ImageBatchPostprocessItem(
            input=input_path,
            output=output_path,
            details=item_details,
        )
        for input_path, output_path, item_details in zip(
            inputs,
            outputs,
            details,
            strict=True,
        )
    )
    return ImageBatchPostprocessResult(
        model=model,
        output_dir=resolved_output_dir,
        outputs=outputs,
        items=items,
        elapsed_ms=(perf_counter() - started) * 1000.0,
    )


def _run_batch(
    inputs: tuple[Path, ...],
    staged_outputs: tuple[Path, ...],
    *,
    model: str,
    progress: StatusReporter,
    long_side: int | None,
    scale: int,
    infer_steps: int,
    cfg_scale: float,
    weak_cond_strength_aelq: float,
    align_method: str,
    tile_size: int,
    seed: int,
    cell_size: int,
    mode: str,
    low_memory: bool,
    force_step: float | None,
) -> tuple[dict[str, Any], ...]:
    if model == VOSR_POSTPROCESS_NAME:
        try:
            batch = upscale_files_with_vosr(
                files=tuple(zip(inputs, staged_outputs, strict=True)),
                scale=scale,
                long_side=long_side,
                infer_steps=infer_steps,
                cfg_scale=cfg_scale,
                weak_cond_strength_aelq=weak_cond_strength_aelq,
                align_method=align_method,
                tile_size=tile_size,
                seed=seed,
                progress=progress,
            )
        except VosrBackendError as error:
            raise ImageBatchPostprocessError(str(error)) from error
        return tuple(_without_paths(item) for item in batch.outputs)

    if model in upscale_model_names():
        try:
            upscaler = IllustrationUpscaler(model_path=upscale_model_path(model))
            results = upscaler.upscale_files(
                tuple(zip(inputs, staged_outputs, strict=True)),
                long_side=long_side,
                progress=progress,
            )
        except (ImageUpscaleError, OSError) as error:
            raise ImageBatchPostprocessError(str(error)) from error
        return tuple(
            {
                "model": result.model_name,
                "model_path": result.model_path.as_posix(),
                "device": result.device,
                "scale": result.scale,
                "natural_width": result.natural_width,
                "natural_height": result.natural_height,
                "target_width": result.target_width,
                "target_height": result.target_height,
                "elapsed_ms": result.elapsed_ms,
            }
            for result in results
        )

    if model == WU_PIXELIZATION_MODEL:
        from aigen.generation.wu_pixelization import (
            WuPixelizationError,
            WuPixelizer,
        )

        progress.begin(len(inputs), "pixelize image batch")
        try:
            with WuPixelizer() as pixelizer:
                results = []
                for input_path, output_path in zip(inputs, staged_outputs, strict=True):
                    result = pixelizer.pixelize(
                        input_path,
                        output_path,
                        cell_size=cell_size,
                    )
                    results.append(_without_paths(result.to_json()))
                    progress.step(f"pixelized {input_path.name}")
        except WuPixelizationError as error:
            raise ImageBatchPostprocessError(str(error)) from error
        return tuple(results)

    from aigen.generation.pixel_art_fixer import (
        PixelArtFixerError,
        fix_pixel_art,
    )

    progress.begin(len(inputs), "fix pixel-art image batch")
    try:
        results = []
        for input_path, output_path in zip(inputs, staged_outputs, strict=True):
            result = fix_pixel_art(
                input_path,
                output_path,
                mode=mode,
                low_memory=low_memory,
                force_step=force_step,
                progress=progress,
            )
            results.append(_without_paths(result.to_json()))
            progress.step(f"fixed {input_path.name}")
    except PixelArtFixerError as error:
        raise ImageBatchPostprocessError(str(error)) from error
    return tuple(results)


def _without_paths(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"input", "output"}
    }
