from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path
import shutil
from typing import Any, TextIO

import numpy as np
from PIL import Image

from aigen.command_io import command_error_payload, dump_json, write_json
from aigen.keyframe_segmentation import (
    AnimeForegroundSegmenter,
    AnimeSegmentationConfig,
    KeyframeSegmentationError,
    Sam2RegionSegmenter,
    Sam2SegmentationConfig,
    SamForegroundSegmenter,
    SamSegmentationConfig,
)
from aigen.progress import StatusReporter


SAM_ENGINES = ("sam2", "sam1", "anime")
SAM_OUTPUT_MODES = ("all", "mask", "cutout", "preview")
SAM_PROMPT_MODES = ("auto", "box", "points", "box+points")


def add_sam_command(subparsers: Any) -> None:
    command = subparsers.add_parser(
        "sam-segment",
        help="Segment an image and write mask/cutout inspection outputs",
    )
    _add_segmentation_arguments(command)
    command.add_argument(
        "--output-mode",
        choices=SAM_OUTPUT_MODES,
        default="all",
    )
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory",
    )


def _add_segmentation_arguments(command: Any) -> None:
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--engine", choices=SAM_ENGINES, default="sam2")
    command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    command.add_argument("--prompt-mode", choices=SAM_PROMPT_MODES, default="auto")
    command.add_argument(
        "--mask-candidate",
        choices=("auto", "1", "2", "3"),
        default="auto",
        help="SAM multimask candidate; auto selects the highest score",
    )
    command.add_argument("--box", type=_parse_box)
    command.add_argument("--positive-points", type=_parse_points)
    command.add_argument("--negative-points", type=_parse_points)
    command.add_argument("--threshold", type=float, default=28.0)
    command.add_argument("--grow", type=int, default=0)
    command.add_argument("--feather", type=int, default=0)
    for name in ("fill-holes", "largest-component", "invert"):
        command.add_argument(
            f"--{name}",
            choices=("false", "true"),
            default="true" if name != "invert" else "false",
        )


def run_sam_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    try:
        result = segment_image(
            args.input,
            engine=args.engine,
            device=args.device,
            prompt_mode=args.prompt_mode,
            mask_candidate=None if args.mask_candidate == "auto" else int(args.mask_candidate) - 1,
            box=args.box,
            positive_points=args.positive_points or (),
            negative_points=args.negative_points or (),
            threshold=args.threshold,
            grow=args.grow,
            feather=args.feather,
            fill_holes=args.fill_holes == "true",
            largest_component=args.largest_component == "true",
            invert=args.invert == "true",
            output_mode=args.output_mode,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            progress=progress,
        )
    except (KeyframeSegmentationError, OSError, ValueError) as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1
    dump_json(stdout, result, pretty=True)
    return 0


def segment_image(
    input_path: Path,
    *,
    engine: str,
    device: str,
    prompt_mode: str,
    mask_candidate: int | None,
    box: tuple[int, int, int, int] | None,
    positive_points: tuple[tuple[int, int], ...],
    negative_points: tuple[tuple[int, int], ...],
    threshold: float,
    grow: int,
    feather: int,
    fill_holes: bool,
    largest_component: bool,
    invert: bool,
    output_mode: str,
    output_dir: Path,
    overwrite: bool,
    progress: StatusReporter,
) -> dict[str, object]:
    if not input_path.is_file():
        raise ValueError(f"Input image does not exist: {input_path}")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise ValueError(f"Output directory exists and overwrite=false: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress.begin(3, f"load {engine}")
    image, mask = _build_mask(
        input_path,
        engine=engine,
        device=device,
        prompt_mode=prompt_mode,
        mask_candidate=mask_candidate,
        box=box,
        positive_points=positive_points,
        negative_points=negative_points,
        threshold=threshold,
        grow=grow,
        feather=feather,
        fill_holes=fill_holes,
        largest_component=largest_component,
        invert=invert,
        progress=progress,
    )
    progress.step("write outputs")
    outputs = _write_outputs(
        image,
        mask,
        input_path=input_path,
        output_dir=output_dir,
        engine=engine,
        output_mode=output_mode,
    )
    result_path = output_dir / "result.json"
    outputs["result"] = result_path.as_posix()
    result = {
        "kind": "sam-segmentation-result",
        "status": "completed",
        "engine": engine,
        "input": input_path.as_posix(),
        "mask_semantics": {
            "white": "selected foreground / repaint region",
            "black": "preserved background",
        },
        "parameters": {
            "prompt_mode": prompt_mode,
            "mask_candidate": mask_candidate,
            "box": list(box) if box is not None else None,
            "positive_points": [list(point) for point in positive_points],
            "negative_points": [list(point) for point in negative_points],
            "threshold": threshold,
            "grow": grow,
            "feather": feather,
            "fill_holes": fill_holes,
            "largest_component": largest_component,
            "invert": invert,
            "output_mode": output_mode,
        },
        "outputs": outputs,
    }
    write_json(result_path, result, pretty=True)
    progress.step("segmentation complete")
    return result


def _build_mask(
    input_path: Path,
    *,
    engine: str,
    device: str,
    prompt_mode: str,
    mask_candidate: int | None,
    box: tuple[int, int, int, int] | None,
    positive_points: tuple[tuple[int, int], ...],
    negative_points: tuple[tuple[int, int], ...],
    threshold: float,
    grow: int,
    feather: int,
    fill_holes: bool,
    largest_component: bool,
    invert: bool,
    progress: StatusReporter,
) -> tuple[Image.Image, np.ndarray]:
    if threshold <= 0:
        raise ValueError("Auto-box threshold must be greater than zero.")
    if grow < -128 or grow > 128:
        raise ValueError("Mask grow/shrink must be between -128 and 128 pixels.")
    if feather < 0 or feather > 128:
        raise ValueError("Mask feather must be between 0 and 128 pixels.")
    _validate_prompt(prompt_mode, box, positive_points)
    with Image.open(input_path) as source:
        image = source.convert("RGB")
    image_array = np.asarray(image, dtype=np.uint8)
    progress.phase(f"load {engine}")
    segmenter = _create_segmenter(engine, device)
    with closing(segmenter):
        progress.phase("segment image")
        mask = _segment(
            segmenter,
            image_array,
            input_path,
            prompt_mode=prompt_mode,
            mask_candidate=mask_candidate,
            box=box,
            positive_points=positive_points,
            negative_points=negative_points,
            threshold=threshold,
        )
        progress.phase("clean mask")
        mask = _clean_mask(
            mask,
            grow=grow,
            feather=feather,
            fill_holes=fill_holes,
            largest_component=largest_component,
            invert=invert,
        )
    return image, mask


def _create_segmenter(engine: str, device: str):
    if engine == "sam2":
        return Sam2RegionSegmenter(Sam2SegmentationConfig(device=device))
    if engine == "sam1":
        return SamForegroundSegmenter(SamSegmentationConfig(device=device))
    if engine == "anime":
        if device != "cuda":
            raise ValueError("Anime segmentation requires the CUDA device.")
        return AnimeForegroundSegmenter(AnimeSegmentationConfig())
    raise ValueError(f"Unknown SAM engine: {engine}")


def _segment(
    segmenter: object,
    image: np.ndarray,
    input_path: Path,
    prompt_mode: str,
    mask_candidate: int | None,
    box: tuple[int, int, int, int] | None,
    positive_points: tuple[tuple[int, int], ...],
    negative_points: tuple[tuple[int, int], ...],
    threshold: float,
) -> np.ndarray:
    if isinstance(segmenter, AnimeForegroundSegmenter):
        if prompt_mode != "auto":
            raise ValueError("Anime segmentation supports automatic foreground prompting only.")
        return segmenter.segment_image(image)
    if prompt_mode == "auto":
        if isinstance(segmenter, (SamForegroundSegmenter, Sam2RegionSegmenter)):
            return segmenter.segment(
                input_path,
                threshold=threshold,
                mask_index=mask_candidate,
            )
        raise TypeError(f"Unsupported segmenter: {type(segmenter).__name__}")
    points = [*positive_points, *negative_points]
    labels = [1] * len(positive_points) + [0] * len(negative_points)
    if isinstance(segmenter, (SamForegroundSegmenter, Sam2RegionSegmenter)):
        if prompt_mode == "box":
            assert box is not None
            return segmenter.segment_image_box(image, box, mask_index=mask_candidate)
        return segmenter.segment_image_prompt(
            image,
            points=points,
            labels=labels,
            box=box if prompt_mode == "box+points" else None,
            mask_index=mask_candidate,
        )
    raise TypeError(f"Unsupported segmenter: {type(segmenter).__name__}")


def _validate_prompt(
    prompt_mode: str,
    box: tuple[int, int, int, int] | None,
    positive_points: tuple[tuple[int, int], ...],
) -> None:
    if prompt_mode == "box" and box is None:
        raise ValueError("Box prompt mode requires a box.")
    if prompt_mode == "points" and not positive_points:
        raise ValueError("Point prompt mode requires at least one positive point.")
    if prompt_mode == "box+points" and (box is None or not positive_points):
        raise ValueError("Box + point prompt mode requires a box and a positive point.")


def _parse_points(value: str) -> tuple[tuple[int, int], ...]:
    points = []
    for item in value.split(";"):
        try:
            coordinates = tuple(int(part.strip()) for part in item.split(","))
        except ValueError as error:
            raise argparse.ArgumentTypeError("Points must be x,y;x,y;...") from error
        if len(coordinates) != 2:
            raise argparse.ArgumentTypeError("Points must be x,y;x,y;...")
        points.append(coordinates)
    if not points:
        raise argparse.ArgumentTypeError("At least one point is required.")
    return tuple(points)


def _clean_mask(
    mask: np.ndarray,
    *,
    grow: int,
    feather: int,
    fill_holes: bool,
    largest_component: bool,
    invert: bool,
) -> np.ndarray:
    from scipy import ndimage

    values = np.asarray(mask, dtype=np.float32)
    binary = values > 0.5
    if grow > 0:
        binary = ndimage.binary_dilation(binary, iterations=grow)
    elif grow < 0:
        binary = ndimage.binary_erosion(binary, iterations=-grow)
    if fill_holes:
        binary = ndimage.binary_fill_holes(binary)
    if largest_component:
        labels, count = ndimage.label(binary)
        if count:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            binary = labels == sizes.argmax()
    values = binary.astype(np.float32)
    if feather:
        values = ndimage.gaussian_filter(values, sigma=max(0.5, feather / 2.0))
        values = np.clip(values, 0.0, 1.0)
    if invert:
        values = 1.0 - values
    return values


def _write_outputs(
    image: Image.Image,
    mask: np.ndarray,
    *,
    input_path: Path,
    output_dir: Path,
    engine: str,
    output_mode: str,
) -> dict[str, str]:
    mask_image = Image.fromarray(_mask_image(mask), mode="L")
    stem = f"{input_path.stem}-{engine}"
    outputs: dict[str, str] = {}
    if output_mode in {"all", "mask"}:
        outputs["mask"] = _save(mask_image, output_dir / f"{stem}-mask.png")
    if output_mode in {"all", "cutout"}:
        cutout = image.convert("RGBA")
        cutout.putalpha(mask_image)
        outputs["cutout"] = _save(cutout, output_dir / f"{stem}-cutout.png")
    if output_mode in {"all", "preview"}:
        preview = _preview(image, mask_image)
        outputs["preview"] = _save(preview, output_dir / f"{stem}-preview.png")
    return outputs


def _mask_image(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Segmentation mask must be two-dimensional, got {values.shape}.")
    return np.rint(np.clip(values, 0.0, 1.0) * 255).astype(np.uint8)


def _preview(image: Image.Image, mask: Image.Image) -> Image.Image:
    highlight = Image.new("RGB", image.size, (255, 48, 128))
    tinted = Image.composite(highlight, image, mask)
    return Image.blend(image, tinted, 0.45)


def _save(image: Image.Image, path: Path) -> str:
    image.save(path)
    return path.as_posix()


def _parse_box(value: str) -> tuple[int, int, int, int]:
    try:
        coordinates = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Box must be x1,y1,x2,y2.") from error
    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError("Box must be x1,y1,x2,y2.")
    left, top, right, bottom = coordinates
    if right <= left or bottom <= top:
        raise argparse.ArgumentTypeError("Box must have positive width and height.")
    return coordinates
