from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.generation.image_batch_postprocess import (
    IMAGE_BATCH_DEFAULT_CELL_SIZE,
    IMAGE_BATCH_DEFAULT_FIXER_MODE,
    ImageBatchPostprocessError,
    image_batch_postprocess_model_names,
    postprocess_image_batch,
)
from aigen.generation.vosr_backend import (
    VOSR_DEFAULT_ALIGN_METHOD,
    VOSR_DEFAULT_CFG_SCALE,
    VOSR_DEFAULT_INFER_STEPS,
    VOSR_DEFAULT_SCALE,
    VOSR_DEFAULT_SEED,
    VOSR_DEFAULT_TILE_SIZE,
    VOSR_DEFAULT_WEAK_COND_STRENGTH_AELQ,
)
from aigen.progress import StatusReporter


def add_image_batch_postprocess_command(subparsers: Any) -> None:
    command = subparsers.add_parser(
        "image-postprocess-batch",
        help="Post-process an ordered image sequence with one retained model session",
    )
    command.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Input image; repeat in frame order",
    )
    command.add_argument(
        "--output-name",
        action="append",
        help="Output basename; repeat once per input when names must differ",
    )
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--model",
        choices=image_batch_postprocess_model_names(),
        required=True,
    )
    output_size = command.add_mutually_exclusive_group()
    output_size.add_argument("--scale", type=int, default=VOSR_DEFAULT_SCALE)
    output_size.add_argument("--long-side", type=int)
    command.add_argument("--infer-steps", type=int, default=VOSR_DEFAULT_INFER_STEPS)
    command.add_argument("--cfg-scale", type=float, default=VOSR_DEFAULT_CFG_SCALE)
    command.add_argument(
        "--weak-cond-strength-aelq",
        type=float,
        default=VOSR_DEFAULT_WEAK_COND_STRENGTH_AELQ,
    )
    command.add_argument(
        "--align-method",
        choices=("wavelet", "adain", "nofix"),
        default=VOSR_DEFAULT_ALIGN_METHOD,
    )
    command.add_argument("--tile-size", type=int, default=VOSR_DEFAULT_TILE_SIZE)
    command.add_argument("--seed", type=int, default=VOSR_DEFAULT_SEED)
    command.add_argument(
        "--cell-size",
        type=int,
        default=IMAGE_BATCH_DEFAULT_CELL_SIZE,
    )
    command.add_argument(
        "--mode",
        choices=("full", "fast"),
        default=IMAGE_BATCH_DEFAULT_FIXER_MODE,
    )
    command.add_argument("--low-memory", action="store_true")
    command.add_argument("--force-step", type=float)


def run_image_batch_postprocess_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    try:
        result = postprocess_image_batch(
            args.input,
            args.output_dir,
            model=args.model,
            progress=progress,
            output_names=args.output_name,
            long_side=args.long_side,
            scale=args.scale,
            infer_steps=args.infer_steps,
            cfg_scale=args.cfg_scale,
            weak_cond_strength_aelq=args.weak_cond_strength_aelq,
            align_method=args.align_method,
            tile_size=args.tile_size,
            seed=args.seed,
            cell_size=args.cell_size,
            mode=args.mode,
            low_memory=args.low_memory,
            force_step=args.force_step,
        )
    except (ImageBatchPostprocessError, OSError) as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1
    dump_json(stdout, result.to_json(), pretty=True)
    return 0
