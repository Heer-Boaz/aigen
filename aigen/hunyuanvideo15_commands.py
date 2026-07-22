from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.progress import StatusReporter


def add_hunyuanvideo15_command(subparsers: Any) -> None:
    command = subparsers.add_parser(
        "hunyuanvideo15-i2v",
        help="Generate 480p I2V with Tencent HunyuanVideo-1.5 step-distilled",
    )
    command.add_argument("--image", type=Path, required=True)
    command.add_argument("--prompt", required=True, help="Motion and camera instruction")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--steps", type=int, choices=(8, 12), default=8)
    command.add_argument("--frames", type=int, default=49)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument(
        "--overlap-group-offloading",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use extra host RAM to overlap transformer transfers (default: enabled)",
    )


def run_hunyuanvideo15_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    from aigen.generation.hunyuanvideo15 import (
        HunyuanVideo15Error,
        generate_hunyuanvideo15_i2v,
    )

    try:
        result = generate_hunyuanvideo15_i2v(
            prompt=args.prompt,
            image=args.image,
            output=args.output,
            steps=args.steps,
            frames=args.frames,
            seed=args.seed,
            overlap_group_offloading=args.overlap_group_offloading,
            progress=progress,
        )
    except HunyuanVideo15Error as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1
    dump_json(stdout, result.to_json(), pretty=True)
    return 0
