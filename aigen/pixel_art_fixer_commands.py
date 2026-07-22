from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.progress import StatusReporter


def add_pixel_art_fixer_command(subparsers: Any) -> None:
    command = subparsers.add_parser(
        "pixel-art-fixer",
        help="Detect and reconstruct a native pixel-art grid",
    )
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--mode", choices=("full", "fast"), default="full")
    command.add_argument("--low-memory", action="store_true")
    command.add_argument("--force-step", type=float)


def run_pixel_art_fixer_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    from aigen.generation.pixel_art_fixer import PixelArtFixerError, fix_pixel_art

    try:
        result = fix_pixel_art(
            args.input,
            args.output,
            mode=args.mode,
            low_memory=args.low_memory,
            force_step=args.force_step,
            progress=progress,
        )
    except PixelArtFixerError as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1
    dump_json(stdout, result.to_json(), pretty=True)
    return 0
