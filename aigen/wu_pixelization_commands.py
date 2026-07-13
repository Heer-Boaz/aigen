from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.progress import StatusReporter


def add_wu_pixelization_command(subparsers: Any) -> None:
    command = subparsers.add_parser("pixel-art-wu")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--cell-size", type=int, required=True)


def run_wu_pixelization_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    from aigen.generation.wu_pixelization import (
        WuPixelizationError,
        pixelize_with_wu,
    )

    progress.phase("pixelize image")
    try:
        result = pixelize_with_wu(
            args.input,
            args.output,
            cell_size=args.cell_size,
        )
    except WuPixelizationError as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1
    dump_json(stdout, result.to_json(), pretty=True)
    return 0
