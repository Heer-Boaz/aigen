from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.generation.ltx23_keyframes import LTX23_DEFAULT_FPS
from aigen.progress import StatusReporter


def add_ltx23_command(subparsers: Any) -> None:
    command = subparsers.add_parser(
        "ltx23-keyframes",
        help="Generate LTX-2.3 video from start, end, and positioned keyframes",
    )
    command.add_argument("--prompt", required=True, help="Motion and camera instruction")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--resolution", required=True, help="Output size as WIDTHxHEIGHT")
    command.add_argument("--frames", type=int, default=49)
    command.add_argument(
        "--fps",
        type=int,
        default=LTX23_DEFAULT_FPS,
        help="Frames per second used for generation and output timing",
    )
    command.add_argument("--steps", type=int, default=8)
    command.add_argument(
        "--solver",
        choices=("distilled_8_steps", "euler", "res2s"),
        default="distilled_8_steps",
    )
    command.add_argument("--seed", type=int, default=42)
    command.add_argument(
        "--keyframe",
        nargs=2,
        action="append",
        required=True,
        metavar=("IMAGE", "FRAME"),
        help="Image path and zero-based output frame index; repeat for every keyframe",
    )


def run_ltx23_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    from aigen.generation.ltx23_keyframes import (
        Ltx23Keyframe,
        Ltx23KeyframesError,
        generate_ltx23_keyframes,
    )

    try:
        keyframes = tuple(
            Ltx23Keyframe(image=Path(image), frame=int(frame))
            for image, frame in args.keyframe
        )
        result = generate_ltx23_keyframes(
            prompt=args.prompt,
            keyframes=keyframes,
            output=args.output,
            resolution=args.resolution,
            frames=args.frames,
            fps=args.fps,
            steps=args.steps,
            solver=args.solver,
            seed=args.seed,
            progress=progress,
        )
    except (Ltx23KeyframesError, ValueError) as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1
    dump_json(stdout, result.to_json(), pretty=True)
    return 0
