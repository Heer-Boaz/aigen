from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.generation.animegen_i2v import (
    ANIMEGEN_DEFAULT_FPS,
    ANIMEGEN_DEFAULT_FRAMES,
    ANIMEGEN_DEFAULT_PRECISION,
    ANIMEGEN_PRECISIONS,
)
from aigen.progress import StatusReporter


def add_animegen_command(subparsers: Any) -> None:
    command = subparsers.add_parser(
        "animegen-i2v",
        help="Generate anime I2V with the official AnimeGen-I2V Lightning recipe",
    )
    command.add_argument("--image", type=Path, required=True, help="Start frame")
    command.add_argument("--last-image", type=Path, help="Optional end frame")
    command.add_argument("--prompt", required=True, help="Motion instruction")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--frames", type=int, default=ANIMEGEN_DEFAULT_FRAMES)
    command.add_argument("--fps", type=int, default=ANIMEGEN_DEFAULT_FPS)
    command.add_argument(
        "--precision",
        choices=ANIMEGEN_PRECISIONS,
        default=ANIMEGEN_DEFAULT_PRECISION,
        help="FP8 layer storage with BF16 compute, or full BF16 storage and compute",
    )
    command.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Seed; repeat to reuse the loaded pipeline for a seed sweep (default: 42)",
    )


def run_animegen_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    from aigen.generation.animegen_i2v import (
        AnimeGenI2VError,
        generate_animegen_i2v,
        generate_animegen_i2v_seed_sweep,
    )

    seeds = tuple(args.seed or (42,))
    try:
        if len(seeds) == 1:
            result = generate_animegen_i2v(
                prompt=args.prompt,
                image=args.image,
                last_image=args.last_image,
                output=args.output,
                frames=args.frames,
                fps=args.fps,
                precision=args.precision,
                seed=seeds[0],
                progress=progress,
            )
            payload = result.to_json()
        else:
            results = generate_animegen_i2v_seed_sweep(
                prompt=args.prompt,
                image=args.image,
                last_image=args.last_image,
                output=args.output,
                frames=args.frames,
                fps=args.fps,
                precision=args.precision,
                seeds=seeds,
                progress=progress,
            )
            payload = {
                "status": "completed",
                "kind": "animegen-i2v-lightning-seed-sweep",
                "seeds": list(seeds),
                "results": [result.to_json() for result in results],
            }
    except AnimeGenI2VError as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1
    dump_json(stdout, payload, pretty=True)
    return 0
