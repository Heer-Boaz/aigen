from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.progress import StatusReporter


def add_flux2_klein_command(subparsers: Any) -> None:
    command = subparsers.add_parser("flux2-klein")
    command.add_argument("--prompt", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--reference", type=Path, action="append", default=[])
    command.add_argument("--width", type=int)
    command.add_argument("--height", type=int)
    command.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Seed; repeat this option to generate a cached seed sweep (default: 42)",
    )
    command.add_argument("--lora", type=Path)
    command.add_argument("--lora-weight", type=float, default=1.0)
    command.add_argument(
        "--strength",
        type=float,
        help="img2img denoise strength in (0, 1]; when set, the first --reference is the init image and its structure/pose is preserved",
    )


def run_flux2_klein_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    from aigen.generation.flux2_klein import (
        Flux2KleinError,
        generate_flux2_klein,
        generate_flux2_klein_seed_sweep,
    )

    try:
        seeds = tuple(args.seed or (42,))
        if len(seeds) == 1:
            result = generate_flux2_klein(
                prompt=args.prompt,
                output=args.output,
                references=args.reference,
                width=args.width,
                height=args.height,
                seed=seeds[0],
                lora=args.lora,
                lora_weight=args.lora_weight,
                strength=args.strength,
                progress=progress,
            )
            payload = result.to_json()
        else:
            result = generate_flux2_klein_seed_sweep(
                prompt=args.prompt,
                output=args.output,
                references=args.reference,
                width=args.width,
                height=args.height,
                seeds=seeds,
                lora=args.lora,
                lora_weight=args.lora_weight,
                strength=args.strength,
                progress=progress,
            )
            payload = {
                "status": "completed",
                "kind": "flux2-klein-seed-sweep",
                "seeds": list(seeds),
                **result.to_json(),
            }
    except Flux2KleinError as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1
    dump_json(stdout, payload, pretty=True)
    return 0
