from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.progress import StatusReporter


IMAGE_EDIT_BACKENDS = ("hidream-o1-full-fp8", "boogu-image-edit-turbo-fp8")


def add_image_edit_command(subparsers: Any) -> None:
    command = subparsers.add_parser(
        "image-edit",
        help="Run a native reference-image editing backend",
    )
    command.add_argument("--backend", choices=IMAGE_EDIT_BACKENDS, required=True)
    command.add_argument("--prompt", required=True)
    command.add_argument("--reference", type=Path, action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--resolution", help="Output size as WIDTHxHEIGHT; default is backend-native")
    command.add_argument("--steps", type=int, help="Inference steps; default is backend-native")
    command.add_argument("--guidance", type=float, help="Guidance; default is backend-native")
    command.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Seed; repeat for an isolated seed sweep (default: 1, 2, 3)",
    )


def run_image_edit_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    from aigen.generation.boogu_image_edit import (
        BOOGU_DEFAULT_GUIDANCE,
        BOOGU_DEFAULT_RESOLUTION,
        BOOGU_DEFAULT_STEPS,
        BooguImageEditError,
        generate_boogu_image_edit_seed_sweep,
    )
    from aigen.generation.hidream_o1_comfy import (
        HIDREAM_DEFAULT_GUIDANCE,
        HIDREAM_DEFAULT_RESOLUTION,
        HIDREAM_DEFAULT_STEPS,
        HiDreamO1Error,
        generate_hidream_o1_seed_sweep,
    )

    try:
        seeds = tuple(args.seed or (1, 2, 3))
        if args.backend == "hidream-o1-full-fp8":
            resolution = args.resolution or HIDREAM_DEFAULT_RESOLUTION
            steps = args.steps if args.steps is not None else HIDREAM_DEFAULT_STEPS
            guidance = args.guidance if args.guidance is not None else HIDREAM_DEFAULT_GUIDANCE
            generate = generate_hidream_o1_seed_sweep
        else:
            resolution = args.resolution or BOOGU_DEFAULT_RESOLUTION
            steps = args.steps if args.steps is not None else BOOGU_DEFAULT_STEPS
            guidance = args.guidance if args.guidance is not None else BOOGU_DEFAULT_GUIDANCE
            generate = generate_boogu_image_edit_seed_sweep
        width_text, height_text = resolution.lower().split("x", maxsplit=1)
        results = generate(
            prompt=args.prompt,
            references=args.reference,
            output=args.output,
            width=int(width_text),
            height=int(height_text),
            seeds=seeds,
            steps=steps,
            guidance=guidance,
            progress=progress,
        )
    except (BooguImageEditError, HiDreamO1Error, ValueError) as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1

    payload = (
        results[0].to_json()
        if len(results) == 1
        else {
            "status": "completed",
            "kind": "image-edit-seed-sweep",
            "backend": args.backend,
            "seeds": list(seeds),
            "results": [result.to_json() for result in results],
        }
    )
    dump_json(stdout, payload, pretty=True)
    return 0
