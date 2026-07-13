from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.character_reference_pack import load_character_reference_pack
from aigen.command_io import command_error_payload, dump_json
from aigen.progress import StatusReporter


def add_pixel_art_command(subparsers: Any) -> None:
    command = subparsers.add_parser("pixel-art")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--prompt")
    command.add_argument("--reference-pack", type=Path)
    command.add_argument("--width", type=int, required=True)
    command.add_argument("--height", type=int, required=True)
    command.add_argument("--colors", type=int, default=8)
    command.add_argument("--steps", type=int, default=10_001)
    command.add_argument("--save-every", type=int, default=0)
    command.add_argument("--seed", type=int, default=0)


def run_pixel_art_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    from aigen.generation.sd_pixl_sdxl import SdPixlError
    from aigen.pixel_art_sd_pixl import convert_to_pixel_art

    reference_images: tuple[Path, ...] = ()
    if args.reference_pack is not None:
        progress.phase("load reference pack")
        reference_images = tuple(
            load_character_reference_pack(args.reference_pack).references.values()
        )
    try:
        result = convert_to_pixel_art(
            args.input,
            args.output,
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            colors=args.colors,
            steps=args.steps,
            save_every=args.save_every,
            seed=args.seed,
            reference_images=reference_images,
            progress=progress,
        )
    except SdPixlError as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1
    dump_json(stdout, result.to_json(), pretty=True)
    return 0
