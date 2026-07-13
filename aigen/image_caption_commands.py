from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.character_reference_pack import load_character_reference_pack
from aigen.progress import StatusReporter


def add_image_caption_command(subparsers: Any) -> None:
    command = subparsers.add_parser("image-caption")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--reference-pack", type=Path)


def run_image_caption_command(
    args: argparse.Namespace,
    stdout: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    from aigen.generation.image_caption_qwen25 import caption_image

    reference_images: tuple[Path, ...] = ()
    if args.reference_pack is not None:
        progress.phase("load reference pack")
        reference_images = tuple(
            load_character_reference_pack(args.reference_pack).references.values()
        )
    progress.phase("caption image")
    stdout.write(f"{caption_image(args.input, reference_images=reference_images)}\n")
    return 0
