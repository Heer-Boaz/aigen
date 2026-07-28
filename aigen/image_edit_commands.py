from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.character_reference_models import CharacterReferenceError
from aigen.command_io import command_error_payload, dump_json
from aigen.generation.image_edit import (
    IMAGE_EDIT_BACKENDS,
    ImageEditError,
    ImageEditRequest,
    run_image_edit,
)
from aigen.image_dimensions import parse_aspect_ratio
from aigen.progress import StatusReporter


class ImageEditCommandError(RuntimeError):
    pass


def add_image_edit_command(subparsers: Any) -> None:
    command = subparsers.add_parser(
        "image-edit",
        help="Edit images with a selected image-generation backend",
    )
    command.add_argument("--backend", choices=IMAGE_EDIT_BACKENDS, required=True)
    command.add_argument("--image", type=Path, action="append")
    command.add_argument(
        "--reference-pack",
        type=Path,
        action="append",
        help="Named visual reference pack; repeat to combine packs",
    )
    command.add_argument("--prompt", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Seed; repeat for a seed sweep (default: 0)",
    )
    command.add_argument("--width", type=int, help="Output width; requires --height")
    command.add_argument("--height", type=int, help="Output height; requires --width")
    command.add_argument(
        "--aspect-ratio",
        type=parse_aspect_ratio,
        help="Recommended backend canvas for W:H; cannot be combined with --width/--height",
    )
    command.add_argument(
        "--steps",
        type=int,
        help="Inference steps; default is backend-native",
    )
    command.add_argument(
        "--guidance",
        type=float,
        help="CFG scale; default is backend-native",
    )
    command.add_argument(
        "--strength",
        type=float,
        help="Image-to-image denoise strength; default is backend-native",
    )
    command.add_argument(
        "--sampler",
        help="Backend sampler; default is backend-native",
    )
    command.add_argument(
        "--scheduler",
        help="Backend sigma scheduler; default is backend-native",
    )
    command.add_argument(
        "--lora",
        type=Path,
        action="append",
        help="Backend-compatible LoRA SafeTensors; repeat to combine LoRAs",
    )
    command.add_argument(
        "--lora-weight",
        type=float,
        action="append",
        help="LoRA strengths in --lora order (default: 1.0 for all)",
    )
    command.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory",
    )


def run_image_edit_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    try:
        request = ImageEditRequest(
            backend=args.backend,
            prompt=args.prompt,
            output_dir=args.output_dir,
            images=tuple(args.image or ()),
            reference_packs=tuple(args.reference_pack or ()),
            seeds=tuple(args.seed or (0,)),
            width=args.width,
            height=args.height,
            aspect_ratio=args.aspect_ratio,
            steps=args.steps,
            guidance=args.guidance,
            strength=args.strength,
            sampler=args.sampler,
            scheduler=args.scheduler,
            loras=tuple(args.lora or ()),
            lora_weights=tuple(args.lora_weight or ()),
            overwrite=args.overwrite,
        )
        result = run_image_edit(request, progress=progress)
    except ImageEditError as error:
        command_error = ImageEditCommandError(str(error))
        dump_json(stderr, command_error_payload(command_error), pretty=True)
        return 1
    except (CharacterReferenceError, OSError, ValueError) as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1

    dump_json(stdout, result.to_json(), pretty=True)
    return 0
