from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from aigen.animegen_commands import add_animegen_command, run_animegen_command
from aigen.brief_commands import add_brief_commands, run_brief_command
from aigen.character_commands import add_character_commands, run_character_command
from aigen.flux2_commands import add_flux2_klein_command, run_flux2_klein_command
from aigen.hunyuanvideo15_commands import (
    add_hunyuanvideo15_command,
    run_hunyuanvideo15_command,
)
from aigen.image_caption_commands import add_image_caption_command, run_image_caption_command
from aigen.image_edit_commands import add_image_edit_command, run_image_edit_command
from aigen.keyframe_commands import add_keyframe_commands, run_keyframe_command
from aigen.lora_commands import add_lora_commands, run_lora_command
from aigen.ltx23_commands import add_ltx23_command, run_ltx23_command
from aigen.model_commands import add_model_commands, run_model_command
from aigen.pixel_art_commands import add_pixel_art_command, run_pixel_art_command
from aigen.pixel_art_fixer_commands import (
    add_pixel_art_fixer_command,
    run_pixel_art_fixer_command,
)
from aigen.sam_commands import add_sam_command, run_sam_command
from aigen.progress import StatusReporter, open_cli_progress
from aigen.video_postprocess_commands import (
    add_video_postprocess_commands,
    run_video_postprocess_command,
)
from aigen.wu_pixelization_commands import (
    add_wu_pixelization_command,
    run_wu_pixelization_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aigen",
        description="AI character generation pipeline tooling",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_animegen_command(subparsers)
    add_brief_commands(subparsers)
    add_character_commands(subparsers)
    add_flux2_klein_command(subparsers)
    add_hunyuanvideo15_command(subparsers)
    add_image_caption_command(subparsers)
    add_image_edit_command(subparsers)
    add_keyframe_commands(subparsers)
    add_lora_commands(subparsers)
    add_ltx23_command(subparsers)
    add_model_commands(subparsers)
    add_pixel_art_command(subparsers)
    add_pixel_art_fixer_command(subparsers)
    add_sam_command(subparsers)
    add_video_postprocess_commands(subparsers)
    add_wu_pixelization_command(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run_command(args)


def _run_command(args: argparse.Namespace) -> int:
    progress = open_cli_progress()
    with progress:
        exit_code = _run_command_with_progress(args, progress)
        progress.finish("completed" if exit_code == 0 else "failed")
        return exit_code


def _run_command_with_progress(args: argparse.Namespace, progress: StatusReporter) -> int:
    if args.command == "animegen-i2v":
        return run_animegen_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "briefs":
        return run_brief_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "characters":
        return run_character_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "flux2-klein":
        return run_flux2_klein_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "hunyuanvideo15-i2v":
        return run_hunyuanvideo15_command(
            args,
            sys.stdout,
            sys.stderr,
            progress=progress,
        )
    if args.command == "image-caption":
        return run_image_caption_command(args, sys.stdout, progress=progress)
    if args.command == "image-edit":
        return run_image_edit_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "keyframes":
        return run_keyframe_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "lora":
        return run_lora_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "ltx23-keyframes":
        return run_ltx23_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "models":
        return run_model_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "pixel-art":
        return run_pixel_art_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "pixel-art-fixer":
        return run_pixel_art_fixer_command(
            args,
            sys.stdout,
            sys.stderr,
            progress=progress,
        )
    if args.command == "pixel-art-wu":
        return run_wu_pixelization_command(
            args,
            sys.stdout,
            sys.stderr,
            progress=progress,
        )
    if args.command == "sam-segment":
        return run_sam_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "video-postprocess":
        return run_video_postprocess_command(
            args,
            sys.stdout,
            sys.stderr,
            progress=progress,
        )
    raise RuntimeError("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
