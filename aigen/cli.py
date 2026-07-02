from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from aigen.brief_commands import add_brief_commands, run_brief_command
from aigen.character_commands import add_character_commands, run_character_command
from aigen.keyframe_commands import add_keyframe_commands, run_keyframe_command
from aigen.lora_commands import add_lora_commands, run_lora_command
from aigen.model_commands import add_model_commands, run_model_command
from aigen.progress import StatusReporter, open_cli_progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aigen",
        description="AI character generation pipeline tooling",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_brief_commands(subparsers)
    add_character_commands(subparsers)
    add_keyframe_commands(subparsers)
    add_lora_commands(subparsers)
    add_model_commands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run_command(args)


def _run_command(args: argparse.Namespace) -> int:
    progress = open_cli_progress(args)
    with progress:
        exit_code = _run_command_with_progress(args, progress)
        progress.finish("completed" if exit_code == 0 else "failed")
        return exit_code


def _run_command_with_progress(args: argparse.Namespace, progress: StatusReporter) -> int:
    if args.command == "briefs":
        return run_brief_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "characters":
        return run_character_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "keyframes":
        return run_keyframe_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "lora":
        return run_lora_command(args, sys.stdout, sys.stderr, progress=progress)
    if args.command == "models":
        return run_model_command(args, sys.stdout, sys.stderr, progress=progress)

    raise RuntimeError("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
