from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from aigen.brief_commands import add_brief_commands, run_brief_command
from aigen.character_commands import add_character_commands, run_character_command
from aigen.keyframe_commands import add_keyframe_commands, run_keyframe_command
from aigen.lora_commands import add_lora_commands, run_lora_command
from aigen.model_commands import add_model_commands, run_model_command


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

    if args.command == "briefs":
        return run_brief_command(args, sys.stdout, sys.stderr)
    if args.command == "characters":
        return run_character_command(args, sys.stdout, sys.stderr)
    if args.command == "keyframes":
        return run_keyframe_command(args, sys.stdout, sys.stderr)
    if args.command == "lora":
        return run_lora_command(args, sys.stdout, sys.stderr)
    if args.command == "models":
        return run_model_command(args, sys.stdout, sys.stderr)

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
