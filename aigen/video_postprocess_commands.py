from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.generation.video_postprocess import (
    VideoPostprocessError,
    create_video_contact_sheet,
    extract_video_frames,
)
from aigen.progress import StatusReporter


def add_video_postprocess_commands(subparsers: Any) -> None:
    command = subparsers.add_parser(
        "video-postprocess",
        help="Post-process generated videos",
    )
    operations = command.add_subparsers(
        dest="video_postprocess_operation",
        required=True,
    )
    extract = operations.add_parser(
        "extract-frames",
        help="Extract every decoded video frame as a lossless PNG",
    )
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    contact_sheet = operations.add_parser(
        "contact-sheet",
        help="Render every decoded video frame into one contact sheet",
    )
    contact_sheet.add_argument("--input", type=Path, required=True)
    contact_sheet.add_argument("--output", type=Path, required=True)


def run_video_postprocess_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    try:
        if args.video_postprocess_operation == "extract-frames":
            payload = extract_video_frames(
                args.input,
                args.output_dir,
                progress=progress,
            ).to_json()
        elif args.video_postprocess_operation == "contact-sheet":
            progress.phase("create video contact sheet")
            output = create_video_contact_sheet(args.input, args.output)
            payload = {
                "status": "completed",
                "kind": "video-contact-sheet",
                "video": args.input.expanduser().resolve().as_posix(),
                "output": output.as_posix(),
            }
        else:
            raise RuntimeError("unsupported video post-processing operation")
    except VideoPostprocessError as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1
    dump_json(stdout, payload, pretty=True)
    return 0
