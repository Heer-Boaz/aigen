from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.generation.video_postprocess import (
    VideoPostprocessError,
    assemble_video_frames,
    create_video_contact_sheet,
    extract_video_frames,
)
from aigen.progress import StatusReporter
from aigen.manifest_io import ManifestIOError, read_json
from aigen.media_timing import AudioTrack, FrameTimeline, load_audio_track


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
    assemble = operations.add_parser("assemble", help="Assemble frame files using their saved timeline")
    assemble.add_argument("--frames-manifest", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--audio-policy", choices=("preserve", "remove", "replace"), default="preserve")
    assemble.add_argument("--audio", type=Path)
    assemble.add_argument("--audio-stream", type=int)
    assemble.add_argument("--background", default="black", help="Background when flattening transparent frames")


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
        elif args.video_postprocess_operation == "assemble":
            manifest = read_json(args.frames_manifest, label="frame sequence")
            base_dir = args.frames_manifest.expanduser().resolve().parent
            if args.audio_policy == "replace":
                if args.audio is None:
                    raise VideoPostprocessError("replace audio requires --audio")
                audio = load_audio_track(args.audio, stream_index=args.audio_stream)
            else:
                if args.audio is not None or args.audio_stream is not None:
                    raise VideoPostprocessError("--audio and --audio-stream require --audio-policy replace")
                audio = AudioTrack.model_validate(manifest["audio"]) if args.audio_policy == "preserve" and manifest.get("audio") else None
                if audio is not None:
                    audio = audio.model_copy(update={"path": (base_dir / audio.path).as_posix()})
            info = assemble_video_frames(
                tuple(base_dir / path for path in manifest["paths"]), args.output,
                timeline=FrameTimeline.model_validate(manifest["timeline"]), audio=audio,
                progress=progress, background=args.background,
            )
            payload = {"status": "completed", "kind": "video-assembly", "output": args.output.resolve().as_posix(),
                       "measured": info.model_dump(mode="json"), "audio_policy": args.audio_policy}
        else:
            raise RuntimeError("unsupported video post-processing operation")
    except (VideoPostprocessError, ManifestIOError, ValueError, KeyError) as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1
    dump_json(stdout, payload, pretty=True)
    return 0
