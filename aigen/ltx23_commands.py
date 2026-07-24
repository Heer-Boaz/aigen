from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.generation.ltx23_keyframes import (
    LTX23_DEFAULT_CONDITIONING_STRENGTH,
    LTX23_DEFAULT_FPS,
    LTX23_DEFAULT_MODEL,
    LTX23_DEFAULT_NEGATIVE_PROMPT,
    LTX23_DEFAULT_PHASES,
    LTX23_MODEL_TYPES,
)
from aigen.progress import StatusReporter


def add_ltx23_command(subparsers: Any) -> None:
    command = subparsers.add_parser(
        "ltx23-keyframes",
        help="Generate LTX-2.3 video from one or more positioned keyframes",
    )
    command.add_argument("--prompt", required=True, help="Motion and camera instruction")
    command.add_argument(
        "--negative-prompt",
        default=LTX23_DEFAULT_NEGATIVE_PROMPT,
        help="Artifacts to suppress during guided generation",
    )
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--resolution", required=True, help="Output size as WIDTHxHEIGHT")
    command.add_argument("--frames", type=int, default=121)
    command.add_argument(
        "--fps",
        type=int,
        default=LTX23_DEFAULT_FPS,
        help="Frames per second used for generation and output timing",
    )
    command.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Stage-1 inference steps (default: 15 for HQ Res2S)",
    )
    command.add_argument(
        "--phases",
        type=int,
        choices=(1, 2),
        default=LTX23_DEFAULT_PHASES,
        help="Generation phases; one phase generates directly at the requested resolution",
    )
    command.add_argument(
        "--solver",
        choices=("distilled_8_steps", "euler", "res2s"),
        default="res2s",
    )
    command.add_argument(
        "--conditioning-strength",
        type=float,
        default=LTX23_DEFAULT_CONDITIONING_STRENGTH,
        help="Keyframe conditioning strength from 0 to 1 (default: 1)",
    )
    command.add_argument(
        "--model",
        choices=tuple(LTX23_MODEL_TYPES),
        default=LTX23_DEFAULT_MODEL,
        help="LTX-2.3 Dev transformer precision; defaults to nvfp4",
    )
    command.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Seed; repeat this option to generate a cached seed sweep (default: 42)",
    )
    command.add_argument(
        "--keyframe",
        nargs=2,
        action="append",
        required=True,
        metavar=("IMAGE", "FRAME"),
        help="Image path and zero-based output frame index; repeat for every keyframe",
    )


def run_ltx23_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    from aigen.generation.ltx23_keyframes import (
        Ltx23Keyframe,
        Ltx23KeyframesError,
        generate_ltx23_keyframes,
        generate_ltx23_keyframes_seed_sweep,
    )

    try:
        keyframes = tuple(
            Ltx23Keyframe(image=Path(image), frame=int(frame))
            for image, frame in args.keyframe
        )
        seeds = tuple(args.seed or (42,))
        if len(seeds) == 1:
            result = generate_ltx23_keyframes(
                prompt=args.prompt,
                keyframes=keyframes,
                output=args.output,
                resolution=args.resolution,
                frames=args.frames,
                fps=args.fps,
                steps=args.steps,
                phases=args.phases,
                solver=args.solver,
                negative_prompt=args.negative_prompt,
                conditioning_strength=args.conditioning_strength,
                model=args.model,
                seed=seeds[0],
                progress=progress,
            )
            payload = result.to_json()
        else:
            results = generate_ltx23_keyframes_seed_sweep(
                prompt=args.prompt,
                keyframes=keyframes,
                output=args.output,
                resolution=args.resolution,
                frames=args.frames,
                fps=args.fps,
                steps=args.steps,
                phases=args.phases,
                solver=args.solver,
                negative_prompt=args.negative_prompt,
                conditioning_strength=args.conditioning_strength,
                model=args.model,
                seeds=seeds,
                progress=progress,
            )
            payload = {
                "status": "completed",
                "kind": "ltx-2.3-keyframe-conditioned-video-seed-sweep",
                "seeds": list(seeds),
                "results": [result.to_json() for result in results],
            }
    except (Ltx23KeyframesError, ValueError) as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1
    dump_json(stdout, payload, pretty=True)
    return 0
