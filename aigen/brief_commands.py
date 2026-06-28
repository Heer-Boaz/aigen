from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.keyframe_briefs import (
    KeyframeBriefError,
    execute_keyframe_brief,
    keyframe_brief_plan_schema,
    keyframe_brief_schema,
    materialize_keyframe_brief,
    plan_keyframe_brief,
)
from aigen.keyframe_judge import (
    DEFAULT_JUDGE_ID,
    DEFAULT_JUDGE_QUANTIZATION,
    DEFAULT_JUDGE_REPO_ID,
    DEFAULT_JUDGE_REVISION,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    KeyframeJudgeConfig,
    KeyframeJudgeError,
)
from aigen.keyframe_examples import KeyframeExampleError
from aigen.keyframe_memory import KeyframeMemoryError
from aigen.keyframes import KeyframeJobError
from aigen.runtime_profiles import MODELS_ROOT, PROJECT_ROOT


def add_brief_commands(subparsers: Any) -> None:
    briefs = subparsers.add_parser("briefs", help="Model-planned keyframe briefs")
    brief_subparsers = briefs.add_subparsers(dest="briefs_command", required=True)

    schema = brief_subparsers.add_parser("schema", help="Write the keyframe brief JSON schema")
    schema.add_argument("--compact", action="store_true", help="Write compact JSON")

    plan_schema = brief_subparsers.add_parser("plan-schema", help="Write the generated brief-plan JSON schema")
    plan_schema.add_argument("--compact", action="store_true", help="Write compact JSON")

    plan = brief_subparsers.add_parser("plan", help="Use the local VLM to plan a keyframe job from a brief")
    plan.add_argument("brief", type=Path, help="Keyframe brief JSON")
    _add_judge_args(plan, max_new_tokens=1400)
    plan.add_argument("--compact", action="store_true", help="Write compact JSON")

    materialize = brief_subparsers.add_parser(
        "materialize",
        help="Extract example controls and write the planned keyframe job",
    )
    materialize.add_argument("brief", type=Path, help="Keyframe brief JSON")
    materialize.add_argument("--pose-device", default="cpu", help="DWPose device")
    materialize.add_argument("--compact", action="store_true", help="Write compact JSON")

    run = brief_subparsers.add_parser("run", help="Plan, materialize and run a keyframe brief")
    run.add_argument("brief", type=Path, help="Keyframe brief JSON")
    _add_judge_args(run, max_new_tokens=1400)
    run.add_argument("--pose-device", default="cpu", help="DWPose device")
    run.add_argument("--compact", action="store_true", help="Write compact JSON")


def run_brief_command(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    try:
        if args.briefs_command == "schema":
            dump_json(stdout, keyframe_brief_schema(), pretty=not args.compact)
            return 0
        if args.briefs_command == "plan-schema":
            dump_json(stdout, keyframe_brief_plan_schema(), pretty=not args.compact)
            return 0
        if args.briefs_command == "plan":
            dump_json(
                stdout,
                plan_keyframe_brief(args.brief, _judge_config(args), project_root=PROJECT_ROOT),
                pretty=not args.compact,
            )
            return 0
        if args.briefs_command == "materialize":
            dump_json(
                stdout,
                materialize_keyframe_brief(args.brief, project_root=PROJECT_ROOT, pose_device=args.pose_device),
                pretty=not args.compact,
            )
            return 0
        dump_json(
            stdout,
            execute_keyframe_brief(
                args.brief,
                _judge_config(args),
                project_root=PROJECT_ROOT,
                pose_device=args.pose_device,
            ),
            pretty=not args.compact,
        )
        return 0
    except (
        KeyframeBriefError,
        KeyframeExampleError,
        KeyframeJobError,
        KeyframeMemoryError,
        KeyframeJudgeError,
    ) as error:
        dump_json(stderr, command_error_payload(error), pretty=not args.compact)
        return 1


def _add_judge_args(parser: argparse.ArgumentParser, *, max_new_tokens: int) -> None:
    parser.add_argument("--judge", default=DEFAULT_JUDGE_ID, help="Planner id recorded in the brief plan")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_ROOT / "vlm/Qwen/Qwen2.5-VL-7B-Instruct",
        help="Local Qwen2.5-VL-7B-Instruct model directory",
    )
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype for planner model weights")
    parser.add_argument("--attention-impl", default="sdpa", help="Transformers attention implementation")
    parser.add_argument(
        "--quantization",
        choices=("bitsandbytes-8bit", "bitsandbytes-4bit", "none"),
        default=DEFAULT_JUDGE_QUANTIZATION,
        help="Local inference quantization for the 7B planner",
    )
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS, help="Minimum Qwen visual pixels")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Maximum Qwen visual pixels")
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens, help="Planner response token budget")
    parser.add_argument("--temperature", type=float, default=0.0, help="Planner sampling temperature")


def _judge_config(args: argparse.Namespace) -> KeyframeJudgeConfig:
    return KeyframeJudgeConfig(
        judge_id=args.judge,
        model=args.model,
        repo_id=DEFAULT_JUDGE_REPO_ID,
        revision=DEFAULT_JUDGE_REVISION,
        dtype=args.dtype,
        attention_impl=args.attention_impl,
        quantization=args.quantization,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
