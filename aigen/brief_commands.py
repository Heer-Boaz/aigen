from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.keyframe_brief_models import KeyframeBriefError, keyframe_brief_plan_schema, keyframe_brief_schema
from aigen.keyframe_brief_planner import plan_keyframe_brief
from aigen.judge_cli import add_judge_runtime_args, judge_config_from_args
from aigen.keyframe_briefs import (
    execute_keyframe_brief,
    materialize_keyframe_brief,
)
from aigen.keyframe_judge import KeyframeJudgeError
from aigen.keyframe_examples import KeyframeExampleError
from aigen.keyframe_job_models import KeyframeJobError
from aigen.keyframe_memory import KeyframeMemoryError
from aigen.manifest_io import ManifestIOError
from aigen.runtime_profiles import PROJECT_ROOT


def add_brief_commands(subparsers: Any) -> None:
    briefs = subparsers.add_parser("briefs", help="Model-planned keyframe briefs")
    brief_subparsers = briefs.add_subparsers(dest="briefs_command", required=True)

    schema = brief_subparsers.add_parser("schema", help="Write the keyframe brief JSON schema")
    schema.add_argument("--compact", action="store_true", help="Write compact JSON")

    plan_schema = brief_subparsers.add_parser("plan-schema", help="Write the generated brief-plan JSON schema")
    plan_schema.add_argument("--compact", action="store_true", help="Write compact JSON")

    plan = brief_subparsers.add_parser("plan", help="Use the local VLM to plan a keyframe job from a brief")
    plan.add_argument("brief", type=Path, help="Keyframe brief JSON")
    add_judge_runtime_args(plan, role="planner", max_new_tokens=1400)
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
    add_judge_runtime_args(run, role="planner", max_new_tokens=1400)
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
                plan_keyframe_brief(args.brief, judge_config_from_args(args), project_root=PROJECT_ROOT),
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
                judge_config_from_args(args),
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
        ManifestIOError,
    ) as error:
        dump_json(stderr, command_error_payload(error), pretty=not args.compact)
        return 1
