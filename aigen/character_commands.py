from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.character_view_models import (
    CharacterViewError,
    character_view_bank_schema,
    character_view_job_schema,
    load_character_view_job,
)
from aigen.character_views import (
    accept_character_view,
    plan_character_view_job,
    run_character_view_job,
    validate_character_view_job,
)
from aigen.command_io import command_error_payload, dump_json
from aigen.manifest_io import ManifestIOError
from aigen.runtime_profiles import PROJECT_ROOT, keyframe_profile_for_name


def add_character_commands(subparsers: Any) -> None:
    characters = subparsers.add_parser("characters", help="Character view bank tools")
    character_subparsers = characters.add_subparsers(dest="characters_command", required=True)

    view_schema = character_subparsers.add_parser("view-schema", help="Write the character-view job schema")
    view_schema.add_argument("--compact", action="store_true", help="Write compact JSON")

    bank_schema = character_subparsers.add_parser("view-bank-schema", help="Write the character view-bank schema")
    bank_schema.add_argument("--compact", action="store_true", help="Write compact JSON")

    view_validate = character_subparsers.add_parser("view-validate", help="Validate a character-view job")
    view_validate.add_argument("job", type=Path, help="Character-view job JSON")
    view_validate.add_argument("--compact", action="store_true", help="Write compact JSON")

    view_plan = character_subparsers.add_parser("view-plan", help="Resolve a character-view job")
    view_plan.add_argument("job", type=Path, help="Character-view job JSON")
    view_plan.add_argument("--compact", action="store_true", help="Write compact JSON")

    view_run = character_subparsers.add_parser("view-run", help="Run a character-view job")
    view_run.add_argument("job", type=Path, help="Character-view job JSON")
    view_run.add_argument("--compact", action="store_true", help="Write compact JSON")

    view_accept = character_subparsers.add_parser(
        "view-accept",
        help="Accept one generated candidate as a canonical character view",
    )
    view_accept.add_argument("job", type=Path, help="Character-view job JSON")
    view_accept.add_argument("--run-dir", type=Path, required=True, help="Completed character-view run directory")
    view_accept.add_argument("--candidate", required=True, help="Candidate name to accept")
    view_accept.add_argument("--compact", action="store_true", help="Write compact JSON")


def run_character_command(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    try:
        if args.characters_command == "view-schema":
            dump_json(stdout, character_view_job_schema(), pretty=not args.compact)
            return 0
        if args.characters_command == "view-bank-schema":
            dump_json(stdout, character_view_bank_schema(), pretty=not args.compact)
            return 0
        if args.characters_command == "view-validate":
            dump_json(
                stdout,
                validate_character_view_job(args.job, project_root=PROJECT_ROOT),
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "view-plan":
            dump_json(
                stdout,
                plan_character_view_job(args.job, project_root=PROJECT_ROOT),
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "view-run":
            dump_json(
                stdout,
                run_character_view_job(
                    args.job,
                    keyframe_profile_for_name(load_character_view_job(args.job).pipeline.profile),
                    project_root=PROJECT_ROOT,
                ),
                pretty=not args.compact,
            )
            return 0
        dump_json(
            stdout,
            accept_character_view(
                args.job,
                run_dir=args.run_dir,
                candidate=args.candidate,
                project_root=PROJECT_ROOT,
            ),
            pretty=not args.compact,
        )
        return 0
    except (CharacterViewError, ManifestIOError) as error:
        dump_json(stderr, command_error_payload(error), pretty=not args.compact)
        return 1
