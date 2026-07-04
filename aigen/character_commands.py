from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.character_reference_models import CharacterReferenceError
from aigen.character_reference_pack import (
    build_character_reference_pack,
    parse_character_reference_args,
    parse_character_reference_pack,
)
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
from aigen.generation.qwen_image_edit_identity import (
    DEFAULT_QWEN_IDENTITY_MAX_SEQUENCE_LENGTH,
    DEFAULT_QWEN_IDENTITY_MAX_SIDE,
    DEFAULT_QWEN_IDENTITY_PROFILE,
    DEFAULT_QWEN_IDENTITY_SEED,
    QwenImageEditIdentityError,
    parse_qwen_identity_reference_args,
    qwen_identity_cli_case_names,
    qwen_image_edit_identity_profile_for_name,
    qwen_image_edit_identity_profile_names,
    run_qwen_image_edit_identity,
)
from aigen.keyframe_memory import KeyframeMemoryError
from aigen.judge_cli import add_judge_runtime_args, judge_config_from_args
from aigen.manifest_io import ManifestIOError
from aigen.progress import StatusReporter
from aigen.runtime_profiles import PROJECT_ROOT, keyframe_profile_for_name
from aigen.vlm_qwen import QwenVlmError


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

    reference_pack = character_subparsers.add_parser(
        "reference-pack",
        help="Build and parse multi-reference character identity packs",
    )
    reference_pack_subparsers = reference_pack.add_subparsers(dest="reference_pack_command", required=True)

    reference_pack_build = reference_pack_subparsers.add_parser(
        "build",
        help="Build a named character reference pack",
    )
    reference_pack_build.add_argument("--character-id", required=True, help="Character id")
    reference_pack_build.add_argument(
        "--reference",
        action="append",
        required=True,
        help="Named reference image as name=path; supported names: front, portrait, side, back, body_shape",
    )
    reference_pack_build.add_argument("--output-dir", type=Path, required=True, help="Reference pack directory")
    reference_pack_build.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    reference_pack_build.add_argument("--compact", action="store_true", help="Write compact JSON")

    reference_pack_parse = reference_pack_subparsers.add_parser(
        "parse",
        help="Use the local Qwen VLM to write an identity profile from a reference pack",
    )
    reference_pack_parse.add_argument("pack", type=Path, help="reference_pack.json")
    reference_pack_parse.add_argument("--output", type=Path, help="Generated identity_profile.json")
    reference_pack_parse.add_argument("--overwrite", action="store_true", help="Replace an existing identity profile")
    add_judge_runtime_args(reference_pack_parse, role="reference parser", max_new_tokens=1600)
    reference_pack_parse.add_argument("--compact", action="store_true", help="Write compact JSON")

    qwen_identity = character_subparsers.add_parser(
        "qwen-identity-run",
        help="Run a fixed multi-reference Qwen Image Edit identity smoke",
    )
    qwen_identity.add_argument(
        "--reference",
        action="append",
        required=True,
        help="Named reference image as name=path; supported names: front, portrait, side, back, body_shape",
    )
    qwen_identity.add_argument(
        "--case",
        action="append",
        choices=qwen_identity_cli_case_names(),
        help="Identity case to generate; defaults to all fixed cases",
    )
    qwen_identity.add_argument("--output-dir", type=Path, required=True, help="Directory for generated images")
    qwen_identity.add_argument(
        "--profile",
        default=DEFAULT_QWEN_IDENTITY_PROFILE,
        choices=qwen_image_edit_identity_profile_names(),
        help="Qwen Image Edit model profile",
    )
    qwen_identity.add_argument(
        "--max-side",
        type=int,
        default=DEFAULT_QWEN_IDENTITY_MAX_SIDE,
        help="Longest generated/reference side for the smoke run",
    )
    qwen_identity.add_argument(
        "--steps",
        type=int,
        help="Qwen Image Edit denoising steps; defaults to the selected profile",
    )
    qwen_identity.add_argument(
        "--true-cfg-scale",
        type=float,
        help="Classifier-free guidance scale; defaults to the selected profile",
    )
    qwen_identity.add_argument(
        "--guidance-scale",
        type=float,
        help="Guidance-distilled model scale; defaults to the selected profile",
    )
    qwen_identity.add_argument("--seed", type=int, default=DEFAULT_QWEN_IDENTITY_SEED, help="Base seed")
    qwen_identity.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_QWEN_IDENTITY_MAX_SEQUENCE_LENGTH,
        help="Maximum prompt token sequence length",
    )
    qwen_identity.add_argument(
        "--nunchaku-blocks-on-gpu",
        type=int,
        help="Explicit slow-fit Nunchaku layer offload; leave unset for direct GPU execution",
    )
    qwen_identity.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    qwen_identity.add_argument("--compact", action="store_true", help="Write compact JSON")


def run_character_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
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
                    progress=progress,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "view-accept":
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
        if args.characters_command == "reference-pack":
            if args.reference_pack_command == "build":
                dump_json(
                    stdout,
                    build_character_reference_pack(
                        character_id=args.character_id,
                        references=parse_character_reference_args(args.reference, Path.cwd()),
                        output_dir=args.output_dir,
                        overwrite=args.overwrite,
                    ),
                    pretty=not args.compact,
                )
                return 0
            if args.reference_pack_command == "parse":
                dump_json(
                    stdout,
                    parse_character_reference_pack(
                        args.pack,
                        judge_config_from_args(args),
                        output_path=args.output,
                        overwrite=args.overwrite,
                        progress=progress,
                    ),
                    pretty=not args.compact,
                )
                return 0
        if args.characters_command == "qwen-identity-run":
            dump_json(
                stdout,
                run_qwen_image_edit_identity(
                    references=parse_qwen_identity_reference_args(args.reference, Path.cwd()),
                    output_dir=args.output_dir,
                    profile=qwen_image_edit_identity_profile_for_name(args.profile),
                    cases=args.case or (),
                    max_side=args.max_side,
                    steps=args.steps,
                    true_cfg_scale=args.true_cfg_scale,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                    max_sequence_length=args.max_sequence_length,
                    overwrite=args.overwrite,
                    nunchaku_blocks_on_gpu=args.nunchaku_blocks_on_gpu,
                    progress=progress,
                ),
                pretty=not args.compact,
            )
            return 0
    except (
        CharacterReferenceError,
        CharacterViewError,
        QwenImageEditIdentityError,
        KeyframeMemoryError,
        ManifestIOError,
        QwenVlmError,
    ) as error:
        dump_json(stderr, command_error_payload(error), pretty=not args.compact)
        return 1
