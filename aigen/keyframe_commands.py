from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.judge_cli import add_judge_runtime_args, judge_config_from_args
from aigen.keyframe_examples import (
    KeyframeExampleError,
    KeyframeExampleExtractionConfig,
    extract_keyframe_example,
)
from aigen.keyframe_judge import (
    KeyframeJudgeError,
    judge_keyframe_run,
)
from aigen.keyframe_control_audit import (
    KeyframeControlAuditError,
    run_keyframe_control_audit,
)
from aigen.keyframe_memory import KeyframeMemoryError
from aigen.keyframe_polish import (
    KeyframePolishError,
    diagnose_keyframe_polish,
    keyframe_polish_job_schema,
    keyframe_polish_plan_schema,
    load_keyframe_polish_job,
    plan_keyframe_polish,
    preview_keyframe_polish_job,
    run_keyframe_polish_job,
    select_keyframe_polish,
    validate_keyframe_polish_job,
)
from aigen.keyframe_pose import KeyframePoseError
from aigen.keyframe_refine_models import (
    KeyframeRefineError,
    keyframe_refine_job_schema,
    load_keyframe_refine_job,
)
from aigen.keyframe_refine import (
    plan_keyframe_refine_job,
    run_keyframe_refine_job,
    validate_keyframe_refine_job,
)
from aigen.keyframe_score import (
    DEFAULT_SCORER_ID,
    KeyframeScoreConfig,
    KeyframeScoreError,
    score_keyframe_run,
    select_scored_keyframe_run,
)
from aigen.keyframe_job_models import (
    KeyframeJobError,
    keyframe_job_schema,
    load_keyframe_job,
)
from aigen.keyframes import (
    plan_keyframe_job,
    run_keyframe_audit_variant_job,
    run_keyframe_job,
    validate_keyframe_job,
)
from aigen.manifest_io import ManifestIOError
from aigen.runtime_profiles import (
    PROJECT_ROOT,
    keyframe_profile_for_name,
    keyframe_refine_profile_for_name,
)


def add_keyframe_commands(subparsers: Any) -> None:
    keyframes = subparsers.add_parser("keyframes", help="JSON-first character keyframe jobs")
    keyframe_subparsers = keyframes.add_subparsers(dest="keyframes_command", required=True)

    schema = keyframe_subparsers.add_parser("schema", help="Write the keyframe JSON schema to stdout")
    schema.add_argument("--compact", action="store_true", help="Write compact JSON")

    refine_schema = keyframe_subparsers.add_parser(
        "refine-schema",
        help="Write the keyframe refine JSON schema to stdout",
    )
    refine_schema.add_argument("--compact", action="store_true", help="Write compact JSON")

    polish_schema = keyframe_subparsers.add_parser(
        "polish-schema",
        help="Write the keyframe polish JSON schema to stdout",
    )
    polish_schema.add_argument("--compact", action="store_true", help="Write compact JSON")

    polish_plan_schema = keyframe_subparsers.add_parser(
        "polish-plan-schema",
        help="Write the generated keyframe polish plan JSON schema to stdout",
    )
    polish_plan_schema.add_argument("--compact", action="store_true", help="Write compact JSON")

    validate = keyframe_subparsers.add_parser("validate", help="Validate a keyframe job without running the GPU")
    validate.add_argument("job", type=Path, help="Keyframe job JSON")
    validate.add_argument("--compact", action="store_true", help="Write compact JSON")

    plan = keyframe_subparsers.add_parser("plan", help="Resolve a keyframe job without running the GPU")
    plan.add_argument("job", type=Path, help="Keyframe job JSON")
    plan.add_argument("--compact", action="store_true", help="Write compact JSON")

    run = keyframe_subparsers.add_parser("run", help="Run a resolved keyframe job")
    run.add_argument("job", type=Path, help="Keyframe job JSON")
    run.add_argument("--compact", action="store_true", help="Write compact JSON")

    run_audit_variant = keyframe_subparsers.add_parser(
        "run-audit-variant",
        help="Run one isolated control-audit variant job",
    )
    run_audit_variant.add_argument("job", type=Path, help="Control-audit variant job JSON")
    run_audit_variant.add_argument("--compact", action="store_true", help="Write compact JSON")

    extract_example = keyframe_subparsers.add_parser(
        "extract-example",
        help="Extract source pose, gray, lineart, softedge and mask controls from a keyframe example",
    )
    extract_example.add_argument("--source", type=Path, required=True, help="Source example image")
    extract_example.add_argument("--output-dir", type=Path, required=True, help="Directory for extracted assets")
    extract_example.add_argument("--name", required=True, help="Asset filename prefix")
    extract_example.add_argument("--width", type=int, required=True, help="Output condition width")
    extract_example.add_argument("--height", type=int, required=True, help="Output condition height")
    extract_example.add_argument(
        "--mirror-x",
        action="store_true",
        help="Mirror the example horizontally before extracting conditions",
    )
    extract_example.add_argument("--pose-device", default="cpu", help="DWPose device")
    extract_example.add_argument("--compact", action="store_true", help="Write compact JSON")

    refine_validate = keyframe_subparsers.add_parser(
        "refine-validate",
        help="Validate a keyframe refine job without running the GPU",
    )
    refine_validate.add_argument("job", type=Path, help="Keyframe refine job JSON")
    refine_validate.add_argument("--compact", action="store_true", help="Write compact JSON")

    refine_plan = keyframe_subparsers.add_parser(
        "refine-plan",
        help="Resolve a keyframe refine job without running the GPU",
    )
    refine_plan.add_argument("job", type=Path, help="Keyframe refine job JSON")
    refine_plan.add_argument("--compact", action="store_true", help="Write compact JSON")

    refine_run = keyframe_subparsers.add_parser(
        "refine-run",
        help="Run a keyframe local inpaint refine job",
    )
    refine_run.add_argument("job", type=Path, help="Keyframe refine job JSON")
    refine_run.add_argument("--compact", action="store_true", help="Write compact JSON")

    polish_plan = keyframe_subparsers.add_parser("polish-plan", help="Resolve a keyframe polish job without loading models")
    polish_plan.add_argument("job", type=Path, help="Keyframe polish job JSON")
    polish_plan.add_argument("--compact", action="store_true", help="Write compact JSON")

    polish_diagnose = keyframe_subparsers.add_parser(
        "polish-diagnose",
        help="Use the local VLM to plan model-discovered local polish regions",
    )
    polish_diagnose.add_argument("job", type=Path, help="Keyframe polish job JSON")
    add_judge_runtime_args(polish_diagnose, role="planner", max_new_tokens=1200)
    polish_diagnose.add_argument("--compact", action="store_true", help="Write compact JSON")

    polish_validate = keyframe_subparsers.add_parser(
        "polish-validate",
        help="Validate a keyframe polish job without running the GPU",
    )
    polish_validate.add_argument("job", type=Path, help="Keyframe polish job JSON")
    polish_validate.add_argument("--compact", action="store_true", help="Write compact JSON")

    polish_preview = keyframe_subparsers.add_parser(
        "polish-preview",
        help="Resolve a keyframe polish job without running the GPU",
    )
    polish_preview.add_argument("job", type=Path, help="Keyframe polish job JSON")
    polish_preview.add_argument("--compact", action="store_true", help="Write compact JSON")

    polish_run = keyframe_subparsers.add_parser(
        "polish-run",
        help="Run selected-candidate local detail polish",
    )
    polish_run.add_argument("job", type=Path, help="Keyframe polish job JSON")
    polish_run.add_argument("--compact", action="store_true", help="Write compact JSON")

    polish_select = keyframe_subparsers.add_parser(
        "polish-select",
        help="Use the local VLM to select local polish variants and write final_composite.png",
    )
    polish_select.add_argument("job", type=Path, help="Keyframe polish job JSON")
    add_judge_runtime_args(polish_select, role="selector", max_new_tokens=700)
    polish_select.add_argument("--compact", action="store_true", help="Write compact JSON")

    judge = keyframe_subparsers.add_parser("judge", help="Judge a completed keyframe run with a local VLM")
    judge.add_argument("run_dir", type=Path, help="Completed keyframe run directory")
    add_judge_runtime_args(judge, role="judge", max_new_tokens=900)
    judge.add_argument("--compact", action="store_true", help="Write compact JSON")

    score = keyframe_subparsers.add_parser("score", help="Score a completed keyframe run against its conditions")
    score.add_argument("run_dir", type=Path, help="Completed keyframe run directory")
    score.add_argument("--scorer", default=DEFAULT_SCORER_ID, help="Scorer id used for output directory naming")
    score.add_argument(
        "--contour-radius",
        type=int,
        default=8,
        help="Pixel radius used when matching candidate boundaries to target contours",
    )
    score.add_argument(
        "--distance-scale-px",
        type=float,
        default=48.0,
        help="Pixel distance that maps target-contour drift to zero distance score",
    )
    score.add_argument("--compact", action="store_true", help="Write compact JSON")

    score_select = keyframe_subparsers.add_parser(
        "score-select",
        help="Select/reject outputs from a condition score result",
    )
    score_select.add_argument("run_dir", type=Path, help="Completed keyframe run directory")
    score_select.add_argument("--from-scorer", default=DEFAULT_SCORER_ID, help="Scorer id to read")
    score_select.add_argument("--top-k", type=int, default=1, help="Number of ranked candidates to select")
    score_select.add_argument("--compact", action="store_true", help="Write compact JSON")

    control_audit = keyframe_subparsers.add_parser(
        "control-audit",
        help="Run fixed-seed control-off/strong-control ablations for a completed keyframe run",
    )
    control_audit.add_argument("run_dir", type=Path, help="Completed keyframe run directory")
    control_audit.add_argument("--seed", type=int, help="Seed to reuse for every audit variant")
    control_audit.add_argument("--compact", action="store_true", help="Write compact JSON")


def run_keyframe_command(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    try:
        if args.keyframes_command == "schema":
            dump_json(stdout, keyframe_job_schema(), pretty=not args.compact)
            return 0
        if args.keyframes_command == "refine-schema":
            dump_json(stdout, keyframe_refine_job_schema(), pretty=not args.compact)
            return 0
        if args.keyframes_command == "polish-schema":
            dump_json(stdout, keyframe_polish_job_schema(), pretty=not args.compact)
            return 0
        if args.keyframes_command == "polish-plan-schema":
            dump_json(stdout, keyframe_polish_plan_schema(), pretty=not args.compact)
            return 0
        if args.keyframes_command == "extract-example":
            dump_json(
                stdout,
                extract_keyframe_example(
                    KeyframeExampleExtractionConfig(
                        source=args.source,
                        output_dir=args.output_dir,
                        name=args.name,
                        width=args.width,
                        height=args.height,
                        mirror_x=args.mirror_x,
                        pose_device=args.pose_device,
                    )
                ),
                pretty=not args.compact,
            )
            return 0
        if args.keyframes_command == "judge":
            dump_json(
                stdout,
                judge_keyframe_run(
                    args.run_dir,
                    judge_config_from_args(args),
                    project_root=PROJECT_ROOT,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.keyframes_command == "score":
            dump_json(
                stdout,
                score_keyframe_run(
                    args.run_dir,
                    KeyframeScoreConfig(
                        scorer_id=args.scorer,
                        contour_radius=args.contour_radius,
                        distance_scale_px=args.distance_scale_px,
                    ),
                    project_root=PROJECT_ROOT,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.keyframes_command == "score-select":
            dump_json(
                stdout,
                select_scored_keyframe_run(
                    args.run_dir,
                    scorer_id=args.from_scorer,
                    top_k=args.top_k,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.keyframes_command == "control-audit":
            dump_json(
                stdout,
                run_keyframe_control_audit(
                    args.run_dir,
                    project_root=PROJECT_ROOT,
                    seed=args.seed,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.keyframes_command == "polish-plan":
            dump_json(
                stdout,
                plan_keyframe_polish(
                    args.job,
                    project_root=PROJECT_ROOT,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.keyframes_command == "polish-diagnose":
            dump_json(
                stdout,
                diagnose_keyframe_polish(
                    args.job,
                    config=judge_config_from_args(args),
                    project_root=PROJECT_ROOT,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.keyframes_command == "polish-select":
            dump_json(
                stdout,
                select_keyframe_polish(
                    args.job,
                    config=judge_config_from_args(args),
                    project_root=PROJECT_ROOT,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.keyframes_command == "polish-validate":
            dump_json(
                stdout,
                validate_keyframe_polish_job(args.job, project_root=PROJECT_ROOT),
                pretty=not args.compact,
            )
            return 0
        if args.keyframes_command in {"polish-preview", "polish-run"}:
            profile = _keyframe_polish_profile_for_job(args.job)
            if args.keyframes_command == "polish-preview":
                dump_json(
                    stdout,
                    preview_keyframe_polish_job(args.job, profile, project_root=PROJECT_ROOT),
                    pretty=not args.compact,
                )
                return 0
            dump_json(
                stdout,
                run_keyframe_polish_job(args.job, profile, project_root=PROJECT_ROOT),
                pretty=not args.compact,
            )
            return 0
        if args.keyframes_command in {"refine-validate", "refine-plan", "refine-run"}:
            profile = _keyframe_refine_profile_for_job(args.job)
            if args.keyframes_command == "refine-validate":
                dump_json(
                    stdout,
                    validate_keyframe_refine_job(args.job, profile, project_root=PROJECT_ROOT),
                    pretty=not args.compact,
                )
                return 0
            if args.keyframes_command == "refine-plan":
                dump_json(
                    stdout,
                    plan_keyframe_refine_job(args.job, profile, project_root=PROJECT_ROOT),
                    pretty=not args.compact,
                )
                return 0
            dump_json(
                stdout,
                run_keyframe_refine_job(args.job, profile, project_root=PROJECT_ROOT),
                pretty=not args.compact,
            )
            return 0
        profile = _keyframe_profile_for_job(args.job)
        if args.keyframes_command == "validate":
            dump_json(
                stdout,
                validate_keyframe_job(args.job, profile, project_root=PROJECT_ROOT),
                pretty=not args.compact,
            )
            return 0
        if args.keyframes_command == "plan":
            dump_json(
                stdout,
                plan_keyframe_job(args.job, profile, project_root=PROJECT_ROOT),
                pretty=not args.compact,
            )
            return 0
        if args.keyframes_command == "run-audit-variant":
            dump_json(
                stdout,
                run_keyframe_audit_variant_job(args.job, profile, project_root=PROJECT_ROOT),
                pretty=not args.compact,
            )
            return 0
        dump_json(
            stdout,
            run_keyframe_job(args.job, profile, project_root=PROJECT_ROOT),
            pretty=not args.compact,
        )
        return 0
    except (
        KeyframeExampleError,
        KeyframeJobError,
        KeyframeMemoryError,
        KeyframeJudgeError,
        KeyframePoseError,
        KeyframeScoreError,
        KeyframeControlAuditError,
        KeyframeRefineError,
        KeyframePolishError,
        ManifestIOError,
    ) as error:
        dump_json(stderr, command_error_payload(error), pretty=not args.compact)
        return 1


def _keyframe_profile_for_job(job_path: Path):
    return keyframe_profile_for_name(load_keyframe_job(job_path).pipeline.profile)


def _keyframe_refine_profile_for_job(job_path: Path):
    return keyframe_refine_profile_for_name(load_keyframe_refine_job(job_path).pipeline.profile)


def _keyframe_polish_profile_for_job(job_path: Path):
    return keyframe_refine_profile_for_name(load_keyframe_polish_job(job_path).pipeline.profile)
