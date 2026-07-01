from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.judge_cli import add_judge_runtime_args, judge_config_from_args
from aigen.lora_control_audit import (
    DEFAULT_CONTROLNET_MODEL as DEFAULT_CONTROL_AUDIT_CONTROLNET_MODEL,
    DEFAULT_FLUX_BASE_MODEL as DEFAULT_CONTROL_AUDIT_BASE_MODEL,
    DEFAULT_NUNCHAKU_FLUX_TRANSFORMER as DEFAULT_CONTROL_AUDIT_NUNCHAKU_TRANSFORMER,
    LoraControlAuditConfig,
    LoraControlAuditError,
    build_lora_control_audit_plan,
    run_lora_control_audit,
)
from aigen.lora_candidates import (
    LoraCandidateError,
    build_lora_candidate_evidence,
    plan_lora_candidates,
    run_lora_candidate_plan,
    review_lora_candidates,
)
from aigen.lora_candidate_models import LoraCandidateBriefError, lora_candidate_brief_schema
from aigen.lora_candidate_judge import LoraCandidateJudgeError, judge_lora_candidate_evidence
from aigen.lora_candidate_planner import LoraCandidateBriefPlanConfig, plan_lora_candidate_brief
from aigen.lora_candidate_profiles import (
    LORA_CANDIDATE_PROFILE,
    LoraCandidateProfileError,
    lora_candidate_profile_for_name,
)
from aigen.lora_canon import LoraCanonError, audit_lora_dataset_source, init_lora_canon
from aigen.lora_dataset_models import LoraDatasetError, lora_dataset_schema
from aigen.lora_datasets import build_lora_dataset
from aigen.lora_training import (
    DEFAULT_BASE_MODEL,
    DEFAULT_TRAINER_SCRIPT,
    LoraLocalTrainConfig,
    LoraTrainingError,
    build_lora_train_plan,
    lora_training_preflight,
    run_lora_training_plan,
)
from aigen.manifest_io import ManifestIOError
from aigen.progress import StatusReporter
from aigen.runtime_profiles import PROJECT_ROOT
from aigen.vlm_qwen import QwenVlmError


def add_lora_commands(subparsers: Any) -> None:
    lora = subparsers.add_parser("lora", help="LoRA dataset and training preparation tools")
    lora_subparsers = lora.add_subparsers(dest="lora_command", required=True)

    schema = lora_subparsers.add_parser("dataset-schema", help="Write the LoRA dataset JSON schema")
    schema.add_argument("--compact", action="store_true", help="Write compact JSON")

    candidate_brief_schema = lora_subparsers.add_parser(
        "candidate-brief-schema",
        help="Write the LoRA candidate brief JSON schema",
    )
    candidate_brief_schema.add_argument("--compact", action="store_true", help="Write compact JSON")

    candidate_brief_plan = lora_subparsers.add_parser(
        "candidate-brief-plan",
        help="Use the local VLM to write a LoRA candidate brief from approved canon images",
    )
    candidate_brief_plan.add_argument("canon_dir", type=Path, help="Approved LoRA canon directory")
    candidate_brief_plan.add_argument("--output", type=Path, required=True, help="Generated candidate brief JSON")
    candidate_brief_plan.add_argument(
        "--candidate-output-dir",
        type=Path,
        required=True,
        help="Directory where candidate-plan should write the generated batch",
    )
    candidate_brief_plan.add_argument("--width", type=_positive_int, default=576)
    candidate_brief_plan.add_argument("--height", type=_positive_int, default=864)
    candidate_brief_plan.add_argument("--steps", type=_positive_int, default=24)
    candidate_brief_plan.add_argument("--seed-start", type=_non_negative_int, default=1)
    candidate_brief_plan.add_argument("--seeds-per-candidate", type=_positive_int, default=256)
    candidate_brief_plan.add_argument("--candidate-count", type=_positive_int, default=12)
    add_judge_runtime_args(candidate_brief_plan, role="candidate planner", max_new_tokens=3200)
    candidate_brief_plan.add_argument("--compact", action="store_true", help="Write compact JSON")

    dataset_build = lora_subparsers.add_parser("dataset-build", help="Build a curated LoRA training dataset")
    dataset_build.add_argument("spec", type=Path, help="LoRA dataset spec JSON")
    dataset_build.add_argument("--compact", action="store_true", help="Write compact JSON")

    canon_init = lora_subparsers.add_parser("canon-init", help="Create a human-approved LoRA canon folder")
    canon_init.add_argument("--character-id", required=True, help="Character id")
    canon_init.add_argument("--trigger-token", required=True, help="LoRA trigger token")
    canon_init.add_argument("--identity-prompt", required=True, help="Identity caption without the trigger token")
    canon_init.add_argument(
        "--anchor",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Human-approved canonical image; repeat for multiple anchors",
    )
    canon_init.add_argument("--approved-by", default="user", help="Approver recorded in the canon manifest")
    canon_init.add_argument("--output-dir", type=Path, help="Canon output directory")
    canon_init.add_argument("--overwrite", action="store_true", help="Replace an existing canon output directory")
    canon_init.add_argument("--compact", action="store_true", help="Write compact JSON")

    dataset_audit = lora_subparsers.add_parser(
        "dataset-audit",
        help="Build contact sheets and audit manifests for a canon folder or candidate pool",
    )
    dataset_audit.add_argument("source_dir", type=Path, help="Canon folder or candidate image directory")
    dataset_audit.add_argument("--output-dir", type=Path, help="Audit output directory")
    dataset_audit.add_argument("--overwrite", action="store_true", help="Replace an existing audit output directory")
    dataset_audit.add_argument("--compact", action="store_true", help="Write compact JSON")

    candidate_plan = lora_subparsers.add_parser(
        "candidate-plan",
        help="Plan a high-volume strict-filter LoRA candidate batch from a candidate brief",
    )
    candidate_plan.add_argument("brief", type=Path, help="LoRA candidate brief JSON")
    candidate_plan.add_argument("--compact", action="store_true", help="Write compact JSON")

    candidate_run = lora_subparsers.add_parser(
        "candidate-run",
        help="Generate images for a LoRA candidate batch plan",
    )
    candidate_run.add_argument("candidate_dir", type=Path, help="Candidate batch directory")
    candidate_run.add_argument("--profile", default=LORA_CANDIDATE_PROFILE.name, help="LoRA candidate generation profile")
    candidate_run.add_argument("--guidance-scale", type=_positive_float, default=2.5)
    candidate_run.add_argument("--max-sequence-length", type=_positive_int, default=128)
    candidate_run.add_argument("--overwrite", action="store_true", help="Replace existing candidate images")
    candidate_run.add_argument("--compact", action="store_true", help="Write compact JSON")

    candidate_evidence = lora_subparsers.add_parser(
        "candidate-evidence",
        help="Build LoRA candidate crop evidence for human review",
    )
    candidate_evidence.add_argument("candidate_dir", type=Path, help="Candidate batch directory")
    candidate_evidence.add_argument("--overwrite", action="store_true", help="Replace an existing evidence directory")
    candidate_evidence.add_argument("--compact", action="store_true", help="Write compact JSON")

    candidate_judge = lora_subparsers.add_parser(
        "candidate-judge",
        help="Judge LoRA candidate evidence with the local VLM before human review",
    )
    candidate_judge.add_argument("candidate_dir", type=Path, help="Candidate batch directory with evidence/review_items.json")
    candidate_judge.add_argument("--overwrite", action="store_true", help="Replace existing candidate judge output")
    add_judge_runtime_args(candidate_judge, role="candidate judge", max_new_tokens=900)
    candidate_judge.add_argument("--compact", action="store_true", help="Write compact JSON")

    candidate_review = lora_subparsers.add_parser(
        "candidate-review",
        help="Write human-approved LoRA candidates and quota report",
    )
    candidate_review.add_argument("candidate_dir", type=Path, help="Candidate batch directory with evidence/passed.json")
    candidate_review.add_argument(
        "--accept",
        action="append",
        required=True,
        metavar="NAME",
        help="Candidate name approved for LoRA training; repeat for multiple accepted candidates",
    )
    candidate_review.add_argument("--approved-by", default="user", help="Approver recorded in accepted.json")
    candidate_review.add_argument("--overwrite", action="store_true", help="Replace an existing review directory")
    candidate_review.add_argument("--compact", action="store_true", help="Write compact JSON")

    training_preflight = lora_subparsers.add_parser(
        "training-preflight",
        help="Report whether the local GPU is suitable for FLUX LoRA training",
    )
    training_preflight.add_argument("dataset_dir", type=Path, help="Built LoRA dataset directory")
    training_preflight.add_argument("--compact", action="store_true", help="Write compact JSON")

    train_plan = lora_subparsers.add_parser("train-plan", help="Build the local 16GB FLUX LoRA training plan")
    _add_train_args(train_plan)
    train_plan.add_argument("--compact", action="store_true", help="Write compact JSON")

    train_run = lora_subparsers.add_parser("train-run", help="Run the local 16GB FLUX LoRA training plan")
    _add_train_args(train_run)
    train_run.add_argument("--dry-run", action="store_true", help="Write the plan without launching training")
    train_run.add_argument("--compact", action="store_true", help="Write compact JSON")

    control_audit_plan = lora_subparsers.add_parser(
        "control-audit-plan",
        help="Plan a plain FLUX ControlNet audit for a trained character LoRA",
    )
    control_audit_plan.add_argument("lora_run_dir", type=Path, help="Completed LoRA training output directory")
    control_audit_plan.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="NAME=CONTROL_IMAGE",
        help="Control audit case; repeat for each pose/control image",
    )
    control_audit_plan.add_argument(
        "--baseline",
        action="append",
        required=True,
        metavar="NAME=BASELINE_IMAGE",
        help="Kontext-reference baseline image for the same case; repeat once for each --case",
    )
    control_audit_plan.add_argument(
        "--case-prompt",
        action="append",
        required=True,
        metavar="NAME=PROMPT",
        help="Curated case prompt; repeat once for each --case",
    )
    control_audit_plan.add_argument("--output-dir", type=Path, help="Control audit output directory")
    control_audit_plan.add_argument("--lora-weights", type=Path, help="LoRA weights safetensors path")
    control_audit_plan.add_argument(
        "--base-model",
        type=Path,
        default=DEFAULT_CONTROL_AUDIT_BASE_MODEL,
        help="Local FLUX.1-dev base pipeline directory",
    )
    control_audit_plan.add_argument(
        "--controlnet-model",
        type=Path,
        default=DEFAULT_CONTROL_AUDIT_CONTROLNET_MODEL,
        help="Local Union-Pro ControlNet directory",
    )
    control_audit_plan.add_argument(
        "--nunchaku-transformer",
        type=Path,
        default=DEFAULT_CONTROL_AUDIT_NUNCHAKU_TRANSFORMER,
        help="Plain FLUX.1-dev Nunchaku transformer",
    )
    control_audit_plan.add_argument("--trigger-token", help="LoRA trigger token")
    control_audit_plan.add_argument("--width", type=_positive_int, default=512)
    control_audit_plan.add_argument("--height", type=_positive_int, default=768)
    control_audit_plan.add_argument("--steps", type=_positive_int, default=20)
    control_audit_plan.add_argument("--max-sequence-length", type=_positive_int, default=128)
    control_audit_plan.add_argument("--guidance-scale", type=_positive_float, default=2.5)
    control_audit_plan.add_argument("--controlnet-conditioning-scale", type=_positive_float, default=0.8)
    control_audit_plan.add_argument("--control-guidance-end", type=_positive_float, default=0.65)
    control_audit_plan.add_argument("--lora-strength", type=_positive_float, default=1.0)
    control_audit_plan.add_argument("--seed", type=_non_negative_int, default=1)
    control_audit_plan.add_argument("--compact", action="store_true", help="Write compact JSON")

    control_audit_run = lora_subparsers.add_parser(
        "control-audit-run",
        help="Run a plain FLUX ControlNet audit for a trained character LoRA",
    )
    control_audit_run.add_argument("lora_run_dir", type=Path, help="Completed LoRA training output directory")
    control_audit_run.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="NAME=CONTROL_IMAGE",
        help="Control audit case; repeat for each pose/control image",
    )
    control_audit_run.add_argument(
        "--baseline",
        action="append",
        required=True,
        metavar="NAME=BASELINE_IMAGE",
        help="Kontext-reference baseline image for the same case; repeat once for each --case",
    )
    control_audit_run.add_argument(
        "--case-prompt",
        action="append",
        required=True,
        metavar="NAME=PROMPT",
        help="Curated case prompt; repeat once for each --case",
    )
    control_audit_run.add_argument("--output-dir", type=Path, help="Control audit output directory")
    control_audit_run.add_argument("--lora-weights", type=Path, help="LoRA weights safetensors path")
    control_audit_run.add_argument(
        "--base-model",
        type=Path,
        default=DEFAULT_CONTROL_AUDIT_BASE_MODEL,
        help="Local FLUX.1-dev base pipeline directory",
    )
    control_audit_run.add_argument(
        "--controlnet-model",
        type=Path,
        default=DEFAULT_CONTROL_AUDIT_CONTROLNET_MODEL,
        help="Local Union-Pro ControlNet directory",
    )
    control_audit_run.add_argument(
        "--nunchaku-transformer",
        type=Path,
        default=DEFAULT_CONTROL_AUDIT_NUNCHAKU_TRANSFORMER,
        help="Plain FLUX.1-dev Nunchaku transformer",
    )
    control_audit_run.add_argument("--trigger-token", help="LoRA trigger token")
    control_audit_run.add_argument("--width", type=_positive_int, default=512)
    control_audit_run.add_argument("--height", type=_positive_int, default=768)
    control_audit_run.add_argument("--steps", type=_positive_int, default=20)
    control_audit_run.add_argument("--max-sequence-length", type=_positive_int, default=128)
    control_audit_run.add_argument("--guidance-scale", type=_positive_float, default=2.5)
    control_audit_run.add_argument("--controlnet-conditioning-scale", type=_positive_float, default=0.8)
    control_audit_run.add_argument("--control-guidance-end", type=_positive_float, default=0.65)
    control_audit_run.add_argument("--lora-strength", type=_positive_float, default=1.0)
    control_audit_run.add_argument("--seed", type=_non_negative_int, default=1)
    control_audit_run.add_argument("--compact", action="store_true", help="Write compact JSON")


def run_lora_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    try:
        if args.lora_command == "dataset-schema":
            dump_json(stdout, lora_dataset_schema(), pretty=not args.compact)
            return 0
        if args.lora_command == "candidate-brief-schema":
            dump_json(stdout, lora_candidate_brief_schema(), pretty=not args.compact)
            return 0
        if args.lora_command == "candidate-brief-plan":
            payload = plan_lora_candidate_brief(
                args.canon_dir,
                judge_config_from_args(args),
                output_path=args.output,
                plan_config=LoraCandidateBriefPlanConfig(
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    seed_start=args.seed_start,
                    seeds_per_candidate=args.seeds_per_candidate,
                    candidate_count=args.candidate_count,
                    candidate_output_dir=args.candidate_output_dir,
                ),
                project_root=PROJECT_ROOT,
                progress=progress,
            )
            dump_json(stdout, payload, pretty=not args.compact)
            return 0
        if args.lora_command == "dataset-build":
            progress.phase("build dataset")
            dump_json(
                stdout,
                build_lora_dataset(args.spec, progress=progress),
                pretty=not args.compact,
            )
            return 0
        if args.lora_command == "canon-init":
            payload = init_lora_canon(
                character_id=args.character_id,
                trigger_token=args.trigger_token,
                identity_prompt=args.identity_prompt,
                anchor_specs=args.anchor,
                output_dir=args.output_dir,
                approved_by=args.approved_by,
                overwrite=args.overwrite,
                progress=progress,
            )
            dump_json(stdout, payload, pretty=not args.compact)
            return 0
        if args.lora_command == "dataset-audit":
            payload = audit_lora_dataset_source(
                args.source_dir,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                progress=progress,
            )
            dump_json(stdout, payload, pretty=not args.compact)
            return 0
        if args.lora_command == "candidate-plan":
            payload = plan_lora_candidates(
                brief_path=args.brief,
                progress=progress,
            )
            dump_json(stdout, payload, pretty=not args.compact)
            return 0
        if args.lora_command == "candidate-run":
            payload = run_lora_candidate_plan(
                args.candidate_dir,
                profile=lora_candidate_profile_for_name(args.profile),
                guidance_scale=args.guidance_scale,
                max_sequence_length=args.max_sequence_length,
                overwrite=args.overwrite,
                progress=progress,
            )
            dump_json(stdout, payload, pretty=not args.compact)
            return 0
        if args.lora_command == "candidate-evidence":
            payload = build_lora_candidate_evidence(
                args.candidate_dir,
                overwrite=args.overwrite,
                progress=progress,
            )
            dump_json(stdout, payload, pretty=not args.compact)
            return 0
        if args.lora_command == "candidate-judge":
            payload = judge_lora_candidate_evidence(
                args.candidate_dir,
                judge_config_from_args(args),
                overwrite=args.overwrite,
                progress=progress,
            )
            dump_json(stdout, payload, pretty=not args.compact)
            return 0
        if args.lora_command == "candidate-review":
            payload = review_lora_candidates(
                args.candidate_dir,
                accepted_names=args.accept,
                approved_by=args.approved_by,
                overwrite=args.overwrite,
                progress=progress,
            )
            dump_json(stdout, payload, pretty=not args.compact)
            return 0
        if args.lora_command == "training-preflight":
            payload = lora_training_preflight(args.dataset_dir.resolve())
        elif args.lora_command == "train-plan":
            payload = build_lora_train_plan(
                args.dataset_dir.resolve(),
                output_dir=args.output_dir,
                trainer_script=args.trainer_script,
                base_model=args.base_model,
                config=_train_config(args),
            )
        elif args.lora_command == "train-run":
            payload = run_lora_training_plan(
                args.dataset_dir.resolve(),
                output_dir=args.output_dir,
                trainer_script=args.trainer_script,
                base_model=args.base_model,
                config=_train_config(args),
                dry_run=args.dry_run,
                progress=progress,
            )
        elif args.lora_command == "control-audit-plan":
            payload = build_lora_control_audit_plan(
                args.lora_run_dir,
                case_specs=args.case,
                baseline_specs=args.baseline,
                case_prompt_specs=args.case_prompt,
                output_dir=args.output_dir,
                lora_weights=args.lora_weights,
                base_model=args.base_model,
                controlnet_model=args.controlnet_model,
                nunchaku_transformer=args.nunchaku_transformer,
                trigger_token=args.trigger_token,
                config=LoraControlAuditConfig(
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    max_sequence_length=args.max_sequence_length,
                    guidance_scale=args.guidance_scale,
                    controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                    control_guidance_end=args.control_guidance_end,
                    lora_strength=args.lora_strength,
                    seed=args.seed,
                ),
            )
        elif args.lora_command == "control-audit-run":
            payload = run_lora_control_audit(
                args.lora_run_dir,
                case_specs=args.case,
                baseline_specs=args.baseline,
                case_prompt_specs=args.case_prompt,
                output_dir=args.output_dir,
                lora_weights=args.lora_weights,
                base_model=args.base_model,
                controlnet_model=args.controlnet_model,
                nunchaku_transformer=args.nunchaku_transformer,
                trigger_token=args.trigger_token,
                config=LoraControlAuditConfig(
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    max_sequence_length=args.max_sequence_length,
                    guidance_scale=args.guidance_scale,
                    controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                    control_guidance_end=args.control_guidance_end,
                    lora_strength=args.lora_strength,
                    seed=args.seed,
                ),
                progress=progress,
            )
        dump_json(stdout, payload, pretty=not args.compact)
        return 0
    except (
        LoraCanonError,
        LoraDatasetError,
        LoraTrainingError,
        LoraControlAuditError,
        LoraCandidateJudgeError,
        LoraCandidateError,
        LoraCandidateBriefError,
        LoraCandidateProfileError,
        QwenVlmError,
        ManifestIOError,
    ) as error:
        dump_json(stderr, command_error_payload(error), pretty=not args.compact)
        return 1


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset_dir", type=Path, help="Built LoRA dataset directory")
    parser.add_argument("--output-dir", type=Path, help="LoRA training output directory")
    _add_train_runtime_args(parser)


def _add_train_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trainer-script",
        type=Path,
        default=DEFAULT_TRAINER_SCRIPT,
        help="Pinned Diffusers FLUX DreamBooth-LoRA trainer script",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=DEFAULT_BASE_MODEL,
        help="Local FLUX Diffusers model directory for LoRA training",
    )
    parser.add_argument("--resolution", type=_positive_int, default=512)
    parser.add_argument("--rank", type=_positive_int, default=4)
    parser.add_argument("--lora-alpha", type=_positive_int, default=4)
    parser.add_argument("--max-train-steps", type=_positive_int, default=800)
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int, default=4)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-4)
    parser.add_argument("--max-sequence-length", type=_positive_int, default=128)
    parser.add_argument("--seed", type=_non_negative_int, default=1)
    parser.add_argument("--mixed-precision", choices=("fp16", "bf16"), default="bf16")


def _train_config(args: argparse.Namespace) -> LoraLocalTrainConfig:
    return LoraLocalTrainConfig(
        resolution=args.resolution,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        max_train_steps=args.max_train_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_sequence_length=args.max_sequence_length,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed
