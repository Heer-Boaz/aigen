from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.lora_control_audit import (
    DEFAULT_CONTROLNET_MODEL as DEFAULT_CONTROL_AUDIT_CONTROLNET_MODEL,
    DEFAULT_FLUX_BASE_MODEL as DEFAULT_CONTROL_AUDIT_BASE_MODEL,
    DEFAULT_NUNCHAKU_FLUX_TRANSFORMER as DEFAULT_CONTROL_AUDIT_NUNCHAKU_TRANSFORMER,
    LoraControlAuditConfig,
    LoraControlAuditError,
    build_lora_control_audit_plan,
    run_lora_control_audit,
)
from aigen.lora_dataset_models import LoraDatasetError, lora_dataset_schema
from aigen.lora_datasets import build_lora_dataset
from aigen.lora_smoke import LoraSmokeError, run_lora_smoke
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


def add_lora_commands(subparsers: Any) -> None:
    lora = subparsers.add_parser("lora", help="LoRA dataset and training preparation tools")
    lora_subparsers = lora.add_subparsers(dest="lora_command", required=True)

    schema = lora_subparsers.add_parser("dataset-schema", help="Write the LoRA dataset JSON schema")
    schema.add_argument("--compact", action="store_true", help="Write compact JSON")

    dataset_build = lora_subparsers.add_parser("dataset-build", help="Build a curated LoRA training dataset")
    dataset_build.add_argument("spec", type=Path, help="LoRA dataset spec JSON")
    dataset_build.add_argument("--compact", action="store_true", help="Write compact JSON")

    smoke = lora_subparsers.add_parser(
        "smoke",
        help="Build an anchor-approved identity LoRA smoke dataset and local train plan",
    )
    smoke.add_argument("--id", required=True, help="Smoke run id")
    smoke.add_argument("--character-id", required=True, help="Character id")
    smoke.add_argument("--trigger-token", required=True, help="LoRA trigger token")
    smoke.add_argument(
        "--anchor",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Human-approved anchor image; repeat for multiple anchors",
    )
    smoke.add_argument(
        "--identity-caption",
        required=True,
        help="Identity-only caption template; do not include action pose text",
    )
    smoke.add_argument("--tag", action="append", default=[], help="Additional caption tag; repeat as needed")
    smoke.add_argument("--approved-by", default="user", help="Approver recorded in the smoke manifest")
    smoke.add_argument("--output-dir", required=True, type=Path, help="Smoke output directory")
    smoke.add_argument("--overwrite", action="store_true", help="Replace an existing smoke output directory")
    _add_train_runtime_args(smoke)
    smoke.add_argument("--compact", action="store_true", help="Write compact JSON")

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
    control_audit_plan.add_argument(
        "--identity-prompt",
        required=True,
        help="Curated identity prompt used for LoRA audit inference; must include the trigger token",
    )
    control_audit_plan.add_argument("--width", type=_positive_int, default=512)
    control_audit_plan.add_argument("--height", type=_positive_int, default=768)
    control_audit_plan.add_argument("--steps", type=_positive_int, default=20)
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
    control_audit_run.add_argument(
        "--identity-prompt",
        required=True,
        help="Curated identity prompt used for LoRA audit inference; must include the trigger token",
    )
    control_audit_run.add_argument("--width", type=_positive_int, default=512)
    control_audit_run.add_argument("--height", type=_positive_int, default=768)
    control_audit_run.add_argument("--steps", type=_positive_int, default=20)
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
        if args.lora_command == "dataset-build":
            progress.phase("build dataset")
            dump_json(
                stdout,
                build_lora_dataset(args.spec, progress=progress),
                pretty=not args.compact,
            )
            return 0
        if args.lora_command == "smoke":
            payload = run_lora_smoke(
                smoke_id=args.id,
                character_id=args.character_id,
                trigger_token=args.trigger_token,
                anchor_specs=args.anchor,
                identity_caption=args.identity_caption,
                tags=args.tag,
                approved_by=args.approved_by,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                trainer_script=args.trainer_script,
                base_model=args.base_model,
                config=_train_config(args),
                progress=progress,
            )
        elif args.lora_command == "training-preflight":
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
                identity_prompt=args.identity_prompt,
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
                    guidance_scale=args.guidance_scale,
                    controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                    control_guidance_end=args.control_guidance_end,
                    lora_strength=args.lora_strength,
                    seed=args.seed,
                ),
            )
        else:
            payload = run_lora_control_audit(
                args.lora_run_dir,
                case_specs=args.case,
                identity_prompt=args.identity_prompt,
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
    except (LoraDatasetError, LoraTrainingError, LoraSmokeError, LoraControlAuditError, ManifestIOError) as error:
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
        help="Local 4-bit FLUX base model directory",
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
