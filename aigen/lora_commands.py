from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
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


def add_lora_commands(subparsers: Any) -> None:
    lora = subparsers.add_parser("lora", help="LoRA dataset and training preparation tools")
    lora_subparsers = lora.add_subparsers(dest="lora_command", required=True)

    schema = lora_subparsers.add_parser("dataset-schema", help="Write the LoRA dataset JSON schema")
    schema.add_argument("--compact", action="store_true", help="Write compact JSON")

    dataset_build = lora_subparsers.add_parser("dataset-build", help="Build a curated LoRA training dataset")
    dataset_build.add_argument("spec", type=Path, help="LoRA dataset spec JSON")
    dataset_build.add_argument("--compact", action="store_true", help="Write compact JSON")

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
        else:
            payload = run_lora_training_plan(
                args.dataset_dir.resolve(),
                output_dir=args.output_dir,
                trainer_script=args.trainer_script,
                base_model=args.base_model,
                config=_train_config(args),
                dry_run=args.dry_run,
                progress=progress,
            )
        dump_json(stdout, payload, pretty=not args.compact)
        return 0
    except (LoraDatasetError, LoraTrainingError, ManifestIOError) as error:
        dump_json(stderr, command_error_payload(error), pretty=not args.compact)
        return 1


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset_dir", type=Path, help="Built LoRA dataset directory")
    parser.add_argument("--output-dir", type=Path, help="LoRA training output directory")
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
