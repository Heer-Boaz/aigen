from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.lora_dataset_models import LoraDatasetError, lora_dataset_schema
from aigen.lora_datasets import build_lora_dataset, flux_lora_training_preflight
from aigen.manifest_io import ManifestIOError


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


def run_lora_command(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    try:
        if args.lora_command == "dataset-schema":
            dump_json(stdout, lora_dataset_schema(), pretty=not args.compact)
            return 0
        if args.lora_command == "dataset-build":
            dump_json(
                stdout,
                build_lora_dataset(args.spec),
                pretty=not args.compact,
            )
            return 0
        dump_json(
            stdout,
            flux_lora_training_preflight(args.dataset_dir.resolve()),
            pretty=not args.compact,
        )
        return 0
    except (LoraDatasetError, ManifestIOError) as error:
        dump_json(stderr, command_error_payload(error), pretty=not args.compact)
        return 1
