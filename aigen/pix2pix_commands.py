from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import command_error_payload, dump_json
from aigen.manifest_io import ManifestIOError
from aigen.pix2pix.errors import Pix2PixError
from aigen.progress import StatusReporter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pix2pix-baseline.json"


def add_pix2pix_commands(subparsers: Any) -> None:
    command = subparsers.add_parser(
        "pix2pix",
        help="train and run the paired U-Net/PatchGAN pixel-art translator",
    )
    actions = command.add_subparsers(dest="pix2pix_action", required=True)

    actions.add_parser("contract", help="print the paired dataset contract")

    audit = actions.add_parser("audit", help="fully validate and fingerprint a dataset")
    audit.add_argument("dataset", type=Path)

    train = actions.add_parser("train", help="train the pix2pix baseline")
    train.add_argument("dataset", type=Path)
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--device", choices=("cuda", "cpu"), default="cuda")

    infer = actions.add_parser("infer", help="run an exported generator")
    infer.add_argument("--model-dir", type=Path, required=True)
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--output-size", type=int)
    infer.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    infer.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")

    evaluate = actions.add_parser(
        "evaluate",
        help="write predictions and metrics for an audited split",
    )
    evaluate.add_argument("--model-dir", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--num-workers", type=int, default=4)
    evaluate.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    evaluate.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")

    iro_plan = actions.add_parser(
        "iro-plan",
        help="materialize an immutable quota-balanced iRO renderer plan",
    )
    iro_plan.add_argument("--config", type=Path, required=True)
    iro_plan.add_argument("--output-dir", type=Path, required=True)

    iro_render = actions.add_parser(
        "iro-render",
        help="resume native iRO APNG acquisition for a corpus plan",
    )
    iro_render.add_argument("corpus", type=Path)

    iro_select = actions.add_parser(
        "iro-select",
        help="deduplicate and select one native target per planned request",
    )
    iro_select.add_argument("corpus", type=Path)

    iro_sources = actions.add_parser(
        "iro-generate-sources",
        help="resume atomically sharded FLUX reverse-source generation",
    )
    iro_sources.add_argument("corpus", type=Path)

    iro_qwen_sources = actions.add_parser(
        "iro-generate-qwen-sources",
        help="resume atomically sharded Qwen reverse-source generation",
    )
    iro_qwen_sources.add_argument("corpus", type=Path)
    iro_qwen_sources.add_argument("--config", type=Path, required=True)

    iro_prepare = actions.add_parser(
        "iro-prepare",
        help="assemble and audit the final native-128 paired dataset",
    )
    iro_prepare.add_argument("corpus", type=Path)

    iro_qwen_prepare = actions.add_parser(
        "iro-prepare-qwen",
        help="assemble and audit the Qwen native-128 paired dataset",
    )
    iro_qwen_prepare.add_argument("corpus", type=Path)


def run_pix2pix_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    try:
        result = _run_pix2pix_action(args, progress)
    except (Pix2PixError, ManifestIOError) as error:
        dump_json(stderr, command_error_payload(error), pretty=True)
        return 1
    dump_json(stdout, result, pretty=True)
    return 0


def _run_pix2pix_action(
    args: argparse.Namespace,
    progress: StatusReporter,
) -> dict[str, object]:
    if args.pix2pix_action == "contract":
        from aigen.pix2pix.dataset import dataset_contract

        return dataset_contract()
    if args.pix2pix_action == "audit":
        from aigen.pix2pix.dataset import audit_dataset

        progress.phase("audit paired dataset")
        return audit_dataset(args.dataset).to_json()
    if args.pix2pix_action == "train":
        from aigen.pix2pix.training import train_pix2pix

        return train_pix2pix(
            args.dataset,
            args.config,
            args.output_dir,
            resume_checkpoint=args.resume,
            device_name=args.device,
            progress=progress,
        )
    if args.pix2pix_action == "infer":
        from aigen.pix2pix.inference import run_inference

        progress.phase("run pix2pix generator")
        return run_inference(
            args.model_dir,
            args.input,
            args.output,
            device_name=args.device,
            precision=args.precision,
            output_size=args.output_size,
        )
    if args.pix2pix_action == "evaluate":
        from aigen.pix2pix.validation import evaluate_model

        return evaluate_model(
            args.model_dir,
            args.dataset,
            args.output_dir,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
            precision=args.precision,
            progress=progress,
        )
    if args.pix2pix_action == "iro-plan":
        from aigen.pix2pix.iro_corpus import plan_iro_corpus

        progress.phase("plan quota-balanced iRO corpus")
        return plan_iro_corpus(args.config, args.output_dir)
    if args.pix2pix_action == "iro-render":
        from aigen.pix2pix.iro_corpus import render_iro_corpus

        return render_iro_corpus(args.corpus, progress=progress)
    if args.pix2pix_action == "iro-select":
        from aigen.pix2pix.iro_corpus import select_iro_targets

        progress.phase("select unique native iRO targets")
        return select_iro_targets(args.corpus)
    if args.pix2pix_action == "iro-generate-sources":
        from aigen.pix2pix.flux_source_corpus import generate_flux_sources

        return generate_flux_sources(args.corpus, progress=progress)
    if args.pix2pix_action == "iro-generate-qwen-sources":
        from aigen.pix2pix.qwen_source_corpus import generate_qwen_sources

        return generate_qwen_sources(
            args.corpus,
            args.config,
            progress=progress,
        )
    if args.pix2pix_action == "iro-prepare":
        from aigen.pix2pix.corpus_dataset import prepare_iro_dataset

        progress.phase("assemble audited pix2pix dataset")
        return prepare_iro_dataset(args.corpus)
    if args.pix2pix_action == "iro-prepare-qwen":
        from aigen.pix2pix.corpus_dataset import prepare_iro_qwen_dataset

        progress.phase("assemble audited Qwen pix2pix dataset")
        return prepare_iro_qwen_dataset(args.corpus)
    raise RuntimeError("unsupported pix2pix action")
