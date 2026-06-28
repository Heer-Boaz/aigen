from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.command_io import dump_json, write_json
from aigen.models.downloads import (
    ModelDownloadError,
    download_models,
    load_download_manifest,
)


def add_model_commands(subparsers: Any) -> None:
    models = subparsers.add_parser("models", help="Model download tools")
    model_subparsers = models.add_subparsers(dest="models_command", required=True)

    download = model_subparsers.add_parser(
        "download",
        help="Download Hugging Face model repos from a model download manifest",
    )
    download.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Model download manifest JSON",
    )
    download.add_argument(
        "--models-root",
        type=Path,
        required=True,
        help="Directory where downloaded model categories are stored",
    )
    download.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path to write the download result; defaults to stdout",
    )
    download.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned downloads without downloading model files",
    )
    download.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of pretty printed JSON",
    )


def run_model_command(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    try:
        result = download_models(
            load_download_manifest(args.manifest),
            args.models_root,
            dry_run=args.dry_run,
        )
    except ModelDownloadError as error:
        dump_json(stderr, error.to_json(), pretty=not args.compact)
        return 1
    payload = result.to_json()
    if args.output:
        write_json(args.output, payload, pretty=not args.compact)
    else:
        dump_json(stdout, payload, pretty=not args.compact)
    return 0
