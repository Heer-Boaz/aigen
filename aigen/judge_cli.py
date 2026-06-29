from __future__ import annotations

import argparse
from pathlib import Path

from aigen.vlm_qwen import (
    DEFAULT_JUDGE_ID,
    DEFAULT_JUDGE_QUANTIZATION,
    DEFAULT_JUDGE_REPO_ID,
    DEFAULT_JUDGE_REVISION,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    QwenVlmConfig,
)
from aigen.runtime_profiles import MODELS_ROOT


def add_judge_runtime_args(parser: argparse.ArgumentParser, *, role: str, max_new_tokens: int) -> None:
    parser.add_argument("--judge", default=DEFAULT_JUDGE_ID, help=f"{role.capitalize()} id recorded in outputs")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_ROOT / "vlm/Qwen/Qwen2.5-VL-7B-Instruct",
        help="Local Qwen2.5-VL-7B-Instruct model directory",
    )
    parser.add_argument("--dtype", default="bfloat16", help=f"Torch dtype for {role} model weights")
    parser.add_argument("--attention-impl", default="sdpa", help="Transformers attention implementation")
    parser.add_argument(
        "--quantization",
        choices=("bitsandbytes-8bit", "none"),
        default=DEFAULT_JUDGE_QUANTIZATION,
        help=f"Local inference quantization for the {role}",
    )
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS, help="Minimum Qwen visual pixels")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Maximum Qwen visual pixels")
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens, help=f"{role.capitalize()} response token budget")
    parser.add_argument("--temperature", type=float, default=0.0, help=f"{role.capitalize()} sampling temperature")


def judge_config_from_args(args: argparse.Namespace) -> QwenVlmConfig:
    return QwenVlmConfig(
        judge_id=args.judge,
        model=args.model,
        repo_id=DEFAULT_JUDGE_REPO_ID,
        revision=DEFAULT_JUDGE_REVISION,
        dtype=args.dtype,
        attention_impl=args.attention_impl,
        quantization=args.quantization,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
