from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TextIO

from aigen.generation.character_concept import (
    DEFAULT_NEGATIVE_PROMPT,
    CharacterConceptError,
    run_character_concept,
)
from aigen.generation.pixel_art import (
    PixelArtError,
    run_pixel_art,
)
from aigen.generation.kontext_pose_control import (
    CharacterKontextPoseError,
    run_character_kontext_pose_control,
)
from aigen.generation.nunchaku_kontext import (
    NunchakuKontextError,
    run_nunchaku_kontext,
)
from aigen.generation.pose_control import (
    CharacterPoseError,
    run_character_pose_control,
)
from aigen.character_views import (
    CharacterViewError,
    accept_character_view,
    character_view_bank_schema,
    character_view_job_schema,
    left_profile_view_template,
    load_character_view_job,
    plan_character_view_job,
    run_character_view_job,
    validate_character_view_job,
)
from aigen.keyframes import (
    KeyframeJobError,
    KeyframeProfile,
    c2_profile_template,
    keyframe_job_schema,
    load_keyframe_job,
    plan_keyframe_job,
    run_keyframe_job,
    validate_keyframe_job,
)
from aigen.keyframe_judge import (
    DEFAULT_CALIBRATION_FIXTURE,
    DEFAULT_JUDGE_ID,
    DEFAULT_JUDGE_QUANTIZATION,
    DEFAULT_JUDGE_REPO_ID,
    DEFAULT_JUDGE_REVISION,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    KeyframeJudgeConfig,
    KeyframeJudgeError,
    calibrate_keyframe_judge,
    judge_keyframe_run,
    select_keyframe_run,
)
from aigen.keyframe_score import (
    DEFAULT_SCORER_ID,
    KeyframeScoreError,
    KeyframeScoreConfig,
    select_scored_keyframe_run,
    score_keyframe_run,
)
from aigen.keyframe_examples import (
    KeyframeExampleError,
    KeyframeExampleExtractionConfig,
    extract_keyframe_example,
)
from aigen.keyframe_pose import KeyframePoseError
from aigen.keyframe_refine import (
    KeyframeRefineError,
    KeyframeRefineProfile,
    keyframe_refine_job_schema,
    load_keyframe_refine_job,
    plan_keyframe_refine_job,
    run_keyframe_refine_job,
    run_keyframe_refine_variant,
    validate_keyframe_refine_job,
)
from aigen.keyframe_polish import (
    KeyframePolishError,
    diagnose_keyframe_polish,
    keyframe_polish_job_schema,
    load_keyframe_polish_job,
    plan_keyframe_polish_job,
    run_keyframe_polish_job,
    validate_keyframe_polish_job,
)
from aigen.models.downloads import (
    ModelDownloadError,
    download_models,
    load_download_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "aigen" / "models"


@dataclass(frozen=True)
class CharacterConceptCliProfile:
    model: str
    transformer_single_file: Path | None
    dtype: str
    steps: int
    guidance_scale: float
    true_cfg_scale: float
    width: int
    height: int
    framing: str
    cpu_offload: bool
    seed: int


@dataclass(frozen=True)
class CharacterPoseCliProfile:
    base_model: str
    controlnet_model: str
    dtype: str
    steps: int
    guidance_scale: float
    true_cfg_scale: float
    width: int
    height: int
    cpu_offload: bool
    controlnet_conditioning_scale: float
    control_guidance_start: float
    control_guidance_end: float
    control_mode: int | None
    seed: int


@dataclass(frozen=True)
class CharacterKontextPoseCliProfile:
    model: str
    controlnet_model: str
    transformer_single_file: Path | None
    dtype: str
    steps: int
    guidance_scale: float
    true_cfg_scale: float
    width: int
    height: int
    reference_max_area: int
    max_sequence_length: int
    framing: str
    pipeline_cpu_offload: bool
    vae_tiling: bool
    controlnet_conditioning_scale: float
    control_guidance_start: float
    control_guidance_end: float
    seed: int


@dataclass(frozen=True)
class NunchakuKontextCliProfile:
    base_model: str
    transformer_model: str
    dtype: str
    steps: int
    guidance_scale: float
    true_cfg_scale: float
    width: int
    height: int
    reference_max_area: int
    max_sequence_length: int
    framing: str
    seed: int


@dataclass(frozen=True)
class NunchakuKontextPoseCliProfile:
    model: str
    controlnet_model: str
    nunchaku_transformer_model: Path
    attention_impl: str
    dtype: str
    steps: int
    guidance_scale: float
    true_cfg_scale: float
    width: int
    height: int
    reference_max_area: int
    max_sequence_length: int
    framing: str
    pipeline_cpu_offload: bool
    nunchaku_layer_offload: bool
    vae_tiling: bool
    controlnet_conditioning_scale: float
    control_guidance_start: float
    control_guidance_end: float
    seed: int


@dataclass(frozen=True)
class KeyframeRefineCliProfile:
    model: str
    nunchaku_transformer_model: Path | None
    attention_impl: str | None
    dtype: str
    pipeline_cpu_offload: bool
    vae_tiling: bool


@dataclass(frozen=True)
class PixelArtCliProfile:
    backend: str
    model: str
    dtype: str
    norm: str
    n_blocks: int


PIXEL_ART_PROFILES = {
    "local": PixelArtCliProfile(
        backend="cyclegan",
        model=(MODELS_ROOT / "pixel-art/cyclegan").as_posix(),
        dtype="float32",
        norm="instance",
        n_blocks=9,
    ),
}


CHARACTER_CONCEPT_PROFILES = {
    "local": CharacterConceptCliProfile(
        model=(MODELS_ROOT / "diffusers/eramth/flux-kontext-4bit-fp4").as_posix(),
        transformer_single_file=None,
        dtype="bfloat16",
        steps=20,
        guidance_scale=2.5,
        true_cfg_scale=1.0,
        width=768,
        height=1152,
        framing="full-body",
        cpu_offload=False,
        seed=1,
    ),
    "production": CharacterConceptCliProfile(
        model=(MODELS_ROOT / "diffusers/eramth/flux-kontext-4bit-fp4").as_posix(),
        transformer_single_file=MODELS_ROOT
        / "bfl/FLUX.1-Kontext-dev-single-file/flux1-kontext-dev.safetensors",
        dtype="bfloat16",
        steps=24,
        guidance_scale=2.5,
        true_cfg_scale=1.0,
        width=768,
        height=1152,
        framing="full-body",
        cpu_offload=True,
        seed=1,
    ),
}

CHARACTER_POSE_PROFILES = {
    "flux-pose-4bit": CharacterPoseCliProfile(
        base_model=(MODELS_ROOT / "diffusers/black-forest-labs/FLUX.1-dev-bnb-4bit").as_posix(),
        controlnet_model=(
            MODELS_ROOT / "diffusers/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
        ).as_posix(),
        dtype="bfloat16",
        steps=20,
        guidance_scale=3.5,
        true_cfg_scale=1.0,
        width=512,
        height=1024,
        cpu_offload=False,
        controlnet_conditioning_scale=0.9,
        control_guidance_start=0.0,
        control_guidance_end=0.65,
        control_mode=None,
        seed=1,
    ),
    "flux-pose": CharacterPoseCliProfile(
        base_model=(MODELS_ROOT / "diffusers/black-forest-labs/FLUX.1-dev-bf16").as_posix(),
        controlnet_model=(
            MODELS_ROOT / "diffusers/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
        ).as_posix(),
        dtype="bfloat16",
        steps=30,
        guidance_scale=3.5,
        true_cfg_scale=1.0,
        width=768,
        height=1152,
        cpu_offload=True,
        controlnet_conditioning_scale=0.9,
        control_guidance_start=0.0,
        control_guidance_end=0.65,
        control_mode=None,
        seed=1,
    ),
}

CHARACTER_KONTEXT_POSE_PROFILES = {
    "local": CharacterKontextPoseCliProfile(
        model=(MODELS_ROOT / "diffusers/eramth/flux-kontext-4bit-fp4").as_posix(),
        controlnet_model=(
            MODELS_ROOT / "diffusers/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
        ).as_posix(),
        transformer_single_file=None,
        dtype="bfloat16",
        steps=28,
        guidance_scale=3.5,
        true_cfg_scale=1.0,
        width=384,
        height=576,
        reference_max_area=384 * 768,
        max_sequence_length=128,
        framing="full-body",
        pipeline_cpu_offload=False,
        vae_tiling=False,
        controlnet_conditioning_scale=0.50,
        control_guidance_start=0.0,
        control_guidance_end=0.50,
        seed=1,
    ),
    "production": CharacterKontextPoseCliProfile(
        model=(MODELS_ROOT / "diffusers/eramth/flux-kontext-4bit-fp4").as_posix(),
        controlnet_model=(
            MODELS_ROOT / "diffusers/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
        ).as_posix(),
        transformer_single_file=MODELS_ROOT
        / "bfl/FLUX.1-Kontext-dev-single-file/flux1-kontext-dev.safetensors",
        dtype="bfloat16",
        steps=30,
        guidance_scale=3.5,
        true_cfg_scale=1.0,
        width=768,
        height=1152,
        reference_max_area=512 * 1024,
        max_sequence_length=128,
        framing="full-body",
        pipeline_cpu_offload=True,
        vae_tiling=True,
        controlnet_conditioning_scale=0.65,
        control_guidance_start=0.0,
        control_guidance_end=0.65,
        seed=1,
    ),
}

NUNCHAKU_KONTEXT_PROFILES = {
    "local": NunchakuKontextCliProfile(
        base_model=(MODELS_ROOT / "diffusers/eramth/flux-kontext-4bit-fp4").as_posix(),
        transformer_model=(
            MODELS_ROOT
            / "nunchaku/nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-fp4_r32-flux.1-kontext-dev.safetensors"
        ).as_posix(),
        dtype="bfloat16",
        steps=3,
        guidance_scale=2.5,
        true_cfg_scale=1.0,
        width=384,
        height=576,
        reference_max_area=384 * 768,
        max_sequence_length=128,
        framing="full-body",
        seed=1,
    ),
}

NUNCHAKU_KONTEXT_POSE_PROFILES = {
    "benchmark": NunchakuKontextPoseCliProfile(
        model=(MODELS_ROOT / "diffusers/eramth/flux-kontext-4bit-fp4").as_posix(),
        controlnet_model=(
            MODELS_ROOT / "diffusers/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
        ).as_posix(),
        nunchaku_transformer_model=(
            MODELS_ROOT
            / "nunchaku/nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-fp4_r32-flux.1-kontext-dev.safetensors"
        ),
        attention_impl="nunchaku-fp16",
        dtype="bfloat16",
        steps=3,
        guidance_scale=3.5,
        true_cfg_scale=1.0,
        width=384,
        height=576,
        reference_max_area=384 * 768,
        max_sequence_length=128,
        framing="full-body",
        pipeline_cpu_offload=True,
        nunchaku_layer_offload=False,
        vae_tiling=False,
        controlnet_conditioning_scale=0.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        seed=1,
    ),
    "local": NunchakuKontextPoseCliProfile(
        model=(MODELS_ROOT / "diffusers/eramth/flux-kontext-4bit-fp4").as_posix(),
        controlnet_model=(
            MODELS_ROOT / "diffusers/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
        ).as_posix(),
        nunchaku_transformer_model=(
            MODELS_ROOT
            / "nunchaku/nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-fp4_r32-flux.1-kontext-dev.safetensors"
        ),
        attention_impl="nunchaku-fp16",
        dtype="bfloat16",
        steps=20,
        guidance_scale=2.5,
        true_cfg_scale=1.0,
        width=384,
        height=576,
        reference_max_area=384 * 768,
        max_sequence_length=128,
        framing="full-body",
        pipeline_cpu_offload=True,
        nunchaku_layer_offload=False,
        vae_tiling=False,
        controlnet_conditioning_scale=0.50,
        control_guidance_start=0.0,
        control_guidance_end=0.50,
        seed=1,
    ),
    "quality": NunchakuKontextPoseCliProfile(
        model=(MODELS_ROOT / "diffusers/eramth/flux-kontext-4bit-fp4").as_posix(),
        controlnet_model=(
            MODELS_ROOT / "diffusers/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
        ).as_posix(),
        nunchaku_transformer_model=(
            MODELS_ROOT
            / "nunchaku/nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-fp4_r32-flux.1-kontext-dev.safetensors"
        ),
        attention_impl="nunchaku-fp16",
        dtype="bfloat16",
        steps=28,
        guidance_scale=2.5,
        true_cfg_scale=1.0,
        width=512,
        height=768,
        reference_max_area=512 * 1024,
        max_sequence_length=128,
        framing="full-body",
        pipeline_cpu_offload=True,
        nunchaku_layer_offload=False,
        vae_tiling=False,
        controlnet_conditioning_scale=0.50,
        control_guidance_start=0.0,
        control_guidance_end=0.50,
        seed=1,
    ),
}


KEYFRAME_MODEL_REVISIONS = {
    "kontext": {
        "repo_id": "eramth/flux-kontext-4bit-fp4",
        "revision": "499964b43d54eda6ca7e21c346efe18e2a1cdad8",
    },
    "controlnet": {
        "repo_id": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        "revision": "5d700aaad96c5ddcdf8a38ef9b22a82aac2c38e5",
    },
    "nunchaku_transformer": {
        "repo_id": "nunchaku-tech/nunchaku-flux.1-kontext-dev",
        "revision": "70dff7728491f3016e256137e8f7d87812af0b4f",
    },
}


KEYFRAME_REFINE_PROFILES = {
    "kontext-inpaint-local": KeyframeRefineCliProfile(
        model=(MODELS_ROOT / "diffusers/eramth/flux-kontext-4bit-fp4").as_posix(),
        nunchaku_transformer_model=(
            MODELS_ROOT
            / "nunchaku/nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-fp4_r32-flux.1-kontext-dev.safetensors"
        ),
        attention_impl="nunchaku-fp16",
        dtype="bfloat16",
        pipeline_cpu_offload=True,
        vae_tiling=False,
    ),
}


def _dump_json(handle: TextIO, payload: object, pretty: bool) -> None:
    json.dump(
        payload,
        handle,
        ensure_ascii=False,
        indent=2 if pretty else None,
        sort_keys=True,
    )
    handle.write("\n")


def _write_json(path: Path, payload: object, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        _dump_json(handle, payload, pretty)


def _keyframe_profile(profile_name: str) -> KeyframeProfile:
    if profile_name != "nunchaku-kontext-pose-quality":
        raise KeyframeJobError(f"Unknown keyframe profile: {profile_name}")
    profile = NUNCHAKU_KONTEXT_POSE_PROFILES["quality"]
    return KeyframeProfile(
        name=profile_name,
        model=profile.model,
        controlnet_model=profile.controlnet_model,
        nunchaku_transformer_model=profile.nunchaku_transformer_model,
        attention_impl=profile.attention_impl,
        dtype=profile.dtype,
        pipeline_cpu_offload=profile.pipeline_cpu_offload,
        nunchaku_layer_offload=profile.nunchaku_layer_offload,
        vae_tiling=profile.vae_tiling,
        model_revisions=KEYFRAME_MODEL_REVISIONS,
    )


def _keyframe_profile_for_job(job_path: Path) -> KeyframeProfile:
    return _keyframe_profile(load_keyframe_job(job_path).pipeline.profile)


def _keyframe_refine_profile(profile_name: str) -> KeyframeRefineProfile:
    if profile_name not in KEYFRAME_REFINE_PROFILES:
        raise KeyframeRefineError(f"Unknown keyframe refine profile: {profile_name}")
    profile = KEYFRAME_REFINE_PROFILES[profile_name]
    return KeyframeRefineProfile(
        name=profile_name,
        model=profile.model,
        nunchaku_transformer_model=profile.nunchaku_transformer_model,
        attention_impl=profile.attention_impl,
        dtype=profile.dtype,
        pipeline_cpu_offload=profile.pipeline_cpu_offload,
        vae_tiling=profile.vae_tiling,
        model_revisions={
            "kontext": KEYFRAME_MODEL_REVISIONS["kontext"],
            "nunchaku_transformer": KEYFRAME_MODEL_REVISIONS["nunchaku_transformer"],
        },
    )


def _keyframe_refine_profile_for_job(job_path: Path) -> KeyframeRefineProfile:
    return _keyframe_refine_profile(load_keyframe_refine_job(job_path).pipeline.profile)


def _keyframe_polish_profile_for_job(job_path: Path) -> KeyframeRefineProfile:
    return _keyframe_refine_profile(load_keyframe_polish_job(job_path).pipeline.profile)


def _add_keyframe_commands(subparsers: Any) -> None:
    keyframes = subparsers.add_parser("keyframes", help="JSON-first character keyframe jobs")
    keyframe_subparsers = keyframes.add_subparsers(dest="keyframes_command", required=True)

    init = keyframe_subparsers.add_parser("init", help="Write a keyframe job template to stdout")
    init.add_argument(
        "--template",
        choices=("c2-profile",),
        required=True,
        help="Template keyframe job to emit",
    )
    init.add_argument("--compact", action="store_true", help="Write compact JSON")

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

    validate = keyframe_subparsers.add_parser("validate", help="Validate a keyframe job without running the GPU")
    validate.add_argument("job", type=Path, help="Keyframe job JSON")
    validate.add_argument("--compact", action="store_true", help="Write compact JSON")

    plan = keyframe_subparsers.add_parser("plan", help="Resolve a keyframe job without running the GPU")
    plan.add_argument("job", type=Path, help="Keyframe job JSON")
    plan.add_argument("--compact", action="store_true", help="Write compact JSON")

    run = keyframe_subparsers.add_parser("run", help="Run a resolved keyframe job")
    run.add_argument("job", type=Path, help="Keyframe job JSON")
    run.add_argument("--compact", action="store_true", help="Write compact JSON")

    extract_example = keyframe_subparsers.add_parser(
        "extract-example",
        help="Extract pose, contour and boundary assets from a source keyframe example",
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

    refine_run_variant = keyframe_subparsers.add_parser("refine-run-variant", help=argparse.SUPPRESS)
    refine_run_variant.add_argument("job", type=Path, help=argparse.SUPPRESS)
    refine_run_variant.add_argument("--variant", required=True, help=argparse.SUPPRESS)
    refine_run_variant.add_argument("--compact", action="store_true", help=argparse.SUPPRESS)

    polish_diagnose = keyframe_subparsers.add_parser(
        "polish-diagnose",
        help="Use the local VLM to decide which polish-job regions need local inpaint",
    )
    polish_diagnose.add_argument("job", type=Path, help="Keyframe polish job JSON")
    polish_diagnose.add_argument("--judge", default=DEFAULT_JUDGE_ID, help="Judge id recorded in diagnosis")
    polish_diagnose.add_argument(
        "--model",
        type=Path,
        default=MODELS_ROOT / "vlm/Qwen/Qwen2.5-VL-7B-Instruct",
        help="Local Qwen2.5-VL-7B-Instruct model directory",
    )
    polish_diagnose.add_argument("--dtype", default="bfloat16", help="Torch dtype for judge model weights")
    polish_diagnose.add_argument("--attention-impl", default="sdpa", help="Transformers attention implementation")
    polish_diagnose.add_argument(
        "--quantization",
        choices=("bitsandbytes-8bit", "bitsandbytes-4bit", "none"),
        default=DEFAULT_JUDGE_QUANTIZATION,
        help="Local inference quantization for polish diagnosis",
    )
    polish_diagnose.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS, help="Minimum Qwen visual pixels")
    polish_diagnose.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Maximum Qwen visual pixels")
    polish_diagnose.add_argument("--max-new-tokens", type=int, default=700, help="Diagnosis response token budget")
    polish_diagnose.add_argument("--temperature", type=float, default=0.0, help="Diagnosis sampling temperature")
    polish_diagnose.add_argument("--compact", action="store_true", help="Write compact JSON")

    polish_validate = keyframe_subparsers.add_parser(
        "polish-validate",
        help="Validate a keyframe polish job without running the GPU",
    )
    polish_validate.add_argument("job", type=Path, help="Keyframe polish job JSON")
    polish_validate.add_argument("--compact", action="store_true", help="Write compact JSON")

    polish_plan = keyframe_subparsers.add_parser(
        "polish-plan",
        help="Resolve a keyframe polish job without running the GPU",
    )
    polish_plan.add_argument("job", type=Path, help="Keyframe polish job JSON")
    polish_plan.add_argument("--compact", action="store_true", help="Write compact JSON")

    polish_run = keyframe_subparsers.add_parser(
        "polish-run",
        help="Run selected-candidate local detail polish",
    )
    polish_run.add_argument("job", type=Path, help="Keyframe polish job JSON")
    polish_run.add_argument("--compact", action="store_true", help="Write compact JSON")

    judge = keyframe_subparsers.add_parser("judge", help="Judge a completed keyframe run with a local VLM")
    judge.add_argument("run_dir", type=Path, help="Completed keyframe run directory")
    judge.add_argument("--judge", default=DEFAULT_JUDGE_ID, help="Judge id used for output directory naming")
    judge.add_argument(
        "--model",
        type=Path,
        default=MODELS_ROOT / "vlm/Qwen/Qwen2.5-VL-7B-Instruct",
        help="Local Qwen2.5-VL-7B-Instruct model directory",
    )
    judge.add_argument("--dtype", default="bfloat16", help="Torch dtype for judge model weights")
    judge.add_argument(
        "--attention-impl",
        default="sdpa",
        help="Transformers attention implementation for the judge model",
    )
    judge.add_argument(
        "--quantization",
        choices=("bitsandbytes-8bit", "bitsandbytes-4bit", "none"),
        default=DEFAULT_JUDGE_QUANTIZATION,
        help="Local inference quantization for the 7B judge; 8-bit is the calibration default",
    )
    judge.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS, help="Minimum Qwen visual pixels")
    judge.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Maximum Qwen visual pixels")
    judge.add_argument("--max-new-tokens", type=int, default=900, help="Judge response token budget")
    judge.add_argument("--temperature", type=float, default=0.0, help="Judge sampling temperature")
    judge.add_argument("--pairwise-top-k", type=int, default=3, help="Final pairwise ranking candidate count")
    judge.add_argument("--compact", action="store_true", help="Write compact JSON")

    calibrate = keyframe_subparsers.add_parser(
        "judge-calibrate",
        help="Calibrate a completed judge result against a golden fixture",
    )
    calibrate.add_argument("run_dir", type=Path, help="Completed keyframe run directory")
    calibrate.add_argument("--from-judge", default=DEFAULT_JUDGE_ID, help="Judge id to read")
    calibrate.add_argument(
        "--fixture",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_CALIBRATION_FIXTURE,
        help="Judge calibration fixture JSON",
    )
    calibrate.add_argument("--compact", action="store_true", help="Write compact JSON")

    score = keyframe_subparsers.add_parser("score", help="Score a completed keyframe run against its conditions")
    score.add_argument("run_dir", type=Path, help="Completed keyframe run directory")
    score.add_argument("--scorer", default=DEFAULT_SCORER_ID, help="Scorer id used for output directory naming")
    score.add_argument(
        "--foreground-threshold",
        type=float,
        default=28.0,
        help="RGB distance from border-estimated background for foreground extraction",
    )
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
    score_select.add_argument("--compact", action="store_true", help="Write compact JSON")

    select = keyframe_subparsers.add_parser("select", help="Select/reject outputs from a keyframe judge result")
    select.add_argument("run_dir", type=Path, help="Completed keyframe run directory")
    select.add_argument("--from-judge", default=DEFAULT_JUDGE_ID, help="Judge id to read")
    select.add_argument("--top", type=int, default=1, help="Number of top-ranked candidates to select")
    select.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="Allow selection even when judge calibration is missing or failed",
    )
    select.add_argument("--compact", action="store_true", help="Write compact JSON")


def _add_model_commands(subparsers: Any) -> None:
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


def _add_character_commands(subparsers: Any) -> None:
    characters = subparsers.add_parser("characters", help="Versioned character view bank tools")
    character_subparsers = characters.add_subparsers(dest="characters_command", required=True)

    view_init = character_subparsers.add_parser("view-init", help="Write a character-view job template to stdout")
    view_init.add_argument("--template", choices=("ai46-left-profile",), required=True)
    view_init.add_argument("--compact", action="store_true", help="Write compact JSON")

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


def _add_generate_commands(subparsers: Any) -> None:
    generate = subparsers.add_parser("generate", help="Run private generation pipelines")
    generate_subparsers = generate.add_subparsers(dest="generate_command", required=True)

    _add_character_concept_command(generate_subparsers)
    _add_character_kontext_pose_command(generate_subparsers)
    _add_character_nunchaku_kontext_command(generate_subparsers)
    _add_character_nunchaku_kontext_pose_command(generate_subparsers)
    _add_character_pose_command(generate_subparsers)
    _add_pixel_art_command(generate_subparsers)


def _add_character_concept_command(generate_subparsers: Any) -> None:
    concept = generate_subparsers.add_parser(
        "character-concept",
        help="Generate a character concept image from a reference image and prompt",
    )
    concept.add_argument(
        "--profile",
        choices=tuple(CHARACTER_CONCEPT_PROFILES),
        default="local",
        help="Generation profile: local is fast 4-bit iteration, production uses the BF16 FLUX transformer",
    )
    concept.add_argument(
        "--model",
        default=argparse.SUPPRESS,
        help="Local FLUX Kontext model path or cached repo id",
    )
    concept.add_argument(
        "--transformer-single-file",
        type=Path,
        default=argparse.SUPPRESS,
        help="Optional Flux transformer safetensors file to use instead of the model folder transformer",
    )
    concept.add_argument("--reference-image", type=Path, required=True, help="Reference character image")
    concept.add_argument("--prompt", required=True, help="Generation instruction")
    concept.add_argument("--output", type=Path, required=True, help="Image path to write")
    concept.add_argument("--device", default="cuda", help="Torch device")
    concept.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default=argparse.SUPPRESS,
        help="Torch dtype for pipeline weights",
    )
    concept.add_argument("--steps", type=int, default=argparse.SUPPRESS, help="Denoising steps")
    concept.add_argument(
        "--guidance-scale",
        type=float,
        default=argparse.SUPPRESS,
        help="Prompt guidance scale",
    )
    concept.add_argument(
        "--true-cfg-scale",
        type=float,
        default=argparse.SUPPRESS,
        help="Kontext true CFG scale used with the negative prompt",
    )
    concept.add_argument("--width", type=int, default=argparse.SUPPRESS, help="Generated image width")
    concept.add_argument("--height", type=int, default=argparse.SUPPRESS, help="Generated image height")
    concept.add_argument(
        "--framing",
        choices=("full-body", "portrait"),
        default=argparse.SUPPRESS,
        help="Character composition contract",
    )
    concept.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Prompt content to avoid; defaults to the built-in character-art negative prompt",
    )
    concept.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Deterministic seed")
    offload = concept.add_mutually_exclusive_group()
    offload.add_argument(
        "--cpu-offload",
        dest="cpu_offload",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use Diffusers model CPU offload",
    )
    offload.add_argument(
        "--no-cpu-offload",
        dest="cpu_offload",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Keep the pipeline on the selected device instead of using CPU offload",
    )
    concept.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of pretty printed JSON",
    )


def _add_pixel_art_command(generate_subparsers: Any) -> None:
    pixel = generate_subparsers.add_parser(
        "pixel-art",
        help="Pixelize an input image into native low-res pixel art (img2img GAN, no Stable Diffusion / FLUX)",
    )
    pixel.add_argument(
        "--profile",
        choices=tuple(PIXEL_ART_PROFILES),
        default="local",
        help="Generation profile; local targets the RTX 5070 Ti pure-PyTorch CycleGAN backend",
    )
    pixel.add_argument(
        "--backend",
        default=argparse.SUPPRESS,
        help="Pixel-art backend id (e.g. cyclegan)",
    )
    pixel.add_argument(
        "--model",
        default=argparse.SUPPRESS,
        help="CycleGAN generator checkpoint, or a directory containing latest_net_G.pth",
    )
    pixel.add_argument(
        "--input-image",
        type=Path,
        required=True,
        help="Source image to pixelize; feed a low-res image to get genuine low-res output",
    )
    pixel.add_argument("--output", type=Path, required=True, help="Image path to write")
    pixel.add_argument("--device", default="cuda", help="Torch device")
    pixel.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default=argparse.SUPPRESS,
        help="Torch dtype for the generator weights",
    )
    pixel.add_argument(
        "--norm",
        choices=("instance", "batch"),
        default=argparse.SUPPRESS,
        help="Normalization layer the checkpoint was trained with (CycleGAN default: instance)",
    )
    pixel.add_argument(
        "--n-blocks",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of ResNet blocks in the generator (CycleGAN default: 9)",
    )
    pixel.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of pretty printed JSON",
    )


def _add_character_pose_command(generate_subparsers: Any) -> None:
    pose = generate_subparsers.add_parser(
        "character-pose",
        help="Generate a character image with FLUX ControlNet pose conditioning",
    )
    pose.add_argument(
        "--profile",
        choices=tuple(CHARACTER_POSE_PROFILES),
        default="flux-pose",
        help="Pose-control generation profile",
    )
    pose.add_argument(
        "--base-model",
        default=argparse.SUPPRESS,
        help="Local FLUX.1-dev Diffusers model path",
    )
    pose.add_argument(
        "--controlnet-model",
        default=argparse.SUPPRESS,
        help="Local FLUX pose ControlNet model path",
    )
    pose.add_argument(
        "--pose-image",
        type=Path,
        required=True,
        help="DWPose/OpenPose-style control image",
    )
    pose.add_argument("--prompt", required=True, help="Generation instruction")
    pose.add_argument("--output", type=Path, required=True, help="Image path to write")
    pose.add_argument("--device", default="cuda", help="Torch device")
    pose.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default=argparse.SUPPRESS,
        help="Torch dtype for pipeline weights",
    )
    pose.add_argument("--steps", type=int, default=argparse.SUPPRESS, help="Denoising steps")
    pose.add_argument(
        "--guidance-scale",
        type=float,
        default=argparse.SUPPRESS,
        help="Prompt guidance scale",
    )
    pose.add_argument(
        "--true-cfg-scale",
        type=float,
        default=argparse.SUPPRESS,
        help="FLUX true CFG scale used with the negative prompt",
    )
    pose.add_argument("--width", type=int, default=argparse.SUPPRESS, help="Generated image width")
    pose.add_argument("--height", type=int, default=argparse.SUPPRESS, help="Generated image height")
    pose.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Prompt content to avoid; defaults to the built-in character-art negative prompt",
    )
    pose.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Deterministic seed")
    pose.add_argument(
        "--controlnet-conditioning-scale",
        type=float,
        default=argparse.SUPPRESS,
        help="Pose ControlNet strength",
    )
    pose.add_argument(
        "--control-guidance-start",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of denoising where ControlNet starts applying",
    )
    pose.add_argument(
        "--control-guidance-end",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of denoising where ControlNet stops applying",
    )
    pose.add_argument(
        "--control-mode",
        type=int,
        default=argparse.SUPPRESS,
        help="Optional ControlNet-Union mode id for models that use mode embeddings",
    )
    offload = pose.add_mutually_exclusive_group()
    offload.add_argument(
        "--cpu-offload",
        dest="cpu_offload",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use Diffusers model CPU offload",
    )
    offload.add_argument(
        "--no-cpu-offload",
        dest="cpu_offload",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Keep the pipeline on the selected device instead of using CPU offload",
    )
    pose.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of pretty printed JSON",
    )


def _add_character_kontext_pose_command(generate_subparsers: Any) -> None:
    pose = generate_subparsers.add_parser(
        "character-kontext-pose",
        help="Generate a same-character image with Kontext identity and pose ControlNet",
    )
    pose.add_argument(
        "--profile",
        choices=tuple(CHARACTER_KONTEXT_POSE_PROFILES),
        default="local",
        help="Kontext pose-control generation profile",
    )
    pose.add_argument(
        "--model",
        default=argparse.SUPPRESS,
        help="Local FLUX Kontext model path",
    )
    pose.add_argument(
        "--controlnet-model",
        default=argparse.SUPPRESS,
        help="Local FLUX pose ControlNet model path",
    )
    pose.add_argument(
        "--transformer-single-file",
        type=Path,
        default=argparse.SUPPRESS,
        help="Optional Flux Kontext transformer safetensors file",
    )
    pose.add_argument(
        "--pose-image",
        type=Path,
        required=True,
        help="DWPose/OpenPose-style control image",
    )
    pose.add_argument(
        "--reference-image",
        type=Path,
        required=True,
        help="Reference character image",
    )
    pose.add_argument("--prompt", required=True, help="Generation instruction")
    pose.add_argument("--output", type=Path, required=True, help="Image path to write")
    pose.add_argument("--device", default="cuda", help="Torch device")
    pose.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default=argparse.SUPPRESS,
        help="Torch dtype for pipeline weights",
    )
    pose.add_argument("--steps", type=int, default=argparse.SUPPRESS, help="Denoising steps")
    pose.add_argument(
        "--guidance-scale",
        type=float,
        default=argparse.SUPPRESS,
        help="Prompt guidance scale",
    )
    pose.add_argument(
        "--true-cfg-scale",
        type=float,
        default=argparse.SUPPRESS,
        help="FLUX true CFG scale used with the negative prompt",
    )
    pose.add_argument("--width", type=int, default=argparse.SUPPRESS, help="Generated image width")
    pose.add_argument("--height", type=int, default=argparse.SUPPRESS, help="Generated image height")
    pose.add_argument(
        "--reference-max-area",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum pixel area for the Kontext reference image before VAE encoding",
    )
    pose.add_argument(
        "--max-sequence-length",
        type=int,
        default=argparse.SUPPRESS,
        help="T5 text token budget for the prompt",
    )
    pose.add_argument(
        "--framing",
        choices=("full-body", "portrait"),
        default=argparse.SUPPRESS,
        help="Character composition contract",
    )
    pose.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Prompt content to avoid; defaults to the built-in character-art negative prompt",
    )
    pose.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Deterministic seed")
    pose.add_argument(
        "--controlnet-conditioning-scale",
        type=float,
        default=argparse.SUPPRESS,
        help="Pose ControlNet strength",
    )
    pose.add_argument(
        "--control-guidance-start",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of denoising where ControlNet starts applying",
    )
    pose.add_argument(
        "--control-guidance-end",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of denoising where ControlNet stops applying",
    )
    offload = pose.add_mutually_exclusive_group()
    offload.add_argument(
        "--cpu-offload",
        dest="pipeline_cpu_offload",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable Diffusers pipeline CPU offload",
    )
    offload.add_argument(
        "--no-cpu-offload",
        dest="pipeline_cpu_offload",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Keep Diffusers pipeline components on the target device",
    )
    tiling = pose.add_mutually_exclusive_group()
    tiling.add_argument(
        "--vae-tiling",
        dest="vae_tiling",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable VAE tiling to reduce peak memory during encode/decode",
    )
    tiling.add_argument(
        "--no-vae-tiling",
        dest="vae_tiling",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable VAE tiling for smaller preview runs",
    )
    pose.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of pretty printed JSON",
    )


def _add_character_nunchaku_kontext_command(generate_subparsers: Any) -> None:
    concept = generate_subparsers.add_parser(
        "character-nunchaku-kontext",
        help="Benchmark same-character Kontext generation with a Nunchaku transformer",
    )
    concept.add_argument(
        "--profile",
        choices=tuple(NUNCHAKU_KONTEXT_PROFILES),
        default="local",
        help="Nunchaku Kontext generation profile",
    )
    concept.add_argument(
        "--base-model",
        default=argparse.SUPPRESS,
        help="Local Diffusers FLUX Kontext component folder",
    )
    concept.add_argument(
        "--transformer-model",
        default=argparse.SUPPRESS,
        help="Local Nunchaku FLUX Kontext transformer safetensors file",
    )
    concept.add_argument("--reference-image", type=Path, required=True, help="Reference character image")
    concept.add_argument("--prompt", required=True, help="Generation instruction")
    concept.add_argument("--output", type=Path, required=True, help="Image path to write")
    concept.add_argument("--device", default="cuda", help="Torch device")
    concept.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default=argparse.SUPPRESS,
        help="Torch dtype for non-transformer pipeline components",
    )
    concept.add_argument("--steps", type=int, default=argparse.SUPPRESS, help="Denoising steps")
    concept.add_argument(
        "--guidance-scale",
        type=float,
        default=argparse.SUPPRESS,
        help="Prompt guidance scale",
    )
    concept.add_argument(
        "--true-cfg-scale",
        type=float,
        default=argparse.SUPPRESS,
        help="Kontext true CFG scale used with the negative prompt",
    )
    concept.add_argument("--width", type=int, default=argparse.SUPPRESS, help="Generated image width")
    concept.add_argument("--height", type=int, default=argparse.SUPPRESS, help="Generated image height")
    concept.add_argument(
        "--reference-max-area",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum pixel area for the Kontext reference image before VAE encoding",
    )
    concept.add_argument(
        "--max-sequence-length",
        type=int,
        default=argparse.SUPPRESS,
        help="T5 text token budget for the prompt",
    )
    concept.add_argument(
        "--framing",
        choices=("full-body", "portrait"),
        default=argparse.SUPPRESS,
        help="Character composition contract",
    )
    concept.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Prompt content to avoid; defaults to the built-in character-art negative prompt",
    )
    concept.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Deterministic seed")
    concept.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of pretty printed JSON",
    )


def _add_character_nunchaku_kontext_pose_command(generate_subparsers: Any) -> None:
    pose = generate_subparsers.add_parser(
        "character-nunchaku-kontext-pose",
        help="Run fused Kontext identity plus pose ControlNet with a Nunchaku transformer",
    )
    pose.add_argument(
        "--profile",
        choices=tuple(NUNCHAKU_KONTEXT_POSE_PROFILES),
        default="local",
        help="Nunchaku Kontext pose generation profile",
    )
    pose.add_argument("--model", default=argparse.SUPPRESS, help="Local Diffusers FLUX Kontext component folder")
    pose.add_argument("--controlnet-model", default=argparse.SUPPRESS, help="Local FLUX pose ControlNet folder")
    pose.add_argument(
        "--nunchaku-transformer-model",
        type=Path,
        default=argparse.SUPPRESS,
        help="Local Nunchaku FLUX Kontext transformer safetensors file",
    )
    pose.add_argument(
        "--attention-impl",
        default=argparse.SUPPRESS,
        help="Nunchaku attention implementation, for example nunchaku-fp16",
    )
    pose.add_argument("--reference-image", type=Path, required=True, help="Reference character image")
    pose.add_argument("--pose-image", type=Path, required=True, help="Pose control image")
    pose.add_argument("--prompt", required=True, help="Generation instruction")
    pose.add_argument("--output", type=Path, required=True, help="Image path to write")
    pose.add_argument("--device", default="cuda", help="Torch device")
    pose.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default=argparse.SUPPRESS,
        help="Torch dtype for non-transformer pipeline components",
    )
    pose.add_argument("--steps", type=int, default=argparse.SUPPRESS, help="Denoising steps")
    pose.add_argument("--guidance-scale", type=float, default=argparse.SUPPRESS, help="Prompt guidance scale")
    pose.add_argument(
        "--true-cfg-scale",
        type=float,
        default=argparse.SUPPRESS,
        help="Kontext true CFG scale used with the negative prompt",
    )
    pose.add_argument("--width", type=int, default=argparse.SUPPRESS, help="Generated image width")
    pose.add_argument("--height", type=int, default=argparse.SUPPRESS, help="Generated image height")
    pose.add_argument(
        "--reference-max-area",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum pixel area for the Kontext reference image before VAE encoding",
    )
    pose.add_argument(
        "--max-sequence-length",
        type=int,
        default=argparse.SUPPRESS,
        help="T5 text token budget for the prompt",
    )
    pose.add_argument(
        "--framing",
        choices=("full-body", "portrait"),
        default=argparse.SUPPRESS,
        help="Character composition contract",
    )
    pose.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Prompt content to avoid; defaults to the built-in character-art negative prompt",
    )
    pose.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Deterministic seed")
    pose.add_argument(
        "--controlnet-conditioning-scale",
        type=float,
        default=argparse.SUPPRESS,
        help="Pose ControlNet conditioning scale",
    )
    pose.add_argument(
        "--control-guidance-start",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of the denoising schedule where ControlNet starts",
    )
    pose.add_argument(
        "--control-guidance-end",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of the denoising schedule where ControlNet stops",
    )
    tiling = pose.add_mutually_exclusive_group()
    tiling.add_argument(
        "--vae-tiling",
        dest="vae_tiling",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable VAE tiling to reduce peak memory during encode/decode",
    )
    tiling.add_argument(
        "--no-vae-tiling",
        dest="vae_tiling",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable VAE tiling for smaller preview runs",
    )
    offload = pose.add_mutually_exclusive_group()
    offload.add_argument(
        "--cpu-offload",
        dest="pipeline_cpu_offload",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable Diffusers pipeline CPU offload",
    )
    offload.add_argument(
        "--no-cpu-offload",
        dest="pipeline_cpu_offload",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Keep Diffusers pipeline components on the target device",
    )
    nunchaku_offload = pose.add_mutually_exclusive_group()
    nunchaku_offload.add_argument(
        "--nunchaku-layer-offload",
        dest="nunchaku_layer_offload",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable Nunchaku internal transformer layer offload",
    )
    nunchaku_offload.add_argument(
        "--no-nunchaku-layer-offload",
        dest="nunchaku_layer_offload",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Keep Nunchaku transformer layers resident",
    )
    pose.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of pretty printed JSON",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aigen",
        description="AI character generation pipeline tooling",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_generate_commands(subparsers)
    _add_character_commands(subparsers)
    _add_keyframe_commands(subparsers)
    _add_model_commands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "models" and args.models_command == "download":
        try:
            result = download_models(
                load_download_manifest(args.manifest),
                args.models_root,
                dry_run=args.dry_run,
            )
        except ModelDownloadError as error:
            _dump_json(sys.stderr, error.to_json(), pretty=not args.compact)
            return 1
        payload = result.to_json()
        if args.output:
            _write_json(args.output, payload, pretty=not args.compact)
        else:
            _dump_json(sys.stdout, payload, pretty=not args.compact)
        return 0

    if args.command == "characters":
        try:
            if args.characters_command == "view-init":
                _dump_json(sys.stdout, left_profile_view_template(), pretty=not args.compact)
                return 0
            if args.characters_command == "view-schema":
                _dump_json(sys.stdout, character_view_job_schema(), pretty=not args.compact)
                return 0
            if args.characters_command == "view-bank-schema":
                _dump_json(sys.stdout, character_view_bank_schema(), pretty=not args.compact)
                return 0
            if args.characters_command == "view-validate":
                _dump_json(
                    sys.stdout,
                    validate_character_view_job(args.job, project_root=PROJECT_ROOT),
                    pretty=not args.compact,
                )
                return 0
            if args.characters_command == "view-plan":
                _dump_json(
                    sys.stdout,
                    plan_character_view_job(args.job, project_root=PROJECT_ROOT),
                    pretty=not args.compact,
                )
                return 0
            if args.characters_command == "view-run":
                _dump_json(
                    sys.stdout,
                    run_character_view_job(
                        args.job,
                        _keyframe_profile(load_character_view_job(args.job).pipeline.profile),
                        project_root=PROJECT_ROOT,
                    ),
                    pretty=not args.compact,
                )
                return 0
            if args.characters_command == "view-accept":
                _dump_json(
                    sys.stdout,
                    accept_character_view(
                        args.job,
                        run_dir=args.run_dir,
                        candidate=args.candidate,
                        project_root=PROJECT_ROOT,
                    ),
                    pretty=not args.compact,
                )
                return 0
        except CharacterViewError as error:
            _dump_json(
                sys.stderr,
                {
                    "schema_version": 1,
                    "status": "error",
                    "error": error.__class__.__name__,
                    "message": str(error),
                },
                pretty=not args.compact,
            )
            return 1

    if args.command == "keyframes":
        try:
            if args.keyframes_command == "init":
                _dump_json(sys.stdout, c2_profile_template(), pretty=not args.compact)
                return 0
            if args.keyframes_command == "schema":
                _dump_json(sys.stdout, keyframe_job_schema(), pretty=not args.compact)
                return 0
            if args.keyframes_command == "refine-schema":
                _dump_json(sys.stdout, keyframe_refine_job_schema(), pretty=not args.compact)
                return 0
            if args.keyframes_command == "polish-schema":
                _dump_json(sys.stdout, keyframe_polish_job_schema(), pretty=not args.compact)
                return 0
            if args.keyframes_command == "extract-example":
                _dump_json(
                    sys.stdout,
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
                _dump_json(
                    sys.stdout,
                    judge_keyframe_run(
                        args.run_dir,
                        KeyframeJudgeConfig(
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
                            pairwise_top_k=args.pairwise_top_k,
                        ),
                        project_root=PROJECT_ROOT,
                    ),
                    pretty=not args.compact,
                )
                return 0
            if args.keyframes_command == "judge-calibrate":
                _dump_json(
                    sys.stdout,
                    calibrate_keyframe_judge(
                        args.run_dir,
                        judge_id=args.from_judge,
                        fixture_path=args.fixture,
                    ),
                    pretty=not args.compact,
                )
                return 0
            if args.keyframes_command == "score":
                _dump_json(
                    sys.stdout,
                    score_keyframe_run(
                        args.run_dir,
                        KeyframeScoreConfig(
                            scorer_id=args.scorer,
                            foreground_threshold=args.foreground_threshold,
                            contour_radius=args.contour_radius,
                            distance_scale_px=args.distance_scale_px,
                        ),
                        project_root=PROJECT_ROOT,
                    ),
                    pretty=not args.compact,
                )
                return 0
            if args.keyframes_command == "score-select":
                _dump_json(
                    sys.stdout,
                    select_scored_keyframe_run(
                        args.run_dir,
                        scorer_id=args.from_scorer,
                    ),
                    pretty=not args.compact,
                )
                return 0
            if args.keyframes_command == "select":
                _dump_json(
                    sys.stdout,
                    select_keyframe_run(
                        args.run_dir,
                        judge_id=args.from_judge,
                        top=args.top,
                        allow_uncalibrated=args.allow_uncalibrated,
                    ),
                    pretty=not args.compact,
                )
                return 0
            if args.keyframes_command == "polish-diagnose":
                _dump_json(
                    sys.stdout,
                    diagnose_keyframe_polish(
                        args.job,
                        config=KeyframeJudgeConfig(
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
                            pairwise_top_k=0,
                        ),
                        project_root=PROJECT_ROOT,
                    ),
                    pretty=not args.compact,
                )
                return 0
            if args.keyframes_command in {"polish-validate", "polish-plan", "polish-run"}:
                profile = _keyframe_polish_profile_for_job(args.job)
                if args.keyframes_command == "polish-validate":
                    _dump_json(
                        sys.stdout,
                        validate_keyframe_polish_job(args.job, profile, project_root=PROJECT_ROOT),
                        pretty=not args.compact,
                    )
                    return 0
                if args.keyframes_command == "polish-plan":
                    _dump_json(
                        sys.stdout,
                        plan_keyframe_polish_job(args.job, profile, project_root=PROJECT_ROOT),
                        pretty=not args.compact,
                    )
                    return 0
                _dump_json(
                    sys.stdout,
                    run_keyframe_polish_job(args.job, profile, project_root=PROJECT_ROOT),
                    pretty=not args.compact,
                )
                return 0
            if args.keyframes_command in {"refine-validate", "refine-plan", "refine-run", "refine-run-variant"}:
                profile = _keyframe_refine_profile_for_job(args.job)
                if args.keyframes_command == "refine-validate":
                    _dump_json(
                        sys.stdout,
                        validate_keyframe_refine_job(args.job, profile, project_root=PROJECT_ROOT),
                        pretty=not args.compact,
                    )
                    return 0
                if args.keyframes_command == "refine-plan":
                    _dump_json(
                        sys.stdout,
                        plan_keyframe_refine_job(args.job, profile, project_root=PROJECT_ROOT),
                        pretty=not args.compact,
                    )
                    return 0
                if args.keyframes_command == "refine-run-variant":
                    _dump_json(
                        sys.stdout,
                        run_keyframe_refine_variant(
                            args.job,
                            profile,
                            variant_name=args.variant,
                            project_root=PROJECT_ROOT,
                        ),
                        pretty=not args.compact,
                    )
                    return 0
                _dump_json(
                    sys.stdout,
                    run_keyframe_refine_job(args.job, profile, project_root=PROJECT_ROOT),
                    pretty=not args.compact,
                )
                return 0
            profile = _keyframe_profile_for_job(args.job)
            if args.keyframes_command == "validate":
                _dump_json(
                    sys.stdout,
                    validate_keyframe_job(args.job, profile, project_root=PROJECT_ROOT),
                    pretty=not args.compact,
                )
                return 0
            if args.keyframes_command == "plan":
                _dump_json(
                    sys.stdout,
                    plan_keyframe_job(args.job, profile, project_root=PROJECT_ROOT),
                    pretty=not args.compact,
                )
                return 0
            if args.keyframes_command == "run":
                _dump_json(
                    sys.stdout,
                    run_keyframe_job(args.job, profile, project_root=PROJECT_ROOT),
                    pretty=not args.compact,
                )
                return 0
        except (
            KeyframeExampleError,
            KeyframeJobError,
            KeyframeJudgeError,
            KeyframePoseError,
            KeyframeScoreError,
            KeyframeRefineError,
            KeyframePolishError,
        ) as error:
            _dump_json(
                sys.stderr,
                {
                    "schema_version": 1,
                    "status": "error",
                    "error": error.__class__.__name__,
                    "message": str(error),
                },
                pretty=not args.compact,
            )
            return 1

    if args.command == "generate" and args.generate_command == "character-concept":
        values = vars(args)
        profile = CHARACTER_CONCEPT_PROFILES[args.profile]
        try:
            result = run_character_concept(
                values.get("model", profile.model),
                args.reference_image,
                args.output,
                args.prompt,
                device=args.device,
                dtype=values.get("dtype", profile.dtype),
                steps=values.get("steps", profile.steps),
                guidance_scale=values.get("guidance_scale", profile.guidance_scale),
                true_cfg_scale=values.get("true_cfg_scale", profile.true_cfg_scale),
                width=values.get("width", profile.width),
                height=values.get("height", profile.height),
                framing=values.get("framing", profile.framing),
                negative_prompt=args.negative_prompt,
                seed=values.get("seed", profile.seed),
                cpu_offload=values.get("cpu_offload", profile.cpu_offload),
                transformer_single_file=values.get("transformer_single_file", profile.transformer_single_file),
            )
        except CharacterConceptError as error:
            _dump_json(
                sys.stderr,
                {
                    "schema_version": 1,
                    "status": "error",
                    "error": error.__class__.__name__,
                    "message": str(error),
                },
                pretty=not args.compact,
            )
            return 1
        _dump_json(sys.stdout, result.to_json(), pretty=not args.compact)
        return 0

    if args.command == "generate" and args.generate_command == "pixel-art":
        values = vars(args)
        profile = PIXEL_ART_PROFILES[args.profile]
        try:
            result = run_pixel_art(
                values.get("model", profile.model),
                args.input_image,
                args.output,
                backend=values.get("backend", profile.backend),
                device=args.device,
                dtype=values.get("dtype", profile.dtype),
                norm=values.get("norm", profile.norm),
                n_blocks=values.get("n_blocks", profile.n_blocks),
            )
        except PixelArtError as error:
            _dump_json(
                sys.stderr,
                {
                    "schema_version": 1,
                    "status": "error",
                    "error": error.__class__.__name__,
                    "message": str(error),
                },
                pretty=not args.compact,
            )
            return 1
        _dump_json(sys.stdout, result.to_json(), pretty=not args.compact)
        return 0

    if args.command == "generate" and args.generate_command == "character-pose":
        values = vars(args)
        profile = CHARACTER_POSE_PROFILES[args.profile]
        try:
            result = run_character_pose_control(
                values.get("base_model", profile.base_model),
                values.get("controlnet_model", profile.controlnet_model),
                args.pose_image,
                args.output,
                args.prompt,
                device=args.device,
                dtype=values.get("dtype", profile.dtype),
                steps=values.get("steps", profile.steps),
                guidance_scale=values.get("guidance_scale", profile.guidance_scale),
                true_cfg_scale=values.get("true_cfg_scale", profile.true_cfg_scale),
                width=values.get("width", profile.width),
                height=values.get("height", profile.height),
                negative_prompt=args.negative_prompt,
                seed=values.get("seed", profile.seed),
                cpu_offload=values.get("cpu_offload", profile.cpu_offload),
                controlnet_conditioning_scale=values.get(
                    "controlnet_conditioning_scale",
                    profile.controlnet_conditioning_scale,
                ),
                control_guidance_start=values.get("control_guidance_start", profile.control_guidance_start),
                control_guidance_end=values.get("control_guidance_end", profile.control_guidance_end),
                control_mode=values.get("control_mode", profile.control_mode),
            )
        except CharacterPoseError as error:
            _dump_json(
                sys.stderr,
                {
                    "schema_version": 1,
                    "status": "error",
                    "error": error.__class__.__name__,
                    "message": str(error),
                },
                pretty=not args.compact,
            )
            return 1
        _dump_json(sys.stdout, result.to_json(), pretty=not args.compact)
        return 0

    if args.command == "generate" and args.generate_command == "character-kontext-pose":
        values = vars(args)
        profile = CHARACTER_KONTEXT_POSE_PROFILES[args.profile]
        try:
            result = run_character_kontext_pose_control(
                values.get("model", profile.model),
                values.get("controlnet_model", profile.controlnet_model),
                args.reference_image,
                args.pose_image,
                args.output,
                args.prompt,
                device=args.device,
                dtype=values.get("dtype", profile.dtype),
                steps=values.get("steps", profile.steps),
                guidance_scale=values.get("guidance_scale", profile.guidance_scale),
                true_cfg_scale=values.get("true_cfg_scale", profile.true_cfg_scale),
                width=values.get("width", profile.width),
                height=values.get("height", profile.height),
                reference_max_area=values.get("reference_max_area", profile.reference_max_area),
                max_sequence_length=values.get("max_sequence_length", profile.max_sequence_length),
                framing=values.get("framing", profile.framing),
                negative_prompt=args.negative_prompt,
                seed=values.get("seed", profile.seed),
                pipeline_cpu_offload=values.get("pipeline_cpu_offload", profile.pipeline_cpu_offload),
                vae_tiling=values.get("vae_tiling", profile.vae_tiling),
                controlnet_conditioning_scale=values.get(
                    "controlnet_conditioning_scale",
                    profile.controlnet_conditioning_scale,
                ),
                control_guidance_start=values.get("control_guidance_start", profile.control_guidance_start),
                control_guidance_end=values.get("control_guidance_end", profile.control_guidance_end),
                transformer_single_file=values.get("transformer_single_file", profile.transformer_single_file),
            )
        except CharacterKontextPoseError as error:
            _dump_json(
                sys.stderr,
                {
                    "schema_version": 1,
                    "status": "error",
                    "error": error.__class__.__name__,
                    "message": str(error),
                },
                pretty=not args.compact,
            )
            return 1
        _dump_json(sys.stdout, result.to_json(), pretty=not args.compact)
        return 0

    if args.command == "generate" and args.generate_command == "character-nunchaku-kontext":
        values = vars(args)
        profile = NUNCHAKU_KONTEXT_PROFILES[args.profile]
        try:
            result = run_nunchaku_kontext(
                values.get("base_model", profile.base_model),
                values.get("transformer_model", profile.transformer_model),
                args.reference_image,
                args.output,
                args.prompt,
                device=args.device,
                dtype=values.get("dtype", profile.dtype),
                steps=values.get("steps", profile.steps),
                guidance_scale=values.get("guidance_scale", profile.guidance_scale),
                true_cfg_scale=values.get("true_cfg_scale", profile.true_cfg_scale),
                width=values.get("width", profile.width),
                height=values.get("height", profile.height),
                reference_max_area=values.get("reference_max_area", profile.reference_max_area),
                max_sequence_length=values.get("max_sequence_length", profile.max_sequence_length),
                framing=values.get("framing", profile.framing),
                negative_prompt=args.negative_prompt,
                seed=values.get("seed", profile.seed),
            )
        except NunchakuKontextError as error:
            _dump_json(
                sys.stderr,
                {
                    "schema_version": 1,
                    "status": "error",
                    "error": error.__class__.__name__,
                    "message": str(error),
                },
                pretty=not args.compact,
            )
            return 1
        _dump_json(sys.stdout, result.to_json(), pretty=not args.compact)
        return 0

    if args.command == "generate" and args.generate_command == "character-nunchaku-kontext-pose":
        values = vars(args)
        profile = NUNCHAKU_KONTEXT_POSE_PROFILES[args.profile]
        try:
            result = run_character_kontext_pose_control(
                values.get("model", profile.model),
                values.get("controlnet_model", profile.controlnet_model),
                args.reference_image,
                args.pose_image,
                args.output,
                args.prompt,
                device=args.device,
                dtype=values.get("dtype", profile.dtype),
                steps=values.get("steps", profile.steps),
                guidance_scale=values.get("guidance_scale", profile.guidance_scale),
                true_cfg_scale=values.get("true_cfg_scale", profile.true_cfg_scale),
                width=values.get("width", profile.width),
                height=values.get("height", profile.height),
                reference_max_area=values.get("reference_max_area", profile.reference_max_area),
                max_sequence_length=values.get("max_sequence_length", profile.max_sequence_length),
                framing=values.get("framing", profile.framing),
                negative_prompt=args.negative_prompt,
                seed=values.get("seed", profile.seed),
                pipeline_cpu_offload=values.get("pipeline_cpu_offload", profile.pipeline_cpu_offload),
                nunchaku_layer_offload=values.get("nunchaku_layer_offload", profile.nunchaku_layer_offload),
                vae_tiling=values.get("vae_tiling", profile.vae_tiling),
                controlnet_conditioning_scale=values.get(
                    "controlnet_conditioning_scale",
                    profile.controlnet_conditioning_scale,
                ),
                control_guidance_start=values.get("control_guidance_start", profile.control_guidance_start),
                control_guidance_end=values.get("control_guidance_end", profile.control_guidance_end),
                nunchaku_transformer_model=values.get(
                    "nunchaku_transformer_model",
                    profile.nunchaku_transformer_model,
                ),
                attention_impl=values.get("attention_impl", profile.attention_impl),
            )
        except CharacterKontextPoseError as error:
            _dump_json(
                sys.stderr,
                {
                    "schema_version": 1,
                    "status": "error",
                    "error": error.__class__.__name__,
                    "message": str(error),
                },
                pretty=not args.compact,
            )
            return 1
        _dump_json(sys.stdout, result.to_json(), pretty=not args.compact)
        return 0

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
