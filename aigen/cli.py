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
from aigen.generation.kontext_pose_control import (
    CharacterKontextPoseError,
    run_character_kontext_pose_control,
)
from aigen.generation.pose_control import (
    CharacterPoseError,
    run_character_pose_control,
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
    cpu_offload: bool
    vae_tiling: bool
    controlnet_conditioning_scale: float
    control_guidance_start: float
    control_guidance_end: float
    seed: int


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
        cpu_offload=False,
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
        cpu_offload=True,
        vae_tiling=True,
        controlnet_conditioning_scale=0.65,
        control_guidance_start=0.0,
        control_guidance_end=0.65,
        seed=1,
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


def _add_generate_commands(subparsers: Any) -> None:
    generate = subparsers.add_parser("generate", help="Run private generation pipelines")
    generate_subparsers = generate.add_subparsers(dest="generate_command", required=True)

    _add_character_concept_command(generate_subparsers)
    _add_character_kontext_pose_command(generate_subparsers)
    _add_character_pose_command(generate_subparsers)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aigen",
        description="AI character generation pipeline tooling",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_generate_commands(subparsers)
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
                cpu_offload=values.get("cpu_offload", profile.cpu_offload),
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

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
