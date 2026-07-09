from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.character_instruction_parser import CharacterInstructionParserConfig
from aigen.character_reference_models import CharacterReferenceError
from aigen.character_reference_pack import (
    build_character_reference_pack,
    parse_character_reference_args,
    parse_character_reference_pack,
)
from aigen.character_region_plan import (
    CharacterRegionPlanError,
    parse_character_region_args,
    plan_character_regions,
)
from aigen.character_qwen_edit import (
    QwenCharacterEditError,
    plan_qwen_character_edit,
    qwen_character_edit_case_names,
    run_qwen_character_edit,
)
from aigen.character_qwen_refine import (
    QwenCharacterRefineError,
    plan_qwen_character_refine,
    run_qwen_character_refine,
)
from aigen.character_view_models import (
    CharacterViewError,
    character_view_bank_schema,
    character_view_job_schema,
    load_character_view_job,
)
from aigen.character_views import (
    accept_character_view,
    plan_character_view_job,
    run_character_view_job,
    validate_character_view_job,
)
from aigen.command_io import command_error_payload, dump_json
from aigen.generation.qwen_image_edit_identity import (
    DEFAULT_QWEN_IDENTITY_MAX_SEQUENCE_LENGTH,
    DEFAULT_QWEN_IDENTITY_MAX_SIDE,
    DEFAULT_QWEN_IDENTITY_PROFILE,
    DEFAULT_QWEN_IDENTITY_SEED,
    QwenImageEditIdentityError,
    parse_qwen_identity_reference_args,
    qwen_identity_cli_case_names,
    qwen_image_edit_identity_profile_for_name,
    qwen_image_edit_identity_model_names,
    qwen_image_edit_identity_profile_names,
    run_qwen_image_edit_identity,
)
from aigen.keyframe_memory import KeyframeMemoryError
from aigen.keyframe_grounding import GroundingConfig, KeyframeGroundingError
from aigen.keyframe_segmentation import Sam2SegmentationConfig, KeyframeSegmentationError
from aigen.judge_cli import add_judge_runtime_args, judge_config_from_args
from aigen.manifest_io import ManifestIOError
from aigen.progress import StatusReporter
from aigen.runtime_profiles import MODELS_ROOT, PROJECT_ROOT, keyframe_profile_for_name
from aigen.text_llm import TextLlmConfig
from aigen.vlm_qwen import (
    DEFAULT_JUDGE_ID,
    DEFAULT_JUDGE_QUANTIZATION,
    DEFAULT_JUDGE_REPO_ID,
    DEFAULT_JUDGE_REVISION,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    QwenVlmConfig,
    QwenVlmError,
)


DEFAULT_INSTRUCTION_PARSER_ID = "qwen3-8b-instruction-parser"
DEFAULT_INSTRUCTION_PARSER_MODEL = MODELS_ROOT / "llm/Qwen/Qwen3-8B"
DEFAULT_INSTRUCTION_PARSER_DTYPE = "bfloat16"
DEFAULT_INSTRUCTION_PARSER_QUANTIZATION = "bitsandbytes-8bit"
DEFAULT_INSTRUCTION_PARSER_MAX_NEW_TOKENS = 700
DEFAULT_INSTRUCTION_PARSER_TEMPERATURE = 0.0
DEFAULT_INSTRUCTION_PARSER_ENABLE_THINKING = False
DEFAULT_EDIT_PLANNER_ID = DEFAULT_JUDGE_ID
DEFAULT_EDIT_PLANNER_MODEL = MODELS_ROOT / "vlm/Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_EDIT_PLANNER_DTYPE = "bfloat16"
DEFAULT_EDIT_PLANNER_ATTENTION_IMPL = "sdpa"
DEFAULT_EDIT_PLANNER_QUANTIZATION = DEFAULT_JUDGE_QUANTIZATION
DEFAULT_EDIT_PLANNER_MIN_PIXELS = DEFAULT_MIN_PIXELS
DEFAULT_EDIT_PLANNER_MAX_PIXELS = DEFAULT_MAX_PIXELS
DEFAULT_EDIT_PLANNER_MAX_NEW_TOKENS = 2200
DEFAULT_EDIT_PLANNER_TEMPERATURE = 0.0


def default_instruction_parser_config() -> CharacterInstructionParserConfig:
    return CharacterInstructionParserConfig(
        text_llm=TextLlmConfig(
            parser_id=DEFAULT_INSTRUCTION_PARSER_ID,
            model=DEFAULT_INSTRUCTION_PARSER_MODEL,
            dtype=DEFAULT_INSTRUCTION_PARSER_DTYPE,
            quantization=DEFAULT_INSTRUCTION_PARSER_QUANTIZATION,
            max_new_tokens=DEFAULT_INSTRUCTION_PARSER_MAX_NEW_TOKENS,
            temperature=DEFAULT_INSTRUCTION_PARSER_TEMPERATURE,
            enable_thinking=DEFAULT_INSTRUCTION_PARSER_ENABLE_THINKING,
        )
    )


def default_edit_planner_vlm_config() -> QwenVlmConfig:
    return QwenVlmConfig(
        judge_id=DEFAULT_EDIT_PLANNER_ID,
        model=DEFAULT_EDIT_PLANNER_MODEL,
        repo_id=DEFAULT_JUDGE_REPO_ID,
        revision=DEFAULT_JUDGE_REVISION,
        dtype=DEFAULT_EDIT_PLANNER_DTYPE,
        attention_impl=DEFAULT_EDIT_PLANNER_ATTENTION_IMPL,
        quantization=DEFAULT_EDIT_PLANNER_QUANTIZATION,
        min_pixels=DEFAULT_EDIT_PLANNER_MIN_PIXELS,
        max_pixels=DEFAULT_EDIT_PLANNER_MAX_PIXELS,
        max_new_tokens=DEFAULT_EDIT_PLANNER_MAX_NEW_TOKENS,
        temperature=DEFAULT_EDIT_PLANNER_TEMPERATURE,
    )


def add_character_commands(subparsers: Any) -> None:
    characters = subparsers.add_parser("characters", help="Character view bank tools")
    character_subparsers = characters.add_subparsers(dest="characters_command", required=True)

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

    reference_pack = character_subparsers.add_parser(
        "reference-pack",
        help="Build and parse multi-reference character identity packs",
    )
    reference_pack_subparsers = reference_pack.add_subparsers(dest="reference_pack_command", required=True)

    reference_pack_build = reference_pack_subparsers.add_parser(
        "build",
        help="Build a named character reference pack",
    )
    reference_pack_build.add_argument("--character-id", required=True, help="Character id")
    reference_pack_build.add_argument(
        "--reference",
        action="append",
        required=True,
        help="Named reference image as name=path; names are pack-local ids and optional role hints",
    )
    reference_pack_build.add_argument("--output-dir", type=Path, required=True, help="Reference pack directory")
    reference_pack_build.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    reference_pack_build.add_argument("--compact", action="store_true", help="Write compact JSON")

    reference_pack_parse = reference_pack_subparsers.add_parser(
        "parse",
        help="Use the local Qwen VLM to write an identity profile from a reference pack",
    )
    reference_pack_parse.add_argument("pack", type=Path, help="reference_pack.json")
    reference_pack_parse.add_argument("--output", type=Path, help="Generated identity_profile.json")
    reference_pack_parse.add_argument("--overwrite", action="store_true", help="Replace an existing identity profile")
    add_judge_runtime_args(reference_pack_parse, role="reference parser", max_new_tokens=1600)
    reference_pack_parse.add_argument("--compact", action="store_true", help="Write compact JSON")

    qwen_identity = character_subparsers.add_parser(
        "qwen-identity-run",
        help="Run fixed multi-reference Qwen Image Edit identity cases from explicit refs",
    )
    qwen_identity.add_argument(
        "--reference",
        action="append",
        required=True,
        help="Explicit role reference image as role=path; supported roles: front, portrait, side, back, body_shape",
    )
    qwen_identity.add_argument(
        "--case",
        action="append",
        choices=qwen_identity_cli_case_names(),
        help="Identity case to generate; defaults to all fixed cases",
    )
    qwen_identity.add_argument("--output-dir", type=Path, required=True, help="Directory for generated images")
    qwen_identity.add_argument(
        "--profile",
        default=DEFAULT_QWEN_IDENTITY_PROFILE,
        choices=qwen_image_edit_identity_profile_names(),
        help="Qwen Image Edit model profile",
    )
    qwen_identity.add_argument(
        "--max-side",
        type=int,
        default=DEFAULT_QWEN_IDENTITY_MAX_SIDE,
        help="Longest generated/reference side for the smoke run",
    )
    qwen_identity.add_argument(
        "--steps",
        type=int,
        help="Qwen Image Edit denoising steps; defaults to the selected profile",
    )
    qwen_identity.add_argument(
        "--true-cfg-scale",
        type=float,
        help="Classifier-free guidance scale; defaults to the selected profile",
    )
    qwen_identity.add_argument(
        "--guidance-scale",
        type=float,
        help="Guidance-distilled model scale; defaults to the selected profile",
    )
    qwen_identity.add_argument("--seed", type=int, default=DEFAULT_QWEN_IDENTITY_SEED, help="Base seed")
    qwen_identity.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_QWEN_IDENTITY_MAX_SEQUENCE_LENGTH,
        help="Maximum prompt token sequence length",
    )
    qwen_identity.add_argument(
        "--nunchaku-blocks-on-gpu",
        type=int,
        help="Explicit slow-fit Nunchaku layer offload; leave unset for direct GPU execution",
    )
    qwen_identity.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    qwen_identity.add_argument("--compact", action="store_true", help="Write compact JSON")

    qwen_edit_plan = character_subparsers.add_parser(
        "qwen-edit-plan",
        help="Plan pack/profile-driven Qwen Image Edit character cases without loading generation models",
    )
    qwen_edit_plan.add_argument("--pack", type=Path, required=True, help="reference_pack.json")
    qwen_edit_plan.add_argument(
        "--case",
        action="append",
        choices=qwen_character_edit_case_names(),
        help="Edit case to plan; defaults to identity view cases",
    )
    qwen_edit_plan.add_argument(
        "--instruction",
        help="Optional user request for the VLM edit planner; defaults to the selected case name",
    )
    qwen_edit_plan.add_argument("--candidates", type=int, default=2, help="Candidates per case")
    qwen_edit_plan.add_argument("--compact", action="store_true", help="Write compact JSON")

    qwen_edit = character_subparsers.add_parser(
        "qwen-edit-run",
        help="Run pack/profile-driven Qwen Image Edit character cases",
    )
    qwen_edit.add_argument("--pack", type=Path, required=True, help="reference_pack.json")
    qwen_edit.add_argument(
        "--case",
        action="append",
        choices=qwen_character_edit_case_names(),
        help="Edit case to generate; defaults to identity view cases",
    )
    qwen_edit.add_argument(
        "--instruction",
        help="Optional user request for the VLM edit planner; defaults to the selected case name",
    )
    qwen_edit.add_argument("--output-dir", type=Path, required=True, help="Directory for generated images")
    qwen_edit.add_argument(
        "--model",
        dest="profile",
        default=DEFAULT_QWEN_IDENTITY_PROFILE,
        choices=qwen_image_edit_identity_model_names(),
        help="Qwen Image Edit model",
    )
    qwen_edit.add_argument(
        "--max-side",
        type=int,
        default=DEFAULT_QWEN_IDENTITY_MAX_SIDE,
        help="Longest generated/reference side",
    )
    qwen_edit.add_argument(
        "--steps",
        type=int,
        help="Qwen Image Edit denoising steps; defaults to the selected profile",
    )
    qwen_edit.add_argument(
        "--true-cfg-scale",
        type=float,
        help="Classifier-free guidance scale; defaults to the selected profile",
    )
    qwen_edit.add_argument(
        "--guidance-scale",
        type=float,
        help="Guidance-distilled model scale; defaults to the selected profile",
    )
    qwen_edit.add_argument("--seed", type=int, default=DEFAULT_QWEN_IDENTITY_SEED, help="Base seed")
    qwen_edit.add_argument(
        "--candidates",
        type=int,
        default=2,
        help="Candidates per case",
    )
    qwen_edit.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_QWEN_IDENTITY_MAX_SEQUENCE_LENGTH,
        help="Maximum prompt token sequence length",
    )
    qwen_edit.add_argument(
        "--nunchaku-blocks-on-gpu",
        type=int,
        help="Explicit slow-fit Nunchaku layer offload; leave unset for direct GPU execution",
    )
    qwen_edit.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    qwen_edit.add_argument("--compact", action="store_true", help="Write compact JSON")

    qwen_refine_plan = character_subparsers.add_parser(
        "qwen-edit-refine-plan",
        help="Plan a masked Qwen Image Edit repair from a selected candidate and reference pack",
    )
    qwen_refine_plan.add_argument("--pack", type=Path, required=True, help="reference_pack.json")
    qwen_refine_plan.add_argument(
        "--identity-profile",
        type=Path,
        help="identity_profile.json; defaults to identity_profile.json next to the pack",
    )
    qwen_refine_plan.add_argument("--image", type=Path, required=True, help="Selected candidate image to refine")
    qwen_refine_plan.add_argument("--mask", type=Path, help="White-on-black repaint mask")
    qwen_refine_plan.add_argument("--region-plan", type=Path, help="characters region-plan result.json")
    qwen_refine_plan.add_argument("--region", help="Region name inside --region-plan")
    qwen_refine_plan.add_argument("--instruction", required=True, help="Local repair instruction")
    qwen_refine_plan.add_argument("--candidates", type=int, default=2, help="Candidates to generate")
    qwen_refine_plan.add_argument("--compact", action="store_true", help="Write compact JSON")

    qwen_refine = character_subparsers.add_parser(
        "qwen-edit-refine",
        help="Run masked Qwen Image Edit repair candidates from a selected image and reference pack",
    )
    qwen_refine.add_argument("--pack", type=Path, required=True, help="reference_pack.json")
    qwen_refine.add_argument(
        "--identity-profile",
        type=Path,
        help="identity_profile.json; defaults to identity_profile.json next to the pack",
    )
    qwen_refine.add_argument("--image", type=Path, required=True, help="Selected candidate image to refine")
    qwen_refine.add_argument("--mask", type=Path, help="White-on-black repaint mask")
    qwen_refine.add_argument("--region-plan", type=Path, help="characters region-plan result.json")
    qwen_refine.add_argument("--region", help="Region name inside --region-plan")
    qwen_refine.add_argument("--instruction", required=True, help="Local repair instruction")
    qwen_refine.add_argument("--output-dir", type=Path, required=True, help="Directory for refine candidates")
    qwen_refine.add_argument(
        "--model",
        dest="profile",
        default=DEFAULT_QWEN_IDENTITY_PROFILE,
        choices=qwen_image_edit_identity_model_names(),
        help="Qwen Image Edit model",
    )
    qwen_refine.add_argument(
        "--max-side",
        type=int,
        default=DEFAULT_QWEN_IDENTITY_MAX_SIDE,
        help="Longest source/reference side before prompt conditioning",
    )
    qwen_refine.add_argument(
        "--steps",
        type=int,
        help="Qwen Image Edit denoising steps; defaults to the selected profile",
    )
    qwen_refine.add_argument(
        "--true-cfg-scale",
        type=float,
        help="Classifier-free guidance scale; defaults to the selected profile",
    )
    qwen_refine.add_argument(
        "--guidance-scale",
        type=float,
        help="Guidance-distilled model scale; defaults to the selected profile",
    )
    qwen_refine.add_argument(
        "--strength",
        type=float,
        default=0.6,
        help="Inpaint denoise strength; higher changes the masked area more",
    )
    qwen_refine.add_argument(
        "--padding-mask-crop",
        type=int,
        help="Optional inpaint crop padding around the white mask region",
    )
    qwen_refine.add_argument("--seed", type=int, default=DEFAULT_QWEN_IDENTITY_SEED, help="Base seed")
    qwen_refine.add_argument("--candidates", type=int, default=2, help="Candidates to generate")
    qwen_refine.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_QWEN_IDENTITY_MAX_SEQUENCE_LENGTH,
        help="Maximum prompt token sequence length",
    )
    qwen_refine.add_argument(
        "--nunchaku-blocks-on-gpu",
        type=int,
        help="Explicit slow-fit Nunchaku layer offload; leave unset for direct GPU execution",
    )
    qwen_refine.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    qwen_refine.add_argument("--compact", action="store_true", help="Write compact JSON")

    region_plan = character_subparsers.add_parser(
        "region-plan",
        help="Ground requested character regions and write SAM2 masks without generation",
    )
    region_plan.add_argument("--image", type=Path, required=True, help="Image to ground and mask")
    region_plan.add_argument(
        "--region",
        action="append",
        required=True,
        help="Region request as NAME=TEXT, for example face='visible face'",
    )
    region_plan.add_argument("--output-dir", type=Path, required=True, help="Output directory for masks and debug sheet")
    region_plan.add_argument("--dino-model", type=Path, default=GroundingConfig.dino_model, help="GroundingDINO model dir")
    region_plan.add_argument("--florence-model", type=Path, default=GroundingConfig.florence_model, help="Florence-2 model dir")
    region_plan.add_argument("--sam2-model", type=Path, default=Sam2SegmentationConfig.model, help="SAM2 model dir")
    region_plan.add_argument("--device", default="cuda", help="Device for grounding and SAM2")
    region_plan.add_argument("--grounding-threshold", type=float, default=GroundingConfig.threshold, help="GroundingDINO box threshold")
    region_plan.add_argument("--text-threshold", type=float, default=GroundingConfig.text_threshold, help="GroundingDINO text threshold")
    region_plan.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    region_plan.add_argument("--compact", action="store_true", help="Write compact JSON")


def run_character_command(
    args: argparse.Namespace,
    stdout: TextIO,
    stderr: TextIO,
    *,
    progress: StatusReporter,
) -> int:
    try:
        if args.characters_command == "view-schema":
            dump_json(stdout, character_view_job_schema(), pretty=not args.compact)
            return 0
        if args.characters_command == "view-bank-schema":
            dump_json(stdout, character_view_bank_schema(), pretty=not args.compact)
            return 0
        if args.characters_command == "view-validate":
            dump_json(
                stdout,
                validate_character_view_job(args.job, project_root=PROJECT_ROOT),
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "view-plan":
            dump_json(
                stdout,
                plan_character_view_job(args.job, project_root=PROJECT_ROOT),
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "view-run":
            dump_json(
                stdout,
                run_character_view_job(
                    args.job,
                    keyframe_profile_for_name(load_character_view_job(args.job).pipeline.profile),
                    project_root=PROJECT_ROOT,
                    progress=progress,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "view-accept":
            dump_json(
                stdout,
                accept_character_view(
                    args.job,
                    run_dir=args.run_dir,
                    candidate=args.candidate,
                    project_root=PROJECT_ROOT,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "reference-pack":
            if args.reference_pack_command == "build":
                dump_json(
                    stdout,
                    build_character_reference_pack(
                        character_id=args.character_id,
                        references=parse_character_reference_args(args.reference, Path.cwd()),
                        output_dir=args.output_dir,
                        overwrite=args.overwrite,
                    ),
                    pretty=not args.compact,
                )
                return 0
            if args.reference_pack_command == "parse":
                dump_json(
                    stdout,
                    parse_character_reference_pack(
                        args.pack,
                        judge_config_from_args(args),
                        output_path=args.output,
                        overwrite=args.overwrite,
                        progress=progress,
                    ),
                    pretty=not args.compact,
                )
                return 0
        if args.characters_command == "qwen-identity-run":
            dump_json(
                stdout,
                run_qwen_image_edit_identity(
                    references=parse_qwen_identity_reference_args(args.reference, Path.cwd()),
                    output_dir=args.output_dir,
                    profile=qwen_image_edit_identity_profile_for_name(args.profile),
                    cases=args.case or (),
                    max_side=args.max_side,
                    steps=args.steps,
                    true_cfg_scale=args.true_cfg_scale,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                    max_sequence_length=args.max_sequence_length,
                    overwrite=args.overwrite,
                    nunchaku_blocks_on_gpu=args.nunchaku_blocks_on_gpu,
                    progress=progress,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "qwen-edit-plan":
            dump_json(
                stdout,
                plan_qwen_character_edit(
                    pack_path=args.pack,
                    instruction_parser_config=default_instruction_parser_config(),
                    vlm_config=default_edit_planner_vlm_config(),
                    cases=args.case or (),
                    instruction=args.instruction,
                    candidates_per_case=args.candidates,
                    progress=progress,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "qwen-edit-run":
            dump_json(
                stdout,
                run_qwen_character_edit(
                    pack_path=args.pack,
                    output_dir=args.output_dir,
                    profile=qwen_image_edit_identity_profile_for_name(args.profile),
                    instruction_parser_config=default_instruction_parser_config(),
                    vlm_config=default_edit_planner_vlm_config(),
                    cases=args.case or (),
                    instruction=args.instruction,
                    max_side=args.max_side,
                    steps=args.steps,
                    true_cfg_scale=args.true_cfg_scale,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                    max_sequence_length=args.max_sequence_length,
                    candidates_per_case=args.candidates,
                    overwrite=args.overwrite,
                    nunchaku_blocks_on_gpu=args.nunchaku_blocks_on_gpu,
                    progress=progress,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "qwen-edit-refine-plan":
            dump_json(
                stdout,
                plan_qwen_character_refine(
                    pack_path=args.pack,
                    identity_profile_path=args.identity_profile,
                    source_image_path=args.image,
                    mask_path=args.mask,
                    region_plan_path=args.region_plan,
                    region_name=args.region,
                    instruction=args.instruction,
                    candidates=args.candidates,
                    progress=progress,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "qwen-edit-refine":
            dump_json(
                stdout,
                run_qwen_character_refine(
                    pack_path=args.pack,
                    identity_profile_path=args.identity_profile,
                    source_image_path=args.image,
                    mask_path=args.mask,
                    region_plan_path=args.region_plan,
                    region_name=args.region,
                    instruction=args.instruction,
                    output_dir=args.output_dir,
                    profile=qwen_image_edit_identity_profile_for_name(args.profile),
                    max_side=args.max_side,
                    steps=args.steps,
                    true_cfg_scale=args.true_cfg_scale,
                    guidance_scale=args.guidance_scale,
                    strength=args.strength,
                    padding_mask_crop=args.padding_mask_crop,
                    seed=args.seed,
                    max_sequence_length=args.max_sequence_length,
                    candidates=args.candidates,
                    overwrite=args.overwrite,
                    nunchaku_blocks_on_gpu=args.nunchaku_blocks_on_gpu,
                    progress=progress,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "region-plan":
            dump_json(
                stdout,
                plan_character_regions(
                    image_path=args.image,
                    regions=parse_character_region_args(args.region),
                    output_dir=args.output_dir,
                    overwrite=args.overwrite,
                    grounding_config=GroundingConfig(
                        dino_model=args.dino_model,
                        florence_model=args.florence_model,
                        device=args.device,
                        threshold=args.grounding_threshold,
                        text_threshold=args.text_threshold,
                    ),
                    segmentation_config=Sam2SegmentationConfig(
                        model=args.sam2_model,
                        device=args.device,
                    ),
                    progress=progress,
                ),
                pretty=not args.compact,
            )
            return 0
    except (
        CharacterReferenceError,
        CharacterRegionPlanError,
        QwenCharacterEditError,
        QwenCharacterRefineError,
        CharacterViewError,
        QwenImageEditIdentityError,
        KeyframeGroundingError,
        KeyframeSegmentationError,
        KeyframeMemoryError,
        ManifestIOError,
        QwenVlmError,
    ) as error:
        dump_json(stderr, command_error_payload(error), pretty=not args.compact)
        return 1
