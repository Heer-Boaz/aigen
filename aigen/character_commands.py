from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from aigen.character_reference_models import CharacterReferenceError
from aigen.character_reference_pack import (
    build_character_reference_pack,
    parse_character_reference_args,
    parse_character_reference_files,
)
from aigen.character_region_plan import (
    CharacterRegionPlanError,
    parse_character_region_args,
    plan_character_regions,
)
from aigen.character_qwen_edit import (
    DEFAULT_QWEN_POSE_MODE,
    QWEN_POSE_MODES,
    QWEN_STRUCTURE_CONTROL_NAMES,
    QwenCharacterEditError,
    run_qwen_character_edit,
)
from aigen.generation.image_upscale import (
    DEFAULT_UPSCALE_MODEL,
    ImageUpscaleError,
    upscale_image,
    upscale_model_names,
)
from aigen.generation.vosr_backend import (
    VOSR_DEFAULT_ALIGN_METHOD,
    VOSR_DEFAULT_CFG_SCALE,
    VOSR_DEFAULT_INFER_STEPS,
    VOSR_DEFAULT_SCALE,
    VOSR_DEFAULT_SEED,
    VOSR_DEFAULT_TILE_SIZE,
    VOSR_DEFAULT_WEAK_COND_STRENGTH_AELQ,
    VosrBackendError,
    upscale_files_with_vosr,
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
    DEFAULT_QWEN_INPAINT_PROFILE,
    DEFAULT_QWEN_UPSCALE_LONG_SIDE,
    QwenImageEditIdentityError,
    parse_qwen_aspect_ratio,
    qwen_image_edit_identity_profile_for_name,
    qwen_image_edit_inpaint_model_names,
    qwen_image_edit_identity_model_names,
)
from aigen.keyframe_memory import KeyframeMemoryError
from aigen.keyframe_grounding import GroundingConfig, KeyframeGroundingError
from aigen.keyframe_segmentation import Sam2SegmentationConfig, KeyframeSegmentationError
from aigen.manifest_io import ManifestIOError
from aigen.progress import StatusReporter
from aigen.runtime_profiles import PROJECT_ROOT, keyframe_profile_for_name
from aigen.vlm_qwen import QwenVlmError


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
        help="Build multi-reference character image packs",
    )
    reference_pack_subparsers = reference_pack.add_subparsers(dest="reference_pack_command", required=True)

    reference_pack_build = reference_pack_subparsers.add_parser(
        "build",
        help="Build a named character reference pack",
    )
    reference_pack_build.add_argument("--character-id", required=True, help="Character id")
    reference_pack_inputs = reference_pack_build.add_mutually_exclusive_group(required=True)
    reference_pack_inputs.add_argument(
        "--reference",
        action="append",
        help="Named reference image as name=path; names are stable pack-local handles",
    )
    reference_pack_inputs.add_argument(
        "--file",
        action="append",
        type=Path,
        help="Reference image path; repeat for each image and use each filename stem as its pack handle",
    )
    reference_pack_build.add_argument("--output-dir", type=Path, required=True, help="Reference pack directory")
    reference_pack_build.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    reference_pack_build.add_argument("--compact", action="store_true", help="Write compact JSON")

    qwen_edit = character_subparsers.add_parser(
        "qwen-edit-run",
        help="Run one free-instruction Qwen Image Edit request",
    )
    qwen_edit.add_argument("--pack", type=Path, help="Reference pack for reference-conditioned generation")
    qwen_edit.add_argument(
        "--image",
        action="append",
        type=Path,
        help="Model input image in Picture order; repeat for each image",
    )
    qwen_edit.add_argument(
        "--instruction",
        required=True,
        help="Free edit or generation instruction",
    )
    qwen_edit.add_argument(
        "--pose-source",
        type=Path,
        help="Source image used natively or through DWPose for pose_transfer",
    )
    qwen_edit.add_argument(
        "--pose-mode",
        choices=QWEN_POSE_MODES,
        default=DEFAULT_QWEN_POSE_MODE,
        help=f"Pose transfer strategy; defaults to {DEFAULT_QWEN_POSE_MODE}",
    )
    qwen_edit.add_argument(
        "--structure-source",
        type=Path,
        help="Scene composition image used to derive a depth or edge control",
    )
    qwen_edit.add_argument(
        "--structure-control",
        choices=QWEN_STRUCTURE_CONTROL_NAMES,
        help="Structural scene control to derive from --structure-source",
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
        help="Upper cap for the generated/reference long side",
    )
    qwen_edit.add_argument(
        "--aspect-ratio",
        type=parse_qwen_aspect_ratio,
        help="Raw canvas aspect as W:H; source or structural input owns the aspect when omitted",
    )
    qwen_edit.add_argument(
        "--upscale-long-side",
        type=int,
        default=DEFAULT_QWEN_UPSCALE_LONG_SIDE,
        help=f"Upscaled long side in pixels; defaults to {DEFAULT_QWEN_UPSCALE_LONG_SIDE}",
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
        default=1,
        help="Number of outputs; defaults to one",
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

    postprocess = character_subparsers.add_parser(
        "postprocess",
        help="Upscale one existing character image",
    )
    postprocess.add_argument("--image", type=Path, required=True, help="Input image")
    postprocess.add_argument("--output", type=Path, required=True, help="Upscaled output image")
    postprocess.add_argument(
        "--long-side",
        type=int,
        default=DEFAULT_QWEN_UPSCALE_LONG_SIDE,
        help=f"Output long side in pixels; defaults to {DEFAULT_QWEN_UPSCALE_LONG_SIDE}",
    )
    postprocess.add_argument(
        "--model",
        choices=upscale_model_names(),
        default=DEFAULT_UPSCALE_MODEL,
        help=f"Upscale model; defaults to {DEFAULT_UPSCALE_MODEL}",
    )
    postprocess.add_argument("--compact", action="store_true", help="Write compact JSON")

    vosr_upscale = character_subparsers.add_parser(
        "vosr-upscale",
        help="VOSR-1.4B-ms upscale",
    )
    vosr_upscale.set_defaults(compact=False)
    vosr_upscale.add_argument("--input", type=Path, required=True)
    vosr_upscale.add_argument("--output", type=Path, required=True)
    vosr_size = vosr_upscale.add_mutually_exclusive_group()
    vosr_size.add_argument("--scale", type=int, default=VOSR_DEFAULT_SCALE)
    vosr_size.add_argument("--long-side", type=int)
    vosr_upscale.add_argument("--infer-steps", type=int, default=VOSR_DEFAULT_INFER_STEPS)
    vosr_upscale.add_argument("--cfg-scale", type=float, default=VOSR_DEFAULT_CFG_SCALE)
    vosr_upscale.add_argument(
        "--weak-cond-strength-aelq",
        type=float,
        default=VOSR_DEFAULT_WEAK_COND_STRENGTH_AELQ,
    )
    vosr_upscale.add_argument(
        "--align-method",
        choices=("wavelet", "adain", "nofix"),
        default=VOSR_DEFAULT_ALIGN_METHOD,
    )
    vosr_upscale.add_argument("--tile-size", type=int, default=VOSR_DEFAULT_TILE_SIZE)
    vosr_upscale.add_argument("--seed", type=int, default=VOSR_DEFAULT_SEED)

    qwen_refine_plan = character_subparsers.add_parser(
        "qwen-edit-refine-plan",
        help="Plan a masked Qwen Image Edit repair from a selected candidate and reference pack",
    )
    qwen_refine_plan.add_argument("--pack", type=Path, required=True, help="reference_pack.json")
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
    qwen_refine.add_argument("--image", type=Path, required=True, help="Selected candidate image to refine")
    qwen_refine.add_argument("--mask", type=Path, help="White-on-black repaint mask")
    qwen_refine.add_argument("--region-plan", type=Path, help="characters region-plan result.json")
    qwen_refine.add_argument("--region", help="Region name inside --region-plan")
    qwen_refine.add_argument("--instruction", required=True, help="Local repair instruction")
    qwen_refine.add_argument("--output-dir", type=Path, required=True, help="Directory for refine candidates")
    qwen_refine.add_argument(
        "--model",
        dest="profile",
        default=DEFAULT_QWEN_INPAINT_PROFILE,
        choices=qwen_image_edit_inpaint_model_names(),
        help="Qwen Image Edit model",
    )
    qwen_refine.add_argument(
        "--max-side",
        type=int,
        help="Optional upper cap for the native source long side",
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
    region_plan.add_argument("--florence-model", type=Path, default=GroundingConfig.florence_model, help="Florence-2 model dir")
    region_plan.add_argument("--sam2-model", type=Path, default=Sam2SegmentationConfig.model, help="SAM2 model dir")
    region_plan.add_argument("--device", default="cuda", help="Device for Florence-2 and SAM2")
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
                        references=(
                            parse_character_reference_files(args.file, Path.cwd())
                            if args.file
                            else parse_character_reference_args(args.reference, Path.cwd())
                        ),
                        output_dir=args.output_dir,
                        overwrite=args.overwrite,
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
                    instruction=args.instruction,
                    source_image_paths=args.image or (),
                    max_side=args.max_side,
                    steps=args.steps,
                    true_cfg_scale=args.true_cfg_scale,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                    max_sequence_length=args.max_sequence_length,
                    candidates_per_case=args.candidates,
                    aspect_ratio=args.aspect_ratio,
                    upscale_long_side=args.upscale_long_side,
                    overwrite=args.overwrite,
                    nunchaku_blocks_on_gpu=args.nunchaku_blocks_on_gpu,
                    pose_source_path=args.pose_source,
                    pose_mode=args.pose_mode,
                    structure_source_path=args.structure_source,
                    structure_control=args.structure_control,
                    progress=progress,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "postprocess":
            dump_json(
                stdout,
                upscale_image(
                    input_path=args.image,
                    output_path=args.output,
                    long_side=args.long_side,
                    model=args.model,
                    progress=progress,
                ),
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "vosr-upscale":
            batch = upscale_files_with_vosr(
                files=((args.input, args.output),),
                scale=args.scale,
                long_side=args.long_side,
                infer_steps=args.infer_steps,
                cfg_scale=args.cfg_scale,
                weak_cond_strength_aelq=args.weak_cond_strength_aelq,
                align_method=args.align_method,
                tile_size=args.tile_size,
                seed=args.seed,
                progress=progress,
            )
            dump_json(
                stdout,
                batch.outputs[0] | {"elapsed_ms": batch.elapsed_ms},
                pretty=not args.compact,
            )
            return 0
        if args.characters_command == "qwen-edit-refine-plan":
            dump_json(
                stdout,
                plan_qwen_character_refine(
                    pack_path=args.pack,
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
                        florence_model=args.florence_model,
                        device=args.device,
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
        VosrBackendError,
        KeyframeGroundingError,
        KeyframeSegmentationError,
        KeyframeMemoryError,
        ImageUpscaleError,
        ManifestIOError,
        QwenVlmError,
    ) as error:
        dump_json(stderr, command_error_payload(error), pretty=not args.compact)
        return 1
