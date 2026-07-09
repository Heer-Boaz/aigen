from __future__ import annotations

import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import gc
from math import gcd, lcm, log
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import Any

from PIL import Image

from aigen.character_reference_models import CHARACTER_REFERENCE_NAME_SET
from aigen.generation.image_upscale import (
    REALESRGAN_ANIME_X4_MODEL_NAME,
    RealESRGANAnimeUpscaler,
)
from aigen.generation.prompt_encoding import tensor_to_device
from aigen.generation.qwen_prompt_encoding import (
    QWEN_IMAGE_EDIT_NEGATIVE_PROMPT,
    QwenImageEditPromptEmbedding,
    QwenImageEditPromptRequest,
    encode_qwen_image_edit_prompts,
)
from aigen.generation.runtime_diagnostics import (
    cuda_memory_stats,
    elapsed_ms,
    module_device_report,
    synchronized_time,
)
from aigen.generation.runtime_types import resolve_torch_dtype
from aigen.image_assets import image_asset_json
from aigen.keyframe_image_ops import save_contact_sheet
from aigen.keyframe_memory import NvidiaSmiMemorySampler, nvidia_smi_preflight_limit
from aigen.manifest_io import resolve_existing_path, write_json
from aigen.progress import StatusReporter
from aigen.runtime_profiles import MODELS_ROOT


DEFAULT_QWEN_IDENTITY_PROFILE = "nunchaku-qwen-edit-2509-fp4-r32-lightning-4step"
DEFAULT_QWEN_IDENTITY_MAX_SIDE = 640
DEFAULT_QWEN_IDENTITY_SEED = 0
DEFAULT_QWEN_IDENTITY_MAX_SEQUENCE_LENGTH = 512
QWEN_IDENTITY_PREFLIGHT_LIMIT_MB = 4096
QWEN_IDENTITY_REFERENCE_NAMES = CHARACTER_REFERENCE_NAME_SET
QWEN_OUTPUT_FORMATS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "2:3": (2, 3),
    "3:2": (3, 2),
    "3:4": (3, 4),
    "4:3": (4, 3),
    "9:16": (9, 16),
    "16:9": (16, 9),
}
QWEN_OUTPUT_FORMAT_ORDER = ("1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9")
QWEN_OUTPUT_RESOLUTIONS = {
    "1k": 1024,
    "2k": 2048,
}
QWEN_OUTPUT_RESOLUTION_ORDER = ("1k", "2k")


class QwenImageEditIdentityError(RuntimeError):
    pass


class QwenImageEditIdentityDependencyError(QwenImageEditIdentityError):
    pass


@dataclass(frozen=True)
class QwenImageEditIdentityProfile:
    name: str
    base_model: str
    base_repo_id: str
    base_revision: str
    dtype: str
    nunchaku_transformer_model: Path
    nunchaku_repo_id: str
    nunchaku_revision: str
    nunchaku_variant: str
    default_steps: int
    default_true_cfg_scale: float
    default_guidance_scale: float
    scheduler_config: dict[str, Any] | None
    local_files_only: bool = True


@dataclass(frozen=True)
class QwenIdentityCase:
    name: str
    references: tuple[str, ...]
    prompt: str
    portrait_canvas: bool = False
    edit_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class QwenIdentityReferenceStep:
    cases: tuple[QwenIdentityCase, ...]
    reference_images: dict[str, Image.Image]


@dataclass(frozen=True)
class QwenIdentityPromptConditioningStep:
    embeddings: dict[str, QwenImageEditPromptEmbedding]
    elapsed_ms: float


@dataclass(frozen=True)
class QwenIdentityDenoiseStep:
    outputs: list[dict[str, Any]]
    elapsed_ms: float


@dataclass(frozen=True)
class QwenIdentityPostprocessStep:
    outputs: list[dict[str, Any]]
    elapsed_ms: float


@dataclass(frozen=True)
class QwenOutputSpec:
    output_format: str
    resolution: str
    raw_width: int
    raw_height: int
    final_width: int
    final_height: int


@dataclass(frozen=True)
class QwenImageEditLatentResult:
    name: str
    case_name: str | None
    candidate_index: int
    latents: Any
    width: int
    height: int
    seed: int
    timings_ms: dict[str, Any]


@dataclass(frozen=True)
class QwenImageEditInpaintReference:
    name: str
    role: str
    purpose: str
    path: Path


@dataclass(frozen=True)
class QwenImageEditInpaintDenoiseStep:
    outputs: list[dict[str, Any]]
    elapsed_ms: float


@dataclass(frozen=True)
class QwenImageEditInpaintCanvas:
    source_image: Image.Image
    mask_image: Image.Image
    width: int
    height: int
    overlay_source_image: Image.Image | None = None
    overlay_mask_image: Image.Image | None = None
    crop_coords: tuple[int, int, int, int] | None = None


QWEN_EDIT_2509_REVISION = "d3968ef930e841f4c73640fb8afa3b306a78167e"
NUNCHAKU_QWEN_EDIT_2509_REVISION = "e93a5fb77403d02a5a73c7cc8707b292c6ebc659"

QWEN_EDIT_2509_LOCAL_MODEL = (MODELS_ROOT / "diffusers/Qwen/Qwen-Image-Edit-2509").as_posix()
NUNCHAKU_QWEN_EDIT_2509_DIR = MODELS_ROOT / "nunchaku/nunchaku-ai/nunchaku-qwen-image-edit-2509"
QWEN_IMAGE_EDIT_2509_LIGHTNING_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


QWEN_IDENTITY_PROFILES: dict[str, QwenImageEditIdentityProfile] = {
    DEFAULT_QWEN_IDENTITY_PROFILE: QwenImageEditIdentityProfile(
        name=DEFAULT_QWEN_IDENTITY_PROFILE,
        base_model=QWEN_EDIT_2509_LOCAL_MODEL,
        base_repo_id="Qwen/Qwen-Image-Edit-2509",
        base_revision=QWEN_EDIT_2509_REVISION,
        dtype="bfloat16",
        nunchaku_transformer_model=NUNCHAKU_QWEN_EDIT_2509_DIR
        / "lightning-251115/svdq-fp4_r32-qwen-image-edit-2509-lightning-4steps-251115.safetensors",
        nunchaku_repo_id="nunchaku-ai/nunchaku-qwen-image-edit-2509",
        nunchaku_revision=NUNCHAKU_QWEN_EDIT_2509_REVISION,
        nunchaku_variant="fp4_r32_lightning_4steps_251115",
        default_steps=4,
        default_true_cfg_scale=1.0,
        default_guidance_scale=1.0,
        scheduler_config=QWEN_IMAGE_EDIT_2509_LIGHTNING_SCHEDULER_CONFIG,
    ),
    "nunchaku-qwen-edit-2509-fp4-r32-lightning-8step": QwenImageEditIdentityProfile(
        name="nunchaku-qwen-edit-2509-fp4-r32-lightning-8step",
        base_model=QWEN_EDIT_2509_LOCAL_MODEL,
        base_repo_id="Qwen/Qwen-Image-Edit-2509",
        base_revision=QWEN_EDIT_2509_REVISION,
        dtype="bfloat16",
        nunchaku_transformer_model=NUNCHAKU_QWEN_EDIT_2509_DIR
        / "lightning-251115/svdq-fp4_r32-qwen-image-edit-2509-lightning-8steps-251115.safetensors",
        nunchaku_repo_id="nunchaku-ai/nunchaku-qwen-image-edit-2509",
        nunchaku_revision=NUNCHAKU_QWEN_EDIT_2509_REVISION,
        nunchaku_variant="fp4_r32_lightning_8steps_251115",
        default_steps=8,
        default_true_cfg_scale=1.0,
        default_guidance_scale=1.0,
        scheduler_config=QWEN_IMAGE_EDIT_2509_LIGHTNING_SCHEDULER_CONFIG,
    ),
    "nunchaku-qwen-edit-2509-fp4-r32": QwenImageEditIdentityProfile(
        name="nunchaku-qwen-edit-2509-fp4-r32",
        base_model=QWEN_EDIT_2509_LOCAL_MODEL,
        base_repo_id="Qwen/Qwen-Image-Edit-2509",
        base_revision=QWEN_EDIT_2509_REVISION,
        dtype="bfloat16",
        nunchaku_transformer_model=NUNCHAKU_QWEN_EDIT_2509_DIR / "svdq-fp4_r32-qwen-image-edit-2509.safetensors",
        nunchaku_repo_id="nunchaku-ai/nunchaku-qwen-image-edit-2509",
        nunchaku_revision=NUNCHAKU_QWEN_EDIT_2509_REVISION,
        nunchaku_variant="fp4_r32",
        default_steps=40,
        default_true_cfg_scale=4.0,
        default_guidance_scale=1.0,
        scheduler_config=None,
    ),
    "nunchaku-qwen-edit-2509-fp4-r128": QwenImageEditIdentityProfile(
        name="nunchaku-qwen-edit-2509-fp4-r128",
        base_model=QWEN_EDIT_2509_LOCAL_MODEL,
        base_repo_id="Qwen/Qwen-Image-Edit-2509",
        base_revision=QWEN_EDIT_2509_REVISION,
        dtype="bfloat16",
        nunchaku_transformer_model=NUNCHAKU_QWEN_EDIT_2509_DIR / "svdq-fp4_r128-qwen-image-edit-2509.safetensors",
        nunchaku_repo_id="nunchaku-ai/nunchaku-qwen-image-edit-2509",
        nunchaku_revision=NUNCHAKU_QWEN_EDIT_2509_REVISION,
        nunchaku_variant="fp4_r128",
        default_steps=40,
        default_true_cfg_scale=4.0,
        default_guidance_scale=1.0,
        scheduler_config=None,
    ),
}
QWEN_IDENTITY_PROFILE_ALIASES: dict[str, str] = {
    "nunchaku-qwen-edit-2509-r32-4step": DEFAULT_QWEN_IDENTITY_PROFILE,
    "nunchaku-qwen-edit-2509-r32-8step": "nunchaku-qwen-edit-2509-fp4-r32-lightning-8step",
    "nunchaku-qwen-edit-2509-r32": "nunchaku-qwen-edit-2509-fp4-r32",
    "nunchaku-qwen-edit-2509-r128": "nunchaku-qwen-edit-2509-fp4-r128",
}


QWEN_IDENTITY_CASES: dict[str, QwenIdentityCase] = {
    "front": QwenIdentityCase(
        name="front",
        references=("front", "portrait", "side"),
        prompt=(
            "Use the input images as references for the same character. Generate a clean full-body front view "
            "character reference image, neutral standing pose, plain light studio background. Preserve the face, "
            "hair, outfit design, body proportions, age, colors, and art style. Do not add props, text, extra "
            "characters, or alternate outfits."
        ),
    ),
    "side": QwenIdentityCase(
        name="side",
        references=("side", "portrait", "front"),
        prompt=(
            "Use the input images as references for the same character. Generate a clean full-body left side view "
            "character reference image, neutral standing pose, plain light studio background. Preserve the face, "
            "hair, outfit design, body proportions, age, colors, and art style. Do not add props, text, extra "
            "characters, or alternate outfits."
        ),
    ),
    "right_profile": QwenIdentityCase(
        name="right_profile",
        references=("side", "portrait", "front"),
        prompt=(
            "Use the input images as references for the same character. Generate a clean full-body right profile "
            "view character reference image, neutral standing pose, plain light studio background. Preserve the "
            "face, hair, outfit design, body proportions, age, colors, and art style. Do not add props, text, extra "
            "characters, or alternate outfits."
        ),
    ),
    "back": QwenIdentityCase(
        name="back",
        references=("back", "portrait", "front"),
        prompt=(
            "Use the input images as references for the same character. Generate a clean full-body back view "
            "character reference image, neutral standing pose, plain light studio background. Preserve the hair, "
            "outfit design, body proportions, colors, and art style. Do not add props, text, extra characters, or "
            "alternate outfits."
        ),
    ),
    "three_quarter": QwenIdentityCase(
        name="three_quarter",
        references=("front", "portrait", "side"),
        prompt=(
            "Use the input images as references for the same character. Generate a clean full-body three-quarter "
            "front view character reference image, neutral standing pose, plain light studio background. Preserve "
            "the face, hair, outfit design, body proportions, age, colors, and art style. Do not add props, text, "
            "extra characters, or alternate outfits."
        ),
    ),
    "portrait": QwenIdentityCase(
        name="portrait",
        references=("portrait", "front", "side"),
        prompt=(
            "Use the input images as references for the same character. Generate a clean shoulders-up portrait, "
            "front-facing, plain light studio background. Preserve the face, hair, outfit details visible near the "
            "neck and shoulders, age, colors, and art style. Do not add props, text, extra characters, or alternate "
            "outfits."
        ),
        portrait_canvas=True,
    ),
}
QWEN_IDENTITY_CASE_ALIASES = {
    "right-profile": "right_profile",
    "three-quarter": "three_quarter",
}


def qwen_image_edit_identity_profile_for_name(profile_name: str) -> QwenImageEditIdentityProfile:
    profile_name = QWEN_IDENTITY_PROFILE_ALIASES.get(profile_name, profile_name)
    try:
        return QWEN_IDENTITY_PROFILES[profile_name]
    except KeyError as error:
        allowed = ", ".join(qwen_image_edit_identity_model_names())
        raise QwenImageEditIdentityError(
            f"Unknown Qwen identity model/profile {profile_name}; expected one of: {allowed}"
        ) from error


def qwen_image_edit_identity_profile_names() -> tuple[str, ...]:
    return tuple(QWEN_IDENTITY_PROFILES)


def qwen_image_edit_identity_model_names() -> tuple[str, ...]:
    return tuple(QWEN_IDENTITY_PROFILE_ALIASES) + qwen_image_edit_identity_profile_names()


def qwen_output_format_names() -> tuple[str, ...]:
    return QWEN_OUTPUT_FORMAT_ORDER


def qwen_output_resolution_names() -> tuple[str, ...]:
    return QWEN_OUTPUT_RESOLUTION_ORDER


def qwen_identity_case_names() -> tuple[str, ...]:
    return tuple(QWEN_IDENTITY_CASES)


def qwen_identity_cli_case_names() -> tuple[str, ...]:
    return tuple(QWEN_IDENTITY_CASES) + tuple(QWEN_IDENTITY_CASE_ALIASES)


def parse_qwen_identity_reference_args(reference_args: Sequence[str], base_dir: Path) -> dict[str, Path]:
    references: dict[str, Path] = {}
    for raw_reference in reference_args:
        name, separator, raw_path = raw_reference.partition("=")
        if separator != "=" or not name or not raw_path:
            raise QwenImageEditIdentityError(f"Reference must be name=path: {raw_reference}")
        if name not in QWEN_IDENTITY_REFERENCE_NAMES:
            allowed = ", ".join(sorted(QWEN_IDENTITY_REFERENCE_NAMES))
            raise QwenImageEditIdentityError(f"Unknown reference name {name}; expected one of: {allowed}")
        references[name] = resolve_existing_path(raw_path, base_dir)
    return references


def run_qwen_image_edit_identity(
    *,
    references: Mapping[str, Path],
    output_dir: Path,
    profile: QwenImageEditIdentityProfile,
    cases: Sequence[str],
    max_side: int,
    steps: int | None,
    true_cfg_scale: float | None,
    guidance_scale: float | None,
    seed: int,
    max_sequence_length: int,
    overwrite: bool,
    nunchaku_blocks_on_gpu: int | None,
    progress: StatusReporter,
    output_format: str | None = None,
    resolution: str | None = None,
) -> dict[str, Any]:
    selected_cases = _selected_cases(cases)
    return run_qwen_image_edit_cases(
        references=references,
        output_dir=output_dir,
        profile=profile,
        edit_cases=selected_cases,
        max_side=max_side,
        steps=steps,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale,
        seed=seed,
        max_sequence_length=max_sequence_length,
        candidates_per_case=1,
        overwrite=overwrite,
        nunchaku_blocks_on_gpu=nunchaku_blocks_on_gpu,
        output_format=output_format,
        resolution=resolution,
        result_kind="qwen-image-edit-identity-result",
        manifest_context=None,
        progress=progress,
    )


def run_qwen_image_edit_cases(
    *,
    references: Mapping[str, Path],
    output_dir: Path,
    profile: QwenImageEditIdentityProfile,
    edit_cases: Sequence[QwenIdentityCase],
    max_side: int,
    steps: int | None,
    true_cfg_scale: float | None,
    guidance_scale: float | None,
    seed: int,
    max_sequence_length: int,
    candidates_per_case: int,
    overwrite: bool,
    nunchaku_blocks_on_gpu: int | None,
    output_format: str | None,
    resolution: str | None,
    result_kind: str,
    manifest_context: Mapping[str, Any] | None,
    progress: StatusReporter,
) -> dict[str, Any]:
    resolved_steps = profile.default_steps if steps is None else steps
    resolved_true_cfg_scale = profile.default_true_cfg_scale if true_cfg_scale is None else true_cfg_scale
    resolved_guidance_scale = profile.default_guidance_scale if guidance_scale is None else guidance_scale
    _validate_generation_settings(
        max_side=max_side,
        steps=resolved_steps,
        true_cfg_scale=resolved_true_cfg_scale,
        guidance_scale=resolved_guidance_scale,
        max_sequence_length=max_sequence_length,
        candidates_per_case=candidates_per_case,
        nunchaku_blocks_on_gpu=nunchaku_blocks_on_gpu,
    )
    output_spec = _qwen_output_spec(output_format=output_format, resolution=resolution, max_side=max_side)
    selected_cases = tuple(edit_cases)
    _validate_edit_cases(selected_cases)
    _validate_reference_pack(references, selected_cases)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise QwenImageEditIdentityError(f"Output exists and overwrite=false: {output_dir.as_posix()}")
        shutil.rmtree(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    preflight = nvidia_smi_preflight_limit(QWEN_IDENTITY_PREFLIGHT_LIMIT_MB)
    memory_sampler = NvidiaSmiMemorySampler(preflight)
    memory_sampler.start()
    memory: dict[str, Any] | None = None
    session: QwenImageEditIdentitySession | None = None
    try:
        reference_step = _prepare_qwen_identity_references(
            references=references,
            selected_cases=selected_cases,
            max_side=max_side,
            output_spec=output_spec,
            progress=progress,
        )
        prompt_step = _encode_qwen_identity_prompts(
            profile=profile,
            reference_step=reference_step,
            true_cfg_scale=resolved_true_cfg_scale,
            max_sequence_length=max_sequence_length,
            progress=progress,
        )
        session = QwenImageEditIdentitySession(
            profile,
            nunchaku_blocks_on_gpu=nunchaku_blocks_on_gpu,
            progress=progress,
        )
        torch = session.torch
        if torch.cuda.is_available():
            progress.phase("reset qwen denoise memory stats")
            torch.cuda.reset_peak_memory_stats("cuda")
        total_start = synchronized_time(torch)
        denoise_step = _run_qwen_identity_denoise_step(
            session=session,
            reference_step=reference_step,
            prompt_step=prompt_step,
            raw_dir=raw_dir,
            max_side=max_side,
            output_spec=output_spec,
            steps=resolved_steps,
            true_cfg_scale=resolved_true_cfg_scale,
            guidance_scale=resolved_guidance_scale,
            seed=seed,
            max_sequence_length=max_sequence_length,
            candidates_per_case=candidates_per_case,
            progress=progress,
        )
        environment = session.environment()
        model_load_ms = session.model_load_ms
        session.close()
        session = None
        postprocess_step = _postprocess_qwen_identity_outputs(
            raw_outputs=denoise_step.outputs,
            output_dir=output_dir,
            output_spec=output_spec,
            progress=progress,
        )
        contact_sheet = output_dir / "contact_sheet.png"
        save_contact_sheet(
            [{"name": output["name"], "path": output["image"]["path"]} for output in postprocess_step.outputs],
            contact_sheet,
            thumb_width=192,
            max_label_chars=24,
        )
        memory = memory_sampler.stop()
        result = {
            "status": "completed",
            "kind": result_kind,
            "profile": _profile_json(profile, nunchaku_blocks_on_gpu=nunchaku_blocks_on_gpu),
            "generation": {
                "max_side": max_side,
                "num_images_per_case": candidates_per_case,
                "max_references_per_case": 3,
                "steps": resolved_steps,
                "true_cfg_scale": resolved_true_cfg_scale,
                "guidance_scale": resolved_guidance_scale,
                "negative_prompt": QWEN_IMAGE_EDIT_NEGATIVE_PROMPT if resolved_true_cfg_scale > 1.0 else None,
                "seed": seed,
                "max_sequence_length": max_sequence_length,
                "output_canvas": _output_spec_json(output_spec) if output_spec is not None else {"mode": "reference_anchor"},
            },
            "references": {name: image_asset_json(path) for name, path in sorted(references.items())},
            "outputs": postprocess_step.outputs,
            "timings_ms": {
                "prompt_encode_ms": prompt_step.elapsed_ms,
                "model_load_ms": model_load_ms,
                "denoise_ms": denoise_step.elapsed_ms,
                "postprocess_ms": postprocess_step.elapsed_ms,
                "total_ms": elapsed_ms(total_start, synchronized_time(torch)),
            },
            "memory": cuda_memory_stats(torch, "cuda") | memory,
            "environment": environment
            | {
                "postprocess": "realesrgan_anime_upscale"
                if output_spec is not None
                else "raw_image_copy",
            },
            "output": {
                "directory": output_dir.as_posix(),
                "raw": raw_dir.as_posix(),
                "contact_sheet": contact_sheet.as_posix(),
                "result": (output_dir / "result.json").as_posix(),
            },
        }
        if manifest_context is not None:
            result["plan"] = dict(manifest_context)
        write_json(output_dir / "result.json", result)
        return result
    finally:
        if session is not None:
            session.close()
        if memory is None:
            memory_sampler.stop()


def run_qwen_image_edit_inpaint_candidates(
    *,
    source_image: Path,
    mask_image: Path,
    reference_images: Sequence[QwenImageEditInpaintReference],
    output_dir: Path,
    profile: QwenImageEditIdentityProfile,
    prompt: str,
    max_side: int,
    steps: int | None,
    true_cfg_scale: float | None,
    guidance_scale: float | None,
    strength: float,
    padding_mask_crop: int | None,
    seed: int,
    max_sequence_length: int,
    candidates: int,
    overwrite: bool,
    nunchaku_blocks_on_gpu: int | None,
    result_kind: str,
    manifest_context: Mapping[str, Any] | None,
    progress: StatusReporter,
) -> dict[str, Any]:
    resolved_steps = profile.default_steps if steps is None else steps
    resolved_true_cfg_scale = profile.default_true_cfg_scale if true_cfg_scale is None else true_cfg_scale
    resolved_guidance_scale = profile.default_guidance_scale if guidance_scale is None else guidance_scale
    _validate_generation_settings(
        max_side=max_side,
        steps=resolved_steps,
        true_cfg_scale=resolved_true_cfg_scale,
        guidance_scale=resolved_guidance_scale,
        max_sequence_length=max_sequence_length,
        candidates_per_case=candidates,
        nunchaku_blocks_on_gpu=nunchaku_blocks_on_gpu,
    )
    _validate_inpaint_settings(strength=strength, padding_mask_crop=padding_mask_crop)
    source_image = source_image.resolve()
    mask_image = mask_image.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise QwenImageEditIdentityError(f"Output exists and overwrite=false: {output_dir.as_posix()}")
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    preflight = nvidia_smi_preflight_limit(QWEN_IDENTITY_PREFLIGHT_LIMIT_MB)
    memory_sampler = NvidiaSmiMemorySampler(preflight)
    memory_sampler.start()
    memory: dict[str, Any] | None = None
    session: QwenImageEditInpaintSession | None = None
    try:
        progress.phase("prepare qwen refine images")
        source = _load_reference_image(source_image, max_side=max_side)
        mask = _load_mask_image(mask_image, target_size=source.size)
        prompt_references = [source] + [
            _load_reference_image(reference.path, max_side=max_side)
            for reference in reference_images
        ]
        prompt_step = _encode_qwen_inpaint_prompt(
            profile=profile,
            prompt=prompt,
            prompt_references=prompt_references,
            true_cfg_scale=resolved_true_cfg_scale,
            max_sequence_length=max_sequence_length,
            progress=progress,
        )
        session = QwenImageEditInpaintSession(
            profile,
            nunchaku_blocks_on_gpu=nunchaku_blocks_on_gpu,
            progress=progress,
        )
        torch = session.torch
        if torch.cuda.is_available():
            progress.phase("reset qwen refine memory stats")
            torch.cuda.reset_peak_memory_stats("cuda")
        total_start = synchronized_time(torch)
        denoise_step = _run_qwen_inpaint_denoise_step(
            session=session,
            source=source,
            mask=mask,
            prompt_step=prompt_step,
            images_dir=images_dir,
            steps=resolved_steps,
            true_cfg_scale=resolved_true_cfg_scale,
            guidance_scale=resolved_guidance_scale,
            strength=strength,
            padding_mask_crop=padding_mask_crop,
            seed=seed,
            max_sequence_length=max_sequence_length,
            candidates=candidates,
            progress=progress,
        )
        contact_sheet = output_dir / "contact_sheet.png"
        save_contact_sheet(
            [{"name": output["name"], "path": output["image"]["path"]} for output in denoise_step.outputs],
            contact_sheet,
            thumb_width=192,
            max_label_chars=24,
        )
        memory = memory_sampler.stop()
        result = {
            "status": "completed",
            "kind": result_kind,
            "profile": _profile_json(profile, nunchaku_blocks_on_gpu=nunchaku_blocks_on_gpu)
            | {"pipeline": "diffusers.QwenImageEditInpaintPipeline"},
            "generation": {
                "num_images": candidates,
                "steps": resolved_steps,
                "true_cfg_scale": resolved_true_cfg_scale,
                "guidance_scale": resolved_guidance_scale,
                "negative_prompt": QWEN_IMAGE_EDIT_NEGATIVE_PROMPT if resolved_true_cfg_scale > 1.0 else None,
                "strength": strength,
                "padding_mask_crop": padding_mask_crop,
                "seed": seed,
                "max_sequence_length": max_sequence_length,
            },
            "source_image": image_asset_json(source_image),
            "mask_image": image_asset_json(mask_image),
            "mask_semantics": {
                "white": "repainted",
                "black": "preserved",
            },
            "reference_images": [
                {
                    "name": reference.name,
                    "role": reference.role,
                    "purpose": reference.purpose,
                    "image": image_asset_json(reference.path),
                }
                for reference in reference_images
            ],
            "prompt": prompt,
            "outputs": denoise_step.outputs,
            "timings_ms": {
                "prompt_encode_ms": prompt_step.elapsed_ms,
                "model_load_ms": session.model_load_ms,
                "denoise_ms": denoise_step.elapsed_ms,
                "total_ms": elapsed_ms(total_start, synchronized_time(torch)),
            },
            "memory": cuda_memory_stats(torch, "cuda") | memory,
            "environment": session.environment(),
            "output": {
                "directory": output_dir.as_posix(),
                "images": images_dir.as_posix(),
                "contact_sheet": contact_sheet.as_posix(),
                "result": (output_dir / "result.json").as_posix(),
            },
        }
        if manifest_context is not None:
            result["plan"] = dict(manifest_context)
        write_json(output_dir / "result.json", result)
        return result
    finally:
        if session is not None:
            session.close()
        if memory is None:
            memory_sampler.stop()


def _prepare_qwen_identity_references(
    *,
    references: Mapping[str, Path],
    selected_cases: Sequence[QwenIdentityCase],
    max_side: int,
    output_spec: QwenOutputSpec | None,
    progress: StatusReporter,
) -> QwenIdentityReferenceStep:
    progress.phase("prepare qwen identity references")
    used_reference_names = _used_reference_names(selected_cases)
    target_size = None if output_spec is None else (output_spec.raw_width, output_spec.raw_height)
    return QwenIdentityReferenceStep(
        cases=tuple(selected_cases),
        reference_images={
            name: _load_reference_image(references[name], max_side=max_side, target_size=target_size)
            for name in used_reference_names
        },
    )


def _encode_qwen_identity_prompts(
    *,
    profile: QwenImageEditIdentityProfile,
    reference_step: QwenIdentityReferenceStep,
    true_cfg_scale: float,
    max_sequence_length: int,
    progress: StatusReporter,
) -> QwenIdentityPromptConditioningStep:
    embeddings, encode_ms = encode_qwen_image_edit_prompts(
        profile.base_model,
        requests=[
            QwenImageEditPromptRequest(
                name=case.name,
                prompt=case.prompt,
                reference_images=tuple(reference_step.reference_images[name] for name in case.references),
            )
            for case in reference_step.cases
        ],
        dtype=profile.dtype,
        true_cfg_scale=true_cfg_scale,
        max_sequence_length=max_sequence_length,
        progress=progress,
    )
    return QwenIdentityPromptConditioningStep(embeddings=embeddings, elapsed_ms=encode_ms)


def _encode_qwen_inpaint_prompt(
    *,
    profile: QwenImageEditIdentityProfile,
    prompt: str,
    prompt_references: Sequence[Image.Image],
    true_cfg_scale: float,
    max_sequence_length: int,
    progress: StatusReporter,
) -> QwenIdentityPromptConditioningStep:
    embeddings, encode_ms = encode_qwen_image_edit_prompts(
        profile.base_model,
        requests=[
            QwenImageEditPromptRequest(
                name="refine",
                prompt=prompt,
                reference_images=tuple(prompt_references),
            )
        ],
        dtype=profile.dtype,
        true_cfg_scale=true_cfg_scale,
        max_sequence_length=max_sequence_length,
        progress=progress,
    )
    return QwenIdentityPromptConditioningStep(embeddings=embeddings, elapsed_ms=encode_ms)


def _run_qwen_identity_denoise_step(
    *,
    session: QwenImageEditIdentitySession,
    reference_step: QwenIdentityReferenceStep,
    prompt_step: QwenIdentityPromptConditioningStep,
    raw_dir: Path,
    max_side: int,
    output_spec: QwenOutputSpec | None,
    steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    seed: int,
    max_sequence_length: int,
    candidates_per_case: int,
    progress: StatusReporter,
) -> QwenIdentityDenoiseStep:
    start = synchronized_time(session.torch)
    denoised: list[QwenImageEditLatentResult] = []
    total_cases = len(reference_step.cases)
    progress.begin(total_cases * candidates_per_case * steps, "denoise qwen identity cases")
    for index, case in enumerate(reference_step.cases):
        anchor_image = reference_step.reference_images[case.references[0]]
        width, height = _case_canvas(case, anchor_image=anchor_image, max_side=max_side, output_spec=output_spec)
        for candidate_index in range(candidates_per_case):
            case_seed = seed + index * candidates_per_case + candidate_index
            output_name = _case_output_name(case.name, candidate_index, candidates_per_case)
            latents, timings = session.denoise_to_latents(
                reference_images=[reference_step.reference_images[name] for name in case.references],
                prompt_embedding=prompt_step.embeddings[case.name],
                case_name=output_name,
                width=width,
                height=height,
                steps=steps,
                true_cfg_scale=true_cfg_scale,
                guidance_scale=guidance_scale,
                seed=case_seed,
                max_sequence_length=max_sequence_length,
                case_index=index + 1,
                case_total=total_cases,
                progress=progress,
            )
            denoised.append(
                QwenImageEditLatentResult(
                    name=output_name,
                    case_name=case.name,
                    candidate_index=candidate_index,
                    latents=latents,
                    width=width,
                    height=height,
                    seed=case_seed,
                    timings_ms=timings,
                )
            )
    session.release_denoise_models_for_decode(progress)
    progress.begin(len(denoised), "decode qwen identity latents")
    outputs: list[dict[str, Any]] = []
    for denoised_result in denoised:
        case = next(case for case in reference_step.cases if case.name == denoised_result.case_name)
        image, decode_ms = session.decode_latents(
            denoised_result.latents,
            width=denoised_result.width,
            height=denoised_result.height,
            output_name=denoised_result.name,
            progress=progress,
        )
        progress.phase(f"save qwen identity case {denoised_result.name}")
        raw_image_path = raw_dir / f"{denoised_result.name}.png"
        image.save(raw_image_path)
        output = {
            "name": denoised_result.name,
            "case": case.name,
            "candidate_index": denoised_result.candidate_index,
            "seed": denoised_result.seed,
            "raw_width": denoised_result.width,
            "raw_height": denoised_result.height,
            "references": list(case.references),
            "prompt": case.prompt,
            "raw_image": image_asset_json(raw_image_path),
            "timings_ms": denoised_result.timings_ms | {"vae_decode_ms": decode_ms},
        }
        if case.edit_context is not None:
            output["edit_context"] = case.edit_context
        outputs.append(output)
        progress.step(f"decoded qwen identity case {denoised_result.name}")
    return QwenIdentityDenoiseStep(outputs=outputs, elapsed_ms=elapsed_ms(start, synchronized_time(session.torch)))


def _postprocess_qwen_identity_outputs(
    *,
    raw_outputs: Sequence[dict[str, Any]],
    output_dir: Path,
    output_spec: QwenOutputSpec | None,
    progress: StatusReporter,
) -> QwenIdentityPostprocessStep:
    start = perf_counter()
    if output_spec is None:
        progress.begin(len(raw_outputs), "copy qwen raw images")
        outputs = []
        for raw_output in raw_outputs:
            output = dict(raw_output)
            raw_path = Path(output["raw_image"]["path"])
            image_path = output_dir / f"{output['name']}.png"
            shutil.copyfile(raw_path, image_path)
            output["width"] = output["raw_width"]
            output["height"] = output["raw_height"]
            output["image"] = image_asset_json(image_path)
            output["postprocess"] = {"mode": "raw_copy"}
            outputs.append(output)
            progress.step(f"copied qwen raw image {output['name']}")
        return QwenIdentityPostprocessStep(outputs=outputs, elapsed_ms=elapsed_ms(start, perf_counter()))

    target_size = (output_spec.final_width, output_spec.final_height)
    for raw_output in raw_outputs:
        raw_size = (raw_output["raw_width"], raw_output["raw_height"])
        if raw_size != (output_spec.raw_width, output_spec.raw_height):
            raise QwenImageEditIdentityError(
                f"Raw output {raw_output['name']} has size {raw_size[0]}x{raw_size[1]}, "
                f"expected {output_spec.raw_width}x{output_spec.raw_height}"
            )
        if raw_size[0] * target_size[1] != raw_size[1] * target_size[0]:
            raise QwenImageEditIdentityError(
                f"Postprocess target {target_size[0]}x{target_size[1]} changes the aspect ratio of "
                f"raw output {raw_output['name']} ({raw_size[0]}x{raw_size[1]})"
            )

    progress.phase("load anime upscaler")
    upscaler = RealESRGANAnimeUpscaler()
    outputs = []
    for raw_output in raw_outputs:
        output = dict(raw_output)
        raw_path = Path(output["raw_image"]["path"])
        with Image.open(raw_path) as raw_image:
            upscaled = upscaler.upscale(raw_image, target_size=target_size, progress=progress)
        image_path = output_dir / f"{output['name']}.png"
        upscaled.image.save(image_path)
        output["width"] = output_spec.final_width
        output["height"] = output_spec.final_height
        output["image"] = image_asset_json(image_path)
        output["postprocess"] = {
            "mode": "anime_upscale",
            "model": REALESRGAN_ANIME_X4_MODEL_NAME,
            "model_path": upscaled.model_path.as_posix(),
            "scale": upscaled.scale,
            "device": upscaled.device,
            "output_format": output_spec.output_format,
            "resolution": output_spec.resolution,
            "natural_width": upscaled.natural_width,
            "natural_height": upscaled.natural_height,
            "target_width": upscaled.target_width,
            "target_height": upscaled.target_height,
            "elapsed_ms": upscaled.elapsed_ms,
        }
        outputs.append(output)
        progress.phase(f"saved upscaled qwen image {output['name']}")
    return QwenIdentityPostprocessStep(outputs=outputs, elapsed_ms=elapsed_ms(start, perf_counter()))


def _run_qwen_inpaint_denoise_step(
    *,
    session: QwenImageEditInpaintSession,
    source: Image.Image,
    mask: Image.Image,
    prompt_step: QwenIdentityPromptConditioningStep,
    images_dir: Path,
    steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    strength: float,
    padding_mask_crop: int | None,
    seed: int,
    max_sequence_length: int,
    candidates: int,
    progress: StatusReporter,
) -> QwenImageEditInpaintDenoiseStep:
    start = synchronized_time(session.torch)
    denoised: list[QwenImageEditLatentResult] = []
    canvas = _prepare_qwen_inpaint_canvas(session.pipeline, source, mask, padding_mask_crop)
    denoise_steps = _effective_qwen_inpaint_steps(steps=steps, strength=strength)
    progress.begin(candidates * denoise_steps, "denoise qwen refine candidates")
    for candidate_index in range(candidates):
        candidate_seed = seed + candidate_index
        output_name = f"refine_candidate_{candidate_index + 1:02d}" if candidates > 1 else "refine"
        latents, timings = session.denoise_to_latents(
            source_image=canvas.source_image,
            mask_image=canvas.mask_image,
            prompt_embedding=prompt_step.embeddings["refine"],
            output_name=output_name,
            steps=steps,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=candidate_seed,
            max_sequence_length=max_sequence_length,
            denoise_progress_steps=denoise_steps,
            candidate_index=candidate_index + 1,
            candidate_total=candidates,
            progress=progress,
        )
        denoised.append(
            QwenImageEditLatentResult(
                name=output_name,
                case_name=None,
                candidate_index=candidate_index,
                latents=latents,
                width=canvas.width,
                height=canvas.height,
                seed=candidate_seed,
                timings_ms=timings,
            )
        )
    session.release_denoise_models_for_decode(progress)
    progress.begin(len(denoised), "decode qwen refine latents")
    outputs: list[dict[str, Any]] = []
    for denoised_result in denoised:
        image, decode_ms = session.decode_latents(
            denoised_result.latents,
            width=denoised_result.width,
            height=denoised_result.height,
            output_name=denoised_result.name,
            progress=progress,
        )
        if canvas.crop_coords is not None:
            progress.phase(f"apply qwen refine crop overlay {denoised_result.name}")
            image = _apply_qwen_inpaint_crop_overlay(session.pipeline, image, canvas)
        progress.phase(f"save qwen refine candidate {denoised_result.name}")
        image_path = images_dir / f"{denoised_result.name}.png"
        image.save(image_path)
        outputs.append(
            {
                "name": denoised_result.name,
                "candidate_index": denoised_result.candidate_index,
                "seed": denoised_result.seed,
                "prompt": prompt_step.embeddings["refine"].prompt,
                "image": image_asset_json(image_path),
                "timings_ms": denoised_result.timings_ms | {"vae_decode_ms": decode_ms},
            }
        )
        progress.step(f"decoded qwen refine candidate {denoised_result.name}")
    return QwenImageEditInpaintDenoiseStep(outputs=outputs, elapsed_ms=elapsed_ms(start, synchronized_time(session.torch)))


class QwenImageEditIdentitySession:
    def __init__(
        self,
        profile: QwenImageEditIdentityProfile,
        *,
        device: str = "cuda",
        nunchaku_blocks_on_gpu: int | None,
        progress: StatusReporter,
    ) -> None:
        torch, pipeline_class = _load_qwen_image_edit_identity()
        self.torch = torch
        self.device = device
        self.nunchaku_blocks_on_gpu = nunchaku_blocks_on_gpu
        model_load_start = synchronized_time(torch)
        self.pipeline = _build_qwen_image_edit_pipeline(
            torch,
            pipeline_class,
            profile,
            device=device,
            nunchaku_blocks_on_gpu=nunchaku_blocks_on_gpu,
            progress=progress,
        )
        self.model_load_ms = elapsed_ms(model_load_start, synchronized_time(torch))

    def denoise_to_latents(
        self,
        *,
        reference_images: list[Image.Image],
        prompt_embedding: QwenImageEditPromptEmbedding,
        case_name: str,
        width: int,
        height: int,
        steps: int,
        true_cfg_scale: float,
        guidance_scale: float,
        seed: int,
        max_sequence_length: int,
        case_index: int,
        case_total: int,
        progress: StatusReporter,
    ) -> tuple[Any, dict[str, Any]]:
        start = synchronized_time(self.torch)
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        progress.phase(f"prepare qwen identity case {case_name}")
        active_embedding = _prompt_embedding_to_device(
            prompt_embedding,
            device=str(self.pipeline._execution_device),
        )
        active_guidance_scale = guidance_scale if self.pipeline.transformer.config.guidance_embeds else None
        with self.torch.inference_mode(), _qwen_edit_plus_output_vae_dimensions(
            self.pipeline,
            width=width,
            height=height,
        ):
            output = self.pipeline(
                image=reference_images,
                prompt=None,
                negative_prompt=None,
                prompt_embeds=active_embedding.prompt_embeds,
                prompt_embeds_mask=active_embedding.prompt_embeds_mask,
                negative_prompt_embeds=active_embedding.negative_prompt_embeds,
                negative_prompt_embeds_mask=active_embedding.negative_prompt_embeds_mask,
                true_cfg_scale=true_cfg_scale,
                guidance_scale=active_guidance_scale,
                height=height,
                width=width,
                num_inference_steps=steps,
                num_images_per_prompt=1,
                generator=generator,
                max_sequence_length=max_sequence_length,
                callback_on_step_end=_denoise_progress_callback(
                    progress,
                    case_name=case_name,
                    case_index=case_index,
                    case_total=case_total,
                    steps=steps,
                ),
                output_type="latent",
            )
        latents = _detach_latents_to_cpu(output.images)
        del output, active_embedding
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return latents, {"denoise_ms": elapsed_ms(start, synchronized_time(self.torch))}

    def release_denoise_models_for_decode(self, progress: StatusReporter) -> None:
        _release_qwen_denoise_models_for_decode(self.torch, self.pipeline, progress)

    def decode_latents(
        self,
        latents: Any,
        *,
        width: int,
        height: int,
        output_name: str,
        progress: StatusReporter,
    ) -> tuple[Image.Image, float]:
        return _decode_qwen_image_latents(
            self.torch,
            self.pipeline,
            latents,
            width=width,
            height=height,
            output_name=output_name,
            progress=progress,
        )

    def environment(self) -> dict[str, Any]:
        return {
            "device_report": _pipeline_device_report(self.pipeline),
            "prompt_conditioning": "precomputed_qwen_image_edit_prompt_embeds",
        }

    def close(self) -> None:
        del self.pipeline
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class QwenImageEditInpaintSession:
    def __init__(
        self,
        profile: QwenImageEditIdentityProfile,
        *,
        device: str = "cuda",
        nunchaku_blocks_on_gpu: int | None,
        progress: StatusReporter,
    ) -> None:
        torch, pipeline_class = _load_qwen_image_edit_inpaint()
        self.torch = torch
        self.device = device
        self.nunchaku_blocks_on_gpu = nunchaku_blocks_on_gpu
        model_load_start = synchronized_time(torch)
        self.pipeline = _build_qwen_image_edit_pipeline(
            torch,
            pipeline_class,
            profile,
            device=device,
            nunchaku_blocks_on_gpu=nunchaku_blocks_on_gpu,
            progress=progress,
        )
        self.model_load_ms = elapsed_ms(model_load_start, synchronized_time(torch))

    def denoise_to_latents(
        self,
        *,
        source_image: Image.Image,
        mask_image: Image.Image,
        prompt_embedding: QwenImageEditPromptEmbedding,
        output_name: str,
        steps: int,
        true_cfg_scale: float,
        guidance_scale: float,
        strength: float,
        seed: int,
        max_sequence_length: int,
        denoise_progress_steps: int,
        candidate_index: int,
        candidate_total: int,
        progress: StatusReporter,
    ) -> tuple[Any, dict[str, Any]]:
        start = synchronized_time(self.torch)
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        progress.phase(f"prepare qwen refine candidate {output_name}")
        active_embedding = _prompt_embedding_to_device(
            prompt_embedding,
            device=str(self.pipeline._execution_device),
        )
        active_guidance_scale = guidance_scale if self.pipeline.transformer.config.guidance_embeds else None
        with self.torch.inference_mode():
            output = self.pipeline(
                image=source_image,
                mask_image=mask_image,
                prompt=None,
                negative_prompt=None,
                prompt_embeds=active_embedding.prompt_embeds,
                prompt_embeds_mask=active_embedding.prompt_embeds_mask,
                negative_prompt_embeds=active_embedding.negative_prompt_embeds,
                negative_prompt_embeds_mask=active_embedding.negative_prompt_embeds_mask,
                true_cfg_scale=true_cfg_scale,
                guidance_scale=active_guidance_scale,
                strength=strength,
                padding_mask_crop=None,
                num_inference_steps=steps,
                num_images_per_prompt=1,
                generator=generator,
                max_sequence_length=max_sequence_length,
                callback_on_step_end=_denoise_progress_callback(
                    progress,
                    case_name=output_name,
                    case_index=candidate_index,
                    case_total=candidate_total,
                    steps=denoise_progress_steps,
                ),
                output_type="latent",
            )
        latents = _detach_latents_to_cpu(output.images)
        del output, active_embedding
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return latents, {"denoise_ms": elapsed_ms(start, synchronized_time(self.torch))}

    def release_denoise_models_for_decode(self, progress: StatusReporter) -> None:
        _release_qwen_denoise_models_for_decode(self.torch, self.pipeline, progress)

    def decode_latents(
        self,
        latents: Any,
        *,
        width: int,
        height: int,
        output_name: str,
        progress: StatusReporter,
    ) -> tuple[Image.Image, float]:
        return _decode_qwen_image_latents(
            self.torch,
            self.pipeline,
            latents,
            width=width,
            height=height,
            output_name=output_name,
            progress=progress,
        )

    def environment(self) -> dict[str, Any]:
        return {
            "device_report": _pipeline_device_report(self.pipeline),
            "prompt_conditioning": "precomputed_qwen_image_edit_prompt_embeds",
        }

    def close(self) -> None:
        del self.pipeline
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _selected_cases(case_names: Sequence[str]) -> list[QwenIdentityCase]:
    if not case_names:
        case_names = qwen_identity_case_names()
    selected = []
    for case_name in case_names:
        normalized_case_name = QWEN_IDENTITY_CASE_ALIASES.get(case_name, case_name)
        try:
            selected.append(QWEN_IDENTITY_CASES[normalized_case_name])
        except KeyError as error:
            allowed = ", ".join(qwen_identity_cli_case_names())
            raise QwenImageEditIdentityError(f"Unknown Qwen identity case {case_name}; expected one of: {allowed}") from error
    return selected


def _validate_reference_pack(references: Mapping[str, Path], cases: Sequence[QwenIdentityCase]) -> None:
    missing = sorted({name for case in cases for name in case.references if name not in references})
    if missing:
        raise QwenImageEditIdentityError(f"Missing Qwen identity reference(s): {', '.join(missing)}")


def _validate_edit_cases(cases: Sequence[QwenIdentityCase]) -> None:
    if not cases:
        raise QwenImageEditIdentityError("At least one Qwen edit case is required")
    names = [case.name for case in cases]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise QwenImageEditIdentityError(f"Duplicate Qwen edit case(s): {', '.join(duplicates)}")
    for case in cases:
        if len(case.references) > 3:
            raise QwenImageEditIdentityError(f"Qwen edit case {case.name} uses more than 3 references")


def _validate_generation_settings(
    *,
    max_side: int,
    steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    max_sequence_length: int,
    candidates_per_case: int,
    nunchaku_blocks_on_gpu: int | None,
) -> None:
    if max_side < 16:
        raise QwenImageEditIdentityError("max_side must be at least 16")
    if steps < 1:
        raise QwenImageEditIdentityError("steps must be at least 1")
    if true_cfg_scale <= 0:
        raise QwenImageEditIdentityError("true_cfg_scale must be positive")
    if guidance_scale <= 0:
        raise QwenImageEditIdentityError("guidance_scale must be positive")
    if max_sequence_length < 1 or max_sequence_length > 1024:
        raise QwenImageEditIdentityError("max_sequence_length must be between 1 and 1024")
    if candidates_per_case < 1:
        raise QwenImageEditIdentityError("candidates_per_case must be at least 1")
    if nunchaku_blocks_on_gpu is not None and nunchaku_blocks_on_gpu < 1:
        raise QwenImageEditIdentityError("nunchaku_blocks_on_gpu must be at least 1")


def _validate_inpaint_settings(*, strength: float, padding_mask_crop: int | None) -> None:
    if strength <= 0.0 or strength > 1.0:
        raise QwenImageEditIdentityError("strength must be greater than 0 and at most 1")
    if padding_mask_crop is not None and padding_mask_crop < 0:
        raise QwenImageEditIdentityError("padding_mask_crop must be non-negative")


def _case_output_name(case_name: str, candidate_index: int, candidates_per_case: int) -> str:
    if candidates_per_case == 1:
        return case_name
    return f"{case_name}_candidate_{candidate_index + 1:02d}"


def _used_reference_names(cases: Sequence[QwenIdentityCase]) -> tuple[str, ...]:
    names: list[str] = []
    for case in cases:
        for name in case.references:
            if name not in names:
                names.append(name)
    return tuple(names)


def _load_reference_image(
    path: Path,
    *,
    max_side: int,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    if target_size is not None:
        return _fit_image_to_canvas(rgb, target_size=target_size)
    return _fit_image_to_max_side(rgb, max_side=max_side)


def _fit_image_to_canvas(image: Image.Image, *, target_size: tuple[int, int]) -> Image.Image:
    target_width, target_height = target_size
    ratio_divisor = gcd(target_width, target_height)
    width_ratio = target_width // ratio_divisor
    height_ratio = target_height // ratio_divisor
    canvas_scale = max(
        (image.width + width_ratio - 1) // width_ratio,
        (image.height + height_ratio - 1) // height_ratio,
    )
    canvas_size = (width_ratio * canvas_scale, height_ratio * canvas_scale)
    if image.size == canvas_size:
        padded = image
    else:
        padded = Image.new("RGB", canvas_size, "white")
        padded.paste(image, ((canvas_size[0] - image.width) // 2, (canvas_size[1] - image.height) // 2))
    if padded.size == target_size:
        return padded
    if padded.width < target_width:
        canvas = Image.new("RGB", target_size, "white")
        canvas.paste(padded, ((target_width - padded.width) // 2, (target_height - padded.height) // 2))
        return canvas
    return padded.resize(target_size, Image.Resampling.LANCZOS)


def _load_mask_image(path: Path, *, target_size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        mask = image.convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    if mask.getbbox() is None:
        raise QwenImageEditIdentityError(f"Mask image has no white repaint area: {path.as_posix()}")
    return mask


def _fit_image_to_max_side(image: Image.Image, *, max_side: int) -> Image.Image:
    aligned_max_side = _align_to_multiple(max_side, 16)
    width, height = image.size
    if max(width, height) > aligned_max_side:
        if width >= height:
            content_size = (aligned_max_side, max(1, round(height * aligned_max_side / width)))
        else:
            content_size = (max(1, round(width * aligned_max_side / height)), aligned_max_side)
        image = image.resize(content_size, Image.Resampling.LANCZOS)

    canvas_width = _align_up_to_multiple(image.width, 16)
    canvas_height = _align_up_to_multiple(image.height, 16)
    if image.size == (canvas_width, canvas_height):
        return image
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    canvas.paste(image, ((canvas_width - image.width) // 2, (canvas_height - image.height) // 2))
    return canvas


def _case_canvas(
    case: QwenIdentityCase,
    *,
    anchor_image: Image.Image,
    max_side: int,
    output_spec: QwenOutputSpec | None = None,
) -> tuple[int, int]:
    if output_spec is not None:
        return output_spec.raw_width, output_spec.raw_height
    if case.portrait_canvas:
        return _canvas_from_anchor_aspect(anchor_image, max_side=max_side)
    return anchor_image.size


def _canvas_from_anchor_aspect(anchor_image: Image.Image, *, max_side: int) -> tuple[int, int]:
    canvas_long_side = _align_to_multiple(max_side, 16)
    width, height = anchor_image.size
    if width >= height:
        return canvas_long_side, _align_to_multiple(round(canvas_long_side * height / width), 16)
    return _align_to_multiple(round(canvas_long_side * width / height), 16), canvas_long_side


@contextmanager
def _qwen_edit_plus_output_vae_dimensions(pipeline: Any, *, width: int, height: int) -> Iterator[None]:
    call_globals = _qwen_edit_plus_call_globals(pipeline)
    original_calculate_dimensions = call_globals["calculate_dimensions"]
    vae_image_size = call_globals["VAE_IMAGE_SIZE"]

    def calculate_dimensions_for_output_vae(target_area: int, ratio: float) -> tuple[int, int]:
        if target_area == vae_image_size:
            return width, height
        return original_calculate_dimensions(target_area, ratio)

    call_globals["calculate_dimensions"] = calculate_dimensions_for_output_vae
    try:
        yield
    finally:
        call_globals["calculate_dimensions"] = original_calculate_dimensions


def _qwen_edit_plus_call_globals(pipeline: Any) -> dict[str, Any]:
    call = pipeline.__class__.__call__
    while "calculate_dimensions" not in call.__globals__:
        call = call.__wrapped__
    return call.__globals__


def _align_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, value // multiple * multiple)


def _align_up_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, (value + multiple - 1) // multiple * multiple)


def _qwen_output_spec(
    *,
    output_format: str | None,
    resolution: str | None,
    max_side: int,
) -> QwenOutputSpec | None:
    if output_format is None and resolution is None:
        return None
    if output_format is None or resolution is None:
        raise QwenImageEditIdentityError("output_format and resolution must be provided together")
    if output_format not in QWEN_OUTPUT_FORMATS:
        allowed = ", ".join(QWEN_OUTPUT_FORMAT_ORDER)
        raise QwenImageEditIdentityError(f"Unknown output_format {output_format}; expected one of: {allowed}")
    if resolution not in QWEN_OUTPUT_RESOLUTIONS:
        allowed = ", ".join(QWEN_OUTPUT_RESOLUTION_ORDER)
        raise QwenImageEditIdentityError(f"Unknown resolution {resolution}; expected one of: {allowed}")
    raw_width, raw_height = _size_for_output_format(
        output_format,
        long_side=max_side,
        align_to_multiple=16,
    )
    final_width, final_height = _size_for_output_format(
        output_format,
        long_side=QWEN_OUTPUT_RESOLUTIONS[resolution],
        align_to_multiple=None,
    )
    return QwenOutputSpec(
        output_format=output_format,
        resolution=resolution,
        raw_width=raw_width,
        raw_height=raw_height,
        final_width=final_width,
        final_height=final_height,
    )


def _size_for_output_format(
    output_format: str,
    *,
    long_side: int,
    align_to_multiple: int | None,
) -> tuple[int, int]:
    width_ratio, height_ratio = QWEN_OUTPUT_FORMATS[output_format]
    scale = long_side // max(width_ratio, height_ratio)
    if align_to_multiple is not None:
        scale_multiple = lcm(
            align_to_multiple // gcd(width_ratio, align_to_multiple),
            align_to_multiple // gcd(height_ratio, align_to_multiple),
        )
        scale = scale // scale_multiple * scale_multiple
    if scale < 1:
        raise QwenImageEditIdentityError(
            f"Output format {output_format} does not fit within a {long_side}px long side"
        )
    return width_ratio * scale, height_ratio * scale


def _output_spec_json(output_spec: QwenOutputSpec) -> dict[str, Any]:
    return {
        "mode": "explicit_output_format",
        "output_format": output_spec.output_format,
        "resolution": output_spec.resolution,
        "raw_width": output_spec.raw_width,
        "raw_height": output_spec.raw_height,
        "final_width": output_spec.final_width,
        "final_height": output_spec.final_height,
        "upscale_model": REALESRGAN_ANIME_X4_MODEL_NAME,
    }


def _qwen_inpaint_canvas_size(pipeline: Any, image: Image.Image) -> tuple[int, int]:
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_inpaint import calculate_dimensions

    calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image.size[0] / image.size[1])
    multiple_of = pipeline.vae_scale_factor * 2
    width = calculated_width // multiple_of * multiple_of
    height = calculated_height // multiple_of * multiple_of
    return width, height


def _prepare_qwen_inpaint_canvas(
    pipeline: Any,
    source: Image.Image,
    mask: Image.Image,
    padding_mask_crop: int | None,
) -> QwenImageEditInpaintCanvas:
    width, height = _qwen_inpaint_canvas_size(pipeline, source)
    if padding_mask_crop is None:
        return QwenImageEditInpaintCanvas(
            source_image=source,
            mask_image=mask,
            width=width,
            height=height,
        )

    full_source = pipeline.image_processor.resize(source, height, width)
    full_mask = mask.resize((width, height), Image.Resampling.NEAREST)
    crop_coords = pipeline.mask_processor.get_crop_region(full_mask, width, height, pad=padding_mask_crop)
    cropped_source = full_source.crop(crop_coords)
    cropped_mask = full_mask.crop(crop_coords)
    crop_width, crop_height = _qwen_inpaint_canvas_size(pipeline, cropped_source)
    return QwenImageEditInpaintCanvas(
        source_image=cropped_source,
        mask_image=cropped_mask,
        width=crop_width,
        height=crop_height,
        overlay_source_image=full_source,
        overlay_mask_image=full_mask,
        crop_coords=crop_coords,
    )


def _apply_qwen_inpaint_crop_overlay(
    pipeline: Any,
    image: Image.Image,
    canvas: QwenImageEditInpaintCanvas,
) -> Image.Image:
    return pipeline.image_processor.apply_overlay(
        canvas.overlay_mask_image,
        canvas.overlay_source_image,
        image,
        canvas.crop_coords,
    )


def _effective_qwen_inpaint_steps(*, steps: int, strength: float) -> int:
    init_timestep = min(steps * strength, steps)
    t_start = int(max(steps - init_timestep, 0))
    return steps - t_start


def _detach_latents_to_cpu(latents: Any) -> Any:
    if isinstance(latents, (list, tuple)):
        if len(latents) != 1:
            raise QwenImageEditIdentityError("Expected exactly one Qwen latent output")
        latents = latents[0]
    return latents.detach().cpu()


def _release_qwen_denoise_models_for_decode(torch: Any, pipeline: Any, progress: StatusReporter) -> None:
    progress.phase("release qwen transformer before vae decode")
    pipeline.transformer.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _decode_qwen_image_latents(
    torch: Any,
    pipeline: Any,
    latents: Any,
    *,
    width: int,
    height: int,
    output_name: str,
    progress: StatusReporter,
) -> tuple[Image.Image, float]:
    progress.phase(f"decode qwen latents {output_name}")
    start = synchronized_time(torch)
    device = _module_device(pipeline.vae)
    with torch.inference_mode():
        latents = latents.to(device=device, dtype=pipeline.vae.dtype)
        latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
        latents = latents.to(pipeline.vae.dtype)
        latents_mean = (
            torch.tensor(pipeline.vae.config.latents_mean)
            .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipeline.vae.config.latents_std).view(
            1, pipeline.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        decoded = pipeline.vae.decode(latents, return_dict=False)[0][:, :, 0]
        images = pipeline.image_processor.postprocess(decoded, output_type="pil")
    del latents, decoded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return images[0], elapsed_ms(start, synchronized_time(torch))


def _module_device(module: Any) -> Any:
    try:
        return next(module.parameters()).device
    except StopIteration as error:
        raise QwenImageEditIdentityError("Cannot determine Qwen VAE device") from error


def _profile_json(
    profile: QwenImageEditIdentityProfile,
    *,
    nunchaku_blocks_on_gpu: int | None,
) -> dict[str, Any]:
    models: dict[str, Any] = {
        "base": {
            "repo_id": profile.base_repo_id,
            "revision": profile.base_revision,
            "path": profile.base_model,
        },
    }
    models["nunchaku_transformer"] = {
        "repo_id": profile.nunchaku_repo_id,
        "revision": profile.nunchaku_revision,
        "variant": profile.nunchaku_variant,
        "path": profile.nunchaku_transformer_model.as_posix(),
    }
    return {
        "name": profile.name,
        "dtype": profile.dtype,
        "load_strategy": "nunchaku-qwen-image-edit-2509",
        "scheduler": "lightning" if profile.scheduler_config is not None else "default",
        "prompt_conditioning": "precomputed_qwen_image_edit_prompt_embeds",
        "default_steps": profile.default_steps,
        "default_true_cfg_scale": profile.default_true_cfg_scale,
        "default_guidance_scale": profile.default_guidance_scale,
        "nunchaku_blocks_on_gpu": nunchaku_blocks_on_gpu,
        "local_files_only": profile.local_files_only,
        "models": models,
    }


def _build_qwen_image_edit_pipeline(
    torch: Any,
    pipeline_class: Any,
    profile: QwenImageEditIdentityProfile,
    *,
    device: str,
    nunchaku_blocks_on_gpu: int | None,
    progress: StatusReporter,
) -> Any:
    progress.begin(3, "load qwen denoise models")
    torch_dtype = resolve_torch_dtype(torch, profile.dtype, auto_value=None)
    progress.phase("load qwen nunchaku transformer")
    transformer = _load_nunchaku_qwen_transformer(profile.nunchaku_transformer_model, torch_dtype=torch_dtype)
    progress.step("loaded qwen nunchaku transformer")
    pipeline_kwargs = {
        "transformer": transformer,
        "text_encoder": None,
        "tokenizer": None,
        "processor": None,
        "torch_dtype": torch_dtype,
        "local_files_only": profile.local_files_only,
    }
    if profile.scheduler_config is not None:
        pipeline_kwargs["scheduler"] = _build_qwen_scheduler(profile.scheduler_config)
    progress.phase("load qwen image edit pipeline")
    pipeline = pipeline_class.from_pretrained(profile.base_model, **pipeline_kwargs)
    pipeline.set_progress_bar_config(disable=True)
    progress.step("loaded qwen image edit pipeline")
    if nunchaku_blocks_on_gpu is None:
        progress.phase(f"move qwen denoise pipeline to {device}")
        try:
            pipeline.to(device)
        except RuntimeError as exc:
            raise QwenImageEditIdentityDependencyError(
                f"Failed to move Qwen Image Edit denoise pipeline to {device}: {exc}"
            ) from exc
    else:
        progress.phase("configure qwen nunchaku layer offload")
        transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=nunchaku_blocks_on_gpu)
        pipeline._exclude_from_cpu_offload.append("transformer")
        pipeline.enable_sequential_cpu_offload()
    progress.step("qwen denoise pipeline ready")
    return pipeline


def _denoise_progress_callback(
    progress: StatusReporter,
    *,
    case_name: str,
    case_index: int,
    case_total: int,
    steps: int,
) -> Any:
    def callback(_pipeline: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        progress.step(f"denoised {case_name} case {case_index}/{case_total} step {step + 1}/{steps}")
        return callback_kwargs

    return callback


def _prompt_embedding_to_device(
    embedding: QwenImageEditPromptEmbedding,
    *,
    device: str,
) -> QwenImageEditPromptEmbedding:
    return QwenImageEditPromptEmbedding(
        name=embedding.name,
        prompt=embedding.prompt,
        prompt_embeds=embedding.prompt_embeds.to(device),
        prompt_embeds_mask=tensor_to_device(embedding.prompt_embeds_mask, device=device),
        negative_prompt_embeds=tensor_to_device(embedding.negative_prompt_embeds, device=device),
        negative_prompt_embeds_mask=tensor_to_device(embedding.negative_prompt_embeds_mask, device=device),
    )


def _load_nunchaku_qwen_transformer(transformer_model: Path, *, torch_dtype: Any) -> Any:
    try:
        from nunchaku import NunchakuQwenImageTransformer2DModel
    except ImportError as exc:
        raise QwenImageEditIdentityDependencyError("Qwen identity Nunchaku profiles require nunchaku") from exc
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
        transformer_model.resolve().as_posix(),
        torch_dtype=torch_dtype,
        offload=False,
    )
    _install_nunchaku_qwen_txt_seq_lens_forward(transformer)
    return transformer


def _install_nunchaku_qwen_txt_seq_lens_forward(transformer: Any) -> None:
    original_forward = transformer.forward
    original_pos_embed_forward = transformer.pos_embed.forward

    def pos_embed_forward_without_txt_seq_warning(
        self: Any,
        video_fhw: Any,
        txt_seq_lens: list[int] | None = None,
        device: Any = None,
        max_txt_seq_len: Any = None,
    ) -> Any:
        if max_txt_seq_len is None and txt_seq_lens is not None:
            max_txt_seq_len = max(txt_seq_lens) if isinstance(txt_seq_lens, list) else txt_seq_lens
        return original_pos_embed_forward(
            video_fhw,
            txt_seq_lens=None,
            device=device,
            max_txt_seq_len=max_txt_seq_len,
        )

    def forward_with_txt_seq_lens(
        self: Any,
        *,
        hidden_states: Any,
        encoder_hidden_states: Any = None,
        encoder_hidden_states_mask: Any = None,
        timestep: Any = None,
        img_shapes: Any = None,
        txt_seq_lens: list[int] | None = None,
        guidance: Any = None,
        attention_kwargs: Any = None,
        controlnet_block_samples: Any = None,
        return_dict: bool = True,
    ) -> Any:
        if txt_seq_lens is None and encoder_hidden_states is not None:
            txt_seq_lens = [int(encoder_hidden_states.shape[1])]
        return original_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            guidance=guidance,
            attention_kwargs=attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            return_dict=return_dict,
        )

    transformer.pos_embed.forward = MethodType(pos_embed_forward_without_txt_seq_warning, transformer.pos_embed)
    transformer.forward = MethodType(forward_with_txt_seq_lens, transformer)


def _build_qwen_scheduler(scheduler_config: Mapping[str, Any]) -> Any:
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
    except ImportError as exc:
        raise QwenImageEditIdentityDependencyError(
            "Qwen identity lightning profiles require diffusers FlowMatchEulerDiscreteScheduler"
        ) from exc
    return FlowMatchEulerDiscreteScheduler.from_config(dict(scheduler_config))


def _load_qwen_image_edit_identity() -> tuple[Any, Any]:
    try:
        import torch
        from diffusers import QwenImageEditPlusPipeline
        from diffusers.utils import logging as diffusers_logging
    except ImportError as exc:
        raise QwenImageEditIdentityDependencyError(
            "Qwen identity generation requires `pip install -e .[generation]`"
        ) from exc
    diffusers_logging.disable_progress_bar()
    return torch, QwenImageEditPlusPipeline


def _load_qwen_image_edit_inpaint() -> tuple[Any, Any]:
    try:
        import torch
        from diffusers import QwenImageEditInpaintPipeline
        from diffusers.utils import logging as diffusers_logging
    except ImportError as exc:
        raise QwenImageEditIdentityDependencyError(
            "Qwen refine generation requires `pip install -e .[generation]`"
        ) from exc
    diffusers_logging.disable_progress_bar()
    return torch, QwenImageEditInpaintPipeline


def _pipeline_device_report(pipeline: Any) -> dict[str, Any]:
    components = {}
    for name in ("transformer", "vae"):
        component = getattr(pipeline, name, None)
        if component is not None:
            components[name] = module_device_report(component)
    return {
        "pipeline_class": type(pipeline).__qualname__,
        "execution_device": str(getattr(pipeline, "_execution_device", "")),
        "model_cpu_offload_seq": getattr(pipeline, "model_cpu_offload_seq", ""),
        "components": components,
    }
