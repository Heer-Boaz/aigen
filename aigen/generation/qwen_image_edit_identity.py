from __future__ import annotations

import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import gc
from math import gcd, log, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import MethodType
from typing import Any

from PIL import Image

from aigen.generation.image_generation_requests import (
    ImageGenerationCaseRequest,
    ImageGenerationOutputRequest,
)
from aigen.generation.image_upscale import size_for_long_side
from aigen.generation.qwen_image_edit_lightx2v import (
    LIGHTX2V_QWEN_EDIT_2511_PROFILE,
    QWEN_IMAGE_EDIT_LIGHTX2V_PROFILES,
    QwenImageEditLightX2VError,
    QwenImageEditLightX2VProfile,
    lightx2v_profile_json,
    run_lightx2v_qwen_image_edit,
)
from aigen.generation.prompt_encoding import tensor_to_device
from aigen.lora_weights import LoraLoadSpec
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
from aigen.generation.vosr_backend import (
    VOSR_MODEL_NAME,
    VOSR_POSTPROCESS_NAME,
    upscale_files_with_vosr,
)
from aigen.image_assets import image_asset_json
from aigen.image_dimensions import closest_aspect_match
from aigen.keyframe_image_ops import exact_outside_mask_diff, save_contact_sheet
from aigen.keyframe_memory import NvidiaSmiMemorySampler, nvidia_smi_preflight_limit
from aigen.manifest_io import write_json
from aigen.progress import StatusReporter
from aigen.runtime_profiles import MODELS_ROOT


DEFAULT_QWEN_INPAINT_PROFILE = "nunchaku-qwen-edit-2509-fp4-r32-lightning-4step"
DEFAULT_QWEN_IDENTITY_PROFILE = LIGHTX2V_QWEN_EDIT_2511_PROFILE
DEFAULT_QWEN_IDENTITY_MAX_SIDE = 1792
DEFAULT_QWEN_IDENTITY_SEED = 0
DEFAULT_QWEN_IDENTITY_MAX_SEQUENCE_LENGTH = 512
QWEN_IDENTITY_PREFLIGHT_LIMIT_MB = 4096
QWEN_IMAGE_EDIT_2511_NATIVE_PIXELS = 1152 * 1536
QWEN_IMAGE_EDIT_2511_MAX_CUSTOM_SIDE = 1664
QWEN_IMAGE_EDIT_2511_NATIVE_CANVASES = {
    (16, 9): (1664, 928),
    (9, 16): (928, 1664),
    (1, 1): (1328, 1328),
    (4, 3): (1472, 1104),
    (3, 4): (1104, 1472),
    (3, 2): (1584, 1056),
    (2, 3): (1056, 1584),
}
DEFAULT_QWEN_ASPECT_RATIO = (3, 4)
DEFAULT_QWEN_UPSCALE_LONG_SIDE = 2048


class QwenImageEditIdentityError(RuntimeError):
    pass


class QwenImageEditIdentityDependencyError(QwenImageEditIdentityError):
    pass


def qwen_image_edit_2511_native_canvas_size(
    aspect_ratio: tuple[int, int],
    *,
    closest: bool,
) -> tuple[int, int]:
    if aspect_ratio in QWEN_IMAGE_EDIT_2511_NATIVE_CANVASES:
        return QWEN_IMAGE_EDIT_2511_NATIVE_CANVASES[aspect_ratio]
    if closest:
        return closest_aspect_match(
            aspect_ratio,
            tuple(QWEN_IMAGE_EDIT_2511_NATIVE_CANVASES.values()),
        )
    supported = ", ".join(
        f"{width}:{height}" for width, height in QWEN_IMAGE_EDIT_2511_NATIVE_CANVASES
    )
    raise QwenImageEditIdentityError(
        f"Qwen-Image-Edit-2511 has no native {aspect_ratio[0]}:{aspect_ratio[1]} canvas; "
        f"use --width/--height or one of: {supported}"
    )


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


QwenImageEditProfile = QwenImageEditIdentityProfile | QwenImageEditLightX2VProfile


@dataclass(frozen=True)
class QwenIdentityCase:
    name: str
    references: tuple[str, ...]
    prompt: str
    source_images: tuple[str, ...] = ()
    guides: tuple[str, ...] = ()
    controls: tuple[str, ...] = ()
    edit_context: dict[str, Any] | None = None
    seeds: tuple[int, ...] | None = None


@dataclass(frozen=True)
class QwenIdentityReferenceStep:
    cases: tuple[QwenIdentityCase, ...]
    source_images: dict[str, Image.Image]
    reference_images: dict[str, Image.Image]
    guide_images: dict[str, Image.Image]
    control_images: dict[tuple[str, str], Image.Image]
    canvas_sizes: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class QwenControlImage:
    image: Image.Image
    content_box: tuple[int, int, int, int] | None = None


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
class QwenImageEditInpaintDenoiseStep:
    outputs: list[dict[str, Any]]
    elapsed_ms: float


@dataclass(frozen=True)
class QwenImageEditInpaintCanvas:
    source_image: Image.Image
    mask_image: Image.Image
    width: int
    height: int
    overlay_source_image: Image.Image
    overlay_mask_image: Image.Image
    crop_coords: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class QwenImageEditInpaintReferenceLatents:
    packed_latents: Any
    shapes: tuple[tuple[int, int, int], ...]


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


QWEN_IDENTITY_PROFILES: dict[str, QwenImageEditProfile] = {
    DEFAULT_QWEN_INPAINT_PROFILE: QwenImageEditIdentityProfile(
        name=DEFAULT_QWEN_INPAINT_PROFILE,
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
QWEN_IDENTITY_PROFILES.update(QWEN_IMAGE_EDIT_LIGHTX2V_PROFILES)
QWEN_IDENTITY_PROFILE_ALIASES: dict[str, str] = {
    "nunchaku-qwen-edit-2509-r32-4step": DEFAULT_QWEN_INPAINT_PROFILE,
    "nunchaku-qwen-edit-2509-r32-8step": "nunchaku-qwen-edit-2509-fp4-r32-lightning-8step",
    "nunchaku-qwen-edit-2509-r32": "nunchaku-qwen-edit-2509-fp4-r32",
    "nunchaku-qwen-edit-2509-r128": "nunchaku-qwen-edit-2509-fp4-r128",
}


def qwen_image_edit_identity_profile_for_name(profile_name: str) -> QwenImageEditProfile:
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


def qwen_image_edit_inpaint_profile_names() -> tuple[str, ...]:
    return tuple(
        name
        for name, profile in QWEN_IDENTITY_PROFILES.items()
        if isinstance(profile, QwenImageEditIdentityProfile)
    )


def qwen_image_edit_inpaint_model_names() -> tuple[str, ...]:
    return tuple(QWEN_IDENTITY_PROFILE_ALIASES) + qwen_image_edit_inpaint_profile_names()


def qwen_image_edit_identity_model_names() -> tuple[str, ...]:
    return tuple(QWEN_IDENTITY_PROFILE_ALIASES) + qwen_image_edit_identity_profile_names()


def _native_canvas_pixels(profile: QwenImageEditProfile) -> int | None:
    if isinstance(profile, QwenImageEditLightX2VProfile):
        return QWEN_IMAGE_EDIT_2511_NATIVE_PIXELS
    return None


def run_qwen_image_edit_cases(
    *,
    source_images: Mapping[str, Path],
    references: Mapping[str, Path],
    guides: Mapping[str, Path],
    controls: Mapping[str, QwenControlImage],
    output_dir: Path,
    profile: QwenImageEditProfile,
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
    aspect_ratio: tuple[int, int] | None,
    canvas_size: tuple[int, int] | None,
    upscale_long_side: int,
    postprocess: str,
    result_kind: str,
    manifest_context: Mapping[str, Any] | None,
    progress: StatusReporter,
    loras: Sequence[LoraLoadSpec] = (),
) -> dict[str, Any]:
    resolved_loras = tuple(loras)
    if resolved_loras:
        if not isinstance(profile, QwenImageEditLightX2VProfile):
            raise QwenImageEditIdentityError(
                "LoRA loading is supported only by the Qwen-Image-Edit-2511 LightX2V backend"
            )
        from aigen.lora_weights import QWEN_IMAGE_ARCHITECTURE, inspect_lora_weights

        checked_loras = []
        for lora in resolved_loras:
            lora_info = inspect_lora_weights(lora.path)
            if lora_info.architecture != QWEN_IMAGE_ARCHITECTURE:
                raise QwenImageEditIdentityError(
                    f"Qwen-Image-Edit-2511 cannot load a {lora_info.architecture} LoRA: "
                    f"{lora_info.path}"
                )
            checked_loras.append(LoraLoadSpec(path=lora_info.path, weight=lora.weight))
        resolved_loras = tuple(checked_loras)
    _validate_qwen_canvas_size(canvas_size, aspect_ratio=aspect_ratio)
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
    native_canvas_pixels = _native_canvas_pixels(profile)
    if upscale_long_side < 1:
        raise QwenImageEditIdentityError("upscale_long_side must be at least 1")
    selected_cases = tuple(edit_cases)
    _validate_edit_cases(selected_cases)
    _validate_image_inputs(source_images, references, guides, controls, selected_cases)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise QwenImageEditIdentityError(f"Output exists and overwrite=false: {output_dir.as_posix()}")
        shutil.rmtree(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    control_paths = _materialize_qwen_control_images(controls, output_dir)

    if isinstance(profile, QwenImageEditLightX2VProfile):
        if nunchaku_blocks_on_gpu is not None:
            raise QwenImageEditIdentityError(
                "--nunchaku-blocks-on-gpu does not apply to the Qwen-Image-Edit-2511 LightX2V backend"
            )
        try:
            return _run_qwen_image_edit_cases_lightx2v(
                source_images=source_images,
                references=references,
                guides=guides,
                controls=controls,
                control_paths=control_paths,
                output_dir=output_dir,
                raw_dir=raw_dir,
                profile=profile,
                selected_cases=selected_cases,
                max_side=max_side,
                native_canvas_pixels=native_canvas_pixels,
                steps=resolved_steps,
                true_cfg_scale=resolved_true_cfg_scale,
                guidance_scale=resolved_guidance_scale,
                seed=seed,
                max_sequence_length=max_sequence_length,
                candidates_per_case=candidates_per_case,
                aspect_ratio=aspect_ratio,
                canvas_size=canvas_size,
                upscale_long_side=upscale_long_side,
                postprocess=postprocess,
                result_kind=result_kind,
                manifest_context=manifest_context,
                loras=resolved_loras,
                progress=progress,
            )
        except QwenImageEditLightX2VError as error:
            raise QwenImageEditIdentityError(str(error)) from error

    preflight = nvidia_smi_preflight_limit(QWEN_IDENTITY_PREFLIGHT_LIMIT_MB)
    memory_sampler = NvidiaSmiMemorySampler(preflight)
    memory_sampler.start()
    memory: dict[str, Any] | None = None
    session: QwenImageEditIdentitySession | None = None
    try:
        reference_step = _prepare_qwen_identity_references(
            source_images=source_images,
            references=references,
            guides=guides,
            controls=controls,
            selected_cases=selected_cases,
            max_side=max_side,
            native_canvas_pixels=native_canvas_pixels,
            aspect_ratio=aspect_ratio,
            canvas_size=canvas_size,
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
            upscale_long_side=upscale_long_side,
            postprocess=postprocess,
            progress=progress,
        )
        contact_sheet = output_dir / "contact_sheet.png"
        candidate_columns = max(
            len(case.seeds) if case.seeds is not None else candidates_per_case
            for case in selected_cases
        )
        save_contact_sheet(
            [{"name": output["name"], "path": output["image"]["path"]} for output in postprocess_step.outputs],
            contact_sheet,
            thumb_width=192,
            max_label_chars=24,
            max_columns=candidate_columns if candidate_columns > 1 else 8,
        )
        memory = memory_sampler.stop()
        result = {
            "status": "completed",
            "kind": result_kind,
            "profile": _profile_json(profile, nunchaku_blocks_on_gpu=nunchaku_blocks_on_gpu),
            "generation": {
                "max_side": max_side,
                "num_images_per_case": candidates_per_case,
                "max_source_images_per_case": max(len(case.source_images) for case in selected_cases),
                "max_references_per_case": max(len(case.references) for case in selected_cases),
                "max_guides_per_case": max(len(case.guides) for case in selected_cases),
                "max_controls_per_case": max(len(case.controls) for case in selected_cases),
                "max_inputs_per_case": max(
                    len(case.source_images) + len(case.references) + len(case.guides) + len(case.controls)
                    for case in selected_cases
                ),
                "steps": resolved_steps,
                "true_cfg_scale": resolved_true_cfg_scale,
                "guidance_scale": resolved_guidance_scale,
                "negative_prompt": QWEN_IMAGE_EDIT_NEGATIVE_PROMPT if resolved_true_cfg_scale > 1.0 else None,
                "seed": seed,
                "max_sequence_length": max_sequence_length,
                "native_canvas_pixels": native_canvas_pixels,
                "output_canvas": _output_canvas_json(
                    aspect_ratio=aspect_ratio,
                    canvas_size=canvas_size,
                    native_canvas_pixels=native_canvas_pixels,
                    upscale_long_side=upscale_long_side,
                ),
            },
            "source_images": {name: image_asset_json(path) for name, path in sorted(source_images.items())},
            "references": {name: image_asset_json(path) for name, path in sorted(references.items())},
            "guides": {name: image_asset_json(path) for name, path in sorted(guides.items())},
            "controls": {name: image_asset_json(path) for name, path in sorted(control_paths.items())},
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
                "postprocess": VOSR_POSTPROCESS_NAME,
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


def _run_qwen_image_edit_cases_lightx2v(
    *,
    source_images: Mapping[str, Path],
    references: Mapping[str, Path],
    guides: Mapping[str, Path],
    controls: Mapping[str, QwenControlImage],
    control_paths: Mapping[str, Path],
    output_dir: Path,
    raw_dir: Path,
    profile: QwenImageEditLightX2VProfile,
    selected_cases: tuple[QwenIdentityCase, ...],
    max_side: int,
    native_canvas_pixels: int | None,
    steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    seed: int,
    max_sequence_length: int,
    candidates_per_case: int,
    aspect_ratio: tuple[int, int] | None,
    canvas_size: tuple[int, int] | None,
    upscale_long_side: int,
    postprocess: str,
    result_kind: str,
    manifest_context: Mapping[str, Any] | None,
    loras: tuple[LoraLoadSpec, ...],
    progress: StatusReporter,
) -> dict[str, Any]:
    preflight = nvidia_smi_preflight_limit(QWEN_IDENTITY_PREFLIGHT_LIMIT_MB)
    memory_sampler = NvidiaSmiMemorySampler(preflight)
    memory_sampler.start()
    memory: dict[str, Any] | None = None
    try:
        reference_step = _prepare_qwen_identity_references(
            source_images=source_images,
            references=references,
            guides=guides,
            controls=controls,
            selected_cases=selected_cases,
            max_side=max_side,
            native_canvas_pixels=native_canvas_pixels,
            aspect_ratio=aspect_ratio,
            canvas_size=canvas_size,
            progress=progress,
        )
        with TemporaryDirectory(prefix="aigen-qwen-2511-inputs-") as temporary_dir:
            staged_paths = _stage_lightx2v_input_images(reference_step, Path(temporary_dir))
            requests = []
            for case_index, case in enumerate(selected_cases):
                case_seeds = case.seeds or tuple(
                    seed + case_index * candidates_per_case + candidate_index
                    for candidate_index in range(candidates_per_case)
                )
                width, height = reference_step.canvas_sizes[case.name]
                requests.append(
                    ImageGenerationCaseRequest(
                        name=case.name,
                        prompt=case.prompt,
                        image_paths=tuple(staged_paths[id(image)] for image in _case_input_images(reference_step, case)),
                        width=width,
                        height=height,
                        outputs=tuple(
                            ImageGenerationOutputRequest(
                                name=_case_output_name(case.name, candidate_index, len(case_seeds)),
                                seed=case_seed,
                                path=raw_dir
                                / f"{_case_output_name(case.name, candidate_index, len(case_seeds))}.png",
                            )
                            for candidate_index, case_seed in enumerate(case_seeds)
                        ),
                    )
                )
            backend_result = run_lightx2v_qwen_image_edit(
                profile=profile,
                cases=tuple(requests),
                steps=steps,
                true_cfg_scale=true_cfg_scale,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                loras=loras,
                progress=progress,
            )

        cases_by_name = {case.name: case for case in selected_cases}
        raw_outputs = []
        for backend_output in backend_result.outputs:
            case = cases_by_name[backend_output["case"]]
            case_request = next(request for request in requests if request.name == case.name)
            raw_image_path = Path(backend_output["path"])
            raw_output = {
                "name": backend_output["name"],
                "case": case.name,
                "candidate_index": next(
                    index
                    for index, output in enumerate(case_request.outputs)
                    if output.name == backend_output["name"]
                ),
                "seed": backend_output["seed"],
                "raw_width": backend_output["width"],
                "raw_height": backend_output["height"],
                "source_images": list(case.source_images),
                "references": list(case.references),
                "guides": list(case.guides),
                "controls": list(case.controls),
                "prompt": case.prompt,
                "raw_image": image_asset_json(raw_image_path),
                "timings_ms": {
                    "denoise_ms": backend_output["denoise_ms"],
                    "vae_decode_ms": backend_output["vae_decode_ms"],
                },
            }
            if case.edit_context is not None:
                raw_output["edit_context"] = case.edit_context
            raw_outputs.append(raw_output)

        postprocess_step = _postprocess_qwen_identity_outputs(
            raw_outputs=raw_outputs,
            output_dir=output_dir,
            upscale_long_side=upscale_long_side,
            postprocess=postprocess,
            progress=progress,
        )
        contact_sheet = output_dir / "contact_sheet.png"
        candidate_columns = max(
            len(case.seeds) if case.seeds is not None else candidates_per_case
            for case in selected_cases
        )
        save_contact_sheet(
            [{"name": output["name"], "path": output["image"]["path"]} for output in postprocess_step.outputs],
            contact_sheet,
            thumb_width=192,
            max_label_chars=24,
            max_columns=candidate_columns if candidate_columns > 1 else 8,
        )
        memory = memory_sampler.stop() | backend_result.memory
        result = {
            "status": "completed",
            "kind": result_kind,
            "profile": lightx2v_profile_json(profile),
            "generation": {
                "max_side": max_side,
                "num_images_per_case": candidates_per_case,
                "max_source_images_per_case": max(len(case.source_images) for case in selected_cases),
                "max_references_per_case": max(len(case.references) for case in selected_cases),
                "max_guides_per_case": max(len(case.guides) for case in selected_cases),
                "max_controls_per_case": max(len(case.controls) for case in selected_cases),
                "max_inputs_per_case": max(
                    len(case.source_images) + len(case.references) + len(case.guides) + len(case.controls)
                    for case in selected_cases
                ),
                "steps": steps,
                "true_cfg_scale": true_cfg_scale,
                "guidance_scale": guidance_scale,
                "negative_prompt": None,
                "seed": seed,
                "max_sequence_length": max_sequence_length,
                "native_canvas_pixels": native_canvas_pixels,
                "output_canvas": _output_canvas_json(
                    aspect_ratio=aspect_ratio,
                    canvas_size=canvas_size,
                    native_canvas_pixels=native_canvas_pixels,
                    upscale_long_side=upscale_long_side,
                ),
            },
            "source_images": {name: image_asset_json(path) for name, path in sorted(source_images.items())},
            "references": {name: image_asset_json(path) for name, path in sorted(references.items())},
            "guides": {name: image_asset_json(path) for name, path in sorted(guides.items())},
            "controls": {name: image_asset_json(path) for name, path in sorted(control_paths.items())},
            "outputs": postprocess_step.outputs,
            "timings_ms": backend_result.timings_ms
            | {
                "postprocess_ms": postprocess_step.elapsed_ms,
                "total_ms": backend_result.timings_ms["total_ms"] + postprocess_step.elapsed_ms,
            },
            "memory": memory,
            "environment": backend_result.environment
            | {
                "postprocess": VOSR_POSTPROCESS_NAME,
            },
            "output": {
                "directory": output_dir.as_posix(),
                "raw": raw_dir.as_posix(),
                "contact_sheet": contact_sheet.as_posix(),
                "result": (output_dir / "result.json").as_posix(),
            },
        }
        if loras:
            result["generation"]["loras"] = [lora.to_json() for lora in loras]
        if manifest_context is not None:
            result["plan"] = dict(manifest_context)
        write_json(output_dir / "result.json", result)
        return result
    finally:
        if memory is None:
            memory_sampler.stop()


def _stage_lightx2v_input_images(
    reference_step: QwenIdentityReferenceStep,
    directory: Path,
) -> dict[int, Path]:
    images = (
        tuple(reference_step.source_images.values())
        + tuple(reference_step.reference_images.values())
        + tuple(reference_step.guide_images.values())
        + tuple(reference_step.control_images.values())
    )
    staged_paths = {}
    for index, image in enumerate(images):
        image_id = id(image)
        if image_id in staged_paths:
            continue
        path = directory / f"input_{index:03d}.png"
        image.save(path)
        staged_paths[image_id] = path
    return staged_paths


def run_qwen_image_edit_inpaint_candidates(
    *,
    source_image: Path,
    mask_image: Path,
    reference_images: Mapping[str, Path],
    output_dir: Path,
    profile: QwenImageEditIdentityProfile,
    prompt: str,
    max_side: int | None,
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
        source, mask = _load_qwen_inpaint_images(
            source_image,
            mask_image,
            max_side=max_side,
        )
        loaded_reference_images = [
            _load_reference_image(reference_path, max_side=max(source.size))
            for reference_path in reference_images.values()
        ]
        prompt_references = [source, *loaded_reference_images]
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
            reference_images=loaded_reference_images,
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
        preservation_failures = [
            output["name"]
            for output in denoise_step.outputs
            if not output["pixel_diff"]["passed"]
        ]
        memory = memory_sampler.stop()
        result = {
            "status": "failed" if preservation_failures else "completed",
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
                    "name": reference_name,
                    "image": image_asset_json(reference_path),
                }
                for reference_name, reference_path in reference_images.items()
            ],
            "prompt": prompt,
            "outputs": denoise_step.outputs,
            "preservation": {
                "outside_mask_unchanged": not preservation_failures,
                "failed_outputs": preservation_failures,
            },
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
        result_path = output_dir / "result.json"
        write_json(result_path, result)
        if preservation_failures:
            failed = ", ".join(preservation_failures)
            raise QwenImageEditIdentityError(
                f"Qwen refine changed pixels outside the repaint mask for {failed}; see {result_path.as_posix()}"
            )
        return result
    finally:
        if session is not None:
            session.close()
        if memory is None:
            memory_sampler.stop()


def _prepare_qwen_identity_references(
    *,
    source_images: Mapping[str, Path],
    references: Mapping[str, Path],
    guides: Mapping[str, Path],
    controls: Mapping[str, QwenControlImage],
    selected_cases: Sequence[QwenIdentityCase],
    max_side: int,
    native_canvas_pixels: int | None,
    aspect_ratio: tuple[int, int] | None,
    canvas_size: tuple[int, int] | None,
    progress: StatusReporter,
) -> QwenIdentityReferenceStep:
    progress.phase("prepare qwen identity references")
    loaded_source_images = {
        name: _load_reference_image(source_images[name], max_side=max_side)
        for name in _used_source_image_names(selected_cases)
    }
    used_reference_names = _used_reference_names(selected_cases)
    reference_images = {
        name: _load_reference_image(references[name], max_side=max_side)
        for name in used_reference_names
    }
    guide_images = {
        name: _load_reference_image(guides[name], max_side=max_side)
        for name in _used_guide_names(selected_cases)
    }
    fitted_controls: dict[tuple[str, tuple[int, int]], Image.Image] = {}
    case_control_images: dict[tuple[str, str], Image.Image] = {}
    canvas_sizes: dict[str, tuple[int, int]] = {}
    for case in selected_cases:
        control = controls[case.controls[0]] if case.controls else None
        source_image = loaded_source_images[case.source_images[0]] if case.source_images else None
        guide_image = guide_images[case.guides[-1]] if case.guides else None
        target_size = _case_canvas(
            source_image=source_image,
            guide_image=guide_image,
            control=control,
            max_side=max_side,
            native_canvas_pixels=native_canvas_pixels,
            aspect_ratio=aspect_ratio,
            canvas_size=canvas_size,
        )
        canvas_sizes[case.name] = target_size
        for control_name in case.controls:
            cache_key = (control_name, target_size)
            if cache_key not in fitted_controls:
                fitted_controls[cache_key] = _load_control_image(
                    controls[control_name],
                    target_size=target_size,
                )
            case_control_images[(case.name, control_name)] = fitted_controls[cache_key]
    return QwenIdentityReferenceStep(
        cases=tuple(selected_cases),
        source_images=loaded_source_images,
        reference_images=reference_images,
        guide_images=guide_images,
        control_images=case_control_images,
        canvas_sizes=canvas_sizes,
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
                reference_images=_case_input_images(reference_step, case),
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
    total_outputs = sum(
        len(case.seeds) if case.seeds is not None else candidates_per_case
        for case in reference_step.cases
    )
    progress.begin(total_outputs * steps, "denoise qwen identity cases")
    for index, case in enumerate(reference_step.cases):
        width, height = reference_step.canvas_sizes[case.name]
        case_seeds = case.seeds or tuple(
            seed + index * candidates_per_case + candidate_index
            for candidate_index in range(candidates_per_case)
        )
        for candidate_index, case_seed in enumerate(case_seeds):
            output_name = _case_output_name(case.name, candidate_index, len(case_seeds))
            latents, timings = session.denoise_to_latents(
                reference_images=list(_case_input_images(reference_step, case)),
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
    decode_device = session.release_denoise_models_for_decode(progress)
    progress.begin(len(denoised), "decode qwen identity latents")
    outputs: list[dict[str, Any]] = []
    for denoised_result in denoised:
        case = next(case for case in reference_step.cases if case.name == denoised_result.case_name)
        image, decode_ms = session.decode_latents(
            denoised_result.latents,
            width=denoised_result.width,
            height=denoised_result.height,
            output_name=denoised_result.name,
            device=decode_device,
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
            "source_images": list(case.source_images),
            "references": list(case.references),
            "guides": list(case.guides),
            "controls": list(case.controls),
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
    upscale_long_side: int,
    postprocess: str,
    progress: StatusReporter,
) -> QwenIdentityPostprocessStep:
    if postprocess == "none":
        return _passthrough_qwen_identity_outputs(
            raw_outputs=raw_outputs,
            output_dir=output_dir,
            progress=progress,
        )
    batch = upscale_files_with_vosr(
        files=tuple(
            (
                Path(raw_output["raw_image"]["path"]),
                output_dir / f"{raw_output['name']}.png",
            )
            for raw_output in raw_outputs
        ),
        long_side=upscale_long_side,
        progress=progress,
    )
    outputs = []
    for raw_output, upscaled in zip(raw_outputs, batch.outputs, strict=True):
        output = dict(raw_output)
        output["width"] = upscaled["target_width"]
        output["height"] = upscaled["target_height"]
        output["image"] = upscaled["output"]
        output["postprocess"] = {
            "mode": "vosr_upscale",
            "backend": upscaled["backend"],
            "model": upscaled["model"],
            "model_revision": upscaled["model_revision"],
            "scale": upscaled["scale"],
            "device": upscaled["device"],
            "long_side": upscale_long_side,
            "target_width": upscaled["target_width"],
            "target_height": upscaled["target_height"],
            "infer_steps": upscaled["infer_steps"],
            "cfg_scale": upscaled["cfg_scale"],
            "weak_cond_strength_aelq": upscaled["weak_cond_strength_aelq"],
            "align_method": upscaled["align_method"],
            "tile_size": upscaled["tile_size"],
            "seed": upscaled["seed"],
        }
        outputs.append(output)
        progress.phase(f"saved upscaled qwen image {output['name']}")
    return QwenIdentityPostprocessStep(outputs=outputs, elapsed_ms=batch.elapsed_ms)


def _passthrough_qwen_identity_outputs(
    *,
    raw_outputs: Sequence[dict[str, Any]],
    output_dir: Path,
    progress: StatusReporter,
) -> QwenIdentityPostprocessStep:
    """Publish the raw denoise output as the candidate, skipping the upscaler.

    Pixel art is finished the moment it leaves the model: the VOSR upscaler resamples the
    hard block edges and leaves halos around them, so its output is strictly worse than
    what it was given. It also costs about as much as a third of the denoise.
    """
    start = perf_counter()
    outputs = []
    for raw_output in raw_outputs:
        raw_path = Path(raw_output["raw_image"]["path"])
        output_path = output_dir / f"{raw_output['name']}.png"
        shutil.copyfile(raw_path, output_path)
        output = dict(raw_output)
        output["width"] = raw_output["raw_width"]
        output["height"] = raw_output["raw_height"]
        output["image"] = image_asset_json(output_path)
        output["postprocess"] = {"mode": "none"}
        outputs.append(output)
        progress.phase(f"kept raw qwen image {output['name']}")
    return QwenIdentityPostprocessStep(outputs=outputs, elapsed_ms=(perf_counter() - start) * 1000)


def _run_qwen_inpaint_denoise_step(
    *,
    session: QwenImageEditInpaintSession,
    source: Image.Image,
    mask: Image.Image,
    reference_images: Sequence[Image.Image],
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
    reference_latents = session.encode_reference_latents(reference_images, progress=progress)
    denoise_steps = _effective_qwen_inpaint_steps(steps=steps, strength=strength)
    progress.begin(candidates * denoise_steps, "denoise qwen refine candidates")
    for candidate_index in range(candidates):
        candidate_seed = seed + candidate_index
        output_name = f"refine_candidate_{candidate_index + 1:02d}" if candidates > 1 else "refine"
        latents, timings = session.denoise_to_latents(
            source_image=canvas.source_image,
            mask_image=canvas.mask_image,
            reference_latents=reference_latents,
            prompt_embedding=prompt_step.embeddings["refine"],
            output_name=output_name,
            width=canvas.width,
            height=canvas.height,
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
    decode_device = session.release_denoise_models_for_decode(progress)
    progress.begin(len(denoised), "decode qwen refine latents")
    outputs: list[dict[str, Any]] = []
    for denoised_result in denoised:
        image, decode_ms = session.decode_latents(
            denoised_result.latents,
            width=denoised_result.width,
            height=denoised_result.height,
            output_name=denoised_result.name,
            device=decode_device,
            progress=progress,
        )
        progress.phase(f"apply qwen refine mask overlay {denoised_result.name}")
        image = _apply_qwen_inpaint_overlay(session.pipeline, image, canvas)
        pixel_diff = exact_outside_mask_diff(
            canvas.overlay_source_image,
            image,
            canvas.overlay_mask_image,
        )
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
                "pixel_diff": pixel_diff,
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
        with self.torch.inference_mode():
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

    def release_denoise_models_for_decode(self, progress: StatusReporter) -> Any:
        return _release_qwen_denoise_models_for_decode(self.torch, self.pipeline, progress)

    def decode_latents(
        self,
        latents: Any,
        *,
        width: int,
        height: int,
        output_name: str,
        device: Any,
        progress: StatusReporter,
    ) -> tuple[Image.Image, float]:
        return _decode_qwen_image_latents(
            self.torch,
            self.pipeline,
            latents,
            width=width,
            height=height,
            output_name=output_name,
            device=device,
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

    def encode_reference_latents(
        self,
        reference_images: Sequence[Image.Image],
        *,
        progress: StatusReporter,
    ) -> QwenImageEditInpaintReferenceLatents:
        progress.phase("encode qwen refine reference latents")
        return _encode_qwen_inpaint_reference_latents(
            self.torch,
            self.pipeline,
            reference_images,
        )

    def denoise_to_latents(
        self,
        *,
        source_image: Image.Image,
        mask_image: Image.Image,
        reference_latents: QwenImageEditInpaintReferenceLatents,
        prompt_embedding: QwenImageEditPromptEmbedding,
        output_name: str,
        width: int,
        height: int,
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
        with _qwen_inpaint_reference_latent_conditioning(
            self.torch,
            self.pipeline,
            reference_latents,
        ), _qwen_inpaint_output_dimensions(self.pipeline, width=width, height=height):
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

    def release_denoise_models_for_decode(self, progress: StatusReporter) -> Any:
        return _release_qwen_denoise_models_for_decode(self.torch, self.pipeline, progress)

    def decode_latents(
        self,
        latents: Any,
        *,
        width: int,
        height: int,
        output_name: str,
        device: Any,
        progress: StatusReporter,
    ) -> tuple[Image.Image, float]:
        return _decode_qwen_image_latents(
            self.torch,
            self.pipeline,
            latents,
            width=width,
            height=height,
            output_name=output_name,
            device=device,
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


def _validate_image_inputs(
    source_images: Mapping[str, Path],
    references: Mapping[str, Path],
    guides: Mapping[str, Path],
    controls: Mapping[str, QwenControlImage],
    cases: Sequence[QwenIdentityCase],
) -> None:
    missing_source_images = sorted(
        {name for case in cases for name in case.source_images if name not in source_images}
    )
    if missing_source_images:
        raise QwenImageEditIdentityError(f"Missing Qwen source image(s): {', '.join(missing_source_images)}")
    missing_references = sorted({name for case in cases for name in case.references if name not in references})
    if missing_references:
        raise QwenImageEditIdentityError(
            f"Missing Qwen identity reference(s): {', '.join(missing_references)}"
        )
    missing_guides = sorted({name for case in cases for name in case.guides if name not in guides})
    if missing_guides:
        raise QwenImageEditIdentityError(f"Missing Qwen visual guide(s): {', '.join(missing_guides)}")
    missing_controls = sorted({name for case in cases for name in case.controls if name not in controls})
    if missing_controls:
        raise QwenImageEditIdentityError(f"Missing Qwen control image(s): {', '.join(missing_controls)}")


def _validate_edit_cases(cases: Sequence[QwenIdentityCase]) -> None:
    if not cases:
        raise QwenImageEditIdentityError("At least one Qwen edit case is required")
    names = [case.name for case in cases]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise QwenImageEditIdentityError(f"Duplicate Qwen edit case(s): {', '.join(duplicates)}")
    for case in cases:
        if not case.source_images and not case.references:
            raise QwenImageEditIdentityError(f"Qwen edit case {case.name} has no input images")
        if case.seeds is not None:
            if not case.seeds:
                raise QwenImageEditIdentityError(f"Qwen edit case {case.name} has no seeds")
            if len(set(case.seeds)) != len(case.seeds):
                raise QwenImageEditIdentityError(f"Qwen edit case {case.name} has duplicate seeds")
            if any(seed < 0 for seed in case.seeds):
                raise QwenImageEditIdentityError(f"Qwen edit case {case.name} has a negative seed")


def _validate_generation_settings(
    *,
    max_side: int | None,
    steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    max_sequence_length: int,
    candidates_per_case: int,
    nunchaku_blocks_on_gpu: int | None,
) -> None:
    if max_side is not None and max_side < 32:
        raise QwenImageEditIdentityError("max_side must be at least 32")
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


def _validate_qwen_canvas_size(
    canvas_size: tuple[int, int] | None,
    *,
    aspect_ratio: tuple[int, int] | None,
) -> None:
    if canvas_size is None:
        return
    if aspect_ratio is not None:
        raise QwenImageEditIdentityError("canvas_size and aspect_ratio are mutually exclusive")
    width, height = canvas_size
    if width < 256 or height < 256 or width % 16 or height % 16:
        raise QwenImageEditIdentityError(
            "Qwen-Image-Edit-2511 canvas dimensions must be multiples of 16 and at least 256"
        )
    if max(width, height) > QWEN_IMAGE_EDIT_2511_MAX_CUSTOM_SIDE:
        raise QwenImageEditIdentityError(
            f"Qwen-Image-Edit-2511 canvas dimensions must not exceed "
            f"{QWEN_IMAGE_EDIT_2511_MAX_CUSTOM_SIDE}px per side"
        )


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


def _used_source_image_names(cases: Sequence[QwenIdentityCase]) -> tuple[str, ...]:
    names: list[str] = []
    for case in cases:
        for name in case.source_images:
            if name not in names:
                names.append(name)
    return tuple(names)


def _used_guide_names(cases: Sequence[QwenIdentityCase]) -> tuple[str, ...]:
    names: list[str] = []
    for case in cases:
        for name in case.guides:
            if name not in names:
                names.append(name)
    return tuple(names)


def _case_input_images(
    reference_step: QwenIdentityReferenceStep,
    case: QwenIdentityCase,
) -> tuple[Image.Image, ...]:
    return tuple(
        reference_step.source_images[name]
        for name in case.source_images
    ) + tuple(
        reference_step.reference_images[name]
        for name in case.references
    ) + tuple(
        reference_step.guide_images[name]
        for name in case.guides
    ) + tuple(reference_step.control_images[(case.name, name)] for name in case.controls)


def _load_reference_image(
    path: Path,
    *,
    max_side: int | None,
) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    return _fit_image_to_max_side(rgb, max_side=max_side)


def _load_control_image(
    control: QwenControlImage,
    *,
    target_size: tuple[int, int],
) -> Image.Image:
    rgb = control.image.convert("RGB")
    if control.content_box is not None:
        rgb = _crop_control_to_aspect(rgb, content_box=control.content_box, target_size=target_size)
        return rgb.resize(target_size, Image.Resampling.LANCZOS)
    return _fit_image_to_canvas(rgb, target_size=target_size, fill="black")


def _materialize_qwen_control_images(
    controls: Mapping[str, QwenControlImage],
    output_dir: Path,
) -> dict[str, Path]:
    if not controls:
        return {}
    controls_dir = output_dir / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: controls_dir / f"{name}.png" for name in controls}
    for name, path in paths.items():
        controls[name].image.convert("RGB").save(path)
    return paths


def _crop_control_to_aspect(
    image: Image.Image,
    *,
    content_box: tuple[int, int, int, int],
    target_size: tuple[int, int],
) -> Image.Image:
    left, top, right, bottom = content_box
    target_width, target_height = target_size
    ratio_divisor = gcd(target_width, target_height)
    width_ratio = target_width // ratio_divisor
    height_ratio = target_height // ratio_divisor
    scale = max(
        (right - left + width_ratio - 1) // width_ratio,
        (bottom - top + height_ratio - 1) // height_ratio,
    )
    crop_width = width_ratio * scale
    crop_height = height_ratio * scale
    crop_left = (left + right - crop_width) // 2
    crop_top = (top + bottom - crop_height) // 2
    crop_right = crop_left + crop_width
    crop_bottom = crop_top + crop_height
    pad_left = max(0, -crop_left)
    pad_top = max(0, -crop_top)
    pad_right = max(0, crop_right - image.width)
    pad_bottom = max(0, crop_bottom - image.height)
    if pad_left or pad_top or pad_right or pad_bottom:
        canvas = Image.new(
            image.mode,
            (image.width + pad_left + pad_right, image.height + pad_top + pad_bottom),
            "black",
        )
        canvas.paste(image, (pad_left, pad_top))
        image = canvas
        crop_left += pad_left
        crop_top += pad_top
    return image.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


def _fit_image_to_canvas(
    image: Image.Image,
    *,
    target_size: tuple[int, int],
    fill: str = "white",
) -> Image.Image:
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
        padded = Image.new("RGB", canvas_size, fill)
        padded.paste(image, ((canvas_size[0] - image.width) // 2, (canvas_size[1] - image.height) // 2))
    if padded.size == target_size:
        return padded
    if padded.width < target_width:
        canvas = Image.new("RGB", target_size, fill)
        canvas.paste(padded, ((target_width - padded.width) // 2, (target_height - padded.height) // 2))
        return canvas
    return padded.resize(target_size, Image.Resampling.LANCZOS)


def _load_qwen_inpaint_images(
    source_path: Path,
    mask_path: Path,
    *,
    max_side: int | None,
) -> tuple[Image.Image, Image.Image]:
    with Image.open(source_path) as source_image:
        source = source_image.convert("RGB")
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
    if mask.size != source.size:
        raise QwenImageEditIdentityError(
            f"Mask image size {mask.width}x{mask.height} does not match source image size "
            f"{source.width}x{source.height}: {mask_path.as_posix()}"
        )
    source = _fit_image_to_max_side(source, max_side=max_side)
    mask = _fit_image_to_max_side(
        mask,
        max_side=max_side,
        fill=0,
        resample=Image.Resampling.NEAREST,
    )
    if mask.getbbox() is None:
        raise QwenImageEditIdentityError(f"Mask image has no white repaint area: {mask_path.as_posix()}")
    return source, mask


def _fit_image_to_max_side(
    image: Image.Image,
    *,
    max_side: int | None,
    fill: str | int = "white",
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    width, height = image.size
    if max_side is not None and max(width, height) > _align_to_multiple(max_side, 16):
        aligned_max_side = _align_to_multiple(max_side, 16)
        if width >= height:
            content_size = (aligned_max_side, max(1, round(height * aligned_max_side / width)))
        else:
            content_size = (max(1, round(width * aligned_max_side / height)), aligned_max_side)
        image = image.resize(content_size, resample)

    canvas_width = _align_up_to_multiple(image.width, 16)
    canvas_height = _align_up_to_multiple(image.height, 16)
    if image.size == (canvas_width, canvas_height):
        return image
    canvas = Image.new(image.mode, (canvas_width, canvas_height), fill)
    canvas.paste(image, ((canvas_width - image.width) // 2, (canvas_height - image.height) // 2))
    return canvas


def _case_canvas(
    *,
    source_image: Image.Image | None,
    guide_image: Image.Image | None,
    control: QwenControlImage | None,
    max_side: int,
    native_canvas_pixels: int | None,
    aspect_ratio: tuple[int, int] | None,
    canvas_size: tuple[int, int] | None,
) -> tuple[int, int]:
    if canvas_size is not None:
        return canvas_size
    if aspect_ratio is not None:
        width_ratio, height_ratio = aspect_ratio
    elif control is not None:
        width_ratio, height_ratio = control.image.size
    elif source_image is not None:
        width_ratio, height_ratio = source_image.size
    elif guide_image is not None:
        width_ratio, height_ratio = guide_image.size
    else:
        width_ratio, height_ratio = DEFAULT_QWEN_ASPECT_RATIO
    if native_canvas_pixels is not None:
        return _size_for_target_area(
            width_ratio,
            height_ratio,
            target_pixels=native_canvas_pixels,
            max_side=max_side,
        )
    return size_for_long_side(width_ratio, height_ratio, long_side=max_side, align_to_multiple=16)


@contextmanager
def _qwen_inpaint_output_dimensions(pipeline: Any, *, width: int, height: int) -> Iterator[None]:
    call_globals = _qwen_image_pipeline_call_globals(pipeline)
    original_calculate_dimensions = call_globals["calculate_dimensions"]

    def calculate_native_dimensions(_target_area: int, _ratio: float) -> tuple[int, int, None]:
        return width, height, None

    call_globals["calculate_dimensions"] = calculate_native_dimensions
    try:
        yield
    finally:
        call_globals["calculate_dimensions"] = original_calculate_dimensions


def _qwen_image_pipeline_call_globals(pipeline: Any) -> dict[str, Any]:
    call = pipeline.__class__.__call__
    while "calculate_dimensions" not in call.__globals__:
        call = call.__wrapped__
    return call.__globals__


def _align_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, value // multiple * multiple)


def _align_up_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, (value + multiple - 1) // multiple * multiple)


def _size_for_target_area(
    width_ratio: int,
    height_ratio: int,
    *,
    target_pixels: int,
    max_side: int,
) -> tuple[int, int]:
    ratio = width_ratio / height_ratio
    width = round(sqrt(target_pixels * ratio) / 16) * 16
    height = round(sqrt(target_pixels / ratio) / 16) * 16
    if max(width, height) <= max_side:
        return width, height
    scale = max_side / max(width, height)
    return (
        _align_to_multiple(round(width * scale), 16),
        _align_to_multiple(round(height * scale), 16),
    )


def _output_canvas_json(
    *,
    aspect_ratio: tuple[int, int] | None,
    canvas_size: tuple[int, int] | None,
    native_canvas_pixels: int | None,
    upscale_long_side: int,
) -> dict[str, Any]:
    if canvas_size is not None:
        mode = "explicit_size"
        aspect_owner = "explicit_size"
        target_pixels = canvas_size[0] * canvas_size[1]
    else:
        mode = "model_native_area" if native_canvas_pixels is not None else "long_side"
        aspect_owner = "explicit" if aspect_ratio is not None else "control_source_guide_or_default"
        target_pixels = native_canvas_pixels
    payload: dict[str, Any] = {
        "mode": mode,
        "aspect_owner": aspect_owner,
        "aspect_ratio": list(aspect_ratio) if aspect_ratio is not None else None,
        "target_pixels": target_pixels,
        "alignment": 16,
        "postprocess": VOSR_POSTPROCESS_NAME,
        "upscale_long_side": upscale_long_side,
        "upscale_model": VOSR_MODEL_NAME,
    }
    if canvas_size is not None:
        payload["target_size"] = list(canvas_size)
    return payload


def _qwen_inpaint_canvas_size(pipeline: Any, image: Image.Image) -> tuple[int, int]:
    multiple_of = pipeline.vae_scale_factor * 2
    return (
        _align_up_to_multiple(image.width, multiple_of),
        _align_up_to_multiple(image.height, multiple_of),
    )


def _encode_qwen_inpaint_reference_latents(
    torch: Any,
    pipeline: Any,
    reference_images: Sequence[Image.Image],
) -> QwenImageEditInpaintReferenceLatents:
    from diffusers import QwenImageEditPlusPipeline

    device = pipeline._execution_device
    dtype = pipeline.vae.dtype
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    packed_latents = []
    shapes = []
    with torch.inference_mode():
        for image in reference_images:
            width, height = _qwen_inpaint_canvas_size(pipeline, image)
            image_tensor = pipeline.image_processor.preprocess(
                image,
                height=height,
                width=width,
            ).unsqueeze(2).to(device=device, dtype=dtype)
            image_latents = QwenImageEditPlusPipeline._encode_vae_image(
                pipeline,
                image_tensor,
                generator=None,
            )
            latent_height, latent_width = image_latents.shape[3:]
            packed_latents.append(
                pipeline._pack_latents(
                    image_latents,
                    1,
                    num_channels_latents,
                    latent_height,
                    latent_width,
                )
            )
            shapes.append((1, latent_height // 2, latent_width // 2))
    return QwenImageEditInpaintReferenceLatents(
        packed_latents=torch.cat(packed_latents, dim=1),
        shapes=tuple(shapes),
    )


@contextmanager
def _qwen_inpaint_reference_latent_conditioning(
    torch: Any,
    pipeline: Any,
    reference_latents: QwenImageEditInpaintReferenceLatents,
) -> Iterator[None]:
    original_forward = pipeline.transformer.forward

    def forward_with_reference_latents(_transformer: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs["hidden_states"] = torch.cat(
            (kwargs["hidden_states"], reference_latents.packed_latents),
            dim=1,
        )
        kwargs["img_shapes"] = [
            [*image_shapes, *reference_latents.shapes]
            for image_shapes in kwargs["img_shapes"]
        ]
        return original_forward(*args, **kwargs)

    pipeline.transformer.forward = MethodType(forward_with_reference_latents, pipeline.transformer)
    try:
        yield
    finally:
        pipeline.transformer.forward = original_forward


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
            overlay_source_image=source,
            overlay_mask_image=mask,
        )

    full_source = source
    full_mask = mask
    crop_coords = pipeline.mask_processor.get_crop_region(full_mask, width, height, pad=padding_mask_crop)
    cropped_source = full_source.crop(crop_coords)
    cropped_mask = full_mask.crop(crop_coords)
    return QwenImageEditInpaintCanvas(
        source_image=cropped_source,
        mask_image=cropped_mask,
        width=width,
        height=height,
        overlay_source_image=full_source,
        overlay_mask_image=full_mask,
        crop_coords=crop_coords,
    )


def _apply_qwen_inpaint_overlay(
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


def _release_qwen_denoise_models_for_decode(torch: Any, pipeline: Any, progress: StatusReporter) -> Any:
    progress.phase("release qwen transformer before vae decode")
    decode_device = pipeline._execution_device
    pipeline.transformer.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return decode_device


def _decode_qwen_image_latents(
    torch: Any,
    pipeline: Any,
    latents: Any,
    *,
    width: int,
    height: int,
    output_name: str,
    device: Any,
    progress: StatusReporter,
) -> tuple[Image.Image, float]:
    progress.phase(f"decode qwen latents {output_name}")
    start = synchronized_time(torch)
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
