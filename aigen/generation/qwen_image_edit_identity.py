from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log
from pathlib import Path
from types import MethodType
from typing import Any

from PIL import Image

from aigen.character_reference_models import CHARACTER_REFERENCE_NAME_SET
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
    normalized_instruction: dict[str, Any] | None = None


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
    try:
        return QWEN_IDENTITY_PROFILES[profile_name]
    except KeyError as error:
        raise QwenImageEditIdentityError(f"Unknown Qwen identity profile: {profile_name}") from error


def qwen_image_edit_identity_profile_names() -> tuple[str, ...]:
    return tuple(QWEN_IDENTITY_PROFILES)


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
    selected_cases = tuple(edit_cases)
    _validate_edit_cases(selected_cases)
    _validate_reference_pack(references, selected_cases)
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
    session: QwenImageEditIdentitySession | None = None
    try:
        reference_step = _prepare_qwen_identity_references(
            references=references,
            selected_cases=selected_cases,
            max_side=max_side,
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
            images_dir=images_dir,
            max_side=max_side,
            steps=resolved_steps,
            true_cfg_scale=resolved_true_cfg_scale,
            guidance_scale=resolved_guidance_scale,
            seed=seed,
            max_sequence_length=max_sequence_length,
            candidates_per_case=candidates_per_case,
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
            },
            "references": {name: image_asset_json(path) for name, path in sorted(references.items())},
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
    progress: StatusReporter,
) -> QwenIdentityReferenceStep:
    progress.phase("prepare qwen identity references")
    used_reference_names = _used_reference_names(selected_cases)
    return QwenIdentityReferenceStep(
        cases=tuple(selected_cases),
        reference_images={
            name: _load_reference_image(references[name], max_side=max_side) for name in used_reference_names
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


def _run_qwen_identity_denoise_step(
    *,
    session: QwenImageEditIdentitySession,
    reference_step: QwenIdentityReferenceStep,
    prompt_step: QwenIdentityPromptConditioningStep,
    images_dir: Path,
    max_side: int,
    steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    seed: int,
    max_sequence_length: int,
    candidates_per_case: int,
    progress: StatusReporter,
) -> QwenIdentityDenoiseStep:
    start = synchronized_time(session.torch)
    outputs: list[dict[str, Any]] = []
    total_cases = len(reference_step.cases)
    progress.begin(total_cases * candidates_per_case * steps, "denoise qwen identity cases")
    for index, case in enumerate(reference_step.cases):
        width, height = _case_canvas(case, max_side=max_side)
        for candidate_index in range(candidates_per_case):
            case_seed = seed + index * candidates_per_case + candidate_index
            output_name = _case_output_name(case.name, candidate_index, candidates_per_case)
            image, timings = session.generate(
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
            progress.phase(f"save qwen identity case {output_name}")
            image_path = images_dir / f"{output_name}.png"
            image.save(image_path)
            output = {
                "name": output_name,
                "case": case.name,
                "candidate_index": candidate_index,
                "seed": case_seed,
                "width": width,
                "height": height,
                "references": list(case.references),
                "prompt": case.prompt,
                "image": image_asset_json(image_path),
                "timings_ms": timings,
            }
            if case.normalized_instruction is not None:
                output["normalized_instruction"] = case.normalized_instruction
            outputs.append(output)
    return QwenIdentityDenoiseStep(outputs=outputs, elapsed_ms=elapsed_ms(start, synchronized_time(session.torch)))


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

    def generate(
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
    ) -> tuple[Image.Image, dict[str, Any]]:
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
                output_type="pil",
            )
        return output.images[0], {"pipeline_ms": elapsed_ms(start, synchronized_time(self.torch))}

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


def _load_reference_image(path: Path, *, max_side: int) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    return _fit_image_to_max_side(rgb, max_side=max_side)


def _fit_image_to_max_side(image: Image.Image, *, max_side: int) -> Image.Image:
    longest_side = max(image.size)
    if longest_side <= max_side:
        return image
    scale = max_side / longest_side
    width = max(16, _align_to_multiple(round(image.width * scale), 16))
    height = max(16, _align_to_multiple(round(image.height * scale), 16))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _case_canvas(case: QwenIdentityCase, *, max_side: int) -> tuple[int, int]:
    height = _align_to_multiple(max_side, 16)
    if case.portrait_canvas:
        return height, height
    return _align_to_multiple(round(height * 2 / 3), 16), height


def _align_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, value // multiple * multiple)


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
