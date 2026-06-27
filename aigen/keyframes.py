from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Thread
from typing import Any, Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aigen.generation.kontext_pose_control import (
    CharacterKontextPoseSession,
    KontextControlCondition,
    _generation_environment,
    cuda_memory_stats,
    fit_size_to_area,
    synchronized_time,
)
from aigen.prompt_tokens import count_kontext_prompt_tokens


KEYFRAME_JOB_SCHEMA = "schemas/keyframe-job.schema.json"
KEYFRAME_KIND = "character-keyframe"
KEYFRAME_SCHEMA_VERSION = 1
NVIDIA_SMI_PREFLIGHT_LIMIT_MB = 1800
NVIDIA_SMI_SAMPLE_SECONDS = 0.25
FLUX_TOKEN_SIZE = 16
VRAM_ESTIMATE_BASELINE_FRAMEBUFFER_MB = 700
VRAM_ESTIMATE_SAFETY_MARGIN_MB = 256
VRAM_ESTIMATE_BASE_PEAK_MB = 15200
VRAM_ESTIMATE_GENERATED_TOKEN_NUMERATOR = 22
VRAM_ESTIMATE_REFERENCE_TOKEN_NUMERATOR = 5
VRAM_ESTIMATE_TOKEN_DENOMINATOR = 100
VRAM_ESTIMATE_TRUE_CFG_MB = 250
VRAM_ESTIMATE_HIGH_GENERATED_TOKEN_THRESHOLD = 2100


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PipelineSpec(StrictModel):
    profile: str


class PathSpec(StrictModel):
    path: str


class IdentityPrimerSpec(StrictModel):
    view: Literal["front", "left_profile", "right_profile", "back"]
    path: str


class CharacterSpec(StrictModel):
    id: str
    identity_primer: IdentityPrimerSpec


class KeyframeSpec(StrictModel):
    action: str
    phase: str
    direction: Literal["left", "right"]
    camera: Literal["orthographic-side"]


class AssetSpec(StrictModel):
    pose: PathSpec
    contour: PathSpec | None = None
    boundary_mask: PathSpec | None = None
    depth: PathSpec | None = None
    softedge: PathSpec | None = None


class PromptSpec(StrictModel):
    clip: str
    t5: str
    negative: str | None = None
    true_cfg_scale: float


class CanvasSpec(StrictModel):
    width: int
    height: int
    reference_max_area: int
    max_sequence_length: int


class SamplingSpec(StrictModel):
    steps: int
    guidance_scale: float


class ControlConditionSpec(StrictModel):
    name: str
    type: Literal["pose", "canny", "softedge", "depth"]
    image: str
    scale: float
    start: float
    end: float
    residual_mask: str | None = None


class VariantSpec(StrictModel):
    name: str
    seed: int


class OutputSpec(StrictModel):
    directory: str
    filename: str
    overwrite: bool
    save_conditions: bool
    save_contact_sheet: bool


class AcceptanceSpec(StrictModel):
    manual: list[str]
    minimum_passing_variants: int


class KeyframeJobSpec(StrictModel):
    schema_path: str = Field(alias="$schema")
    schema_version: Literal[1]
    kind: Literal["character-keyframe"]
    id: str
    pipeline: PipelineSpec
    character: CharacterSpec
    keyframe: KeyframeSpec
    assets: AssetSpec
    prompt: PromptSpec
    canvas: CanvasSpec
    sampling: SamplingSpec
    conditions: list[ControlConditionSpec]
    variants: list[VariantSpec]
    output: OutputSpec
    acceptance: AcceptanceSpec


@dataclass(frozen=True)
class KeyframeProfile:
    name: str
    model: str
    controlnet_model: str
    nunchaku_transformer_model: Path
    attention_impl: str
    dtype: str
    pipeline_cpu_offload: bool
    nunchaku_layer_offload: bool
    vae_tiling: bool
    model_revisions: dict[str, dict[str, str]]


class KeyframeJobError(RuntimeError):
    pass


class NvidiaSmiMemorySampler:
    def __init__(self, preflight: dict[str, int]) -> None:
        self.preflight = preflight
        self.peak_used_mb = preflight["nvidia_smi_preflight_used_mb"]
        self.device_total_mb = preflight["nvidia_smi_device_total_mb"]
        self._stop = Event()
        self._thread = Thread(target=self._sample_loop, daemon=True)

    def start(self) -> None:
        if self.device_total_mb:
            self._thread.start()

    def stop(self) -> dict[str, int]:
        if self._thread.is_alive():
            self._stop.set()
            self._thread.join()
        return {
            **self.preflight,
            "nvidia_smi_peak_used_mb": self.peak_used_mb,
        }

    def _sample_loop(self) -> None:
        while not self._stop.wait(NVIDIA_SMI_SAMPLE_SECONDS):
            snapshot = _nvidia_smi_memory_snapshot()
            self.peak_used_mb = max(self.peak_used_mb, snapshot["nvidia_smi_used_mb"])


def keyframe_job_schema() -> dict[str, Any]:
    schema = KeyframeJobSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def c2_profile_template() -> dict[str, Any]:
    spec = KeyframeJobSpec(
        **{
            "$schema": "../../schemas/keyframe-job.schema.json",
            "schema_version": KEYFRAME_SCHEMA_VERSION,
            "kind": KEYFRAME_KIND,
            "id": "ai46.walk.contact.left.v1",
            "pipeline": {"profile": "nunchaku-kontext-pose-quality"},
            "character": {
                "id": "ai46",
                "identity_primer": {
                    "view": "front",
                    "path": "../../assets/characters/ai46/views/front_v1.png",
                },
            },
            "keyframe": {
                "action": "walk",
                "phase": "contact",
                "direction": "left",
                "camera": "orthographic-side",
            },
            "assets": {
                "pose": {"path": "../../assets/poses/ai46_walk_contact_pose.png"},
                "contour": {"path": "../../assets/contours/ai46_walk_contact_canny.png"},
                "boundary_mask": {"path": "../../assets/masks/ai46_walk_contact_boundary.png"},
            },
            "prompt": {
                "clip": (
                    "Same anime girl in a strict orthographic left-facing 90-degree side profile, "
                    "platformer character side-view walking pose, exactly one eye visible, nose "
                    "and chin in silhouette, short light-brown bob haircut, brown leather jacket, "
                    "white shirt, blue tie, brown shorts, blue thigh-high socks, brown boots."
                ),
                "t5": (
                    "Same anime girl in a strict orthographic left-facing 90-degree side profile, "
                    "platformer character side-view walking pose, exactly one eye visible, nose and "
                    "chin in silhouette, shoulders overlapping in depth, chest facing sideways. Keep "
                    "the same short light-brown bob haircut, blue eyes, white shirt, blue tie, brown "
                    "leather jacket, brown shorts, gloves, blue thigh-high socks, and brown boots. "
                    "Clean plain neutral studio background."
                ),
                "true_cfg_scale": 1.0,
            },
            "canvas": {
                "width": 512,
                "height": 768,
                "reference_max_area": 524288,
                "max_sequence_length": 128,
            },
            "sampling": {
                "steps": 28,
                "guidance_scale": 2.5,
            },
            "conditions": [
                {
                    "name": "pose",
                    "type": "pose",
                    "image": "pose",
                    "scale": 0.55,
                    "start": 0.0,
                    "end": 0.55,
                },
                {
                    "name": "profile_contour",
                    "type": "canny",
                    "image": "contour",
                    "residual_mask": "boundary_mask",
                    "scale": 0.35,
                    "start": 0.0,
                    "end": 0.40,
                },
            ],
            "variants": [
                {"name": "seed_001", "seed": 1},
                {"name": "seed_002", "seed": 2},
                {"name": "seed_003", "seed": 3},
                {"name": "seed_004", "seed": 4},
            ],
            "output": {
                "directory": "../../runs/keyframes/ai46/walk_contact/v1",
                "filename": "{id}__{variant}.png",
                "overwrite": False,
                "save_conditions": True,
                "save_contact_sheet": True,
            },
            "acceptance": {
                "manual": [
                    "exactly one eye visible",
                    "short bob preserved",
                    "jacket, tie, shorts, socks and boots preserved",
                    "feet fully visible",
                    "strict side profile",
                    "pose reads as walk contact frame",
                ],
                "minimum_passing_variants": 3,
            },
        }
    )
    return spec.model_dump(mode="json", by_alias=True, exclude_none=True)


def load_keyframe_job(path: Path) -> KeyframeJobSpec:
    try:
        return KeyframeJobSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise KeyframeJobError(f"Invalid keyframe job {path}: {error}") from error


def resolve_keyframe_job(
    job_path: Path,
    profile: KeyframeProfile,
    *,
    project_root: Path,
    check_outputs: bool,
) -> dict[str, Any]:
    spec = load_keyframe_job(job_path)
    return resolve_keyframe_spec(spec, job_path, profile, project_root=project_root, check_outputs=check_outputs)


def resolve_keyframe_spec(
    spec: KeyframeJobSpec,
    job_path: Path,
    profile: KeyframeProfile,
    *,
    project_root: Path,
    check_outputs: bool,
) -> dict[str, Any]:
    if spec.pipeline.profile != profile.name:
        raise KeyframeJobError(f"Job uses profile {spec.pipeline.profile}, but CLI resolved {profile.name}")

    assets = _resolve_assets(spec, job_path.parent)
    _validate_conditions(spec, assets)
    _validate_asset_dimensions(spec, assets)
    _validate_masks(spec, assets)
    output_dir = _resolve_output_dir(spec.output.directory, job_path.parent)
    outputs = _planned_outputs(spec, output_dir)
    if check_outputs and not spec.output.overwrite:
        existing = [output for output in outputs if Path(output["path"]).exists()]
        if existing:
            raise KeyframeJobError(f"Output exists and overwrite=false: {existing[0]['path']}")

    prompt_tokens = count_kontext_prompt_tokens(profile.model, spec.prompt.clip, spec.prompt.t5)
    if prompt_tokens.clip > prompt_tokens.clip_limit:
        raise KeyframeJobError(f"CLIP prompt has {prompt_tokens.clip} tokens, limit is {prompt_tokens.clip_limit}")
    if prompt_tokens.t5 > spec.canvas.max_sequence_length:
        raise KeyframeJobError(
            f"T5 prompt has {prompt_tokens.t5} tokens, max_sequence_length is {spec.canvas.max_sequence_length}"
        )
    if spec.prompt.negative is not None and spec.prompt.true_cfg_scale <= 1.0:
        raise KeyframeJobError("negative prompt is configured but true_cfg_scale <= 1.0")
    if spec.prompt.true_cfg_scale > 1.0 and spec.prompt.negative is None:
        raise KeyframeJobError("true_cfg_scale > 1.0 requires prompt.negative")

    condition_plan = [_condition_plan(condition, spec.sampling.steps) for condition in spec.conditions]
    inactive = [condition["name"] for condition in condition_plan if condition["active_steps"] == 0]
    if inactive:
        raise KeyframeJobError(f"Condition has zero active steps: {inactive[0]}")

    token_metadata = _planned_token_metadata(spec, assets)
    vram_plan = _vram_plan(spec, token_metadata)
    return {
        "schema_version": KEYFRAME_SCHEMA_VERSION,
        "kind": "resolved-character-keyframe",
        "job_path": job_path.resolve().as_posix(),
        "job_id": spec.id,
        "profile": _profile_json(profile),
        "character": {
            "id": spec.character.id,
            "identity_primer": {
                "view": spec.character.identity_primer.view,
                **assets["identity_primer"],
            },
        },
        "keyframe": spec.keyframe.model_dump(mode="json"),
        "assets": {name: value for name, value in assets.items() if name != "identity_primer"},
        "prompt": spec.prompt.model_dump(mode="json", exclude_none=True),
        "canvas": spec.canvas.model_dump(mode="json"),
        "sampling": spec.sampling.model_dump(mode="json"),
        "conditions": [condition.model_dump(mode="json", exclude_none=True) for condition in spec.conditions],
        "condition_plan": condition_plan,
        "variants": [variant.model_dump(mode="json") for variant in spec.variants],
        "output": {
            **spec.output.model_dump(mode="json"),
            "directory": output_dir.as_posix(),
            "files": outputs,
        },
        "acceptance": spec.acceptance.model_dump(mode="json"),
        "tokens": {
            "clip": prompt_tokens.clip,
            "clip_limit": prompt_tokens.clip_limit,
            "t5": prompt_tokens.t5,
            "t5_limit": spec.canvas.max_sequence_length,
        },
        "token_metadata": token_metadata,
        "vram_plan": vram_plan,
        "git_commit": _git_commit(project_root),
        "spec_sha256": _sha256_bytes(job_path.read_bytes()),
    }


def validate_keyframe_job(job_path: Path, profile: KeyframeProfile, *, project_root: Path) -> dict[str, Any]:
    resolved = resolve_keyframe_job(job_path, profile, project_root=project_root, check_outputs=False)
    return {
        "status": "valid",
        "job_id": resolved["job_id"],
        "profile": resolved["profile"]["name"],
        "tokens": resolved["tokens"],
        "condition_plan": resolved["condition_plan"],
        "outputs": resolved["output"]["files"],
    }


def plan_keyframe_job(job_path: Path, profile: KeyframeProfile, *, project_root: Path) -> dict[str, Any]:
    return resolve_keyframe_job(job_path, profile, project_root=project_root, check_outputs=True)


def run_keyframe_job(job_path: Path, profile: KeyframeProfile, *, project_root: Path) -> dict[str, Any]:
    spec = load_keyframe_job(job_path)
    return run_keyframe_spec(spec, job_path, profile, project_root=project_root)


def run_keyframe_spec(
    spec: KeyframeJobSpec,
    job_path: Path,
    profile: KeyframeProfile,
    *,
    project_root: Path,
) -> dict[str, Any]:
    resolved = resolve_keyframe_spec(spec, job_path, profile, project_root=project_root, check_outputs=True)
    memory_sampler = NvidiaSmiMemorySampler(_nvidia_smi_keyframe_preflight(resolved["vram_plan"]))
    output_dir = Path(resolved["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved.json", resolved)
    if spec.output.save_conditions:
        _save_conditions(resolved, output_dir)

    memory_sampler.start()
    try:
        session = CharacterKontextPoseSession(
            profile.model,
            profile.controlnet_model,
            dtype=profile.dtype,
            nunchaku_transformer_model=profile.nunchaku_transformer_model,
            attention_impl=profile.attention_impl,
            pipeline_cpu_offload=profile.pipeline_cpu_offload,
            nunchaku_layer_offload=profile.nunchaku_layer_offload,
            vae_tiling=profile.vae_tiling,
        )
        try:
            torch = session.torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats("cuda")
            total_start = synchronized_time(torch)
            prepared = session.prepare(
                reference_image=Path(resolved["character"]["identity_primer"]["path"]),
                pose_image=Path(resolved["assets"]["pose"]["path"]),
                prompt=spec.prompt.clip,
                t5_prompt=spec.prompt.t5,
                negative_prompt=spec.prompt.negative,
                true_cfg_scale=spec.prompt.true_cfg_scale,
                width=spec.canvas.width,
                height=spec.canvas.height,
                reference_max_area=spec.canvas.reference_max_area,
                max_sequence_length=spec.canvas.max_sequence_length,
                steps=spec.sampling.steps,
                guidance_scale=spec.sampling.guidance_scale,
                seed=spec.variants[0].seed,
            )
            control_images, control_repeats = _prepare_control_images(session, prepared, spec, resolved)
            masks = _prepare_masks(session, prepared, spec, resolved)
            session.pipeline.maybe_free_model_hooks()
            denoised = []
            for variant in spec.variants:
                result = session.pipeline.denoise_prepared(
                    prepared,
                    name=variant.name,
                    seed=variant.seed,
                    controlnet_conditioning_scale=spec.conditions[0].scale,
                    control_guidance_start=spec.conditions[0].start,
                    control_guidance_end=spec.conditions[0].end,
                    control_conditions=[
                        KontextControlCondition(
                            condition.name,
                            control_images[condition.image],
                            condition.scale,
                            condition.start,
                            condition.end,
                            control_repeats[condition.image],
                            masks[condition.residual_mask] if condition.residual_mask else None,
                        )
                        for condition in spec.conditions
                    ],
                )
                denoised.append(replace(result, latents=result.latents.detach().cpu()))
                del result
            images, decode_ms = session.decode_many(prepared, denoised, chunk_size=1)
            outputs = []
            for image, result, planned in zip(images, denoised, resolved["output"]["files"], strict=True):
                output_path = Path(planned["path"])
                image.save(output_path)
                outputs.append(
                    {
                        **planned,
                        "seed": result.seed,
                        "controlnet_active_steps": result.controlnet_active_steps,
                        "controlnet_step_ms": result.controlnet_step_ms,
                        "transformer_step_ms": result.transformer_step_ms,
                        "controlnet_metadata": result.controlnet_metadata,
                        "timings_ms": result.timings_ms,
                    }
                )
            if spec.output.save_contact_sheet:
                _save_contact_sheet(outputs, output_dir / "contact_sheet.png")
            result_json = {
                "status": "completed",
                "job_id": spec.id,
                "spec_sha256": resolved["spec_sha256"],
                "git_commit": resolved["git_commit"],
                "models": resolved["profile"]["models"],
                "assets": resolved["assets"] | {"identity_primer": resolved["character"]["identity_primer"]},
                "outputs": outputs,
                "effective_config": resolved,
                "token_metadata": prepared.token_metadata,
                "timings_ms": {
                    "model_load_ms": session.model_load_ms,
                    "decode_ms": decode_ms,
                    "total_ms": (synchronized_time(torch) - total_start) * 1000,
                },
                "memory": cuda_memory_stats(torch, "cuda") | memory_sampler.stop(),
                "environment": _generation_environment(torch, session.pipeline),
            }
            _write_json(output_dir / "result.json", result_json)
            return result_json
        finally:
            session.close()
    finally:
        memory_sampler.stop()


def _profile_json(profile: KeyframeProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "dtype": profile.dtype,
        "attention_impl": profile.attention_impl,
        "pipeline_cpu_offload": profile.pipeline_cpu_offload,
        "nunchaku_layer_offload": profile.nunchaku_layer_offload,
        "vae_tiling": profile.vae_tiling,
        "models": {
            **profile.model_revisions,
            "kontext": {
                **profile.model_revisions["kontext"],
                "path": profile.model,
            },
            "controlnet": {
                **profile.model_revisions["controlnet"],
                "path": profile.controlnet_model,
            },
            "nunchaku_transformer": {
                **profile.model_revisions["nunchaku_transformer"],
                "path": profile.nunchaku_transformer_model.resolve().as_posix(),
            },
        },
    }


def _resolve_assets(spec: KeyframeJobSpec, base_dir: Path) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {
        "identity_primer": _asset_json(_resolve_path(spec.character.identity_primer.path, base_dir)),
        "pose": _asset_json(_resolve_path(spec.assets.pose.path, base_dir)),
    }
    for name in ("contour", "boundary_mask", "depth", "softedge"):
        asset = getattr(spec.assets, name)
        if asset:
            assets[name] = _asset_json(_resolve_path(asset.path, base_dir))
    return assets


def _validate_conditions(spec: KeyframeJobSpec, assets: dict[str, dict[str, Any]]) -> None:
    for condition in spec.conditions:
        if condition.image not in assets:
            raise KeyframeJobError(f"Condition {condition.name} references unknown image asset: {condition.image}")
        if condition.residual_mask and condition.residual_mask not in assets:
            raise KeyframeJobError(
                f"Condition {condition.name} references unknown residual mask: {condition.residual_mask}"
            )


def _validate_asset_dimensions(spec: KeyframeJobSpec, assets: dict[str, dict[str, Any]]) -> None:
    for name, asset in assets.items():
        if name == "identity_primer":
            continue
        if asset["width"] != spec.canvas.width or asset["height"] != spec.canvas.height:
            raise KeyframeJobError(
                f"Asset {name} must be {spec.canvas.width}x{spec.canvas.height}, "
                f"got {asset['width']}x{asset['height']}"
            )


def _validate_masks(spec: KeyframeJobSpec, assets: dict[str, dict[str, Any]]) -> None:
    for condition in spec.conditions:
        if not condition.residual_mask:
            continue
        with Image.open(assets[condition.residual_mask]["path"]) as image:
            extrema = image.convert("L").getextrema()
        if extrema == (0, 0) or extrema == (255, 255):
            raise KeyframeJobError(f"Residual mask is not usable: {condition.residual_mask}")


def _condition_plan(condition: ControlConditionSpec, steps: int) -> dict[str, Any]:
    active_steps = sum(
        condition.scale > 0.0
        and condition.start <= i / steps
        and (i + 1) / steps <= condition.end
        for i in range(steps)
    )
    return {
        "name": condition.name,
        "type": condition.type,
        "scale": condition.scale,
        "start": condition.start,
        "end": condition.end,
        "active_steps": active_steps,
    }


def _planned_token_metadata(spec: KeyframeJobSpec, assets: dict[str, dict[str, Any]]) -> dict[str, int]:
    reference_width, reference_height = fit_size_to_area(
        assets["identity_primer"]["width"],
        assets["identity_primer"]["height"],
        max_area=spec.canvas.reference_max_area,
        multiple_of=FLUX_TOKEN_SIZE,
    )
    generated_tokens = _flux_tokens(spec.canvas.width, spec.canvas.height)
    reference_tokens = _flux_tokens(reference_width, reference_height)
    text_tokens = spec.canvas.max_sequence_length
    return {
        "reference_width": reference_width,
        "reference_height": reference_height,
        "reference_tokens": reference_tokens,
        "generated_tokens": generated_tokens,
        "text_tokens": text_tokens,
        "total_tokens": generated_tokens + reference_tokens + text_tokens,
    }


def _flux_tokens(width: int, height: int) -> int:
    return (width // FLUX_TOKEN_SIZE) * (height // FLUX_TOKEN_SIZE)


def _vram_plan(spec: KeyframeJobSpec, token_metadata: dict[str, int]) -> dict[str, Any]:
    true_cfg_extra_mb = VRAM_ESTIMATE_TRUE_CFG_MB if spec.prompt.true_cfg_scale > 1.0 else 0
    high_generated_token_extra_mb = max(
        0,
        token_metadata["generated_tokens"] - VRAM_ESTIMATE_HIGH_GENERATED_TOKEN_THRESHOLD,
    )
    token_mb = (
        token_metadata["generated_tokens"] * VRAM_ESTIMATE_GENERATED_TOKEN_NUMERATOR
        + token_metadata["reference_tokens"] * VRAM_ESTIMATE_REFERENCE_TOKEN_NUMERATOR
    ) // VRAM_ESTIMATE_TOKEN_DENOMINATOR
    estimated_clean_peak_mb = (
        VRAM_ESTIMATE_BASE_PEAK_MB + token_mb + true_cfg_extra_mb + high_generated_token_extra_mb
    )
    return {
        "method": "nunchaku-kontext-controlnet-local-v1",
        "baseline_framebuffer_mb": VRAM_ESTIMATE_BASELINE_FRAMEBUFFER_MB,
        "safety_margin_mb": VRAM_ESTIMATE_SAFETY_MARGIN_MB,
        "estimated_clean_peak_mb": estimated_clean_peak_mb,
        "true_cfg_extra_mb": true_cfg_extra_mb,
        "high_generated_token_extra_mb": high_generated_token_extra_mb,
        "generated_tokens": token_metadata["generated_tokens"],
        "reference_tokens": token_metadata["reference_tokens"],
        "text_tokens": token_metadata["text_tokens"],
        "total_tokens": token_metadata["total_tokens"],
    }


def _planned_outputs(spec: KeyframeJobSpec, output_dir: Path) -> list[dict[str, str | int]]:
    return [
        {
            "name": variant.name,
            "seed": variant.seed,
            "path": (output_dir / spec.output.filename.format(id=spec.id, variant=variant.name)).as_posix(),
        }
        for variant in spec.variants
    ]


def _prepare_control_images(
    session: CharacterKontextPoseSession,
    prepared: Any,
    spec: KeyframeJobSpec,
    resolved: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    from PIL import Image

    control_images = {"pose": prepared.control_image}
    control_repeats = {"pose": prepared.controlnet_blocks_repeat}
    for condition in spec.conditions:
        if condition.image in control_images:
            continue
        control_image, blocks_repeat, _prepare_ms = session.prepare_control_condition(
            prepared,
            pose_image=Image.open(resolved["assets"][condition.image]["path"]).convert("RGB"),
            seed=spec.variants[0].seed,
        )
        control_images[condition.image] = control_image
        control_repeats[condition.image] = blocks_repeat
    return control_images, control_repeats


def _prepare_masks(
    session: CharacterKontextPoseSession,
    prepared: Any,
    spec: KeyframeJobSpec,
    resolved: dict[str, Any],
) -> dict[str, Any]:
    masks = {}
    for condition in spec.conditions:
        if not condition.residual_mask or condition.residual_mask in masks:
            continue
        masks[condition.residual_mask] = session.prepare_residual_mask(
            prepared,
            Image.open(resolved["assets"][condition.residual_mask]["path"]).convert("RGB"),
        )
    return masks


def _save_conditions(resolved: dict[str, Any], output_dir: Path) -> None:
    condition_dir = output_dir / "conditions"
    condition_dir.mkdir(parents=True, exist_ok=True)
    for name, asset in resolved["assets"].items():
        shutil.copy2(asset["path"], condition_dir / f"{name}{Path(asset['path']).suffix}")


def _save_contact_sheet(outputs: list[dict[str, Any]], output_path: Path) -> None:
    images = [Image.open(output["path"]).convert("RGB") for output in outputs]
    thumb_w = 256
    thumb_h = max(1, int(thumb_w * images[0].height / images[0].width))
    label_h = 32
    sheet = Image.new("RGB", (thumb_w * len(images), thumb_h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, output) in enumerate(zip(images, outputs, strict=True)):
        x = index * thumb_w
        sheet.paste(image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, label_h))
        draw.text((x + 8, 8), output["name"], fill="black")
    sheet.save(output_path)


def _asset_json(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        mode = image.mode
        width, height = image.size
    return {
        "path": path.as_posix(),
        "sha256": _sha256_bytes(path.read_bytes()),
        "mode": mode,
        "width": width,
        "height": height,
    }


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise KeyframeJobError(f"Missing path: {path.as_posix()}")
    return path


def _resolve_output_dir(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_commit(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
    ).strip()


def _nvidia_smi_preflight() -> dict[str, int]:
    return _nvidia_smi_preflight_limit(NVIDIA_SMI_PREFLIGHT_LIMIT_MB)


def _nvidia_smi_keyframe_preflight(vram_plan: dict[str, Any]) -> dict[str, int]:
    if not _cuda_available():
        return {
            "nvidia_smi_preflight_used_mb": 0,
            "nvidia_smi_device_total_mb": 0,
            "nvidia_smi_preflight_utilization_gpu": 0,
            "vram_estimated_required_mb": 0,
            "vram_estimated_headroom_mb": 0,
        }

    snapshot = _nvidia_smi_memory_snapshot()
    extra_framebuffer_mb = max(0, snapshot["nvidia_smi_used_mb"] - vram_plan["baseline_framebuffer_mb"])
    required_mb = vram_plan["estimated_clean_peak_mb"] + extra_framebuffer_mb + vram_plan["safety_margin_mb"]
    headroom_mb = snapshot["nvidia_smi_device_total_mb"] - required_mb
    if headroom_mb < 0:
        raise KeyframeJobError(
            "Estimated VRAM requirement exceeds available framebuffer: "
            f"need about {required_mb} MB including margin, "
            f"GPU has {snapshot['nvidia_smi_device_total_mb']} MB, "
            f"currently used {snapshot['nvidia_smi_used_mb']} MB. "
            "Close GPU consumers or lower output/reference tokens."
        )
    return {
        "nvidia_smi_preflight_used_mb": snapshot["nvidia_smi_used_mb"],
        "nvidia_smi_device_total_mb": snapshot["nvidia_smi_device_total_mb"],
        "nvidia_smi_preflight_utilization_gpu": snapshot["nvidia_smi_utilization_gpu"],
        "vram_estimated_required_mb": required_mb,
        "vram_estimated_headroom_mb": headroom_mb,
    }


def _nvidia_smi_preflight_limit(limit_mb: int) -> dict[str, int]:
    if not _cuda_available():
        return {
            "nvidia_smi_preflight_used_mb": 0,
            "nvidia_smi_device_total_mb": 0,
            "nvidia_smi_preflight_utilization_gpu": 0,
        }

    snapshot = _nvidia_smi_memory_snapshot()
    limit_mb = int(os.environ.get("AIGEN_NVIDIA_SMI_PREFLIGHT_LIMIT_MB", limit_mb))
    if snapshot["nvidia_smi_used_mb"] > limit_mb:
        raise KeyframeJobError(
            "GPU framebuffer is not clean enough for a high-resolution keyframe run: "
            f"{snapshot['nvidia_smi_used_mb']} MB used before model load; "
            f"limit is {limit_mb} MB"
        )
    return {
        "nvidia_smi_preflight_used_mb": snapshot["nvidia_smi_used_mb"],
        "nvidia_smi_device_total_mb": snapshot["nvidia_smi_device_total_mb"],
        "nvidia_smi_preflight_utilization_gpu": snapshot["nvidia_smi_utilization_gpu"],
    }


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def _nvidia_smi_memory_snapshot() -> dict[str, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    values = output.splitlines()[0].split(",")
    return {
        "nvidia_smi_used_mb": int(values[0].strip()),
        "nvidia_smi_device_total_mb": int(values[1].strip()),
        "nvidia_smi_utilization_gpu": int(values[2].strip()),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
