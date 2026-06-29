from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from aigen.generation.kontext_pose_control import (
    CharacterKontextPoseSession,
    KontextControlCondition,
    _generation_environment,
)
from aigen.generation.runtime_diagnostics import cuda_memory_stats, synchronized_time
from aigen.image_assets import image_asset_json
from aigen.keyframe_memory import (
    NvidiaSmiMemorySampler,
    keyframe_vram_plan,
    nvidia_smi_keyframe_preflight,
    planned_token_metadata,
)
from aigen.keyframe_job_models import (
    KEYFRAME_SCHEMA_VERSION,
    ControlConditionSpec,
    KeyframeJobError,
    KeyframeJobSpec,
    load_keyframe_job,
)
from aigen.keyframe_image_ops import save_contact_sheet
from aigen.keyframe_profiles import KeyframeProfile
from aigen.manifest_io import (
    resolve_existing_path,
    resolve_output_path,
    sha256_bytes,
    write_json,
)
from aigen.prompt_tokens import count_kontext_prompt_tokens


def resolve_keyframe_job(
    job_path: Path,
    profile: KeyframeProfile,
    *,
    project_root: Path,
    check_outputs: bool,
    require_active_conditions: bool = True,
) -> dict[str, Any]:
    spec = load_keyframe_job(job_path)
    return resolve_keyframe_spec(
        spec,
        job_path,
        profile,
        project_root=project_root,
        check_outputs=check_outputs,
        require_active_conditions=require_active_conditions,
    )


def resolve_keyframe_spec(
    spec: KeyframeJobSpec,
    job_path: Path,
    profile: KeyframeProfile,
    *,
    project_root: Path,
    check_outputs: bool,
    require_active_conditions: bool = True,
) -> dict[str, Any]:
    if spec.pipeline.profile != profile.name:
        raise KeyframeJobError(f"Job uses profile {spec.pipeline.profile}, but CLI resolved {profile.name}")

    assets = _resolve_assets(spec, job_path.parent)
    _validate_conditions(spec, assets)
    _validate_asset_dimensions(spec, assets)
    _validate_masks(spec, assets)
    output_dir = resolve_output_path(spec.output.directory, job_path.parent)
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
    if require_active_conditions and inactive:
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
        "spec_sha256": sha256_bytes(job_path.read_bytes()),
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


def run_keyframe_audit_variant_job(job_path: Path, profile: KeyframeProfile, *, project_root: Path) -> dict[str, Any]:
    spec = load_keyframe_job(job_path)
    return run_keyframe_spec(
        spec,
        job_path,
        profile,
        project_root=project_root,
        require_active_conditions=False,
    )


def run_keyframe_spec(
    spec: KeyframeJobSpec,
    job_path: Path,
    profile: KeyframeProfile,
    *,
    project_root: Path,
    require_active_conditions: bool = True,
) -> dict[str, Any]:
    resolved = resolve_keyframe_spec(
        spec,
        job_path,
        profile,
        project_root=project_root,
        check_outputs=True,
        require_active_conditions=require_active_conditions,
    )
    memory_sampler = NvidiaSmiMemorySampler(nvidia_smi_keyframe_preflight(resolved["vram_plan"]))
    output_dir = Path(resolved["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved.json", resolved)
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
            session.pipeline.maybe_free_model_hooks()
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
                save_contact_sheet(outputs, output_dir / "contact_sheet.png", thumb_width=256, label_x=8)
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
            write_json(output_dir / "result.json", result_json)
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
        "identity_primer": image_asset_json(resolve_existing_path(spec.character.identity_primer.path, base_dir)),
        "pose": image_asset_json(resolve_existing_path(spec.assets.pose.path, base_dir)),
    }
    for name in (
        "contour",
        "canny_lineart",
        "boundary_mask",
        "depth",
        "softedge",
        "gray",
        "filled_silhouette",
        "full_silhouette_mask",
        "arm_hand_mask",
    ):
        asset = getattr(spec.assets, name)
        if asset:
            assets[name] = image_asset_json(resolve_existing_path(asset.path, base_dir))
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
    return planned_token_metadata(
        identity_width=assets["identity_primer"]["width"],
        identity_height=assets["identity_primer"]["height"],
        canvas_width=spec.canvas.width,
        canvas_height=spec.canvas.height,
        reference_max_area=spec.canvas.reference_max_area,
        max_sequence_length=spec.canvas.max_sequence_length,
    )


def _vram_plan(spec: KeyframeJobSpec, token_metadata: dict[str, int]) -> dict[str, Any]:
    return keyframe_vram_plan(
        canvas_width=spec.canvas.width,
        canvas_height=spec.canvas.height,
        true_cfg_scale=spec.prompt.true_cfg_scale,
        token_metadata=token_metadata,
    )


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
        with Image.open(resolved["assets"][condition.image]["path"]) as image:
            pose_image = image.convert("RGB")
        control_image, blocks_repeat, _prepare_ms = session.prepare_control_condition(
            prepared,
            pose_image=pose_image,
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
        with Image.open(resolved["assets"][condition.residual_mask]["path"]) as image:
            residual_mask = image.convert("RGB")
        masks[condition.residual_mask] = session.prepare_residual_mask(
            prepared,
            residual_mask,
        )
    return masks


def _save_conditions(resolved: dict[str, Any], output_dir: Path) -> None:
    condition_dir = output_dir / "conditions"
    condition_dir.mkdir(parents=True, exist_ok=True)
    for name, asset in resolved["assets"].items():
        shutil.copy2(asset["path"], condition_dir / f"{name}{Path(asset['path']).suffix}")


def _git_commit(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
    ).strip()
