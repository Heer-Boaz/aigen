from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from aigen.generation.kontext_pose_control import CharacterKontextPoseSession, KontextControlCondition
from aigen.generation.runtime_diagnostics import cuda_memory_stats, synchronized_time
from aigen.keyframe_image_ops import save_contact_sheet
from aigen.keyframe_job_models import ControlConditionSpec, KeyframeJobSpec, load_keyframe_job
from aigen.keyframe_memory import NvidiaSmiMemorySampler, nvidia_smi_keyframe_preflight
from aigen.keyframe_score import DEFAULT_SCORER_ID, KeyframeScoreConfig, score_keyframe_run
from aigen.keyframes import _generation_environment, resolve_keyframe_spec
from aigen.manifest_io import read_json, sha256_bytes, write_json


CONTROL_AUDIT_SCHEMA_VERSION = 1
CONTROL_AUDIT_SCORER_ID = "control-audit"
CONTROL_AUDIT_MINIMAL_PROMPT = (
    "Same character, full body, clean neutral background, preserve outfit and side-view character design."
)
MATERIAL_SCORE_DELTA = 0.05
LATENT_DELTA_EPSILON = 1e-5


class KeyframeControlAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class ControlAuditVariant:
    name: str
    conditions: list[ControlConditionSpec]


def run_keyframe_control_audit(
    run_dir: Path,
    *,
    project_root: Path,
    seed: int | None = None,
    scorer_id: str = CONTROL_AUDIT_SCORER_ID,
    score_runner: Any | None = None,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.resolve()
    source_result = read_json(resolved_run_dir / "result.json", label="source keyframe result")
    source_config = source_result["effective_config"]
    source_job_path = Path(source_config["job_path"])
    source_spec = load_keyframe_job(source_job_path)
    source_profile = _profile_from_resolved(source_config)
    source_resolved = resolve_keyframe_spec(
        source_spec,
        source_job_path,
        source_profile,
        project_root=project_root,
        check_outputs=False,
    )
    audit_seed = seed if seed is not None else int(source_result["outputs"][0]["seed"])
    variants = _audit_variants(source_spec)
    audit_dir = resolved_run_dir / "control_audit"
    if audit_dir.exists():
        shutil.rmtree(audit_dir)
    audit_dir.mkdir(parents=True)
    _save_audit_conditions(source_resolved, audit_dir)

    memory_sampler = NvidiaSmiMemorySampler(nvidia_smi_keyframe_preflight(source_resolved["vram_plan"]))
    memory_sampler.start()
    memory_stats: dict[str, Any] | None = None
    session = CharacterKontextPoseSession(
        source_profile.model,
        source_profile.controlnet_model,
        dtype=source_profile.dtype,
        nunchaku_transformer_model=source_profile.nunchaku_transformer_model,
        attention_impl=source_profile.attention_impl,
        pipeline_cpu_offload=source_profile.pipeline_cpu_offload,
        nunchaku_layer_offload=source_profile.nunchaku_layer_offload,
        vae_tiling=source_profile.vae_tiling,
    )
    try:
        torch = session.torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats("cuda")
        total_start = synchronized_time(torch)
        prepared = session.prepare(
            reference_image=Path(source_resolved["character"]["identity_primer"]["path"]),
            pose_image=Path(source_resolved["assets"]["pose"]["path"]),
            prompt=CONTROL_AUDIT_MINIMAL_PROMPT,
            t5_prompt=CONTROL_AUDIT_MINIMAL_PROMPT,
            negative_prompt=None,
            true_cfg_scale=1.0,
            width=source_spec.canvas.width,
            height=source_spec.canvas.height,
            reference_max_area=source_spec.canvas.reference_max_area,
            max_sequence_length=source_spec.canvas.max_sequence_length,
            steps=source_spec.sampling.steps,
            guidance_scale=source_spec.sampling.guidance_scale,
            seed=audit_seed,
        )
        control_images, control_repeats = _prepare_audit_control_images(
            session,
            prepared,
            source_resolved,
            audit_seed,
            variants,
        )
        masks = _prepare_audit_masks(session, prepared, source_resolved, variants)
        control_tensor_metadata = _save_control_tensors(torch, control_images, audit_dir / "control_tensors")
        control_debug_metadata = _save_control_debug_images(
            session,
            prepared,
            source_resolved,
            control_images,
            control_repeats,
            audit_dir / "control_debug",
        )
        session.pipeline.maybe_free_model_hooks()

        denoised = []
        for variant in variants:
            result = session.pipeline.denoise_prepared(
                prepared,
                name=variant.name,
                seed=audit_seed,
                controlnet_conditioning_scale=variant.conditions[0].scale,
                control_guidance_start=variant.conditions[0].start,
                control_guidance_end=variant.conditions[0].end,
                collect_control_stats=True,
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
                    for condition in variant.conditions
                ],
            )
            denoised.append(result)
        latent_deltas = _latent_deltas(torch, denoised)
        session.pipeline.maybe_free_model_hooks()
        images, decode_ms = session.decode_many(prepared, denoised, chunk_size=1)

        outputs = []
        for image, result, variant in zip(images, denoised, variants, strict=True):
            output_path = audit_dir / f"{variant.name}.png"
            image.save(output_path)
            outputs.append(
                {
                    "name": variant.name,
                    "seed": result.seed,
                    "path": output_path.as_posix(),
                    "controlnet_active_steps": result.controlnet_active_steps,
                    "controlnet_step_ms": result.controlnet_step_ms,
                    "transformer_step_ms": result.transformer_step_ms,
                    "controlnet_metadata": result.controlnet_metadata,
                    "timings_ms": result.timings_ms,
                    "latent_delta_vs_control_off": latent_deltas[variant.name],
                }
            )

        save_contact_sheet(outputs, audit_dir / "contact_sheet.png", thumb_width=256, label_x=8)
        run_result = {
            "status": "completed",
            "job_id": f"{source_result['job_id']}.control-audit",
            "source_run_dir": resolved_run_dir.as_posix(),
            "source_job_id": source_result["job_id"],
            "git_commit": source_resolved["git_commit"],
            "models": source_resolved["profile"]["models"],
            "assets": source_resolved["assets"] | {"identity_primer": source_resolved["character"]["identity_primer"]},
            "outputs": outputs,
            "effective_config": {
                **source_resolved,
                "prompt": {
                    "clip": CONTROL_AUDIT_MINIMAL_PROMPT,
                    "t5": CONTROL_AUDIT_MINIMAL_PROMPT,
                    "true_cfg_scale": 1.0,
                },
                "audit_variants": [_variant_json(variant, source_spec.sampling.steps) for variant in variants],
            },
            "token_metadata": prepared.token_metadata,
            "timings_ms": {
                "model_load_ms": session.model_load_ms,
                "decode_ms": decode_ms,
                "total_ms": (synchronized_time(torch) - total_start) * 1000,
            },
            "memory": cuda_memory_stats(torch, "cuda") | memory_sampler.stop(),
            "environment": _generation_environment(torch, session.pipeline),
        }
        memory_stats = run_result["memory"]
        write_json(audit_dir / "result.json", run_result)
    finally:
        session.close()
        if memory_stats is None:
            memory_sampler.stop()

    if score_runner is None:
        score_runner = score_keyframe_run
    score_result = score_runner(
        audit_dir,
        KeyframeScoreConfig(scorer_id=scorer_id),
        project_root=project_root,
    )
    audit_payload = _audit_payload(
        source_result,
        audit_dir,
        audit_seed,
        variants,
        run_result,
        score_result,
        control_tensor_metadata,
        control_debug_metadata,
        scorer_id,
    )
    write_json(audit_dir / "audit.json", audit_payload)
    return audit_payload


def _profile_from_resolved(resolved: dict[str, Any]) -> Any:
    from aigen.keyframe_profiles import KeyframeProfile

    profile = resolved["profile"]
    return KeyframeProfile(
        name=profile["name"],
        model=profile["models"]["kontext"]["path"],
        controlnet_model=profile["models"]["controlnet"]["path"],
        nunchaku_transformer_model=Path(profile["models"]["nunchaku_transformer"]["path"]),
        dtype=profile["dtype"],
        attention_impl=profile["attention_impl"],
        pipeline_cpu_offload=profile["pipeline_cpu_offload"],
        nunchaku_layer_offload=profile["nunchaku_layer_offload"],
        vae_tiling=profile["vae_tiling"],
        model_revisions={
            "kontext": {
                key: value for key, value in profile["models"]["kontext"].items() if key != "path"
            },
            "controlnet": {
                key: value for key, value in profile["models"]["controlnet"].items() if key != "path"
            },
            "nunchaku_transformer": {
                key: value
                for key, value in profile["models"]["nunchaku_transformer"].items()
                if key != "path"
            },
        },
    )


def _audit_variants(spec: KeyframeJobSpec) -> list[ControlAuditVariant]:
    return [
        ControlAuditVariant(
            "control_off",
            [
                _condition("pose", "pose", "pose", 0.0, 0.0, 0.65),
                _condition("canny_lineart", "canny", "canny_lineart", 0.0, 0.0, 0.80),
                _condition("softedge", "softedge", "softedge", 0.0, 0.0, 0.80),
                _condition("gray", "gray", "gray", 0.0, 0.0, 0.80),
            ],
        ),
        ControlAuditVariant("current_contour_baseline", [condition.model_copy() for condition in spec.conditions]),
        ControlAuditVariant("pose_only_strong", [_condition("pose", "pose", "pose", 0.90, 0.0, 0.65)]),
        ControlAuditVariant(
            "canny_lineart_unmasked",
            [_condition("canny_lineart", "canny", "canny_lineart", 0.70, 0.0, 0.80)],
        ),
        ControlAuditVariant(
            "softedge_unmasked",
            [_condition("softedge", "softedge", "softedge", 0.70, 0.0, 0.80)],
        ),
        ControlAuditVariant("gray_unmasked", [_condition("gray", "gray", "gray", 0.90, 0.0, 0.80)]),
        ControlAuditVariant(
            "pose_plus_softedge",
            [
                _condition("pose", "pose", "pose", 0.80, 0.0, 0.65),
                _condition("softedge", "softedge", "softedge", 0.55, 0.0, 0.80),
            ],
        ),
        ControlAuditVariant(
            "pose_plus_gray",
            [
                _condition("pose", "pose", "pose", 0.80, 0.0, 0.65),
                _condition("gray", "gray", "gray", 0.65, 0.0, 0.80),
            ],
        ),
        ControlAuditVariant(
            "pose_plus_canny_lineart",
            [
                _condition("pose", "pose", "pose", 0.80, 0.0, 0.65),
                _condition("canny_lineart", "canny", "canny_lineart", 0.55, 0.0, 0.80),
            ],
        ),
    ]


def _condition(
    name: str,
    condition_type: str,
    image: str,
    scale: float,
    start: float,
    end: float,
) -> ControlConditionSpec:
    return ControlConditionSpec(name=name, type=condition_type, image=image, scale=scale, start=start, end=end)


def _prepare_audit_control_images(
    session: CharacterKontextPoseSession,
    prepared: Any,
    resolved: dict[str, Any],
    seed: int,
    variants: list[ControlAuditVariant],
) -> tuple[dict[str, Any], dict[str, bool]]:
    control_images = {"pose": prepared.control_image}
    control_repeats = {"pose": prepared.controlnet_blocks_repeat}
    for condition in _unique_conditions(variants):
        if condition.image in control_images:
            continue
        with Image.open(resolved["assets"][condition.image]["path"]) as image:
            control_image = image.convert("RGB")
        prepared_control, blocks_repeat, _prepare_ms = session.prepare_control_condition(
            prepared,
            pose_image=control_image,
            seed=seed,
        )
        control_images[condition.image] = prepared_control
        control_repeats[condition.image] = blocks_repeat
    return control_images, control_repeats


def _prepare_audit_masks(
    session: CharacterKontextPoseSession,
    prepared: Any,
    resolved: dict[str, Any],
    variants: list[ControlAuditVariant],
) -> dict[str, Any]:
    masks = {}
    for condition in _unique_conditions(variants):
        if not condition.residual_mask or condition.residual_mask in masks:
            continue
        with Image.open(resolved["assets"][condition.residual_mask]["path"]) as image:
            mask_image = image.convert("RGB")
        masks[condition.residual_mask] = session.prepare_residual_mask(prepared, mask_image)
    return masks


def _unique_conditions(variants: list[ControlAuditVariant]) -> list[ControlConditionSpec]:
    conditions = {}
    for variant in variants:
        for condition in variant.conditions:
            conditions[condition.image, condition.residual_mask] = condition
    return list(conditions.values())


def _variant_json(variant: ControlAuditVariant, steps: int) -> dict[str, Any]:
    return {
        "name": variant.name,
        "conditions": [
            condition.model_dump(mode="json", exclude_none=True) | {"active_steps": _active_steps(condition, steps)}
            for condition in variant.conditions
        ],
    }


def _active_steps(condition: ControlConditionSpec, steps: int) -> int:
    return sum(
        condition.scale > 0.0
        and condition.start <= index / steps
        and (index + 1) / steps <= condition.end
        for index in range(steps)
    )


def _save_audit_conditions(resolved: dict[str, Any], audit_dir: Path) -> None:
    condition_dir = audit_dir / "conditions"
    condition_dir.mkdir(parents=True)
    for name, asset in resolved["assets"].items():
        shutil.copy2(asset["path"], condition_dir / f"{name}{Path(asset['path']).suffix}")


def _save_control_tensors(torch: Any, control_images: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True)
    metadata = {}
    for name, tensor in control_images.items():
        tensor_path = output_dir / f"{name}.pt"
        torch.save(tensor.detach().cpu(), tensor_path)
        values = tensor.detach().float()
        metadata[name] = {
            "path": tensor_path.as_posix(),
            "sha256": _tensor_sha256(torch, tensor),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "min": float(values.min().cpu().item()),
            "max": float(values.max().cpu().item()),
            "mean": float(values.mean().cpu().item()),
            "std": float(values.std(unbiased=False).cpu().item()),
        }
    return metadata


def _save_control_debug_images(
    session: CharacterKontextPoseSession,
    prepared: Any,
    resolved: dict[str, Any],
    control_images: dict[str, Any],
    control_repeats: dict[str, bool],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True)
    metadata = {}
    for name, tensor in control_images.items():
        asset = resolved["assets"][name]
        suffix = Path(asset["path"]).suffix
        original_path = output_dir / f"{name}_original{suffix}"
        preprocessed_path = output_dir / f"{name}_preprocessed.png"
        roundtrip_path = output_dir / f"{name}_vae_roundtrip.png"
        shutil.copy2(asset["path"], original_path)
        with Image.open(asset["path"]) as image:
            image.convert("RGB").resize(
                (prepared.width, prepared.height),
                Image.Resampling.LANCZOS,
            ).save(preprocessed_path)
        session.decode_control_condition(
            prepared,
            tensor,
            controlnet_blocks_repeat=control_repeats[name],
        ).save(roundtrip_path)
        metadata[name] = {
            "original": original_path.as_posix(),
            "preprocessed": preprocessed_path.as_posix(),
            "vae_roundtrip": roundtrip_path.as_posix(),
            "controlnet_blocks_repeat": control_repeats[name],
        }
    return metadata


def _tensor_sha256(torch: Any, tensor: Any) -> str:
    cpu_tensor = tensor.detach().cpu().contiguous()
    if cpu_tensor.dtype == torch.bfloat16:
        cpu_tensor = cpu_tensor.view(torch.int16)
    return sha256_bytes(cpu_tensor.numpy().tobytes())


def _latent_deltas(torch: Any, denoised: list[Any]) -> dict[str, dict[str, float]]:
    control_off = next(result.latents for result in denoised if result.name == "control_off").detach().float().cpu()
    deltas = {}
    for result in denoised:
        delta = result.latents.detach().float().cpu() - control_off
        deltas[result.name] = {
            "l2": float(torch.linalg.vector_norm(delta).item()),
            "mean_abs": float(delta.abs().mean().item()),
        }
    return deltas


def _audit_payload(
    source_result: dict[str, Any],
    audit_dir: Path,
    seed: int,
    variants: list[ControlAuditVariant],
    run_result: dict[str, Any],
    score_result: dict[str, Any],
    control_tensor_metadata: dict[str, Any],
    control_debug_metadata: dict[str, Any],
    scorer_id: str,
) -> dict[str, Any]:
    score_by_name = {candidate["candidate"]: candidate for candidate in score_result["candidates"]}
    score_deltas = _score_deltas(score_by_name)
    pass_status = _audit_pass_status(run_result, score_by_name, score_deltas)
    return {
        "schema_version": CONTROL_AUDIT_SCHEMA_VERSION,
        "status": "passed" if pass_status["passed"] else "failed",
        "passed": pass_status["passed"],
        "blockers": pass_status["blockers"],
        "source_run_dir": source_result["effective_config"]["output"]["directory"],
        "source_job_id": source_result["job_id"],
        "seed": seed,
        "minimal_prompt": {
            "clip": CONTROL_AUDIT_MINIMAL_PROMPT,
            "t5": CONTROL_AUDIT_MINIMAL_PROMPT,
            "true_cfg_scale": 1.0,
        },
        "variants": [_variant_json(variant, source_result["effective_config"]["sampling"]["steps"]) for variant in variants],
        "control_tensors": control_tensor_metadata,
        "control_debug": control_debug_metadata,
        "score_deltas_vs_control_off": score_deltas,
        "plain_flux_controlnet_strong": {
            "status": "not_run",
            "reason": "No plain FLUX ControlNet generation owner exists in this codebase yet.",
        },
        "outputs": {
            "audit": (audit_dir / "audit.json").as_posix(),
            "result": (audit_dir / "result.json").as_posix(),
            "contact_sheet": (audit_dir / "contact_sheet.png").as_posix(),
            "scores": score_result["outputs"]["scores"],
            "ranked_contact_sheet": score_result["outputs"]["ranked_contact_sheet"],
            "condition_evidence_ranked": score_result["outputs"]["condition_evidence_ranked"],
            "pose_evidence_ranked": score_result["outputs"]["pose_evidence_ranked"],
        },
        "generation_outputs": run_result["outputs"],
        "memory": run_result["memory"],
        "timings_ms": run_result["timings_ms"],
    }


def _score_deltas(score_by_name: dict[str, Any]) -> dict[str, dict[str, float]]:
    control_off = score_by_name["control_off"]["scores"]
    deltas = {}
    for name, candidate in score_by_name.items():
        deltas[name] = {
            score_name: float(score - control_off[score_name])
            for score_name, score in candidate["scores"].items()
        }
    return deltas


def _audit_pass_status(
    run_result: dict[str, Any],
    score_by_name: dict[str, Any],
    score_deltas: dict[str, dict[str, float]],
) -> dict[str, Any]:
    strong_names = (
        "pose_only_strong",
        "canny_lineart_unmasked",
        "softedge_unmasked",
        "gray_unmasked",
        "pose_plus_softedge",
        "pose_plus_gray",
        "pose_plus_canny_lineart",
    )
    latent_deltas = {
        output["name"]: output["latent_delta_vs_control_off"]["mean_abs"]
        for output in run_result["outputs"]
    }
    best_condition_delta = max(score_deltas[name]["condition"] for name in strong_names if name in score_deltas)
    best_pose_delta = max(score_deltas[name]["pose"] for name in strong_names if name in score_deltas)
    strongest_latent_delta = max(latent_deltas[name] for name in strong_names if name in latent_deltas)
    blockers = []
    if strongest_latent_delta <= LATENT_DELTA_EPSILON:
        blockers.append("strong_control_latents_match_control_off")
    if max(best_condition_delta, best_pose_delta) <= MATERIAL_SCORE_DELTA:
        blockers.append("strong_control_not_materially_closer_to_conditions")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "best_condition_delta": best_condition_delta,
        "best_pose_delta": best_pose_delta,
        "strongest_latent_delta": strongest_latent_delta,
        "control_off_scores": score_by_name["control_off"]["scores"],
    }
