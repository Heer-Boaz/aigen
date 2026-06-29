from __future__ import annotations

import subprocess
from contextlib import closing
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image
from pydantic import ValidationError

from aigen.generation.runtime_diagnostics import cuda_memory_stats, elapsed_ms, synchronized_time
from aigen.image_assets import image_asset_json
from aigen.keyframe_image_ops import (
    mask_overlay,
    outside_mask_change,
    paste_refined_crop,
    save_contact_sheet,
)
from aigen.keyframe_memory import NvidiaSmiMemorySampler, nvidia_smi_preflight
from aigen.keyframe_polish_context import (
    load_polish_context,
)
from aigen.keyframe_polish_models import (
    POLISH_OPERATION_PROFILES,
    KeyframePolishError,
    KeyframePolishJobSpec,
    KeyframePolishPlan,
)
from aigen.keyframe_polish_masks import (
    PolishMaskPlan,
    build_polish_mask_plans,
    load_polish_mask_plans,
    mask_artifact_dir,
    mask_plan_json,
    planned_polish_outputs,
    polish_region_token_metadata,
    polish_region_variants,
)
from aigen.keyframe_polish_planner import (
    parse_polish_plan,
    polish_planner_prompt,
    save_polish_planner_evidence,
    validate_polish_plan,
)
from aigen.keyframe_polish_selection import (
    parse_polish_region_selection,
    polish_selection_prompt,
    save_polish_selection_evidence,
    validate_polish_region_selection,
)
from aigen.keyframe_profiles import KeyframeRefineProfile
from aigen.keyframe_refine import KontextInpaintRefiner
from aigen.manifest_io import read_json, resolve_existing_path, resolve_output_path, sha256_bytes, write_json
from aigen.progress import StatusReporter
from aigen.vlm_qwen import QwenVlm, QwenVlmConfig


KEYFRAME_POLISH_JOB_SCHEMA = "schemas/keyframe-polish-job.schema.json"
KEYFRAME_POLISH_PLAN_SCHEMA = "schemas/keyframe-polish-plan.schema.json"


def keyframe_polish_job_schema() -> dict[str, Any]:
    schema = KeyframePolishJobSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def keyframe_polish_plan_schema() -> dict[str, Any]:
    schema = KeyframePolishPlan.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_keyframe_polish_job(path: Path) -> KeyframePolishJobSpec:
    try:
        return KeyframePolishJobSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise KeyframePolishError(f"Invalid keyframe polish job {path}: {error}") from error


def load_keyframe_polish_plan(path: Path) -> KeyframePolishPlan:
    try:
        return KeyframePolishPlan.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise KeyframePolishError(f"Invalid keyframe polish plan {path}: {error}") from error


def plan_keyframe_polish(
    job_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    spec = load_keyframe_polish_job(job_path)
    context = load_polish_context(spec, job_path)
    plan_path = resolve_output_path(spec.plan.path, job_path.parent)
    output_dir = resolve_output_path(spec.output.directory, job_path.parent)
    return {
        "status": "planned",
        "job_path": job_path.resolve().as_posix(),
        "job_id": spec.id,
        "base": {
            "run_dir": context.base_dir.as_posix(),
            "candidate": spec.base.candidate,
            "image": image_asset_json(context.base_path),
            "base_job_id": context.result["job_id"],
        },
        "character": {
            "id": spec.character.id,
            "identity_primer": {
                "view": spec.character.identity_primer.view,
                **image_asset_json(context.identity_primer_path),
            },
        },
        "planner": spec.planner.model_dump(mode="json"),
        "plan_path": plan_path.as_posix(),
        "plan_exists": plan_path.is_file(),
        "micro_sweep": spec.micro_sweep.model_dump(mode="json"),
        "output": {
            **spec.output.model_dump(mode="json"),
            "directory": output_dir.as_posix(),
        },
        "acceptance": spec.acceptance.model_dump(mode="json"),
        "git_commit": _git_commit(project_root),
        "spec_sha256": sha256_bytes(job_path.read_bytes()),
    }


def diagnose_keyframe_polish(
    job_path: Path,
    *,
    config: QwenVlmConfig,
    project_root: Path,
) -> dict[str, Any]:
    spec = load_keyframe_polish_job(job_path)
    context = load_polish_context(spec, job_path)
    plan_path = resolve_output_path(spec.plan.path, job_path.parent)
    evidence_dir = plan_path.with_suffix("")
    evidence = save_polish_planner_evidence(context, evidence_dir)
    prompt = polish_planner_prompt(spec, context, evidence.prompt_order)
    with closing(QwenVlm(config)) as active_runner:
        raw_text = active_runner.judge_candidate(prompt, evidence.image_paths)
        plan = parse_polish_plan(raw_text)
        validate_polish_plan(spec, plan, context.base_image.size)
        payload = {
            "status": "completed",
            "job_path": job_path.resolve().as_posix(),
            "run_dir": context.base_dir.as_posix(),
            "candidate": spec.base.candidate,
            "git_commit": _git_commit(project_root),
            "planner": {
                "id": config.judge_id,
                "repo_id": config.repo_id,
                "revision": config.revision,
                "quantization": config.quantization,
                "min_pixels": config.min_pixels,
                "max_pixels": config.max_pixels,
                "temperature": config.temperature,
                "device_report": active_runner.device_report,
            },
            "evidence_images": [path.as_posix() for path in evidence.image_paths],
            "evidence_order": evidence.prompt_order,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "polish_plan": plan.model_dump(mode="json"),
            "raw_response": raw_text,
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, payload)
    return payload


def resolve_keyframe_polish_job(
    job_path: Path,
    profile: KeyframeRefineProfile,
    *,
    project_root: Path,
    check_outputs: bool,
) -> dict[str, Any]:
    spec = load_keyframe_polish_job(job_path)
    if spec.pipeline.profile != profile.name:
        raise KeyframePolishError(f"Job uses profile {spec.pipeline.profile}, but CLI resolved {profile.name}")
    context = load_polish_context(spec, job_path)
    plan_payload = read_json(resolve_existing_path(spec.plan.path, job_path.parent), label="keyframe polish plan")
    plan = KeyframePolishPlan.model_validate(plan_payload["polish_plan"])
    validate_polish_plan(spec, plan, context.base_image.size)
    output_dir = resolve_output_path(spec.output.directory, job_path.parent)
    outputs = planned_polish_outputs(spec, plan, output_dir)
    if check_outputs and not spec.output.overwrite:
        existing = [output for output in outputs if Path(output["path"]).exists()]
        if existing:
            raise KeyframePolishError(f"Output exists and overwrite=false: {existing[0]['path']}")
    token_metadata = polish_region_token_metadata(profile.model, plan)
    return {
        "kind": "resolved-keyframe-polish",
        "job_path": job_path.resolve().as_posix(),
        "job_id": spec.id,
        "profile": _profile_json(profile),
        "base": {
            "run_dir": context.base_dir.as_posix(),
            "candidate": spec.base.candidate,
            "image": image_asset_json(context.base_path),
            "base_job_id": context.result["job_id"],
        },
        "character": {
            "id": spec.character.id,
            "identity_primer": {
                "view": spec.character.identity_primer.view,
                **image_asset_json(context.identity_primer_path),
            },
        },
        "plan_path": resolve_existing_path(spec.plan.path, job_path.parent).as_posix(),
        "polish_plan": plan.model_dump(mode="json"),
        "operation_profiles": POLISH_OPERATION_PROFILES,
        "micro_sweep": spec.micro_sweep.model_dump(mode="json"),
        "output": {
            **spec.output.model_dump(mode="json"),
            "directory": output_dir.as_posix(),
            "files": outputs,
        },
        "acceptance": spec.acceptance.model_dump(mode="json"),
        "tokens": token_metadata,
        "git_commit": _git_commit(project_root),
        "spec_sha256": sha256_bytes(job_path.read_bytes()),
    }


def validate_keyframe_polish_job(
    job_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    resolved = plan_keyframe_polish(job_path, project_root=project_root)
    return {
        "status": "valid",
        "job_id": resolved["job_id"],
        "base": resolved["base"],
        "plan_path": resolved["plan_path"],
        "plan_exists": resolved["plan_exists"],
        "output": resolved["output"],
    }


def preview_keyframe_polish_job(
    job_path: Path,
    profile: KeyframeRefineProfile,
    *,
    project_root: Path,
) -> dict[str, Any]:
    resolved = resolve_keyframe_polish_job(job_path, profile, project_root=project_root, check_outputs=True)
    context = load_polish_context(load_keyframe_polish_job(job_path), job_path)
    mask_plans = build_polish_mask_plans(
        context.base_image,
        context.identity_primer,
        resolved,
    )
    return {
        **resolved,
        "mask_plan": [mask_plan_json(plan) for plan in mask_plans],
    }


def run_keyframe_polish_job(
    job_path: Path,
    profile: KeyframeRefineProfile,
    *,
    project_root: Path,
    progress: StatusReporter,
) -> dict[str, Any]:
    progress.phase("resolve polish job")
    spec = load_keyframe_polish_job(job_path)
    resolved = resolve_keyframe_polish_job(job_path, profile, project_root=project_root, check_outputs=True)
    context = load_polish_context(spec, job_path)
    output_dir = Path(resolved["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    progress.step("build polish masks")
    mask_plans = build_polish_mask_plans(
        context.base_image,
        context.identity_primer,
        resolved,
    )
    _save_mask_artifacts(mask_plans, output_dir)
    resolved = {**resolved, "mask_plan": [mask_plan_json(plan, output_dir) for plan in mask_plans]}
    write_json(output_dir / "resolved.json", resolved)
    if spec.output.save_debug_images:
        progress.phase("save polish debug images")
        _save_debug_images(context.base_image, mask_plans, output_dir)
    if not mask_plans:
        result = {
            "status": "completed",
            "job_id": spec.id,
            "spec_sha256": resolved["spec_sha256"],
            "git_commit": resolved["git_commit"],
            "models": resolved["profile"]["models"],
            "assets": {
                "identity_primer": resolved["character"]["identity_primer"],
                "base_image": resolved["base"]["image"],
            },
            "outputs": [],
            "effective_config": resolved,
            "timings_ms": {
                "model_load_ms": 0,
                "total_ms": 0,
            },
            "memory": {},
            "device_report": {},
            "environment": {},
        }
        write_json(output_dir / "result.json", result)
        return result

    memory_sampler = NvidiaSmiMemorySampler(nvidia_smi_preflight())
    memory_sampler.start()
    try:
        total_start = perf_counter()
        progress.step("load polish refiner")
        with closing(KontextInpaintRefiner(profile)) as refiner:
            torch_module = refiner.torch
            outputs = []
            variant_total = sum(len(polish_region_variants(spec, mask_plan)) for mask_plan in mask_plans)
            variant_index = 0
            for mask_index, mask_plan in enumerate(mask_plans, start=1):
                region_dir = output_dir / "regions" / mask_plan.region_id
                region_dir.mkdir(parents=True, exist_ok=True)
                for variant in polish_region_variants(spec, mask_plan):
                    variant_index += 1
                    progress.phase(
                        f"polish {mask_plan.region_id} ({mask_index}/{len(mask_plans)}), "
                        f"{variant['name']} ({variant_index}/{variant_total})"
                    )
                    variant_start = synchronized_time(torch_module)
                    base_crop = context.base_image.crop(mask_plan.crop_box)
                    mask_crop = mask_plan.feather_mask.crop(mask_plan.crop_box)
                    factor = mask_plan.parameters.crop_upsample_factor
                    if factor != 1.0:
                        up_size = (round(base_crop.width * factor), round(base_crop.height * factor))
                        base_crop = base_crop.resize(up_size, Image.Resampling.LANCZOS)
                        mask_crop = mask_crop.resize(up_size, Image.Resampling.LANCZOS)
                    refined_crop = refiner.refine(
                        base_crop=base_crop,
                        mask_crop=mask_crop,
                        reference_image=mask_plan.reference_card,
                        clip_prompt=variant["clip_prompt"],
                        t5_prompt=variant["t5_prompt"],
                        negative_prompt=variant["negative_prompt"],
                        true_cfg_scale=mask_plan.parameters.true_cfg_scale,
                        steps=mask_plan.parameters.steps,
                        guidance_scale=mask_plan.parameters.guidance_scale,
                        strength=variant["strength"],
                        max_sequence_length=mask_plan.parameters.max_sequence_length,
                        seed=variant["seed"],
                    )
                    polished = paste_refined_crop(
                        context.base_image,
                        refined_crop.convert("RGB"),
                        mask_plan.feather_mask,
                        mask_plan.crop_box,
                    )
                    path = region_dir / f"{variant['name']}.png"
                    polished.save(path)
                    outputs.append(
                        {
                            "name": variant["name"],
                            "region_id": mask_plan.region_id,
                            "label": mask_plan.label,
                            "seed": variant["seed"],
                            "strength": variant["strength"],
                            "path": path.as_posix(),
                            "timings_ms": {
                                "polish_ms": elapsed_ms(variant_start, synchronized_time(torch_module)),
                            },
                            "mask_change": outside_mask_change(context.base_image, polished, mask_plan.feather_mask),
                        }
                    )
                    progress.step(f"polished {variant['name']} ({variant_index}/{variant_total})")
            if spec.output.save_contact_sheet:
                progress.phase("write polish contact sheet")
                save_contact_sheet(outputs, output_dir / "contact_sheet.png", thumb_width=192, max_label_chars=28)
            memory = cuda_memory_stats(torch_module, "cuda") | memory_sampler.stop()
            progress.step("write polish result")
            result = {
                "status": "completed",
                "job_id": spec.id,
                "spec_sha256": resolved["spec_sha256"],
                "git_commit": resolved["git_commit"],
                "models": resolved["profile"]["models"],
                "assets": {
                    "identity_primer": resolved["character"]["identity_primer"],
                    "base_image": resolved["base"]["image"],
                },
                "outputs": outputs,
                "effective_config": resolved,
                "timings_ms": {
                    "model_load_ms": refiner.model_load_ms,
                    "total_ms": elapsed_ms(total_start, perf_counter()),
                },
                "memory": memory,
                "device_report": refiner.device_report,
                "environment": _generation_environment(torch_module),
            }
            write_json(output_dir / "result.json", result)
            return result
    finally:
        memory_sampler.stop()


def select_keyframe_polish(
    job_path: Path,
    *,
    config: QwenVlmConfig,
    project_root: Path,
) -> dict[str, Any]:
    spec = load_keyframe_polish_job(job_path)
    context = load_polish_context(spec, job_path)
    output_dir = resolve_output_path(spec.output.directory, job_path.parent)
    result = read_json(output_dir / "result.json", label="keyframe polish result")
    resolved = result["effective_config"]
    plan = KeyframePolishPlan.model_validate(resolved["polish_plan"])
    mask_plans = load_polish_mask_plans(context.base_image, context.identity_primer, resolved["mask_plan"])
    outputs_by_region = _outputs_by_region(result["outputs"])
    with closing(QwenVlm(config)) as active_runner:
        composite = context.base_image.copy()
        selections = []
        for mask_plan in mask_plans:
            candidates = [
                output
                for output in outputs_by_region[mask_plan.region_id]
                if not output["mask_change"]["hard_rejects"]["outside_feather_changed"]
            ]
            if not candidates:
                raise KeyframePolishError(f"No safe polish candidates remain for {mask_plan.region_id}")
            prompt = polish_selection_prompt(spec, plan, mask_plan, candidates)
            image_paths = save_polish_selection_evidence(
                context.base_image,
                context.identity_primer_path,
                mask_plan,
                candidates,
                output_dir,
            )
            raw_text = active_runner.judge_candidate(prompt, image_paths)
            selection = parse_polish_region_selection(raw_text)
            validate_polish_region_selection(selection, mask_plan.region_id, {candidate["name"] for candidate in candidates})
            selected = next(candidate for candidate in candidates if candidate["name"] == selection.best_variant)
            with Image.open(selected["path"]) as selected_image:
                selected_crop = selected_image.convert("RGB").crop(mask_plan.crop_box)
            composite = paste_refined_crop(
                composite,
                selected_crop,
                mask_plan.feather_mask,
                mask_plan.crop_box,
            )
            selections.append(
                {
                    **selection.model_dump(mode="json"),
                    "selected_path": selected["path"],
                    "raw_response": raw_text,
                    "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                }
            )
    final_path = output_dir / "final_composite.png"
    composite.save(final_path)
    payload = {
        "status": "completed",
        "job_id": spec.id,
        "git_commit": _git_commit(project_root),
        "final_composite": image_asset_json(final_path),
        "regions": selections,
        "judge": {
            "id": config.judge_id,
            "repo_id": config.repo_id,
            "revision": config.revision,
            "quantization": config.quantization,
            "min_pixels": config.min_pixels,
            "max_pixels": config.max_pixels,
            "temperature": config.temperature,
            "device_report": active_runner.device_report,
        },
    }
    write_json(output_dir / "polish_selection.json", payload)
    return payload


def _save_debug_images(base: Image.Image, mask_plans: list[PolishMaskPlan], output_dir: Path) -> None:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for mask_plan in mask_plans:
        region_dir = debug_dir / mask_plan.region_id
        region_dir.mkdir(parents=True, exist_ok=True)
        mask_plan.hard_mask.save(region_dir / "mask_hard.png")
        mask_plan.feather_mask.save(region_dir / "mask_feather.png")
        base.crop(mask_plan.crop_box).save(region_dir / "crop.png")
        mask_plan.reference_card.save(region_dir / "reference_detail_card.png")
        mask_overlay(base, mask_plan.feather_mask).save(region_dir / "mask_overlay.png")


def _save_mask_artifacts(mask_plans: list[PolishMaskPlan], output_dir: Path) -> None:
    for mask_plan in mask_plans:
        region_dir = mask_artifact_dir(output_dir, mask_plan.region_id)
        region_dir.mkdir(parents=True, exist_ok=True)
        mask_plan.hard_mask.save(region_dir / "hard.png")
        mask_plan.feather_mask.save(region_dir / "feather.png")


def _outputs_by_region(outputs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for output in outputs:
        grouped.setdefault(output["region_id"], []).append(output)
    return grouped


def _git_commit(project_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()


def _profile_json(profile: KeyframeRefineProfile) -> dict[str, Any]:
    models = {
        "kontext": {
            **profile.model_revisions["kontext"],
            "path": profile.model,
        },
        "nunchaku_transformer": {
            **profile.model_revisions["nunchaku_transformer"],
            "path": profile.nunchaku_transformer_model.resolve().as_posix(),
        },
    }
    return {
        "name": profile.name,
        "dtype": profile.dtype,
        "attention_impl": profile.attention_impl,
        "pipeline_cpu_offload": profile.pipeline_cpu_offload,
        "vae_tiling": profile.vae_tiling,
        "models": models,
    }


def _generation_environment(torch_module: Any) -> dict[str, Any]:
    import diffusers

    environment = {
        "torch_version": torch_module.__version__,
        "torch_cuda_version": torch_module.version.cuda,
        "diffusers_version": diffusers.__version__,
    }
    if torch_module.cuda.is_available():
        environment["gpu_name"] = torch_module.cuda.get_device_name(0)
        environment["compute_capability"] = list(torch_module.cuda.get_device_capability(0))
    return environment
