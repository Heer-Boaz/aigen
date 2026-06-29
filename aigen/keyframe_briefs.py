from __future__ import annotations

from pathlib import Path
from typing import Any

from aigen import keyframe_brief_planner
from aigen.keyframe_examples import KeyframeExampleExtractionConfig, extract_keyframe_example
from aigen.keyframe_judge import KeyframeJudgeConfig, judge_keyframe_run
from aigen.keyframe_brief_models import (
    BriefControlPlanSpec,
    KeyframeBriefError,
    KeyframeBriefPlanSpec,
    KeyframeBriefSpec,
    load_keyframe_brief,
    load_keyframe_brief_plan,
)
from aigen.keyframe_job_models import (
    AcceptanceSpec,
    AssetSpec,
    CharacterSpec,
    ControlConditionSpec,
    KeyframeJobSpec,
    KeyframeSpec,
    OutputSpec,
    PathSpec,
    VariantSpec,
)
from aigen.keyframes import (
    plan_keyframe_job,
    run_keyframe_job,
)
from aigen.keyframe_polish import (
    diagnose_keyframe_polish,
    run_keyframe_polish_job,
    select_keyframe_polish,
)
from aigen.keyframe_score import KeyframeScoreConfig, score_keyframe_run, select_scored_keyframe_run
from aigen.manifest_io import (
    relative_path,
    resolve_existing_path,
    resolve_output_path,
    schema_reference,
    write_json,
)
from aigen.runtime_profiles import keyframe_profile_for_name, keyframe_refine_profile_for_name


def materialize_keyframe_brief(
    brief_path: Path,
    *,
    project_root: Path,
    pose_device: str = "cpu",
) -> dict[str, Any]:
    spec = load_keyframe_brief(brief_path)
    plan_path = resolve_existing_path(spec.output.plan_path, brief_path.parent)
    plan = load_keyframe_brief_plan(plan_path)
    profile = keyframe_profile_for_name(spec.pipeline.profile)
    extraction = extract_keyframe_example(
        KeyframeExampleExtractionConfig(
            source=resolve_existing_path(spec.example.path, brief_path.parent),
            output_dir=resolve_output_path(spec.output.assets_directory, brief_path.parent),
            name=spec.example.name,
            width=plan.canvas.width,
            height=plan.canvas.height,
            mirror_x=spec.example.mirror_x,
            pose_device=pose_device,
        )
    )
    job_path = resolve_output_path(spec.output.job_path, brief_path.parent)
    job = _keyframe_job_from_brief(spec, plan, extraction, brief_path.parent, job_path, project_root)
    write_json(job_path, job.model_dump(mode="json", by_alias=True, exclude_none=True))
    resolved = plan_keyframe_job(job_path, profile, project_root=project_root)
    return {
        "status": "materialized",
        "brief_id": spec.id,
        "plan_path": plan_path.as_posix(),
        "job_path": job_path.as_posix(),
        "extraction": extraction,
        "resolved": {
            "job_id": resolved["job_id"],
            "tokens": resolved["tokens"],
            "condition_plan": resolved["condition_plan"],
            "vram_plan": resolved["vram_plan"],
            "outputs": resolved["output"]["files"],
        },
    }


def run_keyframe_brief(
    brief_path: Path,
    *,
    project_root: Path,
    pose_device: str = "cpu",
) -> dict[str, Any]:
    materialized = materialize_keyframe_brief(brief_path, project_root=project_root, pose_device=pose_device)
    job_path = Path(materialized["job_path"])
    profile = keyframe_profile_for_name(load_keyframe_brief(brief_path).pipeline.profile)
    result = run_keyframe_job(job_path, profile, project_root=project_root)
    return {
        "status": "completed",
        "brief_id": materialized["brief_id"],
        "job_path": job_path.as_posix(),
        "run_dir": result["effective_config"]["output"]["directory"],
        "result": result,
    }


def execute_keyframe_brief(
    brief_path: Path,
    config: KeyframeJudgeConfig,
    *,
    project_root: Path,
    pose_device: str = "cpu",
    runner: Any | None = None,
) -> dict[str, Any]:
    planned = keyframe_brief_planner.plan_keyframe_brief(brief_path, config, project_root=project_root, runner=runner)
    generated = run_keyframe_brief(brief_path, project_root=project_root, pose_device=pose_device)
    run_dir = Path(generated["run_dir"])
    score = score_keyframe_run(run_dir, KeyframeScoreConfig(), project_root=project_root)
    judge = judge_keyframe_run(run_dir, config, project_root=project_root)
    selection = select_scored_keyframe_run(run_dir, top_k=planned["scoring"]["top_k"])
    polish = _polish_selected_candidates(
        brief_path,
        config,
        project_root=project_root,
        selected=selection["selected"],
    )
    return {
        "status": "completed",
        "brief_id": planned["brief_id"],
        "plan_path": planned["plan_path"],
        "job_path": generated["job_path"],
        "run_dir": generated["run_dir"],
        "score": score,
        "judge": judge,
        "selection": selection,
        "polish": polish,
        "result": generated["result"],
    }


def _polish_selected_candidates(
    brief_path: Path,
    config: KeyframeJudgeConfig,
    *,
    project_root: Path,
    selected: list[str],
) -> list[dict[str, Any]]:
    spec = load_keyframe_brief(brief_path)
    plan = load_keyframe_brief_plan(resolve_existing_path(spec.output.plan_path, brief_path.parent))
    profile = keyframe_refine_profile_for_name(plan.polish.profile)
    outputs = []
    for candidate in selected:
        job_path = _write_polish_job(spec, plan, brief_path, candidate, project_root)
        diagnose = diagnose_keyframe_polish(job_path, config=config, project_root=project_root)
        result = run_keyframe_polish_job(job_path, profile, project_root=project_root)
        selection = select_keyframe_polish(job_path, config=config, project_root=project_root)
        outputs.append(
            {
                "candidate": candidate,
                "job_path": job_path.as_posix(),
                "plan_path": diagnose["plan_path"],
                "run_dir": result["effective_config"]["output"]["directory"],
                "final_composite": selection["final_composite"],
            }
        )
    return outputs


def _write_polish_job(
    spec: KeyframeBriefSpec,
    plan: KeyframeBriefPlanSpec,
    brief_path: Path,
    candidate: str,
    project_root: Path,
) -> Path:
    base_output_dir = resolve_output_path(spec.output.plan_path, brief_path.parent).parent / "polish" / candidate
    job_path = base_output_dir / "job.json"
    payload = {
        "$schema": schema_reference(job_path, project_root / "schemas/keyframe-polish-job.schema.json"),
        "schema_version": 1,
        "kind": "keyframe-polish",
        "id": f"{spec.id}.polish.{candidate}",
        "pipeline": {"profile": plan.polish.profile},
        "base": {
            "run_dir": relative_path(resolve_output_path(spec.generation.output_directory, brief_path.parent), job_path.parent),
            "candidate": candidate,
        },
        "character": {
            "id": spec.character.id,
            "identity_primer": plan.identity_primer.model_dump(mode="json"),
        },
        "plan": {"path": "plan.json"},
        "planner": {"max_regions": plan.polish.max_regions},
        "micro_sweep": {
            "strength_offsets": plan.polish.strength_offsets,
            "seed_offsets": plan.polish.seed_offsets,
        },
        "output": {
            "directory": "outputs",
            "overwrite": True,
            "save_debug_images": True,
            "save_contact_sheet": True,
        },
        "acceptance": {
            "manual": [
                *plan.scoring.checks,
                "local polish preserves pose and silhouette",
                "outside feathered mask stays unchanged",
            ]
        },
    }
    write_json(job_path, payload)
    return job_path


def _keyframe_job_from_brief(
    spec: KeyframeBriefSpec,
    plan: KeyframeBriefPlanSpec,
    extraction: dict[str, Any],
    brief_base_dir: Path,
    job_path: Path,
    project_root: Path,
) -> KeyframeJobSpec:
    assets = _assets_from_plan(plan, extraction, job_path.parent)
    conditions = [_condition_from_plan(control) for control in plan.controls]
    job = KeyframeJobSpec(
        **{
            "$schema": schema_reference(job_path, project_root / "schemas/keyframe-job.schema.json"),
            "schema_version": 1,
            "kind": "character-keyframe",
            "id": spec.id,
            "pipeline": spec.pipeline.model_dump(mode="json"),
            "character": CharacterSpec(
                id=spec.character.id,
                identity_primer=plan.identity_primer,
            ).model_dump(mode="json"),
            "keyframe": KeyframeSpec(
                action=spec.request.action,
                phase=spec.request.phase,
                direction=spec.request.direction,
                camera="orthographic-side",
            ).model_dump(mode="json"),
            "assets": assets.model_dump(mode="json", exclude_none=True),
            "prompt": plan.prompt.model_dump(mode="json", exclude_none=True),
            "canvas": plan.canvas.model_dump(mode="json"),
            "sampling": plan.sampling.model_dump(mode="json"),
            "conditions": [condition.model_dump(mode="json", exclude_none=True) for condition in conditions],
            "variants": [
                VariantSpec(name=_seed_name(seed), seed=seed).model_dump(mode="json")
                for seed in range(spec.generation.seed_start, spec.generation.seed_start + spec.generation.seed_count)
            ],
            "output": OutputSpec(
                directory=relative_path(resolve_output_path(spec.generation.output_directory, brief_base_dir), job_path.parent),
                filename=spec.generation.filename,
                overwrite=spec.generation.overwrite,
                save_conditions=spec.generation.save_conditions,
                save_contact_sheet=spec.generation.save_contact_sheet,
            ).model_dump(mode="json"),
            "acceptance": AcceptanceSpec(
                manual=plan.scoring.checks,
                minimum_passing_variants=min(plan.scoring.top_k, spec.generation.seed_count),
            ).model_dump(mode="json"),
        }
    )
    return job


def _assets_from_plan(plan: KeyframeBriefPlanSpec, extraction: dict[str, Any], base_dir: Path) -> AssetSpec:
    extracted_assets = extraction["assets"]
    pose_path = relative_path(Path(extracted_assets["pose"]["path"]), base_dir)
    contour_path = None
    boundary_path = None
    if any(control.source == "example_contour" for control in plan.controls):
        contour_path = relative_path(Path(extracted_assets["contour"]["path"]), base_dir)
    if any(control.residual_mask_source == "example_boundary_mask" for control in plan.controls):
        boundary_path = relative_path(Path(extracted_assets["boundary_mask"]["path"]), base_dir)
    return AssetSpec(
        pose=PathSpec(path=pose_path),
        contour=PathSpec(path=contour_path) if contour_path else None,
        boundary_mask=PathSpec(path=boundary_path) if boundary_path else None,
    )


def _condition_from_plan(control: BriefControlPlanSpec) -> ControlConditionSpec:
    image = "pose" if control.source == "example_pose" else "contour"
    residual_mask = "boundary_mask" if control.residual_mask_source == "example_boundary_mask" else None
    return ControlConditionSpec(
        name=control.name,
        type=control.type,
        image=image,
        scale=control.scale,
        start=control.start,
        end=control.end,
        residual_mask=residual_mask,
    )


def _seed_name(seed: int) -> str:
    return f"seed_{seed:03d}"
