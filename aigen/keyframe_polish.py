from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from scipy import ndimage

from aigen.generation.runtime_diagnostics import cuda_memory_stats, elapsed_ms, synchronized_time
from aigen.keyframe_judge import KeyframeJudgeConfig, QwenKeyframeJudge
from aigen.keyframe_refine import KeyframeRefineProfile, KontextInpaintRefiner
from aigen.keyframes import NvidiaSmiMemorySampler, _nvidia_smi_preflight
from aigen.prompt_tokens import count_kontext_prompt_tokens


KEYFRAME_POLISH_JOB_SCHEMA = "schemas/keyframe-polish-job.schema.json"
KEYFRAME_POLISH_PLAN_SCHEMA = "schemas/keyframe-polish-plan.schema.json"
KEYFRAME_POLISH_SCHEMA_VERSION = 1
POLISH_OPERATIONS = (
    "detail_restore",
    "expression_refine",
    "identity_restore",
    "lineart_sharpen",
    "color_restore",
    "shape_fix",
    "hand_fix",
    "artifact_remove",
)
POLISH_OPERATION_PROFILES: dict[str, dict[str, float | int]] = {
    "detail_restore": {
        "strength_min": 0.18,
        "strength_max": 0.40,
        "steps_min": 14,
        "steps_max": 22,
        "guidance_min": 1.8,
        "guidance_max": 2.6,
        "true_cfg_min": 1.0,
        "true_cfg_max": 1.4,
    },
    "expression_refine": {
        "strength_min": 0.28,
        "strength_max": 0.55,
        "steps_min": 16,
        "steps_max": 24,
        "guidance_min": 2.0,
        "guidance_max": 2.8,
        "true_cfg_min": 1.1,
        "true_cfg_max": 1.6,
    },
    "identity_restore": {
        "strength_min": 0.22,
        "strength_max": 0.48,
        "steps_min": 16,
        "steps_max": 24,
        "guidance_min": 2.0,
        "guidance_max": 2.8,
        "true_cfg_min": 1.1,
        "true_cfg_max": 1.6,
    },
    "lineart_sharpen": {
        "strength_min": 0.16,
        "strength_max": 0.34,
        "steps_min": 12,
        "steps_max": 20,
        "guidance_min": 1.7,
        "guidance_max": 2.5,
        "true_cfg_min": 1.0,
        "true_cfg_max": 1.35,
    },
    "color_restore": {
        "strength_min": 0.14,
        "strength_max": 0.32,
        "steps_min": 12,
        "steps_max": 20,
        "guidance_min": 1.7,
        "guidance_max": 2.4,
        "true_cfg_min": 1.0,
        "true_cfg_max": 1.35,
    },
    "shape_fix": {
        "strength_min": 0.45,
        "strength_max": 0.85,
        "steps_min": 20,
        "steps_max": 28,
        "guidance_min": 2.2,
        "guidance_max": 3.0,
        "true_cfg_min": 1.2,
        "true_cfg_max": 1.8,
    },
    "hand_fix": {
        "strength_min": 0.38,
        "strength_max": 0.75,
        "steps_min": 18,
        "steps_max": 28,
        "guidance_min": 2.1,
        "guidance_max": 3.0,
        "true_cfg_min": 1.15,
        "true_cfg_max": 1.8,
    },
    "artifact_remove": {
        "strength_min": 0.16,
        "strength_max": 0.42,
        "steps_min": 12,
        "steps_max": 22,
        "guidance_min": 1.7,
        "guidance_max": 2.5,
        "true_cfg_min": 1.0,
        "true_cfg_max": 1.4,
    },
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolishPipelineSpec(StrictModel):
    profile: str


class PolishBaseSpec(StrictModel):
    run_dir: str
    candidate: str


class PolishIdentityPrimerSpec(StrictModel):
    view: Literal["front", "left_profile", "right_profile", "back"]
    path: str


class PolishCharacterSpec(StrictModel):
    id: str
    identity_primer: PolishIdentityPrimerSpec


class PolishPlanPathSpec(StrictModel):
    path: str


class PolishPlannerSpec(StrictModel):
    max_regions: int = 4


class PolishMicroSweepSpec(StrictModel):
    strength_offsets: list[float]
    seed_offsets: list[int]


class PolishOutputSpec(StrictModel):
    directory: str
    overwrite: bool
    save_debug_images: bool
    save_contact_sheet: bool


class PolishAcceptanceSpec(StrictModel):
    manual: list[str]


class KeyframePolishJobSpec(StrictModel):
    schema_path: str = Field(alias="$schema")
    schema_version: Literal[1]
    kind: Literal["keyframe-polish"]
    id: str
    pipeline: PolishPipelineSpec
    base: PolishBaseSpec
    character: PolishCharacterSpec
    plan: PolishPlanPathSpec
    planner: PolishPlannerSpec
    micro_sweep: PolishMicroSweepSpec
    output: PolishOutputSpec
    acceptance: PolishAcceptanceSpec


class PlannedPolishParameters(StrictModel):
    strength: float
    steps: int
    guidance_scale: float
    true_cfg_scale: float
    feather_px: int
    crop_padding_px: int
    crop_upsample_factor: float
    max_sequence_length: int


class PlannedPolishRegion(StrictModel):
    id: str
    label: str
    bbox: tuple[int, int, int, int]
    mask_prompt: str
    operation: Literal[
        "detail_restore",
        "expression_refine",
        "identity_restore",
        "lineart_sharpen",
        "color_restore",
        "shape_fix",
        "hand_fix",
        "artifact_remove",
    ]
    reason: str
    reference_crop_requirements: list[str]
    parameters: PlannedPolishParameters
    prompt: str
    negative_prompt: str
    must_not_change: list[str]
    acceptance_checks: list[str]

    @field_validator("reference_crop_requirements", mode="before")
    @classmethod
    def _listify_reference_crop_requirements(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value


class KeyframePolishPlan(StrictModel):
    schema_version: Literal[1]
    kind: Literal["keyframe-polish-plan"]
    job_id: str
    base_candidate: str
    needs_polish: bool
    regions: list[PlannedPolishRegion]
    summary: str


class PolishSelectionCheck(StrictModel):
    target_detail_restored: bool
    identity_preserved: bool
    outside_mask_changed: bool
    pose_changed: bool
    style_match: bool


class PolishRegionSelection(StrictModel):
    region_id: str
    best_variant: str
    passes: bool
    checks: PolishSelectionCheck
    reason: str


@dataclass(frozen=True)
class BoundedPolishParameters:
    strength: float
    steps: int
    guidance_scale: float
    true_cfg_scale: float
    feather_px: int
    crop_padding_px: int
    crop_upsample_factor: float
    max_sequence_length: int


@dataclass(frozen=True)
class PolishMaskPlan:
    region_id: str
    index: int
    label: str
    operation: str
    prompt: str
    negative_prompt: str
    hard_mask: Image.Image
    feather_mask: Image.Image
    crop_box: tuple[int, int, int, int]
    reference_card: Image.Image
    parameters: BoundedPolishParameters


class KeyframePolishError(RuntimeError):
    pass


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
    config: KeyframeJudgeConfig,
    project_root: Path,
    runner: Any | None = None,
) -> dict[str, Any]:
    spec = load_keyframe_polish_job(job_path)
    context = _polish_context(spec, job_path)
    plan_path = _resolve_path_for_write(spec.plan.path, job_path.parent)
    evidence_dir = plan_path.with_suffix("")
    image_paths = _save_planner_evidence(context, evidence_dir)
    prompt = _planner_prompt(spec, context)
    active_runner = runner if runner is not None else QwenKeyframeJudge(config)
    raw_text = active_runner.judge_candidate(prompt, image_paths)
    plan = _parse_polish_plan(raw_text)
    _validate_polish_plan(spec, plan, context.base_image.size)
    payload = {
        "schema_version": 1,
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
            "device_report": _runner_device_report(active_runner),
        },
        "evidence_images": [path.as_posix() for path in image_paths],
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "polish_plan": plan.model_dump(mode="json"),
        "raw_response": raw_text,
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(plan_path, payload)
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
    context = _polish_context(spec, job_path)
    plan_payload = _read_json(_resolve_path(spec.plan.path, job_path.parent))
    plan = KeyframePolishPlan.model_validate(plan_payload["polish_plan"])
    _validate_polish_plan(spec, plan, context.base_image.size)
    output_dir = _resolve_output_dir(spec.output.directory, job_path.parent)
    outputs = _planned_outputs(spec, plan, output_dir)
    if check_outputs and not spec.output.overwrite:
        existing = [output for output in outputs if Path(output["path"]).exists()]
        if existing:
            raise KeyframePolishError(f"Output exists and overwrite=false: {existing[0]['path']}")
    token_metadata = _region_token_metadata(profile, plan)
    return {
        "schema_version": KEYFRAME_POLISH_SCHEMA_VERSION,
        "kind": "resolved-keyframe-polish",
        "job_path": job_path.resolve().as_posix(),
        "job_id": spec.id,
        "profile": _profile_json(profile),
        "base": {
            "run_dir": context.base_dir.as_posix(),
            "candidate": spec.base.candidate,
            "image": _asset_json(context.base_path),
            "source_job_id": context.result["job_id"],
        },
        "character": {
            "id": spec.character.id,
            "identity_primer": {
                "view": spec.character.identity_primer.view,
                **_asset_json(context.identity_primer_path),
            },
        },
        "plan_path": _resolve_path(spec.plan.path, job_path.parent).as_posix(),
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
        "spec_sha256": _sha256_bytes(job_path.read_bytes()),
    }


def validate_keyframe_polish_job(
    job_path: Path,
    profile: KeyframeRefineProfile,
    *,
    project_root: Path,
) -> dict[str, Any]:
    resolved = resolve_keyframe_polish_job(job_path, profile, project_root=project_root, check_outputs=False)
    return {
        "status": "valid",
        "job_id": resolved["job_id"],
        "profile": resolved["profile"]["name"],
        "regions": [region["id"] for region in resolved["polish_plan"]["regions"]],
        "tokens": resolved["tokens"],
        "outputs": resolved["output"]["files"],
    }


def preview_keyframe_polish_job(
    job_path: Path,
    profile: KeyframeRefineProfile,
    *,
    project_root: Path,
) -> dict[str, Any]:
    resolved = resolve_keyframe_polish_job(job_path, profile, project_root=project_root, check_outputs=True)
    context = _polish_context(load_keyframe_polish_job(job_path), job_path)
    mask_plans = build_polish_mask_plans(context.base_image, context.identity_primer, resolved)
    return {
        **resolved,
        "mask_plan": [_mask_plan_json(plan) for plan in mask_plans],
    }


def run_keyframe_polish_job(
    job_path: Path,
    profile: KeyframeRefineProfile,
    *,
    project_root: Path,
) -> dict[str, Any]:
    spec = load_keyframe_polish_job(job_path)
    resolved = resolve_keyframe_polish_job(job_path, profile, project_root=project_root, check_outputs=True)
    context = _polish_context(spec, job_path)
    output_dir = Path(resolved["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_plans = build_polish_mask_plans(context.base_image, context.identity_primer, resolved)
    resolved = {**resolved, "mask_plan": [_mask_plan_json(plan) for plan in mask_plans]}
    _write_json(output_dir / "resolved.json", resolved)
    if spec.output.save_debug_images:
        _save_debug_images(context.base_image, mask_plans, output_dir)

    memory_sampler = NvidiaSmiMemorySampler(_nvidia_smi_preflight())
    memory_sampler.start()
    refiner: KontextInpaintRefiner | None = None
    try:
        total_start = perf_counter()
        refiner = KontextInpaintRefiner(profile)
        torch_module = refiner.torch
        outputs = []
        for mask_plan in mask_plans:
            region_dir = output_dir / "regions" / mask_plan.region_id
            region_dir.mkdir(parents=True, exist_ok=True)
            for variant in _region_variants(spec, mask_plan):
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
                polished = _paste_refined_crop(
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
                        "mask_change": _outside_mask_change(context.base_image, polished, mask_plan.feather_mask),
                    }
                )
        if spec.output.save_contact_sheet:
            _save_contact_sheet(outputs, output_dir / "contact_sheet.png")
        memory = cuda_memory_stats(torch_module, "cuda") | memory_sampler.stop()
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
        _write_json(output_dir / "result.json", result)
        return result
    finally:
        if refiner is not None:
            refiner.close()
        memory_sampler.stop()


def select_keyframe_polish(
    job_path: Path,
    *,
    config: KeyframeJudgeConfig,
    project_root: Path,
    runner: Any | None = None,
) -> dict[str, Any]:
    spec = load_keyframe_polish_job(job_path)
    context = _polish_context(spec, job_path)
    output_dir = _resolve_output_dir(spec.output.directory, job_path.parent)
    result = _read_json(output_dir / "result.json")
    resolved = result["effective_config"]
    plan = KeyframePolishPlan.model_validate(resolved["polish_plan"])
    mask_plans = build_polish_mask_plans(context.base_image, context.identity_primer, resolved)
    outputs_by_region = _outputs_by_region(result["outputs"])
    active_runner = runner if runner is not None else QwenKeyframeJudge(config)
    composite = context.base_image.copy()
    selections = []
    for mask_plan in mask_plans:
        candidates = [
            output
            for output in outputs_by_region[mask_plan.region_id]
            if not output["mask_change"]["hard_rejects"]["outside_feather_changed"]
        ]
        prompt = _selection_prompt(spec, plan, mask_plan, candidates)
        image_paths = _save_selection_evidence(context.base_image, context.identity_primer_path, mask_plan, candidates, output_dir)
        raw_text = active_runner.judge_candidate(prompt, image_paths)
        selection = _parse_region_selection(raw_text)
        _validate_region_selection(selection, mask_plan.region_id, {candidate["name"] for candidate in candidates})
        selected = next(candidate for candidate in candidates if candidate["name"] == selection.best_variant)
        composite = _paste_refined_crop(
            composite,
            Image.open(selected["path"]).convert("RGB").crop(mask_plan.crop_box),
            mask_plan.feather_mask,
            mask_plan.crop_box,
        )
        selections.append(
            {
                **selection.model_dump(mode="json"),
                "selected_path": selected["path"],
                "raw_response": raw_text,
                "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
            }
        )
    final_path = output_dir / "final_composite.png"
    composite.save(final_path)
    payload = {
        "schema_version": 1,
        "status": "completed",
        "job_id": spec.id,
        "git_commit": _git_commit(project_root),
        "final_composite": _asset_json(final_path),
        "regions": selections,
        "judge": {
            "id": config.judge_id,
            "repo_id": config.repo_id,
            "revision": config.revision,
            "quantization": config.quantization,
            "min_pixels": config.min_pixels,
            "max_pixels": config.max_pixels,
            "temperature": config.temperature,
            "device_report": _runner_device_report(active_runner),
        },
    }
    _write_json(output_dir / "polish_selection.json", payload)
    return payload


@dataclass(frozen=True)
class PolishContext:
    base_dir: Path
    result: dict[str, Any]
    base_path: Path
    base_image: Image.Image
    identity_primer_path: Path
    identity_primer: Image.Image


def _polish_context(spec: KeyframePolishJobSpec, job_path: Path) -> PolishContext:
    base_dir = _resolve_output_dir(spec.base.run_dir, job_path.parent)
    result = _read_json(base_dir / "result.json")
    base_output = _base_output(result, spec.base.candidate)
    base_path = Path(base_output["path"]).resolve()
    identity_primer_path = _resolve_path(spec.character.identity_primer.path, job_path.parent)
    return PolishContext(
        base_dir=base_dir,
        result=result,
        base_path=base_path,
        base_image=Image.open(base_path).convert("RGB"),
        identity_primer_path=identity_primer_path,
        identity_primer=Image.open(identity_primer_path).convert("RGB"),
    )


def _save_planner_evidence(context: PolishContext, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    identity = output_dir / "identity_primer.png"
    candidate = output_dir / "candidate.png"
    context.identity_primer.save(identity)
    context.base_image.save(candidate)
    paths.extend((identity, candidate))
    comparison_grid = output_dir / "identity_candidate_comparison_grid.png"
    crop_grid = output_dir / "candidate_crop_grid.png"
    _identity_candidate_comparison_grid(context.identity_primer, context.base_image).save(comparison_grid)
    _candidate_crop_grid(context.base_image).save(crop_grid)
    paths.extend((comparison_grid, crop_grid))
    return paths


def _planner_prompt(spec: KeyframePolishJobSpec, context: PolishContext) -> str:
    effective = context.result["effective_config"]
    acceptance = effective["acceptance"]["manual"]
    keyframe = effective["keyframe"]
    operations = list(POLISH_OPERATION_PROFILES)
    return f"""You are a production art polish planner for character keyframes.

You receive images in this order:
1. approved identity primer
2. selected structural candidate named {spec.base.candidate}
3. identity-primer-vs-candidate normalized crop comparison grid
4. generic candidate crop grid

The selected candidate already has the best structure. Pose, arm shape, silhouette and action readability are not polish targets here.
Plan only local polish regions where the candidate visibly diverges from the identity primer or loses character-specific detail, style, color, expression, clothing details, accessories, material identity, lineart quality, or small artifacts.
The target pose, contour and boundary-mask are already handled by the upstream keyframe run and are intentionally not shown here. Do not infer polish targets from the source pose example.

Do not use a fixed region catalog. Discover the regions from the images. Valid operation types:
{json.dumps(operations)}

Executor operation bounds:
{json.dumps(POLISH_OPERATION_PROFILES, sort_keys=True)}

Target keyframe:
- action: {keyframe["action"]}
- phase: {keyframe["phase"]}
- direction: {keyframe["direction"]}
- camera: {keyframe["camera"]}

Manual acceptance criteria:
{json.dumps(acceptance, ensure_ascii=False)}

Return JSON only. The JSON object must match:
- schema_version: 1
- kind: "keyframe-polish-plan"
- job_id: "{spec.id}"
- base_candidate: "{spec.base.candidate}"
- needs_polish: boolean
- regions: free model-discovered regions, at most {spec.planner.max_regions}
- summary: one sentence

Each region must have:
- id: stable id like region_01
- label: human-readable local detail label
- bbox: [left, top, right, bottom] in candidate pixels
- mask_prompt: text describing the local pixels to mask inside the bbox
- operation: one valid operation type
- reason
- reference_crop_requirements
- parameters: strength, steps, guidance_scale, true_cfg_scale, feather_px, crop_padding_px, crop_upsample_factor, max_sequence_length
- prompt: local positive inpaint prompt
- negative_prompt: local negative prompt
- must_not_change
- acceptance_checks

Inspect the head/face/hair, upper clothing, waist clothing, legs/feet and visible accessories. Pick only the regions with the largest identity/detail mismatch.
Do not propose a region only because it could better match the pose condition or copied source example.
Do not use generic "high quality" prompts.
Each prompt must name the concrete local detail to restore and the identity detail to preserve.
Use anatomically coherent labels and prompts. For example, hair belongs to the head region; do not describe arm, boot or clothing pixels as hair.
Prefer precise small regions over large outfit/body regions. Use lower strength for identity/detail restoration than shape fixes.
"""


def _parse_polish_plan(raw_text: str) -> KeyframePolishPlan:
    data = _json_from_vlm_response(raw_text)
    try:
        return KeyframePolishPlan.model_validate(data)
    except ValidationError as error:
        raise KeyframePolishError(f"Polish planner returned invalid JSON: {error}") from error


def _validate_polish_plan(spec: KeyframePolishJobSpec, plan: KeyframePolishPlan, image_size: tuple[int, int]) -> None:
    if plan.job_id != spec.id:
        raise KeyframePolishError(f"Polish plan is for {plan.job_id}, expected {spec.id}")
    if plan.base_candidate != spec.base.candidate:
        raise KeyframePolishError(f"Polish plan is for {plan.base_candidate}, expected {spec.base.candidate}")
    if len(plan.regions) > spec.planner.max_regions:
        raise KeyframePolishError(f"Polish plan contains {len(plan.regions)} regions, max is {spec.planner.max_regions}")
    ids = [region.id for region in plan.regions]
    if len(ids) != len(set(ids)):
        raise KeyframePolishError("Polish plan contains duplicate region ids")
    width, height = image_size
    for region in plan.regions:
        left, top, right, bottom = region.bbox
        if left < 0 or top < 0 or right > width or bottom > height or left >= right or top >= bottom:
            raise KeyframePolishError(f"Region {region.id} bbox is outside candidate bounds: {region.bbox}")
        if region.operation not in POLISH_OPERATION_PROFILES:
            raise KeyframePolishError(f"Region {region.id} uses unsupported operation {region.operation}")


def build_polish_mask_plans(base: Image.Image, identity_primer: Image.Image, resolved: dict[str, Any]) -> list[PolishMaskPlan]:
    foreground = _foreground_mask(np.asarray(base, dtype=np.uint8))
    plan = KeyframePolishPlan.model_validate(resolved["polish_plan"])
    return [_build_region_mask(index, region, foreground, base, identity_primer) for index, region in enumerate(plan.regions, start=1)]


def _build_region_mask(
    index: int,
    region: PlannedPolishRegion,
    foreground: np.ndarray,
    base: Image.Image,
    identity_primer: Image.Image,
) -> PolishMaskPlan:
    width, height = base.size
    hard = Image.new("L", base.size, 0)
    ImageDraw.Draw(hard).rectangle(region.bbox, fill=255)
    hard_array = np.asarray(hard, dtype=np.uint8) > 0
    hard_array &= foreground
    if not hard_array.any():
        hard_array = np.asarray(hard, dtype=np.uint8) > 0
    parameters = _bound_parameters(region.operation, region.parameters)
    hard_array = ndimage.binary_dilation(hard_array, iterations=max(1, round(width * 0.006)))
    hard = Image.fromarray((hard_array.astype(np.uint8) * 255), mode="L")
    feather = hard.filter(ImageFilter.GaussianBlur(radius=parameters.feather_px))
    crop_box = _expanded_aligned_box(hard_array, parameters.crop_padding_px, width, height)
    return PolishMaskPlan(
        region_id=region.id,
        index=index,
        label=region.label,
        operation=region.operation,
        prompt=region.prompt,
        negative_prompt=region.negative_prompt,
        hard_mask=hard,
        feather_mask=feather,
        crop_box=crop_box,
        reference_card=_reference_detail_card(identity_primer, base, crop_box),
        parameters=parameters,
    )


def _bound_parameters(operation: str, parameters: PlannedPolishParameters) -> BoundedPolishParameters:
    profile = POLISH_OPERATION_PROFILES[operation]
    return BoundedPolishParameters(
        strength=_clamp_float(parameters.strength, profile["strength_min"], profile["strength_max"]),
        steps=_clamp_int(parameters.steps, profile["steps_min"], profile["steps_max"]),
        guidance_scale=_clamp_float(parameters.guidance_scale, profile["guidance_min"], profile["guidance_max"]),
        true_cfg_scale=_clamp_float(parameters.true_cfg_scale, profile["true_cfg_min"], profile["true_cfg_max"]),
        feather_px=max(1, min(32, parameters.feather_px)),
        crop_padding_px=max(16, min(192, parameters.crop_padding_px)),
        crop_upsample_factor=_clamp_float(parameters.crop_upsample_factor, 1.0, 2.0),
        max_sequence_length=max(64, min(256, parameters.max_sequence_length)),
    )


def _region_variants(spec: KeyframePolishJobSpec, mask_plan: PolishMaskPlan) -> list[dict[str, Any]]:
    variants = []
    seen: set[tuple[int, float]] = set()
    for seed_offset in spec.micro_sweep.seed_offsets:
        for strength_offset in spec.micro_sweep.strength_offsets:
            strength = _clamp_float(
                mask_plan.parameters.strength + strength_offset,
                POLISH_OPERATION_PROFILES[mask_plan.operation]["strength_min"],
                POLISH_OPERATION_PROFILES[mask_plan.operation]["strength_max"],
            )
            seed = 1000 + (mask_plan.index * 100) + seed_offset
            key = (seed, strength)
            if key in seen:
                continue
            seen.add(key)
            suffix = f"s{strength:.2f}".replace(".", "p")
            variants.append(
                {
                    "name": f"{mask_plan.region_id}_{suffix}_seed{seed}",
                    "seed": seed,
                    "strength": strength,
                    "clip_prompt": mask_plan.label,
                    "t5_prompt": mask_plan.prompt,
                    "negative_prompt": mask_plan.negative_prompt,
                }
            )
    return variants


def _region_token_metadata(profile: KeyframeRefineProfile, plan: KeyframePolishPlan) -> dict[str, dict[str, int]]:
    metadata = {}
    for region in plan.regions:
        bounded = _bound_parameters(region.operation, region.parameters)
        tokens = count_kontext_prompt_tokens(profile.model, region.label, region.prompt)
        if tokens.clip > tokens.clip_limit:
            raise KeyframePolishError(f"{region.id} CLIP prompt has {tokens.clip} tokens, limit is {tokens.clip_limit}")
        if tokens.t5 > bounded.max_sequence_length:
            raise KeyframePolishError(
                f"{region.id} T5 prompt has {tokens.t5} tokens, "
                f"max_sequence_length is {bounded.max_sequence_length}"
            )
        metadata[region.id] = {
            "clip": tokens.clip,
            "clip_limit": tokens.clip_limit,
            "t5": tokens.t5,
            "t5_limit": bounded.max_sequence_length,
        }
    return metadata


def _planned_outputs(spec: KeyframePolishJobSpec, plan: KeyframePolishPlan, output_dir: Path) -> list[dict[str, Any]]:
    files = []
    for region in plan.regions:
        bounded = _bound_parameters(region.operation, region.parameters)
        seen: set[tuple[int, float]] = set()
        for seed_offset in spec.micro_sweep.seed_offsets:
            for strength_offset in spec.micro_sweep.strength_offsets:
                strength = _clamp_float(
                    bounded.strength + strength_offset,
                    POLISH_OPERATION_PROFILES[region.operation]["strength_min"],
                    POLISH_OPERATION_PROFILES[region.operation]["strength_max"],
                )
                seed = 1000 + (int(region.id.split("_")[-1]) * 100) + seed_offset
                key = (seed, strength)
                if key in seen:
                    continue
                seen.add(key)
                suffix = f"s{strength:.2f}".replace(".", "p")
                name = f"{region.id}_{suffix}_seed{seed}"
                files.append(
                    {
                        "name": name,
                        "region_id": region.id,
                        "seed": seed,
                        "strength": strength,
                        "path": (output_dir / "regions" / region.id / f"{name}.png").as_posix(),
                    }
                )
    return files


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
        _mask_overlay(base, mask_plan.feather_mask).save(region_dir / "mask_overlay.png")


def _save_selection_evidence(
    base: Image.Image,
    identity_primer: Path,
    mask_plan: PolishMaskPlan,
    candidates: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    evidence_dir = output_dir / "selection_evidence" / mask_plan.region_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    primer = evidence_dir / "identity_primer.png"
    overlay = evidence_dir / "base_overlay.png"
    Image.open(identity_primer).convert("RGB").save(primer)
    _mask_overlay(base, mask_plan.feather_mask).save(overlay)
    paths.extend((primer, overlay))
    for candidate in candidates:
        crop = evidence_dir / f"{candidate['name']}_crop.png"
        Image.open(candidate["path"]).convert("RGB").crop(mask_plan.crop_box).save(crop)
        paths.append(crop)
    return paths


def _selection_prompt(
    spec: KeyframePolishJobSpec,
    plan: KeyframePolishPlan,
    mask_plan: PolishMaskPlan,
    candidates: list[dict[str, Any]],
) -> str:
    region = next(region for region in plan.regions if region.id == mask_plan.region_id)
    return f"""You are selecting the best local polish result for one region.

Images:
1. identity primer
2. base candidate with mask overlay
3+ candidate crops in this exact order:
{json.dumps([candidate["name"] for candidate in candidates])}

Region:
- id: {region.id}
- label: {region.label}
- operation: {region.operation}
- reason: {region.reason}
- must_not_change: {json.dumps(region.must_not_change, ensure_ascii=False)}
- acceptance_checks: {json.dumps(region.acceptance_checks, ensure_ascii=False)}

Choose the candidate that restores the target local detail while preserving identity and style. Do not choose a variant that changes pose or looks like a different character.

Return JSON only:
- region_id: "{region.id}"
- best_variant: one candidate name from the list
- passes: boolean
- checks: target_detail_restored, identity_preserved, outside_mask_changed, pose_changed, style_match
- reason: one sentence

Job id: {spec.id}
"""


def _parse_region_selection(raw_text: str) -> PolishRegionSelection:
    data = _json_from_vlm_response(raw_text)
    try:
        return PolishRegionSelection.model_validate(data)
    except ValidationError as error:
        raise KeyframePolishError(f"Polish selector returned invalid JSON: {error}") from error


def _validate_region_selection(selection: PolishRegionSelection, region_id: str, candidates: set[str]) -> None:
    if selection.region_id != region_id:
        raise KeyframePolishError(f"Polish selector returned region {selection.region_id}, expected {region_id}")
    if selection.best_variant not in candidates:
        raise KeyframePolishError(f"Polish selector chose unknown variant {selection.best_variant}")
    if not selection.passes:
        raise KeyframePolishError(f"Polish selector failed {region_id}: {selection.reason}")
    if selection.checks.outside_mask_changed or selection.checks.pose_changed:
        raise KeyframePolishError(f"Polish selector accepted unsafe variant for {region_id}: {selection.reason}")


def _outputs_by_region(outputs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for output in outputs:
        grouped.setdefault(output["region_id"], []).append(output)
    return grouped


def _candidate_crop_grid(base: Image.Image) -> Image.Image:
    foreground = _foreground_mask(np.asarray(base, dtype=np.uint8))
    left, top, right, bottom = _bbox(foreground)
    cols = 3
    rows = 3
    cell_w = 192
    cell_h = 192
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    source_w = right - left
    source_h = bottom - top
    for row in range(rows):
        for col in range(cols):
            crop = (
                left + round(source_w * col / cols),
                top + round(source_h * row / rows),
                left + round(source_w * (col + 1) / cols),
                top + round(source_h * (row + 1) / rows),
            )
            grid.paste(base.crop(crop).resize((cell_w, cell_h), Image.Resampling.LANCZOS), (col * cell_w, row * cell_h))
    return grid


def _identity_candidate_comparison_grid(identity_primer: Image.Image, candidate: Image.Image) -> Image.Image:
    identity_foreground = _foreground_mask(np.asarray(identity_primer, dtype=np.uint8))
    candidate_foreground = _foreground_mask(np.asarray(candidate, dtype=np.uint8))
    identity_box = _bbox(identity_foreground)
    candidate_box = _bbox(candidate_foreground)
    cols = 3
    rows = 3
    crop_w = 144
    crop_h = 144
    label_h = 22
    cell_w = crop_w * 2
    sheet = Image.new("RGB", (cols * cell_w, rows * (crop_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for row in range(rows):
        for col in range(cols):
            x = col * cell_w
            y = row * (crop_h + label_h)
            identity_crop = _normalized_box_crop(identity_primer, identity_box, col, row, cols, rows)
            candidate_crop = _normalized_box_crop(candidate, candidate_box, col, row, cols, rows)
            sheet.paste(identity_crop.resize((crop_w, crop_h), Image.Resampling.LANCZOS), (x, y + label_h))
            sheet.paste(candidate_crop.resize((crop_w, crop_h), Image.Resampling.LANCZOS), (x + crop_w, y + label_h))
            draw.text((x + 4, y + 4), f"identity {row + 1},{col + 1}", fill="black")
            draw.text((x + crop_w + 4, y + 4), f"candidate {row + 1},{col + 1}", fill="black")
    return sheet


def _normalized_box_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    col: int,
    row: int,
    cols: int,
    rows: int,
) -> Image.Image:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    return image.crop(
        (
            left + round(width * col / cols),
            top + round(height * row / rows),
            left + round(width * (col + 1) / cols),
            top + round(height * (row + 1) / rows),
        )
    )


def _reference_detail_card(identity_primer: Image.Image, base: Image.Image, crop_box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = crop_box
    scale_x = identity_primer.width / base.width
    scale_y = identity_primer.height / base.height
    primer_box = (
        max(0, round(left * scale_x)),
        max(0, round(top * scale_y)),
        min(identity_primer.width, round(right * scale_x)),
        min(identity_primer.height, round(bottom * scale_y)),
    )
    crop = identity_primer.crop(primer_box)
    card_w = 768
    card_h = 512
    card = Image.new("RGB", (card_w, card_h), "white")
    full_h = card_h
    full_w = round(identity_primer.width * full_h / identity_primer.height)
    card.paste(identity_primer.resize((full_w, full_h), Image.Resampling.LANCZOS), (0, 0))
    crop_w = card_w - full_w
    crop_h = card_h
    card.paste(crop.resize((crop_w, crop_h), Image.Resampling.LANCZOS), (full_w, 0))
    return card


def _mask_plan_json(mask_plan: PolishMaskPlan) -> dict[str, Any]:
    return {
        "region_id": mask_plan.region_id,
        "label": mask_plan.label,
        "operation": mask_plan.operation,
        "crop_box": list(mask_plan.crop_box),
        "parameters": {
            "strength": mask_plan.parameters.strength,
            "steps": mask_plan.parameters.steps,
            "guidance_scale": mask_plan.parameters.guidance_scale,
            "true_cfg_scale": mask_plan.parameters.true_cfg_scale,
            "feather_px": mask_plan.parameters.feather_px,
            "crop_padding_px": mask_plan.parameters.crop_padding_px,
            "crop_upsample_factor": mask_plan.parameters.crop_upsample_factor,
            "max_sequence_length": mask_plan.parameters.max_sequence_length,
        },
    }


def _mask_overlay(base: Image.Image, mask: Image.Image) -> Image.Image:
    overlay = base.convert("RGBA")
    red = Image.new("RGBA", base.size, (255, 0, 0, 120))
    overlay.alpha_composite(Image.composite(red, Image.new("RGBA", base.size, (0, 0, 0, 0)), mask))
    return overlay.convert("RGB")


def _paste_refined_crop(
    base: Image.Image,
    refined_crop: Image.Image,
    feather_mask: Image.Image,
    crop_box: tuple[int, int, int, int],
) -> Image.Image:
    output = base.copy()
    output.paste(
        refined_crop.resize((crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])),
        crop_box,
        feather_mask.crop(crop_box),
    )
    return output


def _outside_mask_change(base: Image.Image, refined: Image.Image, feather_mask: Image.Image) -> dict[str, Any]:
    base_array = np.asarray(base.convert("RGB"), dtype=np.int16)
    refined_array = np.asarray(refined.convert("RGB"), dtype=np.int16)
    outside = np.asarray(feather_mask, dtype=np.uint8) == 0
    delta = np.abs(base_array - refined_array).max(axis=2)
    changed = delta[outside] > 1
    changed_pixels = int(changed.sum())
    total = int(outside.sum())
    max_delta = int(delta[outside].max()) if total else 0
    return {
        "outside_feather_changed_pixels": changed_pixels,
        "outside_feather_changed_ratio": float(changed_pixels / max(total, 1)),
        "outside_feather_max_delta": max_delta,
        "hard_rejects": {
            "outside_feather_changed": bool(changed_pixels > 0 or max_delta > 1),
        },
    }


def _save_contact_sheet(outputs: list[dict[str, Any]], output_path: Path) -> None:
    images = [Image.open(output["path"]).convert("RGB") for output in outputs]
    thumb_w = 192
    thumb_h = max(1, int(thumb_w * images[0].height / images[0].width))
    label_h = 32
    sheet = Image.new("RGB", (thumb_w * len(images), thumb_h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, output) in enumerate(zip(images, outputs, strict=True)):
        x = index * thumb_w
        sheet.paste(image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, label_h))
        draw.text((x + 6, 8), output["name"][:28], fill="black")
    sheet.save(output_path)


def _foreground_mask(image: np.ndarray, threshold: float = 28.0) -> np.ndarray:
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0).astype(np.float32)
    background = np.median(border, axis=0)
    distance = np.sqrt(((image.astype(np.float32) - background) ** 2).sum(axis=2))
    foreground = distance > threshold
    foreground = ndimage.binary_closing(foreground, structure=np.ones((3, 3), dtype=bool), iterations=2)
    foreground = ndimage.binary_fill_holes(foreground)
    labels, count = ndimage.label(foreground)
    if count == 0:
        return foreground
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise KeyframePolishError("Polish base image contains no foreground subject")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _expanded_aligned_box(mask: np.ndarray, padding: int, width: int, height: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise KeyframePolishError("Polish mask is empty")
    left = max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)
    right = min(width, int(xs.max()) + 1 + padding)
    bottom = min(height, int(ys.max()) + 1 + padding)
    right = min(width, left + _align_up(right - left, 16))
    bottom = min(height, top + _align_up(bottom - top, 16))
    left = max(0, right - _align_up(right - left, 16))
    top = max(0, bottom - _align_up(bottom - top, 16))
    return left, top, right, bottom


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _json_from_vlm_response(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise KeyframePolishError("VLM returned an unterminated Markdown JSON block")
        text = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise KeyframePolishError(f"VLM returned non-JSON output: {error}") from error
    if not isinstance(data, dict):
        raise KeyframePolishError("VLM returned JSON that is not an object")
    return data


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise KeyframePolishError(f"Cannot read keyframe polish input {path.as_posix()}: {error}") from error


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        raise KeyframePolishError(f"Missing path: {path.as_posix()}")
    return path


def _resolve_path_for_write(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_output_dir(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_commit(project_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()


def _profile_json(profile: KeyframeRefineProfile) -> dict[str, Any]:
    models = {
        "kontext": {
            **profile.model_revisions["kontext"],
            "path": profile.model,
        },
    }
    if profile.nunchaku_transformer_model is not None:
        models["nunchaku_transformer"] = {
            **profile.model_revisions["nunchaku_transformer"],
            "path": profile.nunchaku_transformer_model.resolve().as_posix(),
        }
    return {
        "name": profile.name,
        "dtype": profile.dtype,
        "attention_impl": profile.attention_impl,
        "pipeline_cpu_offload": profile.pipeline_cpu_offload,
        "vae_tiling": profile.vae_tiling,
        "models": models,
    }


def _base_output(result: dict[str, Any], candidate: str) -> dict[str, Any]:
    for output in result["outputs"]:
        if output["name"] == candidate:
            return output
    raise KeyframePolishError(f"Base run has no candidate named {candidate}")


def _clamp_float(value: float, lower: float | int, upper: float | int) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _clamp_int(value: int, lower: float | int, upper: float | int) -> int:
    return max(int(lower), min(int(upper), int(value)))


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


def _runner_device_report(runner: Any) -> dict[str, Any]:
    report = getattr(runner, "device_report", {})
    if isinstance(report, dict):
        return report
    return {"value": str(report)}
