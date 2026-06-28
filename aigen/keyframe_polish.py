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
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from scipy import ndimage

from aigen.generation.runtime_diagnostics import cuda_memory_stats, elapsed_ms, synchronized_time
from aigen.keyframe_judge import KeyframeJudgeConfig, QwenKeyframeJudge
from aigen.keyframe_refine import KeyframeRefineProfile, KontextInpaintRefiner
from aigen.keyframes import NvidiaSmiMemorySampler, _nvidia_smi_preflight
from aigen.prompt_tokens import count_kontext_prompt_tokens


KEYFRAME_POLISH_JOB_SCHEMA = "schemas/keyframe-polish-job.schema.json"
KEYFRAME_POLISH_SCHEMA_VERSION = 1


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


class PolishDiagnosisSpec(StrictModel):
    path: str


class PolishDiagnosisRegion(StrictModel):
    name: str
    priority: int
    reason: str


class PolishRegionAssessment(StrictModel):
    needs_polish: bool
    reason: str


class PolishDiagnosisResponse(StrictModel):
    candidate: str
    region_assessments: dict[str, PolishRegionAssessment]
    selected_regions: list[PolishDiagnosisRegion]
    summary: str


class PolishPromptSpec(StrictModel):
    clip: str
    t5: str
    negative: str | None = None
    true_cfg_scale: float


class PolishRegionSpec(StrictModel):
    name: str
    mask: Literal["auto_tie_shirt", "auto_skirt_belt", "auto_face"]
    crop_padding_px: int
    feather_px: int
    prompt: PolishPromptSpec


class PolishSamplingSpec(StrictModel):
    steps: int
    guidance_scale: float
    strength: float
    max_sequence_length: int


class PolishVariantSpec(StrictModel):
    name: str
    seed: int


class PolishOutputSpec(StrictModel):
    directory: str
    filename: str
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
    diagnosis: PolishDiagnosisSpec
    regions: list[PolishRegionSpec]
    sampling: PolishSamplingSpec
    variants: list[PolishVariantSpec]
    output: PolishOutputSpec
    acceptance: PolishAcceptanceSpec


@dataclass(frozen=True)
class PolishMaskPlan:
    region: str
    hard_mask: Image.Image
    feather_mask: Image.Image
    crop_box: tuple[int, int, int, int]


class KeyframePolishError(RuntimeError):
    pass


def keyframe_polish_job_schema() -> dict[str, Any]:
    schema = KeyframePolishJobSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_keyframe_polish_job(path: Path) -> KeyframePolishJobSpec:
    try:
        return KeyframePolishJobSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise KeyframePolishError(f"Invalid keyframe polish job {path}: {error}") from error


def diagnose_keyframe_polish(
    job_path: Path,
    *,
    config: KeyframeJudgeConfig,
    project_root: Path,
    runner: Any | None = None,
) -> dict[str, Any]:
    spec = load_keyframe_polish_job(job_path)
    base_dir = _resolve_output_dir(spec.base.run_dir, job_path.parent)
    result = _read_json(base_dir / "result.json")
    base_output = _base_output(result, spec.base.candidate)
    identity_primer = _resolve_path(spec.character.identity_primer.path, job_path.parent)
    output_path = _resolve_path_for_write(spec.diagnosis.path, job_path.parent)
    base_image_path = Path(base_output["path"]).resolve()
    base = Image.open(base_image_path).convert("RGB")
    available_regions = [region.model_dump(mode="json", exclude_none=True) for region in spec.regions]
    mask_plans = build_polish_mask_plans_for_image(base, available_regions)
    diagnosis_assets_dir = output_path.with_suffix("")
    image_paths = _save_diagnosis_images(base, identity_primer, mask_plans, diagnosis_assets_dir)

    active_runner = runner if runner is not None else QwenKeyframeJudge(config)
    prompt = _diagnosis_prompt(spec.base.candidate, result["effective_config"], mask_plans)
    raw_text = active_runner.judge_candidate(prompt, image_paths)
    response = _parse_diagnosis_response(raw_text)
    _validate_diagnosis_response(response, {region.name for region in spec.regions})
    if response.candidate != spec.base.candidate:
        raise KeyframePolishError(
            f"Polish diagnosis returned candidate {response.candidate}, expected {spec.base.candidate}"
        )
    payload = {
        "schema_version": 1,
        "status": "completed",
        "job_path": job_path.resolve().as_posix(),
        "run_dir": base_dir.as_posix(),
        "candidate": spec.base.candidate,
        "git_commit": _git_commit(project_root),
        "judge": {
            "id": config.judge_id,
            "repo_id": config.repo_id,
            "revision": config.revision,
            "quantization": config.quantization,
            "min_pixels": config.min_pixels,
            "max_pixels": config.max_pixels,
            "temperature": config.temperature,
        },
        "diagnosis_images": [path.as_posix() for path in image_paths],
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "diagnosis": response.model_dump(mode="json"),
        "raw_response": raw_text,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, payload)
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

    base_dir = _resolve_output_dir(spec.base.run_dir, job_path.parent)
    result = _read_json(base_dir / "result.json")
    base_output = _base_output(result, spec.base.candidate)
    base_image = _asset_json(Path(base_output["path"]).resolve())
    identity_primer = _asset_json(_resolve_path(spec.character.identity_primer.path, job_path.parent))
    diagnosis = _read_json(_resolve_path(spec.diagnosis.path, job_path.parent))
    selected_regions = _selected_regions(spec, diagnosis)
    output_dir = _resolve_output_dir(spec.output.directory, job_path.parent)
    outputs = _planned_outputs(spec, output_dir)
    if check_outputs and not spec.output.overwrite:
        existing = [output for output in outputs if Path(output["path"]).exists()]
        if existing:
            raise KeyframePolishError(f"Output exists and overwrite=false: {existing[0]['path']}")

    region_tokens = {}
    for region in selected_regions:
        tokens = count_kontext_prompt_tokens(profile.model, region.prompt.clip, region.prompt.t5)
        if tokens.clip > tokens.clip_limit:
            raise KeyframePolishError(f"{region.name} CLIP prompt has {tokens.clip} tokens, limit is {tokens.clip_limit}")
        if tokens.t5 > spec.sampling.max_sequence_length:
            raise KeyframePolishError(
                f"{region.name} T5 prompt has {tokens.t5} tokens, "
                f"max_sequence_length is {spec.sampling.max_sequence_length}"
            )
        if region.prompt.negative is not None and region.prompt.true_cfg_scale <= 1.0:
            raise KeyframePolishError(f"{region.name} negative prompt is configured but true_cfg_scale <= 1.0")
        if region.prompt.true_cfg_scale > 1.0 and region.prompt.negative is None:
            raise KeyframePolishError(f"{region.name} true_cfg_scale > 1.0 requires prompt.negative")
        region_tokens[region.name] = {
            "clip": tokens.clip,
            "clip_limit": tokens.clip_limit,
            "t5": tokens.t5,
            "t5_limit": spec.sampling.max_sequence_length,
        }

    return {
        "schema_version": KEYFRAME_POLISH_SCHEMA_VERSION,
        "kind": "resolved-keyframe-polish",
        "job_path": job_path.resolve().as_posix(),
        "job_id": spec.id,
        "profile": _profile_json(profile),
        "base": {
            "run_dir": base_dir.as_posix(),
            "candidate": spec.base.candidate,
            "image": base_image,
            "source_job_id": result["job_id"],
        },
        "character": {
            "id": spec.character.id,
            "identity_primer": {
                "view": spec.character.identity_primer.view,
                **identity_primer,
            },
        },
        "diagnosis": {
            "path": _resolve_path(spec.diagnosis.path, job_path.parent).as_posix(),
            "region_assessments": diagnosis["diagnosis"]["region_assessments"],
            "selected_regions": diagnosis["diagnosis"]["selected_regions"],
            "summary": diagnosis["diagnosis"]["summary"],
        },
        "regions": [region.model_dump(mode="json", exclude_none=True) for region in selected_regions],
        "sampling": spec.sampling.model_dump(mode="json"),
        "variants": [variant.model_dump(mode="json") for variant in spec.variants],
        "output": {
            **spec.output.model_dump(mode="json"),
            "directory": output_dir.as_posix(),
            "files": outputs,
        },
        "acceptance": spec.acceptance.model_dump(mode="json"),
        "tokens": region_tokens,
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
        "tokens": resolved["tokens"],
        "outputs": resolved["output"]["files"],
    }


def _selected_regions(
    spec: KeyframePolishJobSpec,
    diagnosis: dict[str, Any],
) -> list[PolishRegionSpec]:
    diagnosed = PolishDiagnosisResponse.model_validate(diagnosis["diagnosis"])
    _validate_diagnosis_response(diagnosed, {region.name for region in spec.regions})
    if diagnosed.candidate != spec.base.candidate:
        raise KeyframePolishError(
            f"Polish diagnosis is for {diagnosed.candidate}, but job base candidate is {spec.base.candidate}"
        )
    regions_by_name = {region.name: region for region in spec.regions}
    selected = []
    for selected_region in sorted(diagnosed.selected_regions, key=lambda region: region.priority):
        if selected_region.name not in regions_by_name:
            raise KeyframePolishError(f"Polish diagnosis selected unavailable region: {selected_region.name}")
        selected.append(regions_by_name[selected_region.name])
    if not selected:
        raise KeyframePolishError("Polish diagnosis selected no regions")
    return selected


def plan_keyframe_polish_job(
    job_path: Path,
    profile: KeyframeRefineProfile,
    *,
    project_root: Path,
) -> dict[str, Any]:
    resolved = resolve_keyframe_polish_job(job_path, profile, project_root=project_root, check_outputs=True)
    mask_plans = build_polish_mask_plans(resolved)
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
    output_dir = Path(resolved["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_plans = build_polish_mask_plans(resolved)
    resolved = {**resolved, "mask_plan": [_mask_plan_json(plan) for plan in mask_plans]}
    _write_json(output_dir / "resolved.json", resolved)
    if spec.output.save_debug_images:
        _save_debug_images(resolved, mask_plans, output_dir)
    selected_regions = [PolishRegionSpec.model_validate(region) for region in resolved["regions"]]

    memory_sampler = NvidiaSmiMemorySampler(_nvidia_smi_preflight())
    memory_sampler.start()
    refiner: KontextInpaintRefiner | None = None
    try:
        total_start = perf_counter()
        refiner = KontextInpaintRefiner(profile)
        torch_module = refiner.torch
        base = Image.open(resolved["base"]["image"]["path"]).convert("RGB")
        identity_primer = Image.open(resolved["character"]["identity_primer"]["path"]).convert("RGB")
        outputs = []
        for variant, planned in zip(spec.variants, resolved["output"]["files"], strict=True):
            variant_start = synchronized_time(torch_module)
            polished = base.copy()
            region_outputs = []
            for region, mask_plan in zip(selected_regions, mask_plans, strict=True):
                base_crop = polished.crop(mask_plan.crop_box)
                mask_crop = mask_plan.feather_mask.crop(mask_plan.crop_box)
                refined_crop = refiner.refine(
                    base_crop=base_crop,
                    mask_crop=mask_crop,
                    reference_image=identity_primer,
                    clip_prompt=region.prompt.clip,
                    t5_prompt=region.prompt.t5,
                    negative_prompt=region.prompt.negative,
                    true_cfg_scale=region.prompt.true_cfg_scale,
                    steps=spec.sampling.steps,
                    guidance_scale=spec.sampling.guidance_scale,
                    strength=spec.sampling.strength,
                    max_sequence_length=spec.sampling.max_sequence_length,
                    seed=variant.seed,
                )
                before_region = polished
                polished = _paste_refined_crop(
                    polished,
                    refined_crop.convert("RGB"),
                    mask_plan.feather_mask,
                    mask_plan.crop_box,
                )
                region_outputs.append(
                    {
                        "name": region.name,
                        "seed": variant.seed,
                        "crop_box": list(mask_plan.crop_box),
                        "mask_change": _outside_mask_change(before_region, polished, mask_plan.feather_mask),
                    }
                )
            output_path = Path(planned["path"])
            polished.save(output_path)
            outputs.append(
                {
                    **planned,
                    "seed": variant.seed,
                    "timings_ms": {
                        "polish_ms": elapsed_ms(variant_start, synchronized_time(torch_module)),
                    },
                    "regions": region_outputs,
                    "hard_rejects": {
                        "outside_feather_changed": any(
                            region["mask_change"]["hard_rejects"]["outside_feather_changed"]
                            for region in region_outputs
                        ),
                    },
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
            "environment": _generation_environment(torch_module),
        }
        _write_json(output_dir / "result.json", result)
        return result
    finally:
        if refiner is not None:
            refiner.close()
        memory_sampler.stop()


def build_polish_mask_plans(resolved: dict[str, Any]) -> list[PolishMaskPlan]:
    base = Image.open(resolved["base"]["image"]["path"]).convert("RGB")
    return build_polish_mask_plans_for_image(base, resolved["regions"])


def build_polish_mask_plans_for_image(base: Image.Image, regions: list[dict[str, Any]]) -> list[PolishMaskPlan]:
    foreground = _foreground_mask(np.asarray(base, dtype=np.uint8))
    foreground = ndimage.binary_dilation(foreground, iterations=5)
    box = _bbox(foreground)
    return [_build_region_mask(region, foreground, box, base.size) for region in regions]


def _build_region_mask(
    region: dict[str, Any],
    foreground: np.ndarray,
    foreground_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> PolishMaskPlan:
    width, height = image_size
    hard = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(hard)
    left, top, right, bottom = foreground_box
    subject_w = right - left
    subject_h = bottom - top
    if region["mask"] == "auto_face":
        cx = left + subject_w * 0.52
        cy = top + subject_h * 0.14
        rx = subject_w * 0.15
        ry = subject_h * 0.13
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    elif region["mask"] == "auto_tie_shirt":
        _draw_ratio_box(draw, foreground_box, (0.48, 0.24, 0.66, 0.54))
    elif region["mask"] == "auto_skirt_belt":
        _draw_ratio_box(draw, foreground_box, (0.38, 0.50, 0.70, 0.67))
    else:
        raise KeyframePolishError(f"Unsupported polish mask: {region['mask']}")

    hard_array = np.asarray(hard, dtype=np.uint8) > 0
    hard_array &= foreground
    hard_array = ndimage.binary_dilation(hard_array, iterations=max(1, round(width * 0.012)))
    hard = Image.fromarray((hard_array.astype(np.uint8) * 255), mode="L")
    feather = hard.filter(ImageFilter.GaussianBlur(radius=region["feather_px"]))
    crop_box = _expanded_aligned_box(hard_array, region["crop_padding_px"], width, height)
    return PolishMaskPlan(
        region=region["name"],
        hard_mask=hard,
        feather_mask=feather,
        crop_box=crop_box,
    )


def _save_diagnosis_images(
    base: Image.Image,
    identity_primer: Path,
    mask_plans: list[PolishMaskPlan],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    primer_path = output_dir / "identity_primer.png"
    candidate_path = output_dir / "candidate.png"
    Image.open(identity_primer).convert("RGB").save(primer_path)
    base.save(candidate_path)
    paths.extend((primer_path, candidate_path))
    for index, mask_plan in enumerate(mask_plans, start=1):
        prefix = f"{index:02d}_{mask_plan.region}"
        overlay_path = output_dir / f"{prefix}_overlay.png"
        crop_path = output_dir / f"{prefix}_crop.png"
        _mask_overlay(base, mask_plan.feather_mask).save(overlay_path)
        base.crop(mask_plan.crop_box).save(crop_path)
        paths.extend((overlay_path, crop_path))
    return paths


def _diagnosis_prompt(
    candidate: str,
    effective_config: dict[str, Any],
    mask_plans: list[PolishMaskPlan],
) -> str:
    acceptance = effective_config["acceptance"]["manual"]
    keyframe = effective_config["keyframe"]
    image_lines = [
        "1. approved character identity primer",
        f"2. full selected candidate: {candidate}",
    ]
    for index, mask_plan in enumerate(mask_plans, start=1):
        overlay_number = 2 + (index * 2) - 1
        crop_number = overlay_number + 1
        image_lines.append(f"{overlay_number}. {mask_plan.region} mask overlay on the candidate")
        image_lines.append(f"{crop_number}. {mask_plan.region} local crop")
    region_lines = [
        f"- {mask_plan.region}: crop_box={list(mask_plan.crop_box)}, guidance={_region_diagnosis_hint(mask_plan.region)}"
        for mask_plan in mask_plans
    ]
    available_names = [mask_plan.region for mask_plan in mask_plans]
    return f"""You are a strict visual art-production QA model for local polish.

You receive these images in order:
{chr(10).join(image_lines)}

The candidate was already selected for structure and pose. Do not reject it for global pose, silhouette, or action framing.
Your job is to inspect each named local crop and decide which local detail regions need inpaint polish while preserving the current pose and silhouette.

Available polish regions:
{chr(10).join(region_lines)}

Use the overlay to locate the region and the crop to judge the local details. The full candidate is only context.
Do not select a region that is already acceptable. Do not invent region names. If a crop visibly disagrees with the identity primer or manual criteria, select that region.

Target keyframe:
- action: {keyframe["action"]}
- phase: {keyframe["phase"]}
- direction: {keyframe["direction"]}
- camera: {keyframe["camera"]}

Manual acceptance criteria:
{json.dumps(acceptance, ensure_ascii=False)}

Return JSON only. The JSON object must contain:
- candidate: exactly "{candidate}"
- region_assessments: an object keyed by every available region name, using only these names: {json.dumps(available_names)}
- selected_regions: every region whose assessment has needs_polish=true, ordered by polish priority
- summary: one sentence

Each region_assessments value has: needs_polish, reason.
Each selected_regions item has: name, priority, reason.
"""


def _region_diagnosis_hint(region: str) -> str:
    hints = {
        "face_expression": "select if the face, visible eye, mouth, expression, or small facial identity needs cleanup",
        "tie_and_shirt": "select if the blue tie, collar, or white shirt is unclear, broken, wrong-colored, or missing",
        "skirt_belt": "select if the brown leather skirt, waist, belt, or panel details became shorts, pants, mushy, or wrong",
    }
    return hints.get(region, "select if this local detail region visibly needs polish")


def _parse_diagnosis_response(raw_text: str) -> PolishDiagnosisResponse:
    data = _json_from_vlm_response(raw_text)
    try:
        return PolishDiagnosisResponse.model_validate(data)
    except ValidationError as error:
        raise KeyframePolishError(f"Polish diagnosis returned invalid JSON: {error}") from error


def _validate_diagnosis_response(response: PolishDiagnosisResponse, available_regions: set[str]) -> None:
    assessment_regions = set(response.region_assessments)
    if assessment_regions != available_regions:
        raise KeyframePolishError(
            f"Polish diagnosis assessed {sorted(assessment_regions)}, expected {sorted(available_regions)}"
        )
    selected_regions = {region.name for region in response.selected_regions}
    if not selected_regions <= available_regions:
        raise KeyframePolishError(f"Polish diagnosis selected unavailable regions: {sorted(selected_regions)}")
    needed_regions = {
        region_name
        for region_name, assessment in response.region_assessments.items()
        if assessment.needs_polish
    }
    if selected_regions != needed_regions:
        raise KeyframePolishError(
            f"Polish diagnosis selected {sorted(selected_regions)}, but assessments require {sorted(needed_regions)}"
        )


def _json_from_vlm_response(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise KeyframePolishError("Polish diagnosis returned an unterminated Markdown JSON block")
        text = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise KeyframePolishError(f"Polish diagnosis returned non-JSON output: {error}") from error
    if not isinstance(data, dict):
        raise KeyframePolishError("Polish diagnosis returned JSON that is not an object")
    return data


def _draw_ratio_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    ratios: tuple[float, float, float, float],
) -> None:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    draw.rectangle(
        (
            left + width * ratios[0],
            top + height * ratios[1],
            left + width * ratios[2],
            top + height * ratios[3],
        ),
        fill=255,
    )


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


def _mask_plan_json(mask_plan: PolishMaskPlan) -> dict[str, Any]:
    return {
        "region": mask_plan.region,
        "crop_box": list(mask_plan.crop_box),
    }


def _save_debug_images(resolved: dict[str, Any], mask_plans: list[PolishMaskPlan], output_dir: Path) -> None:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    base = Image.open(resolved["base"]["image"]["path"]).convert("RGB")
    for mask_plan in mask_plans:
        region_dir = debug_dir / mask_plan.region
        region_dir.mkdir(parents=True, exist_ok=True)
        mask_plan.hard_mask.save(region_dir / "mask_hard.png")
        mask_plan.feather_mask.save(region_dir / "mask_feather.png")
        base.crop(mask_plan.crop_box).save(region_dir / "crop.png")
        mask_plan.feather_mask.crop(mask_plan.crop_box).save(region_dir / "crop_mask.png")
        _mask_overlay(base, mask_plan.feather_mask).save(region_dir / "mask_overlay.png")


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


def _planned_outputs(spec: KeyframePolishJobSpec, output_dir: Path) -> list[dict[str, str | int]]:
    return [
        {
            "name": variant.name,
            "seed": variant.seed,
            "path": (output_dir / spec.output.filename.format(id=spec.id, variant=variant.name)).as_posix(),
        }
        for variant in spec.variants
    ]


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
