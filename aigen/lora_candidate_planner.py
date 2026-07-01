from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import ValidationError

from aigen.lora_candidate_models import (
    LORA_CANDIDATE_BRIEF_SCHEMA,
    LoraCandidateBriefError,
    LoraCandidateBriefSpec,
    LoraCandidateTemplateListSpec,
)
from aigen.lora_canon import load_lora_canon_manifest, lora_canon_images_by_name
from aigen.manifest_io import relative_path, schema_reference, write_json
from aigen.progress import StatusReporter
from aigen.vlm_json import VlmJsonError, json_object_from_vlm_response
from aigen.vlm_qwen import QwenVlm, QwenVlmConfig, qwen_vlm_config_json


@dataclass(frozen=True)
class LoraCandidateBriefPlanConfig:
    width: int
    height: int
    steps: int
    seed_start: int
    seeds_per_candidate: int
    candidate_count: int
    candidate_output_dir: Path


@dataclass(frozen=True)
class LoraCandidatePlannerEvidence:
    image_paths: list[Path]
    order_lines: list[str]


def plan_lora_candidate_brief(
    canon_dir: Path,
    config: QwenVlmConfig,
    *,
    output_path: Path,
    plan_config: LoraCandidateBriefPlanConfig,
    project_root: Path,
    progress: StatusReporter,
) -> dict[str, Any]:
    canon_dir = canon_dir.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_lora_canon_manifest(canon_dir)
    canon_images = lora_canon_images_by_name(manifest, canon_dir)
    evidence = _planner_evidence(manifest, canon_images, output_path.parent / "planner_evidence")
    prompt = _planner_prompt(manifest, canon_images, evidence.order_lines, plan_config)
    raw_path = output_path.with_suffix(".raw.txt")
    prompt_path = output_path.with_suffix(".prompt.txt")
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    progress.phase("plan LoRA candidate brief")
    with closing(QwenVlm(config)) as active_runner:
        raw_text = active_runner.judge_candidate(prompt, evidence.image_paths)
        raw_path.write_text(raw_text + "\n", encoding="utf-8")
        template_list = _template_list_from_response(raw_text, output_path)

    if len(template_list.candidates) != plan_config.candidate_count:
        raise LoraCandidateBriefError(
            f"Invalid LoRA candidate planner response {raw_path.as_posix()}: "
            f"expected {plan_config.candidate_count} candidates, got {len(template_list.candidates)}"
        )
    _validate_identity_primers(template_list, canon_images, raw_path)
    _validate_candidate_prompts(template_list, manifest["character"]["identity_prompt"], raw_path)

    progress.phase("write LoRA candidate brief")
    brief = _candidate_brief(
        manifest,
        canon_dir,
        template_list,
        output_path,
        plan_config,
        project_root,
    )
    write_json(output_path, brief.model_dump(mode="json", by_alias=True))
    return {
        "status": "planned",
        "kind": "lora-candidate-brief-plan",
        "character": manifest["character"]["id"],
        "candidate_count": len(template_list.candidates),
        "planner": qwen_vlm_config_json(config),
        "output": {
            "brief": output_path.as_posix(),
            "prompt": prompt_path.as_posix(),
            "raw_response": raw_path.as_posix(),
            "evidence": (output_path.parent / "planner_evidence").as_posix(),
        },
    }


def _template_list_from_response(raw_text: str, output_path: Path) -> LoraCandidateTemplateListSpec:
    try:
        generated = json_object_from_vlm_response(raw_text)
    except VlmJsonError as error:
        raise LoraCandidateBriefError(f"Invalid LoRA candidate planner response {output_path.as_posix()}: {error}") from error
    try:
        return LoraCandidateTemplateListSpec.model_validate(generated)
    except ValidationError as error:
        raise LoraCandidateBriefError(
            f"Invalid LoRA candidate planner response {output_path.as_posix()}: {error}"
        ) from error


def _candidate_brief(
    manifest: dict[str, Any],
    canon_dir: Path,
    template_list: LoraCandidateTemplateListSpec,
    output_path: Path,
    plan_config: LoraCandidateBriefPlanConfig,
    project_root: Path,
) -> LoraCandidateBriefSpec:
    payload = {
        "$schema": schema_reference(output_path, project_root / LORA_CANDIDATE_BRIEF_SCHEMA),
        "kind": "lora-candidate-brief",
        "id": f"{manifest['character']['id']}.lora.candidates",
        "character": {
            "canon": relative_path(canon_dir, output_path.parent),
        },
        "generation": {
            "width": plan_config.width,
            "height": plan_config.height,
            "steps": plan_config.steps,
            "seed_start": plan_config.seed_start,
            "seeds_per_candidate": plan_config.seeds_per_candidate,
        },
        "candidates": [candidate.model_dump(mode="json") for candidate in template_list.candidates],
        "output": {
            "directory": relative_path(plan_config.candidate_output_dir, output_path.parent),
            "overwrite": True,
        },
    }
    return LoraCandidateBriefSpec.model_validate(payload)


def _validate_identity_primers(
    template_list: LoraCandidateTemplateListSpec,
    canon_images: dict[str, dict[str, Any]],
    raw_path: Path,
) -> None:
    for candidate in template_list.candidates:
        if candidate.identity_primer not in canon_images:
            raise LoraCandidateBriefError(
                f"Invalid LoRA candidate planner response {raw_path.as_posix()}: "
                f"candidate {candidate.name} uses unknown identity_primer {candidate.identity_primer}"
            )


def _validate_candidate_prompts(
    template_list: LoraCandidateTemplateListSpec,
    identity_prompt: str,
    raw_path: Path,
) -> None:
    normalized_identity = _compact_text(identity_prompt).lower()
    for candidate in template_list.candidates:
        normalized_prompt = _compact_text(candidate.prompt.positive).lower()
        if normalized_prompt == normalized_identity:
            raise LoraCandidateBriefError(
                f"Invalid LoRA candidate planner response {raw_path.as_posix()}: "
                f"candidate {candidate.name} prompt only copies the canon identity prompt"
            )


def _planner_evidence(
    manifest: dict[str, Any],
    canon_images: dict[str, dict[str, Any]],
    output_dir: Path,
) -> LoraCandidatePlannerEvidence:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    order_lines: list[str] = []
    for image in manifest["images"]:
        name = image["name"]
        path = Path(canon_images[name]["path"])
        image_paths.append(path)
        order_lines.append(f"canon anchor {name}: {path}")
        for label, crop_path in _detail_crops(path, output_dir, name).items():
            image_paths.append(crop_path)
            order_lines.append(f"canon anchor {name} {label} crop: {crop_path}")
    return LoraCandidatePlannerEvidence(image_paths=image_paths, order_lines=order_lines)


def _detail_crops(path: Path, output_dir: Path, name: str) -> dict[str, Path]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        crops = {
            "upper": rgb.crop((0, 0, width, round(height * 0.38))),
            "middle": rgb.crop((0, round(height * 0.24), width, round(height * 0.76))),
            "lower": rgb.crop((0, round(height * 0.55), width, height)),
        }
    outputs = {}
    for label, crop in crops.items():
        target = output_dir / f"{_slug(name)}_{label}.png"
        crop.save(target)
        outputs[label] = target
    return outputs


def _planner_prompt(
    manifest: dict[str, Any],
    canon_images: dict[str, dict[str, Any]],
    image_order_lines: list[str],
    plan_config: LoraCandidateBriefPlanConfig,
) -> str:
    character = manifest["character"]
    image_order = "\n".join(f"{index}. {line}" for index, line in enumerate(image_order_lines, start=1))
    primer_names = ", ".join(canon_images.keys())
    return f"""You are planning a production LoRA candidate-generation brief from approved character canon images.

The candidate brief will generate candidate images for a character identity LoRA dataset.
The training dataset must contain only canon-worthy identity drawings. Do not plan punch, hit, jump, combat, deformed, exaggerated or malformed action frames.
Inspect the supplied canon images and crops yourself. Describe the character from the images and from the full user identity prompt.
The full user identity prompt is:
{character['identity_prompt']}

Images are supplied in this exact order:
{image_order}

Use the whole prompt above. Preserve every visible identity feature: subject, hair, eyes, upper clothing, neckwear, waist garment, belt, legwear, footwear, body proportions, lineart style and background cleanliness.
The generated candidates should cover useful identity views and mild pose variety for LoRA training: neutral front, left/right profile, mild three-quarter views, back view only when the canon evidence supports it, simple idle, simple walking or standing variations.
Avoid direct source-image copying, sprite-like output, top-down distortion, bottom-up distortion, extreme foreshortening, hard action poses and random costumes.
Every candidate must be full body with a clean neutral background.

Available identity primer names: {primer_names}

Runtime contract:
- canvas: {plan_config.width}x{plan_config.height}
- steps: {plan_config.steps}
- seed_start: {plan_config.seed_start}
- seeds_per_candidate: {plan_config.seeds_per_candidate}
- candidate_count: {plan_config.candidate_count}

Return JSON only. The first character of your response must be "{" and the last character must be "}".
The JSON object must contain exactly one top-level key: candidates.
Do not return the candidates array directly.
candidates must contain exactly {plan_config.candidate_count} objects.
Each candidate object must contain exactly:
- name: short file-safe id, unique, no spaces
- view: concrete visible camera/view description
- pose: concrete mild pose description
- identity_primer: one exact name from the available identity primer names
- prompt: object with exactly one key, positive

Exact top-level JSON shape:
{{
  "candidates": [
    {{
      "name": "descriptive_candidate_id",
      "view": "concrete view",
      "pose": "concrete mild pose",
      "identity_primer": "front",
      "prompt": {{
        "positive": "full visual prompt for this candidate"
      }}
    }}
  ]
}}
The example above shows the shape only. Your real response must include exactly {plan_config.candidate_count} candidate objects inside the candidates array.

The positive prompt must be written by inspecting the images. It must include the character identity, clothing, waist garment, belt if visible, legwear, footwear, view and pose.
Every positive prompt must contain these exact literal phrases:
- full body
- clean neutral background
- clean anime lineart
The view field must describe only camera angle, such as "front view", "left profile view", "right profile view" or "three-quarter front view".
The pose field must describe only body pose, such as "neutral standing pose", "relaxed idle pose" or "mild walk contact pose".
The executor materializes view and pose into the final generation prompt, so the positive prompt may focus on character identity, outfit, background and style.
The positive prompt must not be just the full identity prompt copied verbatim.
Do not include the trigger token in candidate prompts. Candidate prompts are used for pre-LoRA image generation, where the trigger token has no meaning. The executor adds the trigger token only to training captions after human approval.
Do not use a JSON string for prompt. prompt must always be an object with positive.
Do not use numbered template names like front_0, front_1, left_profile_0 or candidate_01. Names must describe the view and pose, such as front_neutral_standing or left_profile_walk_contact.
Do not produce repeated front or left-profile sequences where only a number changes. Every candidate must have a distinct useful view/pose intent.
Do not return placeholder text, markdown or explanations."""


def _slug(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif cleaned and cleaned[-1] != "_":
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "canon"


def _compact_text(value: str) -> str:
    return " ".join(value.split())
