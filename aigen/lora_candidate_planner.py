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
    LoraCandidateTemplateSpec,
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


@dataclass(frozen=True)
class LoraCandidateIntent:
    name: str
    view: str
    pose: str
    identity_primer: str


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
    intents = _candidate_intents(canon_images, plan_config.candidate_count)
    prompt_dir = output_path.with_suffix(".prompts")
    raw_dir = output_path.with_suffix(".raw")
    if prompt_dir.exists():
        for path in prompt_dir.iterdir():
            path.unlink()
    else:
        prompt_dir.mkdir(parents=True)
    if raw_dir.exists():
        for path in raw_dir.iterdir():
            path.unlink()
    else:
        raw_dir.mkdir(parents=True)

    progress.begin(len(intents), "plan LoRA candidate prompts")
    with closing(QwenVlm(config)) as active_runner:
        templates = _candidate_templates_from_vlm(
            active_runner,
            manifest,
            canon_images,
            evidence,
            intents,
            plan_config,
            prompt_dir,
            raw_dir,
            evidence.image_paths,
            progress,
        )
        template_list = LoraCandidateTemplateListSpec(candidates=templates)

    _validate_identity_primers(template_list, canon_images, output_path)
    _validate_candidate_intents(template_list, intents, output_path)
    _validate_candidate_prompts(template_list, manifest["character"]["identity_prompt"], output_path)

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
    output = {
        "brief": output_path.as_posix(),
        "prompts": prompt_dir.as_posix(),
        "raw_responses": raw_dir.as_posix(),
        "evidence": (output_path.parent / "planner_evidence").as_posix(),
    }
    return {
        "status": "planned",
        "kind": "lora-candidate-brief-plan",
        "character": manifest["character"]["id"],
        "candidate_count": len(template_list.candidates),
        "planner": qwen_vlm_config_json(config),
        "output": output,
    }


def _candidate_templates_from_vlm(
    runner: QwenVlm,
    manifest: dict[str, Any],
    canon_images: dict[str, dict[str, Any]],
    evidence: LoraCandidatePlannerEvidence,
    intents: list[LoraCandidateIntent],
    plan_config: LoraCandidateBriefPlanConfig,
    prompt_dir: Path,
    raw_dir: Path,
    image_paths: list[Path],
    progress: StatusReporter,
) -> list[LoraCandidateTemplateSpec]:
    templates = []
    for intent in intents:
        prompt = _single_candidate_prompt(manifest, canon_images, evidence.order_lines, intent, plan_config)
        prompt_path = prompt_dir / f"{intent.name}.txt"
        raw_path = raw_dir / f"{intent.name}.txt"
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        raw_text = runner.judge_candidate(prompt, image_paths)
        raw_path.write_text(raw_text + "\n", encoding="utf-8")
        template = _single_template_from_response_or_repair(
            runner,
            prompt,
            raw_text,
            raw_path,
            prompt_path,
            image_paths,
        )
        _validate_candidate_intent(template, intent, raw_path)
        templates.append(template)
        progress.step(intent.name)
    return templates


def _single_template_from_response_or_repair(
    runner: QwenVlm,
    prompt: str,
    raw_text: str,
    raw_path: Path,
    prompt_path: Path,
    image_paths: list[Path],
) -> LoraCandidateTemplateSpec:
    try:
        return _single_template_from_response(raw_text, raw_path)
    except LoraCandidateBriefError as error:
        repair_prompt_path = prompt_path.with_name(f"{prompt_path.stem}.repair.txt")
        repair_raw_path = raw_path.with_name(f"{raw_path.stem}.repair.txt")
        repair_prompt = _single_candidate_repair_prompt(prompt, raw_text, str(error))
        repair_prompt_path.write_text(repair_prompt + "\n", encoding="utf-8")
        repaired_text = runner.judge_candidate(repair_prompt, image_paths)
        repair_raw_path.write_text(repaired_text + "\n", encoding="utf-8")
        return _single_template_from_response(repaired_text, repair_raw_path)


def _single_template_from_response(raw_text: str, raw_path: Path) -> LoraCandidateTemplateSpec:
    try:
        generated = json_object_from_vlm_response(raw_text)
    except VlmJsonError as error:
        raise LoraCandidateBriefError(f"Invalid LoRA candidate planner response {raw_path.as_posix()}: {error}") from error
    try:
        return LoraCandidateTemplateSpec.model_validate(generated)
    except ValidationError as error:
        raise LoraCandidateBriefError(
            f"Invalid LoRA candidate planner response {raw_path.as_posix()}: {error}"
        ) from error


def _single_candidate_repair_prompt(prompt: str, raw_text: str, validation_error: str) -> str:
    return f"""{prompt}

Your previous response failed validation:
{validation_error}

Return one corrected JSON object for the same requested candidate only.
Do not use Markdown. Do not explain.
Do not reuse any phrase named in the validation error.

Previous response:
{raw_text}"""


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


def _validate_candidate_intents(
    template_list: LoraCandidateTemplateListSpec,
    intents: list[LoraCandidateIntent],
    raw_path: Path,
) -> None:
    expected = {
        intent.name: {
            "view": intent.view,
            "pose": intent.pose,
            "identity_primer": intent.identity_primer,
        }
        for intent in intents
    }
    for candidate in template_list.candidates:
        if candidate.name not in expected:
            raise LoraCandidateBriefError(
                f"Invalid LoRA candidate planner response {raw_path.as_posix()}: "
                f"unexpected candidate {candidate.name}"
            )
        contract = expected[candidate.name]
        changed_contract = (
            candidate.view != contract["view"]
            or candidate.pose != contract["pose"]
            or candidate.identity_primer != contract["identity_primer"]
        )
        if changed_contract:
            raise LoraCandidateBriefError(
                f"Invalid LoRA candidate planner response {raw_path.as_posix()}: "
                f"candidate {candidate.name} changed the requested view, pose or identity_primer"
            )


def _validate_candidate_intent(
    candidate: LoraCandidateTemplateSpec,
    intent: LoraCandidateIntent,
    raw_path: Path,
) -> None:
    if (
        candidate.name != intent.name
        or candidate.view != intent.view
        or candidate.pose != intent.pose
        or candidate.identity_primer != intent.identity_primer
    ):
        raise LoraCandidateBriefError(
            f"Invalid LoRA candidate planner response {raw_path.as_posix()}: "
            f"candidate changed the requested name, view, pose or identity_primer"
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
            order_lines.append(f"canon anchor {name} {label}: {crop_path}")
    return LoraCandidatePlannerEvidence(image_paths=image_paths, order_lines=order_lines)


def _detail_crops(path: Path, output_dir: Path, name: str) -> dict[str, Path]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        crops = {
            "face_hair_detail": rgb.crop((0, 0, width, round(height * 0.38))),
            "torso_outfit_detail": rgb.crop((0, round(height * 0.24), width, round(height * 0.76))),
            "waist_legs_footwear_detail": rgb.crop((0, round(height * 0.55), width, height)),
        }
    outputs = {}
    for label, crop in crops.items():
        target = output_dir / f"{_slug(name)}_{label}.png"
        crop.save(target)
        outputs[label] = target
    return outputs


def _single_candidate_prompt(
    manifest: dict[str, Any],
    canon_images: dict[str, dict[str, Any]],
    image_order_lines: list[str],
    intent: LoraCandidateIntent,
    plan_config: LoraCandidateBriefPlanConfig,
) -> str:
    character = manifest["character"]
    image_order = "\n".join(f"{index}. {line}" for index, line in enumerate(image_order_lines, start=1))
    primer_names = ", ".join(canon_images.keys())
    return f"""You are planning a production LoRA candidate-generation brief from approved character canon images.

Write exactly one pre-LoRA generation prompt for one candidate image.
The generated candidate may later become character identity LoRA training data, but only after model evidence and human approval.
The training dataset must contain only canon-worthy identity drawings. Do not plan punch, hit, jump, combat, deformed, exaggerated or malformed action frames.
Inspect the supplied canon images and detail images yourself. Describe the character from the images and from the full user identity prompt.
The full user identity prompt is:
{character['identity_prompt']}

Images are supplied in this exact order:
{image_order}

Use the whole prompt above. Preserve every visible identity feature: subject, hair, eyes, upper clothing, neckwear, waist garment, belt, legwear, footwear, body proportions, visual style and background cleanliness.
Treat camera or pose wording in the user prompt, such as looking at viewer or standing, as source-image evidence. Omit or adapt it when it conflicts with the candidate view or pose.
Never include the phrase looking at viewer in profile, side, back or rear-view candidate prompts.
For back or rear-view prompts, describe visible rear clothing, hair shape, legwear, footwear and silhouette; do not request visible eyes, smile, blush or viewer-facing face details.
If the requested candidate view is back or rear, prompt.positive must not contain: blue eyes, eye, eyes, smile, smiling, blush, looking at viewer, flat-chested, small breasts.
Detail images in the evidence list are inspection aids only. Never create candidates whose view, pose, name or prompt describes crop, close-up or partial-body framing.
The generated candidates should cover useful identity views and mild pose variety for LoRA training: neutral front, left/right profile, mild three-quarter views, back view only when the canon evidence supports it, simple idle, simple walking or standing variations.
If more distinct candidates are needed, vary full-body mild poses such as relaxed idle, hands-at-sides, small step, contrapposto or neutral stance. Do not use close-up, upper-body, lower-body or crop variants to fill the candidate count.
Changing only identity_primer or name is not a distinct candidate. Every candidate must have a unique view and pose pair. Do not repeat back view neutral standing more than once.
Avoid direct source-image copying, sprite-like output, top-down distortion, bottom-up distortion, extreme foreshortening, hard action poses and random costumes.
Every candidate must show the complete character with a clean simple background.

Available identity primer names: {primer_names}

Requested candidate intent. Do not invent, rename or alter these fields:
- name: {intent.name}
- view: {intent.view}
- pose: {intent.pose}
- identity_primer: {intent.identity_primer}

Runtime contract:
- canvas: {plan_config.width}x{plan_config.height}
- steps: {plan_config.steps}
- seed_start: {plan_config.seed_start}
- seeds_per_candidate: {plan_config.seeds_per_candidate}

Return JSON only. The first character of your response must be "{{" and the last character must be "}}".
The JSON object must contain exactly:
- name: exact requested name
- view: exact requested view
- pose: exact requested pose
- identity_primer: exact requested identity_primer
- prompt: object with exactly one key, positive

Exact JSON shape:
{{
  "name": "{intent.name}",
  "view": "{intent.view}",
  "pose": "{intent.pose}",
  "identity_primer": "{intent.identity_primer}",
  "prompt": {{
    "positive": "full visual prompt for this candidate"
  }}
}}

The positive prompt must be written by inspecting the images. It must include the character identity, clothing, waist garment, belt if visible, legwear, footwear, view, pose, full-body framing, a clean simple background and the visual style inferred from the canon images.
Each positive prompt must explicitly mention the background and the observed visual style in its own words.
The visual style must be a concise art-medium phrase inferred from the images, such as anime-style illustration, clean ink drawing, painterly concept art or another phrase that actually matches the supplied canon.
Do not use filler style phrases like medium style, consistent visual style or detailed character design in place of concrete visual description.
Start each positive prompt with the inferred medium/style phrase, then the exact requested view, then the exact requested pose. For example, a front view candidate must include the words "front view" in positive; a left profile candidate must include "left" or "left-facing"; a right profile candidate must include "right" or "right-facing"; a back view candidate must include "back" or "rear"; a three-quarter view must include "three-quarter" or "3/4".
The generated image sees only prompt.positive. It will not see name, view, pose or identity_primer, so prompt.positive must materialize all of them.
The name must describe the actual output view and pose, not the source primer. A name containing left, right, front or back must match the view field.
The view field must describe only camera angle, such as "front view", "left profile view", "right profile view" or "three-quarter front view".
The pose field must describe only body pose, such as "neutral standing pose", "relaxed idle pose" or "mild walk contact pose".
The positive prompt is the exact generation prompt. Do not rely on the executor to add identity, view or pose text later.
The positive prompt must not be just the full identity prompt copied verbatim.
Do not include the trigger token in candidate prompts. Candidate prompts are used for pre-LoRA image generation, where the trigger token has no meaning. The executor adds the trigger token only to training captions after human approval.
Do not use a JSON string for prompt. prompt must always be an object with positive.
Do not return placeholder text, markdown or explanations."""


def _candidate_intents(canon_images: dict[str, dict[str, Any]], candidate_count: int) -> list[LoraCandidateIntent]:
    front = _primer(canon_images, "front")
    side = _primer(canon_images, "left_profile")
    templates = [
        ("front_neutral_standing", "front view", "neutral standing pose", front),
        ("left_profile_neutral_standing", "left profile view", "neutral standing pose", side),
        ("right_profile_neutral_standing", "right profile view", "neutral standing pose", side),
        ("three_quarter_front_neutral_standing", "three-quarter front view", "neutral standing pose", front),
        ("three_quarter_left_neutral_standing", "three-quarter left view", "neutral standing pose", side),
        ("three_quarter_right_neutral_standing", "three-quarter right view", "neutral standing pose", side),
        ("front_relaxed_idle", "front view", "relaxed idle pose", front),
        ("left_profile_relaxed_idle", "left profile view", "relaxed idle pose", side),
        ("right_profile_relaxed_idle", "right profile view", "relaxed idle pose", side),
        ("left_profile_walk_contact", "left profile view", "mild walk contact pose", side),
        ("right_profile_walk_contact", "right profile view", "mild walk contact pose", side),
        ("front_small_step", "front view", "small step pose", front),
        ("left_profile_small_step", "left profile view", "small step pose", side),
        ("right_profile_small_step", "right profile view", "small step pose", side),
    ]
    back_primer = _back_primer(canon_images)
    if back_primer is not None:
        templates.insert(6, ("back_neutral_standing", "back view", "neutral standing pose", back_primer))
    if candidate_count > len(templates):
        raise LoraCandidateBriefError(f"LoRA candidate planner supports at most {len(templates)} candidate intents")
    return [LoraCandidateIntent(*template) for template in templates[:candidate_count]]


def _primer(canon_images: dict[str, dict[str, Any]], preferred: str) -> str:
    if preferred in canon_images:
        return preferred
    return next(iter(canon_images))


def _back_primer(canon_images: dict[str, dict[str, Any]]) -> str | None:
    for name in ("back", "back_view", "rear", "rear_view"):
        if name in canon_images:
            return name
    return None


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
