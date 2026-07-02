from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    framing: str
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
    evidence = _planner_evidence(manifest, canon_images)
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
            "framing": intent.framing,
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
            or candidate.framing != contract["framing"]
            or candidate.identity_primer != contract["identity_primer"]
        )
        if changed_contract:
            raise LoraCandidateBriefError(
                f"Invalid LoRA candidate planner response {raw_path.as_posix()}: "
                f"candidate {candidate.name} changed the requested view, pose, framing or identity_primer"
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
        or candidate.framing != intent.framing
        or candidate.identity_primer != intent.identity_primer
    ):
        raise LoraCandidateBriefError(
            f"Invalid LoRA candidate planner response {raw_path.as_posix()}: "
            f"candidate changed the requested name, view, pose, framing or identity_primer"
        )


def _validate_candidate_prompts(
    template_list: LoraCandidateTemplateListSpec,
    identity_prompt: str,
    raw_path: Path,
) -> None:
    normalized_identity = _compact_text(identity_prompt).lower()
    for candidate in template_list.candidates:
        normalized_prompt = _compact_text(candidate.prompt.positive).lower()
        if normalized_prompt == normalized_identity or normalized_identity in normalized_prompt:
            raise LoraCandidateBriefError(
                f"Invalid LoRA candidate planner response {raw_path.as_posix()}: "
                f"candidate {candidate.name} prompt copies the full user identity prompt"
            )


def _planner_evidence(
    manifest: dict[str, Any],
    canon_images: dict[str, dict[str, Any]],
) -> LoraCandidatePlannerEvidence:
    image_paths: list[Path] = []
    order_lines: list[str] = []
    for image in manifest["images"]:
        name = image["name"]
        path = Path(canon_images[name]["path"])
        image_paths.append(path)
        order_lines.append(f"canon anchor {name}: {path}")
    return LoraCandidatePlannerEvidence(image_paths=image_paths, order_lines=order_lines)


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

Write exactly one variable generation prompt suffix for one candidate image.
The generated candidate may later become character identity LoRA training data, but only after model evidence and human approval.
The training dataset must contain only canon-worthy identity drawings. Do not plan punch, hit, jump, combat, deformed, exaggerated or malformed action frames.
Inspect the supplied canon images yourself. Describe only the requested view, pose, framing, expression if needed, and clean background in prompt.positive.
The executor prepends this full user identity prompt to every generation:
{character['identity_prompt']}

Images are supplied in this exact order:
{image_order}

Use the canon images and the full user prompt to understand the character, but do not repeat the full user identity prompt in prompt.positive.
prompt.positive is concatenated after the full user identity prompt, so it should only add the requested camera, pose, framing, expression when needed, and background.
Never include the phrase looking at viewer in profile, side, back or rear-view candidate prompts.
For back or rear-view prompts, describe the rear camera angle and visible silhouette; do not request visible eyes, smile, blush or viewer-facing face details.
If the requested candidate view is back or rear, prompt.positive must not contain: eye, eyes, smile, smiling, blush, looking at viewer.
The generated candidates should cover useful identity views, intentional framing variety and mild pose variety for LoRA training: neutral front, left/right profile, mild three-quarter views, back view only when the canon evidence supports it, simple idle, simple walking or standing variations, full-body, thigh-up, waist-up and portrait/detail framing.
If more distinct candidates are needed, vary mild poses such as relaxed idle, hands-at-sides, small step, contrapposto or neutral stance, and vary intentional framing. Do not use accidental cut-off, malformed crops or close-up filler.
Changing only identity_primer or name is not a distinct candidate. Every candidate must have a unique view, pose and framing tuple. Do not repeat back view neutral standing more than once.
Avoid direct source-image copying, sprite-like output, top-down distortion, bottom-up distortion, extreme foreshortening, hard action poses and random costumes.
Every candidate must use a clean simple background. Partial-body candidates are valid only when the requested framing states exactly what is visible.

Available identity primer names: {primer_names}

Requested candidate intent. Do not invent, rename or alter these fields:
- name: {intent.name}
- view: {intent.view}
- pose: {intent.pose}
- framing: {intent.framing}
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
- framing: exact requested framing
- identity_primer: exact requested identity_primer
- prompt: object with exactly one key, positive

Exact JSON shape:
{{
  "name": "{intent.name}",
  "view": "{intent.view}",
  "pose": "{intent.pose}",
  "framing": "{intent.framing}",
  "identity_primer": "{intent.identity_primer}",
  "prompt": {{
    "positive": "variable camera, pose, framing and background suffix for this candidate"
  }}
}}

The positive prompt must be a suffix, not a full prompt. It must include the requested view, requested pose, requested framing and a clean simple background.
The positive prompt must not repeat the full user identity prompt and must not list the whole outfit. The executor prepends the full user identity prompt before this suffix.
For example, a front view candidate must include the words "front view" in positive; a left profile candidate must include "left" or "left-facing"; a right profile candidate must include "right" or "right-facing"; a back view candidate must include "back" or "rear"; a three-quarter view must include "three-quarter" or "3/4".
The generated image sees the full user identity prompt plus prompt.positive. It will not see name, view, pose, framing or identity_primer, so prompt.positive must materialize the requested view, pose and framing.
The name must describe the actual output view and pose, not the source primer. A name containing left, right, front or back must match the view field.
The view field must describe only camera angle, such as "front view", "left profile view", "right profile view" or "three-quarter front view".
The pose field must describe only body pose, such as "neutral standing pose", "relaxed idle pose" or "mild walk contact pose".
The framing field must describe only visible body coverage, such as "full body", "thigh-up", "waist-up", "bust portrait" or "head-and-shoulders portrait".
The positive prompt is not the full generation prompt. The executor adds the full user identity prompt before this suffix.
The positive prompt must not be the full identity prompt copied verbatim.
Do not include the trigger token in candidate prompts. Candidate prompts are used for pre-LoRA image generation, where the trigger token has no meaning. The executor adds the trigger token only to training captions after human approval.
Do not use a JSON string for prompt. prompt must always be an object with positive.
Do not return placeholder text, markdown or explanations."""


def _candidate_intents(canon_images: dict[str, dict[str, Any]], candidate_count: int) -> list[LoraCandidateIntent]:
    front = _primer(canon_images, "front")
    side = _primer(canon_images, "left_profile")
    templates = [
        ("front_neutral_full_body", "front view", "neutral standing pose", "full body", front),
        ("front_neutral_thigh_up", "front view", "neutral standing pose", "thigh-up", front),
        ("front_neutral_waist_up", "front view", "neutral standing pose", "waist-up", front),
        ("front_neutral_portrait", "front view", "neutral standing pose", "head-and-shoulders portrait", front),
        ("left_profile_neutral_full_body", "left profile view", "neutral standing pose", "full body", side),
        ("left_profile_neutral_thigh_up", "left profile view", "neutral standing pose", "thigh-up", side),
        ("right_profile_neutral_full_body", "right profile view", "neutral standing pose", "full body", side),
        ("right_profile_neutral_thigh_up", "right profile view", "neutral standing pose", "thigh-up", side),
        ("three_quarter_front_neutral_thigh_up", "three-quarter front view", "neutral standing pose", "thigh-up", front),
        ("three_quarter_left_neutral_thigh_up", "three-quarter left view", "neutral standing pose", "thigh-up", side),
        ("three_quarter_right_neutral_thigh_up", "three-quarter right view", "neutral standing pose", "thigh-up", side),
        ("front_relaxed_idle_thigh_up", "front view", "relaxed idle pose", "thigh-up", front),
        ("left_profile_relaxed_idle_full_body", "left profile view", "relaxed idle pose", "full body", side),
        ("right_profile_relaxed_idle_full_body", "right profile view", "relaxed idle pose", "full body", side),
        ("left_profile_walk_contact_full_body", "left profile view", "mild walk contact pose", "full body", side),
        ("right_profile_walk_contact_full_body", "right profile view", "mild walk contact pose", "full body", side),
        ("front_small_step_full_body", "front view", "small step pose", "full body", front),
    ]
    back_primer = _back_primer(canon_images)
    if back_primer is not None:
        templates.insert(8, ("back_neutral_full_body", "back view", "neutral standing pose", "full body", back_primer))
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


def _compact_text(value: str) -> str:
    return " ".join(value.split())
