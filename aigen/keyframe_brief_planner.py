from __future__ import annotations

import json
from contextlib import closing, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import ValidationError

from aigen.character_view_models import CharacterViewBankSpec, load_character_view_bank
from aigen.keyframe_brief_models import (
    KEYFRAME_BRIEF_PLAN_KIND,
    KEYFRAME_BRIEF_PLAN_SCHEMA,
    KEYFRAME_BRIEF_PLAN_SCHEMA_VERSION,
    KeyframeBriefError,
    KeyframeBriefPlanSpec,
    KeyframeBriefSpec,
    load_keyframe_brief,
)
from aigen.keyframe_judge import KeyframeJudgeConfig, QwenKeyframeJudge
from aigen.manifest_io import (
    resolve_existing_path,
    resolve_output_path,
    schema_reference,
    sha256_bytes,
    sha256_file,
    write_json,
)
from aigen.vlm_json import VlmJsonError, json_object_from_vlm_response


def plan_keyframe_brief(
    brief_path: Path,
    config: KeyframeJudgeConfig,
    *,
    project_root: Path,
    runner: Any | None = None,
) -> dict[str, Any]:
    spec = load_keyframe_brief(brief_path)
    view_bank_path = resolve_existing_path(spec.character.view_bank.path, brief_path.parent)
    example_path = resolve_existing_path(spec.example.path, brief_path.parent)
    view_bank = load_character_view_bank(view_bank_path)
    view_options = _view_options(view_bank)
    plan_path = resolve_output_path(spec.output.plan_path, brief_path.parent)
    evidence = _planner_evidence_paths(view_options, example_path, plan_path.parent / "planner_evidence")
    prompt = _planner_prompt(spec, view_options, evidence.order_lines)
    raw_path = plan_path.with_suffix(".raw.txt")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(QwenKeyframeJudge(config)) if runner is None else nullcontext(runner) as active_runner:
        raw_text = active_runner.judge_candidate(
            prompt,
            evidence.image_paths,
        )
        raw_path.write_text(raw_text + "\n", encoding="utf-8")
        try:
            plan = _validate_generated_plan(raw_text, spec, config, brief_path, prompt, view_options, plan_path, project_root)
        except ValidationError as error:
            repair_prompt = _planner_repair_prompt(raw_text, error)
            repair_text = active_runner.judge_candidate(repair_prompt, evidence.image_paths)
            plan_path.with_suffix(".repair.raw.txt").write_text(repair_text + "\n", encoding="utf-8")
            try:
                plan = _validate_generated_plan(repair_text, spec, config, brief_path, prompt, view_options, plan_path, project_root)
            except ValidationError as repair_error:
                raise KeyframeBriefError(f"Invalid generated brief plan {plan_path}: {repair_error}") from repair_error
    write_json(plan_path, plan.model_dump(mode="json", by_alias=True, exclude_none=True))
    return {
        "status": "planned",
        "brief_id": spec.id,
        "plan_path": plan_path.as_posix(),
        "raw_response": raw_path.as_posix(),
        "identity_primer": plan.identity_primer.model_dump(mode="json"),
        "controls": [control.model_dump(mode="json", exclude_none=True) for control in plan.controls],
        "scoring": plan.scoring.model_dump(mode="json"),
        "polish": plan.polish.model_dump(mode="json"),
    }


def _validate_generated_plan(
    raw_text: str,
    spec: KeyframeBriefSpec,
    config: KeyframeJudgeConfig,
    brief_path: Path,
    prompt: str,
    view_options: list[dict[str, Any]],
    plan_path: Path,
    project_root: Path,
) -> KeyframeBriefPlanSpec:
    generated = _json_object(raw_text)
    payload = {
        "$schema": schema_reference(plan_path, project_root / KEYFRAME_BRIEF_PLAN_SCHEMA),
        "schema_version": KEYFRAME_BRIEF_PLAN_SCHEMA_VERSION,
        "kind": KEYFRAME_BRIEF_PLAN_KIND,
        "brief_id": spec.id,
        "planner_id": config.judge_id,
        "source_brief_sha256": sha256_file(brief_path),
        "planner_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        **generated,
    }
    plan = KeyframeBriefPlanSpec.model_validate(payload)
    _validate_plan_against_brief(plan, spec, view_options, plan_path)
    return plan


@dataclass(frozen=True)
class PlannerEvidence:
    image_paths: list[Path]
    order_lines: list[str]


def _planner_prompt(spec: KeyframeBriefSpec, view_options: list[dict[str, Any]], image_order_lines: list[str]) -> str:
    view_lines = "\n".join(_view_option_line(option) for option in view_options)
    image_order = "\n".join(f"{index}. {line}" for index, line in enumerate(image_order_lines, start=1))
    contract = {
        "identity_details": {
            "subject": "visible subject type",
            "hair": "hair color, length and shape",
            "face": "visible face and expression details",
            "upper_clothing": "upper-body clothing",
            "neckwear": "tie, scarf, collar item or no visible neckwear",
            "waist_garment": "skirt, shorts, belt, pants waist or no visible waist garment",
            "legwear": "socks, stockings, pants legs or bare legs",
            "footwear": "boots, shoes or no visible footwear",
            "style": "art style, lineart, rendering and palette",
        },
        "identity_description": "caption from identity images",
        "pose_description": "caption from example sprite",
        "platformer_camera_description": "camera/readability interpretation",
        "identity_primer": {
            "view": "one available view name",
            "path": "the exact matching available view path",
        },
        "prompt": {
            "clip": "short prompt built from the supplied images",
            "t5": "detailed prompt built from the supplied images",
            "true_cfg_scale": 1.0,
        },
        "canvas": {
            "width": spec.example.width,
            "height": spec.example.height,
            "reference_max_area": 294912,
            "max_sequence_length": 128,
        },
        "sampling": {
            "steps": 28,
            "guidance_scale": 2.6,
        },
        "controls": [
            {
                "name": "pose",
                "type": "pose",
                "source": "example_pose",
                "scale": 0.7,
                "start": 0.0,
                "end": 0.65,
            },
            {
                "name": "silhouette",
                "type": "softedge",
                "source": "example_softedge",
                "scale": 0.7,
                "start": 0.0,
                "end": 0.8,
                "residual_mask_source": "example_boundary_mask",
            },
        ],
        "scoring": {
            "top_k": min(3, spec.generation.seed_count),
            "priorities": [
                "target pose and source contour match",
                "requested action reads clearly",
                "approved identity primer is preserved",
            ],
            "checks": [
                "whole character and feet remain visible",
                "requested action phase is readable",
                "hair, clothing, legwear and footwear match the identity primer",
            ],
        },
        "polish": {
            "profile": "kontext-inpaint-local",
            "max_regions": 4,
            "strength_offsets": [-0.06, 0.0, 0.06],
            "seed_offsets": [0, 1],
        },
        "rationale": ["one concrete reason"],
    }
    return f"""You are planning a production AI keyframe-generation job.

The user supplies an identity view bank and one example platformer sprite. Plan the generation job.
Do not select the prettiest view. Choose the identity primer that best supports the requested gameplay camera and action readability.
Images are supplied in this exact order:
{image_order}

Inspect the identity images yourself. Describe the character identity, hair, clothing, colors, art style and useful perspective from those images.
The identity view bank is the source of truth for character identity, clothing, hair, colors and style.
Describe the character's lower body in separate parts: waist garment, visible skin, legwear and footwear. Do not collapse socks, stockings or boots into generic pants when the images show separate garments.
The example sprite is only the source of truth for action pose, silhouette intent and gameplay readability.
The example sprite may depict a different character. Never use the example sprite for identity, hair, outfit, colors or style.
Inspect the example sprite yourself. Describe the body pose, hand/arm/leg placement, silhouette, action phase and platformer readability from that image.
The extracted non-pose controls are clean abstract geometry templates rendered from the foreground silhouette. They do not contain sprite colors, sprite texture or internal sprite linework.
Platformer side-view animation may cheat toward the camera when that improves readability. Do not require an exact mathematical 90-degree profile unless the request explicitly says so.
This is a full-body gameplay keyframe. Keep the whole character visible, including both feet; never plan a waist-up, bust, portrait or upper-body crop.

Request:
- character: {spec.character.id}
- action: {spec.request.action}
- phase: {spec.request.phase}
- direction: {spec.request.direction}
- camera: {spec.request.camera}
- description: {spec.request.description}
- output canvas requested by example extraction: {spec.example.width}x{spec.example.height}
- number of generated candidates: {spec.generation.seed_count}

Available identity-primer views:
{view_lines}

Return JSON only. The JSON must use this exact structure and data types; replace the descriptive string values with what you see in the supplied images:
{json.dumps(contract, indent=2)}

Contract details:
- identity_details: structured visual slots from the identity images. Fill every slot with concrete image evidence. Use "no visible ..." only when that garment class is genuinely absent.
- identity_description: visual caption from the identity images, including subject type, hair, clothing, colors and style.
- pose_description: visual caption from the example sprite, including action phase, arms, hands, legs, feet and silhouette.
- platformer_camera_description: visual camera/readability interpretation from the example sprite and selected identity view.
- identity_primer: object with view and exact path from the available views.
- prompt: object with clip, t5, optional negative and required true_cfg_scale. Build both prompts from what you see.
- canvas: object with width={spec.example.width}, height={spec.example.height}, reference_max_area as an integer between 200000 and 524288, max_sequence_length=128.
- sampling: object with steps as an integer between 20 and 32 and guidance_scale as a number between 2.0 and 3.2.
- controls: non-empty array. Each control has name, type, source, scale, start, end and optional residual_mask_source.
- control.source is the extracted asset source and must be exactly "example_pose", "example_canny_lineart" or "example_softedge".
- control.type is the ControlNet condition type and must be exactly "pose", "canny" or "softedge"; never return "source" as a type.
- source "example_pose" must use type "pose".
- source "example_canny_lineart" must use type "canny" and is a clean foreground-geometry outline, not sprite lineart.
- source "example_softedge" must use type "softedge" and is a clean foreground-geometry edge map, not sprite texture.
- residual_mask_source may be "example_boundary_mask", "example_full_silhouette_mask" or "example_arm_hand_mask" when a non-pose control should be spatially limited.
- scoring: object with top_k, priorities and checks for selecting the best generated candidates.
- scoring.top_k must be an integer between 1 and {spec.generation.seed_count}; prefer 3 unless the generation count is smaller.
- scoring.priorities and scoring.checks must be concrete visual criteria for this request. Do not copy schema-descriptive phrases.
- polish: object with profile, max_regions, strength_offsets and seed_offsets for local model-planned polish.
- polish.profile must be exactly "kontext-inpaint-local".
- polish.strength_offsets must be numeric offsets such as [-0.06, 0.0, 0.06].
- polish.seed_offsets must be integer offsets such as [0, 1], not floats.
- rationale must be an array of concrete strings.

Keep prompts specific to the approved identity primer and the example action. Use separate CLIP and T5 prompt text. Do not mention internal filenames in prompts.
Build prompt.clip and prompt.t5 from every identity_details slot plus the example action. Do not omit the waist garment, legwear or footwear.
Build the prompt text from what you see in the supplied identity images and example sprite. Do not reuse generic placeholder identity text.
Choose control strengths from the image evidence. Strong pose control is useful when limb placement matters. Clean softedge or canny geometry control is useful when the source sprite silhouette must have structure authority. Do not request dense gray/source-image control for production keyframes. Do not blindly copy fixed numeric examples.
Never return placeholder strings such as "...".
prompt.true_cfg_scale is required.
Omit prompt.negative when true_cfg_scale is 1.0."""


def _planner_repair_prompt(raw_text: str, error: ValidationError) -> str:
    original = _json_object(raw_text)
    prompt = dict(original.get("prompt", {}))
    if "true_cfg_scale" not in prompt and "negative" not in prompt:
        prompt["true_cfg_scale"] = 1.0
    return f"""Repair this generated keyframe brief plan JSON so it satisfies the schema.

Do not reinterpret the images.
Do not change the visual decisions, identity details, prompt wording, controls, scoring, polish settings or rationale unless a validation error explicitly requires it.
Preserve every original key and value that is not named by the validation error.
Return JSON only.

Validation error:
{error}

Rules:
- prompt.true_cfg_scale is required. If prompt.negative is absent, set prompt.true_cfg_scale to 1.0.
- Copy prompt.clip and prompt.t5 exactly from the original JSON.
- Do not add prompt.negative unless the original JSON already had prompt.negative.
- controls[].type must be "pose", "canny" or "softedge".
- polish.profile must be "kontext-inpaint-local".
- polish.seed_offsets must contain integers.
- rationale must be an array of strings.

For this repair, the prompt object must be:
{json.dumps(prompt, indent=2)}

Original JSON:
{raw_text}"""


def _validate_plan_against_brief(
    plan: KeyframeBriefPlanSpec,
    spec: KeyframeBriefSpec,
    view_options: list[dict[str, Any]],
    plan_path: Path,
) -> None:
    expected_paths = {(option["view"], option["path"]) for option in view_options}
    if (plan.identity_primer.view, plan.identity_primer.path) not in expected_paths:
        raise KeyframeBriefError(f"Invalid generated brief plan {plan_path}: identity_primer must use an available view path")
    if plan.canvas.width != spec.example.width or plan.canvas.height != spec.example.height:
        raise KeyframeBriefError(f"Invalid generated brief plan {plan_path}: canvas must match the brief example size")
    if plan.scoring.top_k > spec.generation.seed_count:
        raise KeyframeBriefError(f"Invalid generated brief plan {plan_path}: scoring.top_k exceeds generated candidate count")


def _view_options(view_bank: CharacterViewBankSpec) -> list[dict[str, Any]]:
    ordered_names = ("front", "left_profile", "right_profile", "back")
    views = view_bank.views
    options = []
    for name in ordered_names:
        if name in views:
            entry = views[name]
            options.append(
                {
                    "view": name,
                    "path": entry.image.path,
                    "camera": entry.view.camera,
                    "pose": entry.view.pose,
                }
            )
    for name, entry in views.items():
        if name not in ordered_names:
            options.append(
                {
                    "view": name,
                    "path": entry.image.path,
                    "camera": entry.view.camera,
                    "pose": entry.view.pose,
                }
            )
    return options


def _planner_evidence_paths(view_options: list[dict[str, Any]], example_path: Path, output_dir: Path) -> PlannerEvidence:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    order_lines: list[str] = []
    for option in view_options:
        view_path = Path(option["path"])
        image_paths.append(view_path)
        order_lines.append(f"identity view {option['view']} full body: {view_path}")
        crops = _identity_detail_crops(view_path, output_dir, option["view"])
        for label, path in crops.items():
            image_paths.append(path)
            order_lines.append(f"identity view {option['view']} {label} detail crop: {path}")
    image_paths.append(example_path)
    order_lines.append(f"example sprite for action/pose only: {example_path}")
    return PlannerEvidence(image_paths=image_paths, order_lines=order_lines)


def _identity_detail_crops(path: Path, output_dir: Path, view_name: str) -> dict[str, Path]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        crops = {
            "torso_waist": rgb.crop((0, round(height * 0.22), width, round(height * 0.68))),
            "lower_body": rgb.crop((0, round(height * 0.42), width, height)),
        }
    outputs = {}
    for label, crop in crops.items():
        target = output_dir / f"{view_name}_{label}.png"
        crop.save(target)
        outputs[label] = target
    return outputs


def _view_option_line(option: dict[str, Any]) -> str:
    return f"- {option['view']}: {option['path']} ({option['camera']}, {option['pose']})."


def _json_object(raw_text: str) -> dict[str, Any]:
    try:
        return json_object_from_vlm_response(raw_text)
    except VlmJsonError as error:
        raise KeyframeBriefError(str(error)) from error
