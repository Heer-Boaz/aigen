from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pydantic import ValidationError

from aigen.keyframe_polish_context import PolishContext
from aigen.keyframe_polish_models import (
    POLISH_OPERATION_PROFILES,
    KeyframePolishError,
    KeyframePolishJobSpec,
    KeyframePolishPlan,
)
from aigen.keyframe_segmentation import foreground_box_mask
from aigen.vlm_json import VlmJsonError, json_object_from_vlm_response


@dataclass(frozen=True)
class PolishPlannerEvidence:
    image_paths: list[Path]
    prompt_order: list[str]


def save_polish_planner_evidence(context: PolishContext, output_dir: Path) -> PolishPlannerEvidence:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    prompt_order = []
    identity = output_dir / "identity_primer.png"
    candidate = output_dir / "candidate.png"
    context.identity_primer.save(identity)
    context.base_image.save(candidate)
    paths.extend((identity, candidate))
    prompt_order.extend(("approved identity primer", f"selected structural candidate {context.base_path.name}"))
    comparison_grid = output_dir / "identity_candidate_comparison_grid.png"
    crop_grid = output_dir / "candidate_crop_grid.png"
    _identity_candidate_comparison_grid(context.identity_primer, context.base_image).save(comparison_grid)
    _candidate_crop_grid(context.base_image).save(crop_grid)
    paths.extend((comparison_grid, crop_grid))
    prompt_order.extend(("identity-primer-vs-candidate normalized crop comparison grid", "generic candidate crop grid"))
    for label, source in _condition_evidence_sources(context).items():
        target = output_dir / f"protected_condition_{label}.png"
        with Image.open(source) as image:
            image.convert("RGB").save(target)
        paths.append(target)
        prompt_order.append(f"protected upstream condition: {label}")
    for label, source in _score_evidence_sources(context).items():
        target = output_dir / f"score_evidence_{label}.png"
        with Image.open(source) as image:
            image.convert("RGB").save(target)
        paths.append(target)
        prompt_order.append(f"model score evidence: {label}")
    return PolishPlannerEvidence(image_paths=paths, prompt_order=prompt_order)


def polish_planner_prompt(spec: KeyframePolishJobSpec, context: PolishContext, evidence_order: list[str]) -> str:
    effective = context.result["effective_config"]
    acceptance = effective["acceptance"]["manual"]
    keyframe = effective["keyframe"]
    operations = list(POLISH_OPERATION_PROFILES)
    return f"""You are a production art polish planner for character keyframes.

You receive images in this order:
{_numbered_lines(evidence_order)}

The selected candidate already has the best structure. Pose, arm shape, silhouette and action readability are not polish targets here.
Plan only local polish regions where the candidate visibly diverges from the identity primer or loses character-specific detail, style, color, expression, clothing details, accessories, material identity, lineart quality, or small artifacts.
The protected upstream condition and score evidence explain what structure should stay fixed. Use it to avoid proposing pose, contour, silhouette or action changes. Do not infer polish targets from the source pose example.

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


def parse_polish_plan(raw_text: str) -> KeyframePolishPlan:
    data = _json_from_vlm_response(raw_text)
    try:
        return KeyframePolishPlan.model_validate(data)
    except ValidationError as error:
        raise KeyframePolishError(f"Polish planner returned invalid JSON: {error}") from error


def validate_polish_plan(spec: KeyframePolishJobSpec, plan: KeyframePolishPlan, image_size: tuple[int, int]) -> None:
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


def _condition_evidence_sources(context: PolishContext) -> dict[str, Path]:
    return {
        name: Path(asset["path"]).resolve()
        for name, asset in context.result["effective_config"]["assets"].items()
        if name in {"pose", "contour", "boundary_mask"}
    }


def _score_evidence_sources(context: PolishContext) -> dict[str, Path]:
    score_dir = context.base_dir / "score"
    if not score_dir.is_dir():
        return {}
    evidence = {}
    patterns = {
        "condition_diff": f"**/{context.candidate}__condition_diff.png",
        "foreground": f"**/{context.candidate}__foreground.png",
        "pose_match": f"**/{context.candidate}__pose_match.png",
    }
    for label, pattern in patterns.items():
        matches = sorted(score_dir.glob(pattern))
        if matches:
            evidence[label] = matches[0]
    return evidence


def _numbered_lines(items: list[str]) -> str:
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def _candidate_crop_grid(base: Image.Image) -> Image.Image:
    foreground = foreground_box_mask(np.asarray(base, dtype=np.uint8))
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
    identity_foreground = foreground_box_mask(np.asarray(identity_primer, dtype=np.uint8))
    candidate_foreground = foreground_box_mask(np.asarray(candidate, dtype=np.uint8))
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


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise KeyframePolishError("Polish base image contains no foreground subject")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _json_from_vlm_response(raw_text: str) -> dict[str, Any]:
    try:
        return json_object_from_vlm_response(raw_text)
    except VlmJsonError as error:
        raise KeyframePolishError(str(error)) from error
