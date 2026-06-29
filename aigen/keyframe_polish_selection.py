from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import ValidationError

from aigen.keyframe_image_ops import mask_overlay
from aigen.keyframe_polish_masks import PolishMaskPlan
from aigen.keyframe_polish_models import (
    KeyframePolishError,
    KeyframePolishJobSpec,
    KeyframePolishPlan,
    PolishRegionSelection,
)
from aigen.vlm_json import VlmJsonError, json_object_from_vlm_response


def save_polish_selection_evidence(
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
    reference_card = evidence_dir / "reference_detail_card.png"
    overlay = evidence_dir / "base_overlay.png"
    with Image.open(identity_primer) as image:
        image.convert("RGB").save(primer)
    mask_plan.reference_card.save(reference_card)
    mask_overlay(base, mask_plan.feather_mask).save(overlay)
    paths.extend((primer, reference_card, overlay))
    for candidate in candidates:
        crop = evidence_dir / f"{candidate['name']}_crop.png"
        with Image.open(candidate["path"]) as image:
            image.convert("RGB").crop(mask_plan.crop_box).save(crop)
        paths.append(crop)
    return paths


def polish_selection_prompt(
    spec: KeyframePolishJobSpec,
    plan: KeyframePolishPlan,
    mask_plan: PolishMaskPlan,
    candidates: list[dict[str, Any]],
) -> str:
    region = next(region for region in plan.regions if region.id == mask_plan.region_id)
    return f"""You are selecting the best local polish result for one region.

Images:
1. identity primer
2. local reference detail card for this region
3. base candidate with mask overlay
4+ candidate crops in this exact order:
{json.dumps([candidate["name"] for candidate in candidates])}

Region:
- id: {region.id}
- label: {region.label}
- operation: {region.operation}
- reason: {region.reason}
- must_not_change: {json.dumps(region.must_not_change, ensure_ascii=False)}
- acceptance_checks: {json.dumps(region.acceptance_checks, ensure_ascii=False)}

Choose the candidate that restores the target local detail while preserving identity and style. Do not choose a variant that changes pose or looks like a different character.
Judge only this local polish region. Do not rank the whole full-frame candidate.

Return JSON only:
- region_id: "{region.id}"
- best_variant: one candidate name from the list
- passes: boolean
- checks: target_detail_restored, identity_preserved, outside_mask_changed, pose_changed, style_match
- reason: one sentence

Job id: {spec.id}
"""


def parse_polish_region_selection(raw_text: str) -> PolishRegionSelection:
    data = _json_from_vlm_response(raw_text)
    try:
        return PolishRegionSelection.model_validate(data)
    except ValidationError as error:
        raise KeyframePolishError(f"Polish selector returned invalid JSON: {error}") from error


def validate_polish_region_selection(
    selection: PolishRegionSelection,
    region_id: str,
    candidates: set[str],
) -> None:
    if selection.region_id != region_id:
        raise KeyframePolishError(f"Polish selector returned region {selection.region_id}, expected {region_id}")
    if selection.best_variant not in candidates:
        raise KeyframePolishError(f"Polish selector chose unknown variant {selection.best_variant}")
    if not selection.passes:
        raise KeyframePolishError(f"Polish selector failed {region_id}: {selection.reason}")
    if selection.checks.outside_mask_changed or selection.checks.pose_changed:
        raise KeyframePolishError(f"Polish selector accepted unsafe variant for {region_id}: {selection.reason}")
    if not selection.checks.target_detail_restored:
        raise KeyframePolishError(f"Polish selector accepted unrestored detail for {region_id}: {selection.reason}")
    if not selection.checks.identity_preserved or not selection.checks.style_match:
        raise KeyframePolishError(f"Polish selector accepted identity/style drift for {region_id}: {selection.reason}")


def _json_from_vlm_response(raw_text: str) -> dict[str, Any]:
    try:
        return json_object_from_vlm_response(raw_text)
    except VlmJsonError as error:
        raise KeyframePolishError(str(error)) from error
