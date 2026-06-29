from __future__ import annotations

from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from aigen.image_assets import image_asset_json
from aigen.keyframe_grounding import GroundedRegionBox, GroundingConfig, GroundingRequest, KeyframeRegionGrounder
from aigen.keyframe_image_ops import bbox_mask_array, expanded_aligned_box
from aigen.keyframe_polish_models import (
    POLISH_OPERATION_PROFILES,
    KeyframePolishError,
    KeyframePolishJobSpec,
    KeyframePolishPlan,
    PlannedPolishParameters,
    PlannedPolishRegion,
)
from aigen.keyframe_segmentation import SamForegroundSegmenter, SamSegmentationConfig
from aigen.prompt_tokens import count_kontext_prompt_tokens


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
    grounding: GroundedRegionBox
    prompt: str
    negative_prompt: str
    hard_mask: Image.Image
    feather_mask: Image.Image
    crop_box: tuple[int, int, int, int]
    reference_card: Image.Image
    parameters: BoundedPolishParameters


def build_polish_mask_plans(
    base: Image.Image,
    identity_primer: Image.Image,
    resolved: dict[str, Any],
    *,
    segmenter: Any | None = None,
    grounder: Any | None = None,
) -> list[PolishMaskPlan]:
    plan = KeyframePolishPlan.model_validate(resolved["polish_plan"])
    if not plan.regions:
        return []
    base_array = np.asarray(base.convert("RGB"), dtype=np.uint8)
    if grounder is None:
        grounder = KeyframeRegionGrounder(GroundingConfig())
    grounded_regions = grounder.ground_regions(
        base,
        [GroundingRequest(prompt=region.mask_prompt, prior_box=region.bbox) for region in plan.regions],
    )

    with ExitStack() as resources:
        if segmenter is None:
            segmenter = resources.enter_context(closing(SamForegroundSegmenter(SamSegmentationConfig())))
        hard_masks = segmenter.segment_image_boxes(
            base_array,
            [grounding.box for grounding in grounded_regions],
        )
        return [
            _build_region_mask(index, region, grounding, hard_array, base, identity_primer)
            for index, (region, grounding, hard_array) in enumerate(
                zip(plan.regions, grounded_regions, hard_masks, strict=True),
                start=1,
            )
        ]


def load_polish_mask_plans(
    base: Image.Image,
    identity_primer: Image.Image,
    mask_plan_manifest: list[dict[str, Any]],
) -> list[PolishMaskPlan]:
    plans = []
    for index, item in enumerate(mask_plan_manifest, start=1):
        with Image.open(item["hard_mask"]["path"]) as image:
            hard_mask = image.convert("L")
        with Image.open(item["feather_mask"]["path"]) as image:
            feather_mask = image.convert("L")
        crop_box = tuple(item["crop_box"])
        parameters = BoundedPolishParameters(**item["parameters"])
        plans.append(
            PolishMaskPlan(
                region_id=item["region_id"],
                index=index,
                label=item["label"],
                operation=item["operation"],
                grounding=GroundedRegionBox(
                    box=tuple(item["grounding"]["box"]),
                    label=item["grounding"]["label"],
                    score=item["grounding"]["score"],
                    source=item["grounding"]["source"],
                    prior_iou=item["grounding"]["prior_iou"],
                ),
                prompt=item["prompt"],
                negative_prompt=item["negative_prompt"],
                hard_mask=hard_mask,
                feather_mask=feather_mask,
                crop_box=crop_box,
                reference_card=reference_detail_card(identity_primer, base, crop_box),
                parameters=parameters,
            )
        )
    return plans


def bound_polish_parameters(operation: str, parameters: PlannedPolishParameters) -> BoundedPolishParameters:
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


def polish_region_variants(spec: KeyframePolishJobSpec, mask_plan: PolishMaskPlan) -> list[dict[str, Any]]:
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


def polish_region_token_metadata(model_path: str, plan: KeyframePolishPlan) -> dict[str, dict[str, int]]:
    metadata = {}
    for region in plan.regions:
        bounded = bound_polish_parameters(region.operation, region.parameters)
        tokens = count_kontext_prompt_tokens(model_path, region.label, region.prompt)
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


def planned_polish_outputs(spec: KeyframePolishJobSpec, plan: KeyframePolishPlan, output_dir: Path) -> list[dict[str, Any]]:
    files = []
    for index, region in enumerate(plan.regions, start=1):
        bounded = bound_polish_parameters(region.operation, region.parameters)
        seen: set[tuple[int, float]] = set()
        for seed_offset in spec.micro_sweep.seed_offsets:
            for strength_offset in spec.micro_sweep.strength_offsets:
                strength = _clamp_float(
                    bounded.strength + strength_offset,
                    POLISH_OPERATION_PROFILES[region.operation]["strength_min"],
                    POLISH_OPERATION_PROFILES[region.operation]["strength_max"],
                )
                seed = 1000 + (index * 100) + seed_offset
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


def reference_detail_card(identity_primer: Image.Image, base: Image.Image, crop_box: tuple[int, int, int, int]) -> Image.Image:
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


def mask_plan_json(mask_plan: PolishMaskPlan, output_dir: Path | None = None) -> dict[str, Any]:
    payload = {
        "region_id": mask_plan.region_id,
        "label": mask_plan.label,
        "operation": mask_plan.operation,
        "prompt": mask_plan.prompt,
        "negative_prompt": mask_plan.negative_prompt,
        "crop_box": list(mask_plan.crop_box),
        "grounding": mask_plan.grounding.to_json(),
        "segmentation": {
            "method": f"{mask_plan.grounding.source}-box-to-sam-mask",
        },
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
    if output_dir is not None:
        region_dir = mask_artifact_dir(output_dir, mask_plan.region_id)
        payload["hard_mask"] = image_asset_json(region_dir / "hard.png")
        payload["feather_mask"] = image_asset_json(region_dir / "feather.png")
    return payload


def mask_artifact_dir(output_dir: Path, region_id: str) -> Path:
    return output_dir / "masks" / region_id


def _build_region_mask(
    index: int,
    region: PlannedPolishRegion,
    grounding: GroundedRegionBox,
    hard_array: np.ndarray,
    base: Image.Image,
    identity_primer: Image.Image,
) -> PolishMaskPlan:
    width, height = base.size
    if hard_array.shape != (height, width):
        raise KeyframePolishError(
            f"Segmentation mask for polish region {region.id} has shape {hard_array.shape}, expected {(height, width)}"
        )
    hard_array &= bbox_mask_array(base.size, grounding.box)
    if not hard_array.any():
        raise KeyframePolishError(f"Segmentation produced no usable mask for polish region {region.id}")
    parameters = bound_polish_parameters(region.operation, region.parameters)
    hard_array = ndimage.binary_dilation(hard_array, iterations=max(1, round(width * 0.006)))
    hard = Image.fromarray((hard_array.astype(np.uint8) * 255), mode="L")
    feather = hard.filter(ImageFilter.GaussianBlur(radius=parameters.feather_px))
    crop_box = expanded_aligned_box(hard_array, parameters.crop_padding_px, width, height)
    return PolishMaskPlan(
        region_id=region.id,
        index=index,
        label=region.label,
        operation=region.operation,
        grounding=grounding,
        prompt=region.prompt,
        negative_prompt=region.negative_prompt,
        hard_mask=hard,
        feather_mask=feather,
        crop_box=crop_box,
        reference_card=reference_detail_card(identity_primer, base, crop_box),
        parameters=parameters,
    )


def _clamp_float(value: float, lower: float | int, upper: float | int) -> float:
    return float(max(float(lower), min(float(upper), value)))


def _clamp_int(value: int, lower: float | int, upper: float | int) -> int:
    return int(max(int(lower), min(int(upper), value)))
