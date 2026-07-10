from __future__ import annotations

import re
import shutil
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from aigen.character_conditioning_planner import CharacterConditioningPlanner
from aigen.image_assets import image_asset_json
from aigen.keyframe_grounding import Florence2RegionGrounder, GroundingConfig, KeyframeGroundingError
from aigen.keyframe_image_ops import mask_overlay, save_contact_sheet
from aigen.keyframe_segmentation import Sam2RegionSegmenter, Sam2SegmentationConfig
from aigen.manifest_io import resolve_existing_path, write_json
from aigen.progress import StatusReporter


CHARACTER_REGION_PLAN_KIND = "character-region-plan"
CHARACTER_INPAINT_MASK_EXPANSION = 4
REGION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class CharacterRegionPlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class CharacterRegionRequest:
    name: str
    prompt: str


def parse_character_region_args(region_args: Sequence[str]) -> tuple[CharacterRegionRequest, ...]:
    if not region_args:
        raise CharacterRegionPlanError("characters region-plan requires at least one --region NAME=TEXT")
    requests: list[CharacterRegionRequest] = []
    seen: set[str] = set()
    for raw_region in region_args:
        name, separator, raw_prompt = raw_region.partition("=")
        name = name.strip()
        prompt = " ".join(raw_prompt.split())
        if separator != "=" or not name or not prompt:
            raise CharacterRegionPlanError(f"Region must be NAME=TEXT: {raw_region}")
        _validate_region_name(name)
        if name in seen:
            raise CharacterRegionPlanError(f"Duplicate region name: {name}")
        seen.add(name)
        requests.append(CharacterRegionRequest(name=name, prompt=prompt))
    return tuple(requests)


def plan_character_regions(
    *,
    image_path: Path,
    regions: Sequence[CharacterRegionRequest],
    output_dir: Path,
    overwrite: bool,
    grounding_config: GroundingConfig,
    segmentation_config: Sam2SegmentationConfig,
    progress: StatusReporter,
) -> dict[str, Any]:
    if not regions:
        raise CharacterRegionPlanError("characters region-plan requires at least one region request")
    conditioning_plan = CharacterConditioningPlanner().plan(
        route_kind="local_repair_or_inpaint",
        available_modes=("region_mask",),
    )
    image_path = resolve_existing_path(image_path.as_posix(), Path.cwd())
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise CharacterRegionPlanError(f"Output exists and overwrite=false: {output_dir.as_posix()}")
        shutil.rmtree(output_dir)
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    progress.phase("load character region image")
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    image_array = np.asarray(image, dtype=np.uint8)

    progress.phase("ground character regions with Florence-2")
    with closing(Florence2RegionGrounder(grounding_config)) as grounder:
        grounded_regions = []
        for region in regions:
            boxes = grounder.ground_boxes(image, region.prompt)
            if not boxes:
                raise KeyframeGroundingError(f"Florence-2 produced no region for '{region.prompt}'")
            grounded_regions.append(boxes[0])

    progress.phase("segment character regions with SAM2")
    with closing(Sam2RegionSegmenter(segmentation_config)) as segmenter:
        masks = segmenter.segment_image_boxes(image_array, [region.box for region in grounded_regions])

    planned_regions = []
    sheet_entries = []
    for region, grounding, mask_array in zip(regions, grounded_regions, masks, strict=True):
        if mask_array.shape != image_array.shape[:2]:
            raise CharacterRegionPlanError(
                f"SAM2 mask for region {region.name} has shape {mask_array.shape}, expected {image_array.shape[:2]}"
            )
        if not mask_array.any():
            raise CharacterRegionPlanError(f"SAM2 produced no usable mask for region {region.name}")
        mask_array = ndimage.binary_dilation(
            mask_array,
            iterations=CHARACTER_INPAINT_MASK_EXPANSION,
        )
        mask_path = masks_dir / f"{region.name}.png"
        overlay_path = overlays_dir / f"{region.name}.png"
        mask_image = Image.fromarray(mask_array.astype(np.uint8) * 255, mode="L")
        mask_image.save(mask_path)
        mask_overlay(image, mask_image).save(overlay_path)
        sheet_entries.append({"name": region.name, "path": overlay_path.as_posix()})
        planned_regions.append(
            {
                "name": region.name,
                "prompt": region.prompt,
                "grounding": grounding.to_json(),
                "segmentation": {
                    "method": f"{grounding.source}-box-to-sam2-mask",
                    "model": segmentation_config.model.as_posix(),
                    "inpaint_mask_expansion": CHARACTER_INPAINT_MASK_EXPANSION,
                    "mask": image_asset_json(mask_path),
                    "overlay": image_asset_json(overlay_path),
                },
            }
        )
        progress.step(f"planned region {region.name}")

    debug_sheet = output_dir / "debug_sheet.png"
    save_contact_sheet(sheet_entries, debug_sheet, thumb_width=320, max_label_chars=32)
    payload = {
        "status": "completed",
        "kind": CHARACTER_REGION_PLAN_KIND,
        "image": image_asset_json(image_path),
        "conditioning": conditioning_plan.model_dump(mode="json"),
        "regions": planned_regions,
        "models": {
            "florence2": grounding_config.florence_model.as_posix(),
            "sam2": segmentation_config.model.as_posix(),
        },
        "output": {
            "directory": output_dir.as_posix(),
            "masks": masks_dir.as_posix(),
            "overlays": overlays_dir.as_posix(),
            "debug_sheet": debug_sheet.as_posix(),
            "result": (output_dir / "result.json").as_posix(),
        },
    }
    write_json(output_dir / "result.json", payload)
    return payload


def _validate_region_name(name: str) -> None:
    if REGION_NAME_PATTERN.fullmatch(name) is None:
        raise CharacterRegionPlanError(f"Region name must be file-safe ASCII: {name}")
