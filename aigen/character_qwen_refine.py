from __future__ import annotations

from contextlib import closing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.character_qwen_edit import load_qwen_character_reference_context
from aigen.character_reference_models import CharacterReferenceError, CharacterReferencePackSpec
from aigen.character_reference_selector import ReferenceSelection, select_reference_subset
from aigen.generation.qwen_image_edit_identity import (
    QWEN_IMAGE_EDIT_MAX_INPUT_IMAGES,
    QwenImageEditIdentityProfile,
    run_qwen_image_edit_inpaint_candidates,
)
from aigen.image_assets import image_asset_json
from aigen.manifest_io import read_json, resolve_existing_path, write_json
from aigen.progress import StatusReporter
from aigen.vlm_qwen import QwenVlm, QwenVlmConfig


QWEN_CHARACTER_REFINE_KIND = "qwen-character-refine-result"
QWEN_CHARACTER_REFINE_PLAN_KIND = "qwen-character-refine-plan"
QWEN_CHARACTER_REFINE_ROUTE = "local_repair_or_inpaint"
QWEN_CHARACTER_REFINE_OUTPUT_MODE = "masked_refine_candidates"
QWEN_CHARACTER_REFINE_MAX_PACK_REFS = QWEN_IMAGE_EDIT_MAX_INPUT_IMAGES - 1


class QwenCharacterRefineError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlannedQwenCharacterRefine:
    source_image: Path
    mask_image: Path
    references: dict[str, Path]
    prompt: str
    manifest: dict[str, Any]


def plan_qwen_character_refine(
    *,
    pack_path: Path,
    source_image_path: Path,
    mask_path: Path | None,
    region_plan_path: Path | None,
    region_name: str | None,
    instruction: str,
    candidates: int,
    vlm_config: QwenVlmConfig,
    progress: StatusReporter,
) -> dict[str, Any]:
    return {
        "status": "planned",
        **_build_qwen_character_refine_plan(
            pack_path=pack_path,
            source_image_path=source_image_path,
            mask_path=mask_path,
            region_plan_path=region_plan_path,
            region_name=region_name,
            instruction=instruction,
            candidates=candidates,
            vlm_config=vlm_config,
            progress=progress,
        ).manifest,
    }


def run_qwen_character_refine(
    *,
    pack_path: Path,
    source_image_path: Path,
    mask_path: Path | None,
    region_plan_path: Path | None,
    region_name: str | None,
    instruction: str,
    output_dir: Path,
    profile: QwenImageEditIdentityProfile,
    max_side: int,
    steps: int | None,
    true_cfg_scale: float | None,
    guidance_scale: float | None,
    strength: float,
    padding_mask_crop: int | None,
    seed: int,
    max_sequence_length: int,
    candidates: int,
    overwrite: bool,
    nunchaku_blocks_on_gpu: int | None,
    vlm_config: QwenVlmConfig,
    progress: StatusReporter,
) -> dict[str, Any]:
    planned = _build_qwen_character_refine_plan(
        pack_path=pack_path,
        source_image_path=source_image_path,
        mask_path=mask_path,
        region_plan_path=region_plan_path,
        region_name=region_name,
        instruction=instruction,
        candidates=candidates,
        vlm_config=vlm_config,
        progress=progress,
    )
    output_dir = output_dir.resolve()
    result = run_qwen_image_edit_inpaint_candidates(
        source_image=planned.source_image,
        mask_image=planned.mask_image,
        reference_images=planned.references,
        output_dir=output_dir,
        profile=profile,
        prompt=planned.prompt,
        max_side=max_side,
        steps=steps,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale,
        strength=strength,
        padding_mask_crop=padding_mask_crop,
        seed=seed,
        max_sequence_length=max_sequence_length,
        candidates=candidates,
        overwrite=overwrite,
        nunchaku_blocks_on_gpu=nunchaku_blocks_on_gpu,
        result_kind=QWEN_CHARACTER_REFINE_KIND,
        manifest_context=planned.manifest,
        progress=progress,
    )
    refine_plan_path = output_dir / "refine_plan.json"
    write_json(refine_plan_path, {"status": "completed", **planned.manifest})
    result["output"]["refine_plan"] = refine_plan_path.as_posix()
    write_json(output_dir / "result.json", result)
    return result


def _build_qwen_character_refine_plan(
    *,
    pack_path: Path,
    source_image_path: Path,
    mask_path: Path | None,
    region_plan_path: Path | None,
    region_name: str | None,
    instruction: str,
    candidates: int,
    vlm_config: QwenVlmConfig,
    progress: StatusReporter,
) -> PlannedQwenCharacterRefine:
    prompt = " ".join(instruction.split())
    if not prompt:
        raise QwenCharacterRefineError("qwen-edit-refine requires a non-empty --instruction")
    if candidates < 1:
        raise QwenCharacterRefineError("candidates must be at least 1")
    mask_image, mask_source = _resolve_refine_mask(
        mask_path=mask_path,
        region_plan_path=region_plan_path,
        region_name=region_name,
    )
    context = load_qwen_character_reference_context(
        pack_path=pack_path,
        progress=progress,
        phase="load qwen character refine reference pack",
    )
    source_image = resolve_existing_path(source_image_path.as_posix(), Path.cwd())
    selection = _select_refine_references(
        pack=context.pack,
        reference_paths=context.reference_paths,
        instruction=prompt,
        pack_path=context.pack_path,
        vlm_config=vlm_config,
        progress=progress,
    )
    references = {
        reference_name: context.reference_paths[reference_name]
        for reference_name in selection.selected_refs
    }
    manifest = _refine_manifest(
        pack_path=context.pack_path,
        character_id=context.pack.character_id,
        source_image=source_image,
        mask_image=mask_image,
        mask_source=mask_source,
        references=references,
        selection=selection,
        instruction=prompt,
        candidates=candidates,
    )
    return PlannedQwenCharacterRefine(
        source_image=source_image,
        mask_image=mask_image,
        references=references,
        prompt=prompt,
        manifest=manifest,
    )


def _select_refine_references(
    *,
    pack: CharacterReferencePackSpec,
    reference_paths: Mapping[str, Path],
    instruction: str,
    pack_path: Path,
    vlm_config: QwenVlmConfig,
    progress: StatusReporter,
) -> ReferenceSelection:
    needs_selection = len(pack.references) > QWEN_CHARACTER_REFINE_MAX_PACK_REFS
    progress.phase("select qwen character refine references")
    try:
        if needs_selection:
            with closing(QwenVlm(vlm_config)) as runner:
                return select_reference_subset(
                    runner=runner,
                    pack=pack,
                    reference_paths=reference_paths,
                    instruction=instruction,
                    route_kind=QWEN_CHARACTER_REFINE_ROUTE,
                    output_mode=QWEN_CHARACTER_REFINE_OUTPUT_MODE,
                    budget=QWEN_CHARACTER_REFINE_MAX_PACK_REFS,
                    path_label=f"{pack_path.as_posix()}#refine",
                )
        return select_reference_subset(
            runner=None,
            pack=pack,
            reference_paths=reference_paths,
            instruction=instruction,
            route_kind=QWEN_CHARACTER_REFINE_ROUTE,
            output_mode=QWEN_CHARACTER_REFINE_OUTPUT_MODE,
            budget=QWEN_CHARACTER_REFINE_MAX_PACK_REFS,
            path_label=f"{pack_path.as_posix()}#refine",
        )
    except CharacterReferenceError as error:
        raise QwenCharacterRefineError(str(error)) from error


def _resolve_refine_mask(
    *,
    mask_path: Path | None,
    region_plan_path: Path | None,
    region_name: str | None,
) -> tuple[Path, dict[str, Any]]:
    if mask_path is not None and region_plan_path is not None:
        raise QwenCharacterRefineError("Use either --mask or --region-plan, not both")
    if mask_path is not None:
        if region_name is not None:
            raise QwenCharacterRefineError("--region is only valid with --region-plan")
        mask = resolve_existing_path(mask_path.as_posix(), Path.cwd())
        return mask, {
            "type": "mask",
            "mask": mask.as_posix(),
            "conditioning_tools": [],
        }
    if region_plan_path is None:
        raise QwenCharacterRefineError("qwen-edit-refine requires --mask or --region-plan")
    if not region_name:
        raise QwenCharacterRefineError("qwen-edit-refine requires --region with --region-plan")
    region_plan = resolve_existing_path(region_plan_path.as_posix(), Path.cwd())
    payload = read_json(region_plan, label="character region plan")
    if payload.get("status") != "completed" or payload.get("kind") != "character-region-plan":
        raise QwenCharacterRefineError(f"Invalid character region plan: {region_plan.as_posix()}")
    for region in payload["regions"]:
        if region["name"] == region_name:
            mask = resolve_existing_path(region["segmentation"]["mask"]["path"], region_plan.parent)
            return mask, {
                "type": "region-plan",
                "region_plan": region_plan.as_posix(),
                "region": region,
                "conditioning_tools": payload["conditioning"]["planned_tools"],
            }
    raise QwenCharacterRefineError(f"Region plan {region_plan.as_posix()} has no region named {region_name}")


def _refine_manifest(
    *,
    pack_path: Path,
    character_id: str,
    source_image: Path,
    mask_image: Path,
    mask_source: dict[str, Any],
    references: dict[str, Path],
    selection: ReferenceSelection,
    instruction: str,
    candidates: int,
) -> dict[str, Any]:
    return {
        "kind": QWEN_CHARACTER_REFINE_PLAN_KIND,
        "character_id": character_id,
        "reference_pack": pack_path.as_posix(),
        "source_instruction": instruction,
        "prompt_source": "user_instruction",
        "route_kind": QWEN_CHARACTER_REFINE_ROUTE,
        "output_mode": QWEN_CHARACTER_REFINE_OUTPUT_MODE,
        "conditioning_modes": ["region_mask"],
        "conditioning_tools": mask_source["conditioning_tools"],
        "reference_selector": selection.selector,
        "selected_ref_indices": list(selection.selected_ref_indices),
        "refs_used": list(selection.selected_refs),
        "source_image": image_asset_json(source_image),
        "mask_image": image_asset_json(mask_image),
        "mask_source": mask_source,
        "references": [
            {
                "name": reference_name,
                "image": image_asset_json(reference_path),
            }
            for reference_name, reference_path in references.items()
        ],
        "candidates": candidates,
        "prompt": instruction,
    }
