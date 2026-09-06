from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

from aigen.character_reference_models import CharacterReferenceError
from aigen.character_reference_pack import load_character_reference_pack
from aigen.generation.qwen_image_edit_identity import (
    DEFAULT_QWEN_IDENTITY_PROFILE, QwenImageEditProfile,
)
from aigen.character_edit import run_character_edit
from aigen.character_edit_audit import AuditContextImage, CharacterAuditGroup
from aigen.generation.image_edit_batch import ImageEditBatchCase, ImageEditBatchLora, ImageEditBatchRequest, ImageEditMask
from aigen.generation.qwen_image_edit_masked import qwen_masked_canvas_size
from aigen.image_edit_defaults import QWEN_2511_SAMPLER, QWEN_2511_DEFAULT_SCHEDULER
from aigen.image_io import open_image
from aigen.image_assets import image_asset_json
from aigen.manifest_io import read_json, resolve_existing_path, write_json, sha256_file
from aigen.progress import StatusReporter
from aigen.vlm_qwen import QwenVlmConfig


QWEN_CHARACTER_REFINE_KIND = "qwen-character-refine-result"
QWEN_CHARACTER_REFINE_PLAN_KIND = "qwen-character-refine-plan"
QWEN_CHARACTER_REFINE_ROUTE = "local_repair_or_inpaint"
QWEN_CHARACTER_REFINE_OUTPUT_MODE = "masked_refine_candidates"


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
    pack_path: Path | None,
    source_image_path: Path,
    mask_path: Path | None,
    region_plan_path: Path | None,
    region_name: str | None,
    instruction: str,
    candidates: int,
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
            progress=progress,
        ).manifest,
    }


def run_qwen_character_refine(
    *,
    pack_path: Path | None,
    source_image_path: Path,
    mask_path: Path | None,
    region_plan_path: Path | None,
    region_name: str | None,
    instruction: str,
    output_dir: Path,
    profile: QwenImageEditProfile,
    max_side: int | None,
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
    progress: StatusReporter,
    max_iterations: int = 2,
    audit_config: QwenVlmConfig | None = None,
) -> dict[str, Any]:
    if profile.name != DEFAULT_QWEN_IDENTITY_PROFILE:
        raise QwenCharacterRefineError("character masked edits require the explicit Qwen-2511 Lightning profile")
    if padding_mask_crop is not None or nunchaku_blocks_on_gpu is not None:
        raise QwenCharacterRefineError("native Qwen-2511 regional edits use the source canvas and LightX2V block offload")
    planned = _build_qwen_character_refine_plan(
        pack_path=pack_path,
        source_image_path=source_image_path,
        mask_path=mask_path,
        region_plan_path=region_plan_path,
        region_name=region_name,
        instruction=instruction,
        candidates=candidates,
        progress=progress,
    )
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise QwenCharacterRefineError(f"Output exists and overwrite=false: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    refine_plan_path = output_dir / "refine_plan.json"
    write_json(refine_plan_path, planned.manifest)
    result = run_character_masked_edit(
        source_image=planned.source_image, mask_image=planned.mask_image,
        references=tuple(planned.references.values()), instruction=planned.prompt,
        output_dir=output_dir,
        max_side=max_side,
        steps=steps,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale,
        strength=strength,
        seed=seed,
        max_sequence_length=max_sequence_length,
        candidates=candidates,
        max_iterations=max_iterations, audit_config=audit_config,
        progress=progress,
    )
    result["refine_plan"] = refine_plan_path.as_posix()
    write_json(output_dir / "result.json", result)
    return result


def run_character_masked_edit(
    *, source_image: Path, mask_image: Path, references: tuple[Path, ...], instruction: str,
    output_dir: Path, progress: StatusReporter, strength: float = 1.0, candidates: int = 2,
    seed: int = 0, max_side: int | None = None, steps: int | None = None,
    true_cfg_scale: float | None = None, guidance_scale: float | None = None,
    max_sequence_length: int = 512, max_iterations: int = 2, audit_config: QwenVlmConfig | None = None,
    loras: tuple[ImageEditBatchLora, ...] = (), sampler: str = QWEN_2511_SAMPLER,
    scheduler: str = QWEN_2511_DEFAULT_SCHEDULER,
) -> dict[str, Any]:
    if candidates < 1:
        raise QwenCharacterRefineError("regional edits require at least one candidate")
    source_image, mask_image = source_image.expanduser().resolve(), mask_image.expanduser().resolve()
    with open_image(source_image) as source:
        width, height = qwen_masked_canvas_size(source.size, max_side)
    prompt = " ".join(instruction.split())
    cases = tuple(ImageEditBatchCase(
        id=f"candidate-{index}", prompt=prompt, image_paths=(source_image, *references),
        width=width, height=height, seed=seed + index, output_path=output_dir / f"candidate-{index}.png",
        mask=ImageEditMask(source_image=source_image, mask_image=mask_image, strength=strength),
    ) for index in range(candidates))
    group = CharacterAuditGroup(
        id="regional-edit", instruction=prompt, route=QWEN_CHARACTER_REFINE_ROUTE,
        references=(source_image, *references),
        context_images=(AuditContextImage("repaint mask: white is editable; black is preserved", mask_image),),
        candidate_ids=tuple(case.id for case in cases),
    )
    result = run_character_edit(
        ImageEditBatchRequest(backend="qwen-image-edit-2511-lightning", cases=cases, loras=loras,
                             steps=steps, guidance=true_cfg_scale, guidance_scale=guidance_scale,
                             max_sequence_length=max_sequence_length, sampler=sampler, scheduler=scheduler),
        audit_groups=(group,), output_dir=output_dir / "edit", progress=progress,
        max_iterations=max_iterations, audit_config=audit_config,
    ).to_json()
    result["kind"] = QWEN_CHARACTER_REFINE_KIND
    write_json(output_dir / "result.json", result)
    return result


def _build_qwen_character_refine_plan(
    *,
    pack_path: Path | None,
    source_image_path: Path,
    mask_path: Path | None,
    region_plan_path: Path | None,
    region_name: str | None,
    instruction: str,
    candidates: int,
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
    progress.phase("load qwen character refine reference pack")
    try:
        context = load_character_reference_pack(pack_path) if pack_path is not None else None
    except CharacterReferenceError as error:
        raise QwenCharacterRefineError(str(error)) from error
    source_image = resolve_existing_path(source_image_path.as_posix(), Path.cwd())
    if region_plan_path is not None:
        plan_source = read_json(region_plan_path, label="character region plan")["image"]
        if plan_source["sha256"] != sha256_file(source_image):
            raise QwenCharacterRefineError("region plan was produced from a different source image")
    references = dict(context.references) if context is not None else {}
    manifest = _refine_manifest(
        pack_path=context.path if context is not None else None,
        character_id=context.spec.character_id if context is not None else None,
        source_image=source_image,
        mask_image=mask_image,
        mask_source=mask_source,
        references=references,
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
            if sha256_file(mask) != region["segmentation"]["mask"]["sha256"]:
                raise QwenCharacterRefineError("region-plan mask contents changed since the selection was recorded")
            return mask, {
                "type": "region-plan",
                "region_plan": region_plan.as_posix(),
                "region": region,
                "conditioning_tools": payload["conditioning"]["planned_tools"],
            }
    raise QwenCharacterRefineError(f"Region plan {region_plan.as_posix()} has no region named {region_name}")


def _refine_manifest(
    *,
    pack_path: Path | None,
    character_id: str | None,
    source_image: Path,
    mask_image: Path,
    mask_source: dict[str, Any],
    references: dict[str, Path],
    instruction: str,
    candidates: int,
) -> dict[str, Any]:
    return {
        "kind": QWEN_CHARACTER_REFINE_PLAN_KIND,
        **({"character_id": character_id, "reference_pack": pack_path.as_posix()} if pack_path is not None else {}),
        "source_instruction": instruction,
        "prompt_source": "user_instruction",
        "route_kind": QWEN_CHARACTER_REFINE_ROUTE,
        "output_mode": QWEN_CHARACTER_REFINE_OUTPUT_MODE,
        "conditioning_modes": ["region_mask"],
        "conditioning_tools": mask_source["conditioning_tools"],
        "refs_used": list(references),
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
