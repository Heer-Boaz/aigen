from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.character_qwen_edit import load_qwen_character_reference_context
from aigen.character_reference_models import CharacterIdentityProfileSpec
from aigen.generation.qwen_image_edit_identity import (
    QwenImageEditIdentityProfile,
    QwenImageEditInpaintReference,
    run_qwen_image_edit_inpaint_candidates,
)
from aigen.image_assets import image_asset_json
from aigen.manifest_io import read_json, resolve_existing_path, write_json
from aigen.progress import StatusReporter


QWEN_CHARACTER_REFINE_KIND = "qwen-character-refine-result"
QWEN_CHARACTER_REFINE_PLAN_KIND = "qwen-character-refine-plan"
QWEN_CHARACTER_REFINE_MAX_ROUTED_REFS = 4
QWEN_CHARACTER_REFINE_ROUTED_ROLES = ("front", "portrait", "side", "back")
QWEN_CHARACTER_REFINE_ROLE_PURPOSES = {
    "front": "front-view outfit, color, and full-body proportion support",
    "portrait": "face identity and upper-body detail support",
    "side": "side silhouette, body depth, and profile support",
    "back": "back-view outfit, hair, and rear silhouette support",
}


class QwenCharacterRefineError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlannedQwenCharacterRefine:
    source_image: Path
    mask_image: Path
    references: tuple[QwenImageEditInpaintReference, ...]
    prompt: str
    manifest: dict[str, Any]


def plan_qwen_character_refine(
    *,
    pack_path: Path,
    identity_profile_path: Path | None,
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
            identity_profile_path=identity_profile_path,
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
    pack_path: Path,
    identity_profile_path: Path | None,
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
    progress: StatusReporter,
) -> dict[str, Any]:
    planned = _build_qwen_character_refine_plan(
        pack_path=pack_path,
        identity_profile_path=identity_profile_path,
        source_image_path=source_image_path,
        mask_path=mask_path,
        region_plan_path=region_plan_path,
        region_name=region_name,
        instruction=instruction,
        candidates=candidates,
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
    identity_profile_path: Path | None,
    source_image_path: Path,
    mask_path: Path | None,
    region_plan_path: Path | None,
    region_name: str | None,
    instruction: str,
    candidates: int,
    progress: StatusReporter,
) -> PlannedQwenCharacterRefine:
    normalized_instruction_text = " ".join(instruction.split())
    if not normalized_instruction_text:
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
        identity_profile_path=identity_profile_path,
        progress=progress,
        phase="load qwen character refine reference pack",
    )
    source_image = resolve_existing_path(source_image_path.as_posix(), Path.cwd())
    references = _routed_refine_references(
        role_reference_ids=context.role_reference_ids,
        reference_paths=context.reference_paths,
    )
    normalized_instruction = _normalized_refine_instruction(
        source_image=source_image,
        mask_image=mask_image,
        mask_source=mask_source,
        references=references,
        reference_roles_by_id=context.identity_profile.reference_roles,
        identity_profile=context.identity_profile,
        instruction=normalized_instruction_text,
    )
    prompt = _prompt_from_refine_instruction(normalized_instruction)
    manifest = _refine_manifest(
        pack_path=context.pack_path,
        identity_profile_path=context.identity_profile_path,
        character_id=context.pack.character_id,
        source_image=source_image,
        mask_image=mask_image,
        mask_source=mask_source,
        references=references,
        reference_roles_by_id=context.identity_profile.reference_roles,
        identity_profile=context.identity_profile,
        instruction=normalized_instruction_text,
        normalized_instruction=normalized_instruction,
        prompt=prompt,
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
        return mask, {"type": "mask", "mask": mask.as_posix()}
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
            raw_mask = region["segmentation"]["mask"]["path"]
            mask = resolve_existing_path(raw_mask, region_plan.parent)
            return mask, {
                "type": "region-plan",
                "region_plan": region_plan.as_posix(),
                "region": region,
            }
    raise QwenCharacterRefineError(f"Region plan {region_plan.as_posix()} has no region named {region_name}")


def _routed_refine_references(
    *,
    role_reference_ids: Mapping[str, str],
    reference_paths: Mapping[str, Path],
) -> tuple[QwenImageEditInpaintReference, ...]:
    references: list[QwenImageEditInpaintReference] = []
    for role in QWEN_CHARACTER_REFINE_ROUTED_ROLES:
        reference_id = role_reference_ids.get(role)
        if reference_id is None:
            continue
        references.append(
            QwenImageEditInpaintReference(
                name=reference_id,
                role=role,
                purpose=QWEN_CHARACTER_REFINE_ROLE_PURPOSES[role],
                path=reference_paths[reference_id],
            )
        )
        if len(references) == QWEN_CHARACTER_REFINE_MAX_ROUTED_REFS:
            break
    if not references:
        raise QwenCharacterRefineError("qwen-edit-refine needs at least one VLM-inferred non-body reference role")
    return tuple(references)


def _normalized_refine_instruction(
    *,
    source_image: Path,
    mask_image: Path,
    mask_source: Mapping[str, Any],
    references: Sequence[QwenImageEditInpaintReference],
    reference_roles_by_id: Mapping[str, str],
    identity_profile: CharacterIdentityProfileSpec,
    instruction: str,
) -> dict[str, Any]:
    return {
        "task": "identity_refine",
        "source_image": source_image.as_posix(),
        "mask_image": mask_image.as_posix(),
        "mask_source": dict(mask_source),
        "mask_semantics": {
            "white": "repainted",
            "black": "preserved",
        },
        "source_instruction": instruction,
        "refs_used": [reference.name for reference in references],
        "reference_roles": {
            reference.name: reference_roles_by_id[reference.name]
            for reference in references
        },
        "prompt_image_order": [
            {"input_index": 1, "source": "selected_image", "path": source_image.as_posix()},
            *[
                {
                    "input_index": index + 2,
                    "source": "reference_pack",
                    "reference_id": reference.name,
                    "role": reference.role,
                    "purpose": reference.purpose,
                    "path": reference.path.as_posix(),
                }
                for index, reference in enumerate(references)
            ],
        ],
        "identity_profile_used": True,
        "body_proportion_source": identity_profile.body_proportion_source,
        "body_proportion": identity_profile.body_proportion.model_dump(mode="json"),
        "optional_missing_refs": identity_profile.optional_missing_refs,
        "identity": dict(identity_profile.identity),
        "must_preserve": _identity_must_preserve(identity_profile),
        "avoid": list(dict.fromkeys(identity_profile.avoid)),
    }


def _prompt_from_refine_instruction(instruction: Mapping[str, Any]) -> str:
    prompt_contract = {
        "task": "identity_refine",
        "selected_image": {
            "input_index": 1,
            "purpose": "current candidate",
        },
        "mask": instruction["mask_semantics"],
        "repair_instruction": instruction["source_instruction"],
        "reference_routing": _prompt_image_order_contract(instruction["prompt_image_order"]),
        "identity_profile": {
            "identity": instruction["identity"],
            "body_proportion_source": instruction["body_proportion_source"],
            "body_proportion": instruction["body_proportion"],
            "must_preserve": instruction["must_preserve"],
            "avoid": instruction["avoid"],
        },
    }
    return "Apply this structured character refine instruction exactly:\n" + json.dumps(
        prompt_contract,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _refine_manifest(
    *,
    pack_path: Path,
    identity_profile_path: Path,
    character_id: str,
    source_image: Path,
    mask_image: Path,
    mask_source: Mapping[str, Any],
    references: Sequence[QwenImageEditInpaintReference],
    reference_roles_by_id: Mapping[str, str],
    identity_profile: CharacterIdentityProfileSpec,
    instruction: str,
    normalized_instruction: Mapping[str, Any],
    prompt: str,
    candidates: int,
) -> dict[str, Any]:
    return {
        "kind": QWEN_CHARACTER_REFINE_PLAN_KIND,
        "character_id": character_id,
        "reference_pack": pack_path.as_posix(),
        "identity_profile": identity_profile_path.as_posix(),
        "identity_profile_used": True,
        "source_instruction": instruction,
        "normalizer": "identity_profile_refine_instruction_v1",
        "reference_selector": "static_repair_role_routing_1_to_4_refs_v1",
        "routed_reference_limit": QWEN_CHARACTER_REFINE_MAX_ROUTED_REFS,
        "source_image": image_asset_json(source_image),
        "mask_image": image_asset_json(mask_image),
        "mask_source": dict(mask_source),
        "available_reference_roles": dict(reference_roles_by_id),
        "refs_used": [reference.name for reference in references],
        "references": [
            {
                "input_index": index + 2,
                "role": reference.role,
                "reference_id": reference.name,
                "purpose": reference.purpose,
                "image": image_asset_json(reference.path),
            }
            for index, reference in enumerate(references)
        ],
        "candidates": candidates,
        "identity": identity_profile.identity,
        "body_proportion": identity_profile.body_proportion.model_dump(mode="json"),
        "body_proportion_source": identity_profile.body_proportion_source,
        "optional_missing_refs": identity_profile.optional_missing_refs,
        "must_preserve": identity_profile.must_preserve,
        "avoid": identity_profile.avoid,
        "normalized_instruction": dict(normalized_instruction),
        "prompt": prompt,
    }


def _identity_must_preserve(identity_profile: CharacterIdentityProfileSpec) -> list[str]:
    return list(dict.fromkeys(identity_profile.must_preserve + identity_profile.body_proportion.do_not_change))


def _prompt_image_order_contract(prompt_images: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: image[key]
            for key in ("input_index", "source", "reference_id", "role", "purpose")
            if key in image
        }
        for image in prompt_images
    ]
