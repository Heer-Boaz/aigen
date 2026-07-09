from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.character_reference_models import (
    CHARACTER_BODY_PROPORTION_SOURCE,
    CHARACTER_IDENTITY_PROFILE_KIND,
    CHARACTER_REFERENCE_PACK_KIND,
    CharacterIdentityProfileSpec,
    CharacterIdentityProfileOutputSpec,
    CharacterReferenceError,
    CharacterReferencePackOutputSpec,
    CharacterReferencePackSpec,
    ImageAssetSpec,
    load_character_edit_plan_vlm_response,
    load_completed_character_reference_pack,
    load_character_identity_vlm_response,
)
from aigen.image_assets import image_asset_json
from aigen.manifest_io import read_json, resolve_existing_path, write_json
from aigen.progress import StatusReporter
from aigen.vlm_json import VlmJsonError, json_object_from_vlm_response
from aigen.vlm_qwen import QwenVlm, QwenVlmConfig, qwen_vlm_config_json


REFERENCE_PACK_FILENAME = "reference_pack.json"
IDENTITY_PROFILE_FILENAME = "identity_profile.json"


@dataclass(frozen=True)
class CharacterEditPlanParseResult:
    selected_refs: tuple[str, ...]
    selected_planner_refs: tuple[str, ...]
    planner_reference_map: dict[str, str]
    reference_semantics: dict[str, str]
    visual_analysis: dict[str, Any]
    planner_context: dict[str, Any]
    edit_instruction: str
    prompt: str
    planner_image_paths: tuple[Path, ...]
    raw_text: str


def parse_character_reference_args(reference_args: Sequence[str], base_dir: Path) -> dict[str, Path]:
    references: dict[str, Path] = {}
    for raw_reference in reference_args:
        name, separator, raw_path = raw_reference.partition("=")
        name = name.strip()
        if separator != "=" or not name or not raw_path:
            raise CharacterReferenceError(f"Reference must be name=path: {raw_reference}")
        if name in references:
            raise CharacterReferenceError(f"Duplicate reference name: {name}")
        references[name] = resolve_existing_path(raw_path, base_dir)
    return references


def build_character_reference_pack(
    *,
    character_id: str,
    references: Mapping[str, Path],
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    _validate_character_id(character_id)
    _validate_reference_mapping(references)
    output_dir = output_dir.resolve()
    pack_path = output_dir / REFERENCE_PACK_FILENAME
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise CharacterReferenceError(f"Reference pack output exists and overwrite=false: {output_dir.as_posix()}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_assets = {
        name: ImageAssetSpec(**image_asset_json(path)).model_dump(mode="json")
        for name, path in _ordered_references(references).items()
    }
    pack = CharacterReferencePackSpec(
        kind=CHARACTER_REFERENCE_PACK_KIND,
        character_id=character_id,
        references={name: ImageAssetSpec(**asset) for name, asset in reference_assets.items()},
        reference_hints={name: name for name in reference_assets},
        output=CharacterReferencePackOutputSpec(
            directory=output_dir.as_posix(),
            reference_pack=pack_path.as_posix(),
        ),
    )
    payload = {
        "status": "completed",
        **pack.model_dump(mode="json"),
    }
    write_json(pack_path, payload)
    return payload


def parse_character_reference_pack(
    pack_path: Path,
    config: QwenVlmConfig,
    *,
    output_path: Path | None,
    overwrite: bool,
    progress: StatusReporter,
) -> dict[str, Any]:
    pack_path = pack_path.resolve()
    pack = load_completed_character_reference_pack(
        read_json(pack_path, label="character reference pack"),
        path_label=pack_path.as_posix(),
    )
    target_path = _identity_output_path(pack_path, output_path)
    if target_path.exists() and not overwrite:
        raise CharacterReferenceError(f"Identity profile exists and overwrite=false: {target_path.as_posix()}")

    reference_paths = {
        name: resolve_existing_path(asset.path, pack_path.parent) for name, asset in pack.references.items()
    }
    image_paths = list(_ordered_references(reference_paths).values())
    prompt = _identity_parser_prompt(pack)
    progress.phase("parse character references with Qwen VLM")
    with closing(QwenVlm(config)) as runner:
        raw_text = runner.describe_image(prompt, image_paths)
        response = _identity_response_from_raw(raw_text, target_path, reference_ids=frozenset(pack.references))
        profile = CharacterIdentityProfileSpec(
            kind=CHARACTER_IDENTITY_PROFILE_KIND,
            character_id=pack.character_id,
            source_reference_pack=pack_path.as_posix(),
            identity=response.identity,
            body_proportion=response.body_proportion,
            body_proportion_source=CHARACTER_BODY_PROPORTION_SOURCE,
            reference_roles=response.reference_roles,
            optional_missing_refs=_optional_missing_refs(response.reference_roles),
            must_preserve=response.must_preserve,
            avoid=response.avoid,
            parser=qwen_vlm_config_json(config) | {"device_report": runner.device_report},
            output=CharacterIdentityProfileOutputSpec(identity_profile=target_path.as_posix()),
        )

    payload = {
        "status": "completed",
        **profile.model_dump(mode="json"),
    }
    write_json(target_path, payload)
    return payload


def parse_character_edit_plan(
    *,
    runner: QwenVlm,
    pack: CharacterReferencePackSpec,
    reference_paths: Mapping[str, Path],
    case_name: str,
    user_instruction: str,
    planner_context: Mapping[str, Any],
    reference_observations: Mapping[str, Any],
    path_label: str,
) -> CharacterEditPlanParseResult:
    planner_reference_map = character_planner_reference_map(pack)
    planner_reference_ids = tuple(planner_reference_map)
    image_paths = tuple(reference_paths[reference_id] for reference_id in planner_reference_map.values())
    prompt = _edit_plan_parser_prompt(
        pack=pack,
        planner_reference_map=planner_reference_map,
        case_name=case_name,
        user_instruction=user_instruction,
        planner_context=planner_context,
        reference_observations=reference_observations,
    )
    raw_text = runner.describe_image(prompt, list(image_paths))
    response = _edit_plan_response_from_raw(raw_text, path_label, reference_ids=frozenset(planner_reference_ids))
    return CharacterEditPlanParseResult(
        selected_refs=tuple(planner_reference_map[reference_id] for reference_id in response.selected_refs),
        selected_planner_refs=tuple(response.selected_refs),
        planner_reference_map=planner_reference_map,
        reference_semantics=dict(response.reference_semantics),
        visual_analysis=dict(response.visual_analysis),
        planner_context=dict(planner_context),
        edit_instruction=response.edit_instruction.strip(),
        prompt=prompt,
        planner_image_paths=image_paths,
        raw_text=raw_text,
    )


def character_planner_reference_map(pack: CharacterReferencePackSpec) -> dict[str, str]:
    return {
        f"reference{index + 1}": reference_id
        for index, reference_id in enumerate(pack.references)
    }


def _identity_response_from_raw(
    raw_text: str,
    target_path: Path,
    *,
    reference_ids: frozenset[str],
):
    try:
        generated = json_object_from_vlm_response(raw_text)
    except VlmJsonError as error:
        raise CharacterReferenceError(f"Invalid character identity parser response {target_path.as_posix()}: {error}") from error
    return load_character_identity_vlm_response(
        generated,
        path_label=target_path.as_posix(),
        reference_ids=reference_ids,
    )


def _edit_plan_response_from_raw(
    raw_text: str,
    path_label: str,
    *,
    reference_ids: frozenset[str],
):
    try:
        generated = json_object_from_vlm_response(raw_text)
    except VlmJsonError as error:
        raise CharacterReferenceError(
            f"Invalid character edit plan response {path_label}: {error}"
        ) from error
    return load_character_edit_plan_vlm_response(
        generated,
        path_label=path_label,
        reference_ids=reference_ids,
    )


def _identity_parser_prompt(pack: CharacterReferencePackSpec) -> str:
    reference_ids = ", ".join(_ordered_reference_assets(pack.references))
    reference_lines = "\n".join(
        f"- {name}: provided label/hint {pack.reference_hints[name]!r} ({asset.width}x{asset.height}, {asset.mode})"
        for name, asset in _ordered_reference_assets(pack.references).items()
    )
    return f"""You are building a compact machine-readable identity dossier for a local character edit pipeline.

Images are supplied in this exact order:
{reference_lines}

The supplied reference ids are exactly: {reference_ids}.
The provided labels are optional evidence only. Infer each reference role from the pixels yourself.
Allowed reference role values are: front, portrait, side, back, body_shape.

Describe the same character across the references. Extract stable visual identity and body-proportion facts only.
Do not invent names, story, mood, quality boosters, scene details, poster text or layout instructions.

Return exactly one JSON object with this shape:
{{
  "identity": {{
    "hair": "short factual phrase",
    "eyes": "short factual phrase when visible",
    "face": "short factual phrase",
    "neckwear": "short factual phrase when present",
    "top": "short factual phrase",
    "bottom": "short factual phrase",
    "legwear": "short factual phrase",
    "footwear": "short factual phrase",
    "style": "short factual phrase"
  }},
  "body_proportion": {{
    "chest_size": "short factual phrase",
    "build": "short factual phrase",
    "shoulder_width": "short factual phrase",
    "waist": "short factual phrase",
    "hip_skirt_silhouette": "short factual phrase",
    "side_body_thickness": "short factual phrase",
    "leg_proportion": "short factual phrase",
    "skirt_back_shape": "short factual phrase",
    "do_not_change": [
      "short body-proportion invariant"
    ],
    "evidence_refs": [
      "reference id from the supplied image list"
    ]
  }},
  "reference_roles": {{
    "reference id from the supplied image list": "front"
  }},
  "must_preserve": [
    "short visual fact that must survive edits"
  ],
  "avoid": [
    "short visual drift or mistaken substitution to avoid"
  ]
}}

Rules:
- Include exactly one reference_roles key for each supplied reference id.
- reference_roles keys must be exactly the supplied reference ids: {reference_ids}.
- Do not add keys for absent optional roles. If there is no supplied reference id for body_shape, do not create one.
- reference_roles values must be one of: front, portrait, side, back, body_shape.
- A dedicated body_shape image is optional. If none is supplied, infer body_proportion from the whole pack.
- body_proportion.evidence_refs must be a non-empty subset of the supplied reference ids: {reference_ids}.
- Use concise strings, not paragraphs.
- Use plain JSON only. No Markdown.
"""


def _edit_plan_parser_prompt(
    *,
    pack: CharacterReferencePackSpec,
    planner_reference_map: Mapping[str, str],
    case_name: str,
    user_instruction: str,
    planner_context: Mapping[str, Any],
    reference_observations: Mapping[str, Any],
) -> str:
    reference_lines = "\n".join(
        _edit_plan_reference_line(
            input_index=index + 1,
            planner_reference_id=planner_reference_id,
            pack_reference_id=pack_reference_id,
            pack=pack,
        )
        for index, (planner_reference_id, pack_reference_id) in enumerate(planner_reference_map.items())
    )
    reference_id_list = ", ".join(planner_reference_map)
    planner_context_json = json.dumps(planner_context, indent=2, sort_keys=True)
    reference_observations_json = json.dumps(reference_observations, indent=2, sort_keys=True)
    return f"""You are the route-aware multi-reference visual planner for a local character image-edit pipeline.
Step 3 single-reference observation has already been run for every reference.
In this Qwen2.5-VL call, inspect all supplied images yourself and execute PixAI core steps 4 through 7:
4. Identify the relevant subject or subjects for the requested task.
5. Extract task-relevant visual components from the references.
6. Align the same components across references.
7. Choose which components matter for this specific output.

Images are supplied in this exact order:
{reference_lines}

Available reference ids are exactly:
{reference_id_list}

The requested output case is: {case_name}.
The user's requested edit is exactly: {user_instruction!r}.
Planner context before image analysis:
{planner_context_json}

reference_observations:
{reference_observations_json}

Your job:
1. Look at all supplied images yourself.
2. Use the planner context only as task focus and routing intent.
3. Use reference_observations to ensure every reference was considered.
4. If an observation conflicts with the image, trust the image.
5. Write compact visual_analysis covering subject mapping, component evidence, cross-reference alignment, output component priorities, and reference encoding summaries for every available reference.
6. Select 1 to 3 references for Qwen Image Edit after that visual analysis.
7. Write concise reference_semantics for the selected or relevant references.
8. Write one concise edit_instruction for Qwen Image Edit.

Rules:
- The image editor redraws the character; it is not a bitmap rotation, crop, resize or file transform tool.
- Use visual facts only when they are visible in the supplied images.
- Separate user-requested output constraints from reference evidence. Requested background, lighting, gaze, expression, scene, action, or camera view are output constraints; they are not reference evidence unless visible in a supplied image.
- Treat reference backgrounds as context, not automatic identity/component evidence.
- art_style or rendering_style is possible visual evidence; decide its relevance from the images and the task route, and explain it when used.
- Do not add names, story, mood, annotations, text, props, scene details, or extra outputs unless they are explicitly requested by the user, present in the planner context, or visible in the supplied images.
- Do not simply copy reference_observations; use them as audit context.
- Do not choose references that are irrelevant to the requested edit.
- selected_refs must be justified by output_component_priorities and cross_reference_alignment.
- selected_refs must support the high-priority output evidence you choose within the 1 to 3 reference limit.
- If important evidence is taken from a reference that is not selected, explain why in visual_analysis.unselected_important_evidence.
- reference_encoding_summary must include every available reference id: {reference_id_list}.
- low_priority_components_for_this_output must not include high or medium priority components from output_component_priorities.
- Qwen Image Edit accepts one to three selected reference images for this path.
- Keep visual_analysis compact: at most 6 component_evidence items, at most 6 cross_reference_alignment items, at most 6 output_component_priorities, and at most 3 unselected_important_evidence items.
- Use single-line JSON string values. Do not put unescaped quotation marks inside string values.
- Do not include trailing commas.
- Return plain JSON only. No Markdown.

Return exactly one JSON object with this shape:
{{
  "selected_refs": [
    "reference id from the available reference id list"
  ],
  "reference_semantics": {{
    "reference id from the available reference id list": "short semantic label inferred from that image"
  }},
  "visual_analysis": {{
    "reference_encoding_summary": {{
      "reference1": "short visual summary of this supplied image",
      "reference2": "short visual summary of this supplied image"
    }},
    "subject_map": {{
      "primary_subject": {{
        "evidence_refs": [
          "reference1"
        ],
        "notes": "short subject-identification note"
      }}
    }},
    "component_evidence": [
      {{
        "component": "short component name",
        "evidence_refs": [
          "reference1"
        ],
        "task_relevance": "high, medium, or low",
        "visibility": "clear, partial, or unclear"
      }}
    ],
    "cross_reference_alignment": [
      {{
        "component": "short component name",
        "aligned_refs": [
          "reference1"
        ],
        "anchor_ref": "reference1",
        "supporting_refs": [],
        "conflicts": [],
        "confidence": "high, medium, or low"
      }}
    ],
    "output_component_priorities": [
      {{
        "component": "short component name",
        "priority": "high, medium, or low",
        "reason": "short task-specific reason"
      }}
    ],
    "unselected_important_evidence": [],
    "low_priority_components_for_this_output": [
      "short component name"
    ],
    "unresolved_alignment_questions": []
  }},
  "edit_instruction": "one concise instruction"
}}

Validation:
- selected_refs must contain 1 to 3 ids from exactly this list: {reference_id_list}.
- selected_refs order is the order the image editor will receive the images.
- selected_refs must be explainable from visual_analysis.output_component_priorities and visual_analysis.cross_reference_alignment.
- reference_semantics is optional evidence authored by you; its keys, when present, must be from exactly this list: {reference_id_list}.
- visual_analysis is a compact freeform JSON object for audit; use the listed reference-observation and step-4-through-7 keys when they apply and keep values short.
- visual_analysis.reference_encoding_summary must contain every available reference id from exactly this list: {reference_id_list}.
- visual_analysis.output_component_priorities and visual_analysis.low_priority_components_for_this_output must not contradict each other.
- Use visual_analysis.unselected_important_evidence when a reference contributes important evidence but cannot be selected.
- edit_instruction must be a single non-empty string.
- Output must be valid JSON parseable by Python json.loads.
"""


def _edit_plan_reference_line(
    *,
    input_index: int,
    planner_reference_id: str,
    pack_reference_id: str,
    pack: CharacterReferencePackSpec,
) -> str:
    asset = pack.references[pack_reference_id]
    return f"- input {input_index}: reference id {planner_reference_id} ({asset.width}x{asset.height}, {asset.mode})"


def _identity_output_path(pack_path: Path, output_path: Path | None) -> Path:
    if output_path is None:
        return pack_path.parent / IDENTITY_PROFILE_FILENAME
    return output_path.resolve()


def _ordered_references(references: Mapping[str, Path]) -> dict[str, Path]:
    return dict(references)


def _ordered_reference_assets(references: Mapping[str, ImageAssetSpec]) -> dict[str, ImageAssetSpec]:
    return dict(references)


def _validate_character_id(character_id: str) -> None:
    if not character_id.strip():
        raise CharacterReferenceError("character_id must not be empty")


def _validate_reference_mapping(references: Mapping[str, Path]) -> None:
    if not references:
        raise CharacterReferenceError("At least one named reference image is required")
    invalid = sorted(name for name in references if not name.strip())
    if invalid:
        raise CharacterReferenceError("Reference names must not be empty")


def _optional_missing_refs(reference_roles: Mapping[str, str]) -> list[str]:
    if "body_shape" in reference_roles.values():
        return []
    return ["body_shape"]
