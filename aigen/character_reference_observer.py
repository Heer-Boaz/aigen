from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aigen.character_reference_models import CharacterReferencePackSpec
from aigen.character_reference_observation_models import (
    CHARACTER_REFERENCE_OBSERVATION_SET_KIND,
    CharacterReferenceObservationError,
    CharacterReferenceObservationSetSpec,
    load_character_reference_observation,
    load_character_reference_observation_set,
)
from aigen.vlm_json import VlmJsonError, json_object_from_vlm_response
from aigen.vlm_qwen import QwenVlm


def observe_character_references(
    *,
    runner: QwenVlm,
    pack: CharacterReferencePackSpec,
    reference_paths: Mapping[str, Path],
    planner_reference_map: Mapping[str, str],
    planner_context: Mapping[str, Any],
    path_label: str,
) -> CharacterReferenceObservationSetSpec:
    observations = {}
    for planner_reference_id, pack_reference_id in planner_reference_map.items():
        prompt = _reference_observation_prompt(
            pack=pack,
            planner_reference_id=planner_reference_id,
            pack_reference_id=pack_reference_id,
            planner_context=planner_context,
        )
        raw_text = runner.describe_image(prompt, [reference_paths[pack_reference_id]])
        data = _observation_data_from_raw(raw_text, f"{path_label}#{planner_reference_id}")
        observations[planner_reference_id] = load_character_reference_observation(
            data,
            path_label=f"{path_label}#{planner_reference_id}",
            reference_id=planner_reference_id,
        )
    return load_character_reference_observation_set(
        {
            "kind": CHARACTER_REFERENCE_OBSERVATION_SET_KIND,
            "observations": {
                reference_id: observation.model_dump(mode="json")
                for reference_id, observation in observations.items()
            },
        },
        path_label=path_label,
        reference_ids=tuple(planner_reference_map),
    )


def _observation_data_from_raw(raw_text: str, path_label: str) -> dict[str, Any]:
    try:
        return json_object_from_vlm_response(raw_text)
    except VlmJsonError as error:
        raise CharacterReferenceObservationError(
            f"Invalid character reference observation {path_label}: {error}"
        ) from error


def _reference_observation_prompt(
    *,
    pack: CharacterReferencePackSpec,
    planner_reference_id: str,
    pack_reference_id: str,
    planner_context: Mapping[str, Any],
) -> str:
    asset = pack.references[pack_reference_id]
    planner_context_json = json.dumps(planner_context, indent=2, sort_keys=True)
    return f"""You are step 3 of a PixAI-style core pipeline: single-reference visual observation.

You receive exactly one reference image.
Reference id: {planner_reference_id}
Image metadata: {asset.width}x{asset.height}, {asset.mode}

Planner context:
{planner_context_json}

Your job:
1. Inspect exactly this one supplied reference image.
2. Do not compare this image with other references.
3. Determine view_or_framing only from the pixels in this image.
4. Do not use filename, pack reference label, neutral reference id, user request, or planner context as truth about what this image shows.
5. Use only visible reference evidence in this image.
6. Use planner context only to decide which visible evidence may be relevant later.
7. Separate visible reference evidence from user-requested output constraints such as requested background, lighting, gaze, expression, scene, or action.
8. Treat the reference background as context, not automatic identity or component evidence.
9. Include art_style or rendering_style as possible visible evidence when it is present.

Return exactly one JSON object with this shape:
{{
  "reference_id": "{planner_reference_id}",
  "visual_summary": "short visual summary",
  "visible_subjects": [
    {{
      "label": "primary subject or other visible subject",
      "visibility": "clear, partial, or unclear",
      "notes": "short note"
    }}
  ],
  "view_or_framing": "short description",
  "visible_components": [
    {{
      "component": "short component name",
      "visibility": "clear, partial, occluded, or unclear",
      "task_relevance_hint": "high, medium, low, or unknown",
      "notes": "short visible evidence"
    }}
  ],
  "occlusion_or_quality_notes": [],
  "text_or_symbol_notes": [],
  "uncertainties": []
}}

Rules:
- reference_id must be exactly "{planner_reference_id}".
- visual_summary must be non-empty.
- view_or_framing must describe this image, not the requested output.
- visible_components may include art_style or rendering_style when visible.
- Do not list user-requested output constraints as visible components unless they are also visible in this reference image.
- Keep values compact.
- Return plain JSON only. No Markdown.
"""
