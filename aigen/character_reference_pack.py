from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from contextlib import closing
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


def _identity_parser_prompt(pack: CharacterReferencePackSpec) -> str:
    reference_lines = "\n".join(
        f"- {name}: provided label/hint {pack.reference_hints[name]!r} ({asset.width}x{asset.height}, {asset.mode})"
        for name, asset in _ordered_reference_assets(pack.references).items()
    )
    return f"""You are building a compact machine-readable identity dossier for a local character edit pipeline.

Images are supplied in this exact order:
{reference_lines}

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
- Include exactly one reference_roles key for each supplied image.
- reference_roles values must be one of: front, portrait, side, back, body_shape.
- A dedicated body_shape image is optional. If none is supplied, infer body_proportion from the whole pack.
- body_proportion.evidence_refs must name supplied image ids.
- Use concise strings, not paragraphs.
- Use plain JSON only. No Markdown.
"""


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
