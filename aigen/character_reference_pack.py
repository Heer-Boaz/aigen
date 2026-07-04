from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from aigen.character_reference_models import (
    CHARACTER_IDENTITY_PROFILE_KIND,
    CHARACTER_REFERENCE_PACK_KIND,
    CHARACTER_REFERENCE_ROLES,
    CHARACTER_REFERENCE_NAME_SET,
    CharacterIdentityProfileSpec,
    CharacterIdentityProfileOutputSpec,
    CharacterReferenceError,
    CharacterReferencePackOutputSpec,
    CharacterReferencePackSpec,
    ImageAssetSpec,
    load_character_identity_vlm_response,
    load_character_reference_pack_payload,
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
        if separator != "=" or not name or not raw_path:
            raise CharacterReferenceError(f"Reference must be name=path: {raw_reference}")
        if name not in CHARACTER_REFERENCE_NAME_SET:
            allowed = ", ".join(sorted(CHARACTER_REFERENCE_NAME_SET))
            raise CharacterReferenceError(f"Unknown reference name {name}; expected one of: {allowed}")
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
    reference_roles = {name: CHARACTER_REFERENCE_ROLES[name] for name in reference_assets}
    pack = CharacterReferencePackSpec(
        kind=CHARACTER_REFERENCE_PACK_KIND,
        character_id=character_id,
        references={name: ImageAssetSpec(**asset) for name, asset in reference_assets.items()},
        reference_roles=reference_roles,
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
    pack = load_character_reference_pack_payload(
        _strip_status(read_json(pack_path, label="character reference pack")),
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
        response = _identity_response_from_raw(raw_text, target_path)
        profile = CharacterIdentityProfileSpec(
            kind=CHARACTER_IDENTITY_PROFILE_KIND,
            character_id=pack.character_id,
            source_reference_pack=pack_path.as_posix(),
            identity=response.identity,
            reference_roles=_merged_reference_roles(pack.reference_roles, response.reference_roles),
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


def _identity_response_from_raw(raw_text: str, target_path: Path):
    try:
        generated = json_object_from_vlm_response(raw_text)
    except VlmJsonError as error:
        raise CharacterReferenceError(f"Invalid character identity parser response {target_path.as_posix()}: {error}") from error
    return load_character_identity_vlm_response(generated, path_label=target_path.as_posix())


def _identity_parser_prompt(pack: CharacterReferencePackSpec) -> str:
    reference_lines = "\n".join(
        f"- {name}: {pack.reference_roles[name]} ({asset.width}x{asset.height}, {asset.mode})"
        for name, asset in _ordered_reference_assets(pack.references).items()
    )
    return f"""You are building a compact machine-readable identity dossier for a local character edit pipeline.

Images are supplied in this exact order:
{reference_lines}

Describe the same character across the references. Extract stable visual identity facts only.
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
    "body_shape": "short factual phrase",
    "style": "short factual phrase"
  }},
  "reference_roles": {{
    "front": "how this provided reference should be used",
    "portrait": "how this provided reference should be used"
  }},
  "must_preserve": [
    "short visual fact that must survive edits"
  ],
  "avoid": [
    "short visual drift or mistaken substitution to avoid"
  ]
}}

Rules:
- Include only reference_roles keys for images actually supplied.
- Use concise strings, not paragraphs.
- Use plain JSON only. No Markdown.
"""


def _identity_output_path(pack_path: Path, output_path: Path | None) -> Path:
    if output_path is None:
        return pack_path.parent / IDENTITY_PROFILE_FILENAME
    return output_path.resolve()


def _ordered_references(references: Mapping[str, Path]) -> dict[str, Path]:
    return {name: references[name] for name in CHARACTER_REFERENCE_ROLES if name in references}


def _ordered_reference_assets(references: Mapping[str, ImageAssetSpec]) -> dict[str, ImageAssetSpec]:
    return {name: references[name] for name in CHARACTER_REFERENCE_ROLES if name in references}


def _validate_character_id(character_id: str) -> None:
    if not character_id.strip():
        raise CharacterReferenceError("character_id must not be empty")


def _validate_reference_mapping(references: Mapping[str, Path]) -> None:
    if not references:
        raise CharacterReferenceError("At least one named reference image is required")
    unknown = sorted(name for name in references if name not in CHARACTER_REFERENCE_NAME_SET)
    if unknown:
        allowed = ", ".join(sorted(CHARACTER_REFERENCE_NAME_SET))
        raise CharacterReferenceError(f"Unknown reference name(s) {', '.join(unknown)}; expected one of: {allowed}")


def _merged_reference_roles(base_roles: dict[str, str], parsed_roles: dict[str, str]) -> dict[str, str]:
    merged = dict(base_roles)
    for name, role in parsed_roles.items():
        if name not in base_roles:
            raise CharacterReferenceError(f"Identity parser returned a role for unknown reference {name}")
        merged[name] = role
    return merged


def _strip_status(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") == "completed":
        payload = dict(payload)
        del payload["status"]
    return payload
