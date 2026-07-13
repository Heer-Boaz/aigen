from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.character_reference_models import (
    CHARACTER_REFERENCE_PACK_KIND,
    CharacterReferenceError,
    CharacterReferencePackOutputSpec,
    CharacterReferencePackSpec,
    ImageAssetSpec,
    load_completed_character_reference_pack,
)
from aigen.image_assets import image_asset_json
from aigen.manifest_io import read_json, resolve_existing_path, write_json


REFERENCE_PACK_FILENAME = "reference_pack.json"


@dataclass(frozen=True)
class LoadedCharacterReferencePack:
    path: Path
    spec: CharacterReferencePackSpec
    references: dict[str, Path]


def load_character_reference_pack(pack_path: Path) -> LoadedCharacterReferencePack:
    resolved_path = pack_path.resolve()
    spec = load_completed_character_reference_pack(
        read_json(resolved_path, label="character reference pack"),
        path_label=resolved_path.as_posix(),
    )
    references = {
        name: resolve_existing_path(asset.path, resolved_path.parent)
        for name, asset in spec.references.items()
    }
    return LoadedCharacterReferencePack(
        path=resolved_path,
        spec=spec,
        references=references,
    )


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


def parse_character_reference_files(reference_files: Sequence[Path], base_dir: Path) -> dict[str, Path]:
    references: dict[str, Path] = {}
    for reference_file in reference_files:
        path = resolve_existing_path(reference_file.as_posix(), base_dir)
        name = path.stem
        if name in references:
            raise CharacterReferenceError(f"Duplicate reference filename: {name}")
        references[name] = path
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
        for name, path in references.items()
    }
    pack = CharacterReferencePackSpec(
        kind=CHARACTER_REFERENCE_PACK_KIND,
        character_id=character_id,
        references={name: ImageAssetSpec(**asset) for name, asset in reference_assets.items()},
        output=CharacterReferencePackOutputSpec(
            directory=output_dir.as_posix(),
            reference_pack=pack_path.as_posix(),
        ),
    )
    payload = {
        "status": "completed",
        **pack.model_dump(mode="json"),
    }
    write_json(pack_path, payload, sort_keys=False)
    return payload


def _validate_character_id(character_id: str) -> None:
    if not character_id.strip():
        raise CharacterReferenceError("character_id must not be empty")


def _validate_reference_mapping(references: Mapping[str, Path]) -> None:
    if not references:
        raise CharacterReferenceError("At least one named reference image is required")
    invalid = sorted(name for name in references if not name.strip())
    if invalid:
        raise CharacterReferenceError("Reference names must not be empty")
