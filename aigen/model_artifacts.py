from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from aigen.manifest_io import sha256_file


MODEL_ARTIFACT_PROVENANCE_FORMAT = "aigen.model-artifacts.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelArtifactComponent:
    name: str
    root: Path
    files: tuple[Path, ...]


def build_model_artifact_provenance(
    components: tuple[ModelArtifactComponent, ...],
) -> dict[str, object]:
    records = []
    for component in sorted(components, key=lambda candidate: candidate.name):
        root = component.root.expanduser().resolve()
        files = []
        for path in sorted(component.files):
            resolved = path.expanduser().resolve()
            relative = resolved.relative_to(root).as_posix()
            files.append(
                {
                    "path": relative,
                    "size_bytes": resolved.stat().st_size,
                    "sha256": sha256_file(resolved),
                }
            )
        records.append(
            {
                "name": component.name,
                "files": files,
            }
        )
    payload = {
        "format": MODEL_ARTIFACT_PROVENANCE_FORMAT,
        "components": records,
    }
    fingerprint = sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        **payload,
        "fingerprint": fingerprint,
    }


def validate_model_artifact_provenance(payload: dict[str, object]) -> None:
    if set(payload) != {"format", "components", "fingerprint"}:
        raise ValueError("invalid model artifact provenance keys")
    if payload["format"] != MODEL_ARTIFACT_PROVENANCE_FORMAT:
        raise ValueError("unsupported model artifact provenance format")
    components = payload["components"]
    if not isinstance(components, list) or not components:
        raise ValueError("model artifact provenance has no components")
    component_names = []
    for component in components:
        if not isinstance(component, dict) or set(component) != {"name", "files"}:
            raise ValueError("invalid model artifact component")
        name = component["name"]
        files = component["files"]
        if not isinstance(name, str) or not name:
            raise ValueError("invalid model artifact component name")
        if not isinstance(files, list) or not files:
            raise ValueError(f"model artifact component {name!r} has no files")
        component_names.append(name)
        file_paths = []
        for file_record in files:
            if not isinstance(file_record, dict) or set(file_record) != {
                "path",
                "size_bytes",
                "sha256",
            }:
                raise ValueError(f"invalid model artifact file in {name!r}")
            path = file_record["path"]
            size_bytes = file_record["size_bytes"]
            checksum = file_record["sha256"]
            if (
                not isinstance(path, str)
                or not path
                or Path(path).is_absolute()
                or ".." in Path(path).parts
            ):
                raise ValueError(f"invalid model artifact path in {name!r}")
            if not isinstance(size_bytes, int) or size_bytes < 0:
                raise ValueError(f"invalid model artifact size in {name!r}")
            if not isinstance(checksum, str) or not _SHA256_PATTERN.fullmatch(
                checksum
            ):
                raise ValueError(f"invalid model artifact checksum in {name!r}")
            file_paths.append(path)
        if file_paths != sorted(file_paths) or len(file_paths) != len(set(file_paths)):
            raise ValueError(f"unordered or duplicate model artifacts in {name!r}")
    if component_names != sorted(component_names) or len(component_names) != len(
        set(component_names)
    ):
        raise ValueError("unordered or duplicate model artifact components")
    expected_fingerprint = _provenance_fingerprint(
        {
            "format": payload["format"],
            "components": components,
        }
    )
    if payload["fingerprint"] != expected_fingerprint:
        raise ValueError("model artifact provenance fingerprint mismatch")


@lru_cache(maxsize=None)
def model_artifact_stat_revision(component: ModelArtifactComponent) -> str:
    root = component.root.expanduser().resolve()
    inventory = []
    for path in component.files:
        resolved = path.expanduser().resolve()
        file_stat = resolved.stat()
        inventory.append(
            (
                resolved.relative_to(root).as_posix(),
                file_stat.st_size,
                file_stat.st_mtime_ns,
                file_stat.st_ctime_ns,
            )
        )
    return sha256(
        json.dumps(inventory, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _provenance_fingerprint(payload: dict[str, object]) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
