from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class ManifestIOError(RuntimeError):
    pass


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ManifestIOError(f"Missing {label}: {path.as_posix()}") from error
    except json.JSONDecodeError as error:
        raise ManifestIOError(f"Invalid {label}: {path.as_posix()}: {error}") from error


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_line(stream: Any, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_existing_path(value: str, base_dir: Path) -> Path:
    path = resolve_output_path(value, base_dir)
    if not path.exists():
        raise ManifestIOError(f"Missing path: {path.as_posix()}")
    return path


def resolve_output_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def relative_path(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()


def schema_reference(target_path: Path, schema_path: Path) -> str:
    try:
        return relative_path(schema_path, target_path.parent)
    except ValueError:
        return schema_path.resolve().as_posix()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def file_manifest(path: Path) -> dict[str, str]:
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
    }
