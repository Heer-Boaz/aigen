from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aigen.manifest_io import write_json_line
from aigen.pix2pix.errors import Pix2PixError


def read_json_records(path: Path, *, label: str) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Pix2PixError(f"cannot read {label}: {path.as_posix()}") from error
    records = []
    for line_number, text in enumerate(lines, start=1):
        if not text:
            raise Pix2PixError(f"blank line in {label} at line {line_number}")
        try:
            record = json.loads(text)
        except json.JSONDecodeError as error:
            raise Pix2PixError(
                f"invalid {label} JSON at line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise Pix2PixError(f"{label} line {line_number} must be a JSON object")
        records.append(record)
    if not records:
        raise Pix2PixError(f"{label} is empty: {path.as_posix()}")
    return tuple(records)


def write_json_records(
    path: Path,
    records: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            write_json_line(stream, record)


def require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    keys = set(payload)
    if keys == expected:
        return
    missing = sorted(expected - keys)
    unexpected = sorted(keys - expected)
    details = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected {', '.join(unexpected)}")
    raise Pix2PixError(f"invalid {label}: {'; '.join(details)}")


def corpus_member(root: Path, relative: str, *, label: str) -> Path:
    member = Path(relative)
    if member.is_absolute():
        raise Pix2PixError(f"{label} must be relative to the corpus root")
    path = (root / member).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise Pix2PixError(f"{label} escapes the corpus root: {relative}") from error
    return path
