from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO


def dump_json(handle: TextIO, payload: object, *, pretty: bool) -> None:
    json.dump(
        payload,
        handle,
        ensure_ascii=False,
        indent=2 if pretty else None,
        sort_keys=True,
    )
    handle.write("\n")


def write_json(path: Path, payload: object, *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        dump_json(handle, payload, pretty=pretty)


def command_error_payload(error: Exception) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "error",
        "error": error.__class__.__name__,
        "message": str(error),
    }
