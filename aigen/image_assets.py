from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from aigen.manifest_io import sha256_file


def image_asset_json(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        mode = image.mode
        width, height = image.size
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "mode": mode,
        "width": width,
        "height": height,
    }


def image_asset_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "sha256", "mode", "width", "height"],
        "properties": {
            "path": {"type": "string"},
            "sha256": {"type": "string"},
            "mode": {"type": "string"},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
        },
    }
