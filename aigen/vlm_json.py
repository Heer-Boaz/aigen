from __future__ import annotations

import json
from typing import Any


class VlmJsonError(RuntimeError):
    pass


def json_object_from_vlm_response(raw_text: str) -> dict[str, Any]:
    text = _json_text(raw_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise VlmJsonError(f"VLM returned non-JSON output: {error}") from error
    if not isinstance(data, dict):
        raise VlmJsonError("VLM returned JSON that is not an object")
    return data


def _json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return text
    opener = lines[0].strip()
    if opener not in {"```json", "```"}:
        return text
    return "\n".join(lines[1:-1]).strip()
