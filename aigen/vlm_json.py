from __future__ import annotations

import json
from typing import Any


class VlmJsonError(RuntimeError):
    pass


def json_object_from_vlm_response(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise VlmJsonError("VLM returned an unterminated Markdown JSON block")
        text = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise VlmJsonError(f"VLM returned non-JSON output: {error}") from error
    if not isinstance(data, dict):
        raise VlmJsonError("VLM returned JSON that is not an object")
    return data
