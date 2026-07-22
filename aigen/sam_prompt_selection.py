from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SAM_PROMPT_SELECTION_KIND = "aigen-sam-prompt-selection"
SAM_PROMPT_SELECTION_VERSION = 1
SAM_PROMPT_MODES = frozenset({"box", "points", "box+points"})


@dataclass(frozen=True)
class SAMPromptSelection:
    image: str
    prompt_mode: str
    box: str
    positive_points: str
    negative_points: str

    def __post_init__(self) -> None:
        if self.prompt_mode not in SAM_PROMPT_MODES:
            raise ValueError(f"Unsupported SAM prompt mode: {self.prompt_mode}")

    def save(self, path: Path) -> None:
        payload = {
            "kind": SAM_PROMPT_SELECTION_KIND,
            "version": SAM_PROMPT_SELECTION_VERSION,
            "image": self.image,
            "prompt_mode": self.prompt_mode,
            "box": self.box,
            "positive_points": self.positive_points,
            "negative_points": self.negative_points,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SAMPromptSelection:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("SAM selection must be a JSON object.")
        if payload.get("kind") != SAM_PROMPT_SELECTION_KIND:
            raise ValueError("File is not an Aigen SAM prompt selection.")
        if payload.get("version") != SAM_PROMPT_SELECTION_VERSION:
            raise ValueError("Unsupported Aigen SAM prompt selection version.")
        values = {}
        for name in ("image", "prompt_mode", "box", "positive_points", "negative_points"):
            value = payload.get(name)
            if not isinstance(value, str):
                raise ValueError(f"SAM selection field {name!r} must be a string.")
            values[name] = value
        return cls(**values)
