from __future__ import annotations

from typing import Any


LORA_HARD_REJECTS = [
    "wrong face",
    "wrong hair length or color",
    "childlike or chibi proportions",
    "wrong visible outfit",
    "missing visible required identity detail",
    "deformed visible anatomy",
    "broken visible hands or feet",
    "framing mismatch",
    "dirty or distracting background",
    "style drift",
    "view label mismatch",
]


def lora_quality_contract() -> dict[str, Any]:
    return {
        "acceptance_rule": "Only canon-worthy images are usable for identity LoRA training.",
        "hard_rejects": LORA_HARD_REJECTS,
    }
