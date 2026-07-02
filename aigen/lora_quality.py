from __future__ import annotations

from typing import Any


LORA_HARD_REJECTS = [
    "wrong face",
    "wrong hair length or color",
    "wrong outfit",
    "missing required neckwear or accessory from the identity prompt",
    "missing required waist or lower-body garment from the identity prompt",
    "missing required belt or waist detail from the identity prompt",
    "missing required legwear from the identity prompt",
    "missing required footwear from the identity prompt",
    "deformed body",
    "broken hands or feet",
    "bad framing or cut-off full body",
    "dirty or distracting background",
    "style drift",
    "view label mismatch",
]


def lora_quality_contract() -> dict[str, Any]:
    return {
        "acceptance_rule": "Only canon-worthy images are usable for identity LoRA training.",
        "hard_rejects": LORA_HARD_REJECTS,
    }
