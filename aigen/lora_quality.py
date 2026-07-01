from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


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
    "bad crop",
    "dirty or distracting background",
    "style drift",
    "view label mismatch",
]

LORA_CROP_REGIONS = [
    "full",
    "face",
    "torso",
    "waist_lower_body",
    "legs_feet",
    "full_silhouette",
]


def lora_quality_contract() -> dict[str, Any]:
    return {
        "acceptance_rule": "Only canon-worthy images are usable for identity LoRA training.",
        "hard_rejects": LORA_HARD_REJECTS,
        "crop_regions": LORA_CROP_REGIONS,
    }


def write_lora_crop_sheet(image_path: Path, output_path: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    boxes = [
        ("full", (0, 0, width, height)),
        ("face", (0, 0, width, max(1, int(height * 0.38)))),
        ("torso", (0, int(height * 0.20), width, max(1, int(height * 0.62)))),
        ("waist/lower body", (0, int(height * 0.43), width, max(1, int(height * 0.74)))),
        ("legs/feet", (0, int(height * 0.58), width, height)),
        ("silhouette", (0, 0, width, height)),
    ]
    thumb_width = 176
    label_height = 28
    sheet = Image.new("RGB", (thumb_width * len(boxes), thumb_width + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, box) in enumerate(boxes):
        crop = image.crop(box)
        crop.thumbnail((thumb_width, thumb_width), Image.Resampling.LANCZOS)
        x = index * thumb_width
        sheet.paste(crop, (x + (thumb_width - crop.width) // 2, label_height + (thumb_width - crop.height) // 2))
        draw.text((x + 6, 7), label, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
