from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from aigen.character_view_models import (
    CharacterViewError,
    ViewBankTrainingValidationSpec,
    load_character_view_bank,
)
from aigen.manifest_io import resolve_existing_path, write_json
from aigen.vlm_qwen import QwenVlm, QwenVlmConfig
from aigen.vlm_json import VlmJsonError, json_object_from_vlm_response


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainingValidationPayload(StrictModel):
    usable_for_lora_training: bool
    hard_rejects: dict[str, bool]
    scores: dict[str, float]
    evidence: dict[str, Any]


def validate_view_bank_entry_for_lora_training(
    bank_path: Path,
    view_name: str,
    config: QwenVlmConfig,
) -> dict[str, Any]:
    bank = load_character_view_bank(bank_path)
    if view_name not in bank.views:
        raise CharacterViewError(f"View bank has no view: {view_name}")
    entry = bank.views[view_name]
    base_dir = bank_path.parent
    source_path = resolve_existing_path(bank.character.source_reference.path, base_dir)
    image_path = resolve_existing_path(entry.image.path, base_dir)
    prompt = _training_validation_prompt(bank.character.id, view_name, entry.view.model_dump(mode="json"))
    with closing(QwenVlm(config)) as judge:
        raw_text = judge.judge_candidate(prompt, [source_path, image_path])
    validation = _training_validation(raw_text, config)
    entry.training_validation = validation
    write_json(bank_path, bank.model_dump(mode="json", exclude_none=True))
    return {
        "status": "validated",
        "character": bank.character.id,
        "view": view_name,
        "bank_path": bank_path.resolve().as_posix(),
        "training_validation": validation.model_dump(mode="json"),
    }


def _training_validation(raw_text: str, config: QwenVlmConfig) -> ViewBankTrainingValidationSpec:
    try:
        payload = TrainingValidationPayload.model_validate(json_object_from_vlm_response(raw_text))
        return ViewBankTrainingValidationSpec.model_validate(
            {
                "validator": config.judge_id,
                "model": config.model.as_posix(),
                **payload.model_dump(mode="json"),
            }
        )
    except (VlmJsonError, ValidationError) as error:
        raise CharacterViewError(f"Invalid character-view training validation JSON: {error}") from error


def _training_validation_prompt(character_id: str, view_name: str, view: dict[str, Any]) -> str:
    return f"""You are a strict visual QA judge for character LoRA training images.

You receive two images in order:
1. The canonical source reference for character {character_id}.
2. The candidate canonical view-bank image named {view_name}.

Judge whether image 2 is safe to use as an identity LoRA training image for the same character.
The image must teach stable character identity and match the required canonical view metadata.

Required view metadata:
{view}

Hard reject if:
- the candidate looks like a different character than the source reference;
- hairstyle, clothing, footwear, colors, or art style no longer match the character;
- the subject is malformed, has broken anatomy, broken hands, broken face, or severe limb errors;
- the background is busy, colored, leaking control artifacts, or not clean enough for identity training;
- the image has obvious artifacts, low quality, blur, pixelated leakage, duplicated body parts, or crop problems;
- the candidate does not match the required canonical view metadata.

Return valid JSON only. The JSON object must contain:
- usable_for_lora_training: boolean
- hard_rejects: object with booleans for identity_mismatch, outfit_mismatch, hairstyle_mismatch, malformed_subject, bad_background, low_image_quality
- scores: object with numeric 0-to-10 values for identity_preservation, outfit_preservation, hairstyle_preservation, anatomy_quality, background_quality, style_consistency, overall
- evidence: object with identity string, quality string, concerns string array

Score guidance:
- 9-10 means excellent training image.
- 7-8 means usable with minor concerns.
- below 7 should usually not be used for LoRA identity training.
- usable_for_lora_training must be false if any hard_reject is true or overall is below 7.

Do not wrap the JSON in Markdown code fences.
"""
