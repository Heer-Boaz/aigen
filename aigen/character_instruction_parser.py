from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from aigen.character_instruction_models import (
    CHARACTER_INSTRUCTION_EDIT_SCOPES,
    CHARACTER_INSTRUCTION_PLAN_KIND,
    CHARACTER_INSTRUCTION_TASK_FAMILIES,
    CharacterInstructionError,
    CharacterInstructionModelResponseSpec,
    CharacterInstructionPlanSpec,
    InstructionEnvelopeSpec,
    character_instruction_model_response_schema,
    load_character_instruction_model_response,
)
from aigen.text_llm import TextLlmConfig, text_llm_config_json, text_llm_runner
from aigen.vlm_json import VlmJsonError, json_object_from_vlm_response


class TextJsonRunner(Protocol):
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> str:
        pass


@dataclass(frozen=True)
class CharacterInstructionParserConfig:
    text_llm: TextLlmConfig


def character_instruction_parser_config_json(config: CharacterInstructionParserConfig) -> dict[str, Any]:
    return text_llm_config_json(config.text_llm)


class CharacterInstructionParser:
    def __init__(
        self,
        config: CharacterInstructionParserConfig,
        *,
        runner: TextJsonRunner | None = None,
    ) -> None:
        self.config = config
        self.runner = runner if runner is not None else text_llm_runner(config.text_llm)

    def parse(self, envelope: InstructionEnvelopeSpec) -> CharacterInstructionPlanSpec:
        normalized_text = normalize_instruction_text(envelope.raw_instruction)
        if not normalized_text:
            raise CharacterInstructionError("character instruction parser requires a non-empty instruction")
        normalized_envelope = envelope.model_copy(update={"raw_instruction": normalized_text})
        raw_text = self.runner.generate_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_parser_prompt(normalized_envelope),
            schema_name="character_instruction_model_response",
            schema=character_instruction_model_response_schema(),
        )
        response_data = _response_data_from_raw(raw_text)
        response = load_character_instruction_model_response(
            response_data,
            path_label="character instruction parser",
        )
        return _canonical_plan(
            envelope=normalized_envelope,
            normalized_text=normalized_text,
            response=response,
            raw_model_response=raw_text,
            parser=character_instruction_parser_config_json(self.config),
        )


def normalize_instruction_text(text: str) -> str:
    translated = text.translate(
        {
            ord("\u2018"): "'",
            ord("\u2019"): "'",
            ord("\u201c"): '"',
            ord("\u201d"): '"',
            ord("\u00a0"): " ",
        }
    )
    translated = translated.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.strip() for line in translated.strip().split("\n")).strip()


def _response_data_from_raw(raw_text: str) -> dict[str, Any]:
    try:
        return json_object_from_vlm_response(raw_text)
    except VlmJsonError as error:
        raise CharacterInstructionError(f"Invalid character instruction parser response: {error}") from error


def _canonical_plan(
    *,
    envelope: InstructionEnvelopeSpec,
    normalized_text: str,
    response: CharacterInstructionModelResponseSpec,
    raw_model_response: str,
    parser: dict[str, Any],
) -> CharacterInstructionPlanSpec:
    task_family = _canonical_task_family(response.task_family, normalized_text)
    edit_scope = _canonical_label(
        response.edit_scope,
        _EDIT_SCOPE_ALIASES,
        allowed=CHARACTER_INSTRUCTION_EDIT_SCOPES,
        default="unknown",
    )
    return CharacterInstructionPlanSpec(
        kind=CHARACTER_INSTRUCTION_PLAN_KIND,
        raw_instruction=envelope.raw_instruction,
        normalized_instruction_text=normalized_text,
        envelope=envelope,
        task_family=task_family,
        edit_scope=edit_scope,
        target_constraints=response.target_constraints,
        parser=parser,
        raw_model_response=raw_model_response,
    )


def _canonical_task_family(raw_value: str, normalized_text: str) -> str:
    canonical = _canonical_label(
        raw_value,
        _TASK_FAMILY_ALIASES,
        allowed=CHARACTER_INSTRUCTION_TASK_FAMILIES,
        default="unknown",
    )
    text = normalized_text.lower()
    if "close-up" in text or "close up" in text or "portrait" in text:
        if _mentions_reference_binding(text):
            return "reference_character_portrait"
    if canonical == "unknown" and _mentions_reference_binding(text):
        return "view_change"
    return canonical


def _parser_prompt(envelope: InstructionEnvelopeSpec) -> str:
    envelope_json = json.dumps(envelope.model_dump(mode="json"), indent=2, sort_keys=True)
    return f"""Read this user instruction and UI/request context as step 1 of a reference-conditioned image pipeline.

Request envelope:
{envelope_json}

Extract only what is written in the user text plus what follows from the request context.
Represent "as shown in referenced images", "same character", and similar phrases as bindings to the supplied references.
Do not inspect or describe the reference images in this step.
Reference images go directly to the image editor; do not serialize them or schedule a visual-analysis stage.

Return exactly one JSON object with this shape:
{{
  "task_family": "broad task label",
  "edit_scope": "global, local, mixed, or unknown",
  "target_constraints": {{
    "framing": ["user-written framing requests"],
    "camera_view": ["user-written camera/view requests"],
    "pose": ["user-written pose requests"],
    "gaze": ["user-written gaze requests"],
    "expression": ["user-written expression requests"],
    "lighting": ["user-written lighting requests"],
    "background": ["user-written background requests"],
    "explicit_style_or_role": ["user-written style, role, genre, or archetype requests"],
    "scene": ["user-written scene/location requests"],
    "action": ["user-written action requests"],
    "mood_or_personality": ["user-written tone or personality direction"],
    "composition": ["user-written layout/composition requests"],
    "text_or_logo": ["user-written text, label, logo, or typography requests"]
  }}
}}

Rules:
- Use broad labels; deterministic code will canonicalize them.
- Keep user-written visual, style, role, scene, action, and mood requests.
- Extract no facts from the reference images.
- Use plain JSON only. No Markdown.
"""


def _canonical_label(
    raw_value: str,
    aliases: dict[str, str],
    *,
    allowed: tuple[str, ...],
    default: str,
) -> str:
    label = _label_key(raw_value)
    canonical = aliases.get(label, label)
    if canonical in allowed:
        return canonical
    return default


def _label_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return re.sub(r"_+", "_", key)


def _mentions_reference_binding(text: str) -> bool:
    return any(pattern.search(text) for pattern in _REFERENCE_BINDING_PATTERNS)


_SYSTEM_PROMPT = """You are an instruction parser for a reference-conditioned image editor.
Return compact JSON for routing. Do not produce hidden reasoning."""

_TASK_FAMILY_ALIASES = {
    "portrait": "reference_character_portrait",
    "close_up": "reference_character_portrait",
    "closeup": "reference_character_portrait",
    "close_up_portrait": "reference_character_portrait",
    "face_closeup": "reference_character_portrait",
    "reference_character_closeup": "reference_character_portrait",
    "reference_character_close_up": "reference_character_portrait",
    "full_body": "reference_character_full_body",
    "character_full_body": "reference_character_full_body",
    "view": "view_change",
    "camera_view": "view_change",
    "pose": "pose_transfer",
    "pose_change": "pose_transfer",
    "repair": "local_repair",
    "local_edit": "local_repair",
    "inpaint": "local_repair",
    "clothing_swap": "outfit_swap",
    "style": "style_transfer",
    "scene": "scene_insertion",
    "scene_generation": "scene_insertion",
    "layout": "layout_or_sheet",
    "sheet": "layout_or_sheet",
    "text": "text_or_label_heavy",
    "label": "text_or_label_heavy",
}

_EDIT_SCOPE_ALIASES = {
    "global_edit": "global",
    "whole_image": "global",
    "new_generation": "global",
    "local_edit": "local",
    "regional": "local",
    "inpaint": "local",
    "hybrid": "mixed",
}

_REFERENCE_BINDING_PATTERNS = (
    re.compile(r"\breferenced? images?\b"),
    re.compile(r"\breference images?\b"),
    re.compile(r"\bsame character\b"),
    re.compile(r"\bas shown\b"),
    re.compile(r"\bshown in (the )?refs?\b"),
)
