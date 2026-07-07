from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from aigen.character_instruction_models import (
    CHARACTER_INSTRUCTION_DOWNSTREAM_REQUIREMENTS,
    CHARACTER_INSTRUCTION_EDIT_SCOPES,
    CHARACTER_INSTRUCTION_PLAN_KIND,
    CHARACTER_INSTRUCTION_SUBJECT_BINDINGS,
    CHARACTER_INSTRUCTION_TASK_FAMILIES,
    CharacterInstructionError,
    CharacterInstructionModelResponseSpec,
    CharacterInstructionPlanSpec,
    InstructionEnvelopeSpec,
    InstructionSubjectBindingSpec,
    InstructionTargetConstraintsSpec,
    character_instruction_model_response_schema,
    load_character_instruction_model_response,
)
from aigen.text_llm import OpenAICompatibleTextLlm, TextLlmConfig, text_llm_config_json
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
        self.runner = runner if runner is not None else OpenAICompatibleTextLlm(config.text_llm)

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
            response_data=response_data,
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
    response_data: dict[str, Any],
    parser: dict[str, Any],
) -> CharacterInstructionPlanSpec:
    task_family = _canonical_task_family(response.task_family, normalized_text)
    edit_scope = _canonical_label(
        response.edit_scope,
        _EDIT_SCOPE_ALIASES,
        allowed=CHARACTER_INSTRUCTION_EDIT_SCOPES,
        default="unknown",
    )
    subject_binding = _canonical_subject_binding(response.subject_binding, normalized_text, envelope)
    downstream_requirements = _canonical_downstream_requirements(
        response.downstream_requirements,
        task_family=task_family,
        edit_scope=edit_scope,
        subject_binding=subject_binding.kind,
        envelope=envelope,
        normalized_text=normalized_text,
        named_external_concepts=response.named_external_concepts,
    )
    _check_step1_identity_leakage(response.target_constraints, normalized_text)
    return CharacterInstructionPlanSpec(
        kind=CHARACTER_INSTRUCTION_PLAN_KIND,
        raw_instruction=envelope.raw_instruction,
        normalized_instruction_text=normalized_text,
        envelope=envelope,
        language=response.language.strip() or "unknown",
        task_family=task_family,
        edit_scope=edit_scope,
        subject_binding=subject_binding,
        target_constraints=response.target_constraints,
        named_external_concepts=_clean_list(response.named_external_concepts),
        downstream_requirements=downstream_requirements,
        ambiguities=_clean_list(response.ambiguities),
        conflicts=_clean_list(response.conflicts),
        parser=parser,
        raw_model_response=response_data,
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


def _canonical_subject_binding(
    subject_binding: InstructionSubjectBindingSpec,
    normalized_text: str,
    envelope: InstructionEnvelopeSpec,
) -> InstructionSubjectBindingSpec:
    kind = _canonical_label(
        subject_binding.kind,
        _SUBJECT_BINDING_ALIASES,
        allowed=CHARACTER_INSTRUCTION_SUBJECT_BINDINGS,
        default="unspecified",
    )
    text = normalized_text.lower()
    if _mentions_image_index(text):
        kind = "image_index_binding"
    elif _mentions_reference_binding(text) and envelope.reference_count > 0:
        kind = "referenced_character"
    elif (
        envelope.ui_mode == "reference_conditioned_generation"
        and envelope.reference_count > 0
        and kind == "unspecified"
    ):
        kind = "referenced_character"
    elif envelope.source_image_present and kind == "unspecified":
        kind = "source_image_subject"
    return InstructionSubjectBindingSpec(
        kind=kind,
        reference_mentions=_clean_list(subject_binding.reference_mentions),
        note=subject_binding.note.strip(),
    )


def _canonical_downstream_requirements(
    raw_requirements: list[str],
    *,
    task_family: str,
    edit_scope: str,
    subject_binding: str,
    envelope: InstructionEnvelopeSpec,
    normalized_text: str,
    named_external_concepts: list[str],
) -> list[str]:
    requirements = {
        _canonical_label(
            value,
            _DOWNSTREAM_REQUIREMENT_ALIASES,
            allowed=CHARACTER_INSTRUCTION_DOWNSTREAM_REQUIREMENTS,
            default="",
        )
        for value in raw_requirements
    }
    requirements.discard("")
    if subject_binding in {"referenced_character", "image_index_binding"}:
        requirements.add("visual_identity_analysis")
    if envelope.reference_count > 1 and subject_binding in {"referenced_character", "image_index_binding"}:
        requirements.add("multi_reference_alignment")
    if named_external_concepts:
        requirements.add("external_concept_resolution")
    if edit_scope == "local" or _mentions_local_repair(normalized_text):
        requirements.add("region_grounding")
        if not envelope.mask_present and not envelope.region_plan_present:
            requirements.add("mask_generation")
    if _mentions_pose_conditioning(normalized_text) or task_family == "pose_transfer":
        requirements.add("pose_conditioning")
    if _mentions_text_rendering(normalized_text) or task_family == "text_or_label_heavy":
        requirements.add("text_rendering_risk")
    if _mentions_visual_disambiguation(normalized_text):
        requirements.add("visual_disambiguation")
    return sorted(requirements, key=CHARACTER_INSTRUCTION_DOWNSTREAM_REQUIREMENTS.index)


def _check_step1_identity_leakage(
    constraints: InstructionTargetConstraintsSpec,
    normalized_text: str,
) -> None:
    generated = " ".join(
        value
        for values in constraints.model_dump(mode="json").values()
        for value in values
    ).lower()
    source = normalized_text.lower()
    leaked_terms = sorted(
        term
        for term in _VISUAL_IDENTITY_TERMS
        if term in generated and term not in source
    )
    if leaked_terms:
        raise CharacterInstructionError(
            "character instruction parser leaked visual identity term(s) before reference analysis: "
            + ", ".join(leaked_terms)
        )


def _parser_prompt(envelope: InstructionEnvelopeSpec) -> str:
    envelope_json = json.dumps(envelope.model_dump(mode="json"), indent=2, sort_keys=True)
    return f"""Read this user instruction and UI/request context as step 1 of a reference-conditioned image pipeline.

Request envelope:
{envelope_json}

Extract only what is written in the user text plus what follows from the request context.
Represent "as shown in referenced images", "same character", and similar phrases as bindings to the supplied references.
Do not inspect or describe the reference images in this step.
Do not add reference-derived identity facts. Later image-analysis stages own those facts.

Return exactly one JSON object with this shape:
{{
  "language": "short language label",
  "task_family": "broad task label",
  "edit_scope": "global, local, mixed, or unknown",
  "subject_binding": {{
    "kind": "referenced_character, image_index_binding, source_image_subject, multiple_subjects, or unspecified",
    "reference_mentions": ["explicit mentions such as Image 1 or referenced images"],
    "note": "short binding note"
  }},
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
  }},
  "named_external_concepts": ["explicit named external concepts"],
  "downstream_requirements": ["broad downstream requirements"],
  "ambiguities": ["real ambiguity that downstream must resolve"],
  "conflicts": ["real conflicts in the request"]
}}

Rules:
- Use broad labels; deterministic code will canonicalize them.
- Keep user-written visual, style, role, scene, action, and mood requests.
- If a reference image must be inspected to know a fact, mark visual identity analysis or visual disambiguation instead of writing the fact.
- Do not output identity, body, must_preserve, avoid, reference_roles, hair, eye, outfit, or body-proportion fields.
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


def _clean_list(values: list[str]) -> list[str]:
    return [value.strip() for value in values if value.strip()]


def _mentions_reference_binding(text: str) -> bool:
    return any(pattern.search(text) for pattern in _REFERENCE_BINDING_PATTERNS)


def _mentions_image_index(text: str) -> bool:
    return any(pattern.search(text) for pattern in _IMAGE_INDEX_PATTERNS)


def _mentions_local_repair(text: str) -> bool:
    return any(pattern.search(text) for pattern in _LOCAL_REPAIR_PATTERNS)


def _mentions_pose_conditioning(text: str) -> bool:
    return any(pattern.search(text) for pattern in _POSE_PATTERNS)


def _mentions_text_rendering(text: str) -> bool:
    return any(pattern.search(text) for pattern in _TEXT_RENDERING_PATTERNS)


def _mentions_visual_disambiguation(text: str) -> bool:
    return _mentions_image_index(text) or any(pattern.search(text) for pattern in _VISUAL_DISAMBIGUATION_PATTERNS)


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

_SUBJECT_BINDING_ALIASES = {
    "reference": "referenced_character",
    "references": "referenced_character",
    "referenced_images": "referenced_character",
    "referenced_character": "referenced_character",
    "same_character": "referenced_character",
    "image_index": "image_index_binding",
    "indexed_reference": "image_index_binding",
    "source_subject": "source_image_subject",
    "source_image": "source_image_subject",
    "multiple": "multiple_subjects",
}

_DOWNSTREAM_REQUIREMENT_ALIASES = {
    "identity_analysis": "visual_identity_analysis",
    "reference_analysis": "visual_identity_analysis",
    "visual_analysis": "visual_identity_analysis",
    "multi_ref_alignment": "multi_reference_alignment",
    "reference_alignment": "multi_reference_alignment",
    "region": "region_grounding",
    "mask": "mask_generation",
    "masking": "mask_generation",
    "pose": "pose_conditioning",
    "text": "text_rendering_risk",
    "external_concept": "external_concept_resolution",
    "concept_resolution": "external_concept_resolution",
    "clarification": "clarification_needed",
    "visual_disambiguation": "visual_disambiguation",
}

_REFERENCE_BINDING_PATTERNS = (
    re.compile(r"\breferenced? images?\b"),
    re.compile(r"\breference images?\b"),
    re.compile(r"\bsame character\b"),
    re.compile(r"\bas shown\b"),
    re.compile(r"\bshown in (the )?refs?\b"),
)

_IMAGE_INDEX_PATTERNS = (
    re.compile(r"\bimage\s*\d+\b"),
    re.compile(r"\bref(erence)?\s*\d+\b"),
    re.compile(r"\b(first|second|third|fourth)\s+(image|ref|reference)\b"),
)

_LOCAL_REPAIR_PATTERNS = (
    re.compile(r"\bfix\b"),
    re.compile(r"\brepair\b"),
    re.compile(r"\bcorrect\b"),
    re.compile(r"\binpaint\b"),
)

_POSE_PATTERNS = (
    re.compile(r"\bpose\b"),
    re.compile(r"\bkeypoint\b"),
    re.compile(r"\bopenpose\b"),
    re.compile(r"\bstanding\b"),
    re.compile(r"\bwalking\b"),
)

_TEXT_RENDERING_PATTERNS = (
    re.compile(r"\btext\b"),
    re.compile(r"\blogo\b"),
    re.compile(r"\blabel\b"),
    re.compile(r"\bspeech bubble\b"),
    re.compile(r"\btypography\b"),
)

_VISUAL_DISAMBIGUATION_PATTERNS = (
    re.compile(r"\bon the left\b"),
    re.compile(r"\bon the right\b"),
    re.compile(r"\bsmall image\b"),
    re.compile(r"\bcorner\b"),
    re.compile(r"\bpanel\b"),
)

_VISUAL_IDENTITY_TERMS = (
    "hair",
    "eyes",
    "eye color",
    "skin tone",
    "body proportion",
    "body shape",
    "chest",
    "waist",
    "hip",
    "silhouette",
)
