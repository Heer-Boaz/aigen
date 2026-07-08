from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aigen.character_instruction_models import CharacterInstructionError, InstructionEnvelopeSpec
from aigen.character_instruction_parser import CharacterInstructionParser, CharacterInstructionParserConfig
from aigen.text_llm import TextLlmConfig


def parser_config() -> CharacterInstructionParserConfig:
    model = Path(tempfile.gettempdir()) / "fake-qwen3-model"
    return CharacterInstructionParserConfig(
        text_llm=TextLlmConfig(
            parser_id="fake-text-parser",
            model=model,
            dtype="bfloat16",
            quantization="bitsandbytes-8bit",
            max_new_tokens=700,
            temperature=0.0,
            enable_thinking=False,
        )
    )


def envelope(raw_instruction: str, *, mask_present: bool = False) -> InstructionEnvelopeSpec:
    return InstructionEnvelopeSpec(
        raw_instruction=raw_instruction,
        ui_mode="reference_conditioned_generation",
        reference_count=3,
        source_image_present=False,
        mask_present=mask_present,
        region_plan_present=False,
        generation_panel_settings={"case": "portrait"},
        requested_model_family="qwen-image-edit",
    )


class FakeTextRunner:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate_json(self, *, system_prompt: str, user_prompt: str, schema_name: str, schema: dict[str, object]) -> str:
        self.prompts.append(user_prompt)
        return json.dumps(self.response)


def base_response(**overrides: object) -> dict[str, object]:
    response: dict[str, object] = {
        "language": "en",
        "task_family": "close-up portrait",
        "edit_scope": "whole image",
        "subject_binding": {
            "kind": "references",
            "reference_mentions": ["referenced images"],
            "note": "use the referenced character",
        },
        "target_constraints": {
            "framing": ["close-up face"],
            "camera_view": [],
            "pose": [],
            "gaze": ["looking left"],
            "expression": ["neutral expression"],
            "lighting": ["neutral lighting"],
            "background": ["white background"],
            "explicit_style_or_role": [],
            "scene": [],
            "action": [],
            "mood_or_personality": [],
            "composition": [],
            "text_or_logo": [],
        },
        "named_external_concepts": [],
        "downstream_requirements": [],
        "ambiguities": [],
        "conflicts": [],
    }
    response.update(overrides)
    return response


class CharacterInstructionParserTests(unittest.TestCase):
    def test_reference_portrait_instruction_requires_visual_analysis_without_identity_facts(self) -> None:
        runner = FakeTextRunner(base_response())
        plan = CharacterInstructionParser(parser_config(), runner=runner).parse(
            envelope(
                "Character as shown in referenced images.\n"
                "Close-up face, looking to the left with neutral expression.\n"
                "Neutral lighting, white background."
            )
        )

        self.assertEqual(plan.task_family, "reference_character_portrait")
        self.assertEqual(plan.subject_binding.kind, "referenced_character")
        self.assertIn("visual_identity_analysis", plan.downstream_requirements)
        self.assertIn("multi_reference_alignment", plan.downstream_requirements)
        self.assertEqual(plan.target_constraints.gaze, ["looking left"])

    def test_user_written_role_scene_action_and_external_concept_survive_step1(self) -> None:
        response = base_response(
            task_family="scene",
            target_constraints={
                "framing": [],
                "camera_view": [],
                "pose": [],
                "gaze": [],
                "expression": [],
                "lighting": [],
                "background": [],
                "explicit_style_or_role": ["noir detective"],
                "scene": ["cafe in Paris"],
                "action": ["drinking coffee"],
                "mood_or_personality": ["heroic", "warm", "protective"],
                "composition": [],
                "text_or_logo": [],
            },
            named_external_concepts=["Konami's Snatcher"],
        )
        plan = CharacterInstructionParser(parser_config(), runner=FakeTextRunner(response)).parse(
            envelope(
                "Character as shown in referenced images, who is a noir detective based on Konami's Snatcher. "
                "Show her drinking coffee in a cafe in Paris. "
                "she is heroic and has a warm and protective personality."
            )
        )

        self.assertEqual(plan.task_family, "scene_insertion")
        self.assertEqual(plan.target_constraints.explicit_style_or_role, ["noir detective"])
        self.assertEqual(plan.target_constraints.action, ["drinking coffee"])
        self.assertEqual(plan.named_external_concepts, ["Konami's Snatcher"])
        self.assertIn("external_concept_resolution", plan.downstream_requirements)
        self.assertIn("visual_identity_analysis", plan.downstream_requirements)

    def test_reference_conditioned_ui_binds_default_case_to_references(self) -> None:
        response = base_response(
            subject_binding={
                "kind": "unspecified",
                "reference_mentions": [],
                "note": "",
            }
        )
        plan = CharacterInstructionParser(parser_config(), runner=FakeTextRunner(response)).parse(envelope("portrait"))

        self.assertEqual(plan.subject_binding.kind, "referenced_character")
        self.assertIn("visual_identity_analysis", plan.downstream_requirements)

    def test_local_repair_without_mask_marks_region_and_mask_requirements(self) -> None:
        response = base_response(
            task_family="repair",
            edit_scope="local edit",
            target_constraints={
                "framing": [],
                "camera_view": [],
                "pose": [],
                "gaze": [],
                "expression": [],
                "lighting": [],
                "background": [],
                "explicit_style_or_role": [],
                "scene": [],
                "action": ["fix the hand"],
                "mood_or_personality": [],
                "composition": [],
                "text_or_logo": [],
            },
        )
        plan = CharacterInstructionParser(parser_config(), runner=FakeTextRunner(response)).parse(
            envelope("Character as shown in referenced images. Fix the hand.")
        )

        self.assertEqual(plan.edit_scope, "local")
        self.assertIn("region_grounding", plan.downstream_requirements)
        self.assertIn("mask_generation", plan.downstream_requirements)

    def test_step1_rejects_unwritten_visual_identity_leakage(self) -> None:
        response = base_response(
            target_constraints={
                "framing": ["portrait"],
                "camera_view": [],
                "pose": [],
                "gaze": [],
                "expression": [],
                "lighting": [],
                "background": [],
                "explicit_style_or_role": ["short hair"],
                "scene": [],
                "action": [],
                "mood_or_personality": [],
                "composition": [],
                "text_or_logo": [],
            }
        )

        with self.assertRaisesRegex(CharacterInstructionError, "visual identity"):
            CharacterInstructionParser(parser_config(), runner=FakeTextRunner(response)).parse(
                envelope("Character as shown in referenced images. Close-up portrait.")
            )


if __name__ == "__main__":
    unittest.main()
