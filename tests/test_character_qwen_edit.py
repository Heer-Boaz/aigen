from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from aigen.character_instruction_models import CharacterInstructionPlanSpec, InstructionEnvelopeSpec
from aigen.character_instruction_parser import CharacterInstructionParserConfig
from aigen.character_qwen_edit import plan_qwen_character_edit
from aigen.character_reference_pack import build_character_reference_pack
from aigen.cli import build_parser
from aigen.progress import SILENT_STATUS
from aigen.text_llm import TextLlmConfig
from aigen.vlm_qwen import QwenVlmConfig


def write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def vlm_config(root: Path) -> QwenVlmConfig:
    return QwenVlmConfig(
        judge_id="qwen2.5-vl-7b",
        model=root / "models" / "qwen",
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        revision="test-revision",
        dtype="bfloat16",
        attention_impl="sdpa",
        quantization="bitsandbytes-8bit",
        min_pixels=1,
        max_pixels=1024,
        max_new_tokens=600,
        temperature=0.0,
    )


def instruction_parser_config() -> CharacterInstructionParserConfig:
    return CharacterInstructionParserConfig(
        text_llm=TextLlmConfig(
            parser_id="fake-text-parser",
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            model="Qwen/Qwen3-8B",
            server_family="vllm",
            api_key_env="AIGEN_TEST_KEY",
            timeout_seconds=1.0,
            max_new_tokens=700,
            temperature=0.0,
            structured_output="json_object",
            enable_thinking=False,
        )
    )


class FakeInstructionParser:
    last: FakeInstructionParser

    def __init__(self, config: CharacterInstructionParserConfig) -> None:
        type(self).last = self
        self.config = config
        self.envelopes: list[InstructionEnvelopeSpec] = []

    def parse(self, envelope: InstructionEnvelopeSpec) -> CharacterInstructionPlanSpec:
        self.envelopes.append(envelope)
        return CharacterInstructionPlanSpec(
            kind="character-instruction-plan",
            raw_instruction=envelope.raw_instruction,
            normalized_instruction_text=envelope.raw_instruction,
            envelope=envelope,
            language="en",
            task_family="reference_character_portrait",
            edit_scope="global",
            subject_binding={
                "kind": "referenced_character",
                "reference_mentions": ["referenced images"],
                "note": "use referenced character",
            },
            target_constraints={
                "framing": ["right side profile"],
                "camera_view": [],
                "pose": [],
                "gaze": [],
                "expression": [],
                "lighting": [],
                "background": [],
                "explicit_style_or_role": [],
                "scene": [],
                "action": [],
                "mood_or_personality": [],
                "composition": [],
                "text_or_logo": [],
            },
            named_external_concepts=[],
            downstream_requirements=["visual_identity_analysis", "multi_reference_alignment"],
            ambiguities=[],
            conflicts=[],
            parser={"id": "fake-text-parser"},
            raw_model_response={},
        )


class FakeEditPlanner:
    last: FakeEditPlanner
    response: dict[str, object] = {
        "selected_refs": ["reference1", "reference2"],
        "reference_semantics": {"reference1": "VLM semantic label", "reference2": "VLM semantic label"},
        "edit_instruction": "VLM-authored right profile instruction",
    }

    def __init__(self, _config: QwenVlmConfig) -> None:
        type(self).last = self
        self.prompts: list[str] = []
        self.image_paths: list[list[Path]] = []
        self.device_report = {"all": [{"device": "cuda:0"}]}
        self.closed = False

    def describe_image(self, prompt: str, image_paths: list[Path]) -> str:
        self.prompts.append(prompt)
        self.image_paths.append(image_paths)
        return json.dumps(type(self).response)

    def close(self) -> None:
        self.closed = True


class CharacterQwenEditTests(unittest.TestCase):
    def test_qwen_edit_run_cli_accepts_plan_model_flag(self) -> None:
        args = build_parser().parse_args(
            [
                "characters",
                "qwen-edit-run",
                "--pack",
                "references/reference_pack.json",
                "--case",
                "right_profile",
                "--model",
                "nunchaku-qwen-edit-2509-r32-4step",
                "--output-dir",
                "runs/right_profile",
            ]
        )

        self.assertEqual(args.profile, "nunchaku-qwen-edit-2509-r32-4step")

    def test_qwen_edit_plan_uses_vlm_selected_refs_without_identity_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs = {
                "asset_a": root / "reference-a.png",
                "asset_b": root / "reference-b.png",
                "asset_c": root / "reference-c.png",
            }
            write_image(refs["asset_a"], (80, 144), (50, 50, 200))
            write_image(refs["asset_b"], (96, 96), (50, 200, 50))
            write_image(refs["asset_c"], (96, 144), (200, 50, 50))
            build_character_reference_pack(
                character_id="subject",
                references=refs,
                output_dir=root / "references",
                overwrite=False,
            )
            pack_path = root / "references" / "reference_pack.json"

            with (
                patch("aigen.character_qwen_edit.CharacterInstructionParser", FakeInstructionParser),
                patch("aigen.character_qwen_edit.QwenVlm", FakeEditPlanner),
            ):
                plan = plan_qwen_character_edit(
                    pack_path=pack_path,
                    instruction_parser_config=instruction_parser_config(),
                    vlm_config=vlm_config(root),
                    cases=["right_profile"],
                    instruction="right side profile",
                    candidates_per_case=2,
                    progress=SILENT_STATUS,
            )

            case = plan["cases"][0]
            self.assertEqual(case["refs_used"], ["asset_a", "asset_b"])
            self.assertEqual(case["prompt"], "VLM-authored right profile instruction")
            self.assertEqual(case["normalized_instruction"]["instruction_plan"]["subject_binding"]["kind"], "referenced_character")
            self.assertEqual(
                case["normalized_instruction"]["task_route_plan"]["route_kind"],
                "portrait_identity_generation",
            )
            self.assertIn(
                "visual_identity_analysis",
                case["normalized_instruction"]["instruction_plan"]["downstream_requirements"],
            )
            self.assertEqual(FakeInstructionParser.last.envelopes[0].reference_count, 3)
            runner = FakeEditPlanner.last
            self.assertTrue(runner.closed)
            self.assertEqual(len(runner.image_paths[0]), 3)
            self.assertIn("Planner context before image analysis", runner.prompts[0])
            self.assertIn("portrait_identity_generation", runner.prompts[0])
            self.assertNotIn("raw_model_response", runner.prompts[0])
            self.assertNotIn("endpoint", runner.prompts[0])


if __name__ == "__main__":
    unittest.main()
