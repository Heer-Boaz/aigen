from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from aigen.character_qwen_edit import plan_qwen_character_edit
from aigen.character_reference_pack import build_character_reference_pack
from aigen.cli import build_parser
from aigen.progress import SILENT_STATUS
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

            with patch("aigen.character_qwen_edit.QwenVlm", FakeEditPlanner):
                plan = plan_qwen_character_edit(
                    pack_path=pack_path,
                    vlm_config=vlm_config(root),
                    cases=["right_profile"],
                    instruction="right side profile",
                    candidates_per_case=2,
                    progress=SILENT_STATUS,
            )

            case = plan["cases"][0]
            self.assertEqual(case["refs_used"], ["asset_a", "asset_b"])
            self.assertEqual(case["prompt"], "VLM-authored right profile instruction")
            runner = FakeEditPlanner.last
            self.assertTrue(runner.closed)
            self.assertEqual(len(runner.image_paths[0]), 3)


if __name__ == "__main__":
    unittest.main()
