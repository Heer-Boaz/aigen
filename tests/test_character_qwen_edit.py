from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from aigen.character_qwen_edit import QwenCharacterEditError, plan_qwen_character_edit
from aigen.character_reference_models import CHARACTER_BODY_PROPORTION_SOURCE
from aigen.character_reference_pack import build_character_reference_pack
from aigen.cli import build_parser
from aigen.generation.qwen_image_edit_identity import (
    QwenIdentityCase,
    QwenIdentityPromptConditioningStep,
    QwenIdentityReferenceStep,
    _run_qwen_identity_denoise_step,
)
from aigen.progress import SILENT_STATUS
from aigen.vlm_qwen import QwenVlmConfig


def write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def body_proportion_payload(*, evidence_refs: list[str]) -> dict[str, object]:
    return {
        "chest_size": "model-extracted chest-size fact",
        "build": "model-extracted build fact",
        "shoulder_width": "model-extracted shoulder-width fact",
        "waist": "model-extracted waist fact",
        "hip_skirt_silhouette": "model-extracted hip/skirt silhouette fact",
        "side_body_thickness": "model-extracted side-body-thickness fact",
        "leg_proportion": "model-extracted leg-proportion fact",
        "skirt_back_shape": "model-extracted back-shape fact",
        "do_not_change": ["model-extracted body invariant"],
        "evidence_refs": evidence_refs,
    }


def write_identity_profile(
    path: Path,
    *,
    pack_path: Path,
    reference_roles: dict[str, str],
    optional_missing_refs: list[str],
) -> None:
    payload = {
        "status": "completed",
        "kind": "character-identity-profile",
        "character_id": "subject",
        "source_reference_pack": pack_path.as_posix(),
        "identity": {
            "hair": "model-extracted hair fact",
            "eyes": "model-extracted eye fact",
            "style": "model-extracted style fact",
        },
        "body_proportion": body_proportion_payload(evidence_refs=list(reference_roles)),
        "body_proportion_source": CHARACTER_BODY_PROPORTION_SOURCE,
        "reference_roles": reference_roles,
        "optional_missing_refs": optional_missing_refs,
        "must_preserve": ["model-extracted visual invariant"],
        "avoid": ["model-extracted visual drift"],
        "parser": {"id": "fake-parser"},
        "output": {"identity_profile": path.as_posix()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


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


class FakeEditInstructionNormalizer:
    last: FakeEditInstructionNormalizer
    response: dict[str, object] = {
        "selected_refs": ["image_a", "image_b"],
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

    def test_qwen_edit_plan_uses_identity_profile_body_proportion_without_reference_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs = {
                "image_a": root / "side.png",
                "image_b": root / "portrait.png",
                "image_c": root / "front.png",
            }
            write_image(refs["image_a"], (80, 144), (50, 50, 200))
            write_image(refs["image_b"], (96, 96), (50, 200, 50))
            write_image(refs["image_c"], (96, 144), (200, 50, 50))
            build_character_reference_pack(
                character_id="subject",
                references=refs,
                output_dir=root / "references",
                overwrite=False,
            )
            pack_path = root / "references" / "reference_pack.json"
            identity_profile_path = root / "references" / "identity_profile.json"
            write_identity_profile(
                identity_profile_path,
                pack_path=pack_path,
                reference_roles={
                    "image_a": "side",
                    "image_b": "portrait",
                    "image_c": "front",
                },
                optional_missing_refs=["body_shape"],
            )

            with patch("aigen.character_qwen_edit.QwenVlm", FakeEditInstructionNormalizer):
                plan = plan_qwen_character_edit(
                    pack_path=pack_path,
                    identity_profile_path=None,
                    vlm_config=vlm_config(root),
                    cases=["right_profile"],
                    instruction="right side profile",
                    candidates_per_case=2,
                    progress=SILENT_STATUS,
                )

            case = plan["cases"][0]
            self.assertNotIn("reference_analysis", plan)
            self.assertEqual(case["refs_used"], ["image_a", "image_b"])
            normalized_instruction = case["normalized_instruction"]
            self.assertCountEqual(normalized_instruction["planner_input_refs"], ["image_a", "image_b", "image_c"])
            self.assertEqual(normalized_instruction["refs_used"], ["image_a", "image_b"])
            self.assertEqual(normalized_instruction["instruction_request"], "right side profile")
            self.assertEqual(normalized_instruction["edit_instruction_source"], "qwen_vlm_edit_planner")
            self.assertEqual(normalized_instruction["edit_instruction"], "VLM-authored right profile instruction")
            self.assertEqual(
                json.loads(normalized_instruction["edit_planner_raw_response"]),
                {
                    "selected_refs": ["image_a", "image_b"],
                    "edit_instruction": "VLM-authored right profile instruction",
                },
            )
            self.assertNotIn("reference_purposes", normalized_instruction)
            self.assertNotIn("view_constraints", normalized_instruction)
            self.assertNotIn("camera", normalized_instruction)
            self.assertNotIn("anchor_ref", normalized_instruction)
            self.assertNotIn("support_refs", normalized_instruction)
            self.assertEqual(case["body_proportion_source"], CHARACTER_BODY_PROPORTION_SOURCE)
            self.assertEqual(case["optional_missing_refs"], ["body_shape"])
            self.assertEqual(
                normalized_instruction["body_proportion"]["do_not_change"],
                ["model-extracted body invariant"],
            )
            self.assertEqual(case["prompt"], "VLM-authored right profile instruction")
            runner = FakeEditInstructionNormalizer.last
            self.assertTrue(runner.closed)
            self.assertEqual(len(runner.image_paths), 1)
            self.assertCountEqual(
                [path.name for path in runner.image_paths[0]],
                ["side.png", "portrait.png", "front.png"],
            )
            self.assertIn("The user's requested edit is exactly: 'right side profile'.", runner.prompts[0])
            self.assertIn("Available reference ids are exactly: image_a, image_b, image_c.", runner.prompts[0])

    def test_qwen_edit_plan_rejects_unknown_vlm_selected_ref(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs = {
                "image_a": root / "side.png",
                "image_b": root / "portrait.png",
                "image_c": root / "front.png",
            }
            write_image(refs["image_a"], (80, 144), (50, 50, 200))
            write_image(refs["image_b"], (96, 96), (50, 200, 50))
            write_image(refs["image_c"], (96, 144), (200, 50, 50))
            build_character_reference_pack(
                character_id="subject",
                references=refs,
                output_dir=root / "references",
                overwrite=False,
            )
            pack_path = root / "references" / "reference_pack.json"
            write_identity_profile(
                root / "references" / "identity_profile.json",
                pack_path=pack_path,
                reference_roles={
                    "image_a": "side",
                    "image_b": "portrait",
                    "image_c": "front",
                },
                optional_missing_refs=["body_shape"],
            )

            previous = FakeEditInstructionNormalizer.response
            FakeEditInstructionNormalizer.response = {
                "selected_refs": ["missing"],
                "edit_instruction": "bad plan",
            }
            try:
                with patch("aigen.character_qwen_edit.QwenVlm", FakeEditInstructionNormalizer):
                    with self.assertRaises(QwenCharacterEditError):
                        plan_qwen_character_edit(
                            pack_path=pack_path,
                            identity_profile_path=None,
                            vlm_config=vlm_config(root),
                            cases=["right_profile"],
                            instruction=None,
                            candidates_per_case=2,
                            progress=SILENT_STATUS,
                        )
            finally:
                FakeEditInstructionNormalizer.response = previous

    def test_qwen_edit_canvas_uses_first_selected_reference_aspect_ratio(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class FakeTorch:
            cuda = FakeCuda()

        class FakeSession:
            def __init__(self) -> None:
                self.torch = FakeTorch()
                self.calls: list[dict[str, object]] = []
                self.released_for_decode = False

            def denoise_to_latents(self, **kwargs):
                self.calls.append(dict(kwargs))
                return object(), {"denoise_ms": 1.0}

            def release_denoise_models_for_decode(self, _progress) -> None:
                self.released_for_decode = True

            def decode_latents(self, _latents, **kwargs) -> tuple[Image.Image, float]:
                return Image.new("RGB", (kwargs["width"], kwargs["height"]), (1, 2, 3)), 2.0

        case = QwenIdentityCase(
            name="right_profile",
            references=("side_ref", "portrait_ref", "front_ref"),
            prompt="same character",
        )
        reference_step = QwenIdentityReferenceStep(
            cases=(case,),
            reference_images={
                "side_ref": Image.new("RGB", (480, 640), (10, 20, 30)),
                "portrait_ref": Image.new("RGB", (432, 640), (40, 50, 60)),
                "front_ref": Image.new("RGB", (416, 640), (70, 80, 90)),
            },
        )
        prompt_step = QwenIdentityPromptConditioningStep(embeddings={"right_profile": object()}, elapsed_ms=0)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = _run_qwen_identity_denoise_step(
                session=FakeSession(),
                reference_step=reference_step,
                prompt_step=prompt_step,
                images_dir=Path(temp_dir),
                max_side=640,
                steps=1,
                true_cfg_scale=1.0,
                guidance_scale=1.0,
                seed=0,
                max_sequence_length=512,
                candidates_per_case=1,
                progress=SILENT_STATUS,
            )

        self.assertEqual(result.outputs[0]["width"], 480)
        self.assertEqual(result.outputs[0]["height"], 640)
        self.assertEqual(result.outputs[0]["image"]["width"], 480)
        self.assertEqual(result.outputs[0]["image"]["height"], 640)


if __name__ == "__main__":
    unittest.main()
