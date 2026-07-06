from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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

            plan = plan_qwen_character_edit(
                pack_path=pack_path,
                identity_profile_path=None,
                cases=["right_profile"],
                instruction="same character",
                candidates_per_case=2,
                progress=SILENT_STATUS,
            )

            case = plan["cases"][0]
            self.assertNotIn("reference_analysis", plan)
            self.assertEqual(case["refs_used"], ["image_a", "image_b", "image_c"])
            normalized_instruction = case["normalized_instruction"]
            self.assertEqual(normalized_instruction["required_roles"], ["side", "portrait", "front"])
            self.assertEqual(normalized_instruction["view_anchor_role"], "side")
            self.assertEqual(normalized_instruction["view_anchor_ref"], "image_a")
            self.assertEqual(normalized_instruction["view_anchor_input_index"], 1)
            self.assertEqual(
                normalized_instruction["reference_purposes"][0],
                {
                    "role": "side",
                    "reference_id": "image_a",
                    "input_index": 1,
                    "purpose": "primary camera angle and side-body geometry anchor",
                },
            )
            self.assertIn("strict right side profile", normalized_instruction["view_constraints"])
            self.assertIn(
                "entire head-to-toe character remains fully visible with a small background margin",
                normalized_instruction["view_constraints"],
            )
            self.assertEqual(case["body_proportion_source"], CHARACTER_BODY_PROPORTION_SOURCE)
            self.assertEqual(case["optional_missing_refs"], ["body_shape"])
            self.assertEqual(
                normalized_instruction["body_proportion"]["do_not_change"],
                ["model-extracted body invariant"],
            )
            self.assertIn("model-extracted body invariant", case["prompt"])
            self.assertIn("Camera/view anchor: input image 1 (image_a, role side)", case["prompt"])
            self.assertIn("entire head-to-toe character remains fully visible", case["prompt"])
            self.assertIn("do not produce a front-facing or three-quarter view", case["prompt"])

    def test_qwen_edit_plan_rejects_duplicate_inferred_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs = {
                "image_a": root / "side-a.png",
                "image_b": root / "side-b.png",
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
                    "image_b": "side",
                    "image_c": "front",
                },
                optional_missing_refs=["body_shape"],
            )

            with self.assertRaises(QwenCharacterEditError):
                plan_qwen_character_edit(
                    pack_path=pack_path,
                    identity_profile_path=None,
                    cases=["right_profile"],
                    instruction=None,
                    candidates_per_case=2,
                    progress=SILENT_STATUS,
                )

    def test_qwen_edit_canvas_uses_view_anchor_reference_aspect_ratio(self) -> None:
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
