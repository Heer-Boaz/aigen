from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from aigen.character_qwen_edit import QwenCharacterEditError, plan_qwen_character_edit
from aigen.character_reference_models import CHARACTER_BODY_PROPORTION_SOURCE
from aigen.character_reference_pack import build_character_reference_pack
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
            self.assertEqual(case["normalized_instruction"]["required_roles"], ["side", "portrait", "front"])
            self.assertEqual(case["body_proportion_source"], CHARACTER_BODY_PROPORTION_SOURCE)
            self.assertEqual(case["optional_missing_refs"], ["body_shape"])
            self.assertEqual(
                case["normalized_instruction"]["body_proportion"]["do_not_change"],
                ["model-extracted body invariant"],
            )
            self.assertIn("model-extracted body invariant", case["prompt"])

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


if __name__ == "__main__":
    unittest.main()
