from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from aigen.character_reference_models import (
    CHARACTER_BODY_PROPORTION_SOURCE,
    CharacterReferenceError,
    load_completed_character_identity_profile,
    load_completed_character_reference_pack,
)
from aigen.character_reference_pack import (
    build_character_reference_pack,
    parse_character_edit_plan,
    parse_character_reference_args,
    parse_character_reference_pack,
)
from aigen.cli import build_parser
from aigen.progress import SILENT_STATUS
from aigen.vlm_qwen import QwenVlmConfig


def write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def parser_config(root: Path) -> QwenVlmConfig:
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
        max_new_tokens=1600,
        temperature=0.0,
    )


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


class FakeReferenceParser:
    last: FakeReferenceParser

    def __init__(self, _config: QwenVlmConfig) -> None:
        type(self).last = self
        self.prompt = ""
        self.image_paths: list[Path] = []
        self.device_report = {"all": [{"device": "cuda:0"}]}
        self.closed = False

    def describe_image(self, prompt: str, image_paths: list[Path]) -> str:
        self.prompt = prompt
        self.image_paths = image_paths
        return json.dumps(
            {
                "identity": {
                    "hair": "model-extracted hair fact",
                    "eyes": "model-extracted eye fact",
                    "style": "model-extracted style fact",
                },
                "body_proportion": body_proportion_payload(
                    evidence_refs=["ref_front", "ref_side"],
                ),
                "reference_roles": {
                    "ref_front": "front",
                    "ref_face": "portrait",
                    "ref_side": "side",
                },
                "must_preserve": ["model-extracted visual invariant"],
                "avoid": ["model-extracted visual drift"],
            }
        )

    def close(self) -> None:
        self.closed = True


class InvalidReferenceParser(FakeReferenceParser):
    def describe_image(self, prompt: str, image_paths: list[Path]) -> str:
        self.prompt = prompt
        self.image_paths = image_paths
        return json.dumps(
            {
                "identity": {"hair": "model-extracted hair fact"},
                "body_proportion": body_proportion_payload(evidence_refs=["missing_ref"]),
                "reference_roles": {"ref_front": "front"},
                "must_preserve": ["model-extracted visual invariant"],
                "avoid": [],
            }
        )


class FakeEditInstructionParser:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.prompt = ""
        self.image_paths: list[Path] = []

    def describe_image(self, prompt: str, image_paths: list[Path]) -> str:
        self.prompt = prompt
        self.image_paths = image_paths
        return json.dumps(self.response)


class CharacterReferencePackTests(unittest.TestCase):
    def test_character_cli_exposes_reference_pack_build_and_parse_only(self) -> None:
        build_args = build_parser().parse_args(
            [
                "characters",
                "reference-pack",
                "build",
                "--character-id",
                "subject",
                "--reference",
                "ref_front=front.png",
                "--output-dir",
                "assets/characters/subject/references",
            ]
        )
        parse_args = build_parser().parse_args(
            [
                "characters",
                "reference-pack",
                "parse",
                "assets/characters/subject/references/reference_pack.json",
            ]
        )

        self.assertEqual(build_args.reference_pack_command, "build")
        self.assertEqual(parse_args.reference_pack_command, "parse")
        self.assertEqual(parse_args.max_new_tokens, 1600)
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["characters", "reference-pack", "analyze", "pack.json"])

    def test_parse_reference_args_accepts_pack_local_ids_and_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_image(root / "front-a.png", (32, 48), (200, 50, 50))
            write_image(root / "front-b.png", (32, 48), (50, 200, 50))

            parsed = parse_character_reference_args(["ref_a=front-a.png"], root)
            self.assertEqual(set(parsed), {"ref_a"})
            with self.assertRaises(CharacterReferenceError):
                parse_character_reference_args(["ref_a=front-a.png", "ref_a=front-b.png"], root)

    def test_build_reference_pack_writes_assets_and_hints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs = {
                "ref_front": root / "front.png",
                "ref_face": root / "face.png",
                "ref_side": root / "side.png",
            }
            write_image(refs["ref_front"], (96, 144), (200, 50, 50))
            write_image(refs["ref_face"], (96, 96), (50, 200, 50))
            write_image(refs["ref_side"], (80, 144), (50, 50, 200))

            result = build_character_reference_pack(
                character_id="subject",
                references=refs,
                output_dir=root / "references",
                overwrite=False,
            )

            pack_path = root / "references" / "reference_pack.json"
            self.assertTrue(pack_path.exists())
            self.assertEqual(result["kind"], "character-reference-pack")
            self.assertEqual(list(result["references"]), ["ref_front", "ref_face", "ref_side"])
            self.assertEqual(result["reference_hints"], {name: name for name in refs})
            self.assertNotIn("reference_roles", result)

    def test_parse_reference_pack_writes_vlm_identity_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs = {
                "ref_front": root / "front.png",
                "ref_face": root / "face.png",
                "ref_side": root / "side.png",
            }
            write_image(refs["ref_front"], (96, 144), (200, 50, 50))
            write_image(refs["ref_face"], (96, 96), (50, 200, 50))
            write_image(refs["ref_side"], (80, 144), (50, 50, 200))
            build_character_reference_pack(
                character_id="subject",
                references=refs,
                output_dir=root / "references",
                overwrite=False,
            )

            with patch("aigen.character_reference_pack.QwenVlm", FakeReferenceParser):
                result = parse_character_reference_pack(
                    root / "references" / "reference_pack.json",
                    parser_config(root),
                    output_path=None,
                    overwrite=False,
                    progress=SILENT_STATUS,
                )

            runner = FakeReferenceParser.last
            self.assertTrue(runner.closed)
            self.assertIn("Infer each reference role from the pixels yourself", runner.prompt)
            self.assertIn("The supplied reference ids are exactly:", runner.prompt)
            self.assertIn("reference_roles keys must be exactly the supplied reference ids:", runner.prompt)
            self.assertIn("ref_front", runner.prompt)
            self.assertIn("ref_face", runner.prompt)
            self.assertIn("ref_side", runner.prompt)
            self.assertIn("If there is no supplied reference id for body_shape, do not create one.", runner.prompt)
            self.assertIn(
                "body_proportion.evidence_refs must be a non-empty subset of the supplied reference ids:",
                runner.prompt,
            )
            self.assertCountEqual([path.name for path in runner.image_paths], ["front.png", "face.png", "side.png"])
            self.assertEqual(result["kind"], "character-identity-profile")
            self.assertEqual(result["body_proportion_source"], CHARACTER_BODY_PROPORTION_SOURCE)
            self.assertEqual(result["optional_missing_refs"], ["body_shape"])
            self.assertEqual(result["reference_roles"]["ref_face"], "portrait")
            self.assertEqual(result["body_proportion"]["do_not_change"], ["model-extracted body invariant"])
            self.assertTrue((root / "references" / "identity_profile.json").exists())

    def test_parse_reference_pack_rejects_unknown_vlm_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs = {"ref_front": root / "front.png"}
            write_image(refs["ref_front"], (96, 144), (200, 50, 50))
            build_character_reference_pack(
                character_id="subject",
                references=refs,
                output_dir=root / "references",
                overwrite=False,
            )

            with patch("aigen.character_reference_pack.QwenVlm", InvalidReferenceParser):
                with self.assertRaises(CharacterReferenceError):
                    parse_character_reference_pack(
                        root / "references" / "reference_pack.json",
                        parser_config(root),
                        output_path=None,
                        overwrite=False,
                        progress=SILENT_STATUS,
                    )

    def test_parse_character_edit_plan_uses_vlm_selected_refs_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs = {
                "ref_front": root / "front.png",
                "ref_face": root / "face.png",
                "ref_side": root / "side.png",
            }
            write_image(refs["ref_front"], (96, 144), (200, 50, 50))
            write_image(refs["ref_face"], (96, 96), (50, 200, 50))
            write_image(refs["ref_side"], (80, 144), (50, 50, 200))
            build_character_reference_pack(
                character_id="subject",
                references=refs,
                output_dir=root / "references",
                overwrite=False,
            )
            pack_path = root / "references" / "reference_pack.json"

            with patch("aigen.character_reference_pack.QwenVlm", FakeReferenceParser):
                parse_character_reference_pack(
                    pack_path,
                    parser_config(root),
                    output_path=None,
                    overwrite=False,
                    progress=SILENT_STATUS,
                )
            profile_payload = json.loads((root / "references" / "identity_profile.json").read_text(encoding="utf-8"))
            loaded_pack = load_completed_character_reference_pack(
                json.loads(pack_path.read_text(encoding="utf-8")),
                path_label=pack_path.as_posix(),
            )
            loaded_profile = load_completed_character_identity_profile(
                profile_payload,
                path_label=(root / "references" / "identity_profile.json").as_posix(),
            )
            runner = FakeEditInstructionParser(
                {
                    "selected_refs": ["ref_side", "ref_face"],
                    "edit_instruction": "model-authored edit instruction",
                }
            )

            result = parse_character_edit_plan(
                runner=runner,
                pack=loaded_pack,
                reference_paths={name: path for name, path in refs.items()},
                identity_profile=loaded_profile,
                case_name="right_profile",
                user_instruction="right side profile",
                path_label="test#right_profile",
            )

            self.assertEqual(result.selected_refs, ("ref_side", "ref_face"))
            self.assertEqual(result.edit_instruction, "model-authored edit instruction")
            self.assertEqual([path.name for path in runner.image_paths], ["face.png", "front.png", "side.png"])
            self.assertIn("The user's requested edit is exactly: 'right side profile'.", runner.prompt)
            self.assertIn("reference id ref_side, inferred role side", runner.prompt)
            self.assertIn("Qwen Image Edit accepts one to three selected reference images", runner.prompt)

    def test_parse_character_edit_plan_rejects_unknown_vlm_selected_ref(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs = {"ref_front": root / "front.png"}
            write_image(refs["ref_front"], (96, 144), (200, 50, 50))
            build_character_reference_pack(
                character_id="subject",
                references=refs,
                output_dir=root / "references",
                overwrite=False,
            )
            pack_path = root / "references" / "reference_pack.json"
            profile_payload = {
                "status": "completed",
                "kind": "character-identity-profile",
                "character_id": "subject",
                "source_reference_pack": pack_path.as_posix(),
                "identity": {"hair": "model-extracted hair fact"},
                "body_proportion": body_proportion_payload(evidence_refs=["ref_front"]),
                "body_proportion_source": CHARACTER_BODY_PROPORTION_SOURCE,
                "reference_roles": {"ref_front": "front"},
                "optional_missing_refs": ["body_shape"],
                "must_preserve": ["model-extracted visual invariant"],
                "avoid": [],
                "parser": {"id": "fake-parser"},
                "output": {"identity_profile": (root / "references" / "identity_profile.json").as_posix()},
            }
            loaded_pack = load_completed_character_reference_pack(
                json.loads(pack_path.read_text(encoding="utf-8")),
                path_label=pack_path.as_posix(),
            )
            loaded_profile = load_completed_character_identity_profile(
                profile_payload,
                path_label="identity_profile.json",
            )

            with self.assertRaises(CharacterReferenceError):
                parse_character_edit_plan(
                    runner=FakeEditInstructionParser(
                        {
                            "selected_refs": ["missing_ref"],
                            "edit_instruction": "model-authored edit instruction",
                        }
                    ),
                    pack=loaded_pack,
                    reference_paths=refs,
                    identity_profile=loaded_profile,
                    case_name="front",
                    user_instruction="front",
                    path_label="test#front",
                )


if __name__ == "__main__":
    unittest.main()
