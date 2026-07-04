from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from aigen.character_reference_models import CharacterReferenceError
from aigen.character_reference_pack import (
    build_character_reference_pack,
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
                    "hair": "short brown bob",
                    "eyes": "blue eyes",
                    "neckwear": "large blue bow",
                    "top": "brown glossy jacket over white shirt",
                    "bottom": "brown leather skirt",
                    "legwear": "blue thigh-highs",
                    "footwear": "brown boots",
                    "body_shape": "small chest and slim build",
                    "style": "anime concept art",
                },
                "reference_roles": {
                    "front": "front outfit and proportions",
                    "portrait": "face, eyes, hair and bow",
                    "side": "profile silhouette",
                },
                "must_preserve": [
                    "large blue bow",
                    "small chest",
                    "brown leather jacket",
                    "blue thigh-highs",
                ],
                "avoid": [
                    "blue necktie",
                    "large chest",
                    "different boots",
                ],
            }
        )

    def close(self) -> None:
        self.closed = True


class CharacterReferencePackTests(unittest.TestCase):
    def test_character_cli_exposes_reference_pack_build_and_parse(self) -> None:
        build_args = build_parser().parse_args(
            [
                "characters",
                "reference-pack",
                "build",
                "--character-id",
                "ai51",
                "--reference",
                "front=front.png",
                "--output-dir",
                "assets/characters/ai51/references",
            ]
        )
        parse_args = build_parser().parse_args(
            [
                "characters",
                "reference-pack",
                "parse",
                "assets/characters/ai51/references/reference_pack.json",
            ]
        )

        self.assertEqual(build_args.characters_command, "reference-pack")
        self.assertEqual(build_args.reference_pack_command, "build")
        self.assertEqual(parse_args.characters_command, "reference-pack")
        self.assertEqual(parse_args.reference_pack_command, "parse")
        self.assertEqual(parse_args.max_new_tokens, 1600)

    def test_parse_reference_args_rejects_duplicate_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_image(root / "front-a.png", (32, 48), (200, 50, 50))
            write_image(root / "front-b.png", (32, 48), (50, 200, 50))

            with self.assertRaises(CharacterReferenceError):
                parse_character_reference_args(["front=front-a.png", "front=front-b.png"], root)

    def test_build_reference_pack_writes_image_assets_and_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "front.png"
            portrait = root / "portrait.png"
            side = root / "side.png"
            write_image(front, (96, 144), (200, 50, 50))
            write_image(portrait, (96, 96), (50, 200, 50))
            write_image(side, (80, 144), (50, 50, 200))

            result = build_character_reference_pack(
                character_id="ai51",
                references={"front": front, "portrait": portrait, "side": side},
                output_dir=root / "references",
                overwrite=False,
            )

            pack_path = root / "references" / "reference_pack.json"
            self.assertTrue(pack_path.exists())
            self.assertEqual(result["kind"], "character-reference-pack")
            self.assertEqual(result["character_id"], "ai51")
            self.assertEqual(list(result["references"]), ["front", "portrait", "side"])
            self.assertEqual(result["references"]["front"]["width"], 96)
            self.assertIn("portrait", result["reference_roles"])

            manifest = json.loads(pack_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["output"]["reference_pack"], pack_path.as_posix())

    def test_parse_reference_pack_writes_identity_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs = {
                "front": root / "front.png",
                "portrait": root / "portrait.png",
                "side": root / "side.png",
            }
            write_image(refs["front"], (96, 144), (200, 50, 50))
            write_image(refs["portrait"], (96, 96), (50, 200, 50))
            write_image(refs["side"], (80, 144), (50, 50, 200))
            build_character_reference_pack(
                character_id="ai51",
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
            self.assertIn("Images are supplied in this exact order", runner.prompt)
            self.assertEqual([path.name for path in runner.image_paths], ["front.png", "portrait.png", "side.png"])
            self.assertEqual(result["kind"], "character-identity-profile")
            self.assertEqual(result["identity"]["neckwear"], "large blue bow")
            self.assertEqual(result["must_preserve"][1], "small chest")
            self.assertEqual(result["avoid"][0], "blue necktie")
            self.assertEqual(result["reference_roles"]["portrait"], "face, eyes, hair and bow")
            self.assertEqual(result["parser"]["device_report"], {"all": [{"device": "cuda:0"}]})
            self.assertTrue((root / "references" / "identity_profile.json").exists())


if __name__ == "__main__":
    unittest.main()
