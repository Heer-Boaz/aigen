from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from aigen.cli import main
from aigen.image_assets import image_asset_json
from aigen.lora_dataset_models import LoraDatasetError, load_lora_dataset_spec
from aigen.lora_datasets import build_lora_dataset
from aigen.manifest_io import write_json


def write_training_source(path: Path, color: tuple[int, int, int], mark: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 128), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((14, 18, 56, 92), outline="white", width=5)
    draw.text((18, 24), mark, fill="black")
    image.save(path)


class LoraDatasetTests(unittest.TestCase):
    def test_cli_lora_dataset_schema_has_no_job_version_field(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = main(["lora", "dataset-schema", "--compact"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["properties"]["kind"]["const"], "lora-dataset")
        self.assertNotIn("schema_version", payload["properties"])

    def test_build_lora_dataset_from_approved_views_and_selected_keyframe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "assets" / "front.png"
            left = root / "assets" / "left.png"
            keyframe = root / "runs" / "keyframes" / "ai51" / "punch" / "seed_060.png"
            write_training_source(front, (180, 50, 60), "F")
            write_training_source(left, (50, 170, 90), "L")
            write_training_source(keyframe, (60, 80, 190), "K")
            bank_path = root / "assets" / "view_bank.json"
            write_json(
                bank_path,
                {
                    "schema_version": 1,
                    "kind": "character-view-bank",
                    "character": {"id": "ai51", "source_reference": image_asset_json(front)},
                    "views": {
                        "front": {
                            "view": {"name": "front", "camera": "orthographic-front", "pose": "source-concept"},
                            "image": image_asset_json(front),
                            "accepted_candidate": "source",
                            "acceptance": {"manual": ["human approved canonical front primer"]},
                        },
                        "left_profile": {
                            "view": {
                                "name": "left_profile",
                                "camera": "orthographic-side",
                                "pose": "neutral-standing",
                            },
                            "image": image_asset_json(left),
                            "accepted_candidate": "seed_002",
                            "accepted_seed": 2,
                            "acceptance": {"manual": ["approved side-profile identity primer"]},
                        },
                    },
                },
            )
            run_dir = root / "runs" / "keyframes" / "ai51" / "punch"
            write_json(
                run_dir / "result.json",
                {
                    "job_id": "ai51.punch.platformer",
                    "effective_config": {
                        "prompt": {
                            "t5": (
                                "pink bob anime character, glossy brown jacket, blue tie, "
                                "brown skirt, platformer attack pose"
                            )
                        },
                        "keyframe": {
                            "action": "punch",
                            "phase": "platformer attack",
                            "direction": "left",
                            "camera": "orthographic-side",
                        },
                    },
                    "outputs": [{"name": "seed_060", "path": keyframe.as_posix(), "seed": 60}],
                },
            )
            write_json(run_dir / "selected.json", {"selected": ["seed_060"]})
            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "ai51_identity_lora_pilot",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {"type": "view_bank", "path": bank_path.as_posix(), "views": ["front", "left_profile"]},
                        {
                            "type": "keyframe_run",
                            "run_dir": run_dir.as_posix(),
                            "selection_path": (run_dir / "selected.json").as_posix(),
                        },
                    ],
                    "output": {
                        "directory": (root / "dataset").as_posix(),
                        "overwrite": True,
                        "validation_ratio": 0.34,
                        "save_contact_sheet": True,
                    },
                },
            )

            result = build_lora_dataset(spec_path)

            output_dir = Path(result["output"]["directory"])
            self.assertEqual(result["accepted_image_count"], 3)
            self.assertEqual(result["split_counts"], {"train": 2, "val": 1})
            self.assertTrue((output_dir / "contact_sheet.png").exists())
            self.assertTrue((output_dir / "captions.txt").exists())
            metadata_lines = (output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(metadata_lines), 3)
            records = [json.loads(line) for line in metadata_lines]
            self.assertTrue(all(record["prompt"].startswith("ai51char, ") for record in records))
            self.assertTrue(any("platformer attack pose" in record["prompt"] for record in records))
            self.assertTrue(all((output_dir / record["caption_file"]).exists() for record in records))

    def test_keyframe_run_source_requires_approved_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dataset.json"
            write_json(
                path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "bad",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [{"type": "keyframe_run", "run_dir": "runs/keyframes/ai51/punch"}],
                    "output": {"directory": "dataset", "overwrite": True, "validation_ratio": 0.1, "save_contact_sheet": True},
                },
            )

            with self.assertRaises(LoraDatasetError):
                load_lora_dataset_spec(path)

    def test_rejects_failed_control_audit_as_dataset_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "audit" / "control_off.png"
            write_training_source(image, (60, 80, 120), "A")
            write_json(root / "audit" / "audit.json", {"passed": False})
            write_json(
                root / "audit" / "result.json",
                {
                    "job_id": "failed.audit",
                    "effective_config": {
                        "prompt": {"t5": "failed audit output"},
                        "keyframe": {
                            "action": "audit",
                            "phase": "failed",
                            "direction": "left",
                            "camera": "orthographic-side",
                        },
                    },
                    "outputs": [{"name": "control_off", "path": image.as_posix()}],
                },
            )
            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "bad",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [{"type": "keyframe_run", "run_dir": "audit", "candidates": ["control_off"]}],
                    "output": {"directory": "dataset", "overwrite": True, "validation_ratio": 0.1, "save_contact_sheet": True},
                },
            )

            with self.assertRaisesRegex(LoraDatasetError, "failed control-audit"):
                build_lora_dataset(spec_path)


if __name__ == "__main__":
    unittest.main()
