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
from aigen.lora_training import _materialize_captioned_train_dataset, build_lora_train_plan
from aigen.manifest_io import write_json
from aigen.progress import SILENT_STATUS


def write_training_source(path: Path, color: tuple[int, int, int], mark: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 128), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((14, 18, 56, 92), outline="white", width=5)
    draw.text((18, 24), mark, fill="black")
    image.save(path)


def write_caption_plan(path: Path, *, view_bank: str, keyframe_run: str) -> None:
    write_json(
        path,
        {
            "$schema": "schemas/keyframe-brief-plan.schema.json",
            "kind": "keyframe-brief-plan",
            "brief_id": "ai51.punch",
            "planner_id": "test-planner",
            "source_brief_sha256": "0" * 64,
            "planner_prompt_sha256": "1" * 64,
            "identity_details": {
                "subject": "anime girl",
                "hair": "short pink bob",
                "face": "blue eyes",
                "upper_clothing": "glossy brown jacket over white shirt",
                "neckwear": "blue tie",
                "waist_garment": "brown leather skirt",
                "legwear": "blue thigh-high socks",
                "footwear": "brown boots",
                "style": "clean anime platformer sprite style",
            },
            "identity_description": "AI51 anime girl with short pink bob and brown leather outfit",
            "pose_description": "left-facing platformer punch-start pose",
            "platformer_camera_description": "readable side-view platformer camera",
            "identity_primer": {"view": "left_profile", "path": "views/left_profile.png"},
            "prompt": {
                "clip": "AI51 platformer keyframe",
                "t5": "AI51 anime girl platformer punch-start keyframe",
                "true_cfg_scale": 1.0,
            },
            "canvas": {"width": 576, "height": 864, "reference_max_area": 294912, "max_sequence_length": 128},
            "sampling": {"steps": 24, "guidance_scale": 2.5},
            "controls": [
                {
                    "name": "example_pose",
                    "type": "pose",
                    "source": "example_pose",
                    "scale": 0.7,
                    "start": 0.0,
                    "end": 0.65,
                }
            ],
            "scoring": {"top_k": 1, "priorities": ["condition match"], "checks": ["identity preserved"]},
            "polish": {
                "profile": "kontext-inpaint-local",
                "max_regions": 1,
                "strength_offsets": [0.0],
                "seed_offsets": [0],
            },
            "lora_captions": {"view_bank": view_bank, "keyframe_run": keyframe_run},
            "rationale": ["test fixture"],
        },
    )


class LoraDatasetTests(unittest.TestCase):
    def test_cli_lora_dataset_schema_has_no_job_version_field(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = main(["lora", "dataset-schema", "--compact"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["properties"]["kind"]["const"], "lora-dataset")
        self.assertNotIn("schema_" "version", payload["properties"])

    def test_build_lora_dataset_from_approved_views_and_selected_keyframe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "assets" / "front.png"
            left = root / "assets" / "left.png"
            back = root / "assets" / "back.png"
            top = root / "assets" / "top.png"
            keyframe = root / "runs" / "keyframes" / "ai51" / "punch" / "seed_060.png"
            write_training_source(front, (180, 50, 60), "F")
            write_training_source(left, (50, 170, 90), "L")
            write_training_source(back, (120, 80, 170), "B")
            write_training_source(top, (90, 130, 180), "T")
            write_training_source(keyframe, (60, 80, 190), "K")
            plan_path = root / "plans" / "punch_plan.json"
            write_caption_plan(
                plan_path,
                view_bank=(
                    "AI51 anime girl character sheet, short pink bob, glossy brown jacket, "
                    "white shirt, blue tie, brown leather skirt, blue thigh-high socks, brown boots"
                ),
                keyframe_run=(
                    "AI51 platformer attack keyframe, same pink bob and brown leather outfit, "
                    "readable side-view action pose"
                ),
            )
            bank_path = root / "assets" / "view_bank.json"
            write_json(
                bank_path,
                {
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
                        "back": {
                            "view": {"name": "back", "camera": "orthographic-back", "pose": "neutral-standing"},
                            "image": image_asset_json(back),
                            "accepted_candidate": "seed_004",
                            "accepted_seed": 4,
                            "acceptance": {"manual": ["approved back-view identity primer"]},
                        },
                        "top": {
                            "view": {"name": "top", "camera": "orthographic-top", "pose": "neutral-standing"},
                            "image": image_asset_json(top),
                            "accepted_candidate": "seed_006",
                            "accepted_seed": 6,
                            "acceptance": {"manual": ["approved top-down identity primer"]},
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
            write_json(
                run_dir / "selected.json",
                {
                    "selection_mode": "condition_score_with_semantic_gate",
                    "scorer": "condition-v1",
                    "semantic_gate": {
                        "passed": ["seed_060"],
                        "blocked": [],
                        "usable_for_auto_select": True,
                        "selection_owner": "condition_score",
                    },
                    "selected": [
                        {
                            "candidate": "seed_060",
                            "scores": {
                                "final": 0.84,
                                "condition": 0.82,
                                "pose": 0.80,
                                "side_profile": 0.86,
                                "artifact": 0.95,
                            },
                            "hard_rejects": {
                                "missing_foreground": False,
                                "weak_condition_match": False,
                                "weak_side_profile": False,
                                "weak_pose_match": False,
                                "artifact_quality_failure": False,
                            },
                            "metrics": {"pose": {"common_keypoints": 12}},
                        }
                    ],
                },
            )
            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "ai51_identity_lora_pilot",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {
                            "type": "view_bank",
                            "path": bank_path.as_posix(),
                            "views": ["front", "left_profile", "back", "top"],
                            "caption_source": {"plan": plan_path.as_posix(), "field": "view_bank"},
                        },
                        {
                            "type": "keyframe_run",
                            "run_dir": run_dir.as_posix(),
                            "selection_path": (run_dir / "selected.json").as_posix(),
                            "caption_source": {"plan": plan_path.as_posix(), "field": "keyframe_run"},
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

            result = build_lora_dataset(spec_path, progress=SILENT_STATUS)

            output_dir = Path(result["output"]["directory"])
            self.assertEqual(result["accepted_image_count"], 5)
            self.assertEqual(result["split_counts"], {"train": 3, "val": 2})
            self.assertTrue((output_dir / "contact_sheet.png").exists())
            self.assertTrue((output_dir / "captions.txt").exists())
            metadata_lines = (output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(metadata_lines), 5)
            records = [json.loads(line) for line in metadata_lines]
            self.assertTrue(all(record["prompt"].startswith("ai51char, ") for record in records))
            self.assertTrue(any(record["name"] == "top" for record in records))
            self.assertTrue(any("readable side view action pose" in record["prompt"] for record in records))
            self.assertTrue(all((output_dir / record["caption_file"]).exists() for record in records))
            keyframe_record = next(record for record in records if record["source_kind"] == "keyframe_run")
            selection = keyframe_record["source_metadata"]["score_selection"]
            self.assertEqual(selection["selection_mode"], "condition_score_with_semantic_gate")
            self.assertEqual(selection["semantic_gate"]["usable_for_auto_select"], True)
            self.assertEqual(selection["scores"]["pose"], 0.80)
            self.assertNotIn("training_preflight", result)
            written_report = json.loads((output_dir / "dataset_report.json").read_text(encoding="utf-8"))
            self.assertNotIn("training_preflight", written_report)

    def test_lora_training_dataset_materializes_prompt_column(self) -> None:
        from datasets import load_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "assets" / "front.png"
            write_training_source(image, (80, 120, 190), "F")
            plan_path = root / "plans" / "punch_plan.json"
            write_caption_plan(
                plan_path,
                view_bank="AI51 approved side-view identity image",
                keyframe_run="AI51 selected keyframe image",
            )
            bank_path = root / "assets" / "view_bank.json"
            write_json(
                bank_path,
                {
                    "kind": "character-view-bank",
                    "character": {"id": "ai51", "source_reference": image_asset_json(image)},
                    "views": {
                        "front": {
                            "view": {"name": "front", "camera": "orthographic-front", "pose": "source-concept"},
                            "image": image_asset_json(image),
                            "accepted_candidate": "source",
                        }
                    },
                },
            )
            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "ai51_identity_lora_pilot",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {
                            "type": "view_bank",
                            "path": bank_path.as_posix(),
                            "views": ["front"],
                            "caption_source": {"plan": plan_path.as_posix(), "field": "view_bank"},
                            "split": "train",
                        }
                    ],
                    "output": {
                        "directory": (root / "dataset").as_posix(),
                        "overwrite": True,
                        "validation_ratio": 0.0,
                        "save_contact_sheet": True,
                    },
                },
            )
            dataset_result = build_lora_dataset(spec_path, progress=SILENT_STATUS)
            train_dataset_dir = root / "lora-output" / "train_dataset"

            _materialize_captioned_train_dataset(Path(dataset_result["output"]["directory"]), train_dataset_dir)

            loaded = load_dataset(train_dataset_dir.as_posix())
            self.assertEqual(loaded["train"].column_names, ["image", "prompt"])
            self.assertEqual(
                loaded["train"][0]["prompt"],
                "ai51char, AI51 approved side view identity image, front, source concept, orthographic front",
            )

    def test_view_bank_source_requires_explicit_views(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "bad",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {
                            "type": "view_bank",
                            "path": "assets/characters/ai51/view_bank.json",
                        }
                    ],
                    "output": {
                        "directory": "dataset",
                        "overwrite": True,
                        "validation_ratio": 0.1,
                        "save_contact_sheet": True,
                    },
                },
            )

            with self.assertRaisesRegex(LoraDatasetError, "views"):
                load_lora_dataset_spec(spec_path)

    def test_view_bank_source_requires_non_empty_view_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "bad",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {
                            "type": "view_bank",
                            "path": "assets/characters/ai51/view_bank.json",
                            "views": [],
                        }
                    ],
                    "output": {
                        "directory": "dataset",
                        "overwrite": True,
                        "validation_ratio": 0.1,
                        "save_contact_sheet": True,
                    },
                },
            )

            with self.assertRaisesRegex(LoraDatasetError, "views"):
                load_lora_dataset_spec(spec_path)

    def test_lora_dataset_rejects_freeform_caption_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "bad",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {
                            "type": "view_bank",
                            "path": "assets/characters/ai51/view_bank.json",
                            "views": ["front"],
                            "caption": "hand-written caption is not accepted",
                        }
                    ],
                    "output": {
                        "directory": "dataset",
                        "overwrite": True,
                        "validation_ratio": 0.1,
                        "save_contact_sheet": True,
                    },
                },
            )

            with self.assertRaisesRegex(LoraDatasetError, "caption"):
                load_lora_dataset_spec(spec_path)

    def test_view_bank_source_requires_caption_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "bad",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {
                            "type": "view_bank",
                            "path": "assets/characters/ai51/view_bank.json",
                            "views": ["front"],
                        }
                    ],
                    "output": {
                        "directory": "dataset",
                        "overwrite": True,
                        "validation_ratio": 0.1,
                        "save_contact_sheet": True,
                    },
                },
            )

            with self.assertRaisesRegex(LoraDatasetError, "caption_source"):
                load_lora_dataset_spec(spec_path)

    def test_keyframe_run_source_requires_caption_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "bad",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {
                            "type": "keyframe_run",
                            "run_dir": "runs/keyframes/ai51/punch",
                            "selection_path": "runs/keyframes/ai51/punch/selected.json",
                        }
                    ],
                    "output": {
                        "directory": "dataset",
                        "overwrite": True,
                        "validation_ratio": 0.1,
                        "save_contact_sheet": True,
                    },
                },
            )

            with self.assertRaisesRegex(LoraDatasetError, "caption_source"):
                load_lora_dataset_spec(spec_path)

    def test_view_bank_source_rejects_keyframe_caption_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "bad",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {
                            "type": "view_bank",
                            "path": "assets/characters/ai51/view_bank.json",
                            "views": ["front"],
                            "caption_source": {"plan": "plans/punch_plan.json", "field": "keyframe_run"},
                        }
                    ],
                    "output": {
                        "directory": "dataset",
                        "overwrite": True,
                        "validation_ratio": 0.1,
                        "save_contact_sheet": True,
                    },
                },
            )

            with self.assertRaisesRegex(LoraDatasetError, "caption_source.field=view_bank"):
                load_lora_dataset_spec(spec_path)

    def test_keyframe_run_source_rejects_view_bank_caption_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "bad",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {
                            "type": "keyframe_run",
                            "run_dir": "runs/keyframes/ai51/punch",
                            "selection_path": "runs/keyframes/ai51/punch/selected.json",
                            "caption_source": {"plan": "plans/punch_plan.json", "field": "view_bank"},
                        }
                    ],
                    "output": {
                        "directory": "dataset",
                        "overwrite": True,
                        "validation_ratio": 0.1,
                        "save_contact_sheet": True,
                    },
                },
            )

            with self.assertRaisesRegex(LoraDatasetError, "caption_source.field=keyframe_run"):
                load_lora_dataset_spec(spec_path)

    def test_lora_train_plan_builds_local_16gb_launch_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "assets" / "front.png"
            write_training_source(image, (80, 120, 190), "F")
            plan_path = root / "plans" / "punch_plan.json"
            write_caption_plan(
                plan_path,
                view_bank="AI51 approved identity image",
                keyframe_run="AI51 selected keyframe image",
            )
            bank_path = root / "assets" / "view_bank.json"
            write_json(
                bank_path,
                {
                    "kind": "character-view-bank",
                    "character": {"id": "ai51", "source_reference": image_asset_json(image)},
                    "views": {
                        "front": {
                            "view": {"name": "front", "camera": "orthographic-front", "pose": "source-concept"},
                            "image": image_asset_json(image),
                            "accepted_candidate": "source",
                        }
                    },
                },
            )
            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "ai51_identity_lora_pilot",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {
                            "type": "view_bank",
                            "path": bank_path.as_posix(),
                            "views": ["front"],
                            "caption_source": {"plan": plan_path.as_posix(), "field": "view_bank"},
                            "split": "train",
                        }
                    ],
                    "output": {
                        "directory": (root / "dataset").as_posix(),
                        "overwrite": True,
                        "validation_ratio": 0.0,
                        "save_contact_sheet": True,
                    },
                },
            )
            dataset_result = build_lora_dataset(spec_path, progress=SILENT_STATUS)
            trainer_script = root / "train_dreambooth_lora_flux.py"
            trainer_script.write_text('check_min_version("0.38.0")\n', encoding="utf-8")
            base_model = root / "models" / "FLUX.1-dev-bnb-4bit"
            (base_model / "transformer").mkdir(parents=True)

            plan = build_lora_train_plan(
                Path(dataset_result["output"]["directory"]),
                trainer_script=trainer_script,
                base_model=base_model,
                output_dir=root / "lora-output",
            )

            command = plan["command"]
            self.assertEqual(plan["status"], "ready_to_launch")
            self.assertEqual(plan["profile"], "flux-lora-local-16gb")
            self.assertEqual(plan["dataset"]["caption_column"], "prompt")
            self.assertEqual(plan["trainer"]["required_instance_prompt"], "ai51char")
            self.assertEqual(plan["model"]["base_model_kind"], "local_bnb_4bit_flux_pipeline")
            self.assertIn(base_model.as_posix(), command)
            self.assertIn("--dataset_name", command)
            self.assertIn((root / "lora-output" / "train_dataset").as_posix(), command)
            self.assertIn("--caption_column", command)
            self.assertIn("prompt", command)
            self.assertNotIn("--instance_data_dir", command)
            self.assertEqual(
                plan["dataset"]["source_train_dir"],
                (Path(dataset_result["output"]["directory"]) / "images" / "train").as_posix(),
            )
            self.assertIn("--resolution", command)
            self.assertIn("512", command)
            self.assertEqual(command.count("--mixed_precision"), 1)
            self.assertIn("--rank", command)
            self.assertIn("4", command)
            self.assertIn("--lora_layers", command)
            self.assertIn("to_q,to_k,to_v,to_out.0", command)
            self.assertIn("--gradient_checkpointing", command)
            self.assertIn("--use_8bit_adam", command)
            self.assertIn("--cache_latents", command)

    def test_lora_train_cli_rejects_invalid_numeric_parameters(self) -> None:
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                main(["lora", "train-plan", "dataset", "--rank", "0"])

        self.assertNotEqual(raised.exception.code, 0)
        self.assertIn("must be greater than 0", stderr.getvalue())

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

    def test_rejects_unscored_keyframe_selection_as_lora_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "run" / "seed_060.png"
            write_training_source(image, (70, 90, 130), "K")
            plan_path = root / "plans" / "punch_plan.json"
            write_caption_plan(
                plan_path,
                view_bank="AI51 approved identity image",
                keyframe_run="AI51 manual selected output",
            )
            write_json(
                root / "run" / "result.json",
                {
                    "job_id": "manual.selection",
                    "effective_config": {
                        "prompt": {"t5": "manual selected output"},
                        "keyframe": {
                            "action": "punch",
                            "phase": "attack",
                            "direction": "left",
                            "camera": "platformer-side-view",
                        },
                    },
                    "outputs": [{"name": "seed_060", "path": image.as_posix(), "seed": 60}],
                },
            )
            write_json(root / "run" / "selected.json", {"selected": ["seed_060"]})
            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "bad",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {
                            "type": "keyframe_run",
                            "run_dir": "run",
                            "selection_path": "run/selected.json",
                            "caption_source": {"plan": plan_path.as_posix(), "field": "keyframe_run"},
                        }
                    ],
                    "output": {"directory": "dataset", "overwrite": True, "validation_ratio": 0.1, "save_contact_sheet": True},
                },
            )

            with self.assertRaisesRegex(LoraDatasetError, "condition_score_with_semantic_gate"):
                build_lora_dataset(spec_path, progress=SILENT_STATUS)

    def test_rejects_failed_control_audit_as_dataset_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "audit" / "control_off.png"
            write_training_source(image, (60, 80, 120), "A")
            plan_path = root / "plans" / "punch_plan.json"
            write_caption_plan(
                plan_path,
                view_bank="AI51 approved identity image",
                keyframe_run="AI51 failed audit output",
            )
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
                root / "audit" / "selected.json",
                {
                    "selection_mode": "condition_score_with_semantic_gate",
                    "scorer": "condition-v1",
                    "semantic_gate": {
                        "passed": ["control_off"],
                        "blocked": [],
                        "usable_for_auto_select": True,
                        "selection_owner": "condition_score",
                    },
                    "selected": [
                        {
                            "candidate": "control_off",
                            "scores": {"final": 0.8},
                            "hard_rejects": {"artifact_quality_failure": False},
                        }
                    ],
                },
            )
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "bad",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {
                            "type": "keyframe_run",
                            "run_dir": "audit",
                            "selection_path": "audit/selected.json",
                            "caption_source": {"plan": plan_path.as_posix(), "field": "keyframe_run"},
                        }
                    ],
                    "output": {"directory": "dataset", "overwrite": True, "validation_ratio": 0.1, "save_contact_sheet": True},
                },
            )

            with self.assertRaisesRegex(LoraDatasetError, "failed control-audit"):
                build_lora_dataset(spec_path, progress=SILENT_STATUS)


if __name__ == "__main__":
    unittest.main()
