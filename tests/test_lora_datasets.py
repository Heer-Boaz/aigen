from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from aigen.cli import main
from aigen.generation.kontext_identity import _pipeline_device_report
from aigen.lora_candidate_models import (
    LoraCandidateBriefError,
    LoraCandidatePromptSpec,
    LoraCandidateTemplateSpec,
    load_lora_candidate_brief,
)
from aigen.lora_dataset_models import LoraDatasetError, load_lora_dataset_spec
from aigen.lora_datasets import build_lora_dataset
from aigen.lora_candidates import LoraCandidateError, plan_lora_candidates
from aigen.lora_training import materialize_captioned_train_dataset, build_lora_train_plan
from aigen.manifest_io import write_json
from aigen.progress import SILENT_STATUS


def write_training_source(path: Path, color: tuple[int, int, int], mark: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 128), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((14, 18, 56, 92), outline="white", width=5)
    draw.text((18, 24), mark, fill="black")
    image.save(path)


class FakeLoraCandidatePlanner:
    def __init__(self) -> None:
        self.prompt = ""
        self.prompts: list[str] = []
        self.image_paths: list[Path] = []
        self.closed = False

    def judge_candidate(self, prompt: str, image_paths: list[Path]) -> str:
        self.prompt = prompt
        self.prompts.append(prompt)
        self.image_paths = image_paths
        if "name: left_profile_neutral_standing" in prompt:
            candidate = {
                "name": "left_profile_neutral_standing",
                "view": "left profile view",
                "pose": "neutral standing pose",
                "identity_primer": "left_profile",
                "prompt": {
                    "positive": (
                        "Anime-style full-body illustration of a girl with short brown hair, blue eyes, brown leather jacket, "
                        "blue necktie, brown leather skirt with belt, blue thigh-highs and "
                        "brown boots, left profile view, neutral standing pose, plain studio background"
                    )
                },
            }
        else:
            candidate = {
                "name": "front_neutral_standing",
                "view": "front view",
                "pose": "neutral standing pose",
                "identity_primer": "front",
                "prompt": {
                    "positive": (
                        "Anime-style full-body illustration of a girl with short brown hair, blue eyes, brown leather jacket, "
                        "blue necktie, brown leather skirt with belt, blue thigh-highs and "
                        "brown boots, front view, neutral standing pose, plain studio background"
                    )
                },
            }
        return json.dumps(candidate)

    def close(self) -> None:
        self.closed = True


class InvalidLoraCandidatePlanner(FakeLoraCandidatePlanner):
    def judge_candidate(self, prompt: str, image_paths: list[Path]) -> str:
        self.prompt = prompt
        self.prompts.append(prompt)
        self.image_paths = image_paths
        return json.dumps(
            {
                "candidates": [
                    {
                        "name": "front_neutral_standing",
                        "view": "front view",
                        "pose": "neutral standing pose",
                        "identity_primer": "front",
                        "prompt": {
                            "positive": (
                                "Anime-style full-body illustration of a girl with short brown hair, blue eyes, brown leather jacket, "
                                "blue necktie, brown leather skirt with belt, blue thigh-highs and "
                                "brown boots, front view, neutral standing pose, plain studio background"
                            )
                        },
                    },
                ]
            }
        )


class FakeLoraCandidateJudge:
    device_report = {"modules": []}

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.image_paths: list[list[Path]] = []
        self.closed = False

    def judge_candidate(self, prompt: str, image_paths: list[Path]) -> str:
        self.prompts.append(prompt)
        self.image_paths.append(image_paths)
        candidate = prompt.split("Generated candidate image named ", 1)[1].split(".", 1)[0]
        return json.dumps(
            {
                "candidate": candidate,
                "pass": True,
                "hard_rejects": {
                    "wrong_face": False,
                    "wrong_hair_length_or_color": False,
                    "wrong_outfit": False,
                    "missing_required_neckwear_or_accessory": False,
                    "missing_required_waist_or_lower_body_garment": False,
                    "missing_required_belt_or_waist_detail": False,
                    "missing_required_legwear": False,
                    "missing_required_footwear": False,
                    "deformed_body": False,
                    "broken_hands_or_feet": False,
                    "bad_crop": False,
                    "dirty_or_distracting_background": False,
                    "style_drift": False,
                    "view_label_mismatch": False,
                },
                "scores": {
                    "identity_preservation": 9,
                    "outfit_preservation": 9,
                    "anatomy_quality": 9,
                    "crop_quality": 9,
                    "background_quality": 9,
                    "style_match": 9,
                    "view_pose_match": 9,
                    "training_usability": 9,
                },
                "evidence": {
                    "identity_match": "Matches the approved primer.",
                    "quality_assessment": "Canon-worthy training image.",
                    "concerns": [],
                },
            }
        )

    def close(self) -> None:
        self.closed = True


class FakeKontextIdentityPipeline:
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"

    def __init__(self) -> None:
        import torch

        self.transformer = torch.nn.Linear(1, 1)
        self.vae = torch.nn.Linear(1, 1)
        self.text_encoder = torch.nn.Linear(1, 1)
        self.text_encoder_2 = torch.nn.Linear(1, 1)


class LoraDatasetTests(unittest.TestCase):
    def test_kontext_identity_device_report_uses_pipeline_components(self) -> None:
        report = _pipeline_device_report(FakeKontextIdentityPipeline())

        self.assertEqual(report["pipeline_class"], "FakeKontextIdentityPipeline")
        self.assertEqual(
            sorted(report["components"]),
            ["text_encoder", "text_encoder_2", "transformer", "vae"],
        )
        self.assertEqual(report["components"]["transformer"]["class"], "Linear")

    def test_lora_candidate_prompt_schema_is_not_character_specific(self) -> None:
        prompt = LoraCandidatePromptSpec(
            positive="matte robot character illustration, rear camera view, neutral studio floor"
        )

        self.assertEqual(
            prompt.positive,
            "matte robot character illustration, rear camera view, neutral studio floor",
        )

    def test_lora_candidate_prompt_must_materialize_view_and_pose(self) -> None:
        with self.assertRaisesRegex(ValueError, "view term"):
            LoraCandidateTemplateSpec(
                name="left_profile_idle",
                view="left profile view",
                pose="idle standing pose",
                identity_primer="front",
                prompt={
                    "positive": (
                        "Anime-style full-body illustration of the approved character, "
                        "idle standing pose, plain studio background"
                    )
                },
            )

        with self.assertRaisesRegex(ValueError, "view term"):
            LoraCandidateTemplateSpec(
                name="right_profile_idle",
                view="right profile view",
                pose="idle standing pose",
                identity_primer="front",
                prompt={
                    "positive": (
                        "Anime-style full-body illustration of the approved character, "
                        "bright studio background, idle standing pose"
                    )
                },
            )

        with self.assertRaisesRegex(ValueError, "requested pose"):
            LoraCandidateTemplateSpec(
                name="front_walk_contact",
                view="front view",
                pose="walk contact pose",
                identity_primer="front",
                prompt={
                    "positive": (
                        "Anime-style full-body illustration of the approved character, "
                        "front view, plain studio background"
                    )
                },
            )

        with self.assertRaisesRegex(ValueError, "front-view gaze"):
            LoraCandidateTemplateSpec(
                name="left_profile_idle",
                view="left profile view",
                pose="idle standing pose",
                identity_primer="left_profile",
                prompt={
                    "positive": (
                        "Anime-style full-body illustration, left profile view, idle standing pose, "
                        "looking at viewer, plain studio background"
                    )
                },
            )

        with self.assertRaisesRegex(ValueError, "name view term"):
            LoraCandidateTemplateSpec(
                name="left_profile_three_quarter",
                view="three-quarter front view",
                pose="neutral standing pose",
                identity_primer="front",
                prompt={
                    "positive": (
                        "Anime-style full-body illustration, three-quarter front view, neutral standing pose, "
                        "plain studio background"
                    )
                },
            )

        with self.assertRaisesRegex(ValueError, "front-facing facial details"):
            LoraCandidateTemplateSpec(
                name="back_neutral",
                view="back view",
                pose="neutral standing pose",
                identity_primer="front",
                prompt={
                    "positive": (
                        "Anime-style full-body illustration, back view, neutral standing pose, "
                        "blue eyes and smiling face, brown leather jacket, leather skirt, plain studio background"
                    )
                },
            )

        with self.assertRaisesRegex(ValueError, "filler style wording"):
            LoraCandidatePromptSpec(
                positive="medium style, front view, neutral standing pose, full body, plain studio background"
            )

    def test_lora_candidate_templates_reject_reused_generation_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "front.png"
            write_training_source(front, (180, 50, 60), "F")
            canon_dir = root / "canon"
            main(
                [
                    "lora",
                    "canon-init",
                    "--character-id",
                    "ai51",
                    "--trigger-token",
                    "ai51char",
                    "--identity-prompt",
                    "1girl, blue eyes, short hair, brown hair, leather skirt, belt, blue necktie",
                    "--anchor",
                    f"front={front.as_posix()}",
                    "--output-dir",
                    canon_dir.as_posix(),
                    "--compact",
                ]
            )
            brief_path = root / "candidate_brief.json"
            write_json(
                brief_path,
                {
                    "$schema": "schemas/lora-candidate-brief.schema.json",
                    "kind": "lora-candidate-brief",
                    "id": "ai51.identity.candidates",
                    "character": {"canon": canon_dir.as_posix()},
                    "generation": {
                        "width": 96,
                        "height": 128,
                        "steps": 20,
                        "seed_start": 20,
                        "seeds_per_candidate": 1,
                    },
                    "candidates": [
                        {
                            "name": "front_neutral",
                            "view": "front view",
                            "pose": "neutral standing",
                            "identity_primer": "front",
                            "prompt": {
                                "positive": (
                                    "Clean illustration style, front view, neutral standing walk contact pose, "
                                    "complete character, plain studio background"
                                ),
                            },
                        },
                        {
                            "name": "front_walk_contact",
                            "view": "front view",
                            "pose": "walk contact",
                            "identity_primer": "front",
                            "prompt": {
                                "positive": (
                                    "Clean illustration style, front view, neutral standing walk contact pose, "
                                    "complete character, plain studio background"
                                ),
                            },
                        },
                    ],
                    "output": {"directory": (root / "candidates").as_posix(), "overwrite": True},
                },
            )

            with self.assertRaisesRegex(LoraCandidateBriefError, "candidate generation prompts must be unique"):
                load_lora_candidate_brief(brief_path)

    def test_lora_canon_init_writes_human_approved_anchor_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "front.png"
            left = root / "left.png"
            write_training_source(front, (180, 50, 60), "F")
            write_training_source(left, (50, 170, 90), "L")
            output_dir = root / "canon"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "lora",
                        "canon-init",
                        "--character-id",
                        "ai51",
                        "--trigger-token",
                        "ai51char",
                        "--identity-prompt",
                        (
                            "1girl, blue eyes, gloves, blue thigh-highs, full body, white blouse, "
                            "button-up shirt, short hair, brown hair, leather skirt, belt, brown long boots, "
                            "collared shirt, looking at viewer, brown leather jacket, sleeved jacket, "
                            "smile, light blush, blue necktie, standing, flat-chested, small breasts"
                        ),
                        "--anchor",
                        f"front={front.as_posix()}",
                        "--anchor",
                        f"left_profile={left.as_posix()}",
                        "--approved-by",
                        "boaz",
                        "--output-dir",
                        output_dir.as_posix(),
                        "--compact",
                    ]
                )

            result = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(result["kind"], "lora-canon")
            self.assertEqual(result["status"], "active")
            self.assertNotIn("schema_version", result)
            self.assertEqual(result["character"]["trigger_token"], "ai51char")
            self.assertEqual(len(result["images"]), 2)
            self.assertTrue((output_dir / "images" / "front.png").exists())
            self.assertTrue((output_dir / "images" / "left_profile.png").exists())
            self.assertTrue((output_dir / "contact_sheet.png").exists())
            self.assertTrue((output_dir / "canon_manifest.json").exists())
            captions = [
                (output_dir / item["training_caption_file"]).read_text(encoding="utf-8")
                for item in result["images"]
            ]
            self.assertTrue(all("training_caption" in item for item in result["images"]))
            self.assertTrue(all("prompt" not in item for item in result["images"]))
            self.assertTrue(all(caption.startswith("ai51char, 1girl, blue eyes") for caption in captions))
            self.assertTrue(any("left profile" in caption for caption in captions))
            left_caption = next(caption for caption in captions if "left profile" in caption)
            self.assertNotIn("looking at viewer", left_caption)
            self.assertEqual(result["images"][0]["approval"]["mode"], "human_approved_canon")

    def test_lora_canon_init_rejects_prompt_with_trigger_token(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "front.png"
            write_training_source(front, (180, 50, 60), "F")
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        "lora",
                        "canon-init",
                        "--character-id",
                        "ai51",
                        "--trigger-token",
                        "ai51char",
                        "--identity-prompt",
                        "AI51CHAR, short pink bob",
                        "--anchor",
                        f"front={front.as_posix()}",
                        "--output-dir",
                        (root / "canon").as_posix(),
                        "--compact",
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("must not include the trigger token", stderr.getvalue())

    def test_lora_candidate_brief_plan_uses_vlm_candidates_from_canon_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "front.png"
            left = root / "left.png"
            write_training_source(front, (180, 50, 60), "F")
            write_training_source(left, (50, 170, 90), "L")
            canon_dir = root / "canon"
            identity_prompt = (
                "1girl, blue eyes, gloves, blue thigh-highs, full body, white blouse, "
                "button-up shirt, short hair, brown hair, leather skirt, belt, brown long boots, "
                "collared shirt, looking at viewer, brown leather jacket, sleeved jacket, "
                "smile, light blush, blue necktie, standing, flat-chested, small breasts"
            )
            main(
                [
                    "lora",
                    "canon-init",
                    "--character-id",
                    "ai51",
                    "--trigger-token",
                    "ai51char",
                    "--identity-prompt",
                    identity_prompt,
                    "--anchor",
                    f"front={front.as_posix()}",
                    "--anchor",
                    f"left_profile={left.as_posix()}",
                    "--output-dir",
                    canon_dir.as_posix(),
                    "--compact",
                ]
            )
            planner = FakeLoraCandidatePlanner()
            brief_path = root / "jobs" / "lora_candidates.json"
            stdout = io.StringIO()

            with (
                patch("aigen.lora_candidate_planner.QwenVlm", return_value=planner),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "lora",
                        "candidate-brief-plan",
                        canon_dir.as_posix(),
                        "--output",
                        brief_path.as_posix(),
                        "--candidate-output-dir",
                        (root / "candidate_runs" / "ai51").as_posix(),
                        "--width",
                        "96",
                        "--height",
                        "128",
                        "--steps",
                        "20",
                        "--seed-start",
                        "30",
                        "--seeds-per-candidate",
                        "3",
                        "--candidate-count",
                        "2",
                        "--compact",
                    ]
                )

            result = json.loads(stdout.getvalue())
            brief = json.loads(brief_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(result["kind"], "lora-candidate-brief-plan")
            self.assertTrue(planner.closed)
            self.assertEqual(len(planner.prompts), 2)
            first_prompt = planner.prompts[0]
            self.assertIn(identity_prompt, first_prompt)
            self.assertIn("Available identity primer names: front, left_profile", first_prompt)
            self.assertIn("Write exactly one pre-LoRA generation prompt", first_prompt)
            self.assertIn('first character of your response must be "{"', first_prompt)
            self.assertIn('last character must be "}"', first_prompt)
            self.assertNotIn("Every positive prompt must contain these exact literal phrases", first_prompt)
            self.assertNotIn("- clean anime lineart", first_prompt)
            self.assertIn("The positive prompt is the exact generation prompt", first_prompt)
            self.assertNotIn("executor materializes", first_prompt)
            self.assertGreaterEqual(len(planner.image_paths), 4)
            self.assertEqual(brief["kind"], "lora-candidate-brief")
            self.assertNotIn("schema_version", brief)
            self.assertEqual(brief["id"], "ai51.lora.candidates")
            self.assertEqual(brief["generation"]["seeds_per_candidate"], 3)
            self.assertEqual(
                [candidate["name"] for candidate in brief["candidates"]],
                ["front_neutral_standing", "left_profile_neutral_standing"],
            )
            self.assertIn("leather skirt with belt", brief["candidates"][0]["prompt"]["positive"])
            self.assertNotIn("ai51char", brief["candidates"][0]["prompt"]["positive"])
            self.assertEqual(brief["candidates"][1]["identity_primer"], "left_profile")
            self.assertTrue((brief_path.with_suffix(".raw") / "front_neutral_standing.txt").exists())
            self.assertTrue((brief_path.with_suffix(".prompts") / "front_neutral_standing.txt").exists())

    def test_lora_candidate_planner_rejects_invalid_vlm_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "front.png"
            left = root / "left.png"
            write_training_source(front, (180, 50, 60), "F")
            write_training_source(left, (50, 170, 90), "L")
            canon_dir = root / "canon"
            identity_prompt = "1girl, blue eyes, short hair, brown hair, leather skirt, belt, blue necktie"
            main(
                [
                    "lora",
                    "canon-init",
                    "--character-id",
                    "ai51",
                    "--trigger-token",
                    "ai51char",
                    "--identity-prompt",
                    identity_prompt,
                    "--anchor",
                    f"front={front.as_posix()}",
                    "--anchor",
                    f"left_profile={left.as_posix()}",
                    "--output-dir",
                    canon_dir.as_posix(),
                    "--compact",
                ]
            )
            brief_path = root / "candidate_brief.json"
            planner = InvalidLoraCandidatePlanner()
            stdout = io.StringIO()
            with (
                patch("aigen.lora_candidate_planner.QwenVlm", return_value=planner),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "lora",
                        "candidate-brief-plan",
                        canon_dir.as_posix(),
                        "--output",
                        brief_path.as_posix(),
                        "--candidate-output-dir",
                        (root / "candidate_runs" / "ai51").as_posix(),
                        "--width",
                        "96",
                        "--height",
                        "128",
                        "--steps",
                        "20",
                        "--seed-start",
                        "30",
                        "--seeds-per-candidate",
                        "3",
                        "--candidate-count",
                        "2",
                        "--compact",
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertFalse(brief_path.exists())

    def test_lora_dataset_audit_accepts_canon_and_pending_loose_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "front.png"
            loose = root / "loose" / "candidate.png"
            write_training_source(front, (180, 50, 60), "F")
            write_training_source(loose, (50, 170, 90), "L")
            canon_dir = root / "canon"

            main(
                [
                    "lora",
                    "canon-init",
                    "--character-id",
                    "ai51",
                    "--trigger-token",
                    "ai51char",
                    "--identity-prompt",
                    "1girl, blue eyes, short hair, brown hair, leather skirt, blue necktie",
                    "--anchor",
                    f"front={front.as_posix()}",
                    "--output-dir",
                    canon_dir.as_posix(),
                    "--compact",
                ]
            )

            canon_stdout = io.StringIO()
            with contextlib.redirect_stdout(canon_stdout):
                canon_exit = main(
                    [
                        "lora",
                        "dataset-audit",
                        canon_dir.as_posix(),
                        "--output-dir",
                        (root / "canon_audit").as_posix(),
                        "--compact",
                    ]
                )
            canon_result = json.loads(canon_stdout.getvalue())
            self.assertEqual(canon_exit, 0)
            self.assertEqual(canon_result["status"], "accepted_canon")
            self.assertEqual(canon_result["counts"], {"images": 1, "accepted": 1, "pending": 0, "rejected": 0})
            self.assertTrue((root / "canon_audit" / "accepted.json").exists())
            self.assertTrue((root / "canon_audit" / "crops" / "front.png").exists())

            loose_stdout = io.StringIO()
            with contextlib.redirect_stdout(loose_stdout):
                loose_exit = main(
                    [
                        "lora",
                        "dataset-audit",
                        (root / "loose").as_posix(),
                        "--output-dir",
                        (root / "loose_audit").as_posix(),
                        "--compact",
                    ]
                )
            loose_result = json.loads(loose_stdout.getvalue())
            self.assertEqual(loose_exit, 0)
            self.assertEqual(loose_result["status"], "needs_human_review")
            self.assertEqual(loose_result["counts"], {"images": 1, "accepted": 0, "pending": 1, "rejected": 0})

    def test_build_lora_dataset_from_canon_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "front.png"
            left = root / "left.png"
            write_training_source(front, (180, 50, 60), "F")
            write_training_source(left, (50, 170, 90), "L")
            canon_dir = root / "canon"
            main(
                [
                    "lora",
                    "canon-init",
                    "--character-id",
                    "ai51",
                    "--trigger-token",
                    "ai51char",
                    "--identity-prompt",
                    "1girl, blue eyes, short hair, brown hair, leather skirt, blue necktie",
                    "--anchor",
                    f"front={front.as_posix()}",
                    "--anchor",
                    f"left_profile={left.as_posix()}",
                    "--output-dir",
                    canon_dir.as_posix(),
                    "--compact",
                ]
            )
            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "ai51_identity_canon",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {
                            "type": "canon",
                            "path": canon_dir.as_posix(),
                            "images": ["front", "left_profile"],
                            "tags": ["approved style note"],
                        }
                    ],
                    "output": {
                        "directory": (root / "dataset").as_posix(),
                        "overwrite": True,
                        "validation_ratio": 0.5,
                        "save_contact_sheet": True,
                    },
                },
            )

            result = build_lora_dataset(spec_path, progress=SILENT_STATUS)

            output_dir = Path(result["output"]["directory"])
            self.assertEqual(result["accepted_image_count"], 2)
            self.assertEqual(result["split_counts"], {"train": 1, "val": 1})
            records = [json.loads(line) for line in (output_dir / "metadata.jsonl").read_text().splitlines()]
            self.assertEqual({record["source_kind"] for record in records}, {"canon"})
            self.assertTrue(all(record["prompt"].startswith("ai51char, 1girl, blue eyes") for record in records))
            self.assertTrue(all("approved style note" not in record["prompt"] for record in records))
            self.assertTrue(all("approved style note" in record["tags"] for record in records))
            self.assertTrue(all(record["source_metadata"]["approval"]["mode"] == "human_approved_canon" for record in records))

    def test_lora_candidate_funnel_builds_dataset_from_human_accepted_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "front.png"
            write_training_source(front, (180, 50, 60), "F")
            canon_dir = root / "canon"
            identity_prompt = (
                "1girl, blue eyes, gloves, blue thigh-highs, full body, white blouse, "
                "button-up shirt, short hair, brown hair, leather skirt, belt, brown long boots, "
                "collared shirt, looking at viewer, brown leather jacket, sleeved jacket, "
                "smile, light blush, blue necktie, standing, flat-chested, small breasts"
            )
            main(
                [
                    "lora",
                    "canon-init",
                    "--character-id",
                    "ai51",
                    "--trigger-token",
                    "ai51char",
                    "--identity-prompt",
                    identity_prompt,
                    "--anchor",
                    f"front={front.as_posix()}",
                    "--output-dir",
                    canon_dir.as_posix(),
                    "--compact",
                ]
            )
            candidate_dir = root / "candidates"
            brief_path = root / "candidate_brief.json"
            write_json(
                brief_path,
                {
                    "$schema": "schemas/lora-candidate-brief.schema.json",
                    "kind": "lora-candidate-brief",
                    "id": "ai51.identity.candidates",
                    "character": {
                        "canon": canon_dir.as_posix(),
                    },
                    "generation": {
                        "width": 96,
                        "height": 128,
                        "steps": 20,
                        "seed_start": 20,
                        "seeds_per_candidate": 2,
                    },
                    "candidates": [
                        {
                            "name": "front_neutral",
                            "view": "front",
                            "pose": "neutral",
                            "identity_primer": "front",
                            "prompt": {
                                "positive": (
                                    f"Anime-style full-body illustration, {identity_prompt}, "
                                    "front view, neutral standing pose, plain studio background"
                                ),
                            },
                        },
                        {
                            "name": "walk_contact",
                            "view": "side",
                            "pose": "walk contact",
                            "identity_primer": "front",
                            "prompt": {
                                "positive": (
                                    f"Anime-style full-body illustration, {identity_prompt}, "
                                    "side-view walk contact pose, plain studio background"
                                ),
                            },
                        },
                    ],
                    "output": {
                        "directory": candidate_dir.as_posix(),
                        "overwrite": True,
                    },
                },
            )
            generate_stdout = io.StringIO()
            with contextlib.redirect_stdout(generate_stdout):
                generate_exit = main(
                    [
                        "lora",
                        "candidate-plan",
                        brief_path.as_posix(),
                        "--compact",
                    ]
                )
            generate_result = json.loads(generate_stdout.getvalue())
            self.assertEqual(generate_exit, 0)
            self.assertEqual(generate_result["kind"], "lora-candidate-plan")
            self.assertNotIn("schema_version", generate_result)
            self.assertEqual(generate_result["counts"]["candidates"], 4)
            self.assertEqual(generate_result["counts"]["candidate_templates"], 2)
            self.assertEqual([candidate["name"] for candidate in generate_result["candidate_templates"]], ["front_neutral", "walk_contact"])
            planned_manifest = json.loads((candidate_dir / "candidates.json").read_text(encoding="utf-8"))
            first_candidate = planned_manifest["candidates"][0]
            self.assertEqual(
                first_candidate["generation_prompt"],
                planned_manifest["candidate_templates"][0]["prompt"]["positive"],
            )
            self.assertIn(identity_prompt, first_candidate["generation_prompt"])
            self.assertNotIn("ai51char", first_candidate["generation_prompt"])
            self.assertEqual(first_candidate["training_caption"], f"ai51char, {first_candidate['generation_prompt']}")
            self.assertIn("generation_prompts/front_neutral_seed_0020.txt", planned_manifest["candidates"][0]["generation_prompt_file"])
            self.assertNotIn("candidates", generate_result)

            selected_name = "front_neutral_seed_0020"
            selected_path = candidate_dir / "images" / f"{selected_name}.png"
            write_training_source(selected_path, (90, 140, 210), "A")

            evidence_stdout = io.StringIO()
            with contextlib.redirect_stdout(evidence_stdout):
                evidence_exit = main(
                    [
                        "lora",
                        "candidate-evidence",
                        candidate_dir.as_posix(),
                        "--compact",
                    ]
                )
            evidence_result = json.loads(evidence_stdout.getvalue())
            self.assertEqual(evidence_exit, 0)
            self.assertEqual(evidence_result["kind"], "lora-candidate-evidence")
            self.assertEqual(evidence_result["counts"], {"candidates": 4, "review_items": 1, "rejected_images": 3})
            self.assertTrue(Path(evidence_result["review_items"][0]["evidence"]["crop_sheet"]).exists())

            unsupported_evidence_stderr = io.StringIO()
            with contextlib.redirect_stderr(unsupported_evidence_stderr):
                with self.assertRaises(SystemExit) as unsupported_evidence:
                    main(
                        [
                            "lora",
                            "candidate-evidence",
                            candidate_dir.as_posix(),
                            "--output-dir",
                            (root / "elsewhere").as_posix(),
                            "--compact",
                        ]
                    )
            self.assertEqual(unsupported_evidence.exception.code, 2)
            self.assertIn("unrecognized arguments: --output-dir", unsupported_evidence_stderr.getvalue())

            early_review_stderr = io.StringIO()
            with contextlib.redirect_stderr(early_review_stderr):
                early_review_exit = main(
                    [
                        "lora",
                        "candidate-review",
                        candidate_dir.as_posix(),
                        "--accept",
                        selected_name,
                        "--compact",
                    ]
                )
            self.assertEqual(early_review_exit, 1)
            self.assertIn("model-passed LoRA candidate items", early_review_stderr.getvalue())

            judge_runner = FakeLoraCandidateJudge()
            judge_stdout = io.StringIO()
            with (
                patch("aigen.lora_candidate_judge.QwenVlm", return_value=judge_runner),
                contextlib.redirect_stdout(judge_stdout),
            ):
                judge_exit = main(
                    [
                        "lora",
                        "candidate-judge",
                        candidate_dir.as_posix(),
                        "--compact",
                    ]
                )
            judge_result = json.loads(judge_stdout.getvalue())
            self.assertEqual(judge_exit, 0)
            self.assertEqual(judge_result["kind"], "lora-candidate-judge")
            self.assertEqual(judge_result["counts"], {"review_items": 1, "passed": 1, "blocked": 0})
            self.assertTrue(judge_runner.closed)
            self.assertIn("canon-worthy training data", judge_runner.prompts[0])
            self.assertEqual(judge_result["selection_gate"]["passed"], [selected_name])
            self.assertTrue((candidate_dir / "evidence" / "passed.json").exists())

            review_stdout = io.StringIO()
            with contextlib.redirect_stdout(review_stdout):
                review_exit = main(
                    [
                        "lora",
                        "candidate-review",
                        candidate_dir.as_posix(),
                        "--accept",
                        selected_name,
                        "--approved-by",
                        "boaz",
                        "--compact",
                    ]
                )
            review_result = json.loads(review_stdout.getvalue())
            self.assertEqual(review_exit, 0)
            self.assertEqual(review_result["kind"], "lora-candidate-review")
            self.assertEqual(review_result["counts"]["model_passed"], 1)
            self.assertEqual(review_result["counts"]["accepted"], 1)
            self.assertEqual(review_result["quota_report"]["by_candidate"], {"front_neutral": 1})
            accepted_path = candidate_dir / "review" / "accepted.json"
            self.assertTrue(accepted_path.exists())

            spec_path = root / "dataset.json"
            write_json(
                spec_path,
                {
                    "$schema": "schemas/lora-dataset.schema.json",
                    "kind": "lora-dataset",
                    "id": "ai51_identity_from_accepted_candidates",
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "sources": [
                        {
                            "type": "candidate_review",
                            "path": accepted_path.as_posix(),
                            "tags": ["canon-worthy accepted candidate"],
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
            dataset = build_lora_dataset(spec_path, progress=SILENT_STATUS)
            self.assertEqual(dataset["accepted_image_count"], 1)
            output_dir = Path(dataset["output"]["directory"])
            records = [json.loads(line) for line in (output_dir / "metadata.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["source_kind"], "candidate_review")
            self.assertEqual(records[0]["source_metadata"]["approval"]["mode"], "human_approved_lora_candidate")
            self.assertEqual(records[0]["source_metadata"]["candidate"]["name"], "front_neutral")
            self.assertTrue(records[0]["prompt"].startswith("ai51char, "))
            self.assertNotIn("canon-worthy accepted candidate", records[0]["prompt"])
            self.assertIn("canon-worthy accepted candidate", records[0]["tags"])

    def test_lora_candidate_plan_rejects_trigger_token_in_generation_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "front.png"
            write_training_source(front, (180, 50, 60), "F")
            canon_dir = root / "canon"
            main(
                [
                    "lora",
                    "canon-init",
                    "--character-id",
                    "ai51",
                    "--trigger-token",
                    "ai51char",
                    "--identity-prompt",
                    "1girl, blue eyes, short hair, brown hair, leather skirt, belt, blue necktie",
                    "--anchor",
                    f"front={front.as_posix()}",
                    "--output-dir",
                    canon_dir.as_posix(),
                    "--compact",
                ]
            )
            candidate_dir = root / "candidates"
            brief_path = root / "candidate_brief.json"
            write_json(
                brief_path,
                {
                    "$schema": "schemas/lora-candidate-brief.schema.json",
                    "kind": "lora-candidate-brief",
                    "id": "ai51.identity.candidates",
                    "character": {"canon": canon_dir.as_posix()},
                    "generation": {
                        "width": 96,
                        "height": 128,
                        "steps": 20,
                        "seed_start": 20,
                        "seeds_per_candidate": 1,
                    },
                    "candidates": [
                        {
                            "name": "front_neutral",
                            "view": "front",
                            "pose": "neutral standing",
                            "identity_primer": "front",
                            "prompt": {
                                "positive": (
                                    "ai51char, anime-style full-body illustration, front view, "
                                    "neutral standing pose, plain studio background"
                                )
                            },
                        }
                    ],
                    "output": {"directory": candidate_dir.as_posix(), "overwrite": True},
                },
            )

            with self.assertRaisesRegex(LoraCandidateError, "LoRA trigger token"):
                plan_lora_candidates(brief_path=brief_path, progress=SILENT_STATUS)
            self.assertFalse(candidate_dir.exists())

    def test_cli_lora_schemas_have_no_job_version_field(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = main(["lora", "dataset-schema", "--compact"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["properties"]["kind"]["const"], "lora-dataset")
        self.assertNotIn("schema_" "version", payload["properties"])

        candidate_stdout = io.StringIO()
        with contextlib.redirect_stdout(candidate_stdout):
            candidate_exit = main(["lora", "candidate-brief-schema", "--compact"])

        candidate_payload = json.loads(candidate_stdout.getvalue())
        self.assertEqual(candidate_exit, 0)
        self.assertEqual(candidate_payload["properties"]["kind"]["const"], "lora-candidate-brief")
        self.assertNotIn("schema_" "version", candidate_payload["properties"])

    def test_lora_training_dataset_materializes_prompt_column(self) -> None:
        from datasets import load_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "assets" / "front.png"
            write_training_source(image, (80, 120, 190), "F")
            canon_dir = root / "canon"
            main(
                [
                    "lora",
                    "canon-init",
                    "--character-id",
                    "ai51",
                    "--trigger-token",
                    "ai51char",
                    "--identity-prompt",
                    "1girl, blue eyes, short hair, brown hair, leather skirt, blue necktie",
                    "--anchor",
                    f"front={image.as_posix()}",
                    "--output-dir",
                    canon_dir.as_posix(),
                    "--compact",
                ]
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
                            "type": "canon",
                            "path": canon_dir.as_posix(),
                            "images": ["front"],
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

            materialize_captioned_train_dataset(Path(dataset_result["output"]["directory"]), train_dataset_dir)

            loaded = load_dataset(train_dataset_dir.as_posix())
            self.assertEqual(loaded["train"].column_names, ["image", "prompt"])
            self.assertEqual(
                loaded["train"][0]["prompt"],
                "ai51char, 1girl, blue eyes, short hair, brown hair, leather skirt, blue necktie, front",
            )

    def test_rejects_keyframe_run_sources_for_identity_lora_dataset(self) -> None:
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

            with self.assertRaisesRegex(LoraDatasetError, "literal_error"):
                load_lora_dataset_spec(spec_path)

    def test_lora_train_plan_builds_local_16gb_launch_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "assets" / "front.png"
            write_training_source(image, (80, 120, 190), "F")
            canon_dir = root / "canon"
            main(
                [
                    "lora",
                    "canon-init",
                    "--character-id",
                    "ai51",
                    "--trigger-token",
                    "ai51char",
                    "--identity-prompt",
                    "1girl, blue eyes, short hair, brown hair, leather skirt, blue necktie",
                    "--anchor",
                    f"front={image.as_posix()}",
                    "--output-dir",
                    canon_dir.as_posix(),
                    "--compact",
                ]
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
                            "type": "canon",
                            "path": canon_dir.as_posix(),
                            "images": ["front"],
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
            for entry in (
                "scheduler",
                "text_encoder",
                "text_encoder_2",
                "tokenizer",
                "tokenizer_2",
                "transformer",
                "vae",
            ):
                (base_model / entry).mkdir(parents=True)
            write_json(base_model / "model_index.json", {"_class_name": "FluxPipeline"})
            write_json(
                base_model / "transformer" / "config.json",
                {
                    "_class_name": "FluxTransformer2DModel",
                    "quantization_config": {
                        "load_in_4bit": True,
                    },
                },
            )

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
            self.assertEqual(plan["model"]["base_model_kind"], "local_4bit_flux_transformer")
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
            self.assertNotIn("--center_crop", command)
            self.assertEqual(command.count("--mixed_precision"), 1)
            self.assertIn("bf16", command)
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

    def test_lora_train_plan_requires_complete_flux_model_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "assets" / "front.png"
            write_training_source(image, (80, 120, 190), "F")
            canon_dir = root / "canon"
            main(
                [
                    "lora",
                    "canon-init",
                    "--character-id",
                    "ai51",
                    "--trigger-token",
                    "ai51char",
                    "--identity-prompt",
                    "1girl, blue eyes, short hair, brown hair, leather skirt, blue necktie",
                    "--anchor",
                    f"front={image.as_posix()}",
                    "--output-dir",
                    canon_dir.as_posix(),
                    "--compact",
                ]
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
                            "type": "canon",
                            "path": canon_dir.as_posix(),
                            "images": ["front"],
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
            base_model = root / "models" / "incomplete-flux"
            (base_model / "transformer").mkdir(parents=True)

            plan = build_lora_train_plan(
                Path(dataset_result["output"]["directory"]),
                trainer_script=trainer_script,
                base_model=base_model,
                output_dir=root / "lora-output",
            )

            self.assertEqual(plan["status"], "missing_local_inputs")
            self.assertIn((base_model / "model_index.json").as_posix(), plan["missing"])
            self.assertIn((base_model / "tokenizer").as_posix(), plan["missing"])
            self.assertIn((base_model / "vae").as_posix(), plan["missing"])

    def test_lora_control_audit_plan_requires_trained_weights_and_plain_nunchaku(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lora_run = root / "lora-run"
            lora_run.mkdir()
            write_json(
                lora_run / "dataset" / "dataset_report.json",
                {
                    "status": "completed",
                    "dataset_id": "ai51_identity",
                    "accepted_image_count": 2,
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "split_counts": {"train": 2, "val": 0},
                    "records": [],
                },
            )
            base_model = root / "models" / "FLUX.1-dev-bf16"
            controlnet_model = root / "models" / "ControlNet"
            control_image = root / "assets" / "side_idle.png"
            baseline_image = root / "assets" / "side_idle_baseline.png"
            write_training_source(control_image, (10, 10, 10), "P")
            write_training_source(baseline_image, (20, 20, 20), "B")
            base_model.mkdir(parents=True)
            controlnet_model.mkdir()
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "lora",
                        "control-audit-plan",
                        lora_run.as_posix(),
                        "--case",
                        f"side_idle={control_image.as_posix()}",
                        "--baseline",
                        f"side_idle={baseline_image.as_posix()}",
                        "--case-prompt",
                        "side_idle=ai51char, approved identity prompt, full body side idle pose",
                        "--base-model",
                        base_model.as_posix(),
                        "--controlnet-model",
                        controlnet_model.as_posix(),
                        "--nunchaku-transformer",
                        (root / "models" / "nunchaku" / "svdq-fp4_r32-flux.1-dev.safetensors").as_posix(),
                        "--compact",
                    ]
                )

            result = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(result["kind"], "lora-control-audit-plan")
            self.assertEqual(result["status"], "missing_local_inputs")
            self.assertEqual(result["trigger_token"], "ai51char")
            self.assertFalse(result["runtime"]["uses_kontext_reference"])
            self.assertEqual(result["runtime"]["reference_tokens"], 0)
            self.assertIn((lora_run / "pytorch_lora_weights.safetensors").as_posix(), result["missing"])
            self.assertIn(
                (root / "models" / "nunchaku" / "svdq-fp4_r32-flux.1-dev.safetensors").as_posix(),
                result["missing"],
            )
            self.assertEqual([case["name"] for case in result["audit_cases"]], ["side_idle"])
            self.assertEqual(
                result["audit_cases"][0]["prompt"],
                "ai51char, approved identity prompt, full body side idle pose",
            )
            self.assertEqual(result["audit_cases"][0]["control_image"]["path"], control_image.resolve().as_posix())
            self.assertEqual(result["audit_cases"][0]["baseline_image"]["path"], baseline_image.resolve().as_posix())

    def test_lora_control_audit_plan_rejects_case_prompt_without_trigger_token(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lora_run = root / "lora-run"
            lora_run.mkdir()
            write_json(
                lora_run / "dataset" / "dataset_report.json",
                {
                    "status": "completed",
                    "dataset_id": "ai51_identity",
                    "accepted_image_count": 1,
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "split_counts": {"train": 1, "val": 0},
                    "records": [],
                },
            )
            control_image = root / "assets" / "side_idle.png"
            baseline_image = root / "assets" / "side_idle_baseline.png"
            write_training_source(control_image, (10, 10, 10), "P")
            write_training_source(baseline_image, (20, 20, 20), "B")
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        "lora",
                        "control-audit-plan",
                        lora_run.as_posix(),
                        "--case",
                        f"side_idle={control_image.as_posix()}",
                        "--baseline",
                        f"side_idle={baseline_image.as_posix()}",
                        "--case-prompt",
                        "side_idle=approved identity prompt without trigger",
                        "--compact",
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("Case prompt must include the LoRA trigger token", stderr.getvalue())

    def test_lora_control_audit_plan_rejects_missing_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lora_run = root / "lora-run"
            lora_run.mkdir()
            write_json(
                lora_run / "dataset" / "dataset_report.json",
                {
                    "status": "completed",
                    "dataset_id": "ai51_identity",
                    "accepted_image_count": 1,
                    "character": {"id": "ai51", "trigger_token": "ai51char"},
                    "split_counts": {"train": 1, "val": 0},
                    "records": [],
                },
            )
            control_image = root / "assets" / "side_idle.png"
            write_training_source(control_image, (10, 10, 10), "P")
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "lora",
                            "control-audit-plan",
                            lora_run.as_posix(),
                            "--case",
                            f"side_idle={control_image.as_posix()}",
                            "--case-prompt",
                            "side_idle=ai51char, approved identity prompt",
                            "--compact",
                        ]
                    )

            self.assertNotEqual(raised.exception.code, 0)
            self.assertIn("required: --baseline", stderr.getvalue())

if __name__ == "__main__":
    unittest.main()
