from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from aigen.cli import build_parser, main
from aigen.generation.flux_components import (
    CLIP_TEXT_ENCODER_COMPONENT,
    T5_TEXT_ENCODER_COMPONENT,
    T5_TOKENIZER_COMPONENT,
)
from aigen.generation.kontext_identity import _pipeline_device_report
from aigen.keyframe_image_ops import save_contact_sheet
from aigen.lora_candidate_models import (
    LoraCandidateBriefError,
    LoraCandidatePromptSpec,
    LoraCandidateTemplateSpec,
    load_lora_candidate_brief,
)
from aigen.lora_dataset_models import LoraDatasetError, load_lora_dataset_spec
from aigen.lora_datasets import build_lora_dataset
from aigen.lora_candidates import LoraCandidateError, plan_lora_candidates, run_lora_candidate_plan
from aigen.lora_candidate_profiles import LoraCandidateProfile
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


class LoraCandidateCommandTests(unittest.TestCase):
    def test_candidate_brief_plan_defaults_to_four_seed_screening_batch(self) -> None:
        args = build_parser().parse_args(
            [
                "lora",
                "candidate-brief-plan",
                "canon",
                "--output",
                "brief.json",
                "--candidate-output-dir",
                "runs/candidates",
            ]
        )

        self.assertEqual(args.seeds_per_candidate, 4)


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
        if "name: left_profile_neutral_full_body" in prompt:
            candidate = {
                "name": "left_profile_neutral_full_body",
                "view": "left profile view",
                "pose": "neutral standing pose",
                "framing": "full body",
                "identity_primer": "left_profile",
                "prompt": {
                    "positive": (
                        "anime-style full-body illustration, left profile view, neutral standing pose, "
                        "plain studio background"
                    )
                },
            }
        elif "name: front_neutral_thigh_up" in prompt:
            candidate = {
                "name": "front_neutral_thigh_up",
                "view": "front view",
                "pose": "neutral standing pose",
                "framing": "thigh-up",
                "identity_primer": "front",
                "prompt": {
                    "positive": (
                        "anime-style thigh-up illustration, front view, neutral standing pose, "
                        "looking at viewer, smile, light blush, plain studio background"
                    )
                },
            }
        else:
            candidate = {
                "name": "front_neutral_full_body",
                "view": "front view",
                "pose": "neutral standing pose",
                "framing": "full body",
                "identity_primer": "front",
                "prompt": {
                    "positive": (
                        "anime-style full-body illustration, front view, neutral standing pose, "
                        "looking at viewer, smile, light blush, plain studio background"
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
                        "framing": "full body",
                        "identity_primer": "front",
                        "prompt": {
                            "positive": (
                                "anime-style full-body illustration, front view, neutral standing pose, "
                                "plain studio background"
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
                    "childlike_or_chibi_proportions": False,
                    "wrong_visible_outfit": False,
                    "missing_visible_required_identity_detail": False,
                    "deformed_visible_anatomy": False,
                    "broken_visible_hands_or_feet": False,
                    "framing_mismatch": False,
                    "dirty_or_distracting_background": False,
                    "style_drift": False,
                    "view_label_mismatch": False,
                },
                "scores": {
                    "identity_preservation": 9,
                    "visible_outfit_preservation": 9,
                    "visible_anatomy_quality": 9,
                    "petite_proportion_preservation": 9,
                    "framing_quality": 9,
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


class FakeLoraFreeCandidateJudge:
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
                    "childlike_or_chibi_proportions": False,
                    "wrong_visible_outfit": False,
                    "missing_visible_required_identity_detail": False,
                    "deformed_visible_anatomy": False,
                    "broken_visible_hands_or_feet": False,
                    "awkward_crop_or_cutoff": False,
                    "dirty_or_distracting_background": False,
                    "style_drift": False,
                },
                "scores": {
                    "identity_preservation": 9,
                    "visible_outfit_preservation": 9,
                    "visible_anatomy_quality": 9,
                    "petite_proportion_preservation": 9,
                    "framing_quality": 9,
                    "background_quality": 9,
                    "style_match": 9,
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


class FakeLoraCandidateCaptioner:
    device_report = {"modules": []}

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.image_paths: list[list[Path]] = []
        self.closed = False

    def describe_image(self, prompt: str, image_paths: list[Path]) -> str:
        self.prompts.append(prompt)
        self.image_paths.append(image_paths)
        return (
            "short brown hair, blue eyes, brown leather jacket, front view, "
            "standing, thigh-up, plain grey background, anime style"
        )

    def close(self) -> None:
        self.closed = True


class FakeKontextIdentityPipeline:
    model_cpu_offload_seq = "text_encoder->t5_text_encoder->transformer->vae"

    def __init__(self) -> None:
        import torch

        self.transformer = torch.nn.Linear(1, 1)
        self.vae = torch.nn.Linear(1, 1)
        setattr(self, CLIP_TEXT_ENCODER_COMPONENT, torch.nn.Linear(1, 1))
        setattr(self, T5_TEXT_ENCODER_COMPONENT, torch.nn.Linear(1, 1))


class FakeCuda:
    def is_available(self) -> bool:
        return False


class FakeTorch:
    cuda = FakeCuda()


class FakeKontextIdentitySession:
    last: FakeKontextIdentitySession

    def __init__(self, *args, **kwargs) -> None:
        self.torch = FakeTorch()
        self.model_load_ms = 7.0
        self.generated: list[dict[str, object]] = []
        self.closed = False
        FakeKontextIdentitySession.last = self

    def generate(self, **kwargs):
        self.generated.append(kwargs)
        image = Image.new("RGB", (kwargs["width"], kwargs["height"]), (100, 120, 140))
        return image, {"pipeline_ms": 3.0}

    def environment(self) -> dict[str, object]:
        return {"prompt_encoding": "precomputed_prompt_embeds"}

    def close(self) -> None:
        self.closed = True


class FakeMemorySampler:
    def __init__(self, preflight: dict[str, int]) -> None:
        self.preflight = preflight
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> dict[str, int]:
        return {**self.preflight, "nvidia_smi_peak_used_mb": self.preflight["nvidia_smi_preflight_used_mb"]}


class LoraDatasetTests(unittest.TestCase):
    def test_kontext_identity_device_report_uses_pipeline_components(self) -> None:
        report = _pipeline_device_report(FakeKontextIdentityPipeline())

        self.assertEqual(report["pipeline_class"], "FakeKontextIdentityPipeline")
        self.assertEqual(
            sorted(report["components"]),
            ["clip_text_encoder", "t5_text_encoder", "transformer", "vae"],
        )
        self.assertEqual(report["components"]["transformer"]["class"], "Linear")

        textless_pipeline = FakeKontextIdentityPipeline()
        setattr(textless_pipeline, CLIP_TEXT_ENCODER_COMPONENT, None)
        setattr(textless_pipeline, T5_TEXT_ENCODER_COMPONENT, None)
        textless_report = _pipeline_device_report(textless_pipeline)
        self.assertEqual(sorted(textless_report["components"]), ["transformer", "vae"])

    def test_contact_sheet_uses_tiled_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs = []
            for index in range(10):
                path = root / f"image_{index}.png"
                write_training_source(path, (index * 20, 80, 160), str(index))
                outputs.append({"name": f"candidate_{index}", "path": path.as_posix()})

            sheet_path = root / "contact_sheet.png"
            save_contact_sheet(outputs, sheet_path, thumb_width=48, max_columns=4)

            with Image.open(sheet_path) as sheet:
                self.assertEqual(sheet.size, (192, 288))

    def test_identity_prompt_is_kept_as_single_generation_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "front.png"
            write_training_source(front, (180, 50, 60), "F")
            identity_prompt = (
                "A young woman with short brown hair styled in a slightly tousled bob, bright blue eyes with "
                "and a small petite build with slender proportions. She wears a fitted brown leather jacket "
                "with silver zipper hardware and structured shoulders, over a crisp white button-up collared "
                "shirt. Around her neck hangs a bold blue necktie, matching her blue thigh-high stockings. "
                "She wears a fitted brown leather mini skirt with a matching leather belt with a silver "
                "rectangular buckle, brown leather knee-high boots with low heels and buckle details, and "
                "brown leather gloves."
            )
            output_dir = root / "canon"

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
                    output_dir.as_posix(),
                    "--compact",
                ]
            )

            manifest = json.loads((output_dir / "canon_manifest.json").read_text(encoding="utf-8"))
            caption = (output_dir / "images" / "front.txt").read_text(encoding="utf-8").strip()
            self.assertEqual(manifest["character"]["identity_prompt"], identity_prompt)
            self.assertNotIn("base_identity_prompt", manifest["character"])
            self.assertNotIn("generation_style_prompt", manifest["character"])
            self.assertNotIn("variable_prompt_terms", manifest["character"])
            self.assertEqual(caption, f"ai51char, {identity_prompt.rstrip('.')}, front")

    def test_lora_candidate_prompt_schema_is_not_character_specific(self) -> None:
        prompt = LoraCandidatePromptSpec(
            positive="rear camera view, neutral standing pose, full body, plain studio background"
        )

        self.assertEqual(
            prompt.positive,
            "rear camera view, neutral standing pose, full body, plain studio background",
        )

    def test_lora_candidate_prompt_must_materialize_view_and_pose(self) -> None:
        with self.assertRaisesRegex(ValueError, "view term"):
            LoraCandidateTemplateSpec(
                name="left_profile_idle",
                view="left profile view",
                pose="idle standing pose",
                framing="full body",
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
                framing="full body",
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
                framing="full body",
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
                framing="full body",
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
                framing="full body",
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
                framing="full body",
                identity_primer="front",
                prompt={
                    "positive": (
                        "Anime-style full-body illustration, back view, neutral standing pose, "
                        "blue eyes and smiling face, brown leather jacket, leather skirt, plain studio background"
                    )
                },
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
                            "framing": "full body",
                            "identity_primer": "front",
                            "prompt": {
                                "positive": (
                                    "Clean illustration style, front view, neutral standing walk contact pose, "
                                    "full body, plain studio background"
                                ),
                            },
                        },
                        {
                            "name": "front_walk_contact",
                            "view": "front view",
                            "pose": "walk contact",
                            "framing": "full body",
                            "identity_primer": "front",
                            "prompt": {
                                "positive": (
                                    "Clean illustration style, front view, neutral standing walk contact pose, "
                                    "full body, plain studio background"
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
            self.assertIn("looking at viewer", left_caption)
            self.assertNotIn("base_identity_prompt", result["character"])
            self.assertNotIn("variable_prompt_terms", result["character"])
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
            self.assertIn("The executor prepends this full user identity prompt", first_prompt)
            self.assertIn("Write exactly one variable generation prompt suffix", first_prompt)
            self.assertIn("Available identity primer names: front, left_profile", first_prompt)
            self.assertIn('first character of your response must be "{"', first_prompt)
            self.assertIn('last character must be "}"', first_prompt)
            self.assertNotIn("Every positive prompt must contain these exact literal phrases", first_prompt)
            self.assertNotIn("- clean anime lineart", first_prompt)
            self.assertIn("The executor adds the full user identity prompt before this suffix", first_prompt)
            self.assertEqual(len(planner.image_paths), 2)
            self.assertEqual(brief["kind"], "lora-candidate-brief")
            self.assertNotIn("schema_version", brief)
            self.assertEqual(brief["id"], "ai51.lora.candidates")
            self.assertEqual(brief["generation"]["seeds_per_candidate"], 3)
            self.assertEqual(
                [candidate["name"] for candidate in brief["candidates"]],
                ["front_neutral_full_body", "front_neutral_thigh_up"],
            )
            self.assertIn("front view", brief["candidates"][0]["prompt"]["positive"])
            self.assertNotIn("leather skirt", brief["candidates"][0]["prompt"]["positive"])
            self.assertNotIn("ai51char", brief["candidates"][0]["prompt"]["positive"])
            self.assertEqual(brief["candidates"][1]["framing"], "thigh-up")
            self.assertTrue((brief_path.with_suffix(".raw") / "front_neutral_full_body.txt").exists())
            self.assertTrue((brief_path.with_suffix(".prompts") / "front_neutral_full_body.txt").exists())

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
            self.assertTrue((root / "canon_audit" / "contact_sheet.png").exists())
            self.assertFalse((root / "canon_audit" / "crops").exists())

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
                            "framing": "full body",
                            "identity_primer": "front",
                            "prompt": {
                                "positive": (
                                    "anime-style full-body illustration, front view, neutral standing pose, "
                                    "looking at viewer, smile, light blush, plain studio background"
                                ),
                            },
                        },
                        {
                            "name": "walk_contact",
                            "view": "side",
                            "pose": "walk contact",
                            "framing": "full body",
                            "identity_primer": "front",
                            "prompt": {
                                "positive": (
                                    "anime-style full-body illustration, side-view walk contact pose, "
                                    "plain studio background"
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
            self.assertTrue(first_candidate["generation_prompt"].startswith(identity_prompt))
            self.assertTrue(
                first_candidate["generation_prompt"].endswith(
                    planned_manifest["candidate_templates"][0]["prompt"]["positive"]
                )
            )
            self.assertIn("brown leather jacket", first_candidate["generation_prompt"])
            self.assertIn("leather skirt", first_candidate["generation_prompt"])
            self.assertNotIn("ai51char", first_candidate["generation_prompt"])
            self.assertEqual(first_candidate["training_caption"], f"ai51char, {first_candidate['generation_prompt']}")
            self.assertIn("smile", first_candidate["training_caption"])
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
            self.assertEqual(evidence_result["review_items"][0]["evidence"]["image"]["path"], selected_path.as_posix())
            self.assertNotIn("crop_sheet", evidence_result["review_items"][0]["evidence"])

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
            self.assertEqual(len(judge_runner.image_paths[0]), 2)
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

    def test_lora_candidate_run_precomputes_unique_prompt_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "front.png"
            write_training_source(front, (180, 50, 60), "F")
            canon_dir = root / "canon"
            identity_prompt = "1girl, blue eyes, brown leather jacket, leather skirt"
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
                    "character": {"canon": canon_dir.as_posix()},
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
                            "pose": "neutral standing",
                            "framing": "full body",
                            "identity_primer": "front",
                            "prompt": {
                                "positive": "front view, neutral standing pose, full body, plain studio background",
                            },
                        },
                        {
                            "name": "side_idle",
                            "view": "side",
                            "pose": "idle standing",
                            "framing": "full body",
                            "identity_primer": "front",
                            "prompt": {
                                "positive": "left side view, idle standing pose, full body, plain studio background",
                            },
                        },
                    ],
                    "output": {"directory": candidate_dir.as_posix(), "overwrite": True},
                },
            )
            plan_lora_candidates(brief_path=brief_path, progress=SILENT_STATUS)
            encoded: dict[str, object] = {}

            def fake_encode(model: str, *, prompts: list[str], dtype: str, max_sequence_length: int):
                encoded.update(
                    {
                        "model": model,
                        "prompts": prompts,
                        "dtype": dtype,
                        "max_sequence_length": max_sequence_length,
                    }
                )
                return {prompt: f"embedding:{index}" for index, prompt in enumerate(prompts)}, 12.5

            profile = LoraCandidateProfile(
                name="test-profile",
                model="/models/kontext",
                nunchaku_transformer_model=Path("/models/nunchaku.safetensors"),
                attention_impl="nunchaku-fp16",
                dtype="bfloat16",
                vae_tiling=False,
                model_revisions={
                    "kontext": {"repo_id": "kontext", "revision": "kontext-rev"},
                    "nunchaku_transformer": {"repo_id": "nunchaku", "revision": "nunchaku-rev"},
                },
            )

            with (
                patch("aigen.lora_candidates.encode_flux_prompts", side_effect=fake_encode),
                patch("aigen.lora_candidates.CharacterKontextIdentitySession", FakeKontextIdentitySession),
                patch(
                    "aigen.lora_candidates.nvidia_smi_preflight_limit",
                    return_value={
                        "nvidia_smi_preflight_used_mb": 100,
                        "nvidia_smi_device_total_mb": 16000,
                        "nvidia_smi_preflight_utilization_gpu": 0,
                    },
                ),
                patch("aigen.lora_candidates.NvidiaSmiMemorySampler", FakeMemorySampler),
            ):
                result = run_lora_candidate_plan(
                    candidate_dir,
                    profile=profile,
                    guidance_scale=2.5,
                    max_sequence_length=128,
                    overwrite=True,
                    progress=SILENT_STATUS,
                )

            session = FakeKontextIdentitySession.last
            self.assertTrue(session.closed)
            self.assertEqual(encoded["model"], "/models/kontext")
            self.assertEqual(len(encoded["prompts"]), 2)
            self.assertEqual(encoded["dtype"], "bfloat16")
            self.assertEqual(encoded["max_sequence_length"], 128)
            self.assertEqual(len(session.generated), 4)
            self.assertEqual(
                [item["prompt_embedding"] for item in session.generated],
                ["embedding:0", "embedding:0", "embedding:1", "embedding:1"],
            )
            self.assertEqual(result["profile"]["prompt_encoding"], "precomputed_prompt_embeds")
            self.assertNotIn("pipeline_cpu_offload", result["profile"])
            self.assertEqual(result["timings_ms"]["prompt_encode_ms"], 12.5)

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
                            "framing": "full body",
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

    def test_lora_freegen_brief_validates_buckets(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from aigen.lora_candidate_models import LoraFreeGenBriefSpec

        base = {
            "$schema": "schemas/lora-freegen-brief.schema.json",
            "kind": "lora-freegen-brief",
            "id": "ai51.lora.freegen",
            "character": {"canon": "canon"},
            "generation": {
                "buckets": [{"width": 96, "height": 128}],
                "steps": 20,
                "seed_start": 0,
                "seeds_per_bucket": 1,
            },
            "output": {"directory": "out", "overwrite": True},
        }
        LoraFreeGenBriefSpec.model_validate(base)

        bad_width = json.loads(json.dumps(base))
        bad_width["generation"]["buckets"] = [{"width": 100, "height": 128}]
        with self.assertRaisesRegex(PydanticValidationError, "divisible by 16"):
            LoraFreeGenBriefSpec.model_validate(bad_width)

        duplicate = json.loads(json.dumps(base))
        duplicate["generation"]["buckets"] = [
            {"width": 96, "height": 128},
            {"width": 96, "height": 128},
        ]
        with self.assertRaisesRegex(PydanticValidationError, "buckets must be unique"):
            LoraFreeGenBriefSpec.model_validate(duplicate)

    def test_lora_freegen_funnel_plans_dedupes_captions_and_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "front.png"
            side = root / "left_profile.png"
            write_training_source(front, (180, 50, 60), "F")
            write_training_source(side, (60, 90, 200), "L")
            identity_prompt = "1girl, short brown hair, blue eyes, brown leather jacket, blue necktie"
            canon_dir = root / "canon"
            with contextlib.redirect_stdout(io.StringIO()):
                canon_exit = main(
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
                        f"left_profile={side.as_posix()}",
                        "--output-dir",
                        canon_dir.as_posix(),
                        "--compact",
                    ]
                )
            self.assertEqual(canon_exit, 0)

            candidate_dir = root / "freegen"
            brief_path = root / "freegen_brief.json"
            write_json(
                brief_path,
                {
                    "$schema": "schemas/lora-freegen-brief.schema.json",
                    "kind": "lora-freegen-brief",
                    "id": "ai51.lora.freegen",
                    "character": {"canon": canon_dir.as_posix()},
                    "generation": {
                        "buckets": [{"width": 96, "height": 128}],
                        "steps": 20,
                        "seed_start": 5,
                        "seeds_per_bucket": 2,
                    },
                    "output": {"directory": candidate_dir.as_posix(), "overwrite": True},
                },
            )

            plan_stdout = io.StringIO()
            with contextlib.redirect_stdout(plan_stdout):
                plan_exit = main(["lora", "freegen-plan", brief_path.as_posix(), "--compact"])
            plan_result = json.loads(plan_stdout.getvalue())
            self.assertEqual(plan_exit, 0)
            self.assertEqual(plan_result["planning_mode"], "free")
            self.assertEqual(
                plan_result["counts"],
                {"identity_primers": 2, "buckets": 1, "seeds_per_bucket": 2, "candidates": 4},
            )
            manifest = json.loads((candidate_dir / "candidates.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [candidate["name"] for candidate in manifest["candidates"]],
                [
                    "free_front_96x128_seed_0005",
                    "free_front_96x128_seed_0006",
                    "free_left_profile_96x128_seed_0005",
                    "free_left_profile_96x128_seed_0006",
                ],
            )
            for candidate in manifest["candidates"]:
                self.assertEqual(candidate["generation_prompt"], identity_prompt)
                self.assertNotIn("training_caption", candidate)
                self.assertEqual(candidate["candidate"]["mode"], "free")
                self.assertTrue((candidate_dir / candidate["generation_prompt_file"]).exists())

            def write_candidate_image(name: str, box: tuple[int, int, int, int]) -> None:
                image = Image.new("RGB", (96, 128), (30, 30, 30))
                draw = ImageDraw.Draw(image)
                draw.rectangle(box, fill="white")
                image.save(candidate_dir / "images" / f"{name}.png")

            write_candidate_image("free_front_96x128_seed_0005", (0, 0, 47, 127))
            write_candidate_image("free_front_96x128_seed_0006", (0, 0, 47, 127))
            write_candidate_image("free_left_profile_96x128_seed_0005", (48, 0, 95, 127))

            evidence_stdout = io.StringIO()
            with contextlib.redirect_stdout(evidence_stdout):
                evidence_exit = main(
                    [
                        "lora",
                        "candidate-evidence",
                        candidate_dir.as_posix(),
                        "--dedupe-threshold",
                        "0",
                        "--compact",
                    ]
                )
            evidence_result = json.loads(evidence_stdout.getvalue())
            self.assertEqual(evidence_exit, 0)
            self.assertEqual(
                evidence_result["counts"],
                {"candidates": 4, "review_items": 2, "rejected_images": 2},
            )
            self.assertEqual(evidence_result["dedupe"], {"threshold": 0, "near_duplicates": 1})
            rejected_by_status = {item["status"]: item for item in evidence_result["rejected_images"]}
            self.assertEqual(
                rejected_by_status["near_duplicate_image"]["evidence"]["near_duplicate_of"],
                "free_front_96x128_seed_0005",
            )
            self.assertIn("missing_candidate_image", rejected_by_status)

            judge_runner = FakeLoraFreeCandidateJudge()
            judge_stdout = io.StringIO()
            with (
                patch("aigen.lora_candidate_judge.QwenVlm", return_value=judge_runner),
                contextlib.redirect_stdout(judge_stdout),
            ):
                judge_exit = main(["lora", "candidate-judge", candidate_dir.as_posix(), "--compact"])
            judge_result = json.loads(judge_stdout.getvalue())
            self.assertEqual(judge_exit, 0)
            self.assertEqual(judge_result["counts"], {"review_items": 2, "passed": 2, "blocked": 0})
            self.assertIn("without a requested view, pose or framing", judge_runner.prompts[0])
            self.assertIn("awkward_crop_or_cutoff", judge_runner.prompts[0])
            self.assertNotIn("view_label_mismatch", judge_runner.prompts[0])

            uncaptioned_review_stderr = io.StringIO()
            with contextlib.redirect_stderr(uncaptioned_review_stderr):
                uncaptioned_review_exit = main(
                    [
                        "lora",
                        "candidate-review",
                        candidate_dir.as_posix(),
                        "--accept",
                        "free_front_96x128_seed_0005",
                        "--compact",
                    ]
                )
            self.assertEqual(uncaptioned_review_exit, 1)
            self.assertIn("candidate-caption", uncaptioned_review_stderr.getvalue())

            captioner = FakeLoraCandidateCaptioner()
            caption_stdout = io.StringIO()
            with (
                patch("aigen.lora_candidate_captions.QwenVlm", return_value=captioner),
                contextlib.redirect_stdout(caption_stdout),
            ):
                caption_exit = main(["lora", "candidate-caption", candidate_dir.as_posix(), "--compact"])
            caption_result = json.loads(caption_stdout.getvalue())
            self.assertEqual(caption_exit, 0)
            self.assertEqual(caption_result["counts"], {"captioned": 2})
            self.assertTrue(captioner.closed)
            self.assertIn("Describe only what is visible", captioner.prompts[0])
            caption_file = candidate_dir / "captions" / "free_front_96x128_seed_0005.txt"
            self.assertEqual(
                caption_file.read_text(encoding="utf-8").strip(),
                "ai51char, short brown hair, blue eyes, brown leather jacket, front view, "
                "standing, thigh-up, plain grey background, anime style",
            )

            review_stdout = io.StringIO()
            with contextlib.redirect_stdout(review_stdout):
                review_exit = main(
                    [
                        "lora",
                        "candidate-review",
                        candidate_dir.as_posix(),
                        "--accept",
                        "free_front_96x128_seed_0005",
                        "--approved-by",
                        "boaz",
                        "--compact",
                    ]
                )
            review_result = json.loads(review_stdout.getvalue())
            self.assertEqual(review_exit, 0)
            accepted = json.loads((candidate_dir / "review" / "accepted.json").read_text(encoding="utf-8"))["items"]
            self.assertEqual(len(accepted), 1)
            self.assertTrue(accepted[0]["training_caption"].startswith("ai51char, short brown hair"))
            self.assertEqual(review_result["quota_report"]["by_view"], {"unspecified": 1})

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
                T5_TEXT_ENCODER_COMPONENT,
                "tokenizer",
                T5_TOKENIZER_COMPONENT,
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
