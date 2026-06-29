from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import ANY, patch

from PIL import Image, ImageDraw

from aigen.cli import main
from aigen.keyframe_brief_models import KeyframeBriefError
from aigen.keyframe_brief_planner import plan_keyframe_brief
from aigen.keyframe_briefs import (
    execute_keyframe_brief,
    materialize_keyframe_brief,
)
from aigen.keyframe_judge import (
    DEFAULT_JUDGE_ID,
    DEFAULT_JUDGE_QUANTIZATION,
    DEFAULT_JUDGE_REPO_ID,
    DEFAULT_JUDGE_REVISION,
    KeyframeJudgeConfig,
)
from aigen.prompt_tokens import PromptTokenCounts


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def image_asset(path: Path, size: tuple[int, int], mode: str = "RGB") -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "sha256": "0" * 64,
        "mode": mode,
        "width": size[0],
        "height": size[1],
    }


def fake_dwpose_control_image(image: Image.Image, **_kwargs: object) -> tuple[Image.Image, dict[str, object]]:
    pose = Image.new("RGB", image.size, "black")
    draw = ImageDraw.Draw(pose)
    draw.line((20, 40, image.width - 20, 40), fill=(255, 0, 0), width=5)
    return pose, {"body_count": 1, "visible_body_keypoints": 12, "mean_body_score": 0.80}


def judge_config() -> KeyframeJudgeConfig:
    return KeyframeJudgeConfig(
        judge_id=DEFAULT_JUDGE_ID,
        model=Path("unused"),
        repo_id=DEFAULT_JUDGE_REPO_ID,
        revision=DEFAULT_JUDGE_REVISION,
        dtype="bfloat16",
        attention_impl="sdpa",
        quantization=DEFAULT_JUDGE_QUANTIZATION,
        min_pixels=1,
        max_pixels=1,
        max_new_tokens=1200,
        temperature=0.0,
    )


class FakeBriefPlanner:
    def __init__(self, identity_primer: Path) -> None:
        self.identity_primer = identity_primer
        self.prompt = ""
        self.image_paths: list[Path] = []

    def judge_candidate(self, prompt: str, image_paths: list[Path]) -> str:
        self.prompt = prompt
        self.image_paths = image_paths
        return json.dumps(
            {
                "identity_details": {
                    "subject": "anime girl",
                    "hair": "short pink bob",
                    "face": "one visible side-profile eye and focused expression",
                    "upper_clothing": "glossy brown jacket over white shirt",
                    "neckwear": "blue tie",
                    "waist_garment": "brown leather skirt",
                    "legwear": "blue thigh-high socks",
                    "footwear": "brown boots",
                    "style": "clean anime lineart with glossy highlights",
                },
                "identity_description": "pink bob, glossy brown jacket, blue tie and brown skirt",
                "pose_description": "left-facing platformer attack start with extended forward arm",
                "platformer_camera_description": "readable side-scroller view with a slight camera cheat toward the viewer",
                "identity_primer": {"view": "left_profile", "path": self.identity_primer.as_posix()},
                "prompt": {
                    "clip": "Pink-bob anime character in a readable platformer side-view attack pose.",
                    "t5": (
                        "Generate a readable platformer side-view attack keyframe. Preserve the short pink bob, "
                        "glossy brown jacket, blue tie, brown skirt, blue socks and brown boots."
                    ),
                    "true_cfg_scale": 1.0,
                },
                "canvas": {"width": 160, "height": 240, "reference_max_area": 294912, "max_sequence_length": 128},
                "sampling": {"steps": 28, "guidance_scale": 2.5},
                "controls": [
                    {"name": "source_pose", "type": "pose", "source": "example_pose", "scale": 0.72, "start": 0.0, "end": 0.65},
                    {
                        "name": "source_contour",
                        "type": "canny",
                        "source": "example_contour",
                        "scale": 0.25,
                        "start": 0.0,
                        "end": 0.35,
                        "residual_mask_source": "example_boundary_mask",
                    },
                ],
                "scoring": {
                    "top_k": 3,
                    "priorities": ["source pose match", "platformer action readability", "identity preservation"],
                    "checks": ["reads as platformer attack", "identity primer preserved", "feet visible"],
                },
                "polish": {
                    "profile": "kontext-inpaint-local",
                    "max_regions": 4,
                    "strength_offsets": [-0.06, 0.0, 0.06],
                    "seed_offsets": [0, 1],
                },
                "rationale": ["left_profile primer reduces camera-yaw negotiation"],
            }
        )


class ClosingFakeBriefPlanner(FakeBriefPlanner):
    def __init__(self, identity_primer: Path) -> None:
        super().__init__(identity_primer)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class InvalidBriefPlanner:
    def judge_candidate(self, _prompt: str, _image_paths: list[Path]) -> str:
        return json.dumps(
            {
                "identity_description": "short pink bob and brown leather outfit",
                "pose_description": "platformer punch",
                "platformer_camera_description": "side-scroller readable camera",
                "identity_primer": {"view": "left_profile", "path": "/tmp/missing.png"},
                "prompt": {"clip": "clip", "t5": "t5"},
                "canvas": {"width": 160, "height": 240, "reference_max_area": 294912, "max_sequence_length": 128},
                "sampling": {"steps": 28, "guidance_scale": 2.5},
                "controls": [
                    {"name": "source_pose", "type": "pose", "source": "example_pose", "scale": 0.72, "start": 0.0, "end": 0.65}
                ],
                "scoring": {"top_k": 3, "priorities": ["condition adherence"], "checks": ["pose readable"]},
                "polish": {
                    "profile": "kontext-inpaint-local",
                    "max_regions": 4,
                    "strength_offsets": [-0.06, 0.0, 0.06],
                    "seed_offsets": [0, 1],
                },
                "rationale": ["test invalid output"],
            }
        )


class PlaceholderBriefPlanner(FakeBriefPlanner):
    def judge_candidate(self, prompt: str, image_paths: list[Path]) -> str:
        data = json.loads(super().judge_candidate(prompt, image_paths))
        data["scoring"] = {
            "top_k": 3,
            "priorities": ["condition-first visual priority"],
            "checks": ["concrete visual check"],
        }
        return json.dumps(data)


class RepairingBriefPlanner(FakeBriefPlanner):
    def __init__(self, identity_primer: Path) -> None:
        super().__init__(identity_primer)
        self.calls = 0

    def judge_candidate(self, prompt: str, image_paths: list[Path]) -> str:
        self.calls += 1
        data = json.loads(super().judge_candidate(prompt, image_paths))
        if self.calls == 1:
            del data["prompt"]["true_cfg_scale"]
        return json.dumps(data)


class KeyframeBriefTests(unittest.TestCase):
    def test_cli_briefs_schema_outputs_schema(self) -> None:
        stdout = StringIO()

        with redirect_stdout(stdout):
            exit_code = main(["briefs", "schema", "--compact"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["properties"]["kind"]["const"], "keyframe-brief")

    def test_plans_keyframe_brief_with_vlm_selected_identity_primer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            brief_path, left_profile = write_brief_fixture(root)
            planner = FakeBriefPlanner(left_profile)

            result = plan_keyframe_brief(
                brief_path,
                judge_config(),
                project_root=Path.cwd(),
                runner=planner,
            )

            plan_path = root / "plans" / "punch_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "planned")
            self.assertEqual(plan["identity_primer"]["view"], "left_profile")
            self.assertEqual(plan["controls"][1]["source"], "example_contour")
            self.assertEqual(plan["polish"]["profile"], "kontext-inpaint-local")
            self.assertEqual(plan["polish"]["strength_offsets"], [-0.06, 0.0, 0.06])
            self.assertNotIn("policy", plan["polish"])
            self.assertEqual(plan["identity_details"]["waist_garment"], "brown leather skirt")
            self.assertIn("Platformer side-view animation may cheat", planner.prompt)
            self.assertIn("hair, clothing, colors and style", planner.prompt)
            self.assertIn("Choose control strengths from the image evidence", planner.prompt)
            self.assertIn('control.type is the ControlNet condition type', planner.prompt)
            self.assertIn('never return "source" as a type', planner.prompt)
            self.assertIn('polish.profile must be exactly "kontext-inpaint-local"', planner.prompt)
            self.assertIn("polish.seed_offsets must be integer offsets", planner.prompt)
            self.assertIn("Describe the character's lower body in separate parts", planner.prompt)
            self.assertIn("This is a full-body gameplay keyframe", planner.prompt)
            self.assertIn("Build prompt.clip and prompt.t5 from every identity_details slot", planner.prompt)
            self.assertIn("The example sprite may depict a different character", planner.prompt)
            self.assertNotIn('"scale": 0.72', planner.prompt)
            self.assertNotIn('"scale": 0.25', planner.prompt)
            self.assertEqual(planner.image_paths[0], root / "assets" / "characters" / "ai51" / "views" / "front.png")
            self.assertEqual(planner.image_paths[-1], root / "examples" / "punch.png")

    def test_plan_keyframe_brief_closes_owned_vlm_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            brief_path, left_profile = write_brief_fixture(root)
            planner = ClosingFakeBriefPlanner(left_profile)

            with patch("aigen.keyframe_brief_planner.QwenKeyframeJudge", return_value=planner):
                plan_keyframe_brief(brief_path, judge_config(), project_root=Path.cwd())

        self.assertTrue(planner.closed)

    def test_materializes_brief_to_keyframe_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            brief_path, left_profile = write_brief_fixture(root)
            planner = FakeBriefPlanner(left_profile)
            plan_keyframe_brief(brief_path, judge_config(), project_root=Path.cwd(), runner=planner)

            with (
                patch("aigen.keyframe_examples._dwpose_control_image", fake_dwpose_control_image),
                patch(
                    "aigen.keyframes.count_kontext_prompt_tokens",
                    return_value=PromptTokenCounts(clip=12, clip_limit=77, t5=24),
                ),
            ):
                result = materialize_keyframe_brief(brief_path, project_root=Path.cwd())

            job_path = Path(result["job_path"])
            job = json.loads(job_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "materialized")
            self.assertEqual(job["character"]["identity_primer"]["view"], "left_profile")
            self.assertEqual(job["conditions"][0]["image"], "pose")
            self.assertEqual(job["conditions"][1]["image"], "contour")
            self.assertEqual(job["conditions"][1]["residual_mask"], "boundary_mask")
            self.assertEqual(job["variants"][0], {"name": "seed_060", "seed": 60})
            self.assertEqual(job["variants"][-1], {"name": "seed_063", "seed": 63})
            self.assertTrue((root / "assets" / "extracted" / "platform_punch_pose.png").exists())

    def test_invalid_generated_plan_keeps_raw_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            brief_path, _left_profile = write_brief_fixture(root)

            with self.assertRaisesRegex(KeyframeBriefError, "Invalid generated brief plan"):
                plan_keyframe_brief(
                    brief_path,
                    judge_config(),
                    project_root=Path.cwd(),
                    runner=InvalidBriefPlanner(),
                )

            self.assertTrue((root / "plans" / "punch_plan.raw.txt").exists())

    def test_invalid_generated_plan_rejects_schema_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            brief_path, left_profile = write_brief_fixture(root)

            with self.assertRaisesRegex(KeyframeBriefError, "scoring priorities and checks must be concrete"):
                plan_keyframe_brief(
                    brief_path,
                    judge_config(),
                    project_root=Path.cwd(),
                    runner=PlaceholderBriefPlanner(left_profile),
                )

    def test_repairs_single_schema_error_without_discarding_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            brief_path, left_profile = write_brief_fixture(root)
            planner = RepairingBriefPlanner(left_profile)

            result = plan_keyframe_brief(
                brief_path,
                judge_config(),
                project_root=Path.cwd(),
                runner=planner,
            )

            plan = json.loads((root / "plans" / "punch_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "planned")
            self.assertEqual(planner.calls, 2)
            self.assertEqual(plan["prompt"]["true_cfg_scale"], 1.0)
            self.assertTrue((root / "plans" / "punch_plan.repair.raw.txt").exists())

    def test_execute_brief_scores_selects_and_polishes_top_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            brief_path, _left_profile = write_brief_fixture(root)
            run_dir = root / "runs" / "keyframes"

            with (
                patch(
                    "aigen.keyframe_brief_planner.plan_keyframe_brief",
                    return_value={
                        "brief_id": "ai51.punch.platformer.left",
                        "plan_path": (root / "plans" / "punch_plan.json").as_posix(),
                        "scoring": {"top_k": 2},
                    },
                ) as plan_mock,
                patch(
                    "aigen.keyframe_briefs.run_keyframe_brief",
                    return_value={
                        "job_path": (root / "jobs" / "punch_job.json").as_posix(),
                        "run_dir": run_dir.as_posix(),
                        "result": {"effective_config": {"output": {"directory": run_dir.as_posix()}}},
                    },
                ) as run_mock,
                patch("aigen.keyframe_briefs.score_keyframe_run", return_value={"status": "completed"}) as score_mock,
                patch("aigen.keyframe_briefs.judge_keyframe_run", return_value={"status": "completed"}) as judge_mock,
                patch(
                    "aigen.keyframe_briefs.select_scored_keyframe_run",
                    return_value={"selected": ["seed_060", "seed_061"]},
                ) as select_mock,
                patch(
                    "aigen.keyframe_briefs._polish_selected_candidates",
                    return_value=[{"candidate": "seed_060"}, {"candidate": "seed_061"}],
                ) as polish_mock,
            ):
                result = execute_keyframe_brief(brief_path, judge_config(), project_root=Path.cwd())

            self.assertEqual(result["selection"]["selected"], ["seed_060", "seed_061"])
            self.assertEqual(result["judge"], {"status": "completed"})
            self.assertEqual(result["polish"], [{"candidate": "seed_060"}, {"candidate": "seed_061"}])
            plan_mock.assert_called_once()
            run_mock.assert_called_once()
            score_mock.assert_called_once_with(run_dir, ANY, project_root=Path.cwd())
            judge_mock.assert_called_once_with(run_dir, judge_config(), project_root=Path.cwd())
            select_mock.assert_called_once_with(run_dir, top_k=2)
            polish_mock.assert_called_once()


def write_brief_fixture(root: Path) -> tuple[Path, Path]:
    front = root / "assets" / "characters" / "ai51" / "views" / "front.png"
    left_profile = root / "assets" / "characters" / "ai51" / "views" / "left_profile.png"
    example = root / "examples" / "punch.png"
    write_image(front, (160, 240), (230, 190, 190))
    write_image(left_profile, (160, 240), (220, 180, 180))
    example.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (48, 72), (0, 255, 0, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 8, 31, 58), fill=(130, 80, 70, 255))
    draw.rectangle((6, 20, 44, 26), fill=(130, 80, 70, 255))
    image.save(example)
    view_bank = root / "assets" / "characters" / "ai51" / "view_bank.json"
    write_json(
        view_bank,
        {
            "schema_version": 1,
            "kind": "character-view-bank",
            "character": {"id": "ai51", "source_reference": image_asset(front, (160, 240))},
            "views": {
                "front": {
                    "image": image_asset(front, (160, 240)),
                    "accepted_candidate": "source",
                    "view": {"name": "front", "camera": "orthographic-front", "pose": "source-concept"},
                },
                "left_profile": {
                    "image": image_asset(left_profile, (160, 240)),
                    "accepted_candidate": "seed_001",
                    "accepted_seed": 1,
                    "view": {"name": "left_profile", "camera": "orthographic-side", "pose": "neutral-standing"},
                },
            },
        },
    )
    brief_path = root / "briefs" / "ai51_punch.json"
    write_json(
        brief_path,
        {
            "$schema": "../schemas/keyframe-brief.schema.json",
            "schema_version": 1,
            "kind": "keyframe-brief",
            "id": "ai51.punch.platformer.left",
            "pipeline": {"profile": "nunchaku-kontext-pose-quality"},
            "character": {"id": "ai51", "view_bank": {"path": "../assets/characters/ai51/view_bank.json"}},
            "request": {
                "action": "punch",
                "phase": "attack-start",
                "direction": "left",
                "camera": "platformer-side-view",
                "description": "Use the example sprite's attack pose as the source action.",
            },
            "example": {
                "path": "../examples/punch.png",
                "name": "platform_punch",
                "width": 160,
                "height": 240,
                "mirror_x": False,
            },
            "generation": {
                "seed_start": 60,
                "seed_count": 4,
                "output_directory": "../runs/keyframes/ai51/punch_platformer/brief_batch",
                "filename": "{id}__{variant}.png",
                "overwrite": True,
                "save_conditions": True,
                "save_contact_sheet": True,
            },
            "output": {
                "assets_directory": "../assets/extracted",
                "plan_path": "../plans/punch_plan.json",
                "job_path": "../jobs/punch_job.json",
            },
        },
    )
    return brief_path, left_profile
