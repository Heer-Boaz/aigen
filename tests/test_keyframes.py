from __future__ import annotations

import json
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw

from aigen.cli import main
from aigen.character_views import (
    accept_character_view,
    left_profile_view_template,
    plan_character_view_job,
    run_character_view_job,
)
from aigen.generation.kontext_pose_control import KontextPoseDenoised
from aigen.keyframes import (
    KeyframeJobError,
    KeyframeProfile,
    c2_profile_template,
    load_keyframe_job,
    plan_keyframe_job,
    run_keyframe_job,
    validate_keyframe_job,
)
from aigen.keyframe_memory import KeyframeMemoryError, nvidia_smi_keyframe_preflight, nvidia_smi_preflight
from aigen.keyframe_judge import (
    DEFAULT_JUDGE_ID,
    DEFAULT_JUDGE_QUANTIZATION,
    DEFAULT_JUDGE_REPO_ID,
    DEFAULT_JUDGE_REVISION,
    KeyframeJudgeConfig,
    QwenKeyframeJudge,
    judge_keyframe_run,
)
from aigen.keyframe_grounding import GroundedRegionBox
from aigen.keyframe_pose import OPENPOSE_BODY_COLORS, PoseKeypoints
from aigen.keyframe_refine import (
    KeyframeRefineProfile,
    plan_keyframe_refine_job,
    run_keyframe_refine_job,
)
from aigen.keyframe_polish import (
    diagnose_keyframe_polish,
    plan_keyframe_polish,
    preview_keyframe_polish_job,
    run_keyframe_polish_job,
    select_keyframe_polish,
)
from aigen.keyframe_score import KeyframeScoreConfig, select_scored_keyframe_run, score_keyframe_run
from aigen.keyframe_examples import KeyframeExampleExtractionConfig, extract_keyframe_example
from aigen.prompt_tokens import PromptTokenCounts


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def write_rectangle_candidate(path: Path, box: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (160, 240), (246, 238, 232))
    draw = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(draw)
    mask_draw.rectangle(box, fill=255)
    image.paste((110, 70, 45), mask=draw)
    image.save(path)


def write_rectangle_contour(path: Path, box: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (160, 240), 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=255, width=3)
    image.save(path)


def write_pose_map(path: Path, points: dict[int, tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (160, 240), "black")
    draw = ImageDraw.Draw(image)
    for index, (x_norm, y_norm) in points.items():
        color = tuple(int(value) for value in OPENPOSE_BODY_COLORS[index])
        x = round(x_norm * image.width)
        y = round(y_norm * image.height)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    image.save(path)


def pose_keypoints(points: dict[int, tuple[float, float]]) -> PoseKeypoints:
    body = torch.full((18, 2), float("nan")).numpy()
    scores = torch.zeros(18).numpy()
    for index, point in points.items():
        body[index] = point
        scores[index] = 1.0
    return PoseKeypoints(points=body, scores=scores, image_size=(160, 240))


class FakePoseExtractor:
    def __init__(self, poses: dict[str, PoseKeypoints]):
        self.poses = poses

    def extract(self, image_path: Path) -> PoseKeypoints:
        return self.poses[image_path.name]


class FakeSegmenter:
    def segment(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        background = pixels[0, 0]
        return np.sqrt(((pixels - background) ** 2).sum(axis=2)) > 20.0

    def segment_image_box(self, image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        left, top, right, bottom = box
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[top:bottom, left:right] = True
        return mask


class FakeGrounder:
    def ground_region(
        self,
        _image: Image.Image,
        prompt: str,
        prior_box: tuple[int, int, int, int],
    ) -> GroundedRegionBox:
        return GroundedRegionBox(
            box=prior_box,
            label=prompt,
            score=1.0,
            source="fake-grounder",
            prior_iou=1.0,
        )


def fake_dwpose_control_image(image: Image.Image, **_kwargs: object) -> tuple[Image.Image, dict[str, object]]:
    pose = Image.new("RGB", image.size, "black")
    draw = ImageDraw.Draw(pose)
    draw.line((20, 30, image.width - 20, 30), fill=(255, 0, 0), width=5)
    return pose, {
        "body_count": 1,
        "visible_body_keypoints": 12,
        "mean_body_score": 0.75,
    }


def write_score_fixture(root: Path) -> tuple[Path, dict[str, PoseKeypoints]]:
    run_dir = root / "runs" / "score"
    reference = root / "assets" / "reference.png"
    pose = root / "assets" / "pose.png"
    contour = root / "assets" / "contour.png"
    mask = root / "assets" / "mask.png"
    target_points = {
        0: (0.52, 0.16),
        1: (0.50, 0.26),
        2: (0.43, 0.30),
        5: (0.57, 0.30),
        8: (0.45, 0.52),
        9: (0.40, 0.72),
        10: (0.36, 0.90),
        11: (0.55, 0.52),
        12: (0.58, 0.72),
        13: (0.65, 0.90),
    }
    write_image(reference, (160, 240), (240, 230, 220))
    write_pose_map(pose, target_points)
    write_rectangle_contour(contour, (60, 30, 100, 210))
    write_image(mask, (160, 240), (255, 255, 255))
    outputs = []
    poses = {}
    candidates = {
        "seed_002": (48, 35, 90, 205),
        "seed_003": (60, 30, 100, 210),
        "seed_005": (20, 55, 75, 190),
    }
    for index, (name, box) in enumerate(candidates.items(), start=2):
        image_path = run_dir / f"{name}.png"
        write_rectangle_candidate(image_path, box)
        outputs.append({"name": name, "seed": index, "path": image_path.as_posix()})
        poses[image_path.name] = pose_keypoints(target_points)
    write_json(
        run_dir / "result.json",
        {
            "status": "completed",
            "job_id": "score.fixture",
            "assets": {
                "identity_primer": {"path": reference.as_posix()},
                "pose": {"path": pose.as_posix()},
                "contour": {"path": contour.as_posix()},
                "boundary_mask": {"path": mask.as_posix()},
            },
            "outputs": outputs,
            "effective_config": {
                "keyframe": {"action": "walk", "phase": "contact", "direction": "left", "camera": "orthographic-side"}
            },
        },
    )
    return run_dir, poses


def write_pose_score_fixture(root: Path) -> tuple[Path, dict[str, PoseKeypoints]]:
    run_dir = root / "runs" / "pose-score"
    reference = root / "assets" / "reference.png"
    pose = root / "assets" / "pose.png"
    contour = root / "assets" / "contour.png"
    mask = root / "assets" / "mask.png"
    target_points = {
        0: (0.52, 0.16),
        1: (0.50, 0.26),
        2: (0.43, 0.30),
        5: (0.57, 0.30),
        8: (0.45, 0.52),
        9: (0.40, 0.72),
        10: (0.36, 0.90),
        11: (0.55, 0.52),
        12: (0.58, 0.72),
        13: (0.65, 0.90),
    }
    shifted_points = {index: (x - 0.09, y + 0.02) for index, (x, y) in target_points.items()}
    write_image(reference, (160, 240), (240, 230, 220))
    write_pose_map(pose, target_points)
    write_rectangle_contour(contour, (60, 30, 100, 210))
    write_image(mask, (160, 240), (255, 255, 255))
    outputs = []
    poses = {}
    for name, candidate_pose in (
        ("seed_002", shifted_points),
        ("seed_003", target_points),
    ):
        image_path = run_dir / f"{name}.png"
        write_rectangle_candidate(image_path, (60, 30, 100, 210))
        outputs.append({"name": name, "seed": int(name[-3:]), "path": image_path.as_posix()})
        poses[image_path.name] = pose_keypoints(candidate_pose)
    write_json(
        run_dir / "result.json",
        {
            "status": "completed",
            "job_id": "pose.score.fixture",
            "assets": {
                "identity_primer": {"path": reference.as_posix()},
                "pose": {"path": pose.as_posix()},
                "contour": {"path": contour.as_posix()},
                "boundary_mask": {"path": mask.as_posix()},
            },
            "outputs": outputs,
            "effective_config": {
                "keyframe": {"action": "walk", "phase": "contact", "direction": "left", "camera": "orthographic-side"}
            },
        },
    )
    return run_dir, poses


def write_keyframe_result(root: Path) -> Path:
    run_dir = root / "runs" / "walk"
    reference = root / "assets" / "reference.png"
    pose = root / "assets" / "pose.png"
    contour = root / "assets" / "contour.png"
    mask = root / "assets" / "mask.png"
    write_image(reference, (512, 768), (220, 180, 180))
    write_image(pose, (640, 960), (0, 255, 0))
    write_image(contour, (640, 960), (255, 255, 255))
    write_image(mask, (640, 960), (96, 96, 96))
    outputs = []
    for index, name in enumerate(("seed_002", "seed_003", "seed_005")):
        image_path = run_dir / f"{name}.png"
        write_image(image_path, (640, 960), (255 - index * 20, 230, 230))
        outputs.append({"name": name, "seed": index + 2, "path": image_path.as_posix()})
    write_json(
        run_dir / "result.json",
        {
            "status": "completed",
            "job_id": "ai46.walk.contact.left",
            "assets": {
                "identity_primer": {"path": reference.as_posix()},
                "pose": {"path": pose.as_posix()},
                "contour": {"path": contour.as_posix()},
                "boundary_mask": {"path": mask.as_posix()},
            },
            "outputs": outputs,
            "effective_config": {
                "keyframe": {
                    "action": "walk",
                    "phase": "contact",
                    "direction": "left",
                    "camera": "orthographic-side",
                },
                "prompt": {
                    "clip": "Same anime girl, strict side profile.",
                    "t5": "Full-body orthographic side-view gameplay keyframe.",
                    "true_cfg_scale": 1.0,
                },
                "acceptance": {"manual": ["strict side profile", "feet fully visible"]},
                "condition_plan": [
                    {"name": "pose", "type": "pose", "active_steps": 15},
                    {"name": "profile_contour", "type": "canny", "active_steps": 12},
                ],
            },
        },
    )
    return run_dir


def write_character_view_fixture(root: Path) -> tuple[Path, Path]:
    source = root / "assets" / "characters" / "ai46" / "views" / "front.png"
    pose = root / "assets" / "views" / "ai46" / "left_profile_pose.png"
    contour = root / "assets" / "views" / "ai46" / "left_profile_contour.png"
    mask = root / "assets" / "views" / "ai46" / "left_profile_boundary.png"
    write_image(source, (160, 240), (240, 230, 220))
    write_image(pose, (512, 768), (0, 255, 0))
    write_image(contour, (512, 768), (255, 255, 255))
    write_image(mask, (512, 768), (128, 128, 128))
    run_dir = root / "runs" / "views"
    candidate = run_dir / "seed_003.png"
    write_image(candidate, (512, 768), (210, 180, 160))
    write_json(
        run_dir / "result.json",
        {
            "status": "completed",
            "job_id": "ai46.left_profile.neutral",
            "outputs": [{"name": "seed_003", "seed": 3, "path": candidate.as_posix()}],
        },
    )
    job_path = root / "jobs" / "ai46_left_profile_view.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        job_path,
        {
            "$schema": "../../schemas/character-view-job.schema.json",
            "schema_version": 1,
            "kind": "character-view",
            "id": "ai46.left_profile.neutral",
            "pipeline": {"profile": "nunchaku-kontext-pose-quality"},
            "character": {"id": "ai46", "source_reference": {"path": "../assets/characters/ai46/views/front.png"}},
            "view": {"name": "left_profile", "camera": "orthographic-side", "pose": "neutral-standing"},
            "assets": {
                "pose": {"path": "../assets/views/ai46/left_profile_pose.png"},
                "contour": {"path": "../assets/views/ai46/left_profile_contour.png"},
                "boundary_mask": {"path": "../assets/views/ai46/left_profile_boundary.png"},
            },
            "prompt": {
                "clip": "Same anime girl, neutral left profile.",
                "t5": "Full-body neutral-standing character turnaround view.",
                "true_cfg_scale": 1.0,
            },
            "canvas": {"width": 512, "height": 768, "reference_max_area": 294912, "max_sequence_length": 128},
            "sampling": {"steps": 28, "guidance_scale": 2.5},
            "conditions": [
                {"name": "pose", "type": "pose", "image": "pose", "scale": 0.5, "start": 0.0, "end": 0.55},
                {
                    "name": "profile_contour",
                    "type": "canny",
                    "image": "contour",
                    "residual_mask": "boundary_mask",
                    "scale": 0.35,
                    "start": 0.0,
                    "end": 0.4,
                },
            ],
            "variants": [{"name": "seed_003", "seed": 3}],
            "output": {
                "directory": "../runs/views",
                "filename": "{id}__{variant}.png",
                "canonical_path": "../assets/characters/ai46/views/left_profile.png",
                "bank_path": "../assets/characters/ai46/view_bank.json",
                "overwrite": False,
                "save_conditions": True,
                "save_contact_sheet": True,
            },
            "acceptance": {"manual": ["strict left profile"], "minimum_passing_variants": 1},
        },
    )
    return job_path, run_dir


def write_refine_fixture(root: Path) -> Path:
    run_dir = root / "runs" / "punch"
    reference = root / "assets" / "AI46.png"
    pose = root / "assets" / "punch_pose.png"
    contour = root / "assets" / "punch_contour.png"
    base_image = run_dir / "seed_005.png"
    write_rectangle_candidate(reference, (54, 30, 108, 220))
    write_rectangle_candidate(base_image, (55, 36, 112, 220))
    write_pose_map(
        pose,
        {
            0: (0.54, 0.16),
            1: (0.52, 0.27),
            2: (0.47, 0.31),
            3: (0.31, 0.31),
            4: (0.16, 0.31),
            5: (0.58, 0.31),
            6: (0.62, 0.43),
            7: (0.65, 0.55),
            8: (0.48, 0.54),
            9: (0.42, 0.74),
            10: (0.36, 0.91),
            11: (0.56, 0.54),
            12: (0.62, 0.74),
            13: (0.68, 0.91),
        },
    )
    write_rectangle_contour(contour, (20, 58, 118, 118))
    write_json(
        run_dir / "resolved.json",
        {
            "keyframe": {
                "action": "punch",
                "phase": "straight-fist",
                "direction": "left",
                "camera": "orthographic-side",
            },
        },
    )
    write_json(
        run_dir / "result.json",
        {
            "status": "completed",
            "job_id": "ai46.punch.fixture",
            "outputs": [{"name": "seed_005", "seed": 5, "path": base_image.as_posix()}],
        },
    )
    job_path = root / "jobs" / "punch_refine.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        job_path,
        {
            "$schema": "../../schemas/keyframe-refine-job.schema.json",
            "schema_version": 1,
            "kind": "keyframe-refine",
            "id": "ai46.punch.straight.fist.refine",
            "pipeline": {"profile": "kontext-inpaint-local"},
            "base": {"run_dir": "../runs/punch", "candidate": "seed_005"},
            "character": {
                "id": "ai46",
                "identity_primer": {"view": "front", "path": "../assets/AI46.png"},
            },
            "region": {
                "name": "punching_arm_fist",
                "mask_source": {
                    "type": "pose_contour_auto",
                    "pose": "../assets/punch_pose.png",
                    "contour": "../assets/punch_contour.png",
                    "candidate_foreground": False,
                },
                "dilate_px": 8,
                "feather_px": 3,
                "crop_padding_px": 24,
            },
            "prompt": {
                "clip": "Closed fist straight punch, preserve character and outfit.",
                "t5": "Replace only the masked punching arm and hand with a compact closed fist.",
                "negative": "open hand, pointing finger",
                "true_cfg_scale": 1.25,
            },
            "sampling": {
                "steps": 4,
                "guidance_scale": 2.5,
                "strength": 0.85,
                "max_sequence_length": 128,
            },
            "variants": [{"name": "refine_001", "seed": 101}],
            "output": {
                "directory": "../runs/punch_refine",
                "filename": "{id}__{variant}.png",
                "overwrite": False,
                "save_debug_images": True,
                "save_contact_sheet": True,
            },
            "acceptance": {"manual": ["closed fist", "front arm extended"]},
        },
    )
    return job_path


def write_polish_fixture(root: Path) -> tuple[Path, Path]:
    run_dir = root / "runs" / "punch"
    reference = root / "assets" / "AI51_left_profile.png"
    base_image = run_dir / "seed_060.png"
    plan = root / "jobs" / "seed_060_polish_plan.json"
    plan.parent.mkdir(parents=True, exist_ok=True)
    write_rectangle_candidate(reference, (54, 30, 108, 220))
    write_rectangle_candidate(base_image, (55, 36, 112, 220))
    write_json(
        run_dir / "result.json",
        {
            "status": "completed",
            "job_id": "ai51.punch.fixture",
            "assets": {
                "identity_primer": {
                    "path": reference.as_posix(),
                    "mode": "RGB",
                    "width": 160,
                    "height": 240,
                    "sha256": "reference",
                }
            },
            "effective_config": {
                "keyframe": {
                    "action": "punch",
                    "phase": "platformer-example",
                    "direction": "left",
                    "camera": "orthographic-side",
                },
                "acceptance": {
                    "manual": [
                        "strict side profile",
                        "front arm and hand pose match the source example",
                    ]
                },
            },
            "outputs": [{"name": "seed_060", "seed": 60, "path": base_image.as_posix()}],
        },
    )
    write_json(
        plan,
        {
            "schema_version": 1,
            "status": "completed",
            "run_dir": run_dir.as_posix(),
            "candidate": "seed_060",
            "polish_plan": {
                "schema_version": 1,
                "kind": "keyframe-polish-plan",
                "job_id": "ai51.punch.seed060.polish",
                "base_candidate": "seed_060",
                "needs_polish": True,
                "regions": [
                    {
                        "id": "region_01",
                        "label": "face expression",
                        "bbox": [48, 28, 105, 92],
                        "mask_prompt": "visible face, eye, mouth and cheek",
                        "operation": "expression_refine",
                        "reason": "expression is weak",
                        "reference_crop_requirements": ["matching face region from identity primer"],
                        "parameters": {
                            "strength": 0.36,
                            "steps": 18,
                            "guidance_scale": 2.2,
                            "true_cfg_scale": 1.25,
                            "feather_px": 3,
                            "crop_padding_px": 16,
                            "crop_upsample_factor": 1.0,
                            "max_sequence_length": 128,
                        },
                        "prompt": "Refine the local face expression while preserving the side-profile head shape.",
                        "negative_prompt": "changed hair length, front view, changed pose",
                        "must_not_change": ["pose", "hair length"],
                        "acceptance_checks": ["expression clearer", "outside mask unchanged"],
                    },
                    {
                        "id": "region_02",
                        "label": "waist outfit details",
                        "bbox": [55, 104, 112, 165],
                        "mask_prompt": "belt, waist and skirt panel details",
                        "operation": "detail_restore",
                        "reason": "waist outfit details are weak",
                        "reference_crop_requirements": ["matching waist outfit region from identity primer"],
                        "parameters": {
                            "strength": 0.30,
                            "steps": 16,
                            "guidance_scale": 2.0,
                            "true_cfg_scale": 1.2,
                            "feather_px": 3,
                            "crop_padding_px": 16,
                            "crop_upsample_factor": 1.0,
                            "max_sequence_length": 128,
                        },
                        "prompt": "Restore local waist outfit details while preserving the existing body pose.",
                        "negative_prompt": "pants, changed legs, changed pose",
                        "must_not_change": ["legs", "pose"],
                        "acceptance_checks": ["outfit detail clearer", "outside mask unchanged"],
                    },
                ],
                "summary": "Polish the face expression and waist outfit details.",
            },
        },
    )
    job_path = root / "jobs" / "punch_polish.json"
    write_json(
        job_path,
        {
            "$schema": "../../schemas/keyframe-polish-job.schema.json",
            "schema_version": 1,
            "kind": "keyframe-polish",
            "id": "ai51.punch.seed060.polish",
            "pipeline": {"profile": "kontext-inpaint-local"},
            "base": {"run_dir": "../runs/punch", "candidate": "seed_060"},
            "character": {
                "id": "ai51",
                "identity_primer": {"view": "left_profile", "path": "../assets/AI51_left_profile.png"},
            },
            "plan": {"path": "seed_060_polish_plan.json"},
            "planner": {"max_regions": 4},
            "micro_sweep": {"strength_offsets": [0.0], "seed_offsets": [1]},
            "output": {
                "directory": "../runs/punch_polish",
                "overwrite": False,
                "save_debug_images": True,
                "save_contact_sheet": True,
            },
            "acceptance": {"manual": ["outside mask unchanged", "model-planned local details improved"]},
        },
    )
    return job_path, plan


def profile() -> KeyframeProfile:
    return KeyframeProfile(
        name="nunchaku-kontext-pose-quality",
        model="/models/kontext",
        controlnet_model="/models/controlnet",
        nunchaku_transformer_model=Path("/models/nunchaku.safetensors"),
        attention_impl="nunchaku-fp16",
        dtype="bfloat16",
        pipeline_cpu_offload=True,
        nunchaku_layer_offload=False,
        vae_tiling=False,
        model_revisions={
            "kontext": {"repo_id": "kontext", "revision": "kontext-revision"},
            "controlnet": {"repo_id": "controlnet", "revision": "controlnet-revision"},
            "nunchaku_transformer": {"repo_id": "nunchaku", "revision": "nunchaku-revision"},
        },
    )


def refine_profile() -> KeyframeRefineProfile:
    return KeyframeRefineProfile(
        name="kontext-inpaint-local",
        model="/models/kontext",
        nunchaku_transformer_model=Path("/models/nunchaku.safetensors"),
        attention_impl="nunchaku-fp16",
        dtype="bfloat16",
        pipeline_cpu_offload=True,
        vae_tiling=False,
        model_revisions={
            "kontext": {"repo_id": "kontext", "revision": "kontext-revision"},
            "nunchaku_transformer": {"repo_id": "nunchaku", "revision": "nunchaku-revision"},
        },
    )


def job_payload(root: Path) -> dict[str, object]:
    reference = root / "assets" / "AI46.png"
    pose = root / "assets" / "pose.png"
    contour = root / "assets" / "contour.png"
    mask = root / "assets" / "mask.png"
    write_image(reference, (1024, 2048), (255, 255, 255))
    write_image(pose, (512, 768), (0, 255, 0))
    write_image(contour, (512, 768), (255, 255, 255))
    write_image(mask, (512, 768), (128, 128, 128))
    return {
        "$schema": "../../schemas/keyframe-job.schema.json",
        "schema_version": 1,
        "kind": "character-keyframe",
        "id": "ai46.walk.contact.left",
        "pipeline": {"profile": "nunchaku-kontext-pose-quality"},
        "character": {
            "id": "ai46",
            "identity_primer": {"view": "front", "path": "assets/AI46.png"},
        },
        "keyframe": {
            "action": "walk",
            "phase": "contact",
            "direction": "left",
            "camera": "orthographic-side",
        },
        "assets": {
            "pose": {"path": "assets/pose.png"},
            "contour": {"path": "assets/contour.png"},
            "boundary_mask": {"path": "assets/mask.png"},
        },
        "prompt": {
            "clip": "Same anime girl, strict side profile.",
            "t5": "Full-body orthographic side-view gameplay keyframe.",
            "true_cfg_scale": 1.0,
        },
        "canvas": {
            "width": 512,
            "height": 768,
            "reference_max_area": 524288,
            "max_sequence_length": 128,
        },
        "sampling": {"steps": 28, "guidance_scale": 2.5},
        "conditions": [
            {
                "name": "pose",
                "type": "pose",
                "image": "pose",
                "scale": 0.55,
                "start": 0.0,
                "end": 0.55,
            },
            {
                "name": "profile_contour",
                "type": "canny",
                "image": "contour",
                "residual_mask": "boundary_mask",
                "scale": 0.35,
                "start": 0.0,
                "end": 0.40,
            },
        ],
        "variants": [
            {"name": "seed_001", "seed": 1},
            {"name": "seed_002", "seed": 2},
        ],
        "output": {
            "directory": "runs/keyframes/ai46/walk_contact/batch",
            "filename": "{id}__{variant}.png",
            "overwrite": False,
            "save_conditions": True,
            "save_contact_sheet": True,
        },
        "acceptance": {"manual": ["strict side profile"], "minimum_passing_variants": 1},
    }


class FakeImage:
    def __init__(self, label: str) -> None:
        self.label = label
        self.width = 512
        self.height = 768

    def save(self, path: Path) -> None:
        Image.new("RGB", (self.width, self.height), (255, 255, 255)).save(path)

    def resize(self, _size: tuple[int, int], _resampling: object) -> FakeImage:
        return self

    def convert(self, _mode: str) -> FakeImage:
        return self


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.free_model_hooks_calls = 0

    def maybe_free_model_hooks(self) -> None:
        self.free_model_hooks_calls += 1

    def denoise_prepared(self, prepared: object, **kwargs: object) -> object:
        if self.free_model_hooks_calls < 1:
            raise AssertionError("prepare-phase models must be released before denoise")
        self.calls.append(kwargs)
        return KontextPoseDenoised(
            name=kwargs["name"],
            latents=torch.ones((1, 2, 3)),
            controlnet_conditioning_scale=kwargs["controlnet_conditioning_scale"],
            control_guidance_start=kwargs["control_guidance_start"],
            control_guidance_end=kwargs["control_guidance_end"],
            seed=kwargs["seed"],
            transformer_step_ms=[2.0],
            controlnet_step_ms=[1.0],
            controlnet_active_steps=15,
            controlnet_metadata={"conditions": [condition.name for condition in kwargs["control_conditions"]]},
            timings_ms={"denoise_ms": 3.0},
        )


class FakeSession:
    instances: list[FakeSession] = []

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.pipeline = FakePipeline()
        self.model_load_ms = 1.0
        self.torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: False,
                reset_peak_memory_stats=lambda _device: None,
            )
        )
        FakeSession.instances.append(self)

    def prepare(self, **kwargs: object) -> object:
        self.prepare_kwargs = kwargs
        return types.SimpleNamespace(
            control_image="pose-control",
            controlnet_blocks_repeat=False,
            token_metadata={"generated_tokens": 1536},
        )

    def prepare_control_condition(self, _prepared: object, *, pose_image: object, seed: int) -> tuple[str, bool, float]:
        return f"control:{seed}:{pose_image.size}", False, 1.0

    def prepare_residual_mask(self, _prepared: object, mask_image: object) -> str:
        return f"mask:{mask_image.size}"

    def decode_many(self, _prepared: object, denoised: list[object], *, chunk_size: int) -> tuple[list[FakeImage], float]:
        if self.pipeline.free_model_hooks_calls < 2:
            raise AssertionError("denoise-phase models must be released before decode")
        return [FakeImage(result.name) for result in denoised], 4.0

    def close(self) -> None:
        self.closed = True


class FakeRefiner:
    instances: list[FakeRefiner] = []

    def __init__(self, _profile: object) -> None:
        self.model_load_ms = 1.0
        self.torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: False,
                reset_peak_memory_stats=lambda _device: None,
            )
        )
        self.pipeline = object()
        self.device_report = {"components": {"transformer": {"parameter_tensors_by_device": {"cuda:0": 1}}}}
        FakeRefiner.instances.append(self)

    def refine(self, *, base_crop: Image.Image, mask_crop: Image.Image, seed: int, **_kwargs: object) -> Image.Image:
        self.seed = seed
        output = base_crop.copy()
        output.paste((20, 40, 210), mask=mask_crop)
        return output

    def close(self) -> None:
        self.closed = True


class FakeJudgeRunner:
    scores = {
        "seed_002": (8.0, 8.0, True, {}),
        "seed_003": (9.0, 9.0, True, {}),
        "seed_005": (
            5.0,
            4.0,
            False,
            {"front_or_three_quarter_view": True, "two_eyes_visible": True},
        ),
    }

    def judge_candidate(self, prompt: str, image_paths: list[Path]) -> str:
        candidate = next(name for name in self.scores if name in prompt)
        condition, side, passes, rejects = self.scores[candidate]
        hard_rejects = {
            "front_or_three_quarter_view": False,
            "two_eyes_visible": False,
            "wrong_direction": False,
            "cropped_feet": False,
            "missing_boots": False,
            "long_hair": False,
            "wrong_outfit": False,
            "pose_not_walk_contact": False,
            "severe_limb_error": False,
        } | rejects
        return json.dumps(
            {
                "candidate": candidate,
                "pass": passes,
                "rank_recommendation": {"seed_003": 1, "seed_002": 2, "seed_005": 3}[candidate],
                "hard_rejects": hard_rejects,
                "scores": {
                    "condition_adherence": condition,
                    "side_profile": side,
                    "pose_match": condition,
                    "contour_match": condition,
                    "identity_preservation": 8,
                    "outfit_preservation": 8,
                    "artifact_quality": 7,
                    "overall": condition,
                },
                "evidence": {
                    "condition_match": f"{candidate} condition evidence",
                    "identity_match": f"{candidate} identity evidence",
                    "concerns": ["minor boot simplification"],
                },
            }
        )

class FakePolishPlanner:
    def judge_candidate(self, prompt: str, image_paths: list[Path]) -> str:
        if len(image_paths) < 3:
            raise AssertionError(f"Expected planner evidence images, got {len(image_paths)}")
        if "Do not use a fixed region catalog" not in prompt:
            raise AssertionError("Polish planner prompt does not forbid fixed region catalogs")
        return json.dumps(
            {
                "schema_version": 1,
                "kind": "keyframe-polish-plan",
                "job_id": "ai51.punch.seed060.polish",
                "base_candidate": "seed_060",
                "needs_polish": True,
                "regions": [
                    {
                        "id": "region_01",
                        "label": "face expression",
                        "bbox": [48, 28, 105, 92],
                        "mask_prompt": "visible face, eye, mouth and cheek",
                        "operation": "expression_refine",
                        "reason": "expression is weak",
                        "reference_crop_requirements": ["matching face region from identity primer"],
                        "parameters": {
                            "strength": 0.36,
                            "steps": 18,
                            "guidance_scale": 2.2,
                            "true_cfg_scale": 1.25,
                            "feather_px": 3,
                            "crop_padding_px": 16,
                            "crop_upsample_factor": 1.0,
                            "max_sequence_length": 128,
                        },
                        "prompt": "Refine the local face expression while preserving the side-profile head shape.",
                        "negative_prompt": "changed hair length, front view, changed pose",
                        "must_not_change": ["pose", "hair length"],
                        "acceptance_checks": ["expression clearer", "outside mask unchanged"],
                    }
                ],
                "summary": "Polish the face expression.",
            }
        )


class FakePolishSelector:
    def judge_candidate(self, prompt: str, image_paths: list[Path]) -> str:
        if "candidate crops" not in prompt:
            raise AssertionError("Polish selector prompt does not expose local candidates")
        region_id = prompt.split("- id: ", 1)[1].splitlines()[0]
        candidate = next(path.stem.removesuffix("_crop") for path in image_paths if path.stem.startswith("region_"))
        return json.dumps(
            {
                "region_id": region_id,
                "best_variant": candidate,
                "passes": True,
                "checks": {
                    "target_detail_restored": True,
                    "identity_preserved": True,
                    "outside_mask_changed": False,
                    "pose_changed": False,
                    "style_match": True,
                },
                "reason": "local detail is clearer",
            }
        )


class KeyframeTests(unittest.TestCase):
    def test_c2_template_has_no_unused_null_fields(self) -> None:
        template = c2_profile_template()

        self.assertNotIn("depth", template["assets"])
        self.assertNotIn("softedge", template["assets"])
        self.assertNotIn("negative", template["prompt"])
        self.assertEqual(template["conditions"][1]["residual_mask"], "boundary_mask")

    def test_cli_init_outputs_keyframe_job_template(self) -> None:
        stdout = StringIO()

        with redirect_stdout(stdout):
            exit_code = main(["keyframes", "init", "--template", "c2-profile"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["kind"], "character-keyframe")
        self.assertEqual(payload["pipeline"]["profile"], "nunchaku-kontext-pose-quality")

    def test_rejects_unknown_json_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = job_payload(root)
            payload["surprise"] = True
            job_path = root / "job.json"
            write_json(job_path, payload)

            with self.assertRaisesRegex(KeyframeJobError, "surprise"):
                load_keyframe_job(job_path)

    def test_plans_keyframe_job_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_path = root / "job.json"
            write_json(job_path, job_payload(root))

            with patch(
                "aigen.keyframes.count_kontext_prompt_tokens",
                return_value=PromptTokenCounts(clip=12, clip_limit=77, t5=24),
            ):
                resolved = plan_keyframe_job(job_path, profile(), project_root=Path.cwd())

        self.assertEqual(resolved["tokens"], {"clip": 12, "clip_limit": 77, "t5": 24, "t5_limit": 128})
        self.assertEqual(resolved["condition_plan"][0]["active_steps"], 15)
        self.assertEqual(resolved["condition_plan"][1]["active_steps"], 11)
        self.assertEqual(len(resolved["output"]["files"]), 2)
        self.assertIn("sha256", resolved["assets"]["pose"])
        self.assertEqual(resolved["character"]["identity_primer"]["view"], "front")
        self.assertEqual(resolved["token_metadata"]["generated_tokens"], 1536)
        self.assertEqual(resolved["token_metadata"]["reference_tokens"], 2048)
        self.assertEqual(resolved["vram_plan"]["method"], "nunchaku-kontext-controlnet-local")

    def test_rejects_keyframe_job_without_identity_primer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = job_payload(root)
            payload["character"] = {"id": "ai46"}
            job_path = root / "job.json"
            write_json(job_path, payload)

            with self.assertRaisesRegex(KeyframeJobError, "identity_primer"):
                load_keyframe_job(job_path)

    def test_rejects_wrong_control_asset_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = job_payload(root)
            write_image(root / "assets" / "pose.png", (832, 1248), (0, 255, 0))
            job_path = root / "job.json"
            write_json(job_path, payload)

            with patch(
                "aigen.keyframes.count_kontext_prompt_tokens",
                return_value=PromptTokenCounts(clip=12, clip_limit=77, t5=24),
            ):
                with self.assertRaisesRegex(KeyframeJobError, "Asset pose must be 512x768"):
                    validate_keyframe_job(job_path, profile(), project_root=Path.cwd())

    def test_run_writes_resolved_result_and_conditions(self) -> None:
        FakeSession.instances.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_path = root / "job.json"
            write_json(job_path, job_payload(root))
            with (
                patch(
                    "aigen.keyframes.count_kontext_prompt_tokens",
                    return_value=PromptTokenCounts(clip=12, clip_limit=77, t5=24),
                ),
                patch("aigen.keyframes.CharacterKontextPoseSession", FakeSession),
                patch("aigen.keyframes.cuda_memory_stats", return_value={"max_allocated_mb": 1}),
                patch("aigen.keyframes._generation_environment", return_value={"env": "fake"}),
                patch(
                    "aigen.keyframes.nvidia_smi_keyframe_preflight",
                    return_value={
                        "nvidia_smi_preflight_used_mb": 0,
                        "nvidia_smi_device_total_mb": 0,
                        "nvidia_smi_preflight_utilization_gpu": 0,
                        "vram_estimated_required_mb": 0,
                        "vram_estimated_headroom_mb": 0,
                    },
                ),
            ):
                result = run_keyframe_job(job_path, profile(), project_root=Path.cwd())

            output_dir = root / "runs" / "keyframes" / "ai46" / "walk_contact" / "batch"
            result_path = output_dir / "result.json"
            resolved_path = output_dir / "resolved.json"

            self.assertTrue(result_path.exists())
            self.assertTrue(resolved_path.exists())
            self.assertTrue((output_dir / "conditions" / "pose.png").exists())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["outputs"][0]["controlnet_metadata"]["conditions"], ["pose", "profile_contour"])
            self.assertEqual(
                FakeSession.instances[0].prepare_kwargs["reference_image"],
                root / "assets" / "AI46.png",
            )
            self.assertEqual(
                FakeSession.instances[0].prepare_kwargs["t5_prompt"],
                "Full-body orthographic side-view gameplay keyframe.",
            )
            self.assertEqual(result["memory"]["nvidia_smi_peak_used_mb"], 0)

    def test_character_view_accept_writes_canonical_view_bank_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_path, run_dir = write_character_view_fixture(root)

            plan = plan_character_view_job(job_path, project_root=Path.cwd())
            accepted = accept_character_view(job_path, run_dir=run_dir, candidate="seed_003", project_root=Path.cwd())
            bank_path = Path(accepted["bank_path"])
            bank = json.loads(bank_path.read_text(encoding="utf-8"))

            self.assertEqual(plan["view"]["name"], "left_profile")
            self.assertTrue(Path(accepted["canonical_path"]).exists())
            self.assertEqual(bank["views"]["left_profile"]["accepted_candidate"], "seed_003")
            self.assertEqual(bank["views"]["left_profile"]["image"]["sha256"], accepted["canonical_sha256"])

    def test_cli_character_view_init_outputs_view_job_template(self) -> None:
        stdout = StringIO()

        with redirect_stdout(stdout):
            exit_code = main(["characters", "view-init", "--template", "ai46-left-profile"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["kind"], "character-view")
        self.assertEqual(payload["character"]["source_reference"]["path"], "../../assets/characters/ai46/views/front.png")
        self.assertEqual(left_profile_view_template()["view"]["name"], "left_profile")

    def test_character_view_run_uses_source_reference_as_front_primer(self) -> None:
        FakeSession.instances.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_path, _run_dir = write_character_view_fixture(root)
            with (
                patch(
                    "aigen.keyframes.count_kontext_prompt_tokens",
                    return_value=PromptTokenCounts(clip=10, clip_limit=77, t5=20),
                ),
                patch("aigen.keyframes.CharacterKontextPoseSession", FakeSession),
                patch("aigen.keyframes.cuda_memory_stats", return_value={"max_allocated_mb": 0}),
                patch("aigen.keyframes._generation_environment", return_value={"env": "fake"}),
                patch(
                    "aigen.keyframes.nvidia_smi_keyframe_preflight",
                    return_value={
                        "nvidia_smi_preflight_used_mb": 0,
                        "nvidia_smi_device_total_mb": 0,
                        "nvidia_smi_preflight_utilization_gpu": 0,
                    },
                ),
            ):
                result = run_character_view_job(job_path, profile(), project_root=Path.cwd())

            output_dir = root / "runs" / "views"
            self.assertTrue((output_dir / "result.json").exists())
            self.assertEqual(result["job_id"], "ai46.left_profile.neutral")
            self.assertEqual(
                FakeSession.instances[0].prepare_kwargs["reference_image"],
                root / "assets" / "characters" / "ai46" / "views" / "front.png",
            )

    def test_extract_keyframe_example_writes_condition_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            image = Image.new("RGBA", (32, 48), (0, 255, 0, 255))
            draw = ImageDraw.Draw(image)
            draw.rectangle((9, 5, 20, 40), fill=(120, 80, 60, 255))
            draw.rectangle((3, 15, 28, 21), fill=(120, 80, 60, 255))
            image.save(source)

            with patch("aigen.keyframe_examples._dwpose_control_image", fake_dwpose_control_image):
                result = extract_keyframe_example(
                    KeyframeExampleExtractionConfig(
                        source=source,
                        output_dir=root / "assets" / "examples",
                        name="punch_example",
                        width=160,
                        height=240,
                        mirror_x=True,
                    )
                )

            assets = result["assets"]
            self.assertTrue(Path(assets["pose"]["path"]).exists())
            self.assertTrue(Path(assets["contour"]["path"]).exists())
            self.assertTrue(Path(assets["boundary_mask"]["path"]).exists())
            self.assertEqual(assets["pose"]["width"], 160)
            self.assertEqual(assets["contour"]["height"], 240)
            self.assertEqual(result["pose"]["visible_body_keypoints"], 12)
            self.assertTrue(result["mirror_x"])

    def test_cli_extract_example_outputs_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            image = Image.new("RGBA", (32, 48), (0, 255, 0, 255))
            draw = ImageDraw.Draw(image)
            draw.rectangle((9, 5, 20, 40), fill=(120, 80, 60, 255))
            image.save(source)
            stdout = StringIO()

            with patch("aigen.keyframe_examples._dwpose_control_image", fake_dwpose_control_image), redirect_stdout(stdout):
                exit_code = main(
                    [
                        "keyframes",
                        "extract-example",
                        "--source",
                        source.as_posix(),
                        "--output-dir",
                        (root / "assets").as_posix(),
                        "--name",
                        "sprite",
                        "--width",
                        "160",
                        "--height",
                        "240",
                        "--mirror-x",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["kind"], "keyframe-example-extraction")
            self.assertEqual(payload["assets"]["metadata"]["path"], (root / "assets" / "sprite_extraction.json").as_posix())

    def test_preflight_rejects_dirty_framebuffer(self) -> None:
        with (
            patch("aigen.keyframe_memory.cuda_available", return_value=True),
            patch(
                "aigen.keyframe_memory.nvidia_smi_memory_snapshot",
                return_value={
                    "nvidia_smi_used_mb": 1801,
                    "nvidia_smi_device_total_mb": 16303,
                    "nvidia_smi_utilization_gpu": 0,
                },
            ),
        ):
            with self.assertRaisesRegex(KeyframeMemoryError, "1801 MB used before model load"):
                nvidia_smi_preflight()

    def test_keyframe_preflight_rejects_estimated_oom(self) -> None:
        with (
            patch("aigen.keyframe_memory.cuda_available", return_value=True),
            patch(
                "aigen.keyframe_memory.nvidia_smi_memory_snapshot",
                return_value={
                    "nvidia_smi_used_mb": 999,
                    "nvidia_smi_device_total_mb": 16303,
                    "nvidia_smi_utilization_gpu": 0,
                },
            ),
        ):
            with self.assertRaisesRegex(KeyframeMemoryError, "Estimated VRAM requirement exceeds"):
                nvidia_smi_keyframe_preflight(
                    {
                        "baseline_framebuffer_mb": 700,
                        "safety_margin_mb": 256,
                        "estimated_clean_peak_mb": 16033,
                        "true_cfg_enabled": False,
                        "true_cfg_extra_mb": 0,
                        "canvas_width": 512,
                        "canvas_height": 768,
                        "generated_tokens": 1536,
                        "reference_tokens": 2048,
                    }
                )

    def test_keyframe_preflight_reports_max_output_canvas(self) -> None:
        with (
            patch("aigen.keyframe_memory.cuda_available", return_value=True),
            patch(
                "aigen.keyframe_memory.nvidia_smi_memory_snapshot",
                return_value={
                    "nvidia_smi_used_mb": 700,
                    "nvidia_smi_device_total_mb": 16303,
                    "nvidia_smi_utilization_gpu": 0,
                },
            ),
        ):
            result = nvidia_smi_keyframe_preflight(
                {
                    "baseline_framebuffer_mb": 700,
                    "safety_margin_mb": 256,
                    "estimated_clean_peak_mb": 15400,
                    "true_cfg_enabled": False,
                    "true_cfg_extra_mb": 0,
                    "canvas_width": 576,
                    "canvas_height": 864,
                    "generated_tokens": 1944,
                    "reference_tokens": 1107,
                }
            )

        self.assertGreater(result["vram_estimated_headroom_mb"], 0)
        self.assertGreaterEqual(result["vram_max_output_canvas"]["generated_tokens"], 1944)
        self.assertEqual(result["vram_max_output_canvas"]["width"] % 16, 0)
        self.assertEqual(result["vram_max_output_canvas"]["height"] % 16, 0)

    def test_keyframe_judge_writes_semantic_gate_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_keyframe_result(root)
            config = KeyframeJudgeConfig(
                judge_id=DEFAULT_JUDGE_ID,
                model=Path("/models/vlm/Qwen/Qwen2.5-VL-7B-Instruct"),
                repo_id=DEFAULT_JUDGE_REPO_ID,
                revision=DEFAULT_JUDGE_REVISION,
                dtype="bfloat16",
                attention_impl="sdpa",
                quantization=DEFAULT_JUDGE_QUANTIZATION,
                min_pixels=1,
                max_pixels=2,
                max_new_tokens=512,
                temperature=0.0,
            )

            judge_result = judge_keyframe_run(
                run_dir,
                config,
                project_root=Path.cwd(),
                runner=FakeJudgeRunner(),
            )
            judge_dir = run_dir / "judge" / DEFAULT_JUDGE_ID
            overlay_exists = Path(judge_result["candidates"][0]["overlay"]).exists()
            prompt_exists = (judge_dir / "prompts" / "seed_002.txt").exists()
            raw_exists = (judge_dir / "raw" / "seed_002.json").exists()

        self.assertEqual(judge_result["semantic_gate"]["passed"], ["seed_002", "seed_003"])
        self.assertEqual(judge_result["semantic_gate"]["blocked"][0]["candidate"], "seed_005")
        self.assertFalse(judge_result["semantic_gate"]["usable_for_auto_select"])
        self.assertEqual(judge_result["semantic_gate"]["selection_owner"], "condition_score")
        self.assertTrue(overlay_exists)
        self.assertTrue(prompt_exists)
        self.assertTrue(raw_exists)

    def test_keyframe_score_ranks_condition_match_from_saved_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir, poses = write_score_fixture(root)

            result = score_keyframe_run(
                run_dir,
                KeyframeScoreConfig(scorer_id="condition-test"),
                project_root=Path.cwd(),
                pose_extractor=FakePoseExtractor(poses),
                segmenter=FakeSegmenter(),
            )
            candidates = {candidate["candidate"]: candidate for candidate in result["candidates"]}

            self.assertEqual(result["ranking"]["best"], "seed_003")
            self.assertLess(
                candidates["seed_002"]["scores"]["final"],
                candidates["seed_003"]["scores"]["final"],
            )
            self.assertTrue(Path(result["outputs"]["scores"]).exists())
            self.assertTrue(Path(result["outputs"]["ranked_contact_sheet"]).exists())
            self.assertTrue(Path(result["outputs"]["condition_evidence_ranked"]).exists())

    def test_keyframe_score_uses_pose_keypoints_when_pose_asset_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir, poses = write_pose_score_fixture(root)

            result = score_keyframe_run(
                run_dir,
                KeyframeScoreConfig(scorer_id="pose-test"),
                project_root=Path.cwd(),
                pose_extractor=FakePoseExtractor(poses),
                segmenter=FakeSegmenter(),
            )
            candidates = {candidate["candidate"]: candidate for candidate in result["candidates"]}

            self.assertEqual(result["ranking"]["best"], "seed_003")
            self.assertGreater(candidates["seed_003"]["scores"]["pose"], candidates["seed_002"]["scores"]["pose"])
            self.assertEqual(candidates["seed_003"]["metrics"]["pose"]["common_keypoints"], 10)
            self.assertTrue(Path(result["outputs"]["pose_evidence_ranked"]).exists())

    def test_keyframe_score_uses_extracted_source_pose_when_metadata_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir, poses = write_pose_score_fixture(root)
            pose_path = root / "assets" / "pose.png"
            normalized_source = root / "assets" / "normalized.png"
            write_image(normalized_source, (160, 240), (240, 230, 220))
            metadata_path = pose_path.with_name(f"{pose_path.stem.removesuffix('_pose')}_extraction.json")
            write_json(
                metadata_path,
                {"assets": {"normalized_source": {"path": normalized_source.as_posix()}}},
            )
            poses[normalized_source.name] = poses["seed_002.png"]

            result = score_keyframe_run(
                run_dir,
                KeyframeScoreConfig(scorer_id="source-pose-test"),
                project_root=Path.cwd(),
                pose_extractor=FakePoseExtractor(poses),
                segmenter=FakeSegmenter(),
            )
            candidates = {candidate["candidate"]: candidate for candidate in result["candidates"]}

            self.assertEqual(result["ranking"]["best"], "seed_002")
            self.assertGreater(candidates["seed_002"]["scores"]["pose"], candidates["seed_003"]["scores"]["pose"])

    def test_keyframe_score_blocks_auto_select_when_pose_scores_degenerate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir, poses = write_pose_score_fixture(root)
            bad_pose = pose_keypoints(
                {
                    0: (0.08, 0.88),
                    1: (0.10, 0.80),
                    2: (0.12, 0.72),
                    5: (0.14, 0.68),
                    8: (0.16, 0.60),
                    9: (0.18, 0.48),
                    10: (0.20, 0.36),
                    11: (0.22, 0.62),
                    12: (0.24, 0.50),
                    13: (0.26, 0.38),
                }
            )
            poses = {name: bad_pose for name in poses}

            result = score_keyframe_run(
                run_dir,
                KeyframeScoreConfig(scorer_id="pose-degenerate-test"),
                project_root=Path.cwd(),
                pose_extractor=FakePoseExtractor(poses),
                segmenter=FakeSegmenter(),
            )

            self.assertFalse(result["selection"]["usable_for_auto_select"])
            self.assertEqual(
                result["selection"]["blockers"],
                ["pose_score_degenerate_all_candidates", "all_candidates_have_hard_rejects"],
            )

            with self.assertRaisesRegex(RuntimeError, "Refusing automatic score selection"):
                select_scored_keyframe_run(run_dir, scorer_id="pose-degenerate-test")

    def test_keyframe_score_select_writes_model_selected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir, poses = write_pose_score_fixture(root)

            score_keyframe_run(
                run_dir,
                KeyframeScoreConfig(scorer_id="pose-select-test"),
                project_root=Path.cwd(),
                pose_extractor=FakePoseExtractor(poses),
                segmenter=FakeSegmenter(),
            )
            selection = select_scored_keyframe_run(run_dir, scorer_id="pose-select-test")

            self.assertEqual(selection["best"], "seed_003")
            self.assertEqual(selection["selected"], ["seed_003"])
            self.assertEqual(selection["rejected"], ["seed_002"])
            self.assertTrue(Path(selection["outputs"]["selected"]).exists())
            self.assertTrue(Path(selection["outputs"]["rejected"]).exists())

    def test_keyframe_score_select_supports_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir, poses = write_score_fixture(root)

            score_keyframe_run(
                run_dir,
                KeyframeScoreConfig(scorer_id="top-k-test"),
                project_root=Path.cwd(),
                pose_extractor=FakePoseExtractor(poses),
                segmenter=FakeSegmenter(),
            )
            selection = select_scored_keyframe_run(run_dir, scorer_id="top-k-test", top_k=2)

            self.assertEqual(selection["best"], "seed_003")
            self.assertEqual(selection["selected"], ["seed_003", "seed_002"])
            self.assertEqual(selection["rejected"], ["seed_005"])

    def test_keyframe_refine_plans_arm_mask_from_target_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_path = write_refine_fixture(root)

            with patch(
                "aigen.keyframe_refine.count_kontext_prompt_tokens",
                return_value=PromptTokenCounts(clip=10, clip_limit=77, t5=18),
            ):
                resolved = plan_keyframe_refine_job(job_path, refine_profile(), project_root=Path.cwd())

        self.assertEqual(resolved["mask_plan"]["front_arm_indices"], [2, 3, 4])
        self.assertEqual(resolved["tokens"]["t5"], 18)
        self.assertLess(resolved["mask_plan"]["crop_box"][0], 40)

    def test_keyframe_refine_run_writes_mask_crop_and_preserves_outside_pixels(self) -> None:
        FakeRefiner.instances.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_path = write_refine_fixture(root)
            with (
                patch(
                    "aigen.keyframe_refine.count_kontext_prompt_tokens",
                    return_value=PromptTokenCounts(clip=10, clip_limit=77, t5=18),
                ),
                patch("aigen.keyframe_refine.KontextInpaintRefiner", FakeRefiner),
                patch("aigen.keyframe_refine.cuda_memory_stats", return_value={"max_allocated_mb": 0}),
                patch("aigen.keyframe_refine._generation_environment", return_value={"env": "fake"}),
                patch(
                    "aigen.keyframe_refine.nvidia_smi_preflight",
                    return_value={
                        "nvidia_smi_preflight_used_mb": 0,
                        "nvidia_smi_device_total_mb": 0,
                    },
                ),
            ):
                result = run_keyframe_refine_job(job_path, refine_profile(), project_root=Path.cwd())

            output_dir = root / "runs" / "punch_refine"
            output_path = Path(result["outputs"][0]["path"])

            self.assertTrue((output_dir / "resolved.json").exists())
            self.assertTrue((output_dir / "debug" / "mask_feather.png").exists())
            self.assertTrue((output_dir / "debug" / "crop.png").exists())
            self.assertTrue((output_dir / "contact_sheet.png").exists())
            self.assertTrue(output_path.exists())
            self.assertFalse(result["outputs"][0]["mask_change"]["hard_rejects"]["outside_feather_changed"])
            self.assertEqual(FakeRefiner.instances[0].seed, 101)
            self.assertTrue(FakeRefiner.instances[0].closed)

    def test_keyframe_polish_plan_is_static_and_model_free(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_path, plan_path = write_polish_fixture(root)

            result = plan_keyframe_polish(job_path, project_root=Path.cwd())

        self.assertEqual(result["status"], "planned")
        self.assertEqual(result["job_id"], "ai51.punch.seed060.polish")
        self.assertEqual(result["plan_path"], plan_path.resolve().as_posix())
        self.assertTrue(result["plan_exists"])
        self.assertEqual(result["base"]["candidate"], "seed_060")

    def test_keyframe_polish_diagnose_writes_model_discovered_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_path, _plan_path = write_polish_fixture(root)

            result = diagnose_keyframe_polish(
                job_path,
                config=KeyframeJudgeConfig(
                    judge_id=DEFAULT_JUDGE_ID,
                    model=Path("/models/vlm/Qwen/Qwen2.5-VL-7B-Instruct"),
                    repo_id=DEFAULT_JUDGE_REPO_ID,
                    revision=DEFAULT_JUDGE_REVISION,
                    dtype="bfloat16",
                    attention_impl="sdpa",
                    quantization=DEFAULT_JUDGE_QUANTIZATION,
                    min_pixels=1,
                    max_pixels=2,
                    max_new_tokens=512,
                    temperature=0.0,
                ),
                project_root=Path.cwd(),
                runner=FakePolishPlanner(),
            )

        self.assertEqual([region["id"] for region in result["polish_plan"]["regions"]], ["region_01"])
        self.assertEqual(result["polish_plan"]["regions"][0]["operation"], "expression_refine")
        self.assertGreaterEqual(len(result["evidence_images"]), 3)

    def test_keyframe_polish_preview_uses_model_plan_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_path, _plan_path = write_polish_fixture(root)

            with patch(
                "aigen.keyframe_polish.count_kontext_prompt_tokens",
                return_value=PromptTokenCounts(clip=8, clip_limit=77, t5=18),
            ):
                resolved = preview_keyframe_polish_job(
                    job_path,
                    refine_profile(),
                    project_root=Path.cwd(),
                    segmenter=FakeSegmenter(),
                    grounder=FakeGrounder(),
                )

        self.assertEqual([region["id"] for region in resolved["polish_plan"]["regions"]], ["region_01", "region_02"])
        self.assertEqual([plan["region_id"] for plan in resolved["mask_plan"]], ["region_01", "region_02"])
        self.assertEqual(resolved["mask_plan"][0]["grounding"]["source"], "fake-grounder")
        self.assertEqual(resolved["mask_plan"][0]["segmentation"]["method"], "fake-grounder-box-to-sam-mask")

    def test_keyframe_polish_run_preserves_outside_pixels(self) -> None:
        FakeRefiner.instances.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_path, _diagnosis_path = write_polish_fixture(root)
            with (
                patch(
                    "aigen.keyframe_polish.count_kontext_prompt_tokens",
                    return_value=PromptTokenCounts(clip=8, clip_limit=77, t5=18),
                ),
                patch("aigen.keyframe_polish.KontextInpaintRefiner", FakeRefiner),
                patch("aigen.keyframe_polish.cuda_memory_stats", return_value={"max_allocated_mb": 0}),
                patch("aigen.keyframe_polish._generation_environment", return_value={"env": "fake"}),
                patch(
                    "aigen.keyframe_polish.nvidia_smi_preflight",
                    return_value={
                        "nvidia_smi_preflight_used_mb": 0,
                        "nvidia_smi_device_total_mb": 0,
                    },
                ),
            ):
                result = run_keyframe_polish_job(
                    job_path,
                    refine_profile(),
                    project_root=Path.cwd(),
                    segmenter=FakeSegmenter(),
                    grounder=FakeGrounder(),
                )

            output_dir = root / "runs" / "punch_polish"
            output_path = Path(result["outputs"][0]["path"])

            self.assertTrue((output_dir / "resolved.json").exists())
            self.assertTrue((output_dir / "debug" / "region_01" / "mask_feather.png").exists())
            self.assertTrue((output_dir / "debug" / "region_02" / "mask_feather.png").exists())
            self.assertTrue((output_dir / "contact_sheet.png").exists())
            self.assertTrue(output_path.exists())
            self.assertFalse(result["outputs"][0]["mask_change"]["hard_rejects"]["outside_feather_changed"])
            self.assertEqual(FakeRefiner.instances[0].seed, 1201)
            self.assertTrue(FakeRefiner.instances[0].closed)

    def test_keyframe_polish_select_writes_final_composite(self) -> None:
        FakeRefiner.instances.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_path, _plan_path = write_polish_fixture(root)
            with (
                patch(
                    "aigen.keyframe_polish.count_kontext_prompt_tokens",
                    return_value=PromptTokenCounts(clip=8, clip_limit=77, t5=18),
                ),
                patch("aigen.keyframe_polish.KontextInpaintRefiner", FakeRefiner),
                patch("aigen.keyframe_polish.cuda_memory_stats", return_value={"max_allocated_mb": 0}),
                patch("aigen.keyframe_polish._generation_environment", return_value={"env": "fake"}),
                patch(
                    "aigen.keyframe_polish.nvidia_smi_preflight",
                    return_value={
                        "nvidia_smi_preflight_used_mb": 0,
                        "nvidia_smi_device_total_mb": 0,
                    },
                ),
            ):
                run_keyframe_polish_job(
                    job_path,
                    refine_profile(),
                    project_root=Path.cwd(),
                    segmenter=FakeSegmenter(),
                    grounder=FakeGrounder(),
                )
            result = select_keyframe_polish(
                job_path,
                config=KeyframeJudgeConfig(
                    judge_id=DEFAULT_JUDGE_ID,
                    model=Path("/models/vlm/Qwen/Qwen2.5-VL-7B-Instruct"),
                    repo_id=DEFAULT_JUDGE_REPO_ID,
                    revision=DEFAULT_JUDGE_REVISION,
                    dtype="bfloat16",
                    attention_impl="sdpa",
                    quantization=DEFAULT_JUDGE_QUANTIZATION,
                    min_pixels=1,
                    max_pixels=2,
                    max_new_tokens=512,
                    temperature=0.0,
                ),
                project_root=Path.cwd(),
                runner=FakePolishSelector(),
            )

            self.assertTrue(Path(result["final_composite"]["path"]).exists())
            self.assertEqual(result["regions"][0]["region_id"], "region_01")

    def test_qwen_judge_reports_missing_local_model_before_loading_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "Missing local judge model"):
                QwenKeyframeJudge(
                    KeyframeJudgeConfig(
                        judge_id=DEFAULT_JUDGE_ID,
                        model=Path(temp_dir) / "missing",
                        repo_id=DEFAULT_JUDGE_REPO_ID,
                        revision=DEFAULT_JUDGE_REVISION,
                        dtype="bfloat16",
                        attention_impl="sdpa",
                        quantization=DEFAULT_JUDGE_QUANTIZATION,
                        min_pixels=1,
                        max_pixels=2,
                        max_new_tokens=512,
                        temperature=0.0,
                    )
                )


if __name__ == "__main__":
    unittest.main()
