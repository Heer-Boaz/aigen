from __future__ import annotations

import json
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from aigen.cli import main
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


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


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
        "id": "ai46.walk.contact.left.v1",
        "pipeline": {"profile": "nunchaku-kontext-pose-quality"},
        "character": {"id": "ai46", "reference": {"path": "assets/AI46.png"}},
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
            "directory": "runs/keyframes/ai46/walk_contact/v1",
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
        self.prepare_models_released = False

    def maybe_free_model_hooks(self) -> None:
        self.prepare_models_released = True

    def denoise_prepared(self, prepared: object, **kwargs: object) -> object:
        if not self.prepare_models_released:
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
        return [FakeImage(result.name) for result in denoised], 4.0

    def close(self) -> None:
        self.closed = True


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

            with patch("aigen.keyframes._count_prompt_tokens", return_value=(12, 77, 24)):
                resolved = plan_keyframe_job(job_path, profile(), project_root=Path.cwd())

        self.assertEqual(resolved["tokens"], {"clip": 12, "clip_limit": 77, "t5": 24, "t5_limit": 128})
        self.assertEqual(resolved["condition_plan"][0]["active_steps"], 15)
        self.assertEqual(resolved["condition_plan"][1]["active_steps"], 11)
        self.assertEqual(len(resolved["output"]["files"]), 2)
        self.assertIn("sha256", resolved["assets"]["pose"])

    def test_rejects_wrong_control_asset_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = job_payload(root)
            write_image(root / "assets" / "pose.png", (832, 1248), (0, 255, 0))
            job_path = root / "job.json"
            write_json(job_path, payload)

            with patch("aigen.keyframes._count_prompt_tokens", return_value=(12, 77, 24)):
                with self.assertRaisesRegex(KeyframeJobError, "Asset pose must be 512x768"):
                    validate_keyframe_job(job_path, profile(), project_root=Path.cwd())

    def test_run_writes_resolved_result_and_conditions(self) -> None:
        FakeSession.instances.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_path = root / "job.json"
            write_json(job_path, job_payload(root))
            with (
                patch("aigen.keyframes._count_prompt_tokens", return_value=(12, 77, 24)),
                patch("aigen.keyframes.CharacterKontextPoseSession", FakeSession),
                patch("aigen.keyframes.cuda_memory_stats", return_value={"max_allocated_mb": 1}),
                patch("aigen.keyframes._generation_environment", return_value={"env": "fake"}),
            ):
                result = run_keyframe_job(job_path, profile(), project_root=Path.cwd())

            output_dir = root / "runs" / "keyframes" / "ai46" / "walk_contact" / "v1"
            result_path = output_dir / "result.json"
            resolved_path = output_dir / "resolved.json"

            self.assertTrue(result_path.exists())
            self.assertTrue(resolved_path.exists())
            self.assertTrue((output_dir / "conditions" / "pose.png").exists())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["outputs"][0]["controlnet_metadata"]["conditions"], ["pose", "profile_contour"])
            self.assertEqual(
                FakeSession.instances[0].prepare_kwargs["t5_prompt"],
                "Full-body orthographic side-view gameplay keyframe.",
            )


if __name__ == "__main__":
    unittest.main()
