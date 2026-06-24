from __future__ import annotations

import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from aigen.cli import CHARACTER_KONTEXT_POSE_PROFILES, main
from aigen.generation.kontext_pose_control import (
    CharacterKontextPoseResult,
    extend_control_residuals,
    run_character_kontext_pose_control,
)


class FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> FakeGenerator:
        self.seed = seed
        return self


class FakeImage:
    def save(self, path: Path) -> None:
        path.write_text("kontext-pose", encoding="utf-8")


class FakePipelineOutput:
    images = [FakeImage()]


class FakeVae:
    def __init__(self) -> None:
        self.slicing_enabled = False
        self.tiling_enabled = False

    def enable_slicing(self) -> None:
        self.slicing_enabled = True

    def enable_tiling(self) -> None:
        self.tiling_enabled = True


class FakePipeline:
    calls: list[dict[str, object]] = []
    pipelines: list[FakePipeline] = []

    def __init__(self) -> None:
        self.call_args: dict[str, object] = {}
        self.cpu_offload_enabled = False
        self.vae = FakeVae()
        self.device: str | None = None

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: object) -> FakePipeline:
        pipeline = FakePipeline()
        cls.calls.append({"model": model, **kwargs})
        cls.pipelines.append(pipeline)
        return pipeline

    def enable_model_cpu_offload(self) -> None:
        self.cpu_offload_enabled = True

    def to(self, device: str) -> FakePipeline:
        self.device = device
        return self

    def __call__(self, **kwargs: object) -> FakePipelineOutput:
        self.call_args = kwargs
        return FakePipelineOutput()


class FakeControlNetModel:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: object) -> str:
        cls.calls.append({"model": model, **kwargs})
        return f"controlnet:{model}"


class FakeTransformerModel:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_single_file(cls, path: str, **kwargs: object) -> str:
        cls.calls.append({"path": path, **kwargs})
        return f"transformer:{path}"


def reset_fakes() -> None:
    FakePipeline.calls.clear()
    FakePipeline.pipelines.clear()
    FakeControlNetModel.calls.clear()
    FakeTransformerModel.calls.clear()


def fake_load_image(path: str) -> str:
    return f"loaded:{path}"


def fake_loader() -> tuple[types.ModuleType, type[FakePipeline], type[FakeControlNetModel], type[FakeTransformerModel], object]:
    torch_module = types.ModuleType("torch")
    torch_module.bfloat16 = "fake-bfloat16"
    torch_module.float16 = "fake-float16"
    torch_module.float32 = "fake-float32"
    torch_module.Generator = FakeGenerator
    return torch_module, FakePipeline, FakeControlNetModel, FakeTransformerModel, fake_load_image


class KontextPoseControlTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_fakes()

    def test_extends_control_residuals_with_zero_reference_suffix(self) -> None:
        sample = torch.ones((2, 3, 4))
        extended = extend_control_residuals((sample,), total_image_tokens=5)

        self.assertEqual(extended[0].shape, (2, 5, 4))
        self.assertTrue(torch.equal(extended[0][:, :3], sample))
        self.assertTrue(torch.equal(extended[0][:, 3:], torch.zeros((2, 2, 4))))

    def test_runs_kontext_pose_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference_image = root / "reference.png"
            pose_image = root / "pose.png"
            output = root / "out" / "pose_result.png"
            reference_image.write_bytes(b"reference")
            pose_image.write_bytes(b"pose")

            with patch(
                "aigen.generation.kontext_pose_control._load_flux_kontext_controlnet",
                side_effect=fake_loader,
            ):
                result = run_character_kontext_pose_control(
                    "/models/flux-kontext",
                    "/models/controlnet",
                    reference_image,
                    pose_image,
                    output,
                    "same character running",
                    device="cuda:0",
                    dtype="bfloat16",
                    steps=28,
                    guidance_scale=3.5,
                    width=768,
                    height=1152,
                    seed=123,
                    controlnet_conditioning_scale=0.65,
                    control_guidance_end=0.65,
                )
            output_text = output.read_text(encoding="utf-8")

        self.assertEqual(output_text, "kontext-pose")
        self.assertEqual(FakeControlNetModel.calls[0]["model"], "/models/controlnet")
        self.assertEqual(FakeControlNetModel.calls[0]["torch_dtype"], "fake-bfloat16")
        self.assertEqual(FakePipeline.calls[0]["model"], "/models/flux-kontext")
        self.assertEqual(FakePipeline.calls[0]["controlnet"], "controlnet:/models/controlnet")
        pipeline = FakePipeline.pipelines[0]
        self.assertEqual(pipeline.device, "cuda:0")
        self.assertFalse(pipeline.cpu_offload_enabled)
        self.assertEqual(pipeline.call_args["image"], f"loaded:{reference_image.resolve().as_posix()}")
        self.assertEqual(pipeline.call_args["control_image"], f"loaded:{pose_image.resolve().as_posix()}")
        self.assertIn("same character running", pipeline.call_args["prompt"])
        self.assertEqual(pipeline.call_args["controlnet_conditioning_scale"], 0.65)
        self.assertEqual(pipeline.call_args["control_guidance_end"], 0.65)
        generator = pipeline.call_args["generator"]
        self.assertIsInstance(generator, FakeGenerator)
        self.assertEqual(generator.seed, 123)
        self.assertEqual(result.reference_image, reference_image.resolve().as_posix())
        self.assertEqual(result.pose_image, pose_image.resolve().as_posix())

    def test_cli_character_kontext_pose_uses_local_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference_image = root / "reference.png"
            pose_image = root / "pose.png"
            output = root / "pose_result.png"
            reference_image.write_bytes(b"reference")
            pose_image.write_bytes(b"pose")
            profile = CHARACTER_KONTEXT_POSE_PROFILES["local"]

            with patch(
                "aigen.cli.run_character_kontext_pose_control",
                return_value=CharacterKontextPoseResult(
                    output_path=output.as_posix(),
                    model=profile.model,
                    controlnet_model=profile.controlnet_model,
                    reference_image=reference_image.as_posix(),
                    pose_image=pose_image.as_posix(),
                    prompt="same character in exact pose",
                    pipeline_prompt="full body\n\nsame character in exact pose",
                    negative_prompt="bad anatomy",
                    width=profile.width,
                    height=profile.height,
                    steps=profile.steps,
                    guidance_scale=profile.guidance_scale,
                    true_cfg_scale=profile.true_cfg_scale,
                    controlnet_conditioning_scale=profile.controlnet_conditioning_scale,
                    control_guidance_start=profile.control_guidance_start,
                    control_guidance_end=profile.control_guidance_end,
                    dtype=profile.dtype,
                    device="cuda",
                    cpu_offload=profile.cpu_offload,
                    transformer_single_file=None,
                    seed=9,
                ),
            ) as run_pose_mock:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "generate",
                            "character-kontext-pose",
                            "--reference-image",
                            str(reference_image),
                            "--pose-image",
                            str(pose_image),
                            "--prompt",
                            "same character in exact pose",
                            "--output",
                            str(output),
                            "--seed",
                            "9",
                            "--compact",
                        ]
                    )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["reference_image"], reference_image.as_posix())
        self.assertEqual(payload["pose_image"], pose_image.as_posix())
        self.assertEqual(run_pose_mock.call_args.args[:5], (
            profile.model,
            profile.controlnet_model,
            reference_image,
            pose_image,
            output,
        ))
        self.assertEqual(run_pose_mock.call_args.kwargs["steps"], 28)
        self.assertEqual(run_pose_mock.call_args.kwargs["controlnet_conditioning_scale"], 0.65)
        self.assertEqual(run_pose_mock.call_args.kwargs["control_guidance_end"], 0.65)


if __name__ == "__main__":
    unittest.main()
