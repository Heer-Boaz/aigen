from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from aigen.cli import CHARACTER_POSE_PROFILES, main
from aigen.generation.pose_control import (
    CharacterPoseResult,
    run_character_pose_control,
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
        path.write_text("pose", encoding="utf-8")


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
    def __init__(self) -> None:
        self.call_args: dict[str, object] | None = None
        self.cpu_offload_enabled = False
        self.vae = FakeVae()
        self.device: str | None = None

    def enable_model_cpu_offload(self) -> None:
        self.cpu_offload_enabled = True

    def to(self, device: str) -> FakePipeline:
        self.device = device
        return self

    def __call__(self, **kwargs: object) -> FakePipelineOutput:
        self.call_args = kwargs
        return FakePipelineOutput()


class FakeFluxControlNetModel:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: object) -> str:
        cls.calls.append({"model": model, **kwargs})
        return f"controlnet:{model}"


class FakeFluxControlNetPipeline:
    calls: list[dict[str, object]] = []
    pipelines: list[FakePipeline] = []

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: object) -> FakePipeline:
        pipeline = FakePipeline()
        cls.calls.append({"model": model, **kwargs})
        cls.pipelines.append(pipeline)
        return pipeline


def reset_pipeline() -> None:
    FakeFluxControlNetModel.calls.clear()
    FakeFluxControlNetPipeline.calls.clear()
    FakeFluxControlNetPipeline.pipelines.clear()


def fake_load_image(path: str) -> str:
    return f"loaded:{path}"


def fake_modules() -> dict[str, types.ModuleType]:
    torch = types.ModuleType("torch")
    torch.bfloat16 = "fake-bfloat16"
    torch.float16 = "fake-float16"
    torch.float32 = "fake-float32"
    torch.Generator = FakeGenerator

    diffusers = types.ModuleType("diffusers")
    diffusers.FluxControlNetModel = FakeFluxControlNetModel
    diffusers.FluxControlNetPipeline = FakeFluxControlNetPipeline

    diffusers_utils = types.ModuleType("diffusers.utils")
    diffusers_utils.load_image = fake_load_image
    return {
        "torch": torch,
        "diffusers": diffusers,
        "diffusers.utils": diffusers_utils,
    }


class CharacterPoseControlTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_pipeline()

    def test_runs_flux_controlnet_pose_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pose_image = root / "pose.png"
            output = root / "out" / "pose_result.png"
            pose_image.write_bytes(b"pose")

            with patch.dict(sys.modules, fake_modules()):
                result = run_character_pose_control(
                    "/models/FLUX.1-dev",
                    "/models/FLUX.1-dev-ControlNet-Union-Pro-2.0",
                    pose_image,
                    output,
                    "same anime girl, burgundy jacket, blue tie",
                    device="cuda:0",
                    dtype="bfloat16",
                    steps=30,
                    guidance_scale=3.5,
                    width=768,
                    height=1152,
                    seed=123,
                    controlnet_conditioning_scale=0.9,
                    control_guidance_end=0.65,
                    control_mode=4,
                )
            output_text = output.read_text(encoding="utf-8")

        self.assertEqual(output_text, "pose")
        self.assertEqual(FakeFluxControlNetModel.calls[0]["model"], "/models/FLUX.1-dev-ControlNet-Union-Pro-2.0")
        self.assertEqual(FakeFluxControlNetModel.calls[0]["torch_dtype"], "fake-bfloat16")
        self.assertEqual(FakeFluxControlNetModel.calls[0]["local_files_only"], True)
        self.assertEqual(FakeFluxControlNetPipeline.calls[0]["model"], "/models/FLUX.1-dev")
        self.assertEqual(
            FakeFluxControlNetPipeline.calls[0]["controlnet"],
            "controlnet:/models/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        )
        self.assertEqual(FakeFluxControlNetPipeline.calls[0]["torch_dtype"], "fake-bfloat16")
        self.assertEqual(FakeFluxControlNetPipeline.calls[0]["local_files_only"], True)
        pipeline = FakeFluxControlNetPipeline.pipelines[0]
        self.assertTrue(pipeline.cpu_offload_enabled)
        self.assertTrue(pipeline.vae.tiling_enabled)
        self.assertTrue(pipeline.vae.slicing_enabled)
        self.assertEqual(pipeline.call_args["prompt"], "same anime girl, burgundy jacket, blue tie")
        self.assertEqual(pipeline.call_args["control_image"], f"loaded:{pose_image.resolve().as_posix()}")
        self.assertEqual(pipeline.call_args["controlnet_conditioning_scale"], 0.9)
        self.assertEqual(pipeline.call_args["control_guidance_start"], 0.0)
        self.assertEqual(pipeline.call_args["control_guidance_end"], 0.65)
        self.assertEqual(pipeline.call_args["control_mode"], 4)
        self.assertEqual(pipeline.call_args["num_inference_steps"], 30)
        generator = pipeline.call_args["generator"]
        self.assertIsInstance(generator, FakeGenerator)
        self.assertEqual(generator.device, "cuda:0")
        self.assertEqual(generator.seed, 123)
        self.assertEqual(result.output_path, output.resolve().as_posix())
        self.assertEqual(result.control_mode, 4)

    def test_cli_character_pose_uses_flux_pose_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pose_image = root / "pose.png"
            output = root / "pose_result.png"
            pose_image.write_bytes(b"pose")
            profile = CHARACTER_POSE_PROFILES["flux-pose"]

            with patch(
                "aigen.cli.run_character_pose_control",
                return_value=CharacterPoseResult(
                    output_path=output.as_posix(),
                    base_model=profile.base_model,
                    controlnet_model=profile.controlnet_model,
                    pose_image=pose_image.as_posix(),
                    prompt="same character in exact pose",
                    negative_prompt="bad anatomy",
                    width=profile.width,
                    height=profile.height,
                    steps=profile.steps,
                    guidance_scale=profile.guidance_scale,
                    true_cfg_scale=profile.true_cfg_scale,
                    controlnet_conditioning_scale=profile.controlnet_conditioning_scale,
                    control_guidance_start=profile.control_guidance_start,
                    control_guidance_end=profile.control_guidance_end,
                    control_mode=profile.control_mode,
                    dtype=profile.dtype,
                    device="cuda",
                    cpu_offload=profile.cpu_offload,
                    seed=9,
                ),
            ) as run_pose_mock:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "generate",
                            "character-pose",
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
        self.assertEqual(payload["base_model"], profile.base_model)
        self.assertEqual(payload["controlnet_model"], profile.controlnet_model)
        run_pose_mock.assert_called_once()
        self.assertEqual(run_pose_mock.call_args.args[:5], (
            profile.base_model,
            profile.controlnet_model,
            pose_image,
            output,
            "same character in exact pose",
        ))
        self.assertEqual(run_pose_mock.call_args.kwargs["steps"], 30)
        self.assertEqual(run_pose_mock.call_args.kwargs["guidance_scale"], 3.5)
        self.assertEqual(run_pose_mock.call_args.kwargs["controlnet_conditioning_scale"], 0.9)
        self.assertEqual(run_pose_mock.call_args.kwargs["control_guidance_end"], 0.65)
        self.assertTrue(run_pose_mock.call_args.kwargs["cpu_offload"])

    def test_cli_character_pose_uses_flux_pose_4bit_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pose_image = root / "pose.png"
            output = root / "pose_result.png"
            pose_image.write_bytes(b"pose")
            profile = CHARACTER_POSE_PROFILES["flux-pose-4bit"]

            with patch(
                "aigen.cli.run_character_pose_control",
                return_value=CharacterPoseResult(
                    output_path=output.as_posix(),
                    base_model=profile.base_model,
                    controlnet_model=profile.controlnet_model,
                    pose_image=pose_image.as_posix(),
                    prompt="same character in exact pose",
                    negative_prompt="bad anatomy",
                    width=profile.width,
                    height=profile.height,
                    steps=profile.steps,
                    guidance_scale=profile.guidance_scale,
                    true_cfg_scale=profile.true_cfg_scale,
                    controlnet_conditioning_scale=profile.controlnet_conditioning_scale,
                    control_guidance_start=profile.control_guidance_start,
                    control_guidance_end=profile.control_guidance_end,
                    control_mode=profile.control_mode,
                    dtype=profile.dtype,
                    device="cuda",
                    cpu_offload=profile.cpu_offload,
                    seed=9,
                ),
            ) as run_pose_mock:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "generate",
                            "character-pose",
                            "--profile",
                            "flux-pose-4bit",
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
        self.assertEqual(payload["base_model"], profile.base_model)
        self.assertIn("FLUX.1-dev-bnb-4bit", payload["base_model"])
        self.assertEqual(run_pose_mock.call_args.kwargs["width"], 512)
        self.assertEqual(run_pose_mock.call_args.kwargs["height"], 1024)
        self.assertEqual(run_pose_mock.call_args.kwargs["steps"], 20)
        self.assertFalse(run_pose_mock.call_args.kwargs["cpu_offload"])


if __name__ == "__main__":
    unittest.main()
