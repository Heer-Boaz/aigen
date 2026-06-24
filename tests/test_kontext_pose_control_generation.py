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
    fit_size_to_area,
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

    def disable_slicing(self) -> None:
        self.slicing_enabled = False

    def enable_tiling(self) -> None:
        self.tiling_enabled = True

    def disable_tiling(self) -> None:
        self.tiling_enabled = False


class FakePipeline:
    calls: list[dict[str, object]] = []
    pipelines: list[FakePipeline] = []

    def __init__(self) -> None:
        self.call_args: dict[str, object] = {}
        self.cpu_offload_enabled = False
        self.vae = FakeVae()
        self.device: str | None = None
        self.last_token_metadata = {
            "reference_width": 384,
            "reference_height": 768,
            "reference_tokens": 1152,
            "generated_tokens": 864,
            "text_tokens": 128,
            "total_tokens": 2144,
        }
        self.last_timings_ms = {
            "prompt_encode_ms": 1.0,
            "reference_vae_ms": 2.0,
            "control_vae_ms": 3.0,
            "controlnet_ms": 4.0,
            "transformer_ms": 5.0,
            "vae_decode_ms": 6.0,
            "pipeline_ms": 21.0,
        }

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
    torch_module.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        reset_peak_memory_stats=lambda _device: None,
        max_memory_allocated=lambda _device: 0,
        synchronize=lambda: None,
    )
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

    def test_fits_reference_size_to_area_and_multiple(self) -> None:
        self.assertEqual(
            fit_size_to_area(1024, 2048, max_area=384 * 768, multiple_of=16),
            (384, 768),
        )

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
                    width=384,
                    height=576,
                    reference_max_area=384 * 768,
                    max_sequence_length=128,
                    seed=123,
                    controlnet_conditioning_scale=0.50,
                    control_guidance_end=0.50,
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
        self.assertFalse(pipeline.vae.tiling_enabled)
        self.assertFalse(pipeline.vae.slicing_enabled)
        self.assertEqual(pipeline.call_args["image"], f"loaded:{reference_image.resolve().as_posix()}")
        self.assertEqual(pipeline.call_args["control_image"], f"loaded:{pose_image.resolve().as_posix()}")
        self.assertIn("same character running", pipeline.call_args["prompt"])
        self.assertEqual(pipeline.call_args["reference_max_area"], 384 * 768)
        self.assertEqual(pipeline.call_args["max_sequence_length"], 128)
        self.assertEqual(pipeline.call_args["controlnet_conditioning_scale"], 0.50)
        self.assertEqual(pipeline.call_args["control_guidance_end"], 0.50)
        generator = pipeline.call_args["generator"]
        self.assertIsInstance(generator, FakeGenerator)
        self.assertEqual(generator.seed, 123)
        self.assertEqual(result.reference_image, reference_image.resolve().as_posix())
        self.assertEqual(result.pose_image, pose_image.resolve().as_posix())
        self.assertEqual(result.reference_tokens, 1152)
        self.assertEqual(result.generated_tokens, 864)
        self.assertEqual(result.total_tokens, 2144)
        self.assertEqual(result.timings_ms["transformer_ms"], 5.0)
        self.assertIn("model_load_ms", result.timings_ms)

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
                    reference_width=384,
                    reference_height=768,
                    reference_max_area=profile.reference_max_area,
                    reference_tokens=1152,
                    generated_tokens=864,
                    text_tokens=128,
                    total_tokens=2144,
                    max_sequence_length=profile.max_sequence_length,
                    timings_ms={
                        "model_load_ms": 0.0,
                        "prompt_encode_ms": 1.0,
                        "reference_vae_ms": 2.0,
                        "control_vae_ms": 3.0,
                        "controlnet_ms": 4.0,
                        "transformer_ms": 5.0,
                        "vae_decode_ms": 6.0,
                        "pipeline_ms": 21.0,
                        "total_ms": 21.0,
                    },
                    peak_vram_mb=0,
                    steps=profile.steps,
                    guidance_scale=profile.guidance_scale,
                    true_cfg_scale=profile.true_cfg_scale,
                    controlnet_conditioning_scale=profile.controlnet_conditioning_scale,
                    control_guidance_start=profile.control_guidance_start,
                    control_guidance_end=profile.control_guidance_end,
                    dtype=profile.dtype,
                    device="cuda",
                    cpu_offload=profile.cpu_offload,
                    vae_tiling=profile.vae_tiling,
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
        self.assertEqual(run_pose_mock.call_args.kwargs["reference_max_area"], 384 * 768)
        self.assertEqual(run_pose_mock.call_args.kwargs["max_sequence_length"], 128)
        self.assertEqual(run_pose_mock.call_args.kwargs["vae_tiling"], False)
        self.assertEqual(run_pose_mock.call_args.kwargs["controlnet_conditioning_scale"], 0.50)
        self.assertEqual(run_pose_mock.call_args.kwargs["control_guidance_end"], 0.50)


if __name__ == "__main__":
    unittest.main()
