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

import torch

from aigen.cli import CHARACTER_KONTEXT_POSE_PROFILES, main
from aigen.generation.kontext_pose_control import (
    CharacterKontextPoseResult,
    KontextPosePrepared,
    add_control_residuals,
    apply_control_residual_mask,
    extend_control_residuals,
    fit_size_to_area,
    residual_suffix_is_zero,
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


class FakeParameter:
    device = "cuda:0"
    dtype = "torch.float16"
    shape = (2, 3)


class FakeTransformer:
    class Config:
        quantization_config = "fake-quantization-config"

    config = Config()

    def named_parameters(self) -> list[tuple[str, FakeParameter]]:
        return [("transformer.block.weight", FakeParameter())]


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
        self.transformer = FakeTransformer()
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
        self.last_controlnet_step_ms = [0.0, 4.0]
        self.last_transformer_step_ms = [2.0, 3.0]
        self.last_controlnet_active_steps = 1
        self.last_controlnet_metadata = {
            "generated_tokens": 864,
            "reference_tokens": 1152,
            "combined_image_tokens": 2016,
            "double_residual_count": 6,
            "single_residual_count": 0,
            "double_residual_shapes": [[1, 2016, 3072]],
            "single_residual_shapes": [],
            "reference_suffix_zero": True,
            "controlnet_blocks_repeat": False,
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


class FakeNunchakuTransformer:
    calls: list[dict[str, object]] = []

    def __init__(self, model: str) -> None:
        self.model = model
        self.attention_impl: str | None = None

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: object) -> FakeNunchakuTransformer:
        transformer = FakeNunchakuTransformer(model)
        cls.calls.append({"model": model, "transformer": transformer, **kwargs})
        return transformer

    def set_attention_impl(self, attention_impl: str) -> None:
        self.attention_impl = attention_impl


def reset_fakes() -> None:
    FakePipeline.calls.clear()
    FakePipeline.pipelines.clear()
    FakeControlNetModel.calls.clear()
    FakeTransformerModel.calls.clear()
    FakeNunchakuTransformer.calls.clear()


def fake_load_image(path: str) -> str:
    return f"loaded:{path}"


def fake_loader() -> tuple[types.ModuleType, type[FakePipeline], type[FakeControlNetModel], type[FakeTransformerModel], object]:
    torch_module = types.ModuleType("torch")
    torch_module.bfloat16 = "fake-bfloat16"
    torch_module.float16 = "fake-float16"
    torch_module.float32 = "fake-float32"
    torch_module.Generator = FakeGenerator
    torch_module.equal = torch.equal
    torch_module.__version__ = "fake-torch"
    torch_module.version = types.SimpleNamespace(cuda="fake-cuda")
    torch_module.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        reset_peak_memory_stats=lambda _device: None,
        max_memory_allocated=lambda _device: 0,
        max_memory_reserved=lambda _device: 0,
        mem_get_info=lambda _device: (0, 0),
        get_device_name=lambda _device: "fake-gpu",
        get_device_capability=lambda _device: (0, 0),
        synchronize=lambda: None,
    )
    return torch_module, FakePipeline, FakeControlNetModel, FakeTransformerModel, fake_load_image


def fake_prepared(seed: int, *, control_image: object = "current-control") -> KontextPosePrepared:
    return KontextPosePrepared(
        prompt_embeds=torch.ones((1, 4, 3)),
        pooled_prompt_embeds=torch.ones((1, 3)),
        text_ids=torch.ones((4, 3)),
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        negative_text_ids=None,
        do_true_cfg=False,
        base_latents=torch.ones((1, 2, 3)),
        image_latents=torch.ones((1, 5, 3)),
        generated_img_ids=torch.ones((2, 3)),
        combined_img_ids=torch.ones((7, 3)),
        control_image=control_image,
        controlnet_blocks_repeat=False,
        transformer_guidance=None,
        controlnet_guidance=None,
        true_cfg_scale=1.0,
        width=384,
        height=576,
        batch_size=1,
        num_images_per_prompt=1,
        num_channels_latents=16,
        dtype=torch.float32,
        device="cpu",
        seed=seed,
        steps=20,
        token_metadata={},
        timings_ms={},
    )


class KontextPoseControlTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_fakes()

    def test_extends_control_residuals_with_zero_reference_suffix(self) -> None:
        sample = torch.ones((2, 3, 4))
        extended = extend_control_residuals((sample,), total_image_tokens=5)

        self.assertEqual(extended[0].shape, (2, 5, 4))
        self.assertTrue(torch.equal(extended[0][:, :3], sample))
        self.assertTrue(torch.equal(extended[0][:, 3:], torch.zeros((2, 2, 4))))
        self.assertTrue(residual_suffix_is_zero(extended, generated_tokens=3))

    def test_sums_control_residuals_before_reference_extension(self) -> None:
        first = [torch.ones((1, 3, 2)), torch.full((1, 3, 2), 2.0)]
        second = [torch.full((1, 3, 2), 3.0), torch.full((1, 3, 2), 4.0)]

        summed = add_control_residuals(None, first)
        summed = add_control_residuals(summed, second)
        extended = extend_control_residuals(summed, total_image_tokens=5)

        self.assertTrue(torch.equal(extended[0][:, :3], torch.full((1, 3, 2), 4.0)))
        self.assertTrue(torch.equal(extended[1][:, :3], torch.full((1, 3, 2), 6.0)))
        self.assertTrue(residual_suffix_is_zero(extended, generated_tokens=3))

    def test_applies_control_residual_mask_per_generated_token(self) -> None:
        samples = [torch.ones((1, 3, 2))]
        mask = torch.tensor([[[1.0], [0.5], [0.0]]])

        masked = apply_control_residual_mask(samples, mask)

        self.assertTrue(
            torch.equal(
                masked[0],
                torch.tensor([[[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]]]),
            )
        )

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
        self.assertEqual(result.transformer_step_ms, [2.0, 3.0])
        self.assertEqual(result.controlnet_step_ms, [0.0, 4.0])
        self.assertEqual(result.controlnet_active_steps, 1)
        self.assertEqual(result.memory["max_allocated_mb"], 0)
        self.assertEqual(result.environment["torch_version"], "fake-torch")
        self.assertEqual(result.parameter_locations[0]["name"], "transformer.block.weight")

    def test_runs_kontext_pose_generation_with_nunchaku_transformer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference_image = root / "reference.png"
            pose_image = root / "pose.png"
            nunchaku_model = root / "nunchaku.safetensors"
            output = root / "out" / "pose_result.png"
            reference_image.write_bytes(b"reference")
            pose_image.write_bytes(b"pose")
            nunchaku_model.write_bytes(b"nunchaku")
            nunchaku = types.ModuleType("nunchaku")
            nunchaku.NunchakuFluxTransformer2dModel = FakeNunchakuTransformer

            with patch.dict(sys.modules, {"nunchaku": nunchaku}):
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
                        steps=3,
                        width=384,
                        height=576,
                        nunchaku_transformer_model=nunchaku_model,
                        attention_impl="nunchaku-fp16",
                    )

        self.assertEqual(FakeNunchakuTransformer.calls[0]["model"], nunchaku_model.resolve().as_posix())
        self.assertEqual(FakeNunchakuTransformer.calls[0]["torch_dtype"], "fake-bfloat16")
        self.assertEqual(FakeNunchakuTransformer.calls[0]["offload"], False)
        transformer = FakeNunchakuTransformer.calls[0]["transformer"]
        self.assertEqual(transformer.attention_impl, "nunchaku-fp16")
        self.assertEqual(FakePipeline.calls[0]["transformer"], transformer)
        self.assertEqual(result.nunchaku_transformer_model, nunchaku_model.resolve().as_posix())
        self.assertEqual(result.attention_impl, "nunchaku-fp16")

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
                    transformer_step_ms=[2.0, 3.0],
                    controlnet_step_ms=[0.0, 4.0],
                    controlnet_active_steps=1,
                    controlnet_metadata={
                        "generated_tokens": 864,
                        "reference_tokens": 1152,
                        "combined_image_tokens": 2016,
                        "double_residual_count": 6,
                        "single_residual_count": 0,
                        "double_residual_shapes": [[1, 2016, 3072]],
                        "single_residual_shapes": [],
                        "reference_suffix_zero": True,
                        "controlnet_blocks_repeat": False,
                    },
                    memory={
                        "max_allocated_mb": 0,
                        "max_reserved_mb": 0,
                        "free_after_run_mb": 0,
                        "device_total_mb": 0,
                    },
                    environment={
                        "torch_version": "fake-torch",
                        "torch_cuda_version": "fake-cuda",
                        "bitsandbytes_version": "fake-bnb",
                        "transformer_class": "FakeTransformer",
                        "transformer_device_map": None,
                        "transformer_quantization_config": "fake-quantization-config",
                    },
                    parameter_locations=[
                        {
                            "name": "transformer.block.weight",
                            "class": "FakeParameter",
                            "device": "cuda:0",
                            "dtype": "torch.float16",
                            "shape": [2, 3],
                        }
                    ],
                    steps=profile.steps,
                    guidance_scale=profile.guidance_scale,
                    true_cfg_scale=profile.true_cfg_scale,
                    controlnet_conditioning_scale=profile.controlnet_conditioning_scale,
                    control_guidance_start=profile.control_guidance_start,
                    control_guidance_end=profile.control_guidance_end,
                    dtype=profile.dtype,
                    device="cuda",
                    pipeline_cpu_offload=profile.pipeline_cpu_offload,
                    nunchaku_layer_offload=False,
                    vae_tiling=profile.vae_tiling,
                    transformer_single_file=None,
                    nunchaku_transformer_model=None,
                    attention_impl=None,
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

    def test_cli_character_nunchaku_kontext_pose_uses_local_profile(self) -> None:
        from aigen.cli import NUNCHAKU_KONTEXT_POSE_PROFILES

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference_image = root / "reference.png"
            pose_image = root / "pose.png"
            output = root / "pose_result.png"
            reference_image.write_bytes(b"reference")
            pose_image.write_bytes(b"pose")
            profile = NUNCHAKU_KONTEXT_POSE_PROFILES["local"]

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
                        "controlnet_ms": 0.0,
                        "transformer_ms": 5.0,
                        "vae_decode_ms": 6.0,
                        "pipeline_ms": 21.0,
                        "total_ms": 21.0,
                    },
                    transformer_step_ms=[2.0, 3.0],
                    controlnet_step_ms=[0.0, 0.0],
                    controlnet_active_steps=0,
                    controlnet_metadata={
                        "generated_tokens": 864,
                        "reference_tokens": 1152,
                        "combined_image_tokens": 2016,
                        "double_residual_count": 0,
                        "single_residual_count": 0,
                        "double_residual_shapes": [],
                        "single_residual_shapes": [],
                        "reference_suffix_zero": True,
                        "controlnet_blocks_repeat": False,
                    },
                    memory={
                        "max_allocated_mb": 0,
                        "max_reserved_mb": 0,
                        "free_after_run_mb": 0,
                        "device_total_mb": 0,
                    },
                    environment={
                        "torch_version": "fake-torch",
                        "torch_cuda_version": "fake-cuda",
                        "bitsandbytes_version": "fake-bnb",
                        "transformer_class": "FakeTransformer",
                        "transformer_device_map": None,
                        "transformer_quantization_config": "fake-quantization-config",
                    },
                    parameter_locations=[],
                    steps=profile.steps,
                    guidance_scale=profile.guidance_scale,
                    true_cfg_scale=profile.true_cfg_scale,
                    controlnet_conditioning_scale=profile.controlnet_conditioning_scale,
                    control_guidance_start=profile.control_guidance_start,
                    control_guidance_end=profile.control_guidance_end,
                    dtype=profile.dtype,
                    device="cuda",
                    pipeline_cpu_offload=profile.pipeline_cpu_offload,
                    nunchaku_layer_offload=profile.nunchaku_layer_offload,
                    vae_tiling=profile.vae_tiling,
                    transformer_single_file=None,
                    nunchaku_transformer_model=profile.nunchaku_transformer_model.as_posix(),
                    attention_impl=profile.attention_impl,
                    seed=9,
                ),
            ) as run_pose_mock:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "generate",
                            "character-nunchaku-kontext-pose",
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
        self.assertEqual(payload["nunchaku_transformer_model"], profile.nunchaku_transformer_model.as_posix())
        self.assertEqual(payload["attention_impl"], "nunchaku-fp16")
        self.assertEqual(run_pose_mock.call_args.args[:6], (
            profile.model,
            profile.controlnet_model,
            reference_image,
            pose_image,
            output,
            "same character in exact pose",
        ))
        self.assertEqual(
            run_pose_mock.call_args.kwargs["nunchaku_transformer_model"],
            profile.nunchaku_transformer_model,
        )
        self.assertEqual(run_pose_mock.call_args.kwargs["attention_impl"], "nunchaku-fp16")
        self.assertEqual(run_pose_mock.call_args.kwargs["pipeline_cpu_offload"], True)
        self.assertEqual(run_pose_mock.call_args.kwargs["nunchaku_layer_offload"], False)
        self.assertEqual(run_pose_mock.call_args.kwargs["steps"], 20)
        self.assertEqual(run_pose_mock.call_args.kwargs["controlnet_conditioning_scale"], 0.50)
        self.assertEqual(run_pose_mock.call_args.kwargs["control_guidance_end"], 0.50)

if __name__ == "__main__":
    unittest.main()
