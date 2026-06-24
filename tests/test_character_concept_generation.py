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

from aigen.cli import CHARACTER_CONCEPT_PROFILES, main
from aigen.generation.character_concept import (
    CharacterConceptResult,
    run_character_concept,
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
        path.write_text("concept", encoding="utf-8")


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


class FakeFluxKontextPipeline:
    calls: list[dict[str, object]] = []
    pipelines: list[FakePipeline] = []

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: object) -> FakePipeline:
        pipeline = FakePipeline()
        cls.calls.append({"model": model, **kwargs})
        cls.pipelines.append(pipeline)
        return pipeline


class FakeFluxTransformer2DModel:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_single_file(cls, path: str, **kwargs: object) -> str:
        cls.calls.append({"path": path, **kwargs})
        return f"transformer:{path}"


def reset_pipeline() -> None:
    FakeFluxKontextPipeline.calls.clear()
    FakeFluxKontextPipeline.pipelines.clear()
    FakeFluxTransformer2DModel.calls.clear()


def fake_load_image(path: str) -> str:
    return f"loaded:{path}"


def fake_modules() -> dict[str, types.ModuleType]:
    torch = types.ModuleType("torch")
    torch.bfloat16 = "fake-bfloat16"
    torch.float16 = "fake-float16"
    torch.float32 = "fake-float32"
    torch.Generator = FakeGenerator

    diffusers = types.ModuleType("diffusers")
    diffusers.FluxKontextPipeline = FakeFluxKontextPipeline
    diffusers.FluxTransformer2DModel = FakeFluxTransformer2DModel

    diffusers_utils = types.ModuleType("diffusers.utils")
    diffusers_utils.load_image = fake_load_image
    return {
        "torch": torch,
        "diffusers": diffusers,
        "diffusers.utils": diffusers_utils,
    }


class CharacterConceptGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_pipeline()

    def test_runs_flux_kontext_reference_image_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "AI51.png"
            output = root / "out" / "concept.png"
            reference.write_bytes(b"image")

            with patch.dict(sys.modules, fake_modules()):
                result = run_character_concept(
                    "/models/FLUX.1-Kontext-dev",
                    reference,
                    output,
                    "create production anime character concept art",
                    device="cuda:0",
                    dtype="bfloat16",
                    steps=32,
                    guidance_scale=2.5,
                    true_cfg_scale=1.8,
                    width=1024,
                    height=1536,
                    seed=123,
                )
            output_text = output.read_text(encoding="utf-8")

        self.assertEqual(result.output_path, output.resolve().as_posix())
        self.assertEqual(output_text, "concept")
        self.assertEqual(FakeFluxKontextPipeline.calls[0]["model"], "/models/FLUX.1-Kontext-dev")
        self.assertNotIn("transformer", FakeFluxKontextPipeline.calls[0])
        self.assertEqual(FakeFluxKontextPipeline.calls[0]["torch_dtype"], "fake-bfloat16")
        self.assertEqual(FakeFluxKontextPipeline.calls[0]["local_files_only"], True)
        self.assertTrue(FakeFluxKontextPipeline.pipelines[0].cpu_offload_enabled)
        self.assertTrue(FakeFluxKontextPipeline.pipelines[0].vae.tiling_enabled)
        self.assertTrue(FakeFluxKontextPipeline.pipelines[0].vae.slicing_enabled)
        prompt = FakeFluxKontextPipeline.pipelines[0].call_args["prompt"]
        self.assertIn("create production anime character concept art", prompt)
        self.assertIn("complete figure and feet visible", prompt)
        self.assertEqual(FakeFluxKontextPipeline.pipelines[0].call_args["width"], 1024)
        self.assertEqual(FakeFluxKontextPipeline.pipelines[0].call_args["height"], 1536)
        self.assertEqual(FakeFluxKontextPipeline.pipelines[0].call_args["max_area"], 1024 * 1536)
        self.assertEqual(FakeFluxKontextPipeline.pipelines[0].call_args["num_inference_steps"], 32)
        self.assertEqual(FakeFluxKontextPipeline.pipelines[0].call_args["guidance_scale"], 2.5)
        self.assertEqual(FakeFluxKontextPipeline.pipelines[0].call_args["true_cfg_scale"], 1.8)
        self.assertIn("cropped feet", FakeFluxKontextPipeline.pipelines[0].call_args["negative_prompt"])
        generator = FakeFluxKontextPipeline.pipelines[0].call_args["generator"]
        self.assertIsInstance(generator, FakeGenerator)
        self.assertEqual(generator.device, "cuda:0")
        self.assertEqual(generator.seed, 123)
        self.assertEqual(result.width, 1024)
        self.assertEqual(result.height, 1536)
        self.assertEqual(result.framing, "full-body")
        result_json = result.to_json()
        self.assertIn("complete figure and feet visible", result_json["pipeline_prompt"])
        self.assertEqual(result_json["steps"], 32)
        self.assertEqual(result_json["guidance_scale"], 2.5)
        self.assertEqual(result_json["true_cfg_scale"], 1.8)
        self.assertEqual(result_json["dtype"], "bfloat16")
        self.assertEqual(result_json["device"], "cuda:0")
        self.assertTrue(result_json["cpu_offload"])
        self.assertIn("cropped feet", result_json["negative_prompt"])
        self.assertIsNone(result_json["transformer_single_file"])

    def test_can_override_transformer_from_single_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "AI51.png"
            output = root / "out" / "concept.png"
            transformer = root / "flux1-kontext-dev.safetensors"
            reference.write_bytes(b"image")
            transformer.write_bytes(b"weights")

            with patch.dict(sys.modules, fake_modules()):
                result = run_character_concept(
                    "/models/flux-kontext-base",
                    reference,
                    output,
                    "create production anime character concept art",
                    transformer_single_file=transformer,
                )

        self.assertEqual(FakeFluxTransformer2DModel.calls[0]["path"], transformer.resolve().as_posix())
        self.assertEqual(FakeFluxTransformer2DModel.calls[0]["config"], "/models/flux-kontext-base")
        self.assertEqual(FakeFluxTransformer2DModel.calls[0]["subfolder"], "transformer")
        self.assertEqual(FakeFluxTransformer2DModel.calls[0]["local_files_only"], True)
        self.assertEqual(
            FakeFluxKontextPipeline.calls[0]["transformer"],
            f"transformer:{transformer.resolve().as_posix()}",
        )
        self.assertEqual(result.transformer_single_file, transformer.resolve().as_posix())

    def test_cli_character_concept_writes_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "AI51.png"
            output = root / "concept.png"
            reference.write_bytes(b"image")
            stdout = io.StringIO()

            with patch(
                "aigen.cli.run_character_concept",
                return_value=CharacterConceptResult(
                    output_path=output.as_posix(),
                    model="/models/FLUX.1-Kontext-dev",
                    reference_image=reference.as_posix(),
                    prompt="make character concept art",
                    pipeline_prompt="make character concept art\n\nfull body",
                    negative_prompt="bad anatomy",
                    width=1024,
                    height=1536,
                    framing="full-body",
                    steps=30,
                    guidance_scale=2.5,
                    true_cfg_scale=1.8,
                    dtype="bfloat16",
                    device="cuda",
                    cpu_offload=True,
                    transformer_single_file="/models/bfl/flux1-kontext-dev.safetensors",
                    seed=9,
                ),
            ) as run_character_concept_mock:
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "generate",
                            "character-concept",
                            "--model",
                            "/models/FLUX.1-Kontext-dev",
                            "--transformer-single-file",
                            "/models/bfl/flux1-kontext-dev.safetensors",
                            "--reference-image",
                            str(reference),
                            "--prompt",
                            "make character concept art",
                            "--output",
                            str(output),
                            "--steps",
                            "30",
                            "--guidance-scale",
                            "2.5",
                            "--true-cfg-scale",
                            "1.8",
                            "--width",
                            "1024",
                            "--height",
                            "1536",
                            "--cpu-offload",
                            "--seed",
                            "9",
                        ]
                    )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["prompt"], "make character concept art")
        self.assertIn("full body", payload["pipeline_prompt"])
        self.assertEqual(payload["steps"], 30)
        self.assertEqual(payload["guidance_scale"], 2.5)
        self.assertEqual(payload["true_cfg_scale"], 1.8)
        self.assertEqual(payload["dtype"], "bfloat16")
        self.assertTrue(payload["cpu_offload"])
        self.assertEqual(payload["transformer_single_file"], "/models/bfl/flux1-kontext-dev.safetensors")
        run_character_concept_mock.assert_called_once()
        self.assertEqual(run_character_concept_mock.call_args.args[:4], (
            "/models/FLUX.1-Kontext-dev",
            reference,
            output,
            "make character concept art",
        ))
        self.assertEqual(run_character_concept_mock.call_args.kwargs["steps"], 30)
        self.assertEqual(run_character_concept_mock.call_args.kwargs["guidance_scale"], 2.5)
        self.assertEqual(run_character_concept_mock.call_args.kwargs["true_cfg_scale"], 1.8)
        self.assertEqual(run_character_concept_mock.call_args.kwargs["width"], 1024)
        self.assertEqual(run_character_concept_mock.call_args.kwargs["height"], 1536)
        self.assertEqual(run_character_concept_mock.call_args.kwargs["framing"], "full-body")
        self.assertEqual(run_character_concept_mock.call_args.kwargs["seed"], 9)
        self.assertTrue(run_character_concept_mock.call_args.kwargs["cpu_offload"])
        self.assertEqual(
            run_character_concept_mock.call_args.kwargs["transformer_single_file"],
            Path("/models/bfl/flux1-kontext-dev.safetensors"),
        )

    def test_cli_production_profile_uses_bf16_transformer_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "AI51.png"
            output = root / "concept.png"
            reference.write_bytes(b"image")
            profile = CHARACTER_CONCEPT_PROFILES["production"]

            with patch(
                "aigen.cli.run_character_concept",
                return_value=CharacterConceptResult(
                    output_path=output.as_posix(),
                    model=profile.model,
                    reference_image=reference.as_posix(),
                    prompt="make character concept art",
                    pipeline_prompt="make character concept art\n\nfull body",
                    negative_prompt="bad anatomy",
                    width=profile.width,
                    height=profile.height,
                    framing=profile.framing,
                    steps=profile.steps,
                    guidance_scale=profile.guidance_scale,
                    true_cfg_scale=profile.true_cfg_scale,
                    dtype=profile.dtype,
                    device="cuda",
                    cpu_offload=profile.cpu_offload,
                    transformer_single_file=profile.transformer_single_file.as_posix(),
                    seed=9,
                ),
            ) as run_character_concept_mock:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main(
                        [
                            "generate",
                            "character-concept",
                            "--profile",
                            "production",
                            "--reference-image",
                            str(reference),
                            "--prompt",
                            "make character concept art",
                            "--output",
                            str(output),
                            "--seed",
                            "9",
                            "--compact",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        run_character_concept_mock.assert_called_once()
        self.assertEqual(run_character_concept_mock.call_args.args[:4], (
            profile.model,
            reference,
            output,
            "make character concept art",
        ))
        self.assertEqual(run_character_concept_mock.call_args.kwargs["dtype"], "bfloat16")
        self.assertEqual(run_character_concept_mock.call_args.kwargs["steps"], profile.steps)
        self.assertEqual(run_character_concept_mock.call_args.kwargs["width"], profile.width)
        self.assertEqual(run_character_concept_mock.call_args.kwargs["height"], profile.height)
        self.assertTrue(run_character_concept_mock.call_args.kwargs["cpu_offload"])
        self.assertEqual(
            run_character_concept_mock.call_args.kwargs["transformer_single_file"],
            profile.transformer_single_file,
        )


if __name__ == "__main__":
    unittest.main()
