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

from PIL import Image

from aigen.cli import NUNCHAKU_KONTEXT_PROFILES, main
from aigen.generation.nunchaku_kontext import (
    NunchakuKontextResult,
    run_nunchaku_kontext,
)


class FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> FakeGenerator:
        self.seed = seed
        return self


class FakeParameter:
    device = "cuda:0"
    dtype = "torch.uint8"
    shape = (4, 1)


class FakeTransformer:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model: str) -> FakeTransformer:
        cls.calls.append({"model": model})
        return FakeTransformer()

    def named_parameters(self) -> list[tuple[str, FakeParameter]]:
        return [("transformer.block.weight", FakeParameter())]


class FakeImage:
    def save(self, path: Path) -> None:
        path.write_text("nunchaku", encoding="utf-8")


class FakePipelineOutput:
    images = [FakeImage()]


class FakeTokenizerOutput:
    input_ids = [1, 2, 3]


class FakeTokenizer:
    def __call__(self, *_args: object, **_kwargs: object) -> FakeTokenizerOutput:
        return FakeTokenizerOutput()


class FakeVae:
    def __init__(self) -> None:
        self.slicing_disabled = False
        self.tiling_disabled = False

    def disable_slicing(self) -> None:
        self.slicing_disabled = True

    def disable_tiling(self) -> None:
        self.tiling_disabled = True


class FakePipeline:
    calls: list[dict[str, object]] = []
    pipelines: list[FakePipeline] = []

    def __init__(self, transformer: FakeTransformer) -> None:
        self.call_args: dict[str, object] = {}
        self.transformer = transformer
        self.tokenizer_2 = FakeTokenizer()
        self.vae = FakeVae()
        self.vae_scale_factor = 8
        self.device: str | None = None

    @classmethod
    def from_pretrained(cls, base_model: str, **kwargs: object) -> FakePipeline:
        pipeline = FakePipeline(kwargs["transformer"])
        cls.calls.append({"base_model": base_model, **kwargs})
        cls.pipelines.append(pipeline)
        return pipeline

    def to(self, device: str) -> FakePipeline:
        self.device = device
        return self

    def __call__(self, **kwargs: object) -> FakePipelineOutput:
        self.call_args = kwargs
        callback = kwargs["callback_on_step_end"]
        for step in range(kwargs["num_inference_steps"]):
            callback(self, step, step, {"latents": f"latents-{step}"})
        return FakePipelineOutput()


def reset_fakes() -> None:
    FakeTransformer.calls.clear()
    FakePipeline.calls.clear()
    FakePipeline.pipelines.clear()


def fake_modules() -> dict[str, types.ModuleType]:
    torch = types.ModuleType("torch")
    torch.bfloat16 = "fake-bfloat16"
    torch.float16 = "fake-float16"
    torch.float32 = "fake-float32"
    torch.Generator = FakeGenerator
    torch.__version__ = "fake-torch"
    torch.version = types.SimpleNamespace(cuda="fake-cuda")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        reset_peak_memory_stats=lambda _device: None,
        synchronize=lambda: None,
        max_memory_allocated=lambda _device: 0,
        max_memory_reserved=lambda _device: 0,
        mem_get_info=lambda _device: (0, 0),
        get_device_name=lambda _device: "fake-gpu",
        get_device_capability=lambda _device: (0, 0),
    )

    diffusers = types.ModuleType("diffusers")
    diffusers.FluxKontextPipeline = FakePipeline

    nunchaku = types.ModuleType("nunchaku")
    nunchaku.NunchakuFluxTransformer2dModel = FakeTransformer
    return {
        "torch": torch,
        "diffusers": diffusers,
        "nunchaku": nunchaku,
    }


class NunchakuKontextTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_fakes()

    def test_runs_nunchaku_kontext_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.png"
            output = root / "out" / "nunchaku.png"
            Image.new("RGB", (1024, 2048), color=(255, 255, 255)).save(reference)

            with patch.dict(sys.modules, fake_modules()):
                with patch("aigen.generation.nunchaku_kontext.version", return_value="fake-nunchaku"):
                    result = run_nunchaku_kontext(
                        "/models/base",
                        "/models/nunchaku.safetensors",
                        reference,
                        output,
                        "same character running",
                        device="cuda:0",
                        dtype="bfloat16",
                        steps=3,
                        width=384,
                        height=576,
                        reference_max_area=384 * 768,
                        max_sequence_length=128,
                        seed=123,
                    )
            output_text = output.read_text(encoding="utf-8")

        self.assertEqual(output_text, "nunchaku")
        self.assertEqual(FakeTransformer.calls[0]["model"], "/models/nunchaku.safetensors")
        self.assertEqual(FakePipeline.calls[0]["base_model"], "/models/base")
        self.assertEqual(FakePipeline.calls[0]["transformer"].__class__, FakeTransformer)
        self.assertEqual(FakePipeline.calls[0]["torch_dtype"], "fake-bfloat16")
        pipeline = FakePipeline.pipelines[0]
        self.assertEqual(pipeline.device, "cuda:0")
        self.assertTrue(pipeline.vae.slicing_disabled)
        self.assertTrue(pipeline.vae.tiling_disabled)
        self.assertEqual(pipeline.call_args["width"], 384)
        self.assertEqual(pipeline.call_args["height"], 576)
        self.assertEqual(pipeline.call_args["max_sequence_length"], 128)
        self.assertEqual(pipeline.call_args["_auto_resize"], False)
        self.assertEqual(pipeline.call_args["generator"].seed, 123)
        self.assertEqual(result.reference_width, 384)
        self.assertEqual(result.reference_height, 768)
        self.assertEqual(result.reference_tokens, 1152)
        self.assertEqual(result.generated_tokens, 864)
        self.assertEqual(result.text_tokens, 128)
        self.assertEqual(result.total_tokens, 2144)
        self.assertEqual(len(result.step_ms), 3)
        self.assertEqual(result.environment["nunchaku_version"], "fake-nunchaku")
        self.assertEqual(result.parameter_locations[0]["name"], "transformer.block.weight")

    def test_cli_character_nunchaku_kontext_uses_local_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.png"
            output = root / "nunchaku.png"
            reference.write_bytes(b"reference")
            profile = NUNCHAKU_KONTEXT_PROFILES["local"]

            with patch(
                "aigen.cli.run_nunchaku_kontext",
                return_value=NunchakuKontextResult(
                    output_path=output.as_posix(),
                    base_model=profile.base_model,
                    transformer_model=profile.transformer_model,
                    reference_image=reference.as_posix(),
                    prompt="same character",
                    pipeline_prompt="full body\n\nsame character",
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
                    steps=profile.steps,
                    step_ms=[10.0, 5.0, 5.5],
                    warm_step_median_ms=5.5,
                    timings_ms={"model_load_ms": 1.0, "generation_ms": 20.0, "total_ms": 21.0},
                    memory={
                        "max_allocated_mb": 0,
                        "max_reserved_mb": 0,
                        "free_after_run_mb": 0,
                        "device_total_mb": 0,
                    },
                    environment={"nunchaku_version": "fake"},
                    parameter_locations=[],
                    guidance_scale=profile.guidance_scale,
                    true_cfg_scale=profile.true_cfg_scale,
                    dtype=profile.dtype,
                    device="cuda",
                    seed=7,
                ),
            ) as run_mock:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "generate",
                            "character-nunchaku-kontext",
                            "--reference-image",
                            str(reference),
                            "--prompt",
                            "same character",
                            "--output",
                            str(output),
                            "--seed",
                            "7",
                            "--compact",
                        ]
                    )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["total_tokens"], 2144)
        self.assertEqual(run_mock.call_args.args[:4], (
            profile.base_model,
            profile.transformer_model,
            reference,
            output,
        ))
        self.assertEqual(run_mock.call_args.kwargs["steps"], 3)
        self.assertEqual(run_mock.call_args.kwargs["reference_max_area"], 384 * 768)
        self.assertEqual(run_mock.call_args.kwargs["max_sequence_length"], 128)


if __name__ == "__main__":
    unittest.main()
