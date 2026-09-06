"""Real Diffusers preprocessing, packing and scheduling with CPU neural doubles."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

from aigen.generation.flux2_klein import (
    Flux2KleinPromptEmbedding, _denoise_flux2_klein_case, _denoise_flux2_klein_cases,
    _prepare_condition_images, _prepare_flux2_klein_cases,
)
from aigen.generation.image_generation_requests import ImageGenerationCaseRequest, ImageGenerationOutputRequest
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps


class RecordingTransformer:
    config = SimpleNamespace(in_channels=128)

    def __init__(self):
        self.context_ids = []

    def to(self, device):
        return self

    def __call__(self, *, hidden_states, img_ids, **kwargs):
        if hidden_states.shape[:2] != img_ids.shape[:2]:
            raise AssertionError("Latent tokens and position IDs disagree")
        self.context_ids.append(img_ids[:, :, 0].unique().tolist())
        return (torch.zeros_like(hidden_states),)


class CpuPipeline(Flux2KleinPipeline):
    def __init__(self):
        self.vae_scale_factor = 8
        self.default_sample_size = 128
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=16)
        self.vae = Mock(dtype=torch.float32)
        self.scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
        self.transformer = RecordingTransformer()
        self.encoded_sizes = []

    def encode_prompt(self, *, prompt_embeds, **kwargs):
        return prompt_embeds, torch.zeros((1, 1, 4))

    def _encode_vae_image(self, image, generator):
        self.encoded_sizes.append(tuple(image.shape[-2:]))
        # Deterministic stand-in for argmax VAE encoding, in Klein's patchified layout.
        pooled = torch.nn.functional.avg_pool2d(image[:, :1], 16)
        return pooled.repeat(1, 128, 1, 1)


@contextmanager
def cpu_execution():
    tensor_to = torch.Tensor.to

    def to_cpu(tensor, *args, **kwargs):
        if args and str(args[0]).startswith("cuda"):
            args = ("cpu", *args[1:])
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return tensor_to(tensor, *args, **kwargs)

    cpu_torch = SimpleNamespace(
        device=lambda _: torch.device("cpu"),
        Generator=lambda **_: torch.Generator(device="cpu"),
        no_grad=torch.no_grad, cat=torch.cat,
    )
    with patch.object(torch.Tensor, "to", to_cpu), patch("aigen.generation.flux2_klein._release_cuda"):
        yield cpu_torch


class KleinConditioningTests(unittest.TestCase):
    def test_reference_aspect_and_resolution_follow_upstream(self):
        pipeline = CpuPipeline()
        sizes = ((320, 640), (1024, 256), (2048, 2048))
        images = [Image.new("RGB", size) for size in sizes]
        try:
            tensors, width, height = _prepare_condition_images(pipeline, images)
            self.assertEqual((width, height), sizes[0])
            self.assertEqual([tuple(t.shape[-2:]) for t in tensors], [(640, 320), (256, 1024), (1024, 1024)])
        finally:
            for image in images:
                image.close()

    def test_canvas_and_batch_order_do_not_change_reference_context(self):
        with TemporaryDirectory() as directory, cpu_execution() as cpu_torch:
            paths = (Path(directory) / "portrait.png", Path(directory) / "landscape.png")
            for path, size, color in zip(paths, ((320, 640), (640, 320)), ("red", "blue"), strict=True):
                Image.new("RGB", size, color).save(path)
            cases = tuple(ImageGenerationCaseRequest(
                name=str(index), prompt="", image_paths=paths, width=w, height=h,
                outputs=(ImageGenerationOutputRequest(name=str(index), seed=index, path=Path(directory) / f"out{index}.png"),),
            ) for index, (w, h) in enumerate(((512, 768), (1536, 1536), (512, 768))))
            embeddings = {"": Flux2KleinPromptEmbedding("", torch.zeros((1, 1, 4)))}

            def prepare(pipeline, requests, strength):
                return _prepare_flux2_klein_cases(pipeline, cases=requests, prompt_embeddings=embeddings,
                                                  torch=cpu_torch, strength=strength, progress=Mock())

            pipeline = CpuPipeline()
            prepared = prepare(pipeline, cases, 0.5)
            # Two native-aspect refs, then one init per distinct output canvas.
            self.assertEqual(pipeline.encoded_sizes, [(640, 320), (320, 640), (768, 512), (1536, 1536)])
            self.assertIs(prepared[0].init_source_latent, prepared[2].init_source_latent)
            # Enlarging a red init fills the canvas; it must not get black crop padding.
            self.assertEqual(prepared[1].init_source_latent.min().item(), 1.0)
            self.assertIs(prepared[0].image_latents, prepared[1].image_latents)
            self.assertEqual(prepared[0].image_latent_ids[:, :, 0].unique().tolist(), [10, 20])
            alone = prepare(CpuPipeline(), (cases[1],), 0.5)[0]
            reversed_cases = prepare(CpuPipeline(), tuple(reversed(cases)), 0.5)
            plain = prepare(CpuPipeline(), (cases[1],), None)[0]
            for other in (alone, reversed_cases[1], plain):
                torch.testing.assert_close(other.image_latents, prepared[1].image_latents, rtol=0, atol=0)
                torch.testing.assert_close(other.image_latent_ids, prepared[1].image_latent_ids, rtol=0, atol=0)
            torch.testing.assert_close(alone.init_source_latent, prepared[1].init_source_latent, rtol=0, atol=0)

            # A real scheduler and latent packing exercise the >1MP strength path and repeated seeds.
            request = replace(cases[1], outputs=(cases[1].outputs[0], cases[1].outputs[0]))
            outputs = _denoise_flux2_klein_case(
                pipeline, case=replace(prepared[1], request=request), torch=cpu_torch,
                compute_empirical_mu=compute_empirical_mu, retrieve_timesteps=retrieve_timesteps,
                sampler="flowmatch-euler", progress=Mock(),
            )
            self.assertEqual(outputs[0].latents.shape, (1, 96 * 96, 128))
            self.assertEqual(outputs[0].latent_ids.shape, (1, 96 * 96, 4))
            unpacked = pipeline._unpack_latents_with_ids(outputs[0].latents, outputs[0].latent_ids)
            self.assertEqual(unpacked.shape, (1, 128, 96, 96))
            self.assertTrue(all(ids == [0, 10, 20] for ids in pipeline.transformer.context_ids))
            torch.testing.assert_close(outputs[0].latents, outputs[1].latents, rtol=0, atol=0)
            full_strength = []
            for case in (plain, prepare(CpuPipeline(), (cases[1],), 1.0)[0]):
                full_strength.append(_denoise_flux2_klein_case(
                    pipeline, case=case, torch=cpu_torch,
                    compute_empirical_mu=compute_empirical_mu, retrieve_timesteps=retrieve_timesteps,
                    sampler="flowmatch-euler", progress=Mock(),
                )[0].latents)
            torch.testing.assert_close(*full_strength, rtol=0, atol=0)

    def test_progress_counts_the_prepared_schedule_for_every_output(self):
        with TemporaryDirectory() as directory, cpu_execution() as cpu_torch:
            source = Path(directory) / "source.png"
            Image.new("RGB", (128, 128), "red").save(source)
            request = ImageGenerationCaseRequest(
                name="schedule", prompt="", image_paths=(source,), width=128, height=128,
                outputs=tuple(ImageGenerationOutputRequest(str(seed), seed, Path(directory)/f"{seed}.png") for seed in (1, 2)),
            )
            embeddings = {"": Flux2KleinPromptEmbedding("", torch.zeros((1, 1, 4)))}
            for strength, refs, expected_steps in ((0.5, (source,), 4), (1.0, (source,), 8), (0.5, (), 8)):
                with self.subTest(strength=strength, references=len(refs)):
                    pipeline = CpuPipeline()
                    prepared = _prepare_flux2_klein_cases(
                        pipeline, cases=(replace(request, image_paths=refs),), prompt_embeddings=embeddings,
                        torch=cpu_torch, strength=strength, progress=Mock(),
                    )
                    progress = Mock()
                    _denoise_flux2_klein_cases(
                        pipeline, prepared_cases=prepared, torch=cpu_torch, compute_empirical_mu=compute_empirical_mu,
                        retrieve_timesteps=retrieve_timesteps, sampler="flowmatch-euler", progress=progress,
                    )
                    progress.begin.assert_called_once_with(expected_steps, f"denoising 0/{expected_steps}")
                    self.assertEqual(progress.step.call_count, expected_steps)


if __name__ == "__main__":
    unittest.main()
