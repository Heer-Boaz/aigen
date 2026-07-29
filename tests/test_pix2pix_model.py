from __future__ import annotations

import unittest

import torch

from aigen.pix2pix.config import ModelConfig
from aigen.pix2pix.model import (
    ConditionalPatchDiscriminator,
    Pix2PixGenerator,
    discriminator_loss,
    generator_loss,
    model_parameter_report,
)


class Pix2PixModelTests(unittest.TestCase):
    def test_reference_model_shapes_and_parameter_counts(self) -> None:
        config = ModelConfig()
        generator = Pix2PixGenerator(config).eval()
        discriminator = ConditionalPatchDiscriminator(config).eval()

        report = model_parameter_report(generator, discriminator)
        self.assertEqual(report["generator"], 54_413_955)
        self.assertEqual(report["discriminator"], 2_768_705)

        source = torch.zeros(1, 3, 256, 256)
        with torch.inference_mode():
            generated = generator(source)
            logits = discriminator(source, generated)
        self.assertEqual(tuple(generated.shape), (1, 3, 256, 256))
        self.assertEqual(tuple(logits.shape), (1, 1, 30, 30))

    def test_losses_are_finite_and_differentiable(self) -> None:
        generated = torch.zeros(1, 3, 8, 8, requires_grad=True)
        target = torch.ones_like(generated)
        fake_logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
        real_logits = torch.ones_like(fake_logits)

        total, adversarial, reconstruction = generator_loss(
            fake_logits,
            generated,
            target,
            lambda_l1=100.0,
        )
        d_loss = discriminator_loss(real_logits, fake_logits)
        (total + d_loss).backward()

        self.assertTrue(torch.isfinite(total))
        self.assertGreater(adversarial.item(), 0)
        self.assertEqual(reconstruction.item(), 1.0)
        self.assertIsNotNone(generated.grad)

    def test_native_128_generator_and_patchgan_variants(self) -> None:
        source = torch.zeros(1, 3, 128, 128)
        expected_discriminators = {
            1: (139_585, (1, 1, 62, 62)),
            3: (2_768_705, (1, 1, 14, 14)),
        }
        for layers, (parameter_count, output_shape) in expected_discriminators.items():
            with self.subTest(discriminator_layers=layers):
                config = ModelConfig(
                    image_size=128,
                    discriminator_layers=layers,
                )
                generator = Pix2PixGenerator(config).eval()
                discriminator = ConditionalPatchDiscriminator(config).eval()
                report = model_parameter_report(generator, discriminator)
                with torch.inference_mode():
                    generated = generator(source)
                    logits = discriminator(source, generated)

                self.assertEqual(report["generator"], 41_828_995)
                self.assertEqual(report["discriminator"], parameter_count)
                self.assertEqual(tuple(generated.shape), (1, 3, 128, 128))
                self.assertEqual(tuple(logits.shape), output_shape)

    def test_native_128_two_billion_parameter_configuration(self) -> None:
        config = ModelConfig(
            image_size=128,
            generator_channels=448,
            discriminator_channels=64,
            discriminator_layers=3,
        )
        generator = Pix2PixGenerator(config, device="meta")
        discriminator = ConditionalPatchDiscriminator(config, device="meta")

        self.assertEqual(
            model_parameter_report(generator, discriminator),
            {
                "generator": 2_048_905_603,
                "discriminator": 2_768_705,
                "total": 2_051_674_308,
            },
        )
