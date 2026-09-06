"""CPU tests in the native LightX2V environment; no neural weights loaded."""
import unittest

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from aigen.generation.qwen_image_edit_sampling import QwenEditScheduler, pack_qwen_repaint_mask
from aigen.image_edit_defaults import QWEN_2511_SAMPLER


class QwenMaskedSamplingTests(unittest.TestCase):
    def scheduler(self, *, sampler=QWEN_2511_SAMPLER, strength=0.6):
        scheduler = object.__new__(QwenEditScheduler)
        scheduler.sampler = sampler
        scheduler.clear_mask()
        scheduler.scheduler = FlowMatchEulerDiscreteScheduler(shift=1.15)
        scheduler.scheduler.set_timesteps(8, device="cpu")
        scheduler.timesteps = scheduler.scheduler.timesteps
        scheduler.timesteps, scheduler.infer_steps = scheduler.get_timesteps(8, strength, "cpu")
        generator = torch.Generator().manual_seed(17)
        scheduler.latents = torch.randn(1, 4, 64, generator=generator)
        scheduler.source_latents = torch.randn(1, 4, 64, generator=generator)
        scheduler.repaint_mask = torch.zeros_like(scheduler.latents)
        scheduler.repaint_mask[:, ::2] = 1
        scheduler.ancestral_generator = torch.Generator().manual_seed(17)
        return scheduler

    def test_mask_keeps_four_spatial_values_per_packed_token(self):
        mask = torch.tensor([[[[0., 0.25], [0.75, 1.]]]])
        full = mask.repeat_interleave(8, 2).repeat_interleave(8, 3)
        packed = pack_qwen_repaint_mask(full, height=16, width=16)
        self.assertEqual(packed.shape, (1, 1, 64))
        torch.testing.assert_close(packed.reshape(16, 4), mask.reshape(1, 4).expand(16, 4))

    def test_strength_initializes_whole_canvas_and_every_step_uses_next_sigma(self):
        for sampler in (QWEN_2511_SAMPLER, "euler-ancestral"):
            for strength in (0.6, 1.0):
                with self.subTest(sampler=sampler, strength=strength):
                    scheduler = self.scheduler(sampler=sampler, strength=strength)
                    noise = scheduler.latents.clone()
                    source = scheduler.source_latents.clone()
                    begin = 8 - round(8 * strength)
                    sigma = scheduler.scheduler.sigmas[begin]
                    scheduler.prepare_i2i_denoise_strength_latents(None)
                    torch.testing.assert_close(scheduler.latents, (1 - sigma) * source + sigma * noise)
                    for index in range(scheduler.infer_steps):
                        scheduler.step_index = index
                        scheduler.noise_pred = torch.full_like(source, 0.4)
                        scheduler.step_post()
                        sigma_next = scheduler.scheduler.sigmas[begin + index + 1]
                        expected = (1 - sigma_next) * source + sigma_next * noise
                        torch.testing.assert_close(scheduler.latents[:, 1::2], expected[:, 1::2])
                    self.assertTrue(torch.equal(scheduler.latents[:, 1::2], source[:, 1::2]))
                    self.assertFalse(torch.equal(scheduler.latents[:, ::2], source[:, ::2]))
                    self.assertTrue(torch.equal(scheduler.source_noise, noise))
                    scheduler.clear_mask()
                    self.assertIsNone(scheduler.source_noise)
                    self.assertIsNone(scheduler.preserved_latents)

    def test_native_strength_rounding_and_zero_step_rejection(self):
        self.assertEqual(self.scheduler(strength=0.6).infer_steps, 5)
        with self.assertRaisesRegex(ValueError, "0 denoising steps"):
            self.scheduler(strength=0.01)

    def test_unmasked_default_matches_the_native_euler_scheduler(self):
        scheduler = self.scheduler(strength=1.0)
        scheduler.clear_mask()
        native = FlowMatchEulerDiscreteScheduler(shift=1.15)
        native.set_timesteps(8, device="cpu")
        expected = scheduler.latents.clone()
        for index, timestep in enumerate(native.timesteps):
            scheduler.step_index = index
            scheduler.noise_pred = torch.full_like(scheduler.latents, 0.4)
            expected = native.step(scheduler.noise_pred, timestep, expected, return_dict=False)[0]
            scheduler.step_post()
            self.assertTrue(torch.equal(scheduler.latents, expected))

    def test_source_shape_error_is_not_resized_or_selected_from_references(self):
        scheduler = self.scheduler()
        scheduler.source_latents = torch.zeros(1, 9, 64)
        with self.assertRaisesRegex(ValueError, "match"):
            scheduler.prepare_i2i_denoise_strength_latents(None)


if __name__ == "__main__":
    unittest.main()
