from __future__ import annotations

import unittest

import torch

from aigen.flux_geometry import fit_size_to_area
from aigen.generation.kontext_pose_control import (
    KontextPosePrepared,
    add_control_residuals,
    apply_control_residual_mask,
    extend_control_residuals,
    residual_suffix_is_zero,
)


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
    def test_extends_control_residuals_with_zero_reference_suffix(self) -> None:
        sample = torch.ones((2, 3, 4))
        extended = extend_control_residuals((sample,), total_image_tokens=5)

        self.assertEqual(extended[0].shape, (2, 5, 4))
        self.assertTrue(torch.equal(extended[0][:, :3], sample))
        self.assertTrue(torch.equal(extended[0][:, 3:], torch.zeros((2, 2, 4))))
        self.assertTrue(residual_suffix_is_zero(extended, generated_tokens=3))

    def test_reuses_control_residual_zero_suffix_cache(self) -> None:
        sample = torch.ones((2, 3, 4))
        cache = {}

        extend_control_residuals((sample,), total_image_tokens=5, zero_suffix_cache=cache)
        cached_suffix = next(iter(cache.values()))
        extend_control_residuals((sample,), total_image_tokens=5, zero_suffix_cache=cache)

        self.assertIs(next(iter(cache.values())), cached_suffix)

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

if __name__ == "__main__":
    unittest.main()
