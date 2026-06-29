from __future__ import annotations

import unittest

from aigen.diffusers_kontext_adapter import kontext_inpaint_text_kwargs


class DiffusersKontextAdapterTests(unittest.TestCase):
    def test_inpaint_text_kwargs_keep_internal_negative_prompt_clean(self) -> None:
        kwargs = kontext_inpaint_text_kwargs(
            clip_prompt="short identity prompt",
            t5_prompt="detailed identity and action prompt",
            negative_prompt="wrong outfit",
        )

        self.assertEqual(kwargs["prompt"], "short identity prompt")
        self.assertEqual(kwargs["prompt_2"], "detailed identity and action prompt")
        self.assertEqual(kwargs["negative_prompt"], "wrong outfit")
        self.assertNotIn("negative_prompt_2", kwargs)


if __name__ == "__main__":
    unittest.main()
