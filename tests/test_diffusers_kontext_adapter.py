from __future__ import annotations

import unittest

from aigen.diffusers_kontext_adapter import (
    DIFFUSERS_CLIP_PROMPT_ARGUMENT,
    DIFFUSERS_T5_PROMPT_ARGUMENT,
    kontext_inpaint_text_kwargs,
)


class DiffusersKontextAdapterTests(unittest.TestCase):
    def test_inpaint_text_kwargs_keep_internal_negative_prompt_clean(self) -> None:
        kwargs = kontext_inpaint_text_kwargs(
            clip_prompt="short identity prompt",
            t5_prompt="detailed identity and action prompt",
            negative_prompt="wrong outfit",
        )

        self.assertEqual(kwargs[DIFFUSERS_CLIP_PROMPT_ARGUMENT], "short identity prompt")
        self.assertEqual(kwargs[DIFFUSERS_T5_PROMPT_ARGUMENT], "detailed identity and action prompt")
        self.assertEqual(kwargs["negative_prompt"], "wrong outfit")
        self.assertEqual(set(kwargs), {DIFFUSERS_CLIP_PROMPT_ARGUMENT, DIFFUSERS_T5_PROMPT_ARGUMENT, "negative_prompt"})


if __name__ == "__main__":
    unittest.main()
