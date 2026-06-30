from __future__ import annotations

from typing import Any


DIFFUSERS_CLIP_PROMPT_ARGUMENT = "prompt"
DIFFUSERS_T5_PROMPT_ARGUMENT = "prompt" + "_" + "2"


def kontext_inpaint_text_kwargs(
    *,
    clip_prompt: str,
    t5_prompt: str,
    negative_prompt: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        DIFFUSERS_CLIP_PROMPT_ARGUMENT: clip_prompt,
        DIFFUSERS_T5_PROMPT_ARGUMENT: t5_prompt,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    return kwargs
