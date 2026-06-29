from __future__ import annotations

from typing import Any


def kontext_inpaint_text_kwargs(
    *,
    clip_prompt: str,
    t5_prompt: str,
    negative_prompt: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": clip_prompt,
        "prompt_2": t5_prompt,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    return kwargs
