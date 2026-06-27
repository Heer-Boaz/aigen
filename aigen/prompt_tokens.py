from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTokenCounts:
    clip: int
    clip_limit: int
    t5: int


def count_kontext_prompt_tokens(model: str, clip_prompt: str, t5_prompt: str) -> PromptTokenCounts:
    from transformers import CLIPTokenizer, T5TokenizerFast

    clip_tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=True)
    t5_tokenizer = T5TokenizerFast.from_pretrained(model, subfolder="tokenizer_2", local_files_only=True)
    clip_tokens = len(clip_tokenizer(clip_prompt, padding=False, truncation=False).input_ids)
    t5_tokens = len(t5_tokenizer(t5_prompt, padding=False, truncation=False).input_ids)
    return PromptTokenCounts(clip=clip_tokens, clip_limit=clip_tokenizer.model_max_length, t5=t5_tokens)
