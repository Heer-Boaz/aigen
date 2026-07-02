from __future__ import annotations

from dataclasses import dataclass

from aigen.generation.flux_components import CLIP_TOKENIZER_COMPONENT, T5_TOKENIZER_COMPONENT


@dataclass(frozen=True)
class PromptTokenCounts:
    clip: int
    clip_limit: int
    t5: int


def count_kontext_prompt_tokens(model: str, clip_prompt: str, t5_prompt: str) -> PromptTokenCounts:
    from transformers import CLIPTokenizer, T5TokenizerFast

    clip_tokenizer = CLIPTokenizer.from_pretrained(model, subfolder=CLIP_TOKENIZER_COMPONENT, local_files_only=True)
    t5_tokenizer = T5TokenizerFast.from_pretrained(model, subfolder=T5_TOKENIZER_COMPONENT, local_files_only=True)
    clip_tokens = len(clip_tokenizer(clip_prompt, padding=False, truncation=False).input_ids)
    t5_tokens = len(t5_tokenizer(t5_prompt, padding=False, truncation=False).input_ids)
    return PromptTokenCounts(clip=clip_tokens, clip_limit=clip_tokenizer.model_max_length, t5=t5_tokens)
