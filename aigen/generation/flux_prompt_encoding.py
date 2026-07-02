from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

from aigen.generation.flux_components import (
    CLIP_TEXT_ENCODER_COMPONENT,
    CLIP_TOKENIZER_COMPONENT,
    T5_TEXT_ENCODER_COMPONENT,
    T5_TOKENIZER_COMPONENT,
)
from aigen.generation.runtime_diagnostics import elapsed_ms, synchronized_time
from aigen.generation.runtime_types import resolve_torch_dtype


class FluxPromptEncodingError(RuntimeError):
    pass


class FluxPromptEncodingDependencyError(FluxPromptEncodingError):
    pass


@dataclass(frozen=True)
class FluxPromptEmbedding:
    prompt: str
    prompt_embeds: Any
    pooled_prompt_embeds: Any


def encode_flux_prompts(
    model: str,
    *,
    prompts: list[str],
    dtype: str,
    max_sequence_length: int,
    device: str = "cuda",
) -> tuple[dict[str, FluxPromptEmbedding], float]:
    torch, CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast = _load_flux_text_dependencies()
    start = synchronized_time(torch)
    torch_dtype = resolve_torch_dtype(torch, dtype, auto_value=None)
    torch_device = torch.device(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained(model, subfolder=CLIP_TOKENIZER_COMPONENT, local_files_only=True)
    t5_tokenizer = T5TokenizerFast.from_pretrained(model, subfolder=T5_TOKENIZER_COMPONENT, local_files_only=True)
    clip_text_encoder = CLIPTextModel.from_pretrained(
        model,
        subfolder=CLIP_TEXT_ENCODER_COMPONENT,
        torch_dtype=torch_dtype,
        local_files_only=True,
    ).to(torch_device)
    t5_text_encoder = T5EncoderModel.from_pretrained(
        model,
        subfolder=T5_TEXT_ENCODER_COMPONENT,
        torch_dtype=torch_dtype,
        local_files_only=True,
    ).to(torch_device)

    embeddings: dict[str, FluxPromptEmbedding] = {}
    with torch.no_grad():
        for prompt in dict.fromkeys(prompts):
            embeddings[prompt] = FluxPromptEmbedding(
                prompt=prompt,
                prompt_embeds=_encode_t5_prompt(
                    t5_tokenizer,
                    t5_text_encoder,
                    prompt=prompt,
                    device=torch_device,
                    max_sequence_length=max_sequence_length,
                ).cpu(),
                pooled_prompt_embeds=_encode_clip_prompt(
                    clip_tokenizer,
                    clip_text_encoder,
                    prompt=prompt,
                    device=torch_device,
                ).cpu(),
            )
    encode_ms = elapsed_ms(start, synchronized_time(torch))

    del clip_text_encoder, t5_text_encoder, clip_tokenizer, t5_tokenizer
    _release_cuda(torch)
    return embeddings, encode_ms


def _load_flux_text_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    except ImportError as exc:
        raise FluxPromptEncodingDependencyError(
            "FLUX prompt encoding requires `pip install -e .[generation]`"
        ) from exc
    return torch, CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


def _encode_clip_prompt(
    tokenizer: Any,
    text_encoder: Any,
    *,
    prompt: str,
    device: Any,
) -> Any:
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=False).pooler_output
    return prompt_embeds.to(dtype=text_encoder.dtype, device=device).view(1, -1)


def _encode_t5_prompt(
    tokenizer: Any,
    text_encoder: Any,
    *,
    prompt: str,
    device: Any,
    max_sequence_length: int,
) -> Any:
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=False)[0]
    return prompt_embeds.to(dtype=text_encoder.dtype, device=device)


def _release_cuda(torch: Any) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
