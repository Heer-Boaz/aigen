from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image

from aigen.generation.prompt_encoding import release_prompt_encoder_memory, tensor_to_cpu
from aigen.generation.runtime_diagnostics import elapsed_ms, synchronized_time
from aigen.generation.runtime_types import resolve_torch_dtype
from aigen.progress import StatusReporter


QWEN_IMAGE_EDIT_NEGATIVE_PROMPT = " "
QWEN_IMAGE_EDIT_PROMPT_ENCODER_SUBFOLDER = "text_encoder_bnb8"
QWEN_IMAGE_EDIT_ATTENTION_IMPL = "sdpa"


class QwenPromptEncodingError(RuntimeError):
    pass


class QwenPromptEncodingDependencyError(QwenPromptEncodingError):
    pass


@dataclass(frozen=True)
class QwenImageEditPromptRequest:
    name: str
    prompt: str
    reference_images: tuple[Image.Image, ...]


@dataclass(frozen=True)
class QwenImageEditPromptEmbedding:
    name: str
    prompt: str
    prompt_embeds: Any
    prompt_embeds_mask: Any | None
    negative_prompt_embeds: Any | None
    negative_prompt_embeds_mask: Any | None


def encode_qwen_image_edit_prompts(
    model: str,
    *,
    requests: Sequence[QwenImageEditPromptRequest],
    dtype: str,
    true_cfg_scale: float,
    max_sequence_length: int,
    device: str = "cuda",
    attention_impl: str = QWEN_IMAGE_EDIT_ATTENTION_IMPL,
    progress: StatusReporter | None = None,
) -> tuple[dict[str, QwenImageEditPromptEmbedding], float]:
    (
        torch,
        pipeline_class,
        text_encoder_class,
        processor_class,
        condition_image_size,
        calculate_dimensions,
    ) = _load_qwen_prompt_encoding_dependencies()
    _suppress_known_prompt_encoder_warnings()
    progress_total = 5 + len(requests)
    if true_cfg_scale > 1.0:
        progress_total += len(requests)
    if progress is not None:
        progress.begin(progress_total, "qwen prompt conditioning")
    _phase(progress, "reset qwen prompt encoder memory stats")
    _reset_peak_memory_stats(torch, device)
    _step(progress, "reset qwen prompt encoder memory stats")
    start = synchronized_time(torch)
    torch_dtype = resolve_torch_dtype(torch, dtype, auto_value=None)
    torch_device = torch.device(device)
    _phase(progress, "load qwen prompt encoder")
    text_encoder = _load_qwen_text_encoder(
        text_encoder_class,
        model,
        torch_dtype=torch_dtype,
        torch_device=torch_device,
        attention_impl=attention_impl,
    )
    _step(progress, "loaded qwen prompt encoder")
    processor = None
    pipeline = None
    embeddings: dict[str, QwenImageEditPromptEmbedding] = {}
    try:
        _phase(progress, "load qwen prompt processor")
        processor = processor_class.from_pretrained(
            model,
            subfolder="processor",
            local_files_only=True,
        )
        _step(progress, "loaded qwen prompt processor")
        _phase(progress, "build qwen prompt conditioning pipeline")
        pipeline = pipeline_class(
            scheduler=None,
            vae=None,
            text_encoder=text_encoder,
            tokenizer=None,
            processor=processor,
            transformer=None,
        )
        pipeline.set_progress_bar_config(disable=True)
        _step(progress, "built qwen prompt conditioning pipeline")

        with torch.inference_mode():
            for request in requests:
                _phase(progress, f"encode qwen prompt {request.name}")
                condition_images = _condition_images(
                    pipeline,
                    request.reference_images,
                    condition_image_size=condition_image_size,
                    calculate_dimensions=calculate_dimensions,
                )
                prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                    prompt=request.prompt,
                    image=condition_images,
                    device=torch_device,
                    num_images_per_prompt=1,
                    max_sequence_length=max_sequence_length,
                )
                prompt_embeds_mask = _prompt_mask_or_ones(torch, prompt_embeds, prompt_embeds_mask)
                negative_prompt_embeds = None
                negative_prompt_embeds_mask = None
                if true_cfg_scale > 1.0:
                    _phase(progress, f"encode qwen negative prompt {request.name}")
                    negative_prompt_embeds, negative_prompt_embeds_mask = pipeline.encode_prompt(
                        prompt=QWEN_IMAGE_EDIT_NEGATIVE_PROMPT,
                        image=condition_images,
                        device=torch_device,
                        num_images_per_prompt=1,
                        max_sequence_length=max_sequence_length,
                    )
                    negative_prompt_embeds_mask = _prompt_mask_or_ones(
                        torch,
                        negative_prompt_embeds,
                        negative_prompt_embeds_mask,
                    )
                    _step(progress, f"encoded qwen negative prompt {request.name}")
                embeddings[request.name] = QwenImageEditPromptEmbedding(
                    name=request.name,
                    prompt=request.prompt,
                    prompt_embeds=prompt_embeds.to("cpu"),
                    prompt_embeds_mask=tensor_to_cpu(prompt_embeds_mask),
                    negative_prompt_embeds=tensor_to_cpu(negative_prompt_embeds),
                    negative_prompt_embeds_mask=tensor_to_cpu(negative_prompt_embeds_mask),
                )
                _step(progress, f"encoded qwen prompt {request.name}")
        return embeddings, elapsed_ms(start, synchronized_time(torch))
    finally:
        _phase(progress, "release qwen prompt encoder")
        del pipeline, processor, text_encoder
        release_prompt_encoder_memory(torch)
        _step(progress, "released qwen prompt encoder")


def _condition_images(
    pipeline: Any,
    reference_images: Sequence[Image.Image],
    *,
    condition_image_size: int,
    calculate_dimensions: Any,
) -> list[Image.Image]:
    condition_images = []
    for image in reference_images:
        condition_width, condition_height = calculate_dimensions(condition_image_size, image.width / image.height)
        condition_images.append(pipeline.image_processor.resize(image, condition_height, condition_width))
    return condition_images


def _prompt_mask_or_ones(torch: Any, prompt_embeds: Any, prompt_embeds_mask: Any | None) -> Any:
    if prompt_embeds_mask is not None:
        return prompt_embeds_mask
    return torch.ones(
        prompt_embeds.shape[:2],
        dtype=torch.long,
        device=prompt_embeds.device,
    )


def _phase(progress: StatusReporter | None, message: str) -> None:
    if progress is not None:
        progress.phase(message)


def _step(progress: StatusReporter | None, message: str) -> None:
    if progress is not None:
        progress.step(message)


def _reset_peak_memory_stats(torch: Any, device: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _suppress_known_prompt_encoder_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization",
        category=UserWarning,
        module=r"bitsandbytes\.autograd\._functions",
    )


def _load_qwen_text_encoder(
    text_encoder_class: Any,
    model: str,
    *,
    torch_dtype: Any,
    torch_device: Any,
    attention_impl: str,
) -> Any:
    return text_encoder_class.from_pretrained(
        model,
        subfolder=QWEN_IMAGE_EDIT_PROMPT_ENCODER_SUBFOLDER,
        torch_dtype=torch_dtype,
        attn_implementation=attention_impl,
        device_map={"": torch_device.index if torch_device.index is not None else 0},
        local_files_only=True,
    )


def _load_qwen_prompt_encoding_dependencies() -> tuple[Any, Any, Any, Any, int, Any]:
    try:
        import torch
        from diffusers import QwenImageEditPlusPipeline
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
            CONDITION_IMAGE_SIZE,
            calculate_dimensions,
        )
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor
    except ImportError as exc:
        raise QwenPromptEncodingDependencyError(
            "Qwen prompt encoding requires `pip install -e .[generation]`"
        ) from exc
    return (
        torch,
        QwenImageEditPlusPipeline,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLProcessor,
        CONDITION_IMAGE_SIZE,
        calculate_dimensions,
    )
