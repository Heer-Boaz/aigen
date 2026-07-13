from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path

import torch

from aigen.runtime_profiles import MODELS_ROOT


QWEN25_VL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
QWEN25_VL_MODEL_ROOT = MODELS_ROOT / "vlm/Qwen/Qwen2.5-VL-7B-Instruct"
_CAPTION_INSTRUCTION = (
    "Describe this image accurately and in detail, including the camera angle and pose."
)
_REFERENCE_CAPTION_INSTRUCTION = (
    "Describe the target image accurately and in detail, including the camera angle and pose."
)


def caption_image(
    image_path: Path,
    *,
    reference_images: Sequence[Path] = (),
) -> str:
    from qwen_vl_utils import process_vision_info
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    processor = AutoProcessor.from_pretrained(
        QWEN25_VL_MODEL_ROOT,
        local_files_only=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN25_VL_MODEL_ROOT,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        local_files_only=True,
    )
    model.eval()
    try:
        if reference_images:
            content = [{"type": "text", "text": "Appearance reference pack:"}]
            content.extend(
                {
                    "type": "image",
                    "image": reference_path.resolve().as_posix(),
                }
                for reference_path in reference_images
            )
            content.extend(
                [
                    {"type": "text", "text": "Target image:"},
                    {
                        "type": "image",
                        "image": image_path.resolve().as_posix(),
                    },
                    {"type": "text", "text": _REFERENCE_CAPTION_INSTRUCTION},
                ]
            )
        else:
            content = [
                {
                    "type": "image",
                    "image": image_path.resolve().as_posix(),
                },
                {"type": "text", "text": _CAPTION_INSTRUCTION},
            ]
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(next(model.parameters()).device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        caption = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return " ".join(caption.split()).strip().strip("`\"'")
    finally:
        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()
