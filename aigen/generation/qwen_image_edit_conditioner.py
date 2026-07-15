from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from diffusers.image_processor import VaeImageProcessor
from transformers import (
    FineGrainedFP8Config,
    Qwen2Tokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLProcessor,
)
from transformers.utils import logging as transformers_logging

from lightx2v.models.input_encoders.hf.qwen25.qwen25_vlforconditionalgeneration import (
    Qwen25_VLForConditionalGeneration_TextEncoder,
)
from lightx2v_platform.base.global_var import AI_DEVICE


class QwenImageEditFp8Conditioner(Qwen25_VLForConditionalGeneration_TextEncoder):
    def __init__(self, config: Any, model_path: Path) -> None:
        self.model_path = model_path
        super().__init__(config)

    def load(self) -> None:
        transformers_logging.disable_progress_bar()
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=AI_DEVICE,
            low_cpu_mem_usage=True,
            quantization_config=FineGrainedFP8Config(
                modules_to_not_convert=["visual", "lm_head"]
            ),
            local_files_only=True,
        )

        tokenizer_path = self.config.get(
            "qwen25vl_tokenizer_path",
            Path(self.config["model_path"]) / "tokenizer",
        )
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
        )
        if self.config["task"] == "i2i":
            self.image_processor = VaeImageProcessor(
                vae_scale_factor=self.config["vae_scale_factor"] * 2
            )
            processor_path = self.config.get(
                "qwen25vl_processor_path",
                Path(self.config["model_path"]) / "processor",
            )
            self.processor = Qwen2VLProcessor.from_pretrained(
                processor_path,
                local_files_only=True,
            )
