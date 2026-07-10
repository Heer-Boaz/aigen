from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.generation.runtime_diagnostics import module_device_report
from aigen.generation.runtime_types import resolve_torch_dtype


DEFAULT_JUDGE_ID = "qwen2.5-vl-7b"
DEFAULT_JUDGE_REPO_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_JUDGE_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
DEFAULT_JUDGE_QUANTIZATION = "bitsandbytes-8bit"
DEFAULT_MAX_PIXELS = 512 * 28 * 28
DEFAULT_MIN_PIXELS = 256 * 28 * 28


class QwenVlmError(RuntimeError):
    pass


@dataclass(frozen=True)
class QwenVlmConfig:
    judge_id: str
    model: Path
    repo_id: str
    revision: str
    dtype: str
    attention_impl: str
    quantization: str
    min_pixels: int
    max_pixels: int
    max_new_tokens: int
    temperature: float


class QwenVlm:
    def __init__(self, config: QwenVlmConfig) -> None:
        validate_local_qwen_model(config)

        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        dtype = _torch_dtype(torch, config.dtype)
        quantization_config = _quantization_config(torch, config)
        device_map = qwen_vlm_device_map(torch)
        try:
            processor = AutoProcessor.from_pretrained(
                config.model.as_posix(),
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                local_files_only=True,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.model.as_posix(),
                torch_dtype=dtype,
                attn_implementation=config.attention_impl,
                device_map=device_map,
                quantization_config=quantization_config,
                local_files_only=True,
            )
        except OSError as error:
            raise QwenVlmError(f"Failed to load local Qwen VLM from {config.model.as_posix()}: {error}") from error
        self.model = model
        self.processor = processor
        self.process_vision_info = process_vision_info
        self.config = config
        self.torch = torch
        self.device_report = module_device_report(self.model)

    def judge_candidate(self, prompt: str, image_paths: list[Path]) -> str:
        return self._generate(prompt, image_paths)

    def describe_image(self, prompt: str, image_paths: list[Path]) -> str:
        return self._generate(prompt, image_paths)

    def select_indices(
        self,
        prompt: str,
        image_paths: list[Path],
        choices: Sequence[tuple[int, ...]],
    ) -> tuple[int, ...]:
        if not choices:
            raise QwenVlmError("Qwen VLM index selection requires at least one valid choice")

        inputs = self._prepare_inputs(prompt, image_paths)
        prompt_length = inputs.input_ids.shape[1]
        tokenizer = self.processor.tokenizer
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise QwenVlmError("Qwen VLM tokenizer has no EOS token for constrained index selection")

        choice_by_tokens: dict[tuple[int, ...], tuple[int, ...]] = {}
        allowed_by_prefix: dict[tuple[int, ...], set[int]] = {}
        max_new_tokens = 0
        for choice in choices:
            choice_text = f"[{','.join(str(index) for index in choice)}]"
            choice_tokens = tuple(tokenizer.encode(choice_text, add_special_tokens=False))
            choice_by_tokens[choice_tokens] = choice
            constrained_tokens = choice_tokens + (eos_token_id,)
            max_new_tokens = max(max_new_tokens, len(constrained_tokens))
            for offset, token_id in enumerate(constrained_tokens):
                allowed_by_prefix.setdefault(constrained_tokens[:offset], set()).add(token_id)

        def allowed_tokens(_batch_id: int, input_ids: Any) -> list[int]:
            generated_prefix = tuple(input_ids[prompt_length:].tolist())
            try:
                return list(allowed_by_prefix[generated_prefix])
            except KeyError as error:
                raise QwenVlmError(
                    f"Qwen VLM produced an invalid constrained index prefix: {generated_prefix}"
                ) from error

        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=min(4, len(choices)),
                early_stopping=True,
                renormalize_logits=True,
                eos_token_id=eos_token_id,
                prefix_allowed_tokens_fn=allowed_tokens,
            )
        selected_tokens = tuple(generated_ids[0, prompt_length:].tolist())
        if selected_tokens and selected_tokens[-1] == eos_token_id:
            selected_tokens = selected_tokens[:-1]
        try:
            return choice_by_tokens[selected_tokens]
        except KeyError as error:
            raise QwenVlmError(
                f"Qwen VLM produced an invalid constrained index selection: {selected_tokens}"
            ) from error

    def close(self) -> None:
        del self.model
        del self.processor
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _generate(self, prompt: str, image_paths: list[Path]) -> str:
        inputs = self._prepare_inputs(prompt, image_paths)
        generate_kwargs: dict[str, Any] = {"max_new_tokens": self.config.max_new_tokens}
        if self.config.temperature > 0.0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.config.temperature
        else:
            generate_kwargs["do_sample"] = False
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
        trimmed_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        return self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _prepare_inputs(self, prompt: str, image_paths: list[Path]) -> Any:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": path.as_posix(),
                        "min_pixels": self.config.min_pixels,
                        "max_pixels": self.config.max_pixels,
                    }
                    for path in image_paths
                ]
                + [{"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(next(self.model.parameters()).device)


def qwen_vlm_config_json(config: QwenVlmConfig) -> dict[str, Any]:
    return {
        "id": config.judge_id,
        "model": config.model.resolve().as_posix(),
        "repo_id": config.repo_id,
        "revision": config.revision,
        "dtype": config.dtype,
        "attention_impl": config.attention_impl,
        "quantization": config.quantization,
        "min_pixels": config.min_pixels,
        "max_pixels": config.max_pixels,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
    }


def validate_local_qwen_model(config: QwenVlmConfig) -> None:
    if not config.model.exists():
        raise QwenVlmError(
            "Missing local Qwen VLM. Download "
            f"{config.repo_id} to {config.model.as_posix()} before running vision judging."
        )
    config_path = config.model / "config.json"
    if not config_path.exists():
        raise QwenVlmError(f"Local Qwen VLM is incomplete; missing {config_path.as_posix()}")
    if not any(config.model.glob("*.safetensors")):
        raise QwenVlmError(
            "Local Qwen VLM is incomplete; missing safetensors weights in "
            f"{config.model.as_posix()}"
        )


def qwen_vlm_device_map(torch: Any) -> dict[str, int | str]:
    if not torch.cuda.is_available():
        raise QwenVlmError("Qwen VLM requires CUDA; CPU inference is not a supported pipeline path")
    return {"": 0}


def _quantization_config(torch: Any, config: QwenVlmConfig) -> Any | None:
    if config.quantization == "none":
        return None
    if config.quantization != "bitsandbytes-8bit":
        raise QwenVlmError(f"Unknown Qwen VLM quantization: {config.quantization}")
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_8bit=True)


def _torch_dtype(torch: Any, dtype: str) -> Any:
    return resolve_torch_dtype(torch, dtype, auto_value="auto")
