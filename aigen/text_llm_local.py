from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from aigen.generation.runtime_types import resolve_torch_dtype


class LocalTextLlmError(RuntimeError):
    pass


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        sys.stdout.write(generate_local_text(request))
    except LocalTextLlmError as error:
        sys.stderr.write(str(error))
        return 1
    return 0


def generate_local_text(request: dict[str, Any]) -> str:
    config = request["config"]
    model_path = Path(config["model"])
    if not model_path.exists():
        raise LocalTextLlmError(f"missing local instruction parser model: {model_path.as_posix()}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    dtype = resolve_torch_dtype(torch, config["dtype"], auto_value="auto")
    quantization_config = _quantization_config(config["quantization"])
    device_map = {"": 0} if torch.cuda.is_available() else None
    if config["quantization"] == "bitsandbytes-8bit" and device_map is None:
        raise LocalTextLlmError("bitsandbytes-8bit instruction parser requires CUDA")

    tokenizer = AutoTokenizer.from_pretrained(model_path.as_posix(), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path.as_posix(),
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        local_files_only=True,
    )
    if device_map is None:
        model = model.to("cpu")

    messages = [
        {"role": "system", "content": request["system_prompt"]},
        {"role": "user", "content": _user_prompt_with_schema(request)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=config["enable_thinking"],
    )
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)
    generate_kwargs: dict[str, Any] = {"max_new_tokens": config["max_new_tokens"]}
    if config["temperature"] > 0.0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = config["temperature"]
    else:
        generate_kwargs["do_sample"] = False
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def _user_prompt_with_schema(request: dict[str, Any]) -> str:
    schema_json = json.dumps(request["schema"], indent=2, sort_keys=True)
    return (
        f"{request['user_prompt']}\n\n"
        f"JSON schema name: {request['schema_name']}\n"
        f"JSON schema:\n{schema_json}\n"
    )


def _quantization_config(quantization: str) -> Any | None:
    if quantization == "none":
        return None
    if quantization != "bitsandbytes-8bit":
        raise LocalTextLlmError(f"Unknown instruction parser quantization: {quantization}")
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_8bit=True)


if __name__ == "__main__":
    raise SystemExit(main())
