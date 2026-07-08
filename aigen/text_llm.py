from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


class TextLlmError(RuntimeError):
    pass


@dataclass(frozen=True)
class TextLlmConfig:
    parser_id: str
    model: Path
    dtype: str
    quantization: Literal["bitsandbytes-8bit", "none"]
    max_new_tokens: int
    temperature: float
    enable_thinking: bool


def text_llm_config_json(config: TextLlmConfig) -> dict[str, Any]:
    return {
        "id": config.parser_id,
        "runtime": "local-transformers-subprocess",
        "model": config.model.as_posix(),
        "dtype": config.dtype,
        "quantization": config.quantization,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "enable_thinking": config.enable_thinking,
    }


class OneShotLocalTextLlm:
    def __init__(self, config: TextLlmConfig) -> None:
        self.config = config

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> str:
        request = {
            "config": text_llm_config_json(self.config),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema_name": schema_name,
            "schema": schema,
        }
        completed = subprocess.run(
            [sys.executable, "-m", "aigen.text_llm_local"],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise TextLlmError(f"Local instruction parser failed: {details}")
        response = completed.stdout.strip()
        if not response:
            raise TextLlmError("Local instruction parser produced empty output")
        return response


def text_llm_runner(config: TextLlmConfig) -> OneShotLocalTextLlm:
    return OneShotLocalTextLlm(config)
