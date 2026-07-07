from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal


class TextLlmError(RuntimeError):
    pass


@dataclass(frozen=True)
class TextLlmConfig:
    parser_id: str
    endpoint: str
    model: str
    server_family: Literal["vllm", "llama.cpp", "openai-compatible"]
    api_key_env: str
    timeout_seconds: float
    max_new_tokens: int
    temperature: float
    structured_output: Literal["json_object", "json_schema", "none"]
    enable_thinking: bool


def text_llm_config_json(config: TextLlmConfig) -> dict[str, Any]:
    return {
        "id": config.parser_id,
        "endpoint": config.endpoint,
        "model": config.model,
        "server_family": config.server_family,
        "api_key_env": config.api_key_env,
        "timeout_seconds": config.timeout_seconds,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "structured_output": config.structured_output,
        "enable_thinking": config.enable_thinking,
    }


class OpenAICompatibleTextLlm:
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
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
        }
        if self.config.temperature <= 0.0:
            body["stream"] = False
        if self.config.structured_output == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif self.config.structured_output == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            }
        if self.config.server_family == "vllm" and not self.config.enable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}

        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = _api_key(self.config.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            self.config.endpoint,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                response_payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise TextLlmError(
                f"Text instruction parser request failed with HTTP {error.code}: {details}"
            ) from error
        except urllib.error.URLError as error:
            raise TextLlmError(f"Text instruction parser endpoint is unavailable: {error.reason}") from error
        except TimeoutError as error:
            raise TextLlmError("Text instruction parser request timed out") from error

        try:
            data = json.loads(response_payload)
        except json.JSONDecodeError as error:
            raise TextLlmError(f"Text instruction parser returned non-JSON API output: {error}") from error
        return _completion_text(data)


def _completion_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TextLlmError("Text instruction parser response has no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise TextLlmError("Text instruction parser response choice is not an object")
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    text = choice.get("text")
    if isinstance(text, str) and text.strip():
        return text
    raise TextLlmError("Text instruction parser response has no message content")


def _api_key(env_name: str) -> str | None:
    if not env_name:
        return None
    value = os.environ.get(env_name)
    if value is None or not value.strip():
        return None
    return value.strip()
