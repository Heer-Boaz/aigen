from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from aigen.text_llm import OneShotLocalTextLlm, TextLlmConfig, TextLlmError, text_llm_config_json
from aigen.text_llm_local import LocalTextLlmError, generate_local_text


def config() -> TextLlmConfig:
    return TextLlmConfig(
        parser_id="qwen3-8b-instruction-parser",
        model=Path("/models/qwen3"),
        dtype="bfloat16",
        quantization="bitsandbytes-8bit",
        max_new_tokens=700,
        temperature=0.0,
        enable_thinking=False,
    )


class TextLlmTests(unittest.TestCase):
    def test_one_shot_runner_sends_single_subprocess_request(self) -> None:
        calls = []

        def fake_run(*args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout='{"ok": true}', stderr="")

        with patch("aigen.text_llm.subprocess.run", fake_run):
            response = OneShotLocalTextLlm(config()).generate_json(
                system_prompt="system",
                user_prompt="user",
                schema_name="schema",
                schema={"type": "object"},
            )

        self.assertEqual(response, '{"ok": true}')
        self.assertEqual(calls[0][0][0][-2:], ["-m", "aigen.text_llm_local"])
        payload = json.loads(calls[0][1]["input"])
        self.assertEqual(payload["config"]["runtime"], "local-transformers-subprocess")
        self.assertEqual(payload["config"]["model"], "/models/qwen3")
        self.assertEqual(payload["system_prompt"], "system")
        self.assertEqual(payload["user_prompt"], "user")
        self.assertEqual(payload["schema_name"], "schema")
        self.assertEqual(payload["schema"], {"type": "object"})

    def test_one_shot_runner_fails_on_subprocess_error(self) -> None:
        def fake_run(*_args, **_kwargs):
            return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")

        with patch("aigen.text_llm.subprocess.run", fake_run):
            with self.assertRaisesRegex(TextLlmError, "boom"):
                OneShotLocalTextLlm(config()).generate_json(
                    system_prompt="system",
                    user_prompt="user",
                    schema_name="schema",
                    schema={},
                )

    def test_local_generator_fails_when_model_is_missing(self) -> None:
        request = {
            "config": text_llm_config_json(config()),
            "system_prompt": "system",
            "user_prompt": "user",
            "schema_name": "schema",
            "schema": {},
        }

        with self.assertRaisesRegex(LocalTextLlmError, "missing local instruction parser model"):
            generate_local_text(request)


if __name__ == "__main__":
    unittest.main()
