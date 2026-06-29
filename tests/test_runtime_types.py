from __future__ import annotations

import unittest

from aigen.generation.runtime_types import resolve_torch_dtype


class FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"


class RuntimeTypesTests(unittest.TestCase):
    def test_resolves_supported_torch_dtypes(self) -> None:
        self.assertEqual(resolve_torch_dtype(FakeTorch, "bfloat16", auto_value=None), "bf16")
        self.assertEqual(resolve_torch_dtype(FakeTorch, "float16", auto_value=None), "fp16")
        self.assertEqual(resolve_torch_dtype(FakeTorch, "float32", auto_value=None), "fp32")

    def test_resolves_auto_to_owner_value(self) -> None:
        self.assertEqual(resolve_torch_dtype(FakeTorch, "auto", auto_value="auto"), "auto")
        self.assertIsNone(resolve_torch_dtype(FakeTorch, "auto", auto_value=None))

    def test_rejects_unknown_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported torch dtype"):
            resolve_torch_dtype(FakeTorch, "float8", auto_value=None)


if __name__ == "__main__":
    unittest.main()
