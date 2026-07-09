from __future__ import annotations

import unittest
from types import FunctionType

from PIL import Image

from aigen.generation.qwen_image_edit_identity import (
    QwenIdentityCase,
    _case_canvas,
    _qwen_edit_plus_output_vae_dimensions,
)


VAE_IMAGE_SIZE = 1024 * 1024


def calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    return target_area, round(ratio * 1000)


class FakeQwenEditPlusPipeline:
    def __call__(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            calculate_dimensions(VAE_IMAGE_SIZE, 0.5),
            calculate_dimensions(384 * 384, 0.5),
        )


def _decorated_pipeline_call(self) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        calculate_dimensions(VAE_IMAGE_SIZE, 0.5),
        calculate_dimensions(384 * 384, 0.5),
    )


def _pipeline_call_wrapper(self):
    return _wrapped_call.__wrapped__(self)


_wrapper_globals: dict[str, object] = {"__name__": "fake_torch_contextlib"}
_wrapped_call = FunctionType(_pipeline_call_wrapper.__code__, _wrapper_globals)
_wrapped_call.__wrapped__ = _decorated_pipeline_call
_wrapper_globals["_wrapped_call"] = _wrapped_call


class DecoratedFakeQwenEditPlusPipeline:
    __call__ = _wrapped_call


class QwenImageEditIdentityCanvasTests(unittest.TestCase):
    def test_portrait_canvas_preserves_anchor_aspect(self) -> None:
        case = QwenIdentityCase(
            name="portrait",
            references=("reference-a",),
            prompt="",
            portrait_canvas=True,
        )
        anchor = Image.new("RGB", (416, 640))

        self.assertEqual((416, 640), _case_canvas(case, anchor_image=anchor, max_side=640))

    def test_output_vae_dimensions_are_scoped_to_pipeline_call(self) -> None:
        pipeline = FakeQwenEditPlusPipeline()
        original_calculate_dimensions = calculate_dimensions

        with _qwen_edit_plus_output_vae_dimensions(pipeline, width=416, height=640):
            self.assertEqual(((416, 640), (384 * 384, 500)), pipeline())

        self.assertIs(original_calculate_dimensions, calculate_dimensions)
        self.assertEqual((VAE_IMAGE_SIZE, 500), calculate_dimensions(VAE_IMAGE_SIZE, 0.5))

    def test_output_vae_dimensions_find_decorated_pipeline_call(self) -> None:
        pipeline = DecoratedFakeQwenEditPlusPipeline()

        with _qwen_edit_plus_output_vae_dimensions(pipeline, width=416, height=640):
            self.assertEqual(((416, 640), (384 * 384, 500)), pipeline())


if __name__ == "__main__":
    unittest.main()
