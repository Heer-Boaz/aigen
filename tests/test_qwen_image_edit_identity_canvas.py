from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FunctionType

from PIL import Image

from aigen.generation.qwen_image_edit_identity import (
    QwenIdentityCase,
    _case_canvas,
    _postprocess_qwen_identity_outputs,
    _qwen_output_spec,
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


class NullProgress:
    @property
    def renders_live(self) -> bool:
        return False

    def __enter__(self) -> "NullProgress":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None

    def begin(self, total: int, phase: str) -> None:
        return None

    def phase(self, text: str) -> None:
        return None

    def step(self, text: str) -> None:
        return None

    def finish(self, status: str) -> None:
        return None


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

    def test_explicit_output_spec_owns_canvas(self) -> None:
        case = QwenIdentityCase(
            name="portrait",
            references=("reference-a",),
            prompt="",
            portrait_canvas=True,
        )
        anchor = Image.new("RGB", (416, 640))
        output_spec = _qwen_output_spec(output_format="16:9", resolution="1k", max_side=640)

        self.assertIsNotNone(output_spec)
        self.assertEqual((640, 352), _case_canvas(case, anchor_image=anchor, max_side=640, output_spec=output_spec))
        self.assertEqual((1024, 576), (output_spec.final_width, output_spec.final_height))

    def test_output_spec_requires_format_and_resolution(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "provided together"):
            _qwen_output_spec(output_format="2:3", resolution=None, max_side=640)

    def test_raw_copy_postprocess_writes_final_to_run_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            raw_dir = output_dir / "raw"
            raw_dir.mkdir()
            raw_image_path = raw_dir / "portrait.png"
            Image.new("RGB", (416, 640)).save(raw_image_path)

            postprocess = _postprocess_qwen_identity_outputs(
                raw_outputs=[
                    {
                        "name": "portrait",
                        "raw_width": 416,
                        "raw_height": 640,
                        "raw_image": {"path": raw_image_path.as_posix()},
                    }
                ],
                output_dir=output_dir,
                output_spec=None,
                progress=NullProgress(),
            )

            self.assertTrue((output_dir / "portrait.png").exists())
            self.assertFalse((output_dir / "images").exists())
            self.assertEqual((416, 640), (postprocess.outputs[0]["width"], postprocess.outputs[0]["height"]))
            self.assertEqual((output_dir / "portrait.png").as_posix(), postprocess.outputs[0]["image"]["path"])

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
