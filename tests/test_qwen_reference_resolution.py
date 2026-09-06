from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from aigen.generation.qwen_image_edit_conditioner import QwenImageEditFp8Conditioner
from aigen.generation.qwen_image_edit_identity import (
    QwenIdentityCase,
    _case_input_images,
    _load_reference_image,
    _prepare_qwen_identity_references,
    _stage_lightx2v_input_images,
)
from aigen.progress import StatusReporter


class QwenReferenceResolutionTests(unittest.TestCase):
    def test_reference_pixels_follow_display_orientation(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.jpg"
            exif = Image.Exif()
            exif[274] = 6
            with Image.new("RGB", (64, 32), "blue") as image:
                image.save(path, exif=exif)
            with _load_reference_image(path, max_side=None) as image:
                self.assertEqual(image.size, (32, 64))
                self.assertIsNone(image.getexif().get(274))

    def test_native_inputs_keep_dimensions_and_order_through_staging(self) -> None:
        sizes = ((2049, 1025), (113, 157), (1009, 529))
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / f"source_{index}.png" for index in range(3)]
            for path, size in zip(paths, sizes, strict=True):
                with Image.new("RGB", size, "blue") as image:
                    image.save(path)
            case = QwenIdentityCase(
                name="edit", source_images=("source",), references=("reference",),
                guides=("guide",), prompt="",
            )
            prepared = _prepare_qwen_identity_references(
                source_images={"source": paths[0]}, references={"reference": paths[1]},
                guides={"guide": paths[2]}, controls={}, selected_cases=(case,),
                max_side=640, input_max_side=None, native_canvas_pixels=None,
                aspect_ratio=None, canvas_size=(640, 480), progress=Mock(spec=StatusReporter),
            )
            inputs = _case_input_images(prepared, case)
            try:
                self.assertEqual(prepared.canvas_sizes[case.name], (640, 480))
                staged = _stage_lightx2v_input_images(prepared, Path(directory))
                for image, size in zip(inputs, sizes, strict=True):
                    self.assertEqual(image.size, size)
                    with Image.open(staged[id(image)]) as saved:
                        self.assertEqual(saved.size, size)
                        self.assertEqual(saved.tobytes(), image.tobytes())
            finally:
                for image in inputs:
                    image.close()

    def test_conditioner_preserves_vae_resolution_and_upstream_semantic_context(self) -> None:
        conditioner = object.__new__(QwenImageEditFp8Conditioner)
        conditioner.CONDITION_IMAGE_SIZE = 384 * 384
        conditioner.image_processor = VaeImageProcessor(vae_scale_factor=16)
        cases = (
            ((2049, 1025), (544, 256)),
            ((113, 157), (320, 448)),
            ((529, 1009), (288, 544)),
            ((1024, 1024), (384, 384)),
        )
        for size, expected_semantic_size in cases:
            with self.subTest(size=size), Image.new("RGB", size) as image:
                semantic, vae, semantic_hw, vae_hw = conditioner.preprocess_image(image)
                self.assertEqual(semantic.size, expected_semantic_size)
                self.assertEqual(semantic_hw, (semantic.height, semantic.width))
                self.assertEqual(vae.shape[:3], (1, 3, 1))
                self.assertEqual(vae_hw, tuple(vae.shape[-2:]))
                for original, aligned in zip((image.height, image.width), vae_hw, strict=True):
                    self.assertEqual(aligned % 16, 0)
                    self.assertLessEqual(aligned, original)
                    self.assertLess(original - aligned, 16)
                semantic.close()

    def test_explicit_legacy_input_limit_keeps_existing_fitting(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            with Image.new("RGB", (131, 77)) as image:
                image.save(path)
            with _load_reference_image(path, max_side=64) as image:
                self.assertEqual(image.size, (64, 48))


if __name__ == "__main__":
    unittest.main()
