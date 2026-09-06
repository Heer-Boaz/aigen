from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from aigen.generation.qwen_image_edit_masked import (
    composite_qwen_masked_output, prepare_qwen_masked_canvas, qwen_masked_canvas_size,
)
from aigen.keyframe_image_ops import exact_outside_mask_diff


class QwenMaskedCanvasTests(unittest.TestCase):
    def test_native_and_capped_canvas_preserves_original_rgba_pixels_outside_mask(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.random.default_rng(11).integers(0, 256, (67, 101, 4), dtype=np.uint8)
            source = root / "source.png"
            Image.fromarray(pixels).save(source)
            mask_pixels = np.zeros((67, 101), dtype=np.uint8)
            mask_pixels[20:50, 30:70] = 255
            mask = root / "mask.png"
            Image.fromarray(mask_pixels).save(mask)
            for cap in (None, 64):
                with self.subTest(cap=cap):
                    size = qwen_masked_canvas_size((101, 67), cap)
                    self.assertEqual(size, (112, 80) if cap is None else (64, 48))
                    canvas = prepare_qwen_masked_canvas(source, mask, size, root / str(cap))
                    decoded = root / f"decoded-{cap}.png"
                    Image.new("RGB", size, "red").save(decoded)
                    output = root / f"output-{cap}.png"
                    result = composite_qwen_masked_output(source, mask, decoded, canvas, output)
                    self.assertTrue(result["passed"])
                    with Image.open(output) as image:
                        actual = np.asarray(image)
                        self.assertEqual(image.size, (101, 67))
                        np.testing.assert_array_equal(actual[mask_pixels == 0], pixels[mask_pixels == 0])
                        np.testing.assert_array_equal(actual[..., 3], pixels[..., 3])
                        self.assertFalse(np.array_equal(actual[mask_pixels > 0], pixels[mask_pixels > 0]))

    def test_alpha_only_corruption_fails_exact_pixel_check(self):
        with Image.new("RGBA", (8, 8), (10, 20, 30, 255)) as source:
            changed = source.copy()
            changed.putpixel((1, 1), (10, 20, 30, 0))
            with Image.new("L", source.size, 0) as mask:
                result = exact_outside_mask_diff(source, changed, mask)
            self.assertFalse(result["passed"])
            self.assertEqual(result["outside_mask_changed_pixels"], 1)


if __name__ == "__main__":
    unittest.main()
