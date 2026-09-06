from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from aigen import sam_commands
from aigen.generation.image_upscale import IllustrationUpscaler
from aigen.generation.image_edit import _image_aspect_ratio
from aigen.generation.flux2_klein import _load_reference_images
from aigen.generation.vosr_backend import _prepare_vosr_file, VosrBackendError
from aigen.image_io import image_alpha, open_image, oriented_image_size
from aigen.keyframe_segmentation import _load_rgb
from aigen.progress import SILENT_STATUS
from aigen.sam_prompt_canvas import SAMPromptCanvas


class ImageIOTests(unittest.TestCase):
    def test_sam_canvas_manual_and_automatic_inputs_share_orientation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "oriented.jpg"
            exif = Image.Exif()
            exif[274] = 6
            pixels = np.arange(64 * 32 * 3, dtype=np.uint8).reshape((32, 64, 3))
            Image.fromarray(pixels).save(path, exif=exif)
            displayed = SAMPromptCanvas._load_image(path)
            np.testing.assert_array_equal(_load_rgb(path), displayed)
            self.assertEqual(displayed.shape[:2], (64, 32))
            self.assertEqual(_image_aspect_ratio(path), (1, 2))
            with _load_reference_images((path,))[0] as reference:
                np.testing.assert_array_equal(np.asarray(reference), displayed)

            def segment(segmenter, image, input_path, **kwargs):
                np.testing.assert_array_equal(image, displayed)
                self.assertEqual(kwargs["positive_points"], ((16, 40),))
                return np.ones(image.shape[:2], dtype=bool)

            with patch.object(sam_commands, "_create_segmenter", return_value=SimpleNamespace(close=lambda: None)), patch.object(sam_commands, "_segment", segment):
                image, mask = sam_commands._build_mask(
                    path, engine="sam2", device="cpu", prompt_mode="points", mask_candidate=None,
                    box=None, positive_points=((16, 40),), negative_points=(), threshold=28,
                    grow=0, feather=0, fill_holes=False, largest_component=False, invert=False, progress=SILENT_STATUS,
                )
                self.assertEqual(image.size, (32, 64))
                self.assertEqual(mask.shape, (64, 32))
                image.close()
            with Image.open(path) as source:
                self.assertEqual(oriented_image_size(source), (32, 64))
            with open_image(path) as source:
                self.assertIsNone(source.getexif().get(274))

    def test_upscalers_preserve_rgba_palette_and_color_key_alpha(self):
        rgba = Image.new("RGBA", (16, 16), (100, 50, 20, 0))
        rgba.paste((100, 50, 20, 255), (4, 4, 12, 12))
        palette = Image.new("P", (16, 16), 0)
        palette.putpalette([100, 50, 20, 20, 50, 100] + [0] * (768 - 6))
        palette.paste(1, (4, 4, 12, 12))
        palette.info["transparency"] = 0
        color_key = Image.new("RGB", (16, 16), (100, 50, 20))
        color_key.paste((20, 50, 100), (4, 4, 12, 12))
        color_key.info["transparency"] = (100, 50, 20)
        upscaler = object.__new__(IllustrationUpscaler)
        upscaler.torch, upscaler.model = torch, Mock()
        upscaler.model_path = Path("cpu-double.pth")
        upscaler.scale, upscaler.tile_size, upscaler.tile_overlap = 2, 512, 32
        with TemporaryDirectory() as directory, patch("aigen.generation.image_upscale._upscale_tensor_tiled", side_effect=lambda model, tensor, **kw: torch.nn.functional.interpolate(tensor, scale_factor=2)):
            for index, source in enumerate((rgba, palette, color_key)):
                with self.subTest(mode=source.mode):
                    path = Path(directory) / f"input{index}.png"
                    source.save(path)
                    with image_alpha(source) as alpha:
                        expected = alpha.resize((32, 32), Image.Resampling.LANCZOS)
                    result = upscaler._upscale_on_device(source, target_size=(32, 32), device="cpu", progress=SILENT_STATUS)
                    self.assertEqual(result.image.mode, "RGBA")
                    np.testing.assert_array_equal(np.asarray(result.image.getchannel("A")), np.asarray(expected))
                    result.image.close()
                    item = _prepare_vosr_file(path, Path(directory) / "output.png", 2, None)
                    self.assertEqual(item.alpha.getextrema(), (0, 255))
                    self.assertEqual(item.alpha.tobytes(), source.convert("RGBA").getchannel("A").tobytes())
                    item.image.close()
                    item.alpha.close()
                    with self.assertRaisesRegex(VosrBackendError, "JPEG cannot preserve alpha"):
                        _prepare_vosr_file(path, Path(directory) / "output.jpg", 2, None)
                    expected.close()
                    source.close()

    def test_file_upscale_uses_oriented_canvas_and_loads_model_once(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "oriented.jpg"
            output = Path(directory) / "output.png"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (64, 32)).save(path, exif=exif)
            upscaler = object.__new__(IllustrationUpscaler)
            upscaler.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            upscaler.model = Mock()
            upscaler.scale, upscaler.tile_size, upscaler.tile_overlap = 2, 512, 32

            def upscale(image, *, target_size, **kwargs):
                self.assertEqual(image.size, (32, 64))
                self.assertEqual(target_size, (64, 128))
                return SimpleNamespace(image=image.resize(target_size), elapsed_ms=0, model_name="cpu", model_path=Path("cpu"),
                                       scale=2, device="cpu", source_width=32, source_height=64, natural_width=64, natural_height=128,
                                       target_width=64, target_height=128)

            upscaler._upscale_on_device = upscale
            upscaler.upscale_files(((path, output),), long_side=128, progress=SILENT_STATUS)
            with Image.open(output) as image:
                self.assertEqual(image.size, (64, 128))
                self.assertIsNone(image.getexif().get(274))
            self.assertEqual(upscaler.model.to.call_count, 2)  # Enter device, then release.


if __name__ == "__main__":
    unittest.main()
