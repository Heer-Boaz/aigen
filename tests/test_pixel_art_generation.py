from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from aigen.cli import PIXEL_ART_PROFILES, main
from aigen.generation import pixel_art
from aigen.generation.pixel_art import (
    PixelArtBackendError,
    PixelArtError,
    PixelArtResult,
    run_pixel_art,
)


def write_input_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 64, 32)).save(path)


class FakeBackend:
    """Stands in for a loaded CycleGAN generator."""

    def __init__(self, out_size: tuple[int, int] | None = None) -> None:
        self._out_size = out_size
        self.seen_size: tuple[int, int] | None = None

    def pixelize(self, image: Image.Image) -> Image.Image:
        self.seen_size = image.size
        size = self._out_size or image.size
        return Image.new("RGB", size, color=(10, 20, 30))


class PixelArtGenerationTests(unittest.TestCase):
    def test_unknown_norm_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "in.png"
            write_input_image(src, (64, 64))
            with self.assertRaises(PixelArtError):
                run_pixel_art(
                    "models/pixel-art/cyclegan",
                    src,
                    Path(temp_dir) / "out.png",
                    norm="layer",
                )

    def test_preserves_native_resolution_img2img(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "in.png"
            output = Path(temp_dir) / "out" / "pixelized.png"
            write_input_image(src, (96, 64))  # non-square, not a multiple-of-4 trap
            with patch.object(pixel_art, "_load_backend", return_value=backend):
                result = run_pixel_art(
                    "models/pixel-art/cyclegan",
                    src,
                    output,
                )

            self.assertTrue(output.exists())
            self.assertIsInstance(result, PixelArtResult)
            self.assertEqual(result.backend, "cyclegan")
            self.assertEqual((result.width, result.height), (96, 64))
            self.assertEqual(backend.seen_size, (96, 64))
            with Image.open(output) as written:
                self.assertEqual(written.size, (96, 64))  # no resampling
            self.assertEqual(result.to_json()["width"], 96)

    def test_size_mismatch_is_rejected(self) -> None:
        backend = FakeBackend(out_size=(48, 32))  # backend resized -> not allowed
        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "in.png"
            write_input_image(src, (96, 64))
            with patch.object(pixel_art, "_load_backend", return_value=backend):
                with self.assertRaises(PixelArtBackendError):
                    run_pixel_art(
                        "models/pixel-art/cyclegan",
                        src,
                        Path(temp_dir) / "out.png",
                    )

    def test_missing_input_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(PixelArtError):
                run_pixel_art(
                    "models/pixel-art/cyclegan",
                    Path(temp_dir) / "does_not_exist.png",
                    Path(temp_dir) / "out.png",
                )

    def test_cli_pixel_art_dispatch(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "in.png"
            output = Path(temp_dir) / "cli_out.png"
            write_input_image(src, (80, 80))
            stdout = io.StringIO()
            with patch.object(pixel_art, "_load_backend", return_value=backend):
                with patch.object(sys, "stdout", stdout):
                    code = main(
                        [
                            "generate",
                            "pixel-art",
                            "--input-image",
                            src.as_posix(),
                            "--output",
                            output.as_posix(),
                        ]
                    )

            self.assertEqual(code, 0)
            self.assertTrue(output.exists())
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["backend"], PIXEL_ART_PROFILES["local"].backend)
            self.assertEqual(payload["norm"], PIXEL_ART_PROFILES["local"].norm)
            self.assertEqual(payload["width"], 80)


if __name__ == "__main__":
    unittest.main()
