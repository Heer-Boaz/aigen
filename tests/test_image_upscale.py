from __future__ import annotations

import unittest

from aigen.generation.image_upscale import _tile_positions, _tile_windows


class ImageUpscaleTests(unittest.TestCase):
    def test_tile_positions_match_comfy_edge_tiles(self) -> None:
        self.assertEqual((0,), _tile_positions(512, tile_size=512, overlap=32))
        self.assertEqual((0, 480), _tile_positions(640, tile_size=512, overlap=32))

    def test_tile_windows_use_small_edge_tiles_without_extra_full_tile(self) -> None:
        windows = _tile_windows(width=640, height=416, tile_size=512, overlap=32)

        self.assertEqual(((0, 0, 512, 416), (480, 0, 160, 416)), windows)


if __name__ == "__main__":
    unittest.main()
