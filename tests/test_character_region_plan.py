from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from aigen.character_region_plan import (
    CharacterRegionPlanError,
    CharacterRegionRequest,
    parse_character_region_args,
    plan_character_regions,
)
from aigen.cli import build_parser
from aigen.keyframe_grounding import GroundedRegionBox, GroundingConfig
from aigen.keyframe_segmentation import Sam2SegmentationConfig
from aigen.progress import SILENT_STATUS


def write_image(path: Path, size: tuple[int, int] = (96, 128)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, "white")
    pixels = image.load()
    for y in range(24, 104):
        for x in range(32, 64):
            pixels[x, y] = (120, 70, 40)
    image.save(path)


class FakeRegionGrounder:
    last: FakeRegionGrounder

    def __init__(self, config: GroundingConfig) -> None:
        type(self).last = self
        self.config = config
        self.requests = []

    def ground_regions(self, image: Image.Image, requests: list[object]) -> list[GroundedRegionBox]:
        self.image_size = image.size
        self.requests = requests
        return [
            GroundedRegionBox(
                box=(10 + index * 16, 20 + index * 16, 42 + index * 16, 60 + index * 16),
                label=request.prompt,
                score=0.9 - index * 0.1,
                source="florence2",
                prior_iou=0.0,
            )
            for index, request in enumerate(requests)
        ]


class FakeSam2Segmenter:
    last: FakeSam2Segmenter

    def __init__(self, config: Sam2SegmentationConfig) -> None:
        type(self).last = self
        self.config = config
        self.boxes = []
        self.closed = False

    def segment_image_boxes(self, image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
        self.image_shape = image.shape
        self.boxes = boxes
        masks = []
        for left, top, right, bottom in boxes:
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[top:bottom, left:right] = True
            masks.append(mask)
        return masks

    def close(self) -> None:
        self.closed = True


class CharacterRegionPlanTests(unittest.TestCase):
    def test_region_plan_cli_accepts_image_regions_and_models(self) -> None:
        args = build_parser().parse_args(
            [
                "characters",
                "region-plan",
                "--image",
                "candidate.png",
                "--region",
                "face=visible face",
                "--output-dir",
                "runs/regions",
                "--sam2-model",
                "aigen/models/segmentation/facebook/sam2.1-hiera-tiny",
            ]
        )

        self.assertEqual(args.characters_command, "region-plan")
        self.assertEqual(args.region, ["face=visible face"])
        self.assertEqual(args.sam2_model, Path("aigen/models/segmentation/facebook/sam2.1-hiera-tiny"))

    def test_parse_character_region_args_rejects_duplicate_or_unsafe_names(self) -> None:
        self.assertEqual(
            parse_character_region_args(["face=visible face"]),
            (CharacterRegionRequest(name="face", prompt="visible face"),),
        )
        with self.assertRaises(CharacterRegionPlanError):
            parse_character_region_args([])
        with self.assertRaises(CharacterRegionPlanError):
            parse_character_region_args(["face=visible face", "face=other face"])
        with self.assertRaises(CharacterRegionPlanError):
            parse_character_region_args(["bad/name=visible face"])

    def test_region_plan_writes_sam2_masks_overlays_and_debug_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "candidate.png"
            output_dir = root / "regions"
            write_image(image_path)

            with (
                patch("aigen.character_region_plan.KeyframeRegionGrounder", FakeRegionGrounder),
                patch("aigen.character_region_plan.Sam2RegionSegmenter", FakeSam2Segmenter),
            ):
                result = plan_character_regions(
                    image_path=image_path,
                    regions=parse_character_region_args(["face=visible face", "bow=neck bow"]),
                    output_dir=output_dir,
                    overwrite=False,
                    grounding_config=GroundingConfig(device="cpu"),
                    segmentation_config=Sam2SegmentationConfig(model=root / "sam2", device="cpu"),
                    progress=SILENT_STATUS,
                )

            self.assertEqual(result["kind"], "character-region-plan")
            self.assertEqual([region["name"] for region in result["regions"]], ["face", "bow"])
            self.assertEqual(FakeRegionGrounder.last.image_size, (96, 128))
            self.assertEqual([request.prior_box for request in FakeRegionGrounder.last.requests], [None, None])
            self.assertEqual(FakeSam2Segmenter.last.boxes, [(10, 20, 42, 60), (26, 36, 58, 76)])
            self.assertTrue(FakeSam2Segmenter.last.closed)
            self.assertTrue((output_dir / "masks" / "face.png").exists())
            self.assertTrue((output_dir / "overlays" / "face.png").exists())
            self.assertTrue((output_dir / "debug_sheet.png").exists())
            self.assertTrue((output_dir / "result.json").exists())
            self.assertEqual(result["regions"][0]["segmentation"]["method"], "florence2-box-to-sam2-mask")

    def test_region_plan_rejects_existing_output_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "candidate.png"
            output_dir = root / "regions"
            write_image(image_path)
            output_dir.mkdir()
            (output_dir / "old.txt").write_text("old", encoding="utf-8")

            with self.assertRaises(CharacterRegionPlanError):
                plan_character_regions(
                    image_path=image_path,
                    regions=parse_character_region_args(["face=visible face"]),
                    output_dir=output_dir,
                    overwrite=False,
                    grounding_config=GroundingConfig(device="cpu"),
                    segmentation_config=Sam2SegmentationConfig(model=root / "sam2", device="cpu"),
                    progress=SILENT_STATUS,
                )


if __name__ == "__main__":
    unittest.main()
