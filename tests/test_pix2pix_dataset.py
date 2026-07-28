from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from aigen.pix2pix.dataset import DATASET_FORMAT, audit_dataset
from aigen.pix2pix.errors import Pix2PixError


class Pix2PixDatasetTests(unittest.TestCase):
    def test_audit_validates_and_fingerprints_aligned_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_dataset(root)

            first = audit_dataset(root)
            second = audit_dataset(root)

            self.assertEqual(first.split_counts["train"], 2)
            self.assertEqual(first.split_counts["validation"], 1)
            self.assertEqual(first.split_group_counts["train"], 2)
            self.assertEqual(first.split_group_counts["validation"], 1)
            self.assertEqual(first.fingerprint, second.fingerprint)
            self.assertEqual([pair.id for pair in first.split("train")], ["train-a", "train-b"])

    def test_audit_rejects_non_rgb_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_dataset(root)
            Image.new("RGBA", (256, 256), (0, 0, 0, 0)).save(
                root / "target" / "validation.png"
            )

            with self.assertRaisesRegex(Pix2PixError, "must be RGB"):
                audit_dataset(root)

    def test_audit_accepts_native_128_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_dataset(root, image_size=128)

            dataset = audit_dataset(root)

            self.assertEqual(dataset.image_size, 128)

    def test_audit_rejects_group_leakage_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_dataset(root)
            pair_manifest = root / "pairs.jsonl"
            records = [
                json.loads(line)
                for line in pair_manifest.read_text(encoding="utf-8").splitlines()
            ]
            records[-1]["group"] = records[0]["group"]
            pair_manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(Pix2PixError, "crosses dataset splits"):
                audit_dataset(root)


def _write_dataset(root: Path, *, image_size: int = 256) -> None:
    (root / "source").mkdir()
    (root / "target").mkdir()
    records = []
    for pair_id, split, value in (
        ("train-a", "train", 32),
        ("train-b", "train", 96),
        ("validation", "validation", 160),
    ):
        source = f"source/{pair_id}.png"
        target = f"target/{pair_id}.png"
        Image.new("RGB", (image_size, image_size), (value, value, value)).save(
            root / source
        )
        Image.new("RGB", (image_size, image_size), (255 - value, value, 0)).save(
            root / target
        )
        records.append(
            {
                "id": pair_id,
                "group": f"subject-{pair_id}",
                "split": split,
                "source": source,
                "target": target,
            }
        )
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "format": DATASET_FORMAT,
                "name": "generic-test",
                "image_size": image_size,
                "pairs": "pairs.jsonl",
            }
        ),
        encoding="utf-8",
    )
    (root / "pairs.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
