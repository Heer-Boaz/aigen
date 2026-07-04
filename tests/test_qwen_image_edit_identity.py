from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from aigen.cli import build_parser
from aigen.generation.qwen_image_edit_identity import (
    DEFAULT_QWEN_IDENTITY_PROFILE,
    QwenImageEditIdentityError,
    parse_qwen_identity_reference_args,
    qwen_image_edit_identity_profile_for_name,
    run_qwen_image_edit_identity,
)
from aigen.progress import SILENT_STATUS


def write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def empty_cache() -> None:
        return None

    @staticmethod
    def reset_peak_memory_stats(_device: str) -> None:
        return None


class FakeGenerator:
    def __init__(self, *, device: str) -> None:
        self.device = device
        self.seed = 0

    def manual_seed(self, seed: int) -> FakeGenerator:
        self.seed = seed
        return self


class FakeTorch:
    cuda = FakeCuda()
    Generator = FakeGenerator

    @staticmethod
    def inference_mode() -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


class FakeQwenIdentitySession:
    last: FakeQwenIdentitySession

    def __init__(self, profile, *, nunchaku_blocks_on_gpu: int | None) -> None:
        type(self).last = self
        self.profile = profile
        self.nunchaku_blocks_on_gpu = nunchaku_blocks_on_gpu
        self.torch = FakeTorch()
        self.model_load_ms = 9.5
        self.generated: list[dict[str, object]] = []
        self.closed = False

    def generate(
        self,
        *,
        reference_images,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        true_cfg_scale: float,
        guidance_scale: float,
        seed: int,
        max_sequence_length: int,
    ):
        self.generated.append(
            {
                "reference_sizes": [image.size for image in reference_images],
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "true_cfg_scale": true_cfg_scale,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "max_sequence_length": max_sequence_length,
            }
        )
        return Image.new("RGB", (width, height), (40 + len(self.generated), 80, 120)), {"pipeline_ms": 1.25}

    def environment(self) -> dict[str, object]:
        return {"profile": self.profile.name}

    def close(self) -> None:
        self.closed = True


class FakeMemorySampler:
    def __init__(self, preflight: dict[str, int]) -> None:
        self.preflight = preflight
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> dict[str, int]:
        return {**self.preflight, "nvidia_smi_peak_used_mb": 2048}


class QwenImageEditIdentityTests(unittest.TestCase):
    def test_character_cli_exposes_qwen_identity_run(self) -> None:
        args = build_parser().parse_args(
            [
                "characters",
                "qwen-identity-run",
                "--reference",
                "front=front.png",
                "--reference",
                "portrait=portrait.png",
                "--reference",
                "side=side.png",
                "--reference",
                "back=back.png",
                "--case",
                "right-profile",
                "--output-dir",
                "runs/qwen",
            ]
        )

        self.assertEqual(args.characters_command, "qwen-identity-run")
        self.assertEqual(args.profile, DEFAULT_QWEN_IDENTITY_PROFILE)
        self.assertEqual(args.reference[0], "front=front.png")
        self.assertEqual(args.case, ["right-profile"])

    def test_parse_reference_args_resolves_named_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            front = root / "front.png"
            write_image(front, (64, 96), (220, 60, 60))

            refs = parse_qwen_identity_reference_args(["front=front.png"], root)

            self.assertEqual(refs, {"front": front.resolve()})

    def test_parse_reference_args_rejects_unknown_ref_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "hat.png"
            write_image(image, (64, 96), (220, 60, 60))

            with self.assertRaises(QwenImageEditIdentityError):
                parse_qwen_identity_reference_args(["hat=hat.png"], root)

    def test_run_qwen_identity_smoke_writes_outputs_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            references = {
                "front": root / "front.png",
                "portrait": root / "portrait.png",
                "side": root / "side.png",
                "back": root / "back.png",
            }
            write_image(references["front"], (220, 420), (210, 60, 60))
            write_image(references["portrait"], (320, 320), (60, 180, 80))
            write_image(references["side"], (240, 420), (70, 80, 210))
            write_image(references["back"], (240, 420), (200, 180, 70))
            output_dir = root / "run"

            with (
                patch(
                    "aigen.generation.qwen_image_edit_identity.nvidia_smi_preflight_limit",
                    return_value={
                        "nvidia_smi_preflight_used_mb": 100,
                        "nvidia_smi_device_total_mb": 16000,
                        "nvidia_smi_preflight_utilization_gpu": 0,
                    },
                ),
                patch("aigen.generation.qwen_image_edit_identity.NvidiaSmiMemorySampler", FakeMemorySampler),
                patch("aigen.generation.qwen_image_edit_identity.QwenImageEditIdentitySession", FakeQwenIdentitySession),
            ):
                result = run_qwen_image_edit_identity(
                    references=references,
                    output_dir=output_dir,
                    profile=qwen_image_edit_identity_profile_for_name(DEFAULT_QWEN_IDENTITY_PROFILE),
                    cases=("front", "portrait"),
                    max_side=128,
                    steps=None,
                    true_cfg_scale=None,
                    guidance_scale=None,
                    seed=11,
                    max_sequence_length=256,
                    overwrite=False,
                    nunchaku_blocks_on_gpu=None,
                    progress=SILENT_STATUS,
                )

            session = FakeQwenIdentitySession.last
            self.assertTrue(session.closed)
            self.assertIsNone(session.nunchaku_blocks_on_gpu)
            self.assertEqual(len(session.generated), 2)
            self.assertEqual(session.generated[0]["width"], 80)
            self.assertEqual(session.generated[0]["height"], 128)
            self.assertEqual(session.generated[1]["width"], 128)
            self.assertEqual(session.generated[1]["height"], 128)
            self.assertEqual(session.generated[0]["steps"], 4)
            self.assertEqual(session.generated[0]["true_cfg_scale"], 1.0)
            self.assertLessEqual(max(session.generated[0]["reference_sizes"][0]), 128)
            self.assertTrue((output_dir / "images" / "front.png").exists())
            self.assertTrue((output_dir / "images" / "portrait.png").exists())
            self.assertTrue((output_dir / "contact_sheet.png").exists())
            self.assertTrue((output_dir / "result.json").exists())
            self.assertEqual(result["memory"]["nvidia_smi_peak_used_mb"], 2048)
            self.assertEqual(result["outputs"][0]["references"], ["front", "portrait", "side"])

            manifest = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["profile"]["load_strategy"], "nunchaku-qwen-image-edit-2509")
            self.assertEqual(manifest["generation"]["max_references_per_case"], 3)
            self.assertEqual(manifest["generation"]["steps"], 4)


if __name__ == "__main__":
    unittest.main()
