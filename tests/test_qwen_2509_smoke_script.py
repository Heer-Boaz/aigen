from __future__ import annotations

import unittest

from tools.smoke_qwen_image_edit_2509_nunchaku import (
    DEFAULT_CHECKPOINT,
    LOW_VRAM_THRESHOLD_MB,
    _default_steps_for_checkpoint,
    _default_true_cfg_scale_for_checkpoint,
    _offload_enabled,
    build_parser,
)


class Qwen2509SmokeScriptTests(unittest.TestCase):
    def test_parser_defaults_to_exact_4step_checkpoint(self) -> None:
        args = build_parser().parse_args(
            [
                "--input-image",
                "reference.png",
                "--output-dir",
                "runs/qwen_2509_smoke",
            ]
        )

        self.assertEqual(args.checkpoint, DEFAULT_CHECKPOINT)
        self.assertEqual(args.max_side, 512)
        self.assertEqual(args.offload_mode, "auto")
        self.assertIsNone(args.steps)

    def test_step_defaults_follow_checkpoint_name(self) -> None:
        self.assertEqual(_default_steps_for_checkpoint(DEFAULT_CHECKPOINT), 4)
        self.assertEqual(
            _default_steps_for_checkpoint(DEFAULT_CHECKPOINT.with_name("svdq-fp4_r32-qwen-image-edit-2509.safetensors")),
            40,
        )

    def test_true_cfg_defaults_follow_checkpoint_kind(self) -> None:
        self.assertEqual(_default_true_cfg_scale_for_checkpoint(DEFAULT_CHECKPOINT), 1.0)
        self.assertEqual(
            _default_true_cfg_scale_for_checkpoint(
                DEFAULT_CHECKPOINT.with_name("svdq-fp4_r32-qwen-image-edit-2509.safetensors")
            ),
            4.0,
        )

    def test_auto_offload_enables_on_16gb_gpu(self) -> None:
        snapshot = {
            "nvidia_smi_device_total_mb": LOW_VRAM_THRESHOLD_MB,
            "nvidia_smi_used_mb": 100,
            "nvidia_smi_utilization_gpu": 0,
        }

        self.assertTrue(_offload_enabled("auto", snapshot))
        self.assertFalse(_offload_enabled("never", snapshot))
        self.assertTrue(_offload_enabled("always", snapshot))


if __name__ == "__main__":
    unittest.main()
