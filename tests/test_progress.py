from __future__ import annotations

import argparse
import io
import unittest
from unittest.mock import patch

from aigen.progress import RuntimeStatus, SILENT_STATUS, open_cli_progress
from aigen.system_telemetry import SystemTelemetry


class FakeTelemetrySampler:
    def sample(self) -> SystemTelemetry:
        return SystemTelemetry(
            cpu_percent=17.5,
            gpu_percent=42,
            vram_used_mb=1234,
            vram_total_mb=16303,
        )


class ProgressTests(unittest.TestCase):
    def test_terminal_status_renders_gpu_and_vram(self) -> None:
        stream = io.StringIO()

        status = RuntimeStatus.terminal(
            label="keyframes run",
            interval_seconds=60.0,
            stream=stream,
            telemetry=FakeTelemetrySampler(),
        )
        with status:
            status.step("denoise seed_060 (1/64)")
            status.finish("completed")

        output = stream.getvalue()
        self.assertIn("keyframes run", output)
        self.assertIn("cpu  17.5%", output)
        self.assertIn("gpu  42%", output)
        self.assertIn("vram 1234/16303 MB", output)
        self.assertIn("completed", output)
        self.assertGreaterEqual(output.count("\r"), 2)
        self.assertEqual(output.count("\n"), 1)
        self.assertTrue(output.endswith("\n"))

    def test_quiet_status_keeps_same_runtime_contract(self) -> None:
        with SILENT_STATUS:
            SILENT_STATUS.phase("load models")
            SILENT_STATUS.step("denoise")
            SILENT_STATUS.finish("completed")

    def test_cli_progress_can_be_disabled_at_runtime_boundary(self) -> None:
        args = argparse.Namespace(command="keyframes", keyframes_command="run")

        with patch.dict("os.environ", {"AIGEN_PROGRESS": "0"}):
            self.assertFalse(open_cli_progress(args).renders_live)

    def test_cli_progress_rejects_unknown_command_contract(self) -> None:
        args = argparse.Namespace(command="nonsense")

        with patch.dict("os.environ", {"AIGEN_PROGRESS": "0"}):
            with self.assertRaisesRegex(RuntimeError, "unsupported command"):
                open_cli_progress(args)


if __name__ == "__main__":
    unittest.main()
