import asyncio
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image
from textual.widgets import Button

from aigen import image_tui
from aigen.tui_process import command_lines, command_process
from aigen.workflow_document_io import save_workflow_document
from aigen.workflow_graph import ImageSourceConfig, ImageSourceNode, WorkflowGraph
from aigen.workflow_run_records import new_workflow_run_id, workflow_run_path


class CommandProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdout_framing_preserves_large_records_split_utf8_and_last_line(self):
        expected = ("x" * 65535 + "雪" * 30_000, "next", "tail")
        code = "import sys; sys.stdout.buffer.write(('x'*65535 + '雪'*30000 + '\\nnext\\ntail').encode())"
        async with command_process([sys.executable, "-c", code], cwd=Path.cwd(), env=os.environ) as process:
            lines = tuple([line async for line in command_lines(process.stdout)])
            self.assertEqual(lines, expected)
            self.assertEqual(await process.wait(), 0)

    async def test_cancellation_kills_resistant_descendant_after_parent_exits(self):
        # Both processes hold stdout. The child ignores TERM; its parent accepts it.
        code = """import os, signal, time
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(os.getpid(), flush=True)
    while True: time.sleep(1)
else:
    while True: time.sleep(1)
"""
        ready = asyncio.Event()
        handles = []

        async def run():
            async with command_process([sys.executable, "-c", code], cwd=Path.cwd(), env=os.environ) as process:
                handles.append(process)
                handles.append(int(await process.stdout.readline()))
                ready.set()
                await process.stdout.read()

        with patch("aigen.tui_process.TERMINATION_GRACE_SECONDS", 0.15):
            task = asyncio.create_task(run())
            await asyncio.wait_for(ready.wait(), 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        self.assertEqual(handles[0].returncode, -signal.SIGTERM)
        self.assertTrue(handles[0].stdout.at_eof())
        # An orphan may briefly remain as an init-owned zombie, but cannot run.
        status = Path(f"/proc/{handles[1]}/stat")
        if status.exists():
            self.assertEqual(status.read_text().split()[2], "Z")

    async def test_cancellation_during_spawn_still_reaps_process(self):
        spawn = asyncio.create_subprocess_exec
        spawned = asyncio.Event()
        release = asyncio.Event()
        handles = []

        async def delayed_spawn(*args, **kwargs):
            process = await spawn(*args, **kwargs)
            handles.append(process)
            spawned.set()
            await release.wait()
            return process

        async def run():
            async with command_process([sys.executable, "-c", "import time; time.sleep(60)"],
                                       cwd=Path.cwd(), env=os.environ):
                self.fail("cancelled spawn must not enter the command body")

        with patch("aigen.tui_process.asyncio.create_subprocess_exec", side_effect=delayed_spawn):
            task = asyncio.create_task(run())
            await asyncio.wait_for(spawned.wait(), 5)
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        self.assertIsNotNone(handles[0].returncode)
        self.assertTrue(handles[0].stdout.at_eof())


class GenerationLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def wait_finished(self, app, pilot):
        async with asyncio.timeout(10):
            while app.generation_worker is not None:
                await pilot.pause(0.02)

    async def test_bad_event_releases_controls_and_allows_another_run(self):
        with TemporaryDirectory() as directory, patch("aigen.tui_process.TERMINATION_GRACE_SECONDS", 0.15):
            with patch.object(image_tui, "STATE_PATH", Path(directory) / "form.json"):
                app = image_tui.ImageGenerationApp()
            async with app.run_test() as pilot:
                errors = []
                with patch.object(app, "_show_error", side_effect=lambda title, message: errors.append(message)):
                    app._start_command([sys.executable, "-c",
                                        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                                        "print('AIGEN_PROGRESS {broken', flush=True); time.sleep(60)"], directory,
                                       action_button_id="generation-action", idle_label="Generate", error_title="Start failed")
                    await self.wait_finished(app, pilot)
                    self.assertIn("Invalid generation event", errors[0])
                    self.assertIsNone(app.process)
                    self.assertFalse(app.query_one("#generation-action", Button).disabled)
                    self.assertFalse(app.query_one("#video-action", Button).disabled)
                    app._start_command([sys.executable, "-c", "print('done')"], directory,
                                       action_button_id="generation-action", idle_label="Generate", error_title="Start failed")
                    await self.wait_finished(app, pilot)
                    self.assertEqual(len(errors), 1)

    async def test_immediate_stop_and_spawn_failure_both_finish(self):
        with TemporaryDirectory() as directory:
            with patch.object(image_tui, "STATE_PATH", Path(directory) / "form.json"):
                app = image_tui.ImageGenerationApp()
            async with app.run_test() as pilot:
                app._start_command([sys.executable, "-c", "import time; time.sleep(60)"], directory,
                                   action_button_id="generation-action", idle_label="Generate", error_title="Start failed")
                app._cancel_generation()
                await self.wait_finished(app, pilot)
                errors = []
                with patch.object(app, "_show_error", side_effect=lambda title, message: errors.append((title, message))):
                    app._start_command([str(Path(directory) / "missing-executable")], directory,
                                       action_button_id="generation-action", idle_label="Generate", error_title="Start failed")
                    await self.wait_finished(app, pilot)
                self.assertEqual(errors[0][0], "Start failed")
                self.assertIsNone(app.process)

    async def test_contact_sheet_cli_and_quit_share_responsive_process_cleanup(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            with patch.object(image_tui, "STATE_PATH", root / "form.json"), \
                 patch("aigen.tui_process.TERMINATION_GRACE_SECONDS", 0.3):
                app = image_tui.ImageGenerationApp()
                async with app.run_test() as pilot:
                    app._start_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                                        "color=c=blue:s=32x32:r=4:d=1", "-pix_fmt", "yuv420p", str(video)], directory,
                                       action_button_id="video-action", idle_label="Generate", error_title="Video test",
                                       contact_sheet_videos=(video,))
                    await self.wait_finished(app, pilot)
                    with Image.open(root / "video-contact.png") as sheet:
                        self.assertGreater(sheet.width, 32)
                    marker = root / "ready"
                    code = ("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                            f"open({str(marker)!r}, 'w').close(); time.sleep(60)")
                    app._start_command([sys.executable, "-c", code], directory,
                                       action_button_id="generation-action", idle_label="Generate", error_title="Quit test")
                    async with asyncio.timeout(5):
                        while not marker.exists():
                            await pilot.pause(0.01)
                    process = app.process
                    quit_task = asyncio.create_task(app._quit())
                    ticks = 0
                    while not quit_task.done():
                        await asyncio.sleep(0.01)
                        ticks += 1
                    await quit_task
                    self.assertGreater(ticks, 2)
                    self.assertEqual(process.returncode, -signal.SIGKILL)
                    self.assertTrue(process.stdout.at_eof())

    async def test_forced_workflow_stop_finalizes_history_and_preserves_published_nodes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (16, 16)).save(source)
            graph = WorkflowGraph(name="Forced stop", nodes=(
                ImageSourceNode(id="source", title="Source", config=ImageSourceConfig(path=str(source))),
            ), connections=())
            document = root / "workflow.json"
            save_workflow_document(graph, document)
            run_id = new_workflow_run_id()
            run_dir = workflow_run_path(root, graph.workflow_id, run_id)
            marker = root / "ready"
            code = """import signal, sys, time
from pathlib import Path
from aigen.progress import SILENT_STATUS
from aigen.workflow_compilation import compile_workflow_run
from aigen.workflow_document_io import load_workflow_document
from aigen.workflow_execution import execute_workflow
signal.signal(signal.SIGTERM, signal.SIG_IGN)
def pause_after_publication(event):
    if event.get('status') == 'completed':
        Path(sys.argv[4]).touch()
        while True: time.sleep(1)
execute_workflow(compile_workflow_run(load_workflow_document(Path(sys.argv[1]))),
                 runs_root=Path(sys.argv[2]), run_id=sys.argv[3], progress=SILENT_STATUS,
                 event_sink=pause_after_publication)
"""
            with patch.object(image_tui, "STATE_PATH", root / "form.json"), \
                 patch("aigen.tui_process.TERMINATION_GRACE_SECONDS", 0.15):
                app = image_tui.ImageGenerationApp()
                async with app.run_test() as pilot:
                    app.workflow_run_state.start(graph)
                    app._start_command([sys.executable, "-c", code, str(document), str(root), run_id, str(marker)],
                                       directory, action_button_id="workflow-run", idle_label="Run",
                                       error_title="Workflow test")
                    app.workflow_run_dir = run_dir
                    async with asyncio.timeout(5):
                        while not marker.exists():
                            await pilot.pause(0.01)
                    process = app.process
                    state_path = run_dir / "run.json"
                    self.assertEqual(json.loads(state_path.read_text())["status"], "running")
                    app._cancel_generation()
                    await self.wait_finished(app, pilot)
                    self.assertEqual(process.returncode, -signal.SIGKILL)
                    state = json.loads(state_path.read_text())
                    self.assertEqual(state["status"], "interrupted")
                    self.assertEqual(list(state["nodes"]), ["source"])
                    self.assertTrue(Path(state["nodes"]["source"]).is_file())
                    self.assertEqual(app.workflow_run_state.status("source"), "completed")
                    self.assertIsNone(app.process)
                    self.assertFalse(app.query_one("#generation-action", Button).disabled)


if __name__ == "__main__":
    unittest.main()
