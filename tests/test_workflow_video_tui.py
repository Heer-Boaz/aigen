from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from textual.widgets import Button, Input

from aigen import image_tui
from aigen.manifest_io import sha256_file
from aigen.media_timing import probe_video, verify_video
from aigen.workflow_graph import (
    AssembleVideoNode, ExtractVideoFramesNode, NodePortRef, VideoSourceConfig,
    VideoSourceNode, WorkflowConnection, WorkflowGraph,
)
from aigen.workflow_media_results import MediaCandidate, resolve_media_result
from aigen.workflow_results_tui import WorkflowResults
from test_workflow_properties import open_editor, select_node
from test_workflow_results_tui import finish_process


class WorkflowVideoUITests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_video_history_open_export_and_portable_frame_archive(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "input.mp4"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=c=red:s=64x80:r=25:d=0.16", "-c:v", "libx264", "-bf", "0", str(source)],
                           check=True, capture_output=True)
            graph = WorkflowGraph(name="Video", nodes=(
                VideoSourceNode(id="source", title="Video input", config=VideoSourceConfig(path=str(source))),
                ExtractVideoFramesNode(id="extract", title="Frames"),
                AssembleVideoNode(id="assemble", title="Video output"),
            ), connections=tuple(WorkflowConnection(id=f"wire{index}", source=NodePortRef(node_id=start, port=port),
                    target=NodePortRef(node_id=end, port=target)) for index, (start, port, end, target) in enumerate((
                        ("source", "video", "extract", "video"), ("extract", "images", "assemble", "images")))))
            with patch.object(image_tui, "DEFAULT_WORKFLOW_RUNS_ROOT", directory / "runs"):
                async with open_editor(graph, directory) as (app, editor, pilot, _):
                    await select_node(app, editor, pilot, "assemble")
                    self.assertTrue(await pilot.click("#workflow-run-target"))
                    await pilot.pause()
                    await finish_process(app, pilot)
                    self.assertTrue(await pilot.click("#workflow-results"))
                    await pilot.pause()
                    results = app.screen
                    self.assertIsInstance(results, WorkflowResults)
                    await results.workers.wait_for_complete()
                    candidate = results.candidates[0]
                    self.assertIsInstance(candidate, MediaCandidate)
                    output = resolve_media_result(candidate)
                    self.assertEqual(output.info.frames, 4)
                    self.assertTrue(results.query_one("#result-select", Button).disabled)
                    with patch("aigen.workflow_results_tui.open_artifact") as opened:
                        self.assertTrue(await pilot.click("#result-open"))
                        await results.workers.wait_for_complete()
                        opened.assert_called_once_with(Path(output.path))
                    exported = directory / "export.mp4"
                    await self.export(pilot, app, results, exported)
                    self.assertEqual(sha256_file(exported), output.content_sha256)
                    self.assertTrue(await pilot.click("#result-close"))
                    await pilot.pause()
                    await select_node(app, editor, pilot, "extract")
                    self.assertTrue(await pilot.click("#workflow-results"))
                    await pilot.pause()
                    frames = app.screen
                    await frames.workers.wait_for_complete()
                    archive_path = directory / "frames.zip"
                    await self.export(pilot, app, frames, archive_path)
                    extracted = directory / "unpacked"
                    with zipfile.ZipFile(archive_path) as archive:
                        self.assertEqual(len(archive.namelist()), 5)
                        archive.extractall(extracted)
                    rebuilt = directory / "rebuilt.mp4"
                    subprocess.run([str(Path.cwd() / ".venv/bin/python"), "-m", "aigen.cli", "video-postprocess", "assemble",
                        "--frames-manifest", str(extracted / "frames.json"), "--output", str(rebuilt)], check=True, capture_output=True)
                    verify_video(probe_video(rebuilt), timeline=output.info.timeline, frames=4)

    @staticmethod
    async def export(pilot, app, results, destination):
        assert await pilot.click("#result-export")
        await pilot.pause()
        app.screen.query_one(Input).value = str(destination)
        await pilot.press("enter")
        await pilot.pause()
        await results.workers.wait_for_complete()


if __name__ == "__main__":
    unittest.main()
