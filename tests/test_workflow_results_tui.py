import asyncio
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image
from textual.containers import ItemGrid
from textual.widgets import Button, Input, Label

from aigen import image_tui
from aigen.manifest_io import sha256_file
from aigen.tui_dialogs import PromptDialog
from aigen.tui_result_records import ResultRecords
from aigen.workflow_artifacts import ImageCollectionArtifact
from aigen.workflow_canvas import WorkflowCanvas
from aigen.workflow_document_io import load_workflow_document
from aigen.workflow_graph import (
    ImageCollectionNode, ImageEditConfig, ImageEditNode, ImagePostprocessNode, ImageSelectionNode,
    ImageSourceConfig, ImageSourceNode, NodePortRef, PixelArtFixerConfig, WorkflowConnection, WorkflowGraph,
)
from aigen.workflow_results import node_result_history, load_node_result, resolve_image_result
from aigen.workflow_results_tui import ResultPreview, WorkflowResults
from test_workflow_properties import menu_command, open_editor, select_node


def cpu_graph(directory):
    nodes = []
    for index in range(2):
        source = directory / f"source{index}.png"
        image = Image.new("RGB", (16, 20), (40, 50, 60))
        image.paste((180 + 20 * index, 60, 20), (4, 4, 12, 16))
        image.resize((64, 80), Image.Resampling.NEAREST).save(source)
        nodes.append(ImageSourceNode(id=f"source{index}", title=f"Input {index}", config=ImageSourceConfig(path=str(source))))
        nodes.append(ImagePostprocessNode(id=f"fix{index}", title=f"Candidate {index}", config=PixelArtFixerConfig(mode="fast", low_memory=False, force_step=4.0)))
    nodes.extend((ImageCollectionNode(id="collection", title="Variants"), ImageSelectionNode(id="choice", title="Selected"),
                  ImagePostprocessNode(id="next", title="Continue", config=PixelArtFixerConfig(mode="fast", low_memory=False, force_step=2.0))))
    routes = (("source0", "image", "fix0", "image", 0), ("source1", "image", "fix1", "image", 0),
              ("fix0", "image", "collection", "images", 0), ("fix1", "image", "collection", "images", 1),
              ("collection", "collection", "choice", "collection", 0), ("choice", "image", "next", "image", 0))
    return WorkflowGraph(name="CPU result flow", nodes=tuple(nodes), connections=tuple(
        WorkflowConnection(id=f"wire{index}", source=NodePortRef(node_id=source, port=output),
                           target=NodePortRef(node_id=target, port=input_port), order=order)
        for index, (source, output, target, input_port, order) in enumerate(routes)
    ))


async def finish_process(app, pilot):
    process = app.process
    for _ in range(600):
        await pilot.pause(0.1)
        if app.generation_worker is None:
            assert process.returncode == 0
            assert process.stdout.at_eof()
            return
    raise AssertionError("CPU workflow did not finish")


class WorkflowResultsUITests(unittest.IsolatedAsyncioTestCase):
    async def test_run_telemetry_does_not_overwrite_active_node_progress(self):
        graph = WorkflowGraph(name="Progress ownership", nodes=(
            ImageEditNode(id="edit", title="Edit", config=ImageEditConfig(backend="flux2-klein")),))
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory)) as (app, editor, pilot, _):
                app.workflow_run_state.start(graph)
                app.active_action_button_id = "workflow-run"
                progress = image_tui.GenerationProgress(
                    phase="denoising 7/8", completed=7, total=8, elapsed_seconds=20,
                    remaining_seconds=3, final=False, cpu_percent=4, gpu_percent=100,
                    vram_used_mb=14000, vram_total_mb=16303)
                app.workflow_node_updated(image_tui.WorkflowNodeUpdated({
                    "node_id": "edit", "status": "running", "progress": asdict(progress)}))
                status = editor.query_one("#workflow-editor-status", Label)
                original = str(status.render())
                self.assertIn("denoising 7/8", original)
                overall = image_tui.GenerationProgress(**{**asdict(progress), "phase": "completed input", "completed": 1, "total": 4})
                app.generation_updated(image_tui.GenerationUpdated(overall))
                self.assertEqual(str(status.render()), original)
                app.active_action_button_id = None

    async def test_cli_collection_history_selection_export_and_continuation_after_restart(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            root = directory / "runs"
            graph = cpu_graph(directory)
            with patch.object(image_tui, "DEFAULT_WORKFLOW_RUNS_ROOT", root):
                async with open_editor(graph, directory) as (app, editor, pilot, document_path):
                    await select_node(app, editor, pilot, "collection")
                    await menu_command(app, pilot, "run-target", context=True)
                    await pilot.pause()
                    await finish_process(app, pilot)
                    statuses = editor.query_one(WorkflowCanvas).runtime_statuses
                    self.assertNotIn("choice", statuses)
                    self.assertEqual(statuses["collection"], "completed")
                    await menu_command(app, pilot, "results", context=True)
                    await pilot.pause()
                    results = app.screen
                    self.assertIsInstance(results, WorkflowResults)
                    await results.workers.wait_for_complete()
                    self.assertEqual(len(results.candidates), 2)
                    button = results.query_one("#candidate-1", Button)
                    button.scroll_visible(animate=False)
                    await pilot.pause()
                    self.assertTrue(await pilot.click("#candidate-1"))
                    await results.workers.wait_for_complete()
                    chosen = results.candidates[1]
                    self.assertEqual(results.highlighted_index, 1)
                    self.assertTrue(await pilot.click("#result-log"))
                    await results.workers.wait_for_complete()
                    await pilot.pause()
                    self.assertIsInstance(app.screen, ResultRecords)
                    await app.screen.workers.wait_for_complete()
                    self.assertTrue(await pilot.click("#result-record-close"))
                    await pilot.pause()
                    export = directory / "export.png"
                    self.assertTrue(await pilot.click("#result-export"))
                    await pilot.pause()
                    self.assertIsInstance(app.screen, PromptDialog)
                    app.screen.query_one(Input).value = str(export)
                    self.assertTrue(await pilot.click("#dialog-ok"))
                    await pilot.pause()
                    await results.workers.wait_for_complete()
                    self.assertEqual(sha256_file(export), sha256_file(Path(chosen.image.path)))
                    with patch("aigen.workflow_results_tui.open_artifact") as opened:
                        self.assertTrue(await pilot.click("#result-open"))
                        await results.workers.wait_for_complete()
                        opened.assert_called_once_with(Path(chosen.image.path))
                    self.assertTrue(await pilot.click("#result-select"))
                    await pilot.pause()
                    await editor.workers.wait_for_complete()
                    saved = load_workflow_document(document_path)
                    self.assertEqual(saved.node("choice").config.selected, chosen.reference)
                    self.assertEqual(resolve_image_result(chosen.reference), chosen.image)

                async with open_editor(saved, directory) as (app, editor, pilot, _):
                    self.assertTrue(await pilot.click("#workflow-run"))
                    await pilot.pause()
                    await finish_process(app, pilot)
                    self.assertEqual(set(editor.query_one(WorkflowCanvas).runtime_statuses), {"choice", "next"})
                    self.assertTrue(node_result_history(root, saved.workflow_id, "next"))
                    await select_node(app, editor, pilot, "collection")
                    await menu_command(app, pilot, "results", context=True)
                    await pilot.pause()
                    await app.screen.workers.wait_for_complete()
                    self.assertEqual(app.screen.highlighted_index, 1)
                    await self.assert_safe_history_switch(app.screen, pilot)

    async def assert_safe_history_switch(self, results, pilot):
        first, old_candidate = results.candidates
        old_view = results._view_id
        path = results.manifest_path
        manifest = load_node_result(path)
        entered, release = asyncio.Event(), asyncio.Event()
        original_remove = ItemGrid.remove_children

        async def delayed_remove(grid, *args, **kwargs):
            if grid.id == "result-gallery":
                entered.set()
                await release.wait()
            return await original_remove(grid, *args, **kwargs)

        def assert_inactive():
            self.assertIsNone(results.highlighted_index)
            for name in ("select", "open", "export", "log"):
                self.assertTrue(results.query_one(f"#result-{name}", Button).disabled)

        view = (manifest, (first,), [("First", Image.new("RGB", (8, 8)), 0)])
        with (patch.object(results, "_read_view", return_value=view) as read_view,
              patch.object(ItemGrid, "remove_children", delayed_remove)):
            worker = results.show_result(path)
            await asyncio.wait_for(entered.wait(), 2)
            self.assertEqual(results.candidates, ())
            self.assertTrue(results.query_one("#result-gallery", ItemGrid).disabled)
            results.post_message(ResultPreview.Highlighted(old_candidate, 1, old_view))
            await pilot.pause()
            assert_inactive()
            release.set()
            await worker.wait()
            await results.workers.wait_for_complete()
            self.assertEqual(results.candidates, (first,))
            self.assertEqual(results.highlighted_index, 0)

            previous_view = results._view_id
            read_view.side_effect = ValueError("Unreadable history")
            await results.show_result(path).wait()
            self.assertEqual(results.candidates, ())
            self.assertIsNone(results.manifest_path)
            self.assertFalse(results.query("#result-gallery ResultPreview"))
            self.assertFalse(results.query("#result-inputs ResultPreview"))
            results.post_message(ResultPreview.Highlighted(first, 0, previous_view))
            await pilot.pause()
            assert_inactive()

    async def test_seed_variants_dialog_is_one_undo_edit(self):
        graph = WorkflowGraph(name="Create variants", nodes=(ImageEditNode(id="edit", title="Edit", config=ImageEditConfig(backend="flux2-klein", seed=10)),))
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory)) as (app, editor, pilot, _):
                await select_node(app, editor, pilot, "edit")
                await menu_command(app, pilot, "variants", context=True)
                await pilot.pause()
                app.screen.query_one(Input).value = "10, 20, 30"
                self.assertTrue(await pilot.click("#dialog-ok"))
                await pilot.pause()
                await editor.workers.wait_for_complete()
                self.assertEqual(len(app.workflow_buffer.document.nodes), 5)
                await menu_command(app, pilot, "undo")
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document, graph)


if __name__ == "__main__":
    unittest.main()
