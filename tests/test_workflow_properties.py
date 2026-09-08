from contextlib import asynccontextmanager
from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image
from textual.widgets import Button, OptionList

from aigen import image_tui
from aigen.generation.image_edit import image_edit_backend_settings
from aigen.workflow_canvas import WorkflowCanvas
from aigen.workflow_compilation import compile_workflow
from aigen.workflow_document_io import load_workflow_document, save_workflow_document
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_graph import (
    AnimeGenI2VNode, CharacterEditConfig, CharacterEditNode, ImageEditConfig, ImageEditNode, ImagePostprocessNode,
    ImageSourceConfig, ImageSourceNode, NodePortRef, PixelArtFixerConfig,
    WorkflowConnection, WorkflowGraph,
)
from aigen.workflow_inspector import WorkflowInspector
from aigen.workflow_property_widgets import PropertyInput, PropertySelect
from aigen.workflow_templates import default_animegen_config


def custom_nodes():
    settings = image_edit_backend_settings("qwen-image-edit-2511-base")
    return (
        ImageEditNode(id="edit", title="Edit", config=ImageEditConfig(
            backend="qwen-image-edit-2511-base", prompt="Keep the image unchanged.",
            steps=37, guidance=5.25, sampler=settings.sampler, scheduler=settings.scheduler,
        )),
        AnimeGenI2VNode(id="anime", title="Video", config=default_animegen_config().model_copy(update={"steps": 9})),
        ImagePostprocessNode(id="fix", title="Grid", config=PixelArtFixerConfig(
            mode="fast", low_memory=False, force_step=4.0,
        )),
    )


@asynccontextmanager
async def open_editor(graph, directory, *, size=(120, 40)):
    document_path = directory / "workflow.json"
    save_workflow_document(graph, document_path)
    with patch.object(image_tui, "STATE_PATH", directory / "form.json"):
        app = image_tui.ImageGenerationApp()
        app.workflow_buffer = WorkflowEditBuffer(load_workflow_document(document_path), document_path=document_path)
        app.workflow_document_paths = [document_path]
        async with app.run_test(size=size) as pilot:
            app._open_workflow_editor()
            await pilot.pause()
            yield app, app.workflow_editor, pilot, document_path


async def select_node(app, editor, pilot, node_id):
    editor.query_one(WorkflowCanvas).set_selected_node(node_id)
    await editor.query_one(WorkflowInspector).show(app.workflow_buffer.document, node_id, None)
    editor._update_actions()
    editor.query_one(WorkflowCanvas).focus()
    await pilot.pause()


async def menu_command(app, pilot, command, *, context=False):
    await pilot.press("shift+f10" if context else "f10")
    await pilot.pause()
    options = app.screen.query_one(OptionList)
    options.highlighted = options.get_option_index(command)
    options.focus()
    await pilot.press("enter")
    await pilot.pause()


def property_widget(editor, widget_type, field):
    return next(widget for widget in editor.query(widget_type) if widget.field_name == field)


class WorkflowPropertyBufferTests(unittest.TestCase):
    def test_same_values_preserve_custom_settings_and_redo(self):
        nodes = custom_nodes()
        buffer = WorkflowEditBuffer(WorkflowGraph(name="Properties", nodes=nodes, connections=()))
        buffer.rename_graph("Renamed")
        buffer.undo()
        before = buffer.document
        for node, field in zip(nodes, ("backend", "sampling", "model"), strict=True):
            with self.subTest(field=field):
                self.assertFalse(buffer.update_node_config(node.id, field, getattr(node.config, field)))
                self.assertIs(buffer.document, before)
                self.assertTrue(buffer.can_redo)
                self.assertFalse(buffer.dirty)

    def test_actual_backend_switch_applies_new_defaults(self):
        buffer = WorkflowEditBuffer(WorkflowGraph(name="Properties", nodes=custom_nodes(), connections=()))
        backend = "flux2-klein"
        settings = image_edit_backend_settings(backend)
        self.assertTrue(buffer.update_node_config("edit", "backend", backend))
        config = buffer.document.node("edit").config
        for field in ("steps", "guidance", "strength", "sampler", "scheduler"):
            self.assertEqual(getattr(config, field), getattr(settings, field))
        buffer.undo()
        self.assertEqual(buffer.document.node("edit").config, custom_nodes()[0].config)

    def test_klein_backend_switch_resets_pose_mode_and_undo_restores_qwen_settings(self):
        node = CharacterEditNode(id="edit", title="Edit", config=CharacterEditConfig(
            backend="qwen-image-edit-2511-lightning", pose_mode="keypoint",
            max_sequence_length=512, guidance_scale=3,
        ))
        buffer = WorkflowEditBuffer(WorkflowGraph(name="Backend defaults", nodes=(node,)))
        buffer.update_node_config(node.id, "backend", "flux2-klein")
        config = buffer.document.node(node.id).config
        self.assertEqual(config.pose_mode, "native")
        self.assertIsNone(config.max_sequence_length)
        self.assertIsNone(config.guidance_scale)
        buffer.undo()
        self.assertEqual(buffer.document.node(node.id).config, node.config)


class WorkflowPropertyUITests(unittest.IsolatedAsyncioTestCase):
    async def test_cpu_run_cache_status_edits_reopen_and_stop(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.png"
            pixels = Image.new("RGBA", (16, 16), (30, 80, 120, 0))
            pixels.paste((200, 90, 20, 255), (4, 4, 12, 12))
            pixels.resize((64, 64), Image.Resampling.NEAREST).save(source)
            graph = WorkflowGraph(name="CPU execution", nodes=(
                ImageSourceNode(id="source", title="Source", config=ImageSourceConfig(path=str(source))),
                custom_nodes()[2],
            ), connections=(WorkflowConnection(id="wire", source=NodePortRef(node_id="source", port="image"),
                                               target=NodePortRef(node_id="fix", port="image")),))
            with patch.object(image_tui, "DEFAULT_WORKFLOW_RUNS_ROOT", directory / "runs"):
                async with open_editor(graph, directory) as (app, editor, pilot, path):
                    async def wait_finished():
                        for _ in range(600):
                            await pilot.pause(0.1)
                            if app.generation_worker is None:
                                return
                        self.fail("CPU subprocess did not finish within 60 seconds")

                    canvas = editor.query_one(WorkflowCanvas)
                    for expected in ("completed", "reused"):
                        if expected == "reused":
                            await editor.workers.wait_for_complete()
                            self.assertEqual(str(editor.query_one("#workflow-run", Button).label), "View results")
                            await menu_command(app, pilot, "run-again")
                        else:
                            self.assertTrue(await pilot.click("#workflow-run"))
                        await pilot.pause()
                        self.assertIsNotNone(app.process)
                        process = app.process
                        self.assertFalse(editor.query_one("#workflow-run", Button).disabled)
                        self.assertEqual(str(editor.query_one("#workflow-run", Button).label), "Stop")
                        await wait_finished()
                        self.assertTrue(process.stdout.at_eof())
                        self.assertEqual(canvas.runtime_statuses["fix"], expected)

                    await select_node(app, editor, pilot, "source")
                    property_widget(editor, PropertyInput, "path").value = str(directory / "replacement.png")
                    self.assertTrue(await editor.commit_pending_property())
                    self.assertEqual(canvas.runtime_statuses, {"source": "outdated", "fix": "outdated"})
                    await menu_command(app, pilot, "close")
                    await pilot.pause()
                    self.assertIsNone(app.workflow_editor)
                    app._open_workflow_editor()
                    await pilot.pause()
                    editor = app.workflow_editor
                    canvas = editor.query_one(WorkflowCanvas)
                    self.assertEqual(canvas.runtime_statuses["fix"], "outdated")
                    await menu_command(app, pilot, "undo")
                    await pilot.pause()
                    self.assertEqual(canvas.runtime_statuses["fix"], "reused")

                    app._start_command(
                        [sys.executable, "-c", "import signal; signal.pause()"], str(directory),
                        action_button_id="workflow-run", idle_label="Run", error_title="CPU stop test", running_label=None,
                    )
                    editor.set_running(True)
                    await pilot.pause()
                    process = app.process
                    self.assertEqual(os.getpgid(process.pid), process.pid)
                    self.assertTrue(await pilot.click("#workflow-run"))
                    await wait_finished()
                    self.assertEqual(process.returncode, -15)
                    self.assertTrue(process.stdout.at_eof())

    async def test_open_select_save_undo_redo_preserve_settings(self):
        graph = WorkflowGraph(name="Properties", nodes=custom_nodes(), connections=())
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory)) as (app, editor, pilot, path):
                original_bytes = path.read_bytes()
                for node in graph.nodes:
                    await select_node(app, editor, pilot, node.id)
                    self.assertEqual(app.workflow_buffer.document, graph)
                    self.assertFalse(app.workflow_buffer.dirty)
                    self.assertFalse(app.workflow_buffer.can_undo)
                await pilot.press("ctrl+s")
                await pilot.pause(0.4)
                self.assertEqual(path.read_bytes(), original_bytes)

                force = property_widget(editor, PropertyInput, "force_step")
                force.value = "2"
                await pilot.press("ctrl+s")
                await pilot.pause()
                self.assertEqual(load_workflow_document(path), app.workflow_buffer.document)
                self.assertEqual(app.workflow_buffer.document.node("fix").config.force_step, 2.0)
                self.assertFalse(app.workflow_buffer.dirty)
                await menu_command(app, pilot, "undo")
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document, graph)
                self.assertTrue(app.workflow_buffer.can_redo)
                await menu_command(app, pilot, "redo")
                await pilot.pause()
                self.assertEqual(load_workflow_document(path), app.workflow_buffer.document)
                self.assertFalse(app.workflow_buffer.dirty)

    async def test_implicit_defaults_compile_open_and_round_trip(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.png"
            Image.new("RGB", (64, 64)).save(source)
            graph = WorkflowGraph(name="Implicit defaults", nodes=(
                ImageSourceNode(id="source", title="Source", config=ImageSourceConfig(path=str(source))),
                ImageEditNode(id="edit", title="Edit", config=ImageEditConfig(
                    backend="qwen-image-edit-2511-base", prompt="Keep the image unchanged.",
                )),
            ), connections=(WorkflowConnection(id="wire", source=NodePortRef(node_id="source", port="image"),
                                               target=NodePortRef(node_id="edit", port="references")),))
            compiled = compile_workflow(graph)
            async with open_editor(graph, directory) as (app, editor, pilot, path):
                await select_node(app, editor, pilot, "edit")
                self.assertEqual(app.workflow_buffer.document, graph)
                for field in ("sampler", "scheduler"):
                    self.assertIsNone(property_widget(editor, PropertySelect, field).value)
                    property_widget(editor, PropertySelect, field).value = getattr(compiled.node("edit").config.settings, field)
                    await pilot.pause()
                    self.assertIsNotNone(getattr(app.workflow_buffer.document.node("edit").config, field))
                    property_widget(editor, PropertySelect, field).value = None
                    await pilot.pause()
                    self.assertIsNone(getattr(app.workflow_buffer.document.node("edit").config, field))
                await pilot.press("ctrl+s")
                await pilot.pause()
                self.assertEqual(load_workflow_document(path), graph)

    async def test_selection_commits_valid_draft_and_rejects_invalid_draft(self):
        graph = WorkflowGraph(name="Drafts", nodes=custom_nodes(), connections=())
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory)) as (app, editor, pilot, path):
                await select_node(app, editor, pilot, "edit")
                property_widget(editor, PropertyInput, "steps").value = "invalid"
                property_widget(editor, PropertySelect, "backend").value = "flux2-klein"
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document, graph)
                self.assertEqual(property_widget(editor, PropertySelect, "backend").value, "qwen-image-edit-2511-base")
                property_widget(editor, PropertyInput, "steps").value = "31"
                property_widget(editor, PropertySelect, "backend").value = "flux2-klein"
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document.node("edit").config.backend, "flux2-klein")
                self.assertEqual(app.workflow_buffer.document.node("edit").config.steps, 4)
                await menu_command(app, pilot, "undo")
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document.node("edit").config.steps, 31)


if __name__ == "__main__":
    unittest.main()
