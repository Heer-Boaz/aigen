from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

from textual import events
from textual.document._document import Selection
from textual.widgets import Label

from aigen.generation.image_edit import image_edit_backend_settings
from aigen.workflow_canvas import WorkflowCanvas
from aigen.workflow_document_io import load_workflow_document
from aigen.workflow_editor import WorkflowEditorBody
from aigen.workflow_graph import CharacterEditConfig, CharacterEditNode, Ltx23Node, WorkflowGraph
from aigen.workflow_inspector import WorkflowInspector
from aigen.workflow_property_widgets import PropertyInput, PropertyRow, PropertySelect, PropertyTextArea
from test_workflow_interactions import click_node, interaction_graph
from test_workflow_properties import open_editor, property_widget, select_node


MULTILINE_TEXT = "First line\n\n  Second line – café 日本語  \n"


async def paste(app, pilot, widget, text):
    widget.focus()
    await pilot.press("f7")
    app.post_message(events.Paste(text))
    await pilot.pause()


class WorkflowTextEditingTests(unittest.IsolatedAsyncioTestCase):
    async def test_multiline_paste_commit_save_and_reload_keep_text_and_selection(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            async with open_editor(interaction_graph(), directory) as (app, editor, pilot, path):
                await select_node(app, editor, pilot, "edit")
                text = property_widget(editor, PropertyTextArea, "prompt")
                await paste(app, pilot, text, MULTILINE_TEXT)
                text.selection = Selection((0, 2), (2, 7))
                selection = text.selection
                await pilot.press("ctrl+enter")
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document.node("edit").config.prompt, MULTILINE_TEXT)
                self.assertEqual(app.workflow_buffer.revision, 1)
                self.assertIs(property_widget(editor, PropertyTextArea, "prompt"), text)
                self.assertEqual(text.selection, selection)
                self.assertTrue(text.has_focus)
                self.assertEqual(editor.query_one(WorkflowInspector).property_drafts(), ())
                await pilot.press("ctrl+s")
                await pilot.pause()
                self.assertEqual(load_workflow_document(path).node("edit").config.prompt, MULTILINE_TEXT)
                self.assertFalse(app.workflow_buffer.dirty)
                self.assertEqual(text.selection, selection)
                self.assertTrue(text.has_focus)

                await pilot.press("ctrl+z")
                await pilot.pause()
                self.assertEqual(text.text, "")
                self.assertEqual(app.workflow_buffer.revision, 1)
                await pilot.press("ctrl+y")
                await pilot.pause()
                self.assertEqual(text.text, MULTILINE_TEXT)
                self.assertEqual(editor.query_one(WorkflowInspector).property_drafts(), ())
                saved = load_workflow_document(path)
            async with open_editor(saved, directory) as (app, editor, pilot, _):
                await select_node(app, editor, pilot, "edit")
                self.assertEqual(property_widget(editor, PropertyTextArea, "prompt").text, MULTILINE_TEXT)
                self.assertEqual(editor.query_one(WorkflowInspector).property_drafts(), ())
                self.assertFalse(app.workflow_buffer.dirty)

    async def test_enter_tab_clipboard_and_undo_stay_in_the_text_editor(self):
        graph = interaction_graph()
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory)) as (app, editor, pilot, _):
                app.workflow_buffer.rename_graph("Graph history")
                await select_node(app, editor, pilot, "edit")
                text = property_widget(editor, PropertyTextArea, "prompt")
                text.focus()
                await pilot.pause()
                self.assertIn("Ctrl+Enter: apply", str(editor.query_one("#workflow-editor-status", Label).content))
                await pilot.press("ctrl+z", "ctrl+y", "ctrl+shift+z")
                self.assertEqual(app.workflow_buffer.document.name, "Graph history")
                await pilot.press("a", "enter", "b")
                await pilot.pause()
                self.assertEqual(text.text, "a\nb")
                self.assertEqual(app.workflow_buffer.revision, 1)
                await pilot.press("f7", "ctrl+c")
                self.assertEqual(app.clipboard, "a\nb")
                self.assertIs(app.screen, editor)
                await pilot.press("delete")
                self.assertEqual(text.text, "")
                self.assertEqual(len(app.workflow_buffer.document.nodes), 3)
                await pilot.press("ctrl+z")
                self.assertEqual(text.text, "a\nb")
                await pilot.press("ctrl+shift+z")
                self.assertEqual(text.text, "")
                await pilot.press("ctrl+z", "tab")
                self.assertIs(app.focused, property_widget(editor, PropertySelect, "seed_mode"))
                await pilot.press("shift+tab")
                self.assertIs(app.focused, text)
                self.assertEqual(text.text, "a\nb")
                await pilot.press("ctrl+enter")
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document.node("edit").config.prompt, "a\nb")
                await pilot.press("end", "c", "ctrl+z")
                self.assertEqual(text.text, "a\nb")
                self.assertEqual(app.workflow_buffer.revision, 2)

    async def test_resize_and_drawer_preserve_selection_scroll_and_history(self):
        with TemporaryDirectory() as directory:
            async with open_editor(interaction_graph(), Path(directory), size=(160, 50)) as (app, editor, pilot, _):
                await select_node(app, editor, pilot, "edit")
                text = property_widget(editor, PropertyTextArea, "prompt")
                value = "\n".join(f"Line {index:02}" for index in range(25))
                await paste(app, pilot, text, value)
                text.selection = Selection((18, 2), (19, 5))
                await pilot.pause()
                selection, scroll = text.selection, text.scroll_offset
                self.assertGreater(scroll.y, 0)
                for size in ((80, 24), (160, 50), (80, 24)):
                    await pilot.resize_terminal(*size)
                    body = editor.query_one(WorkflowEditorBody)
                    body.show_inspector(True)
                    await pilot.pause()
                    self.assertIs(property_widget(editor, PropertyTextArea, "prompt"), text)
                    self.assertEqual(text.text, value)
                    self.assertEqual(text.selection, selection)
                    self.assertEqual(text.scroll_offset, scroll)
                    self.assertGreaterEqual(text.content_size.height, 4)
                    body.show_inspector(False)
                    await pilot.pause()
                    body.show_inspector(True)
                text.focus()
                await pilot.press("ctrl+z")
                self.assertEqual(text.text, "")
                self.assertFalse(app.workflow_buffer.dirty)

    async def test_invalid_other_field_keeps_draft_then_backend_switch_retains_editor(self):
        with TemporaryDirectory() as directory:
            async with open_editor(interaction_graph(), Path(directory)) as (app, editor, pilot, _):
                await select_node(app, editor, pilot, "edit")
                text = property_widget(editor, PropertyTextArea, "prompt")
                await paste(app, pilot, text, MULTILINE_TEXT)
                steps = property_widget(editor, PropertyInput, "steps")
                steps.value = "invalid"
                await pilot.press("ctrl+enter")
                await pilot.pause()
                self.assertFalse(app.workflow_buffer.dirty)
                self.assertEqual(text.text, MULTILINE_TEXT)
                self.assertTrue(steps.has_focus)
                self.assertTrue(steps.has_class("-invalid"))
                steps.value = "31"
                backend = property_widget(editor, PropertySelect, "backend")
                backend.value = "qwen-image-edit-2511-lightning"
                await pilot.pause()
                config = app.workflow_buffer.document.node("edit").config
                self.assertEqual(config.prompt, MULTILINE_TEXT)
                self.assertEqual(config.backend, "qwen-image-edit-2511-lightning")
                self.assertEqual(config.steps, image_edit_backend_settings(config.backend).steps)
                self.assertEqual(app.workflow_buffer.revision, 2)
                self.assertIs(property_widget(editor, PropertyTextArea, "prompt"), text)
                self.assertIs(property_widget(editor, PropertySelect, "backend"), backend)
                text.focus()
                await pilot.press("ctrl+z")
                self.assertEqual(text.text, "")
                self.assertEqual(app.workflow_buffer.revision, 2)

    async def test_node_switch_commits_and_graph_undo_refreshes_text(self):
        graph = interaction_graph()
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory), size=(160, 40)) as (app, editor, pilot, _):
                canvas = editor.query_one(WorkflowCanvas)
                await click_node(pilot, canvas, "edit")
                await paste(app, pilot, property_widget(editor, PropertyTextArea, "prompt"), MULTILINE_TEXT)
                await click_node(pilot, canvas, "other")
                self.assertEqual(app.workflow_buffer.document.node("edit").config.prompt, MULTILINE_TEXT)
                self.assertEqual(property_widget(editor, PropertyTextArea, "prompt").text, "")
                await click_node(pilot, canvas, "edit")
                text = property_widget(editor, PropertyTextArea, "prompt")
                await pilot.press("ctrl+z")
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document, graph)
                self.assertIs(property_widget(editor, PropertyTextArea, "prompt"), text)
                self.assertEqual(text.text, "")
                await pilot.press("ctrl+y")
                await pilot.pause()
                self.assertEqual(text.text, MULTILINE_TEXT)
                text.focus()
                await pilot.press("ctrl+z")
                self.assertEqual(text.text, MULTILINE_TEXT)
                self.assertEqual(editor.query_one(WorkflowInspector).property_drafts(), ())

    async def test_negative_prompt_and_document_replacement_have_independent_text(self):
        graph = WorkflowGraph(name="Video text", nodes=(Ltx23Node(id="video", title="Video"),))
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory)) as (app, editor, pilot, _):
                await select_node(app, editor, pilot, "video")
                negative = property_widget(editor, PropertyTextArea, "negative_prompt")
                await paste(app, pilot, negative, MULTILINE_TEXT)
                await pilot.press("ctrl+s")
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document.node("video").config.negative_prompt, MULTILINE_TEXT)
                self.assertEqual(app.workflow_buffer.document.node("video").config.prompt, "")
                app.workflow_buffer.replace_document(graph)
                await editor.show_replaced_document("Replaced")
                await select_node(app, editor, pilot, "video")
                replacement = property_widget(editor, PropertyTextArea, "negative_prompt")
                self.assertIsNot(replacement, negative)
                self.assertEqual(replacement.text, "")
                replacement.focus()
                await pilot.press("ctrl+z")
                self.assertEqual(replacement.text, "")
                self.assertFalse(app.workflow_buffer.dirty)

    async def test_visibility_changes_keep_editor_and_update_native_options(self):
        graph = WorkflowGraph(name="Conditional properties", nodes=(
            CharacterEditNode(id="edit", title="Edit", config=CharacterEditConfig(backend="flux2-klein")),
        ))
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory)) as (app, editor, pilot, _):
                await select_node(app, editor, pilot, "edit")
                text = property_widget(editor, PropertyTextArea, "prompt")
                await paste(app, pilot, text, MULTILINE_TEXT)
                seed = property_widget(editor, PropertyInput, "seed")
                seed.value = "73"
                property_widget(editor, PropertySelect, "seed_mode").value = "random"
                await pilot.pause()
                self.assertFalse(seed.query_ancestor(PropertyRow).display)
                self.assertEqual(app.workflow_buffer.document.node("edit").config.seed, 73)
                property_widget(editor, PropertySelect, "seed_mode").value = "fixed"
                await pilot.pause()
                self.assertIs(property_widget(editor, PropertyInput, "seed"), seed)
                self.assertTrue(seed.query_ancestor(PropertyRow).display)
                self.assertEqual(seed.value, "73")
                structure = property_widget(editor, PropertySelect, "structure_control")
                self.assertFalse(structure.query_ancestor(PropertyRow).display)
                property_widget(editor, PropertySelect, "backend").value = "qwen-image-edit-2511-lightning"
                await pilot.pause()
                self.assertTrue(structure.query_ancestor(PropertyRow).display)
                pose = property_widget(editor, PropertySelect, "pose_mode")
                pose.value = "keypoint"
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document.node("edit").config.pose_mode, "keypoint")
                property_widget(editor, PropertySelect, "backend").value = "flux2-klein"
                await pilot.pause()
                self.assertEqual(pose.value, "native")
                self.assertFalse(structure.query_ancestor(PropertyRow).display)
                self.assertIs(property_widget(editor, PropertyTextArea, "prompt"), text)
                self.assertEqual(text.text, MULTILINE_TEXT)
                text.focus()
                await pilot.press("ctrl+z")
                self.assertEqual(text.text, "")

    async def test_stop_shortcut_reaches_process_owner_from_multiline_editor(self):
        with TemporaryDirectory() as directory:
            async with open_editor(interaction_graph(), Path(directory)) as (app, editor, pilot, _):
                await select_node(app, editor, pilot, "edit")
                text = property_widget(editor, PropertyTextArea, "prompt")
                await paste(app, pilot, text, MULTILINE_TEXT)
                app._start_command(
                    [sys.executable, "-c", "import signal; signal.pause()"], directory,
                    action_button_id="workflow-run", idle_label="Run", error_title="Stop test", running_label=None,
                )
                editor.set_running(True)
                await pilot.pause()
                process = app.process
                self.assertIsNotNone(process)
                text.focus()
                await pilot.press("shift+f5")
                for _ in range(100):
                    await pilot.pause(0.05)
                    if app.generation_worker is None:
                        break
                self.assertIsNone(app.generation_worker)
                self.assertEqual(process.returncode, -15)
                self.assertTrue(process.stdout.at_eof())
                self.assertEqual(text.text, MULTILINE_TEXT)


if __name__ == "__main__":
    unittest.main()
