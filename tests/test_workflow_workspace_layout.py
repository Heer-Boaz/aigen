from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image
from textual import events
from textual.containers import VerticalScroll
from textual.document._document import Selection
from textual.widgets import Button, Collapsible, Label, Select, TextArea

from aigen.manifest_io import atomic_write_json, sha256_file
from aigen.workflow_artifacts import ImageArtifact, ImageCollectionArtifact
from aigen.workflow_cache import NodeExecutionDetails
from aigen.workflow_canvas import WorkflowCanvas
from aigen.workflow_editor_layout import InspectorDivider, WorkflowEditorBody
from aigen.workflow_graph import ImageCollectionNode, ImageSelectionNode, NodeKind, NodePortRef, WorkflowConnection, WorkflowGraph
from aigen.workflow_inspector import WorkflowInspector
from aigen.workflow_property_widgets import PropertyInput, PropertyTextArea
from aigen.workflow_result_widgets import ResultComparison, ResultPreview
from aigen.workflow_results import NodeResultManifest
from aigen.workflow_results_tui import WorkflowResults
from test_workflow_interactions import interaction_graph
from test_workflow_properties import open_editor, property_widget, select_node


def recorded_candidates(root: Path, count=8):
    source = root / "input.png"
    Image.new("RGB", (48, 64), (30, 60, 100)).save(source)
    checksum = sha256_file(source)
    original = ImageArtifact(path=str(source), identity=checksum, content_sha256=checksum)
    candidates = []
    for index in range(count):
        source = root / f"output{index}.png"
        image = Image.new("RGB", (48, 64), (30 + index * 20, 80, 110))
        image.paste((200, 190, 140), (12, 8, 36, 56))
        image.save(source)
        checksum = sha256_file(source)
        result = NodeResultManifest(node_id=f"source{index}", node_kind=NodeKind.IMAGE_SOURCE,
            title=f"Candidate {index + 1}", signature=f"{index:064x}",
            outputs={"image": ImageArtifact(path=str(source), identity=checksum, content_sha256=checksum)},
            details=NodeExecutionDetails(completed_at="2026-09-08", effective_config={"backend": "fixture", "seed": index},
                                         inputs={"reference": (original,)}, measured_outputs={}))
        path = root / "attempt-1" / "nodes" / result.node_id / "result.json"
        atomic_write_json(path, result.model_dump(mode="json"))
        candidates.append(result.candidate(path, "image"))
    collection = NodeResultManifest(node_id="collection", node_kind=NodeKind.IMAGE_COLLECTION, signature="a" * 64,
        outputs={"collection": ImageCollectionArtifact(candidates=tuple(candidates), identity="collection")})
    path = root / "attempt-1" / "nodes" / "collection" / "result.json"
    atomic_write_json(path, collection.model_dump(mode="json"))
    graph = WorkflowGraph(name="Recorded candidates", nodes=(ImageCollectionNode(id="collection", title="Candidates"),
        ImageSelectionNode(id="choice", title="Chosen output")), connections=(WorkflowConnection(id="wire",
            source=NodePortRef(node_id="collection", port="collection"), target=NodePortRef(node_id="choice", port="collection")),))
    return graph, path


class WorkspaceLayoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_inspector_resize_expand_hide_and_groups_keep_native_drafts(self):
        with TemporaryDirectory() as directory:
            async with open_editor(interaction_graph(), Path(directory), size=(160, 50)) as (app, editor, pilot, _):
                body = editor.query_one(WorkflowEditorBody)
                canvas = editor.query_one(WorkflowCanvas)
                self.assertFalse(body.has_class("inspector-open"))
                self.assertEqual(canvas.region.width, body.region.width)
                await select_node(app, editor, pilot, "edit")
                self.assertTrue(body.has_class("inspector-open"))
                self.assertFalse(any(widget.field_name == "name" for widget in editor.query(PropertyInput)))
                advanced = editor.query_one("#workflow-advanced", Collapsible)
                self.assertTrue(advanced.collapsed)
                text = property_widget(editor, PropertyTextArea, "prompt")
                text.focus()
                value = "First line\n\nLast line"
                app.post_message(events.Paste(value))
                await pilot.pause()
                text.selection = Selection((0, 2), (2, 5))
                selection = text.selection
                panel = editor.query_one("#workflow-inspector-panel")
                width = panel.size.width
                divider = editor.query_one(InspectorDivider)
                x, y = divider.region.x, divider.region.y + 3
                await pilot.mouse_down(offset=(x, y))
                self.assertIs(app.mouse_captured, divider)
                await pilot.hover(offset=(x - 16, y))
                await pilot.mouse_up(offset=(x - 16, y))
                await pilot.pause()
                self.assertGreaterEqual(panel.size.width, width + 15)
                self.assertEqual(app.workflow_buffer.revision, 0)
                self.assertEqual(text.selection, selection)

                self.assertTrue(await pilot.click("#workflow-inspector-expand"))
                await pilot.pause()
                self.assertTrue(body.expanded)
                self.assertFalse(canvas.display)
                self.assertEqual(panel.size.width, body.size.width)
                self.assertGreaterEqual(text.content_size.height, 20)
                restore = editor.query_one("#workflow-inspector-expand", Button)
                self.assertGreaterEqual(restore.content_size.width, restore.label.cell_length)
                await pilot.press("escape")
                self.assertFalse(body.expanded)
                await pilot.press("escape")
                self.assertFalse(body.has_class("inspector-open"))
                self.assertEqual(canvas.region.width, body.region.width)
                body.show_inspector(True)
                for width, height in ((80, 24), (120, 40), (160, 50)):
                    await pilot.resize_terminal(width, height)
                    await pilot.pause()
                    self.assertIs(property_widget(editor, PropertyTextArea, "prompt"), text)
                    self.assertEqual(text.text, value)
                    self.assertEqual(text.selection, selection)
                    self.assertGreaterEqual(text.content_size.height, 4)
                advanced.collapsed = False
                guidance = property_widget(editor, PropertyInput, "guidance")
                guidance.value = "invalid"
                await pilot.pause()
                advanced.collapsed = True
                self.assertFalse(await editor.commit_pending_property())
                await pilot.pause()
                self.assertFalse(advanced.collapsed)
                self.assertTrue(guidance.has_focus)
                self.assertEqual(app.workflow_buffer.revision, 0)
                guidance.value = guidance.original_value
                text.focus()
                await pilot.press("ctrl+z")
                self.assertEqual(text.text, "")
                self.assertEqual(editor.query_one(WorkflowInspector).property_drafts(), ())
                body.show_inspector(False)
                await pilot.press("escape")
                await pilot.pause()
                self.assertIsNone(app.workflow_editor)

    async def test_results_keep_active_output_and_choice_visible_across_sizes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            graph, path = recorded_candidates(root)
            async with open_editor(graph, root, size=(80, 24)) as (app, editor, pilot, _):
                results = WorkflowResults(graph, "choice", root)
                with patch("aigen.workflow_results_tui.node_result_history", return_value=(path,)):
                    app.push_screen(results)
                    await pilot.pause()
                    await results.workers.wait_for_complete()
                comparison = results.query_one(ResultComparison)
                self.assertEqual(results.query_one("#result-details", TextArea).text, "")
                for size in ((80, 24), (120, 40), (160, 50)):
                    await pilot.resize_terminal(*size)
                    await pilot.pause()
                    self.assertFalse(results.query_one("#result-details").display)
                    self.assertTrue(results.query_one("#result-output").display)
                    self.assertGreater(comparison.region.height, 7)
                    for selector in ("#result-selected", "#result-actions"):
                        region = results.query_one(selector).region
                        self.assertEqual(region.intersection(results.region), region)
                    results.query_one("#candidate-0", ResultPreview).focus()
                    await pilot.press("left")
                    await results.workers.wait_for_complete()
                    self.assertEqual(results.highlighted_index, 7)
                    self.assertIn("Output 8/8", str(results.query_one("#result-selected", Label).content))
                    self.assertIn("Use output 8", results.query_one("#result-select", Button).tooltip)
                    gallery = results.query_one("#result-gallery-scroll", VerticalScroll)
                    tile = results.query_one("#candidate-7", ResultPreview)
                    await pilot.pause()
                    self.assertEqual(tile.region.intersection(gallery.content_region), tile.region)
                    self.assertFalse(results.query_one("#result-select", Button).disabled)
                    other_tile = results.query_one("#candidate-6", ResultPreview)
                    self.assertTrue(await pilot.click(offset=(other_tile.region.x + 3, other_tile.region.y + 2)))
                    await results.workers.wait_for_complete()
                    self.assertEqual(results.highlighted_index, 6)
                    self.assertIs(app.screen, results)
                    if size[0] >= 120:
                        results.query_one("#result-view-mode", Select).value = "compare"
                        await pilot.pause()
                        self.assertTrue(results.query_one("#result-inputs").display)
                        self.assertTrue(results.query_one("#result-output").display)
                    await results.inspect_candidate(results.candidates[0], 0, results._view_id).wait()
                await pilot.resize_terminal(80, 24)
                await pilot.pause()
                self.assertTrue(results.query_one("#result-output").display)
                self.assertFalse(results.query_one("#result-inputs").display)
                results.query_one("#result-view-mode", Select).value = "input"
                await pilot.pause()
                self.assertIn("Output 1/8", str(results.query_one("#result-selected", Label).content))
                self.assertTrue(await pilot.click("#result-info"))
                await pilot.pause()
                self.assertTrue(results.query_one("#result-details").display)
                self.assertIn('"seed": 0', results.query_one("#result-details", TextArea).text)
                await pilot.press("escape")
                self.assertIs(app.screen, editor)


if __name__ == "__main__":
    unittest.main()
