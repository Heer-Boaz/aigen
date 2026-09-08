from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

from textual import events
from textual.geometry import Offset
from textual.pilot import _get_mouse_message_arguments
from textual.widgets import Button, Input, OptionList, Select

from aigen.tui_choice_menu import ChoiceMenu
from aigen.workflow_canvas import WorkflowCanvas
from aigen.workflow_connection_dialog import WorkflowConnectionDialog
from aigen.workflow_document_io import load_workflow_document
from aigen.workflow_editor import WorkflowEditorBody
from aigen.workflow_graph import (
    ImageEditConfig, ImageEditNode, ImageSourceConfig, ImageSourceNode,
    NodeLayout, NodePortRef, WorkflowConnection, WorkflowGraph,
)
from aigen.workflow_inspector import ConnectionOrderSelect, WorkflowInspector
from aigen.workflow_property_widgets import PropertyInput
from test_workflow_properties import menu_command, open_editor, property_widget


def interaction_graph():
    return WorkflowGraph(name="Interactions", nodes=(
        ImageSourceNode(id="source", title="Source", layout=NodeLayout(x=4, y=2),
                        config=ImageSourceConfig(path="source.png")),
        ImageEditNode(id="edit", title="Edit", layout=NodeLayout(x=46, y=2),
                      config=ImageEditConfig(backend="flux2-klein")),
        ImageEditNode(id="other", title="Other", layout=NodeLayout(x=46, y=15),
                      config=ImageEditConfig(backend="flux2-klein")),
    ))


def screen_point(canvas, x, y):
    return tuple(canvas.content_region.offset + Offset(x, y) - canvas.scroll_offset)


def node_body(canvas, node_id):
    node = canvas.node_geometry(node_id)
    return screen_point(canvas, node.x + 10, node.y + 1)


async def click_node(pilot, canvas, node_id):
    await pilot.click(offset=node_body(canvas, node_id))
    await pilot.pause()


class WorkflowInteractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_rapid_release_recapture_and_repeated_keys_do_not_lose_edits(self):
        graph = interaction_graph()
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory), size=(160, 40)) as (app, editor, pilot, _):
                canvas = editor.query_one(WorkflowCanvas)
                point = node_body(canvas, "source")
                await pilot.mouse_down(offset=point)
                # Real terminal events may arrive in one batch, without Pilot's
                # usual flush between each mouse/key operation.
                for kind in (events.MouseUp, events.MouseDown):
                    editor._forward_event(kind(**_get_mouse_message_arguments(editor, point, button=1)))
                await pilot.pause()
                self.assertIs(app.mouse_captured, canvas)
                await pilot.mouse_up(offset=(point[0] + 8, point[1] + 3))
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document.node("source").layout, NodeLayout(x=12, y=5))
                for _ in range(3):
                    app.post_message(events.Key("shift+right", None))
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document.node("source").layout, NodeLayout(x=18, y=5))
                self.assertEqual(app.workflow_buffer.revision, 4)
                for _ in range(4):
                    await pilot.press("ctrl+z")
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document, graph)

    async def test_rapid_selection_and_drag_never_project_a_stale_selection(self):
        with TemporaryDirectory() as directory:
            async with open_editor(interaction_graph(), Path(directory), size=(160, 40)) as (app, editor, pilot, _):
                canvas = editor.query_one(WorkflowCanvas)
                inspector = editor.query_one(WorkflowInspector)
                await click_node(pilot, canvas, "edit")
                property_widget(editor, PropertyInput, "steps").value = "31"
                projected = []
                show = inspector.show

                async def observe(document, node_id, connection_id):
                    projected.append((node_id, canvas.selected_node_id))
                    await show(document, node_id, connection_id)

                source = node_body(canvas, "source")
                other = node_body(canvas, "other")
                end = (other[0] + 5, other[1] + 2)
                with patch.object(inspector, "show", side_effect=observe):
                    for kind, point in ((events.MouseDown, source), (events.MouseUp, source),
                                        (events.MouseDown, other), (events.MouseMove, end), (events.MouseUp, end)):
                        editor._forward_event(kind(**_get_mouse_message_arguments(editor, point, button=1)))
                    await pilot.pause()
                self.assertTrue(projected)
                self.assertTrue(all(requested == selected for requested, selected in projected), projected)
                self.assertEqual(app.workflow_buffer.document.node("edit").config.steps, 31)
                self.assertEqual(app.workflow_buffer.document.node("other").layout, NodeLayout(x=51, y=17))
                self.assertEqual(app.workflow_buffer.revision, 2)

    async def test_body_drag_is_one_saved_edit_and_click_is_not_an_edit(self):
        graph = interaction_graph()
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory), size=(160, 40)) as (app, editor, pilot, path):
                canvas = editor.query_one(WorkflowCanvas)
                await click_node(pilot, canvas, "source")
                self.assertEqual(canvas.selected_node_id, "source")
                self.assertEqual(app.workflow_buffer.revision, 0)
                start = node_body(canvas, "source")
                await pilot.mouse_down(offset=start)
                self.assertIs(app.mouse_captured, canvas)
                await pilot.hover(offset=(start[0] + 5, start[1] + 2))
                self.assertEqual(app.workflow_buffer.revision, 0)
                await pilot.mouse_up(offset=(start[0] + 8, start[1] + 3))
                await pilot.pause()
                self.assertIsNone(app.mouse_captured)
                self.assertEqual(app.workflow_buffer.document.node("source").layout, NodeLayout(x=12, y=5))
                self.assertEqual(app.workflow_buffer.revision, 1)
                self.assertIsNone(editor.get_selected_text())
                self.assertFalse(editor.selections)
                await pilot.click(offset=node_body(canvas, "source"), times=2)
                self.assertFalse(editor.selections)
                await pilot.press("ctrl+s")
                await pilot.pause()
                self.assertEqual(load_workflow_document(path), app.workflow_buffer.document)
                await pilot.press("ctrl+z")
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document, graph)
                await pilot.press("ctrl+y")
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document, load_workflow_document(path))

    async def test_draft_is_accepted_before_same_or_other_node_drag(self):
        for target in ("edit", "source"):
            with self.subTest(target=target), TemporaryDirectory() as directory:
                graph = interaction_graph()
                async with open_editor(graph, Path(directory), size=(160, 40)) as (app, editor, pilot, _):
                    canvas = editor.query_one(WorkflowCanvas)
                    await click_node(pilot, canvas, "edit")
                    property_widget(editor, PropertyInput, "steps").value = "31"
                    start = node_body(canvas, target)
                    await pilot.mouse_down(offset=start)
                    self.assertIs(app.mouse_captured, canvas)
                    await pilot.mouse_up(offset=(start[0] + 6, start[1] + 2))
                    await pilot.pause()
                    original = graph.node(target).layout
                    self.assertEqual(app.workflow_buffer.document.node(target).layout,
                                     NodeLayout(x=original.x + 6, y=original.y + 2))
                    self.assertEqual(app.workflow_buffer.document.node("edit").config.steps, 31)
                    self.assertEqual(app.workflow_buffer.revision, 2)
                    await pilot.press("ctrl+z")
                    await pilot.pause()
                    self.assertEqual(app.workflow_buffer.document.node(target).layout, original)
                    self.assertEqual(app.workflow_buffer.document.node("edit").config.steps, 31)
                    await pilot.press("ctrl+z")
                    await pilot.pause()
                    self.assertEqual(app.workflow_buffer.document, graph)

    async def test_invalid_draft_prevents_gesture_and_reopens_hidden_properties(self):
        graph = interaction_graph()
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory), size=(160, 40)) as (app, editor, pilot, _):
                canvas = editor.query_one(WorkflowCanvas)
                await click_node(pilot, canvas, "edit")
                value = property_widget(editor, PropertyInput, "steps")
                value.value = "invalid"
                await pilot.resize_terminal(80, 24)
                await pilot.pause()
                start = node_body(canvas, "source")
                await pilot.mouse_down(offset=start)
                await pilot.mouse_up(offset=(start[0] + 5, start[1] + 2))
                await pilot.pause()
                self.assertEqual(canvas.selected_node_id, "edit")
                self.assertIsNone(app.mouse_captured)
                self.assertEqual(app.workflow_buffer.document, graph)
                self.assertTrue(editor.query_one(WorkflowEditorBody).has_class("inspector-open"))
                self.assertIs(property_widget(editor, PropertyInput, "steps"), value)
                self.assertTrue(value.has_focus)
                self.assertTrue(value.has_class("-invalid"))

    async def test_escape_and_capture_loss_restore_drag_preview(self):
        graph = interaction_graph()
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory), size=(160, 40)) as (app, editor, pilot, _):
                canvas = editor.query_one(WorkflowCanvas)
                for cancel in ("escape", "release"):
                    with self.subTest(cancel=cancel):
                        start = node_body(canvas, "source")
                        await pilot.mouse_down(offset=start)
                        await pilot.hover(offset=(start[0] + 8, start[1] + 4))
                        if cancel == "escape":
                            await pilot.press("escape")
                        else:
                            app.capture_mouse(None)
                        await pilot.pause()
                        await pilot.mouse_up(offset=(start[0] + 8, start[1] + 4))
                        self.assertEqual(app.workflow_buffer.document, graph)
                        self.assertEqual(canvas.node_geometry("source").x, 4)
                        self.assertEqual(canvas.node_geometry("source").y, 2)
                        self.assertIsNone(app.mouse_captured)

    async def test_drag_uses_world_coordinates_after_scrolling_during_gesture(self):
        graph = interaction_graph().model_copy(update={"nodes": (
            ImageSourceNode(id="source", title="Source", layout=NodeLayout(x=90, y=30)),
            ImageSourceNode(id="far", title="Far", layout=NodeLayout(x=200, y=70)),
        )})
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory), size=(160, 40)) as (app, editor, pilot, _):
                canvas = editor.query_one(WorkflowCanvas)
                canvas.scroll_to(70, 20, animate=False, immediate=True)
                await pilot.pause()
                start = node_body(canvas, "source")
                await pilot.mouse_down(offset=start)
                canvas.scroll_to(75, 23, animate=False, immediate=True)
                await pilot.pause()
                await pilot.mouse_up(offset=(start[0] + 4, start[1] + 2))
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document.node("source").layout, NodeLayout(x=99, y=35))
                self.assertEqual(app.workflow_buffer.revision, 1)

    async def test_ports_connect_and_reconnect_without_moving_nodes(self):
        graph = interaction_graph()
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory), size=(160, 40)) as (app, editor, pilot, _):
                canvas = editor.query_one(WorkflowCanvas)
                source = canvas.node_geometry("source").outputs[0]
                target = canvas.node_geometry("edit").inputs[0]
                await pilot.mouse_down(offset=screen_point(canvas, source.x, source.y))
                await pilot.mouse_up(offset=screen_point(canvas, target.x, target.y))
                await pilot.pause()
                connection = app.workflow_buffer.document.connections[0]
                self.assertEqual(connection.source.node_id, "source")
                self.assertEqual(connection.target.node_id, "edit")
                replacement = canvas.node_geometry("other").inputs[0]
                await pilot.mouse_down(offset=screen_point(canvas, target.x, target.y))
                await pilot.mouse_up(offset=screen_point(canvas, replacement.x, replacement.y))
                await pilot.pause()
                self.assertEqual(len(app.workflow_buffer.document.connections), 1)
                self.assertEqual(app.workflow_buffer.document.connections[0].id, connection.id)
                self.assertEqual(app.workflow_buffer.document.connections[0].target.node_id, "other")
                self.assertEqual(app.workflow_buffer.document.nodes, graph.nodes)
                await pilot.press("ctrl+z")
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document.connections, (connection,))

    async def test_context_menu_targets_clicked_node_and_can_be_dismissed(self):
        graph = interaction_graph()
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory), size=(160, 40)) as (app, editor, pilot, _):
                canvas = editor.query_one(WorkflowCanvas)
                await click_node(pilot, canvas, "source")
                await pilot.mouse_down(offset=node_body(canvas, "edit"), button=3)
                await pilot.pause()
                self.assertIsInstance(app.screen, ChoiceMenu)
                self.assertEqual(canvas.selected_node_id, "edit")
                options = app.screen.query_one(OptionList)
                self.assertIsNotNone(options.get_option("variants"))
                await pilot.press("escape")
                self.assertEqual(app.workflow_buffer.document, graph)
                await menu_command(app, pilot, "delete", context=True)
                self.assertEqual(tuple(node.id for node in app.workflow_buffer.document.nodes), ("source", "other"))
                await pilot.press("ctrl+z")
                await pilot.pause()
                self.assertEqual(app.workflow_buffer.document, graph)

    async def test_resize_preserves_draft_and_drawer_leaves_full_height_canvas(self):
        with TemporaryDirectory() as directory:
            async with open_editor(interaction_graph(), Path(directory), size=(160, 40)) as (app, editor, pilot, _):
                canvas = editor.query_one(WorkflowCanvas)
                await click_node(pilot, canvas, "edit")
                inspector = editor.query_one(WorkflowInspector)
                value = property_widget(editor, PropertyInput, "steps")
                value.value = "31"
                for width, height in ((80, 24), (60, 20), (120, 40), (160, 50), (80, 24)):
                    await pilot.resize_terminal(width, height)
                    await pilot.pause()
                    self.assertIs(editor.query_one(WorkflowInspector), inspector)
                    self.assertIs(property_widget(editor, PropertyInput, "steps"), value)
                    self.assertEqual(value.value, "31")
                    self.assertGreaterEqual(canvas.size.height, height - 4)
                    self.assertTrue(editor.query_one("#workflow-run").visible)
                canvas.focus()
                await pilot.press("enter")
                await pilot.pause()
                panel = editor.query_one("#workflow-inspector-panel")
                self.assertEqual(panel.region.right, 80)
                self.assertEqual(panel.region.width, 80)
                close = editor.query_one("#workflow-inspector-close")
                self.assertLessEqual(close.region.right, panel.region.right)
                self.assertTrue(await pilot.click(close))
                await pilot.pause()
                self.assertTrue(canvas.has_focus)
                self.assertFalse(editor.query_one(WorkflowEditorBody).has_class("inspector-open"))

    async def test_input_keys_cannot_delete_nodes_or_undo_graph(self):
        graph = interaction_graph()
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory), size=(160, 40)) as (app, editor, pilot, _):
                canvas = editor.query_one(WorkflowCanvas)
                await click_node(pilot, canvas, "source")
                await pilot.press("shift+right")
                await pilot.pause()
                revision = app.workflow_buffer.revision
                value = property_widget(editor, PropertyInput, "path")
                value.focus()
                await pilot.press("end", "backspace", "home", "delete", "ctrl+z")
                await pilot.pause()
                self.assertEqual(value.value, "ource.pn")
                self.assertEqual(app.workflow_buffer.revision, revision)
                self.assertEqual(len(app.workflow_buffer.document.nodes), 3)
                canvas.focus()
                await pilot.press("delete")
                await pilot.pause()
                self.assertEqual(len(app.workflow_buffer.document.nodes), 2)

    async def test_connection_order_inserts_and_round_trips_in_one_undo(self):
        graph = interaction_graph()
        sources = tuple(ImageSourceNode(id=f"s{index}", title=f"Source {index}", layout=NodeLayout(x=4, y=2 + 8 * index))
                        for index in range(3))
        connections = tuple(WorkflowConnection(id=f"c{index}", source=NodePortRef(node_id=node.id, port="image"),
                            target=NodePortRef(node_id="edit", port="references"), order=index)
                            for index, node in enumerate(sources))
        graph = WorkflowGraph(name="Input order", nodes=(*sources, graph.node("edit")), connections=connections)
        with TemporaryDirectory() as directory:
            async with open_editor(graph, Path(directory), size=(160, 40)) as (app, editor, pilot, path):
                canvas = editor.query_one(WorkflowCanvas)
                # Selecting the wire via the canvas uses its real hit map.
                point = next((x, y) for y in range(canvas.virtual_size.height) for x in range(canvas.virtual_size.width)
                             if canvas._scene.node_at(Offset(x, y)) is None
                             and canvas._scene.wire_at(Offset(x, y)) == "c0"
                             and canvas._scene.input_near(Offset(x, y)) is None
                             and canvas._scene.output_near(Offset(x, y)) is None)
                await pilot.click(offset=screen_point(canvas, *point))
                await pilot.pause()
                self.assertEqual(canvas.selected_connection_id, "c0")
                order = editor.query_one(ConnectionOrderSelect)
                order.focus()
                await pilot.press("enter", "end", "enter")
                await pilot.pause()
                ordered = app.workflow_buffer.document.incoming_connections()["edit"]["references"]
                self.assertEqual(tuple(wire.id for wire in ordered), ("c1", "c2", "c0"))
                self.assertEqual(app.workflow_buffer.revision, 1)
                await pilot.press("ctrl+s")
                await pilot.pause()
                self.assertEqual(load_workflow_document(path), app.workflow_buffer.document)
                await menu_command(app, pilot, "undo")
                self.assertEqual(app.workflow_buffer.document, graph)
                await menu_command(app, pilot, "connect", context=True)
                self.assertIsInstance(app.screen, WorkflowConnectionDialog)
                dialog = app.screen
                self.assertEqual(dialog.query_one("#workflow-connect-source", Select).value, "s0:image")
                self.assertEqual(dialog.query_one("#workflow-connect-target", Select).value, "edit:references")
                await pilot.press("escape")

    async def test_stop_shortcut_reaches_process_owner_through_modal_menu(self):
        with TemporaryDirectory() as directory:
            async with open_editor(interaction_graph(), Path(directory), size=(80, 24)) as (app, editor, pilot, _):
                app._start_command(
                    [sys.executable, "-c", "import signal; signal.pause()"], directory,
                    action_button_id="workflow-run", idle_label="Run", error_title="Stop test", running_label=None,
                )
                editor.set_running(True)
                await pilot.pause()
                process = app.process
                self.assertIsNotNone(process)
                run = editor.query_one("#workflow-run", Button)
                self.assertEqual(str(run.label), "Stop")
                self.assertFalse(run.disabled)
                await pilot.press("f10")
                await pilot.pause()
                self.assertIsInstance(app.screen, ChoiceMenu)
                await pilot.press("shift+f5")
                for _ in range(100):
                    await pilot.pause(0.05)
                    if app.generation_worker is None:
                        break
                self.assertIsNone(app.generation_worker)
                self.assertEqual(process.returncode, -15)
                self.assertTrue(process.stdout.at_eof())
                await pilot.press("escape")
                self.assertEqual(str(run.label), "Run")

    async def test_inspector_menu_and_connect_dialog_have_visible_clickable_controls(self):
        with TemporaryDirectory() as directory:
            async with open_editor(interaction_graph(), Path(directory), size=(80, 24)) as (app, editor, pilot, _):
                canvas = editor.query_one(WorkflowCanvas)
                await click_node(pilot, canvas, "source")
                await pilot.press("enter")
                await pilot.pause()
                actions = editor.query("#workflow-inspector-actions Button")
                self.assertEqual([button.name for button in actions], ["context", "expand-inspector", "hide-inspector"])
                self.assertTrue(await pilot.click("#workflow-context"))
                await pilot.pause()
                menu = app.screen
                options = menu.query_one(OptionList)
                self.assertIsNotNone(options.get_option("results"))
                self.assertIsNotNone(options.get_option("run-target"))
                options.highlighted = options.get_option_index("connect")
                await pilot.press("enter")
                await pilot.pause()
                self.assertIsInstance(app.screen, WorkflowConnectionDialog)
                for button_id, label in (("workflow-connect-cancel", "Cancel"), ("workflow-connect-confirm", "Connect")):
                    button = app.screen.query_one(f"#{button_id}", Button)
                    self.assertGreaterEqual(button.content_region.height, 1)
                    self.assertIn(label, button.render_line(0).text)
                    self.assertLessEqual(button.region.bottom, app.screen.query_one("#workflow-connect-actions").region.bottom)
                self.assertTrue(await pilot.click("#workflow-connect-confirm"))
                await pilot.pause()
                self.assertIs(app.screen, editor)
                connection = app.workflow_buffer.document.connections[0]
                self.assertEqual(connection.source.node_id, "source")
                self.assertEqual(connection.target.node_id, "edit")

    async def test_search_node_picker_and_context_add_position(self):
        with TemporaryDirectory() as directory:
            async with open_editor(interaction_graph(), Path(directory), size=(80, 24)) as (app, editor, pilot, _):
                canvas = editor.query_one(WorkflowCanvas)
                await pilot.mouse_down(offset=screen_point(canvas, 8, 17), button=3)
                await pilot.pause()
                menu = app.screen
                self.assertIsInstance(menu, ChoiceMenu)
                panel = menu.query_one("#choice-menu-panel")
                self.assertLessEqual(panel.region.bottom, 24)
                self.assertLessEqual(panel.region.right, 80)
                options = menu.query_one(OptionList)
                options.highlighted = options.get_option_index("add")
                await pilot.press("enter")
                await pilot.pause()
                self.assertIsInstance(app.screen, ChoiceMenu)
                app.screen.query_one(Input).value = "image"
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                added = app.workflow_buffer.document.node(canvas.selected_node_id)
                self.assertIsInstance(added, ImageSourceNode)
                self.assertEqual(added.layout, NodeLayout(x=8, y=17))
                self.assertEqual(app.workflow_buffer.revision, 1)


if __name__ == "__main__":
    unittest.main()
