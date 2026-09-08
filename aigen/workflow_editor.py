from __future__ import annotations

from pathlib import Path
import asyncio

from pydantic import ValidationError
from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.geometry import Offset
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, TextArea

from aigen.workflow_canvas import WorkflowCanvas
from aigen.workflow_edit_buffer import (
    WorkflowEditBuffer,
    WorkflowPropertyEditError,
)
from aigen.workflow_graph import (
    ImageEditNode,
    ImageSelectionNode,
    ImageResultReference,
    NodeKind,
    NodePortRef,
    WorkflowGraph,
    node_definition,
)
from aigen.workflow_inspector import (
    ConnectionOrderSelect,
    WorkflowInspector,
)
from aigen.workflow_property_widgets import (
    PropertyInput,
    PropertyRow,
    PropertySelect,
    PropertyTextArea,
)
from aigen.workflow_editor_layout import InspectorDivider, WorkflowEditorBody
from aigen.workflow_run_state import WorkflowRunState
from aigen.workflow_results_tui import WorkflowResults
from aigen.workflow_results import has_viewable_output
from aigen.tui_dialogs import PromptDialog
from aigen.tui_action_button import ActionButton
from aigen.workflow_commands import DEFAULT_WORKFLOW_RUNS_ROOT
from aigen.workflow_connection_dialog import WorkflowConnectionDialog
from aigen.tui_choice_menu import ChoiceMenu, MenuChoice
from aigen.manifest_io import ManifestIOError
from aigen.workflow_task import next_workflow_step, project_results, read_workflow_runs


class WorkflowEditor(ModalScreen[None]):
    """Fullscreen visual editor for the persisted workflow graph."""

    BINDINGS = [
        Binding("ctrl+s", "command('save')", show=False),
        Binding("ctrl+enter", "commit_properties", show=False),
        Binding("ctrl+o", "command('load')", show=False),
        Binding("ctrl+z", "command('undo')", show=False),
        Binding("ctrl+y,ctrl+shift+z", "command('redo')", show=False),
        Binding("insert", "command('add')", show=False),
        Binding("delete,backspace", "command('delete')", show=False),
        Binding("f5", "command('run')", show=False),
        Binding("f10", "command('menu')", show=False),
        Binding("shift+f10", "command('context')", show=False),
        Binding("enter", "command('inspect')", show=False),
        Binding("escape", "close_inspector", show=False),
    ]

    DEFAULT_CSS = """
    WorkflowEditor {
        width: 100%;
        height: 100%;
        background: #100d16;
    }

    WorkflowEditor #workflow-editor-shell {
        width: 100%;
        height: 100%;
        background: #100d16;
    }

    WorkflowEditor #workflow-command-bar {
        height: 1;
        padding: 0 1;
        background: #211a2d;
    }

    WorkflowEditor #workflow-editor-title {
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: #d8c5eb;
        text-style: bold;
        text-overflow: ellipsis;
    }

    WorkflowEditor Button {
        width: auto;
        min-width: 3;
        height: 1;
        min-height: 1;
        border: none;
        padding: 0;
    }

    WorkflowEditor WorkflowEditorBody {
        layout: horizontal;
        width: 100%;
        height: 1fr;
        padding: 0;
    }

    WorkflowEditor #workflow-inspector-panel {
        height: 1fr;
        background: #1c1724;
    }

    WorkflowEditor #workflow-inspector-actions {
        height: 1;
        padding: 0 1;
    }

    WorkflowEditor #workflow-inspector-title {
        width: 1fr;
        height: 1;
        color: #d8c5eb;
    }

    WorkflowEditor WorkflowInspector {
        width: 100%;
        min-width: 0;
        height: 1fr;
    }

    WorkflowEditor #workflow-editor-status {
        width: 100%;
        height: 1;
        padding: 0 1;
        color: #b9adc8;
        text-overflow: ellipsis;
    }
    """

    class SaveRequested(Message):
        pass

    class LoadRequested(Message):
        pass

    class RunRequested(Message):
        def __init__(self, document: WorkflowGraph, target_node_ids: tuple[str, ...] | None = None) -> None:
            super().__init__()
            self.document = document
            self.target_node_ids = target_node_ids

    class StopRequested(Message):
        pass

    class QuitRequested(Message):
        pass

    class BrowseRequested(Message):
        def __init__(
            self,
            node_id: str,
            field_name: str,
            current_value: str,
        ) -> None:
            super().__init__()
            self.node_id = node_id
            self.field_name = field_name
            self.current_value = current_value

    def __init__(
        self,
        edit_buffer: WorkflowEditBuffer,
        run_state: WorkflowRunState,
        runs_root: Path = DEFAULT_WORKFLOW_RUNS_ROOT,
    ) -> None:
        super().__init__()
        self._edit_buffer = edit_buffer
        self._run_state = run_state
        self._running = False
        self._runs_root = runs_root
        self._has_drafts = False
        self._records = ()
        self._history_loaded = False
        self._history_error: str | None = None
        self._task_document: WorkflowGraph | None = None
        self._latest_results: dict[str, Path] = {}
        self._applicable_results: dict[str, Path] = {}
        self._result_nodes: tuple[str, ...] = ()
        self._next_step = next_workflow_step(edit_buffer.document, {})

    def compose(self) -> ComposeResult:
        with Container(id="workflow-editor-shell"):
            with Horizontal(id="workflow-command-bar"):
                yield Button("Menu", name="menu", id="workflow-menu", compact=True, tooltip="Workflow commands · F10")
                yield Label(self._title_text(), id="workflow-editor-title", markup=False)
                yield Button("+ Node", name="add", id="workflow-add-node", compact=True, tooltip="Find a node · Insert")
                yield Button("Inspect", name="inspect", id="workflow-inspect", compact=True, tooltip="Show properties · Enter")
                yield Button("Results", name="view-results", id="workflow-results", compact=True)
                yield ActionButton("Run", name="run", id="workflow-run", variant="primary", compact=True, tooltip="Run workflow · F5")
            with WorkflowEditorBody(id="workflow-editor-body"):
                yield WorkflowCanvas(
                    self._edit_buffer.document,
                    prepare_interaction=self.commit_pending_property,
                    selection_changed=self._selection_changed,
                    move_node=self._move_node,
                    connect_ports=self._connect_ports,
                    id="workflow-canvas",
                )
                yield InspectorDivider()
                with Container(id="workflow-inspector-panel"):
                    with Horizontal(id="workflow-inspector-actions"):
                        yield Label("Properties", id="workflow-inspector-title")
                        yield Button("⋯", name="context", id="workflow-context", compact=True,
                                     tooltip="Selection actions · Shift+F10")
                        yield ActionButton("Expand", name="expand-inspector", id="workflow-inspector-expand", compact=True)
                        yield Button("×", name="hide-inspector", id="workflow-inspector-close", compact=True,
                                     tooltip="Close properties · Escape")
                    yield WorkflowInspector(self._edit_buffer.document, None, None, id="workflow-inspector")
            yield Label("Ready", id="workflow-editor-status")

    def on_mount(self) -> None:
        self._update_actions()
        self._set_status(self._selection_hint())
        self.query_one(WorkflowCanvas).focus()
        self.refresh_result_history()

    def set_running(self, running: bool) -> None:
        finished = self._running and not running
        self._running = running
        self.query_one(WorkflowInspector).disabled = running
        self.query_one(WorkflowCanvas).set_editable(not running)
        self._update_actions()
        self._set_status("Workflow running · Shift+F5: stop" if running else self._selection_hint())
        if finished:
            self.refresh_result_history()

    @work(exclusive=True, group="workflow-result-history")
    async def refresh_result_history(self) -> None:
        workflow_id = self._edit_buffer.document.workflow_id
        self._history_loaded = False
        self._history_error = None
        self._update_actions()
        try:
            records = await asyncio.to_thread(read_workflow_runs, self._runs_root, workflow_id)
        except (OSError, ValueError, KeyError, ManifestIOError) as error:
            self._history_error = str(error)
        else:
            self._records = records
            self._history_loaded = True
            self._task_document = None
            self._project_task()
        self._update_actions()

    def _project_task(self) -> None:
        document = self._edit_buffer.document
        if document is not self._task_document:
            self._latest_results, self._applicable_results = project_results(document, self._records)
            self._result_nodes = tuple(node.id for node in document.nodes if has_viewable_output(node) and (
                node.id in self._latest_results or isinstance(node, ImageSelectionNode) and node.config.selected is not None))
            self._next_step = next_workflow_step(document, self._applicable_results)
            self._task_document = document

    def refresh_runtime_statuses(self) -> None:
        self.query_one(WorkflowCanvas).set_runtime_statuses(
            self._run_state.project(self._edit_buffer.document)
        )

    def refresh_runtime_status(self, node_id: str) -> None:
        self.query_one(WorkflowCanvas).set_runtime_status(
            node_id, self._run_state.status(node_id),
        )

    def set_status(self, message: str) -> None:
        self._set_status(message)

    async def show_replaced_document(self, status: str) -> None:
        self._records = ()
        self._latest_results = {}
        self._applicable_results = {}
        self._result_nodes = ()
        self._history_loaded = False
        canvas = self.query_one(WorkflowCanvas)
        canvas.set_runtime_statuses({})
        canvas.set_selection(None, None)
        await self._show_document()
        self._set_status(status)
        self.refresh_result_history()

    def document_saved(self) -> None:
        path = self._edit_buffer.document_path
        assert path is not None
        self.query_one("#workflow-editor-title", Label).update(
            self._title_text()
        )
        self._set_status(f"Saved {path}")

    async def apply_browsed_path(
        self,
        node_id: str,
        field_name: str,
        path: Path,
    ) -> None:
        await self._update_config_field(
            node_id,
            field_name,
            path.as_posix(),
        )

    async def _selection_changed(
        self,
        node_id: str | None,
        connection_id: str | None,
    ) -> None:
        await self.query_one(WorkflowInspector).show(
            self._edit_buffer.document,
            node_id,
            connection_id,
        )
        self._update_actions()
        if not self._running:
            self._set_status(self._selection_hint())

    async def _move_node(
        self,
        node_id: str,
        x: int,
        y: int,
    ) -> None:
        if self._edit_buffer.move_node(
            node_id,
            x=x,
            y=y,
        ):
            await self._show_document()

    async def _connect_ports(
        self,
        source: NodePortRef,
        target: NodePortRef,
        connection_id: str | None = None,
    ) -> None:
        revision = self._edit_buffer.revision
        try:
            connection = (
                self._edit_buffer.connect_ports(source, target)
                if connection_id is None
                else self._edit_buffer.reconnect_connection(
                    connection_id,
                    source,
                    target,
                )
            )
        except (ValidationError, ValueError) as error:
            self.notify(str(error), severity="error")
            return
        canvas = self.query_one(WorkflowCanvas)
        canvas.set_selection(None, connection.id)
        if self._edit_buffer.revision != revision:
            await self._show_document()
        else:
            await self.query_one(WorkflowInspector).show(
                self._edit_buffer.document,
                None,
                connection.id,
            )
            self._update_actions()

    @on(Input.Submitted)
    async def property_submitted(self, event: Input.Submitted) -> None:
        editor = event.input
        if not isinstance(editor, PropertyInput):
            return
        await self.commit_pending_property()

    async def commit_pending_property(self) -> bool:
        inspector = self.query_one(WorkflowInspector)
        drafts = inspector.property_drafts()
        if not drafts:
            return True
        try:
            self._edit_buffer.update_properties(drafts)
        except WorkflowPropertyEditError as error:
            self.query_one(WorkflowEditorBody).show_inspector(True)
            inspector.focus_invalid_draft(error.edit)
            self.notify(str(error), severity="error")
            return False
        inspector.accept_properties()
        await self._show_document()
        return True

    async def action_commit_properties(self) -> None:
        await self.commit_pending_property()

    @on(WorkflowInspector.DraftsChanged)
    def drafts_changed(self, event: WorkflowInspector.DraftsChanged) -> None:
        self._has_drafts = event.modified
        self.query_one("#workflow-editor-title", Label).update(self._title_text())

    @on(Select.Changed)
    async def property_selected(self, event: Select.Changed) -> None:
        editor = event.select
        if isinstance(editor, ConnectionOrderSelect):
            if not editor.is_mounted or event.value == editor.position:
                return
            if not await self.commit_pending_property():
                editor.value = editor.position
                return
            if self._edit_buffer.reorder_connection(editor.connection_id, int(event.value)):
                await self._show_document()
            return
        if not isinstance(editor, PropertySelect) or not editor.is_mounted:
            return
        if event.value == editor.original_value:
            return
        if not await self.commit_pending_property():
            editor.value = editor.original_value
            return
        node = self._edit_buffer.document.node(editor.node_id)
        if getattr(node.config, editor.field_name) != editor.original_value:
            return
        await self._update_config_field(
            editor.node_id,
            editor.field_name,
            event.value,
        )

    @on(Button.Pressed)
    async def button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.has_class("workflow-property-browse"):
            row = event.button.query_ancestor(PropertyRow)
            assert isinstance(row, PropertyRow)
            assert row.node_id is not None
            node_id, field_name = row.node_id, row.field_name
            value = row.query_one(PropertyInput).value
            if await self.commit_pending_property():
                self.post_message(self.BrowseRequested(node_id, field_name, value))
            return
        command = event.button.name
        if command == "context":
            await self.query_one(WorkflowCanvas).action_context_menu(
                Offset(event.button.region.x, event.button.region.bottom),
            )
            return
        if command == "run" and self._running:
            command = "stop"
        if command is not None:
            await self.action_command(command)

    async def action_command(self, command: str) -> None:
        if not self._command_enabled(command):
            return
        if command not in {"menu", "context", "inspect", "hide-inspector", "expand-inspector", "stop", "load", "quit"}:
            if not await self.commit_pending_property():
                return
        canvas = self.query_one(WorkflowCanvas)
        match command:
            case "menu":
                button = self.query_one("#workflow-menu")
                self.app.push_screen(
                    ChoiceMenu("Workflow", self._document_choices(),
                               anchor=Offset(button.region.x, button.region.bottom)),
                    self._menu_chosen,
                )
            case "context":
                await canvas.action_context_menu()
            case "add":
                self._choose_node()
            case "inspect":
                self._show_inspector()
            case "expand-inspector":
                body = self.query_one(WorkflowEditorBody)
                body.expand_inspector(not body.expanded)
            case "hide-inspector":
                self.query_one(WorkflowEditorBody).show_inspector(False)
                canvas.focus()
            case "delete":
                await self._delete_selection()
            case "undo":
                await self._undo()
            case "redo":
                await self._redo()
            case "layout":
                await self._auto_layout()
            case "save":
                self._save()
            case "load":
                self.post_message(self.LoadRequested())
            case "run":
                await self._run()
            case "run-target":
                self.post_message(self.RunRequested(self._edit_buffer.document, (canvas.selected_node_id,)))
            case "run-again":
                self.post_message(self.RunRequested(self._edit_buffer.document))
            case "variants":
                self._variants()
            case "results":
                self._open_node_results(canvas.selected_node_id)
            case "view-results":
                self._view_results()
            case "refresh-results":
                self.refresh_result_history()
            case "connect":
                self._choose_connection()
            case "stop":
                self.post_message(self.StopRequested())
            case "close":
                self.dismiss(None)
            case "quit":
                self.post_message(self.QuitRequested())

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "commit_properties":
            return isinstance(self.focused, PropertyTextArea)
        if action == "command":
            command = str(parameters[0])
            if command in {"undo", "redo"} and isinstance(self.focused, (Input, TextArea)):
                return False
            if command in {"delete", "inspect", "add"} and not self.query_one(WorkflowCanvas).has_focus:
                return False
            return self._command_enabled(command)
        if action == "close_inspector":
            return True
        return super().check_action(action, parameters)

    def _command_enabled(self, command: str) -> bool:
        canvas = self.query_one(WorkflowCanvas)
        node_id = canvas.selected_node_id
        if command == "stop":
            return self._running
        if command in {"view-results", "refresh-results"}:
            return command == "refresh-results" or bool(self._result_nodes)
        if command == "results":
            return node_id is not None and has_viewable_output(self._edit_buffer.document.node(node_id))
        if command in {"menu", "context", "inspect", "hide-inspector", "expand-inspector", "quit"}:
            return True
        if self._running:
            return False
        if command == "undo":
            return self._edit_buffer.can_undo
        if command == "redo":
            return self._edit_buffer.can_redo
        if command == "delete":
            return node_id is not None or canvas.selected_connection_id is not None
        if command == "connect":
            return canvas.can_connect_selection()
        if command == "variants":
            return node_id is not None and isinstance(self._edit_buffer.document.node(node_id), ImageEditNode)
        if command == "run-target":
            return node_id is not None
        if command == "run":
            return bool(self._edit_buffer.document.nodes) and (self._history_loaded or self._history_error is not None)
        if command == "run-again":
            return self._history_loaded and self._next_step.action == "results"
        if command == "layout":
            return bool(self._edit_buffer.document.nodes)
        return True

    def _document_choices(self) -> tuple[MenuChoice, ...]:
        return tuple(
            MenuChoice(command, label, shortcut, not self._command_enabled(command))
            for command, label, shortcut in (
                ("save", "Save workflow", "Ctrl+S"),
                ("load", "Open workflow…", "Ctrl+O"),
                ("undo", "Undo", "Ctrl+Z"),
                ("redo", "Redo", "Ctrl+Y"),
                ("layout", "Arrange nodes", ""),
                ("inspect", "Properties", "Enter"),
                ("view-results", "Browse result history", ""),
                ("run-again", "Run again", ""),
                ("refresh-results", "Refresh results", ""),
                ("close", "Close editor", ""),
                ("quit", "Quit application", "Ctrl+C"),
            )
        )

    def _context_choices(self) -> tuple[MenuChoice, ...]:
        canvas = self.query_one(WorkflowCanvas)
        if canvas.selected_node_id is not None:
            choices = [("inspect", "Properties", "Enter"), ("run-target", "Run to here", "")]
            if self._command_enabled("results"):
                choices.append(("results", "Results", ""))
            if isinstance(self._edit_buffer.document.node(canvas.selected_node_id), ImageEditNode):
                choices.append(("variants", "Seed variants…", ""))
            choices.extend((("connect", "Connect…", ""), ("delete", "Delete node", "Delete")))
        elif canvas.selected_connection_id is not None:
            choices = [("inspect", "Connection properties", "Enter"), ("connect", "Reconnect…", ""),
                       ("delete", "Disconnect", "Delete")]
        else:
            choices = [("add", "Add node…", "Insert"), ("layout", "Arrange nodes", ""),
                       ("inspect", "Workflow properties", "Enter")]
        return tuple(
            MenuChoice(command, label, shortcut, not self._command_enabled(command))
            for command, label, shortcut in choices
        )

    @on(WorkflowCanvas.ContextRequested)
    def context_requested(self, event: WorkflowCanvas.ContextRequested) -> None:
        canvas = self.query_one(WorkflowCanvas)
        if canvas.selected_node_id is not None:
            title = self._edit_buffer.document.node(canvas.selected_node_id).title
        else:
            title = "Connection" if canvas.selected_connection_id is not None else "Canvas"

        async def chosen(command: str | None) -> None:
            if command == "add":
                self._choose_node(event.position)
            elif command is not None:
                await self.action_command(command)
        self.app.push_screen(ChoiceMenu(title, self._context_choices(), anchor=event.anchor), chosen)

    async def _menu_chosen(self, command: str | None) -> None:
        if command is not None:
            await self.action_command(command)

    def _choose_node(self, position: Offset | None = None) -> None:
        async def chosen(value: str | None) -> None:
            if value is not None:
                await self._add_node(NodeKind(value), position)
        self.app.push_screen(
            ChoiceMenu("Add node", tuple(MenuChoice(kind.value, node_definition(kind).label) for kind in NodeKind),
                       searchable=True),
            chosen,
        )

    async def _add_node(self, kind: NodeKind, position: Offset | None) -> None:
        canvas = self.query_one(WorkflowCanvas)
        if position is None:
            position = canvas.scroll_offset + Offset(
                max(2, (canvas.size.width - NODE_WIDTH) // 2), max(2, canvas.size.height // 2),
            )
        node = self._edit_buffer.add_node(
            kind, x=max(0, position.x), y=max(0, position.y),
        )
        canvas.set_selected_node(node.id)
        await self._show_document()

    def _choose_connection(self) -> None:
        canvas = self.query_one(WorkflowCanvas)
        connection_id = canvas.selected_connection_id
        connection = next((wire for wire in self._edit_buffer.document.connections if wire.id == connection_id), None)
        dialog = WorkflowConnectionDialog(self._edit_buffer.document, canvas.selected_node_id, connection=connection)
        if not dialog.can_connect:
            self.notify("Add a node with a compatible input before connecting.")
            return
        async def chosen(endpoints: tuple[NodePortRef, NodePortRef] | None) -> None:
            if endpoints is not None:
                await self._connect_ports(*endpoints, connection_id=connection_id)
        self.app.push_screen(dialog, chosen)

    async def _delete_selection(self) -> None:
        canvas = self.query_one(WorkflowCanvas)
        if canvas.selected_node_id is not None:
            changed = self._edit_buffer.delete_node(
                canvas.selected_node_id
            )
        elif canvas.selected_connection_id is not None:
            changed = self._edit_buffer.delete_connection(
                canvas.selected_connection_id
            )
        else:
            return
        if changed:
            canvas.set_selection(None, None)
            await self._show_document()

    async def _undo(self) -> None:
        label = self._edit_buffer.undo_label
        if self._edit_buffer.undo():
            await self._show_document()
            self._set_status(f"Undid {label}")

    async def _redo(self) -> None:
        label = self._edit_buffer.redo_label
        if self._edit_buffer.redo():
            await self._show_document()
            self._set_status(f"Redid {label}")

    async def _auto_layout(self) -> None:
        if self._edit_buffer.auto_layout():
            await self._show_document()

    def _save(self) -> None:
        self.post_message(self.SaveRequested())

    async def _run(self) -> None:
        if self._history_error is not None:
            self.refresh_result_history()
            return
        self._project_task()
        step = self._next_step
        if step.action == "choose":
            self._open_node_results(step.selection_id, step.result_path)
        elif step.action == "results":
            self._view_results(step.targets)
        elif step.action == "connect":
            self.query_one(WorkflowCanvas).set_selected_node(step.selection_id)
            await self._selection_changed(step.selection_id, None)
            self._show_inspector()
        else:
            self.post_message(self.RunRequested(self._edit_buffer.document, step.targets))

    def _open_node_results(self, node_id: str, initial_path: Path | None = None) -> None:
        self.app.push_screen(WorkflowResults(self._edit_buffer.document, node_id, self._runs_root,
                                            initial_path=initial_path), self._image_selected)

    def _view_results(self, completed: tuple[str, ...] | None = None) -> None:
        document = self._edit_buffer.document
        candidates = tuple(document.node(node_id) for node_id in (completed if completed is not None else self._result_nodes))
        def open_result(node_id: str | None) -> None:
            if node_id is not None:
                self._open_node_results(node_id, self._applicable_results[node_id] if completed is not None else None)

        selected = self.query_one(WorkflowCanvas).selected_node_id
        if selected in {node.id for node in candidates}:
            open_result(selected)
        elif len(candidates) == 1:
            open_result(candidates[0].id)
        else:
            choices = tuple(MenuChoice(node.id, node.title + (" · earlier settings" if node.id not in self._applicable_results
                                and not isinstance(node, ImageSelectionNode) else "")) for node in candidates)
            self.app.push_screen(ChoiceMenu("View results", choices), open_result)

    def _variants(self) -> None:
        node_id = self.query_one(WorkflowCanvas).selected_node_id
        if node_id is None or not isinstance(self._edit_buffer.document.node(node_id), ImageEditNode):
            self.notify("Select an Image edit node to create seed variants.")
            return
        node = self._edit_buffer.document.node(node_id)
        seeds = ", ".join(str(node.config.seed + offset) for offset in range(3))
        self.app.push_screen(PromptDialog("Seed variants", "Distinct integer seeds, separated by commas", seeds),
                             lambda value: self._variants_entered(node_id, value))

    def _variants_entered(self, node_id: str, value: str | None) -> None:
        if value is None:
            return
        try:
            seeds = tuple(int(part.strip()) for part in value.split(","))
            collection_id, _ = self._edit_buffer.create_image_variants(node_id, seeds)
        except ValueError as error:
            self.notify(str(error), severity="error")
            return
        self.query_one(WorkflowCanvas).set_selected_node(collection_id)
        self.run_worker(self._show_document())
        self._set_status("Variants created. Generate choices to compare and choose an output.")

    def _image_selected(self, selection: tuple[str, ImageResultReference] | None) -> None:
        if selection is None:
            return
        node_id, reference = selection
        self._edit_buffer.select_image(node_id, reference)
        self.query_one(WorkflowCanvas).set_selected_node(node_id)
        self.run_worker(self._show_document())
        self.post_message(self.SaveRequested())

    async def _update_config_field(
        self,
        node_id: str,
        field_name: str,
        raw_value: object,
    ) -> bool:
        try:
            changed = self._edit_buffer.update_node_config(
                node_id,
                field_name,
                raw_value,
            )
        except (ValidationError, ValueError) as error:
            self.notify(str(error), severity="error")
            return False
        if changed:
            await self._show_document()
        return True

    async def _show_document(self) -> None:
        canvas = self.query_one(WorkflowCanvas)
        canvas.set_document(self._edit_buffer.document)
        self.refresh_runtime_statuses()
        await self.query_one(WorkflowInspector).show(
            self._edit_buffer.document,
            canvas.selected_node_id,
            canvas.selected_connection_id,
        )
        self.query_one("#workflow-editor-title", Label).update(
            self._title_text()
        )
        self._update_actions()

    @on(WorkflowEditorBody.LayoutChanged)
    def layout_changed(self) -> None:
        self._update_actions()

    @on(events.DescendantFocus)
    def property_focused(self, event: events.DescendantFocus) -> None:
        if not self._running and isinstance(event.widget, (WorkflowCanvas, PropertyInput, PropertySelect, PropertyTextArea)):
            self._set_status(self._selection_hint())

    def _show_inspector(self) -> None:
        self.query_one(WorkflowEditorBody).show_inspector(True)
        self.call_after_refresh(self.query_one(WorkflowInspector).focus)

    async def action_close_inspector(self) -> None:
        body = self.query_one(WorkflowEditorBody)
        if body.expanded:
            body.expand_inspector(False)
        elif body.has_class("inspector-open"):
            body.show_inspector(False)
            self.query_one(WorkflowCanvas).focus()
        else:
            await self.action_command("close")

    def _update_actions(self) -> None:
        self._project_task()
        body = self.query_one(WorkflowEditorBody)
        run = self.query_one("#workflow-run", Button)
        run.label = "Stop" if self._running else "Refresh results" if self._history_error else self._next_step.label
        run.variant = "error" if self._running else "primary"
        run.disabled = not self._command_enabled("stop" if self._running else "run")
        run.tooltip = "Stop workflow · Shift+F5" if self._running else self._next_step.description + " · F5"
        self.query_one("#workflow-results", Button).disabled = not self._command_enabled("view-results")
        self.query_one("#workflow-results").display = self._running or self._next_step.action != "results"
        if not self._running:
            self._set_status(self._selection_hint())
        self.query_one("#workflow-add-node", Button).disabled = self._running
        canvas = self.query_one(WorkflowCanvas)
        body.set_context(canvas.selected_node_id is not None or canvas.selected_connection_id is not None)
        self.query_one("#workflow-inspect").display = not body.has_class("inspector-open")
        self.query_one("#workflow-inspector-expand", Button).label = "Restore" if body.expanded else "Expand"
        self.refresh_bindings()

    def _set_status(self, message: str) -> None:
        self.query_one("#workflow-editor-status", Label).update(message)

    def _selection_hint(self) -> str:
        if self._history_error:
            return f"Cannot read results: {self._history_error}"
        if isinstance(self.focused, PropertyTextArea):
            return "Enter: new line · Ctrl+Enter: apply · Ctrl+S: save · Tab: next field"
        canvas = self.query_one(WorkflowCanvas)
        if canvas.selected_connection_id is not None:
            return "Enter: input properties · Right-click / Shift+F10: reconnect or disconnect"
        if canvas.selected_node_id is not None:
            return self._next_step.description + " · Enter: properties · Shift+F10: actions"
        return self._next_step.description if self._history_loaded else "Reading recorded results…"

    def _title_text(self) -> str:
        dirty = " *" if self._edit_buffer.dirty or self._has_drafts else ""
        return f"{self._edit_buffer.document.name}{dirty}"
