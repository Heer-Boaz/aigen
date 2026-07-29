from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container, ItemGrid
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select

from aigen.workflow_canvas import WorkflowCanvas
from aigen.workflow_edit_buffer import (
    WorkflowEditBuffer,
    WorkflowPropertyEditError,
)
from aigen.workflow_graph import (
    NodeKind,
    NodePortRef,
    WorkflowGraph,
    node_definition,
)
from aigen.workflow_inspector import (
    PropertyInput,
    PropertyRow,
    PropertySelect,
    WorkflowInspector,
)
from aigen.workflow_layout import NODE_WIDTH


class WorkflowEditorBody(Container):
    """Owns the responsive canvas and inspector split."""

    def on_resize(self, event: events.Resize) -> None:
        canvas = self.query_one(WorkflowCanvas)
        inspector = self.query_one(WorkflowInspector)
        horizontal_minimum = (
            self.styles.gutter.width
            + NODE_WIDTH
            + canvas.styles.gutter.width
            + inspector.horizontal_minimum_width
        )
        self.set_class(
            event.size.width < horizontal_minimum,
            "stacked",
        )


class WorkflowEditor(ModalScreen[None]):
    """Fullscreen visual editor for the persisted workflow graph."""

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

    WorkflowEditor #workflow-editor-title {
        width: 100%;
        height: 1;
        padding: 0 1;
        color: #d8c5eb;
        text-style: bold;
    }

    WorkflowEditor #workflow-node-toolbar,
    WorkflowEditor #workflow-document-toolbar {
        width: 100%;
        height: auto;
        grid-gutter: 0;
        padding: 0 1;
    }

    WorkflowEditor #workflow-node-kind {
        height: 1;
        min-width: 20;
        border: none;
        padding: 0;
    }

    WorkflowEditor Button {
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 1;
    }

    WorkflowEditor WorkflowEditorBody {
        layout: horizontal;
        width: 100%;
        height: 1fr;
        padding: 0 1;
    }

    WorkflowEditor WorkflowEditorBody.stacked {
        layout: vertical;
    }

    WorkflowEditor WorkflowEditorBody.stacked > WorkflowCanvas {
        width: 100%;
        height: 3fr;
    }

    WorkflowEditor WorkflowEditorBody.stacked > WorkflowInspector {
        width: 100%;
        min-width: 0;
        height: 2fr;
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
        def __init__(self, document: WorkflowGraph) -> None:
            super().__init__()
            self.document = document

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
    ) -> None:
        super().__init__()
        self._edit_buffer = edit_buffer
        self._running = False

    def compose(self) -> ComposeResult:
        with Container(id="workflow-editor-shell"):
            yield Label(self._title_text(), id="workflow-editor-title")
            yield ItemGrid(
                Select(
                    (
                        (node_definition(kind).label, kind.value)
                        for kind in NodeKind
                    ),
                    value=NodeKind.IMAGE_SOURCE.value,
                    allow_blank=False,
                    compact=True,
                    id="workflow-node-kind",
                    classes="workflow-edit-control",
                ),
                Button(
                    "+ Node",
                    id="workflow-add-node",
                    classes="workflow-edit-control",
                    compact=True,
                ),
                Button(
                    "Delete",
                    id="workflow-delete-node",
                    classes="workflow-edit-control",
                    compact=True,
                ),
                Button(
                    "Earlier",
                    id="workflow-connection-earlier",
                    classes="workflow-edit-control",
                    compact=True,
                ),
                Button(
                    "Later",
                    id="workflow-connection-later",
                    classes="workflow-edit-control",
                    compact=True,
                ),
                Button(
                    "Undo",
                    id="workflow-undo",
                    classes="workflow-edit-control",
                    compact=True,
                ),
                Button(
                    "Redo",
                    id="workflow-redo",
                    classes="workflow-edit-control",
                    compact=True,
                ),
                Button(
                    "Auto layout",
                    id="workflow-auto-layout",
                    classes="workflow-edit-control",
                    compact=True,
                ),
                min_column_width=10,
                stretch_height=False,
                regular=False,
                id="workflow-node-toolbar",
            )
            with WorkflowEditorBody(id="workflow-editor-body"):
                yield WorkflowCanvas(
                    self._edit_buffer.document,
                    id="workflow-canvas",
                )
                yield WorkflowInspector(
                    self._edit_buffer.document,
                    None,
                    None,
                    id="workflow-inspector",
                )
            yield ItemGrid(
                Button(
                    "Save",
                    id="workflow-save",
                    classes="workflow-edit-control",
                    compact=True,
                ),
                Button(
                    "Load",
                    id="workflow-load",
                    classes="workflow-edit-control",
                    compact=True,
                ),
                Button(
                    "Run",
                    id="workflow-run",
                    variant="primary",
                    compact=True,
                ),
                Button(
                    "Stop",
                    id="workflow-stop",
                    compact=True,
                    disabled=True,
                ),
                Button(
                    "Close",
                    id="workflow-close",
                    classes="workflow-run-locked-control",
                    compact=True,
                ),
                Button(
                    "Quit",
                    id="workflow-quit",
                    classes="workflow-run-locked-control",
                    compact=True,
                ),
                min_column_width=9,
                stretch_height=False,
                regular=False,
                id="workflow-document-toolbar",
            )
            yield Label("Ready", id="workflow-editor-status")

    def on_mount(self) -> None:
        self._update_history_actions()
        self.query_one(WorkflowCanvas).focus()

    def set_running(self, running: bool) -> None:
        self._running = running
        for control in self.query(
            ".workflow-edit-control, .workflow-run-locked-control"
        ):
            control.disabled = running
        self.query_one("#workflow-run", Button).disabled = running
        self.query_one("#workflow-stop", Button).disabled = not running
        self.query_one(WorkflowInspector).disabled = running
        self.query_one(WorkflowCanvas).set_editable(not running)
        self._update_history_actions()
        self._set_status("Workflow running" if running else "Ready")

    def set_runtime_statuses(self, statuses: dict[str, str]) -> None:
        self.query_one(WorkflowCanvas).set_runtime_statuses(statuses)

    def set_runtime_status(self, node_id: str, status: str) -> None:
        self.query_one(WorkflowCanvas).set_runtime_status(node_id, status)

    def set_status(self, message: str) -> None:
        self._set_status(message)

    async def show_replaced_document(self, status: str) -> None:
        canvas = self.query_one(WorkflowCanvas)
        canvas.set_runtime_statuses({})
        canvas.set_selection(None, None)
        await self._show_document()
        self._set_status(status)

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

    @on(WorkflowCanvas.SelectionChanged)
    async def canvas_selection_changed(
        self,
        event: WorkflowCanvas.SelectionChanged,
    ) -> None:
        if not await self.commit_pending_property():
            self.query_one(WorkflowCanvas).set_selection(
                event.previous_node_id,
                event.previous_connection_id,
            )
            return
        await self.query_one(WorkflowInspector).show(
            self._edit_buffer.document,
            event.node_id,
            event.connection_id,
        )
        self._update_history_actions()

    @on(WorkflowCanvas.NodeMoved)
    async def canvas_node_moved(
        self,
        event: WorkflowCanvas.NodeMoved,
    ) -> None:
        if not await self.commit_pending_property():
            self.query_one(WorkflowCanvas).set_document(
                self._edit_buffer.document
            )
            return
        if self._edit_buffer.move_node(
            event.node_id,
            x=event.x,
            y=event.y,
        ):
            await self._show_document()

    @on(WorkflowCanvas.ConnectionRequested)
    async def canvas_connection_requested(
        self,
        event: WorkflowCanvas.ConnectionRequested,
    ) -> None:
        if not await self.commit_pending_property():
            return
        await self._connect_ports(
            event.source,
            event.target,
            event.connection_id,
        )

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
            self._update_history_actions()

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
            inspector.focus_invalid_draft(error.edit)
            self.notify(str(error), severity="error")
            return False
        inspector.clear_drafts()
        await self._show_document()
        return True

    @on(Select.Changed)
    async def property_selected(self, event: Select.Changed) -> None:
        editor = event.select
        if not isinstance(editor, PropertySelect) or not editor.is_mounted:
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
        button_id = event.button.id
        browse_request: tuple[str, str, str] | None = None
        if (
            event.button.name is not None
            and event.button.has_class("workflow-property-browse")
        ):
            row = event.button.query_ancestor(PropertyRow)
            assert isinstance(row, PropertyRow)
            assert row.node_id is not None
            browse_request = (
                row.node_id,
                row.field_name,
                row.query_one(PropertyInput).value,
            )

        if (
            button_id
            not in {
                "workflow-stop",
                "workflow-load",
                "workflow-quit",
            }
            and not await self.commit_pending_property()
        ):
            return

        match button_id:
            case "workflow-add-node":
                await self._add_node()
            case "workflow-delete-node":
                await self._delete_selection()
            case "workflow-connection-earlier":
                await self._move_connection(-1)
            case "workflow-connection-later":
                await self._move_connection(1)
            case "workflow-undo":
                await self._undo()
            case "workflow-redo":
                await self._redo()
            case "workflow-auto-layout":
                await self._auto_layout()
            case "workflow-save":
                self._save()
            case "workflow-load":
                self.post_message(self.LoadRequested())
            case "workflow-run":
                self._run()
            case "workflow-stop":
                self.post_message(self.StopRequested())
            case "workflow-close":
                self.dismiss(None)
            case "workflow-quit":
                self.post_message(self.QuitRequested())
            case _:
                if browse_request is not None:
                    node_id, field_name, current_value = browse_request
                    self.post_message(
                        self.BrowseRequested(
                            node_id,
                            field_name,
                            current_value,
                        )
                    )

    async def _add_node(self) -> None:
        value = self.query_one("#workflow-node-kind", Select).value
        assert isinstance(value, str)
        canvas = self.query_one(WorkflowCanvas)
        x = int(canvas.scroll_offset.x + max(2, canvas.size.width // 2))
        y = int(canvas.scroll_offset.y + max(2, canvas.size.height // 2))
        node = self._edit_buffer.add_node(
            NodeKind(value),
            x=x,
            y=y,
        )
        canvas.set_selected_node(node.id)
        await self._show_document()

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

    async def _move_connection(self, direction: int) -> None:
        connection_id = self.query_one(
            WorkflowCanvas
        ).selected_connection_id
        if connection_id is None:
            return
        if self._edit_buffer.move_connection(connection_id, direction):
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

    def _run(self) -> None:
        self.post_message(self.RunRequested(self._edit_buffer.document))

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
        await self.query_one(WorkflowInspector).show(
            self._edit_buffer.document,
            canvas.selected_node_id,
            canvas.selected_connection_id,
        )
        self.query_one("#workflow-editor-title", Label).update(
            self._title_text()
        )
        self._update_history_actions()

    def _update_history_actions(self) -> None:
        self.query_one("#workflow-undo", Button).disabled = (
            self._running or not self._edit_buffer.can_undo
        )
        self.query_one("#workflow-redo", Button).disabled = (
            self._running or not self._edit_buffer.can_redo
        )
        canvas = self.query_one(WorkflowCanvas)
        self.query_one("#workflow-delete-node", Button).disabled = (
            self._running
            or (
                canvas.selected_node_id is None
                and canvas.selected_connection_id is None
            )
        )
        can_move_earlier, can_move_later = self._connection_move_states()
        self.query_one(
            "#workflow-connection-earlier",
            Button,
        ).disabled = self._running or not can_move_earlier
        self.query_one(
            "#workflow-connection-later",
            Button,
        ).disabled = self._running or not can_move_later

    def _connection_move_states(self) -> tuple[bool, bool]:
        connection_id = self.query_one(
            WorkflowCanvas
        ).selected_connection_id
        if connection_id is None:
            return False, False
        return self._edit_buffer.connection_move_capabilities(
            connection_id
        )

    def _set_status(self, message: str) -> None:
        self.query_one("#workflow-editor-status", Label).update(message)

    def _title_text(self) -> str:
        path = (
            self._edit_buffer.document_path.as_posix()
            if self._edit_buffer.document_path is not None
            else "Unsaved"
        )
        dirty = " *" if self._edit_buffer.dirty else ""
        return (
            f"Workflow — {self._edit_buffer.document.name} — "
            f"{path}{dirty}"
        )
