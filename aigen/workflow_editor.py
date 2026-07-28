from __future__ import annotations

import uuid
from enum import Enum
from graphlib import TopologicalSorter
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, ItemGrid, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from aigen.generation.animegen_i2v import (
    ANIMEGEN_PRECISIONS,
    ANIMEGEN_SAMPLINGS,
    animegen_sampling_profile,
)
from aigen.generation.image_batch_postprocess import (
    image_batch_postprocess_model_names,
)
from aigen.generation.image_edit import (
    IMAGE_EDIT_BACKENDS,
    image_edit_backend_settings,
)
from aigen.workflow_canvas import NODE_WIDTH, WorkflowCanvas
from aigen.workflow_compilation import CompiledWorkflow, compile_workflow
from aigen.workflow_document_io import save_workflow_document
from aigen.workflow_graph import (
    AnimeGenI2VNode,
    FramePostprocessNode,
    ImageEditNode,
    ImagePostprocessNode,
    NodeKind,
    NodeLayout,
    NodePortRef,
    WorkflowConnection,
    WorkflowGraph,
    WorkflowNode,
    VosrPostprocessConfig,
    node_definition,
)
from aigen.workflow_templates import (
    create_workflow_node,
    postprocess_config_for_model,
)

_PRESERVE_SELECTION = object()


class PropertyInput(Input):
    def __init__(
        self,
        value: str,
        *,
        node_id: str | None,
        field_name: str,
    ) -> None:
        super().__init__(value, compact=True, classes="workflow-property-editor")
        self.node_id = node_id
        self.field_name = field_name
        self.original_value = value


class PropertySelect(Select[object]):
    def __init__(
        self,
        options: tuple[tuple[str, object], ...],
        value: object,
        *,
        node_id: str,
        field_name: str,
    ) -> None:
        super().__init__(
            options,
            value=value,
            allow_blank=False,
            compact=True,
            classes="workflow-property-editor",
        )
        self.node_id = node_id
        self.field_name = field_name
        self.original_value = value


class PropertyRow(Horizontal):
    def __init__(
        self,
        *,
        node_id: str | None,
        field_name: str,
        label: str,
        value: object,
        annotation: object,
        browse: bool = False,
        options: tuple[tuple[str, object], ...] | None = None,
    ) -> None:
        super().__init__(classes="workflow-property-row")
        self.node_id = node_id
        self.field_name = field_name
        self.label_text = label
        self.value = value
        self.annotation = annotation
        self.browse = browse
        self.options = options

    def compose(self) -> ComposeResult:
        yield Label(self.label_text, classes="workflow-property-label")
        options = (
            self.options
            if self.options is not None
            else _property_options(self.annotation)
        )
        if options is None:
            yield PropertyInput(
                "" if self.value is None else str(self.value),
                node_id=self.node_id,
                field_name=self.field_name,
            )
        else:
            yield PropertySelect(
                options,
                self.value,
                node_id=self.node_id or "",
                field_name=self.field_name,
            )
        if self.browse:
            yield Button(
                "Browse",
                name=self.field_name,
                compact=True,
                classes="workflow-property-browse",
            )


class WorkflowInspector(VerticalScroll):
    DEFAULT_CSS = """
    WorkflowInspector {
        width: 1fr;
        min-width: 24;
        height: 1fr;
        border: solid #5b496d;
        background: #1c1724;
        padding: 0 1;
        scrollbar-size-vertical: 1;
    }

    WorkflowInspector .workflow-inspector-heading {
        height: 1;
        text-style: bold;
        color: #d8c5eb;
    }

    WorkflowInspector .workflow-inspector-kind {
        height: 1;
        color: #9e8cad;
        margin-bottom: 1;
    }

    WorkflowInspector .workflow-property-row {
        width: 100%;
        height: 1;
    }

    WorkflowInspector .workflow-property-label {
        width: 12;
        min-width: 8;
        height: 1;
        content-align-vertical: middle;
        text-overflow: ellipsis;
    }

    WorkflowInspector .workflow-property-editor {
        width: 1fr;
        height: 1;
        border: none;
        padding: 0;
    }

    WorkflowInspector .workflow-property-browse {
        width: 8;
        min-width: 8;
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 1;
    }

    WorkflowInspector .workflow-inspector-empty {
        color: #9e8cad;
        height: auto;
    }
    """

    def __init__(
        self,
        document: WorkflowGraph,
        selected_node_id: str | None,
        selected_connection_id: str | None,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.document = document
        self.selected_node_id = selected_node_id
        self.selected_connection_id = selected_connection_id

    def compose(self) -> ComposeResult:
        yield Label("Workflow", classes="workflow-inspector-heading")
        yield PropertyRow(
            node_id=None,
            field_name="name",
            label="Name",
            value=self.document.name,
            annotation=str,
        )
        if self.selected_node_id is None:
            if self.selected_connection_id is not None:
                connection = next(
                    connection
                    for connection in self.document.connections
                    if connection.id == self.selected_connection_id
                )
                source = self.document.node(connection.source.node_id)
                target = self.document.node(connection.target.node_id)
                yield Label(
                    "Connection",
                    classes="workflow-inspector-heading",
                )
                yield Static(
                    f"{source.title}.{connection.source.port}\n"
                    f"→ {target.title}.{connection.target.port}\n"
                    f"Order {connection.order}",
                    classes="workflow-inspector-empty",
                )
                return
            yield Static(
                "Select a node to edit its properties.",
                classes="workflow-inspector-empty",
            )
            return

        node = self.document.node(self.selected_node_id)
        yield Label(node.title, classes="workflow-inspector-heading")
        yield Label(
            node_definition(node.kind).label,
            classes="workflow-inspector-kind",
        )
        yield PropertyRow(
            node_id=node.id,
            field_name="title",
            label="Title",
            value=node.title,
            annotation=str,
        )
        for field_name in _visible_config_fields(node):
            field = type(node.config).model_fields[field_name]
            yield PropertyRow(
                node_id=node.id,
                field_name=field_name,
                label=_field_label(field_name),
                value=getattr(node.config, field_name),
                annotation=field.annotation,
                browse=field_name == "path",
                options=_node_property_options(node, field_name),
            )

    async def show(
        self,
        document: WorkflowGraph,
        selected_node_id: str | None,
        selected_connection_id: str | None,
    ) -> None:
        self.document = document
        self.selected_node_id = selected_node_id
        self.selected_connection_id = selected_connection_id
        await self.recompose()


class WorkflowConnectionDialog(
    ModalScreen[tuple[NodePortRef, NodePortRef] | None]
):
    DEFAULT_CSS = """
    WorkflowConnectionDialog {
        align: center middle;
        background: #000000 55%;
    }

    WorkflowConnectionDialog #workflow-connect-dialog {
        width: 92%;
        height: auto;
        border: solid #8a66a3;
        background: #1c1724;
        padding: 1 2;
    }

    WorkflowConnectionDialog .workflow-connect-label {
        width: 10;
        height: 1;
    }

    WorkflowConnectionDialog .workflow-connect-select {
        width: 1fr;
        height: 1;
        border: none;
        padding: 0;
    }

    WorkflowConnectionDialog .workflow-connect-row {
        width: 100%;
        height: 1;
    }

    WorkflowConnectionDialog #workflow-connect-actions {
        width: 100%;
        height: 1;
        margin-top: 1;
        align-horizontal: right;
    }

    WorkflowConnectionDialog Button {
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        document: WorkflowGraph,
        selected_node_id: str | None,
    ) -> None:
        super().__init__()
        self.document = document
        self._sources: dict[str, NodePortRef] = {}
        self._targets: dict[str, tuple[NodePortRef, ...]] = {}
        all_targets = tuple(
            (
                node,
                port,
                NodePortRef(node_id=node.id, port=port.name),
            )
            for node in document.nodes
            for port in node_definition(node.kind).inputs
        )
        for node in document.nodes:
            for port in node_definition(node.kind).outputs:
                source = NodePortRef(node_id=node.id, port=port.name)
                compatible = tuple(
                    target
                    for target_node, target_port, target in all_targets
                    if target_node.id != node.id
                    and set(port.artifact_types).intersection(
                        target_port.artifact_types
                    )
                )
                if compatible:
                    key = _endpoint_key(source)
                    self._sources[key] = source
                    self._targets[key] = compatible
        preferred = next(
            (
                key
                for key, source in self._sources.items()
                if source.node_id == selected_node_id
            ),
            None,
        )
        self._selected_source = preferred or next(
            iter(self._sources),
            None,
        )

    @property
    def can_connect(self) -> bool:
        return self._selected_source is not None

    def compose(self) -> ComposeResult:
        assert self._selected_source is not None
        targets = self._targets[self._selected_source]
        with Container(id="workflow-connect-dialog"):
            yield Label("Connect nodes")
            with Horizontal(classes="workflow-connect-row"):
                yield Label("From", classes="workflow-connect-label")
                yield Select(
                    tuple(
                        (
                            _endpoint_label(self.document, source),
                            key,
                        )
                        for key, source in self._sources.items()
                    ),
                    value=self._selected_source,
                    allow_blank=False,
                    compact=True,
                    id="workflow-connect-source",
                    classes="workflow-connect-select",
                )
            with Horizontal(classes="workflow-connect-row"):
                yield Label("To", classes="workflow-connect-label")
                yield Select(
                    tuple(
                        (
                            _endpoint_label(self.document, target),
                            _endpoint_key(target),
                        )
                        for target in targets
                    ),
                    value=_endpoint_key(targets[0]),
                    allow_blank=False,
                    compact=True,
                    id="workflow-connect-target",
                    classes="workflow-connect-select",
                )
            with Horizontal(id="workflow-connect-actions"):
                yield Button("Cancel", id="workflow-connect-cancel")
                yield Button(
                    "Connect",
                    id="workflow-connect-confirm",
                    variant="primary",
                )

    @on(Select.Changed, "#workflow-connect-source")
    def source_changed(self, event: Select.Changed) -> None:
        if not isinstance(event.value, str):
            return
        self._selected_source = event.value
        targets = self._targets[event.value]
        target_select = self.query_one("#workflow-connect-target", Select)
        target_select.set_options(
            tuple(
                (
                    _endpoint_label(self.document, target),
                    _endpoint_key(target),
                )
                for target in targets
            )
        )
        target_select.value = _endpoint_key(targets[0])

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "workflow-connect-cancel":
            self.dismiss(None)
            return
        if event.button.id != "workflow-connect-confirm":
            return
        source_key = self.query_one(
            "#workflow-connect-source",
            Select,
        ).value
        target_key = self.query_one(
            "#workflow-connect-target",
            Select,
        ).value
        assert isinstance(source_key, str)
        assert isinstance(target_key, str)
        target = next(
            target
            for target in self._targets[source_key]
            if _endpoint_key(target) == target_key
        )
        self.dismiss((self._sources[source_key], target))


class WorkflowEditor(ModalScreen[WorkflowGraph | None]):
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

    WorkflowEditor #workflow-editor-body {
        width: 100%;
        height: 1fr;
        padding: 0 1;
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
        def __init__(self, document: WorkflowGraph) -> None:
            super().__init__()
            self.document = document

    class LoadRequested(Message):
        pass

    class RunRequested(Message):
        def __init__(self, workflow: CompiledWorkflow) -> None:
            super().__init__()
            self.workflow = workflow

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
        document: WorkflowGraph,
        *,
        document_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.document = document
        self.document_path = document_path
        self.selected_node_id: str | None = None
        self.selected_connection_id: str | None = None
        self._undo_stack: list[WorkflowGraph] = []
        self._redo_stack: list[WorkflowGraph] = []
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
                    "Connect",
                    id="workflow-connect-nodes",
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
            with Horizontal(id="workflow-editor-body"):
                yield WorkflowCanvas(
                    self.document,
                    id="workflow-canvas",
                )
                yield WorkflowInspector(
                    self.document,
                    self.selected_node_id,
                    self.selected_connection_id,
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

    def set_status(self, message: str) -> None:
        self._set_status(message)

    async def apply_loaded_document(
        self,
        document: WorkflowGraph,
        path: Path,
    ) -> None:
        self.document_path = path
        self.selected_node_id = None
        self.selected_connection_id = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        await self._show_document(document)
        self._set_status(f"Loaded {path}")

    async def apply_browsed_path(
        self,
        node_id: str,
        field_name: str,
        path: Path,
    ) -> None:
        node = self.document.node(node_id)
        await self._update_config_field(
            node,
            field_name,
            path.as_posix(),
        )

    def save_to(self, path: Path) -> None:
        save_workflow_document(self.document, path)
        self.document_path = path
        self.query_one("#workflow-editor-title", Label).update(
            self._title_text()
        )
        self._set_status(f"Saved {path}")

    @on(WorkflowCanvas.SelectionChanged)
    async def canvas_selection_changed(
        self,
        event: WorkflowCanvas.SelectionChanged,
    ) -> None:
        self.selected_node_id = event.node_id
        self.selected_connection_id = event.connection_id
        await self.query_one(WorkflowInspector).show(
            self.document,
            self.selected_node_id,
            self.selected_connection_id,
        )
        self._update_history_actions()

    @on(WorkflowCanvas.NodeMoved)
    async def canvas_node_moved(
        self,
        event: WorkflowCanvas.NodeMoved,
    ) -> None:
        node = self.document.node(event.node_id)
        moved = _replace_node(
            node,
            layout=NodeLayout(x=event.x, y=event.y),
        )
        await self._replace_node(moved)

    @on(WorkflowCanvas.ConnectionRequested)
    async def canvas_connection_requested(
        self,
        event: WorkflowCanvas.ConnectionRequested,
    ) -> None:
        await self._connect_ports(event.source, event.target)

    async def _connect_ports(
        self,
        source: NodePortRef,
        target: NodePortRef,
    ) -> None:
        target_node = self.document.node(target.node_id)
        target_port = node_definition(target_node.kind).input(
            target.port
        )
        assert target_port is not None
        retained = [
            connection
            for connection in self.document.connections
            if target_port.multiple
            or (
                connection.target.node_id != target.node_id
                or connection.target.port != target.port
            )
        ]
        if any(
            connection.source == source
            and connection.target == target
            for connection in retained
        ):
            return
        order = (
            max(
                (
                    connection.order
                    for connection in retained
                    if connection.target == target
                ),
                default=-1,
            )
            + 1
            if target_port.multiple
            else 0
        )
        retained.append(
            WorkflowConnection(
                id=f"connection-{uuid.uuid4().hex}",
                source=source,
                target=target,
                order=order,
            )
        )
        try:
            document = _rebuild_graph(
                self.document,
                connections=retained,
            )
        except ValidationError as error:
            self.notify(str(error), severity="error")
            return
        await self._apply_document(document)

    @on(Input.Submitted)
    async def property_submitted(self, event: Input.Submitted) -> None:
        editor = event.input
        if not isinstance(editor, PropertyInput):
            return
        await self._commit_property_input(editor)

    @on(events.DescendantBlur)
    async def property_blurred(self, event: events.DescendantBlur) -> None:
        if isinstance(event.widget, PropertyInput):
            await self._commit_property_input(event.widget)

    async def _commit_property_input(self, editor: PropertyInput) -> None:
        if editor.value == editor.original_value:
            return
        if editor.node_id is None:
            try:
                document = WorkflowGraph(
                    version=self.document.version,
                    name=editor.value,
                    nodes=self.document.nodes,
                    connections=self.document.connections,
                )
            except ValidationError as error:
                self.notify(str(error), severity="error")
                return
            await self._apply_document(document)
            return
        node = self.document.node(editor.node_id)
        if editor.field_name == "title":
            try:
                changed = _replace_node(node, title=editor.value)
            except ValidationError as error:
                self.notify(str(error), severity="error")
                return
            await self._replace_node(changed)
            return
        await self._update_config_field(
            node,
            editor.field_name,
            editor.value,
        )

    @on(Select.Changed)
    async def property_selected(self, event: Select.Changed) -> None:
        editor = event.select
        if not isinstance(editor, PropertySelect) or not editor.is_mounted:
            return
        node = self.document.node(editor.node_id)
        if getattr(node.config, editor.field_name) != editor.original_value:
            return
        await self._update_config_field(
            node,
            editor.field_name,
            event.value,
        )

    @on(Button.Pressed)
    async def button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        match button_id:
            case "workflow-add-node":
                await self._add_node()
            case "workflow-delete-node":
                await self._delete_selection()
            case "workflow-connect-nodes":
                self._open_connection_dialog()
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
                self.dismiss(self.document)
            case "workflow-quit":
                self.post_message(self.QuitRequested())
            case _:
                if (
                    event.button.name is not None
                    and event.button.has_class("workflow-property-browse")
                ):
                    row = event.button.parent
                    assert isinstance(row, PropertyRow)
                    assert row.node_id is not None
                    current_value = row.query_one(PropertyInput).value
                    self.post_message(
                        self.BrowseRequested(
                            row.node_id,
                            row.field_name,
                            current_value,
                        )
                    )

    def _open_connection_dialog(self) -> None:
        dialog = WorkflowConnectionDialog(
            self.document,
            self.selected_node_id,
        )
        if not dialog.can_connect:
            self.notify(
                "This workflow has no compatible node ports.",
                severity="warning",
            )
            return
        self.app.push_screen(dialog, self._connection_chosen)

    def _connection_chosen(
        self,
        connection: tuple[NodePortRef, NodePortRef] | None,
    ) -> None:
        if connection is None:
            return
        source, target = connection
        self.run_worker(
            self._connect_ports(source, target),
            group="workflow-connect",
            exclusive=True,
        )

    async def _add_node(self) -> None:
        value = self.query_one("#workflow-node-kind", Select).value
        assert isinstance(value, str)
        canvas = self.query_one(WorkflowCanvas)
        x = int(canvas.scroll_offset.x + max(2, canvas.size.width // 2))
        y = int(canvas.scroll_offset.y + max(2, canvas.size.height // 2))
        definition = node_definition(NodeKind(value))
        height = max(
            len(definition.inputs),
            len(definition.outputs),
            1,
        ) + 2
        while any(
            x < geometry.right + 2
            and x + NODE_WIDTH + 1 > geometry.x
            and y < geometry.bottom + 1
            and y + height > geometry.y
            for geometry in (
                canvas.node_geometry(node.id)
                for node in self.document.nodes
            )
        ):
            x += 4
            y += 2
        node = create_workflow_node(NodeKind(value), x=x, y=y)
        document = _rebuild_graph(
            self.document,
            nodes=[*self.document.nodes, node],
        )
        await self._apply_document(document, selected_node_id=node.id)

    async def _delete_selection(self) -> None:
        if self.selected_node_id is not None:
            node_id = self.selected_node_id
            document = _rebuild_graph(
                self.document,
                nodes=[
                    node
                    for node in self.document.nodes
                    if node.id != node_id
                ],
                connections=_normalized_connection_orders(
                    [
                        connection
                        for connection in self.document.connections
                        if connection.source.node_id != node_id
                        and connection.target.node_id != node_id
                    ]
                ),
            )
            await self._apply_document(
                document,
                selected_node_id=None,
                selected_connection_id=None,
            )
            return
        if self.selected_connection_id is None:
            return
        document = _rebuild_graph(
            self.document,
            connections=_normalized_connection_orders(
                [
                    connection
                    for connection in self.document.connections
                    if connection.id != self.selected_connection_id
                ]
            ),
        )
        await self._apply_document(
            document,
            selected_node_id=None,
            selected_connection_id=None,
        )

    async def _move_connection(self, direction: int) -> None:
        if self.selected_connection_id is None:
            return
        selected = next(
            connection
            for connection in self.document.connections
            if connection.id == self.selected_connection_id
        )
        siblings = [
            connection
            for connection in self.document.connections
            if connection.target == selected.target
        ]
        siblings.sort(key=lambda connection: connection.order)
        source_index = siblings.index(selected)
        target_index = source_index + direction
        if not 0 <= target_index < len(siblings):
            return
        other = siblings[target_index]
        connections = [
            _connection_with_order(connection, other.order)
            if connection.id == selected.id
            else (
                _connection_with_order(connection, selected.order)
                if connection.id == other.id
                else connection
            )
            for connection in self.document.connections
        ]
        await self._apply_document(
            _rebuild_graph(self.document, connections=connections)
        )

    async def _undo(self) -> None:
        if not self._undo_stack:
            return
        previous = self._undo_stack.pop()
        self._redo_stack.append(self.document)
        await self._show_document(previous)

    async def _redo(self) -> None:
        if not self._redo_stack:
            return
        following = self._redo_stack.pop()
        self._undo_stack.append(self.document)
        await self._show_document(following)

    async def _auto_layout(self) -> None:
        predecessors = {node.id: set() for node in self.document.nodes}
        for connection in self.document.connections:
            predecessors[connection.target.node_id].add(
                connection.source.node_id
            )
        order = tuple(TopologicalSorter(predecessors).static_order())
        depths: dict[str, int] = {}
        for node_id in order:
            depths[node_id] = max(
                (depths[predecessor] + 1 for predecessor in predecessors[node_id]),
                default=0,
            )
        layers: dict[int, list[str]] = {}
        for node in self.document.nodes:
            layers.setdefault(depths[node.id], []).append(node.id)

        positions: dict[str, NodeLayout] = {}
        for depth, node_ids in layers.items():
            y = 2
            for node_id in node_ids:
                node = self.document.node(node_id)
                definition = node_definition(node.kind)
                positions[node_id] = NodeLayout(
                    x=4 + depth * (NODE_WIDTH + 10),
                    y=y,
                )
                y += max(
                    len(definition.inputs),
                    len(definition.outputs),
                    1,
                ) + 4
        nodes = [
            _replace_node(node, layout=positions[node.id])
            for node in self.document.nodes
        ]
        await self._apply_document(
            _rebuild_graph(self.document, nodes=nodes)
        )

    def _save(self) -> None:
        if self.document_path is None:
            self.post_message(self.SaveRequested(self.document))
            return
        try:
            self.save_to(self.document_path)
        except OSError as error:
            self.notify(str(error), severity="error")

    def _run(self) -> None:
        try:
            workflow = compile_workflow(self.document)
        except ValueError as error:
            self.notify(str(error), severity="error")
            return
        self.post_message(self.RunRequested(workflow))

    async def _update_config_field(
        self,
        node: WorkflowNode,
        field_name: str,
        raw_value: object,
    ) -> None:
        model_field = type(node.config).model_fields[field_name]
        try:
            value = _parse_property(raw_value, model_field.annotation)
            if (
                isinstance(node, (ImagePostprocessNode, FramePostprocessNode))
                and field_name == "model"
            ):
                changed = _replace_node(
                    node,
                    config=postprocess_config_for_model(str(value)),
                )
                await self._replace_node(changed)
                return
            config_payload = node.config.model_dump(mode="python")
            config_payload[field_name] = value
            _apply_owner_defaults(node, field_name, value, config_payload)
            config = type(node.config).model_validate(config_payload)
            changed = _replace_node(node, config=config)
        except (ValidationError, ValueError) as error:
            self.notify(str(error), severity="error")
            return
        await self._replace_node(changed)

    async def _replace_node(self, changed: WorkflowNode) -> None:
        document = _rebuild_graph(
            self.document,
            nodes=[
                changed if node.id == changed.id else node
                for node in self.document.nodes
            ],
        )
        await self._apply_document(document)

    async def _apply_document(
        self,
        document: WorkflowGraph,
        *,
        selected_node_id: str | None | object = _PRESERVE_SELECTION,
        selected_connection_id: str | None | object = _PRESERVE_SELECTION,
    ) -> None:
        if document == self.document:
            return
        self._undo_stack.append(self.document)
        self._redo_stack.clear()
        if selected_node_id is not _PRESERVE_SELECTION:
            assert selected_node_id is None or isinstance(selected_node_id, str)
            self.selected_node_id = selected_node_id
        if selected_connection_id is not _PRESERVE_SELECTION:
            assert (
                selected_connection_id is None
                or isinstance(selected_connection_id, str)
            )
            self.selected_connection_id = selected_connection_id
        await self._show_document(document)

    async def _show_document(self, document: WorkflowGraph) -> None:
        self.document = document
        if (
            self.selected_node_id is not None
            and not any(
                node.id == self.selected_node_id
                for node in document.nodes
            )
        ):
            self.selected_node_id = None
        if (
            self.selected_connection_id is not None
            and not any(
                connection.id == self.selected_connection_id
                for connection in document.connections
            )
        ):
            self.selected_connection_id = None
        canvas = self.query_one(WorkflowCanvas)
        canvas.set_document(document)
        canvas.set_selection(
            self.selected_node_id,
            self.selected_connection_id,
        )
        await self.query_one(WorkflowInspector).show(
            document,
            self.selected_node_id,
            self.selected_connection_id,
        )
        self.query_one("#workflow-editor-title", Label).update(
            self._title_text()
        )
        self._update_history_actions()

    def _update_history_actions(self) -> None:
        self.query_one("#workflow-undo", Button).disabled = (
            self._running or not self._undo_stack
        )
        self.query_one("#workflow-redo", Button).disabled = (
            self._running or not self._redo_stack
        )
        self.query_one("#workflow-delete-node", Button).disabled = (
            self._running
            or (
                self.selected_node_id is None
                and self.selected_connection_id is None
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
        if self.selected_connection_id is None:
            return False, False
        selected = next(
            connection
            for connection in self.document.connections
            if connection.id == self.selected_connection_id
        )
        siblings = sorted(
            (
                connection
                for connection in self.document.connections
                if connection.target == selected.target
            ),
            key=lambda connection: connection.order,
        )
        index = siblings.index(selected)
        return index > 0, index < len(siblings) - 1

    def _set_status(self, message: str) -> None:
        self.query_one("#workflow-editor-status", Label).update(message)

    def _title_text(self) -> str:
        path = (
            self.document_path.as_posix()
            if self.document_path is not None
            else "Unsaved"
        )
        return f"Workflow — {self.document.name} — {path}"


def _rebuild_graph(
    document: WorkflowGraph,
    *,
    nodes: list[WorkflowNode] | None = None,
    connections: list[WorkflowConnection] | None = None,
) -> WorkflowGraph:
    return WorkflowGraph(
        version=document.version,
        name=document.name,
        nodes=document.nodes if nodes is None else nodes,
        connections=(
            document.connections
            if connections is None
            else connections
        ),
    )


def _replace_node(
    node: WorkflowNode,
    *,
    title: str | None = None,
    layout: NodeLayout | None = None,
    config: BaseModel | None = None,
) -> WorkflowNode:
    payload = node.model_dump(mode="python")
    if title is not None:
        payload["title"] = title
    if layout is not None:
        payload["layout"] = layout
    if config is not None:
        payload["config"] = config
    return type(node).model_validate(payload)


def _normalized_connection_orders(
    connections: list[WorkflowConnection],
) -> list[WorkflowConnection]:
    groups: dict[tuple[str, str], list[WorkflowConnection]] = {}
    for connection in connections:
        groups.setdefault(
            (connection.target.node_id, connection.target.port),
            [],
        ).append(connection)
    normalized: dict[str, WorkflowConnection] = {}
    for siblings in groups.values():
        siblings.sort(key=lambda connection: connection.order)
        for order, connection in enumerate(siblings):
            normalized[connection.id] = _connection_with_order(
                connection,
                order,
            )
    return [normalized[connection.id] for connection in connections]


def _connection_with_order(
    connection: WorkflowConnection,
    order: int,
) -> WorkflowConnection:
    payload = connection.model_dump(mode="python")
    payload["order"] = order
    return WorkflowConnection.model_validate(payload)


def _endpoint_key(endpoint: NodePortRef) -> str:
    return f"{endpoint.node_id}:{endpoint.port}"


def _endpoint_label(
    document: WorkflowGraph,
    endpoint: NodePortRef,
) -> str:
    return f"{document.node(endpoint.node_id).title}.{endpoint.port}"


def _field_label(field_name: str) -> str:
    return field_name.replace("_", " ").capitalize()


def _node_property_options(
    node: WorkflowNode,
    field_name: str,
) -> tuple[tuple[str, object], ...] | None:
    values: tuple[str, ...] | None = None
    if isinstance(node, ImageEditNode):
        if field_name == "backend":
            values = IMAGE_EDIT_BACKENDS
        elif field_name in {"sampler", "scheduler"}:
            settings = image_edit_backend_settings(node.config.backend)
            values = getattr(settings, f"{field_name}s")
    elif isinstance(node, (ImagePostprocessNode, FramePostprocessNode)):
        if field_name == "model":
            values = image_batch_postprocess_model_names()
    elif isinstance(node, AnimeGenI2VNode):
        if field_name == "sampling":
            values = ANIMEGEN_SAMPLINGS
        elif field_name == "precision":
            values = ANIMEGEN_PRECISIONS
    if values is None:
        return None
    return tuple((value, value) for value in values)


def _visible_config_fields(node: WorkflowNode) -> tuple[str, ...]:
    fields = tuple(type(node.config).model_fields)
    if (
        isinstance(node, (ImagePostprocessNode, FramePostprocessNode))
        and isinstance(node.config, VosrPostprocessConfig)
    ):
        hidden = (
            "scale"
            if node.config.sizing == "long-side"
            else "long_side"
        )
        return tuple(field for field in fields if field != hidden)
    return fields


def _apply_owner_defaults(
    node: WorkflowNode,
    field_name: str,
    value: object,
    config_payload: dict[str, object],
) -> None:
    if isinstance(node, ImageEditNode) and field_name == "backend":
        settings = image_edit_backend_settings(str(value))
        config_payload.update(
            steps=settings.steps,
            guidance=settings.guidance,
            strength=settings.strength,
            sampler=settings.sampler,
            scheduler=settings.scheduler,
        )
    elif isinstance(node, AnimeGenI2VNode) and field_name == "sampling":
        config_payload["steps"] = animegen_sampling_profile(str(value)).steps


def _property_options(
    annotation: object,
) -> tuple[tuple[str, object], ...] | None:
    if annotation is bool:
        return (("Off", False), ("On", True))
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return tuple((str(member.value), member.value) for member in annotation)
    if get_origin(annotation) is Literal:
        return tuple((str(value), value) for value in get_args(annotation))
    return None


def _parse_property(raw_value: object, annotation: object) -> object:
    if (
        raw_value == ""
        and get_origin(annotation) in {Union, UnionType}
        and type(None) in get_args(annotation)
    ):
        return None
    return TypeAdapter(annotation).validate_python(raw_value)
