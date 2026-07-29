from __future__ import annotations

from enum import Enum
from typing import Literal, get_args, get_origin

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Button, Input, Label, Select, Static

from aigen.generation.animegen_i2v import (
    ANIMEGEN_PRECISIONS,
    ANIMEGEN_SAMPLINGS,
)
from aigen.generation.image_batch_postprocess import (
    image_batch_postprocess_model_names,
)
from aigen.generation.image_edit import (
    IMAGE_EDIT_BACKENDS,
    image_edit_backend_settings,
)
from aigen.workflow_edit_buffer import WorkflowPropertyEdit
from aigen.workflow_graph import (
    AnimeGenI2VNode,
    FramePostprocessNode,
    ImageEditNode,
    ImagePostprocessNode,
    VosrPostprocessConfig,
    WorkflowGraph,
    WorkflowNode,
    node_definition,
)


PROPERTY_LABEL_MIN_WIDTH = 8
PROPERTY_EDITOR_MIN_WIDTH = 8
PROPERTY_BROWSE_MIN_WIDTH = 8
PROPERTY_BROWSE_HORIZONTAL_PADDING = 1
PROPERTY_BROWSE_OUTER_MIN_WIDTH = (
    PROPERTY_BROWSE_MIN_WIDTH
    + 2 * PROPERTY_BROWSE_HORIZONTAL_PADDING
)
INSPECTOR_HORIZONTAL_GUTTER_WIDTH = 4
INSPECTOR_HORIZONTAL_CONTENT_MIN_WIDTH = (
    PROPERTY_LABEL_MIN_WIDTH
    + PROPERTY_EDITOR_MIN_WIDTH
    + PROPERTY_BROWSE_OUTER_MIN_WIDTH
)
INSPECTOR_HORIZONTAL_MIN_WIDTH = (
    INSPECTOR_HORIZONTAL_CONTENT_MIN_WIDTH
    + INSPECTOR_HORIZONTAL_GUTTER_WIDTH
)


class PropertyInput(Input):
    def __init__(
        self,
        value: str,
        *,
        original_value: str,
        node_id: str | None,
        field_name: str,
    ) -> None:
        super().__init__(
            value,
            compact=True,
            classes="workflow-property-editor",
        )
        self.node_id = node_id
        self.field_name = field_name
        self.original_value = original_value


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


class PropertyRow(Container):
    def __init__(
        self,
        *,
        node_id: str | None,
        field_name: str,
        label: str,
        value: object,
        annotation: object,
        draft: str | None = None,
        browse: bool = False,
        options: tuple[tuple[str, object], ...] | None = None,
    ) -> None:
        super().__init__(classes="workflow-property-row")
        self.node_id = node_id
        self.field_name = field_name
        self.label_text = label
        self.value = value
        self.annotation = annotation
        self.draft = draft
        self.browse = browse
        self.options = options

    def compose(self) -> ComposeResult:
        yield Label(self.label_text, classes="workflow-property-label")
        options = (
            self.options
            if self.options is not None
            else _property_options(self.annotation)
        )
        with Container(classes="workflow-property-controls"):
            if options is None:
                original_value = "" if self.value is None else str(self.value)
                yield PropertyInput(
                    (
                        original_value
                        if self.draft is None
                        else self.draft
                    ),
                    original_value=original_value,
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

    def on_resize(self, event: events.Resize) -> None:
        horizontal_minimum = (
            PROPERTY_LABEL_MIN_WIDTH
            + PROPERTY_EDITOR_MIN_WIDTH
            + (
                PROPERTY_BROWSE_OUTER_MIN_WIDTH
                if self.browse
                else 0
            )
        )
        self.set_class(
            event.size.width < horizontal_minimum,
            "stacked",
        )


class WorkflowInspector(VerticalScroll):
    DEFAULT_CSS = """
    WorkflowInspector {
        width: 2fr;
        min-width: %(inspector_min_width)d;
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
        layout: grid;
        grid-size: 2 1;
        grid-columns: 1fr 2fr;
        grid-rows: 1;
        width: 100%%;
        height: 1;
    }

    WorkflowInspector .workflow-property-row.stacked {
        grid-size: 1 2;
        grid-columns: 1fr;
        grid-rows: 1 1;
        height: 2;
    }

    WorkflowInspector .workflow-property-label {
        width: 100%%;
        min-width: %(label_min_width)d;
        height: 1;
        content-align-vertical: middle;
        text-overflow: ellipsis;
    }

    WorkflowInspector .workflow-property-controls {
        layout: grid;
        grid-size: 2 1;
        grid-columns: 1fr auto;
        grid-rows: 1;
        width: 100%%;
        min-width: 0;
        height: 1;
    }

    WorkflowInspector .workflow-property-editor {
        width: 100%%;
        min-width: %(editor_min_width)d;
        height: 1;
        border: none;
        padding: 0;
    }

    WorkflowInspector .workflow-property-browse {
        width: auto;
        min-width: %(browse_min_width)d;
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 %(browse_padding)d;
    }

    WorkflowInspector .workflow-inspector-empty {
        color: #9e8cad;
        height: auto;
    }
    """ % {
        "inspector_min_width": INSPECTOR_HORIZONTAL_MIN_WIDTH,
        "label_min_width": PROPERTY_LABEL_MIN_WIDTH,
        "editor_min_width": PROPERTY_EDITOR_MIN_WIDTH,
        "browse_min_width": PROPERTY_BROWSE_MIN_WIDTH,
        "browse_padding": PROPERTY_BROWSE_HORIZONTAL_PADDING,
    }

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
        self._node_id = selected_node_id
        self._connection_id = selected_connection_id
        self._drafts: dict[
            tuple[str | None, str],
            WorkflowPropertyEdit,
        ] = {}

    @property
    def horizontal_minimum_width(self) -> int:
        return (
            INSPECTOR_HORIZONTAL_CONTENT_MIN_WIDTH
            + self.styles.gutter.width
        )

    def compose(self) -> ComposeResult:
        yield Label("Workflow", classes="workflow-inspector-heading")
        yield PropertyRow(
            node_id=None,
            field_name="name",
            label="Name",
            value=self.document.name,
            annotation=str,
            draft=self._draft_value(None, "name"),
        )
        if self._node_id is None:
            if self._connection_id is not None:
                connection = next(
                    connection
                    for connection in self.document.connections
                    if connection.id == self._connection_id
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

        node = self.document.node(self._node_id)
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
            draft=self._draft_value(node.id, "title"),
        )
        for field_name in _visible_config_fields(node):
            field = type(node.config).model_fields[field_name]
            yield PropertyRow(
                node_id=node.id,
                field_name=field_name,
                label=_field_label(field_name),
                value=getattr(node.config, field_name),
                annotation=field.annotation,
                draft=self._draft_value(node.id, field_name),
                browse=field_name == "path",
                options=_node_property_options(node, field_name),
            )

    @on(Input.Changed)
    def property_changed(self, event: Input.Changed) -> None:
        editor = event.input
        if not isinstance(editor, PropertyInput):
            return
        editor.remove_class("-invalid")
        self._capture_input(editor)

    def property_drafts(self) -> tuple[WorkflowPropertyEdit, ...]:
        drafts: list[WorkflowPropertyEdit] = []
        for editor in self.query(PropertyInput):
            edit = self._capture_input(editor)
            if edit is not None:
                drafts.append(edit)
        return tuple(drafts)

    def clear_drafts(self) -> None:
        self._drafts.clear()

    def focus_invalid_draft(
        self,
        edit: WorkflowPropertyEdit,
    ) -> None:
        editor = next(
            editor
            for editor in self.query(PropertyInput)
            if editor.node_id == edit.node_id
            and editor.field_name == edit.field_name
        )
        editor.add_class("-invalid")
        editor.focus()

    async def show(
        self,
        document: WorkflowGraph,
        selected_node_id: str | None,
        selected_connection_id: str | None,
    ) -> None:
        previous_projection = self._projection()
        self.document = document
        self._node_id = selected_node_id
        self._connection_id = selected_connection_id
        if self._projection() != previous_projection:
            await self.recompose()

    def _capture_input(
        self,
        editor: PropertyInput,
    ) -> WorkflowPropertyEdit | None:
        key = (editor.node_id, editor.field_name)
        if editor.value == editor.original_value:
            self._drafts.pop(key, None)
            return None
        edit = WorkflowPropertyEdit(
            node_id=editor.node_id,
            field_name=editor.field_name,
            raw_value=editor.value,
        )
        self._drafts[key] = edit
        return edit

    def _draft_value(
        self,
        node_id: str | None,
        field_name: str,
    ) -> str | None:
        draft = self._drafts.get((node_id, field_name))
        return None if draft is None else str(draft.raw_value)

    def _projection(self) -> tuple[object, ...]:
        if self._node_id is not None:
            node = self.document.node(self._node_id)
            return (
                self.document.name,
                self._node_id,
                node.kind,
                node.title,
                node.config,
            )
        if self._connection_id is not None:
            connection = next(
                connection
                for connection in self.document.connections
                if connection.id == self._connection_id
            )
            return (
                self.document.name,
                self._connection_id,
                connection.source,
                connection.target,
                connection.order,
                self.document.node(connection.source.node_id).title,
                self.document.node(connection.target.node_id).title,
            )
        return (self.document.name, None)


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
    hidden: set[str] = set()
    if (
        "seed_mode" in type(node.config).model_fields
        and getattr(node.config, "seed_mode") == "random"
    ):
        hidden.add("seed")
    if (
        isinstance(node, (ImagePostprocessNode, FramePostprocessNode))
        and isinstance(node.config, VosrPostprocessConfig)
    ):
        hidden.add(
            "scale"
            if node.config.sizing == "long-side"
            else "long_side"
        )
    return tuple(field for field in fields if field not in hidden)


def _property_options(
    annotation: object,
) -> tuple[tuple[str, object], ...] | None:
    if annotation is bool:
        return (("Off", False), ("On", True))
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return tuple(
            (str(member.value), member.value)
            for member in annotation
        )
    if get_origin(annotation) is Literal:
        return tuple(
            (str(value), value)
            for value in get_args(annotation)
        )
    return None
