from __future__ import annotations

from enum import Enum
from typing import Literal, get_args, get_origin

from textual import events
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Button, Input, Label, Select

from aigen.tui_text_editor import MultilineInput


PROPERTY_LABEL_MIN_WIDTH = 8
PROPERTY_EDITOR_MIN_WIDTH = 8
PROPERTY_BROWSE_MIN_WIDTH = 8
PROPERTY_BROWSE_HORIZONTAL_PADDING = 1
PROPERTY_BROWSE_OUTER_MIN_WIDTH = (
    PROPERTY_BROWSE_MIN_WIDTH + 2 * PROPERTY_BROWSE_HORIZONTAL_PADDING
)
PROPERTY_ROW_MIN_WIDTH = (
    PROPERTY_LABEL_MIN_WIDTH + PROPERTY_EDITOR_MIN_WIDTH + PROPERTY_BROWSE_OUTER_MIN_WIDTH
)


class PropertyInput(Input):
    def __init__(self, value: str, *, node_id: str | None, field_name: str) -> None:
        super().__init__(value, compact=True, classes="workflow-property-editor")
        self.node_id = node_id
        self.field_name = field_name
        self.original_value = value


class PropertyTextArea(MultilineInput):
    def __init__(self, value: str, *, node_id: str | None, field_name: str) -> None:
        super().__init__(
            value,
            classes="workflow-property-editor",
            tooltip="Enter: new line · Ctrl+Enter: apply · Ctrl+S: save workflow",
        )
        self.node_id = node_id
        self.field_name = field_name
        self.original_value = value


class PropertySelect(Select[object]):
    def __init__(
        self,
        options: tuple[tuple[str, object], ...],
        value: object,
        *,
        node_id: str | None,
        field_name: str,
    ) -> None:
        super().__init__(
            options, value=value, allow_blank=False, compact=True,
            classes="workflow-property-editor",
        )
        self.node_id = node_id
        self.field_name = field_name
        self.original_value = value


class PropertyRow(Container):
    DEFAULT_CSS = """
    PropertyRow {
        layout: grid;
        grid-size: 2 1;
        grid-columns: 1fr 2fr;
        grid-rows: 1;
        width: 100%%;
        height: 1;
    }
    PropertyRow.stacked {
        grid-size: 1 2;
        grid-columns: 1fr;
        grid-rows: 1 1;
        height: 2;
    }
    PropertyRow .workflow-property-label {
        width: 100%%;
        min-width: %(label_min_width)d;
        height: 1;
        content-align-vertical: middle;
        text-overflow: ellipsis;
    }
    PropertyRow .workflow-property-controls {
        layout: grid;
        grid-size: 2 1;
        grid-columns: 1fr auto;
        grid-rows: 1;
        width: 100%%;
        min-width: 0;
        height: 1;
    }
    PropertyRow .workflow-property-editor {
        width: 100%%;
        min-width: %(editor_min_width)d;
        height: 1;
        border: none;
        padding: 0;
    }
    PropertyRow .workflow-property-browse {
        width: auto;
        min-width: %(browse_min_width)d;
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 %(browse_padding)d;
    }
    PropertyRow.multiline {
        layout: vertical;
        height: auto;
        margin: 1 0;
    }
    PropertyRow.multiline .workflow-property-controls {
        layout: vertical;
        height: auto;
    }
    PropertyRow PropertyTextArea.workflow-property-editor {
        height: 25vh;
        padding: 0 1;
        scrollbar-size-vertical: 1;
    }
    """ % {
        "label_min_width": PROPERTY_LABEL_MIN_WIDTH,
        "editor_min_width": PROPERTY_EDITOR_MIN_WIDTH,
        "browse_min_width": PROPERTY_BROWSE_MIN_WIDTH,
        "browse_padding": PROPERTY_BROWSE_HORIZONTAL_PADDING,
    }

    def __init__(
        self,
        *,
        node_id: str | None,
        field_name: str,
        label: str,
        value: object,
        annotation: object,
        browse: bool = False,
        multiline: bool = False,
        visible: bool = True,
        placeholder: str = "",
        options: tuple[tuple[str, object], ...] | None = None,
    ) -> None:
        super().__init__(classes="multiline" if multiline else None)
        self.node_id = node_id
        self.field_name = field_name
        self.label_text = label
        self.value = value
        self.annotation = annotation
        self.browse = browse
        self.display = visible
        self.options = options if options is not None else _property_options(annotation)
        self.editor: PropertyInput | PropertyTextArea | PropertySelect
        if self.options is not None:
            self.editor = PropertySelect(
                self.options, value, node_id=node_id, field_name=field_name,
            )
        else:
            widget_type = PropertyTextArea if multiline else PropertyInput
            self.editor = widget_type(
                "" if value is None else str(value), node_id=node_id, field_name=field_name,
            )

        if isinstance(self.editor, PropertyInput):
            self.editor.placeholder = placeholder

    def compose(self) -> ComposeResult:
        yield Label(self.label_text, classes="workflow-property-label")
        with Container(classes="workflow-property-controls"):
            yield self.editor
            if self.browse:
                yield Button(
                    "Browse", name=self.field_name, compact=True,
                    classes="workflow-property-browse",
                )

    def show_value(
        self,
        value: object,
        *,
        options: tuple[tuple[str, object], ...] | None = None,
    ) -> None:
        """Project a changed model value without replacing its editor or an unchanged draft."""
        editor = self.editor
        if isinstance(editor, PropertySelect):
            options = options if options is not None else _property_options(self.annotation)
            if self.options != options or self.value != value:
                with editor.prevent(Select.Changed):
                    if self.options != options:
                        editor.set_options(options)
                    editor.original_value = value
                    editor.value = value
                self.options = options
        elif self.value != value:
            text = "" if value is None else str(value)
            editor.original_value = text
            if isinstance(editor, PropertyTextArea):
                if editor.text != text:
                    editor.load_text(text)
            elif editor.value != text:
                editor.value = text
        self.value = value

    def on_resize(self, event: events.Resize) -> None:
        horizontal_minimum = (
            PROPERTY_LABEL_MIN_WIDTH + PROPERTY_EDITOR_MIN_WIDTH
            + (PROPERTY_BROWSE_OUTER_MIN_WIDTH if self.browse else 0)
        )
        self.set_class(event.size.width < horizontal_minimum, "stacked")


def _property_options(annotation: object) -> tuple[tuple[str, object], ...] | None:
    if annotation is bool:
        return (("Off", False), ("On", True))
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return tuple((str(member.value), member.value) for member in annotation)
    if get_origin(annotation) is Literal:
        return tuple((str(value), value) for value in get_args(annotation))
    return None
