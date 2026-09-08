from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Label, Select, TextArea

from aigen.image_tui_model import DropdownOption, FormField, ImageEditForm
from aigen.postprocess_tui_model import PostprocessForm
from aigen.sam_tui_model import SamEditForm
from aigen.video_tui_model import VideoForm
from aigen.tui_text_editor import MultilineInput

FormModel = ImageEditForm | PostprocessForm | VideoForm | SamEditForm


class PathInput(Input):
    class BrowseRequested(Message):
        def __init__(self, form: FormModel, field: FormField) -> None:
            super().__init__()
            self.form = form
            self.field = field

    def __init__(self, form: FormModel, field: FormField) -> None:
        super().__init__(field.value, compact=True, classes="field-editor")
        self.form = form
        self.field = field

    def on_click(self, event: events.Click) -> None:
        if event.chain == 2:
            event.stop()
            self.post_message(self.BrowseRequested(self.form, self.field))


class FieldRow(Horizontal):
    class Selected(Message):
        def __init__(self, form: FormModel, field: FormField) -> None:
            super().__init__()
            self.form = form
            self.field = field

    def __init__(self, form: FormModel, field: FormField) -> None:
        movable = field.slot_id in form.slot_move_states and field.name != "lora_weight"
        super().__init__(classes="field-row movable" if movable else "field-row")
        self.form = form
        self.field = field
        self.movable = movable
        self.set_class(field.name in {"prompt", "negative_prompt"}, "multiline")
        self.options = self._options()

    def _options(self) -> tuple[DropdownOption, ...] | None:
        options = self.form.dropdown_options(self.field)
        if options is not None and self.field.value not in {option.value for option in options}:
            options = (*options, DropdownOption(self.field.value, self.field.value))
        return options

    def compose(self) -> ComposeResult:
        yield Label(self.field.label, classes="field-label")
        options = self.options
        if options is None:
            if self.has_class("multiline"):
                yield MultilineInput(self.field.value, classes="field-editor",
                                     tooltip="Enter: new line · Tab: next field · Ctrl+Z: undo")
            elif self.field.path_kind is not None:
                yield PathInput(self.form, self.field)
            else:
                yield Input(self.field.value, compact=True, classes="field-editor")
        else:
            yield Select(
                ((option.label, option.value) for option in options),
                allow_blank=False,
                value=self.field.value,
                compact=True,
                classes="field-editor",
            )
        if self.movable:
            can_move_up, can_move_down = self.form.slot_move_states[self.field.slot_id]
            yield Button(
                "↑",
                name="move-up",
                compact=True,
                flat=True,
                disabled=not can_move_up,
                classes="move-control",
            )
            yield Button(
                "↓",
                name="move-down",
                compact=True,
                flat=True,
                disabled=not can_move_down,
                classes="move-control",
            )

    def on_click(self) -> None:
        self.post_message(self.Selected(self.form, self.field))

    def on_descendant_focus(self) -> None:
        self.post_message(self.Selected(self.form, self.field))

    async def show_field(self, field: FormField) -> None:
        self.field = field
        options = self._options()
        movable = field.slot_id in self.form.slot_move_states and field.name != "lora_weight"
        if (options is None) != (self.options is None) or movable != self.movable:
            self.options = options
            self.movable = movable
            self.set_class(movable, "movable")
            await self.recompose()
            return
        self.query_one(Label).update(field.label)
        editor = self.query_one(".field-editor")
        if isinstance(editor, Select):
            with editor.prevent(Select.Changed):
                if options != self.options:
                    editor.set_options((option.label, option.value) for option in options)
                editor.value = field.value
        elif isinstance(editor, TextArea):
            if editor.text != field.value:
                editor.load_text(field.value)
        elif isinstance(editor, Input):
            editor.value = field.value
            if isinstance(editor, PathInput):
                editor.field = field
        self.options = options
        if self.movable:
            up, down = self.form.slot_move_states[field.slot_id]
            for button, enabled in zip(self.query(Button), (up, down), strict=True):
                button.disabled = not enabled

    def on_enter(self) -> None:
        self.add_class("hovered")

    def on_leave(self) -> None:
        self.set_class(self.is_mouse_over, "hovered")


class FormFields(VerticalScroll):
    def __init__(self, form: FormModel, *, id: str) -> None:
        super().__init__(id=id, classes="form-fields")
        self.form = form

    def compose(self) -> ComposeResult:
        yield from (FieldRow(self.form, field) for field in self.form.fields)

    async def refresh_fields(self) -> None:
        rows = {(row.field.name, row.field.slot_id): row for row in self.query_children(FieldRow)}
        ordered = []
        for field in self.form.fields:
            row = rows.pop((field.name, field.slot_id), None)
            if row is None:
                row = FieldRow(self.form, field)
                await self.mount(row)
            else:
                await row.show_field(field)
            ordered.append(row)
        for row in rows.values():
            await row.remove()
        for index, row in enumerate(ordered):
            if self.children[index] is not row:
                self.move_child(row, before=index)
