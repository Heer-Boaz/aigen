from __future__ import annotations

import json
import os
import signal
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ItemGrid, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DirectoryTree,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from aigen.character_reference_models import CharacterReferenceError
from aigen.image_postprocess_tui_model import ImagePostprocessForm
from aigen.image_tui_model import (
    DropdownOption,
    FormField,
    ImageEditForm,
)
from aigen.progress import JSON_PROGRESS_PREFIX, format_duration
from aigen.runtime_profiles import (
    PROJECT_ROOT,
    display_project_path,
    resolve_project_path,
)
from aigen.sam_prompt_canvas import SAMPromptCanvas
from aigen.sam_prompt_dialog import SAMPromptDialog
from aigen.sam_prompt_selection import SAMPromptSelection
from aigen.sam_tui_model import SamEditForm
from aigen.video_tui_model import VideoForm


FormModel = ImageEditForm | ImagePostprocessForm | VideoForm | SamEditForm


@dataclass(frozen=True)
class FieldSelection:
    form: FormModel
    field: FormField


CONFIG_ROOT = (
    Path(os.environ["XDG_CONFIG_HOME"]).expanduser()
    if "XDG_CONFIG_HOME" in os.environ
    else Path.home() / ".config"
)
STATE_PATH = CONFIG_ROOT / "aigen" / "image-tui.json"
IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
CONFIG_EXTENSIONS = frozenset({".json"})
SAM_SELECTION_EXTENSIONS = frozenset({".json"})


@dataclass(frozen=True)
class GenerationProgress:
    phase: str
    completed: int
    total: int
    elapsed_seconds: float
    remaining_seconds: float | None
    final: bool
    cpu_percent: float
    gpu_percent: int | None
    vram_used_mb: int | None
    vram_total_mb: int | None


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

    def compose(self) -> ComposeResult:
        yield Label(self.field.label, classes="field-label")
        options = self.form.dropdown_options(self.field)
        if options is None:
            if self.field.slot_kind in {"image", "keyframe", "reference_pack", "config"} or self.field.name == "output_dir":
                yield PathInput(self.form, self.field)
            else:
                yield Input(self.field.value, compact=True, classes="field-editor")
        else:
            option_values = {option.value for option in options}
            if self.field.value not in option_values:
                options = (*options, DropdownOption(self.field.value, self.field.value))
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


class FilteredDirectoryTree(DirectoryTree):
    def __init__(
        self,
        path: Path,
        *,
        directories_only: bool,
        extensions: frozenset[str],
    ) -> None:
        super().__init__(path)
        self.directories_only = directories_only
        self.extensions = extensions

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return (
            path
            for path in paths
            if not path.name.startswith(".")
            and (
                path.is_dir()
                or (
                    not self.directories_only
                    and path.suffix.casefold() in self.extensions
                )
            )
        )


class MessageDialog(ModalScreen[None]):
    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self.title = title
        self.message = message

    def compose(self) -> ComposeResult:
        with Container(classes="dialog"):
            yield Label(self.title, classes="dialog-title")
            yield Static(self.message, classes="dialog-message")
            yield Button("Close", variant="primary", id="close-dialog", compact=True)

    @on(Button.Pressed, "#close-dialog")
    def close_dialog(self) -> None:
        self.dismiss()


class PromptDialog(ModalScreen[str | None]):
    def __init__(self, title: str, label: str, value: str = "") -> None:
        super().__init__()
        self.title = title
        self.label = label
        self.value = value

    def compose(self) -> ComposeResult:
        with Container(classes="dialog"):
            yield Label(self.title, classes="dialog-title")
            yield Label(self.label)
            yield Input(self.value, id="dialog-input", compact=True)
            with Horizontal(classes="dialog-actions"):
                yield Button("OK", variant="primary", id="dialog-ok", compact=True)
                yield Button("Cancel", id="dialog-cancel", compact=True)

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    @on(Input.Submitted, "#dialog-input")
    @on(Button.Pressed, "#dialog-ok")
    def accept(self) -> None:
        self.dismiss(self.query_one(Input).value)

    @on(Button.Pressed, "#dialog-cancel")
    def cancel(self) -> None:
        self.dismiss(None)


class FileBrowser(ModalScreen[Path | None]):
    def __init__(
        self,
        start: Path,
        *,
        title: str,
        directories_only: bool,
        extensions: frozenset[str],
        select_label: str,
    ) -> None:
        super().__init__()
        self.start = start
        self.title = title
        self.directories_only = directories_only
        self.extensions = extensions
        self.select_label = select_label

    def compose(self) -> ComposeResult:
        with Container(classes="browser-dialog"):
            yield Label(self.title, classes="dialog-title")
            yield Label(self.start.as_posix(), id="browser-path")
            yield FilteredDirectoryTree(
                self.start,
                directories_only=self.directories_only,
                extensions=self.extensions,
            )
            with Horizontal(classes="dialog-actions"):
                yield Button("Up", id="browser-up", compact=True)
                yield Button(
                    self.select_label,
                    variant="primary",
                    id="browser-select",
                    compact=True,
                )
                yield Button("Cancel", id="browser-cancel", compact=True)

    @on(DirectoryTree.FileSelected)
    def file_selected(self, event: DirectoryTree.FileSelected) -> None:
        if not self.directories_only:
            self.dismiss(event.path)

    @on(Button.Pressed, "#browser-up")
    async def go_up(self) -> None:
        tree = self.query_one(FilteredDirectoryTree)
        parent = Path(tree.path).parent
        if parent != Path(tree.path):
            tree.path = parent
            await tree.reload()
            self.query_one("#browser-path", Label).update(parent.as_posix())

    @on(Button.Pressed, "#browser-select")
    def select_current(self) -> None:
        tree = self.query_one(FilteredDirectoryTree)
        node = tree.cursor_node
        if node is None or node.data is None:
            return
        path = node.data.path
        if self.directories_only:
            self.dismiss(path if path.is_dir() else path.parent)
        elif path.is_file():
            self.dismiss(path)

    @on(Button.Pressed, "#browser-cancel")
    def cancel(self) -> None:
        self.dismiss(None)


class GenerationUpdated(Message):
    def __init__(self, progress: GenerationProgress) -> None:
        super().__init__()
        self.progress = progress


class GenerationFinished(Message):
    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = output_dir


class GenerationFailed(Message):
    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


class ImageGenerationApp(App[None]):
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("ctrl+c", "quit", show=False),
        Binding(
            "backspace",
            "remove_hovered_slot",
            show=False,
            priority=True,
        ),
        Binding("backspace", "remove_selected_slot", show=False),
        Binding("delete", "clear_selected_field", show=False),
    ]
    CSS = """
    Screen {
        background: #17131f;
        color: #e7e1ed;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 0;
    }

    .form-fields {
        height: 1fr;
        scrollbar-size: 1 1;
        padding: 0 1;
    }

    .field-row {
        layout: grid;
        grid-size: 4 1;
        grid-columns: 18 1fr 0 0;
        grid-rows: 1;
        height: 1;
    }

    .field-row.movable {
        grid-columns: 18 1fr 3 3;
    }

    .field-row.hovered .field-label {
        background: #30273d;
    }

    .field-row.hovered .field-editor {
        background: #d9d1df;
    }

    .field-row.selected .field-label {
        background: #70598a;
        color: #ffffff;
    }

    .field-row.selected .field-editor {
        background: #c8b8d8;
    }

    .field-label {
        height: 1;
        text-style: bold;
        content-align-vertical: middle;
    }

    .field-editor {
        width: 100%;
        height: 1;
        border: none;
        padding: 0;
        background: #e7e1ed;
        color: #17131f;
    }

    .field-editor:focus {
        background: #ffffff;
        color: #17131f;
        text-style: none;
    }

    .move-control {
        width: 3;
        min-width: 3;
        height: 1;
        min-height: 1;
        border: none;
        padding: 0;
    }

    #actions, #postprocess-actions {
        width: 100%;
        height: auto;
        max-height: 4;
        padding: 0 1;
        grid-gutter: 0;
    }

    #video-actions {
        width: 100%;
        height: auto;
        max-height: 4;
        padding: 0 1;
        grid-gutter: 0;
    }

    #sam-actions {
        width: 100%;
        height: auto;
        max-height: 4;
        padding: 0 1;
        grid-gutter: 0;
    }

    .sam-prompt-dialog-screen {
        align: center middle;
        background: #000000 70%;
    }

    .sam-prompt-dialog {
        width: 96%;
        height: 94%;
        min-width: 60;
        min-height: 20;
        padding: 1 2;
        background: #211a2d;
        border: solid #8c72aa;
    }

    #sam-dialog-canvas {
        width: 1fr;
        height: 1fr;
        min-height: 8;
        border: round #70598a;
        background: #111111;
        content-align: center middle;
    }

    #sam-prompt-dialog-actions {
        width: 100%;
        height: auto;
        max-height: 2;
        padding: 0;
        grid-gutter: 0;
    }

    #sam-prompt-dialog-actions Button {
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 1;
    }

    #actions Button, #video-actions Button, #sam-actions Button, #postprocess-actions Button, .dialog-actions Button {
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 1;
    }

    #generation-progress {
        height: 1;
        display: none;
    }

    #status {
        height: 1;
        padding: 0 1;
        background: #e7e1ed;
        color: #17131f;
    }

    ModalScreen {
        align: center middle;
        background: #000000 55%;
    }

    .dialog {
        width: 80%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: #211a2d;
        border: solid #8c72aa;
    }

    .browser-dialog {
        width: 90%;
        height: 85%;
        padding: 1 2;
        background: #211a2d;
        border: solid #8c72aa;
    }

    .dialog-title {
        height: 1;
        text-style: bold;
    }

    .dialog-message {
        height: auto;
        max-height: 1fr;
        overflow-y: auto;
    }

    .dialog-actions {
        height: 1;
        align-horizontal: center;
    }

    DirectoryTree {
        height: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.form = ImageEditForm()
        self.video_form = VideoForm()
        self.sam_form = SamEditForm()
        self.postprocess_form = ImagePostprocessForm()
        self.configuration_path: Path | None = None
        self.sam_selection_path: Path | None = None
        self.sam_prompt_dialog: SAMPromptDialog | None = None
        self.selected_field: FieldSelection | None = None
        self.process: subprocess.Popen[str] | None = None
        self.active_action_button_id: str | None = None
        self.active_action_idle_label = ""
        self.generation_progress: GenerationProgress | None = None
        self.startup_error: str | None = None
        try:
            if STATE_PATH.exists():
                self.form.load(STATE_PATH)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.startup_error = str(error)
        self.form.set_value(self.form.field("model"), self.form.field("model").value)

    def compose(self) -> ComposeResult:
        with TabbedContent(initial="images", id="tabs"):
            with TabPane("Images", id="images"):
                yield FormFields(self.form, id="image-fields")
            with TabPane("Videos", id="videos"):
                yield FormFields(self.video_form, id="video-fields")
                yield ItemGrid(
                    Button("+ Keyframe", name="add-keyframe", id="video-add-keyframe", compact=True),
                    Button("+ Seed", name="add-video-seed", id="video-add-seed", compact=True),
                    Button("+ Image", name="add-video-image", id="video-add-image", compact=True),
                    Button("Remove", name="remove-video", compact=True),
                    Button("Browse", name="browse-video", compact=True),
                    Button(
                        "Generate",
                        name="video-generate",
                        id="video-action",
                        variant="primary",
                        compact=True,
                    ),
                    min_column_width=12,
                    stretch_height=False,
                    id="video-actions",
                )
            with TabPane("SAM Edit", id="sam-edit"):
                yield FormFields(self.sam_form, id="sam-fields")
                yield ItemGrid(
                    Button("Browse", name="browse-sam", compact=True),
                    Button("Edit prompts", name="sam-edit", id="sam-edit-prompts", compact=True),
                    Button("Clear prompts", name="sam-clear", compact=True),
                    Button(
                        "Run",
                        name="sam-segment",
                        id="sam-action",
                        variant="primary",
                        compact=True,
                    ),
                    min_column_width=12,
                    stretch_height=False,
                    id="sam-actions",
                )
            with TabPane("Post-processing", id="postprocessing"):
                yield FormFields(self.postprocess_form, id="postprocess-fields")
                yield ItemGrid(
                    Button(
                        "Process",
                        name="postprocess",
                        id="postprocess-action",
                        variant="primary",
                        compact=True,
                    ),
                    min_column_width=12,
                    stretch_height=False,
                    id="postprocess-actions",
                )
        yield ItemGrid(
            Button("+ Seed", name="add-seed", compact=True),
            Button("+ Image", name="add-image", compact=True),
            Button("+ Pack", name="add-reference-pack", compact=True),
            Button("+ LoRA", name="add-lora", compact=True),
            Button("Remove", name="remove", compact=True),
            Button("Browse", name="browse", compact=True),
            Button("Use Result", name="use-result", compact=True),
            Button("Save Pack", name="save-pack", compact=True),
            Button("Save Config", name="save-config", compact=True),
            Button("Load Config", name="load-config", compact=True),
            Button(
                "Generate",
                name="generate",
                id="generation-action",
                variant="primary",
                compact=True,
            ),
            Button("Quit", name="quit", compact=True),
            min_column_width=12,
            stretch_height=False,
            id="actions",
        )
        yield ProgressBar(id="generation-progress")
        yield Static("Ready.", id="status")

    def on_mount(self) -> None:
        if self.startup_error is not None:
            self._show_error("Cannot load saved form", self.startup_error)
        self._update_sam_prompt_canvas()

    async def _rebuild_fields(self) -> None:
        await self._rebuild_form(self.form)

    async def _rebuild_form(self, form: FormModel) -> None:
        if form is self.form:
            fields_id = "#image-fields"
        elif form is self.video_form:
            fields_id = "#video-fields"
        elif form is self.sam_form:
            fields_id = "#sam-fields"
        else:
            fields_id = "#postprocess-fields"
        if (
            self.selected_field is not None
            and self.selected_field.form is form
            and not any(field is self.selected_field.field for field in form.fields)
        ):
            self._select_field(None)
        await self.query_one(fields_id, FormFields).recompose()
        self._refresh_selected_field()
        if form is self.video_form:
            self._update_video_actions()
        if form is self.sam_form:
            self._update_sam_prompt_canvas()

    def _update_video_actions(self) -> None:
        self.query_one("#video-add-keyframe", Button).disabled = (
            not self.video_form.can_add_slot("keyframe")
        )
        self.query_one("#video-add-seed", Button).disabled = (
            not self.video_form.can_add_slot("seed")
        )
        self.query_one("#video-add-image", Button).disabled = (
            not self.video_form.can_add_slot("image")
        )

    def _update_sam_prompt_canvas(self) -> None:
        form = self.sam_form
        active = (
            form.field("operation").value == "segment"
            and form.field("engine").value != "anime"
            and form.field("prompt_mode").value in {"box", "points", "box+points"}
        )
        self.query_one("#sam-edit-prompts", Button).disabled = not active
        if self.sam_prompt_dialog is not None:
            self.sam_prompt_dialog.set_state(
                image=form.field("input").value,
                prompt_mode=form.field("prompt_mode").value,
                box=form.field("box").value,
                positive_points=form.field("positive_points").value,
                negative_points=form.field("negative_points").value,
            )

    def _open_sam_prompt_editor(self) -> None:
        form = self.sam_form
        if self.sam_prompt_dialog is not None:
            return
        dialog = SAMPromptDialog(
            image=form.field("input").value,
            prompt_mode=form.field("prompt_mode").value,
            box=form.field("box").value,
            positive_points=form.field("positive_points").value,
            negative_points=form.field("negative_points").value,
        )
        self.sam_prompt_dialog = dialog
        self.push_screen(dialog, self._close_sam_prompt_editor)

    def _close_sam_prompt_editor(self, _: None) -> None:
        self.sam_prompt_dialog = None
        self._update_sam_prompt_canvas()

    def _sam_prompt_selection(self) -> SAMPromptSelection:
        form = self.sam_form
        return SAMPromptSelection(
            image=form.field("input").value,
            prompt_mode=form.field("prompt_mode").value,
            box=form.field("box").value,
            positive_points=form.field("positive_points").value,
            negative_points=form.field("negative_points").value,
        )

    def _choose_sam_selection_directory(self) -> None:
        start = self.sam_selection_path.parent if self.sam_selection_path else PROJECT_ROOT
        self.push_screen(
            FileBrowser(
                start,
                title="Save SAM selection",
                directories_only=True,
                extensions=SAM_SELECTION_EXTENSIONS,
                select_label="Select folder",
            ),
            self._choose_sam_selection_name,
        )

    def _choose_sam_selection_name(self, directory: Path | None) -> None:
        if directory is None:
            return
        name = self.sam_selection_path.name if self.sam_selection_path else "sam-selection.json"
        self.push_screen(
            PromptDialog("Save SAM selection", "Selection filename", name),
            lambda filename: self._save_sam_prompt_selection(directory, filename),
        )

    def _save_sam_prompt_selection(
        self,
        directory: Path | None = None,
        filename: str | None = None,
    ) -> None:
        if directory is None:
            self._choose_sam_selection_directory()
            return
        if filename is None:
            return
        filename = filename.strip()
        if not filename or Path(filename).name != filename:
            self._show_error(
                "Cannot save SAM selection",
                "Selection filename must be a non-empty filename without a path.",
            )
            return
        output = directory / filename
        if output.suffix == "":
            output = output.with_suffix(".json")
        elif output.suffix.casefold() != ".json":
            self._show_error(
                "Cannot save SAM selection",
                "Selection filename must use the .json extension.",
            )
            return
        if output.exists() and output != self.sam_selection_path:
            self._show_error("Cannot save SAM selection", f"Selection already exists: {output}")
            return
        try:
            self._sam_prompt_selection().save(output)
        except (OSError, ValueError) as error:
            self._show_error("Cannot save SAM selection", str(error))
            return
        self.sam_selection_path = output
        self._set_status(f"Saved SAM selection: {display_project_path(output)}")

    def _load_sam_prompt_selection(self) -> None:
        start = self.sam_selection_path or PROJECT_ROOT
        self.push_screen(
            FileBrowser(
                self._browser_start(start.as_posix()),
                title="Load SAM selection",
                directories_only=False,
                extensions=SAM_SELECTION_EXTENSIONS,
                select_label="Load",
            ),
            self._apply_sam_prompt_selection,
        )

    def _apply_sam_prompt_selection(self, path: Path | None) -> None:
        if path is None:
            return
        try:
            selection = SAMPromptSelection.load(path)
        except (OSError, ValueError, TypeError) as error:
            self._show_error("Cannot load SAM selection", str(error))
            return
        form = self.sam_form
        form.set_value(form.field("prompt_mode"), selection.prompt_mode)
        form.set_value(form.field("box"), selection.box)
        form.set_value(form.field("positive_points"), selection.positive_points)
        form.set_value(form.field("negative_points"), selection.negative_points)
        self.sam_selection_path = path
        self._set_status(f"Loaded SAM selection: {display_project_path(path)}")
        self.run_worker(self._rebuild_form(form), group="fields", exclusive=True)

    @on(FieldRow.Selected)
    def field_selected(self, event: FieldRow.Selected) -> None:
        self._select_field(FieldSelection(event.form, event.field))

    def _select_field(self, selection: FieldSelection | None) -> None:
        if (
            self.selected_field is not None
            and selection is not None
            and self.selected_field.form is selection.form
            and self.selected_field.field is selection.field
        ):
            return
        self.selected_field = selection
        self._refresh_selected_field()

    def _refresh_selected_field(self) -> None:
        for row in self.query(FieldRow):
            selected = self.selected_field
            row.set_class(
                selected is not None
                and selected.form is row.form
                and selected.field is row.field,
                "selected",
            )

    def _hovered_row(self) -> FieldRow | None:
        return next(
            (row for row in self.query(FieldRow) if row.has_class("hovered")),
            None,
        )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        hovered = self._hovered_row()
        if action == "remove_hovered_slot":
            return hovered is not None and hovered.field.slot_id is not None
        if action == "remove_selected_slot":
            return hovered is None
        return super().check_action(action, parameters)

    @on(Input.Changed)
    def input_changed(self, event: Input.Changed) -> None:
        row = event.input.parent
        if isinstance(row, FieldRow):
            if row.field.value == event.value:
                return
            row.form.set_value(row.field, event.value)
            self._select_field(FieldSelection(row.form, row.field))
            if row.form is self.sam_form and row.field.name in {
                "input",
                "box",
                "positive_points",
                "negative_points",
            }:
                self._update_sam_prompt_canvas()

    @on(Select.Changed)
    async def select_changed(self, event: Select.Changed) -> None:
        if event.value is Select.NULL:
            return
        row = event.select.parent
        if not isinstance(row, FieldRow):
            return
        value = str(event.value)
        if row.field.value == value:
            return
        self._select_field(FieldSelection(row.form, row.field))
        row.form.set_value(row.field, value)
        if row.field.name in {"operation", "model", "prompt_mode", "engine"}:
            await self._rebuild_form(row.form)

    @on(PathInput.BrowseRequested)
    def browse_requested(self, event: PathInput.BrowseRequested) -> None:
        self._browse_field(event.form, event.field)

    @on(SAMPromptCanvas.PromptChanged)
    def sam_prompt_changed(self, event: SAMPromptCanvas.PromptChanged) -> None:
        form = self.sam_form
        form.set_value(form.field("box"), event.box)
        form.set_value(form.field("positive_points"), event.positive_points)
        form.set_value(form.field("negative_points"), event.negative_points)
        self._update_sam_prompt_canvas()

    @on(TabbedContent.TabActivated)
    def tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self.query_one("#actions").display = event.pane.id == "images"

    @on(Button.Pressed)
    async def button_pressed(self, event: Button.Pressed) -> None:
        button = event.button
        row = button.parent
        if isinstance(row, FieldRow):
            assert row.field.slot_id is not None
            self._select_field(FieldSelection(row.form, row.field))
            row.form.move_slot(
                row.field.slot_id,
                -1 if button.name == "move-up" else 1,
            )
            await self._rebuild_form(row.form)
            return

        action = button.name
        if action is None:
            return
        if action.startswith("add-"):
            if action == "add-video-seed":
                slot_kind = "seed"
                form = self.video_form
            elif action == "add-video-image":
                slot_kind = "image"
                form = self.video_form
            elif action == "add-keyframe":
                slot_kind = "keyframe"
                form = self.video_form
            else:
                slot_kind = action.removeprefix("add-").replace("-", "_")
                form = self.form
            try:
                slot_id = form.add_slot(slot_kind)
            except ValueError as error:
                self._show_error("Cannot add video slot", str(error))
                return
            self._select_field(
                FieldSelection(
                    form,
                    next(field for field in form.fields if field.slot_id == slot_id),
                )
            )
            await self._rebuild_form(form)
        elif action == "remove":
            await self.action_remove_selected_slot()
        elif action == "browse":
            if self.selected_field is not None:
                self._browse_field(
                    self.selected_field.form,
                    self.selected_field.field,
                )
            else:
                self._set_status("Select an Image or Output directory field to browse.")
        elif action == "use-result":
            self._browse_result()
        elif action == "save-pack":
            self.push_screen(
                PromptDialog("Save reference pack", "Pack / character id"),
                self._save_reference_pack,
            )
        elif action == "save-config":
            self._choose_configuration_directory()
        elif action == "load-config":
            self._load_configuration()
        elif action == "generate":
            if self.process is None:
                self._start_generation()
            else:
                self._cancel_generation()
        elif action == "postprocess":
            if self.process is None:
                self._start_postprocess()
            else:
                self._cancel_generation()
        elif action == "video-generate":
            if self.process is None:
                self._start_video()
            else:
                self._cancel_generation()
        elif action == "sam-segment":
            if self.process is None:
                self._start_sam()
            else:
                self._cancel_generation()
        elif action == "remove-video":
            if self.selected_field is not None and self.selected_field.form is self.video_form:
                await self._remove_slot(self.selected_field)
            else:
                self._set_status("Select a video keyframe or seed slot.")
        elif action == "browse-video":
            if self.selected_field is not None and self.selected_field.form is self.video_form:
                self._browse_field(self.selected_field.form, self.selected_field.field)
            else:
                self._set_status("Select a video input or Output directory field to browse.")
        elif action == "browse-sam":
            if self.selected_field is not None and self.selected_field.form is self.sam_form:
                self._browse_field(self.selected_field.form, self.selected_field.field)
            else:
                self._set_status("Select a SAM file or Output directory field to browse.")
        elif action == "sam-edit":
            self._open_sam_prompt_editor()
        elif action == "sam-prompt-clear":
            assert self.sam_prompt_dialog is not None
            self.sam_prompt_dialog.clear_prompts()
        elif action == "sam-prompt-save":
            self._save_sam_prompt_selection()
        elif action == "sam-prompt-load":
            self._load_sam_prompt_selection()
        elif action == "sam-prompt-close":
            assert self.sam_prompt_dialog is not None
            self.sam_prompt_dialog.close()
        elif action == "sam-clear":
            form = self.sam_form
            form.set_value(form.field("box"), "")
            form.set_value(form.field("positive_points"), "")
            form.set_value(form.field("negative_points"), "")
            self._update_sam_prompt_canvas()
        elif action == "quit":
            self.action_quit()

    async def action_remove_selected_slot(self) -> None:
        await self._remove_slot(self.selected_field)

    async def action_remove_hovered_slot(self) -> None:
        hovered = self._hovered_row()
        assert hovered is not None and hovered.field.slot_id is not None
        await self._remove_slot(FieldSelection(hovered.form, hovered.field))

    async def _remove_slot(
        self,
        selection: FieldSelection | None,
    ) -> None:
        if selection is None or selection.field.slot_id is None:
            self._set_status("Select a seed, image, reference pack or LoRA slot.")
            return
        selection.form.remove_slot(selection.field.slot_id)
        if (
            self.selected_field is not None
            and self.selected_field == selection
        ):
            self._select_field(None)
        await self._rebuild_form(selection.form)

    async def action_clear_selected_field(self) -> None:
        if self.selected_field is None:
            self._set_status("Select a field to clear.")
            return
        form = self.selected_field.form
        field = self.selected_field.field
        options = form.dropdown_options(field)
        if options is not None and not any(option.value == "" for option in options):
            self._set_status(f"{field.label} cannot be empty.")
            return
        form.set_value(field, "")
        await self._rebuild_form(form)

    def _browse_field(self, form: FormModel, field: FormField) -> None:
        if field.slot_kind in {"image", "keyframe"}:
            directories_only = False
            title = field.label
            select_label = "Select image"
            extensions = IMAGE_EXTENSIONS
        elif field.slot_kind == "reference_pack":
            directories_only = False
            title = field.label
            select_label = "Select pack"
            extensions = CONFIG_EXTENSIONS
        elif field.slot_kind == "config":
            directories_only = False
            title = field.label
            select_label = "Select JSON"
            extensions = CONFIG_EXTENSIONS
        elif field.name == "output_dir":
            directories_only = True
            title = field.label
            select_label = "Select folder"
            extensions = IMAGE_EXTENSIONS
        else:
            self._set_status("Select a file or Output directory field to browse.")
            return
        start = self._browser_start(field.value)
        self.push_screen(
            FileBrowser(
                start,
                title=title,
                directories_only=directories_only,
                extensions=extensions,
                select_label=select_label,
            ),
            lambda path: self._set_browsed_path(form, field, path),
        )

    def _set_browsed_path(
        self,
        form: FormModel,
        field: FormField,
        path: Path | None,
    ) -> None:
        if path is not None:
            form.set_value(field, display_project_path(path))
            self.run_worker(
                self._rebuild_form(form),
                group="fields",
                exclusive=True,
            )

    def _browse_result(self) -> None:
        output = resolve_project_path(self.form.field("output_dir").value)
        if not output.is_dir():
            self._set_status(f"Output directory does not exist: {output}")
            return
        self.push_screen(
            FileBrowser(
                output,
                title="Use result",
                directories_only=False,
                extensions=IMAGE_EXTENSIONS,
                select_label="Use image",
            ),
            self._use_result,
        )

    def _use_result(self, path: Path | None) -> None:
        if path is None:
            return
        field = next(
            (
                field
                for field in self.form.fields
                if field.slot_kind == "image" and not field.value.strip()
            ),
            None,
        )
        if field is None:
            slot_id = self.form.add_slot("image")
            field = next(field for field in self.form.fields if field.slot_id == slot_id)
        field.value = display_project_path(path)
        self._select_field(FieldSelection(self.form, field))
        self.run_worker(self._rebuild_fields(), group="fields", exclusive=True)

    def _save_reference_pack(self, pack_id: str | None) -> None:
        if pack_id is None:
            return
        pack_id = pack_id.strip()
        if not pack_id or Path(pack_id).name != pack_id:
            self._show_error(
                "Cannot save reference pack",
                "Pack / character id must be a non-empty filename without a path.",
            )
            return
        try:
            output = self.form.save_reference_pack(pack_id)
        except (CharacterReferenceError, OSError) as error:
            self._show_error("Cannot save reference pack", str(error))
            return
        field = next(
            (
                field
                for field in self.form.fields
                if field.slot_kind == "reference_pack" and not field.value.strip()
            ),
            None,
        )
        if field is None:
            slot_id = self.form.add_slot("reference_pack")
            field = next(field for field in self.form.fields if field.slot_id == slot_id)
        field.value = display_project_path(output)
        self._select_field(FieldSelection(self.form, field))
        self._set_status(f"Saved reference pack: {field.value}")
        self.run_worker(self._rebuild_fields(), group="fields", exclusive=True)

    def _choose_configuration_directory(self) -> None:
        start = (
            self.configuration_path.parent
            if self.configuration_path is not None
            else PROJECT_ROOT
        )
        self.push_screen(
            FileBrowser(
                start,
                title="Save configuration",
                directories_only=True,
                extensions=CONFIG_EXTENSIONS,
                select_label="Select folder",
            ),
            self._choose_configuration_name,
        )

    def _choose_configuration_name(self, directory: Path | None) -> None:
        if directory is None:
            return
        name = self.configuration_path.name if self.configuration_path else "image-edit.json"
        self.push_screen(
            PromptDialog("Save configuration", "Configuration filename", name),
            lambda filename: self._save_configuration(directory, filename),
        )

    def _save_configuration(self, directory: Path, filename: str | None) -> None:
        if filename is None:
            return
        filename = filename.strip()
        if not filename or Path(filename).name != filename:
            self._show_error(
                "Cannot save configuration",
                "Configuration filename must be a non-empty filename without a path.",
            )
            return
        output = directory / filename
        if output.suffix == "":
            output = output.with_suffix(".json")
        elif output.suffix.casefold() != ".json":
            self._show_error(
                "Cannot save configuration",
                "Configuration filename must use the .json extension.",
            )
            return
        if output.exists() and output != self.configuration_path:
            self._show_error(
                "Cannot save configuration", f"Configuration already exists: {output}"
            )
            return
        try:
            self.form.save(output)
        except OSError as error:
            self._show_error("Cannot save configuration", str(error))
            return
        self.configuration_path = output
        self._set_status(f"Saved configuration: {display_project_path(output)}")

    def _load_configuration(self) -> None:
        if self.process is not None:
            self._set_status("Stop generation before loading a configuration.")
            return
        start = self.configuration_path or PROJECT_ROOT
        self.push_screen(
            FileBrowser(
                self._browser_start(start.as_posix()),
                title="Load configuration",
                directories_only=False,
                extensions=CONFIG_EXTENSIONS,
                select_label="Load",
            ),
            self._apply_configuration,
        )

    def _apply_configuration(self, path: Path | None) -> None:
        if path is None:
            return
        try:
            self.form.load(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._show_error("Cannot load configuration", str(error))
            return
        self.configuration_path = path
        self._select_field(None)
        self._set_status(f"Loaded configuration: {display_project_path(path)}")
        self.run_worker(self._rebuild_fields(), group="fields", exclusive=True)

    def _start_generation(self) -> None:
        if self.process is not None:
            self._set_status("Generation is already running.")
            return
        try:
            command, output_dir = self.form.generation_command()
        except ValueError as error:
            self._show_error("Cannot start generation", str(error))
            return
        self._start_command(
            command,
            output_dir,
            action_button_id="generation-action",
            idle_label="Generate",
            error_title="Cannot start generation",
        )

    def _start_postprocess(self) -> None:
        if self.process is not None:
            self._set_status("Processing is already running.")
            return
        try:
            command, output_dir = self.postprocess_form.generation_command()
        except ValueError as error:
            self._show_error("Cannot start post-processing", str(error))
            return
        self._start_command(
            command,
            output_dir,
            action_button_id="postprocess-action",
            idle_label="Process",
            error_title="Cannot start post-processing",
        )

    def _start_video(self) -> None:
        if self.process is not None:
            self._set_status("Video generation is already running.")
            return
        try:
            command, output_dir = self.video_form.generation_command()
        except ValueError as error:
            self._show_error("Cannot start video generation", str(error))
            return
        self._start_command(
            command,
            output_dir,
            action_button_id="video-action",
            idle_label="Generate",
            error_title="Cannot start video generation",
        )

    def _start_sam(self) -> None:
        if self.process is not None:
            self._set_status("SAM operation is already running.")
            return
        try:
            command, output_dir = self.sam_form.generation_command()
        except ValueError as error:
            self._show_error("Cannot start SAM operation", str(error))
            return
        self._start_command(
            command,
            output_dir,
            action_button_id="sam-action",
            idle_label="Run",
            error_title="Cannot start SAM operation",
        )

    def _start_command(
        self,
        command: list[str],
        output_dir: str,
        *,
        action_button_id: str,
        idle_label: str,
        error_title: str,
    ) -> None:
        environment = os.environ.copy()
        environment["AIGEN_PROGRESS"] = "json"
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
        except OSError as error:
            self._show_error(error_title, str(error))
            return
        self.process = process
        self.active_action_button_id = action_button_id
        self.active_action_idle_label = idle_label
        self.generation_progress = None
        self._set_status("Starting...")
        self.query_one(f"#{action_button_id}", Button).label = "Stop"
        for button_id in (
            "generation-action",
            "video-action",
            "sam-action",
            "postprocess-action",
        ):
            self.query_one(f"#{button_id}", Button).disabled = button_id != action_button_id
        self.run_worker(
            lambda: self._watch_generation(process, output_dir),
            thread=True,
            name="image-generation",
            exit_on_error=False,
        )

    def _watch_generation(self, process: subprocess.Popen[str], output_dir: str) -> None:
        assert process.stdout is not None
        output_lines: deque[str] = deque(maxlen=200)
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            if not line.startswith(JSON_PROGRESS_PREFIX):
                output_lines.append(line)
                continue
            payload = json.loads(line[len(JSON_PROGRESS_PREFIX) :])
            self.post_message(
                GenerationUpdated(
                    GenerationProgress(
                        phase=payload["phase"],
                        completed=payload["completed"],
                        total=payload["total"],
                        elapsed_seconds=payload["elapsed_seconds"],
                        remaining_seconds=payload["remaining_seconds"],
                        final=payload["final"],
                        cpu_percent=payload["cpu_percent"],
                        gpu_percent=payload["gpu_percent"],
                        vram_used_mb=payload["vram_used_mb"],
                        vram_total_mb=payload["vram_total_mb"],
                    )
                )
            )
        returncode = process.wait()
        if returncode == 0:
            self.post_message(GenerationFinished(output_dir))
        else:
            self.post_message(
                GenerationFailed(self._error_message("\n".join(output_lines), returncode))
            )

    @on(GenerationUpdated)
    def generation_updated(self, event: GenerationUpdated) -> None:
        self.generation_progress = event.progress
        progress_bar = self.query_one("#generation-progress", ProgressBar)
        progress_bar.display = event.progress.total > 0
        if event.progress.total:
            progress_bar.update(
                total=event.progress.total,
                progress=event.progress.completed,
            )
        self._set_status(self._progress_text(event.progress))

    @on(GenerationFinished)
    def generation_finished(self, event: GenerationFinished) -> None:
        self._generation_stopped()
        self._set_status(f"Output: {event.output_dir}")

    @on(GenerationFailed)
    def generation_failed(self, event: GenerationFailed) -> None:
        self._generation_stopped()
        self._show_error("Generation failed", event.error)

    def _generation_stopped(self) -> None:
        self.process = None
        self.generation_progress = None
        self.query_one("#generation-progress", ProgressBar).display = False
        if self.active_action_button_id is not None:
            self.query_one(f"#{self.active_action_button_id}", Button).label = (
                self.active_action_idle_label
            )
        for button_id in (
            "generation-action",
            "video-action",
            "sam-action",
            "postprocess-action",
        ):
            self.query_one(f"#{button_id}", Button).disabled = False
        self.active_action_button_id = None
        self.active_action_idle_label = ""

    def _cancel_generation(self) -> None:
        if self.process is None:
            self._set_status("No generation is running.")
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        self._set_status("Stopping generation...")

    def _stop_generation(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def action_quit(self) -> None:
        try:
            self.form.save(STATE_PATH)
        except OSError as error:
            self._show_error("Cannot save form", str(error))
            return
        self._stop_generation()
        self.exit()

    def _show_error(self, title: str, message: str) -> None:
        self.push_screen(MessageDialog(title, message))

    def _set_status(self, status: str) -> None:
        self.query_one("#status", Static).update(status)

    @staticmethod
    def _browser_start(value: str) -> Path:
        path = resolve_project_path(value) if value.strip() else PROJECT_ROOT
        if path.is_dir():
            return path
        if path.parent.is_dir():
            return path.parent
        return PROJECT_ROOT

    @staticmethod
    def _error_message(output: str, returncode: int) -> str:
        if output:
            try:
                payload = json.loads(output)
            except json.JSONDecodeError:
                return output
            message = payload.get("message")
            return message if isinstance(message, str) else output
        return f"Image generation exited with code {returncode}."

    @staticmethod
    def _progress_text(progress: GenerationProgress) -> str:
        parts = [progress.phase]
        if progress.total and not progress.final and progress.completed < progress.total:
            parts.append(
                "eta --:--"
                if progress.remaining_seconds is None
                else f"eta {format_duration(progress.remaining_seconds)}"
            )
        parts.append(f"elapsed {format_duration(progress.elapsed_seconds)}")
        parts.append(f"cpu {progress.cpu_percent:5.1f}%")
        if progress.gpu_percent is None:
            parts.extend(("gpu n/a", "vram n/a"))
        else:
            assert progress.vram_used_mb is not None
            assert progress.vram_total_mb is not None
            vram_percent = round(progress.vram_used_mb * 100 / progress.vram_total_mb)
            parts.extend(
                (
                    f"gpu {progress.gpu_percent:3d}%",
                    f"vram {progress.vram_used_mb}/{progress.vram_total_mb} MB ({vram_percent}%)",
                )
            )
        return " | ".join(parts)


def main() -> None:
    ImageGenerationApp().run()


if __name__ == "__main__":
    main()
