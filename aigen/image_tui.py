from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from uuid import uuid4

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widgets import (
    Button,
    Input,
    ProgressBar,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.worker import Worker, WorkerCancelled

from aigen.character_reference_models import CharacterReferenceError
from aigen.generation.video_postprocess import contact_sheet_path
from aigen.tui_process import command_lines, command_process
from aigen.image_tui_footer import ImageTUIFooter
from aigen.image_tui_model import (
    FormField,
    ImageEditForm,
)
from aigen.postprocess_tui_model import PostprocessForm
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
from aigen.tui_file_browser import FileBrowser
from aigen.tui_dialogs import (
    ConfirmationDialog,
    MessageDialog,
    PromptDialog,
)
from aigen.video_tui_model import VideoForm
from aigen.workflow_commands import DEFAULT_WORKFLOW_RUNS_ROOT
from aigen.workflow_document_list import WorkflowDocumentList
from aigen.workflow_document_io import (
    load_workflow_document,
    save_workflow_document,
)
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_run_state import WorkflowRunState
from aigen.workflow_run_records import finalize_terminated_workflow, new_workflow_run_id, workflow_run_path
from aigen.workflow_editor import WorkflowEditor
from aigen.workflow_execution import WORKFLOW_EVENT_PREFIX
from aigen.workflow_graph import (
    AudioSourceNode,
    ImageSourceNode,
    LoraSourceNode,
    ReferencePackNode,
    VideoSourceNode,
    WorkflowGraph,
)
from aigen.workflow_templates import character_workflow_template, image_style_workflow_template, keyframed_video_workflow_template
from aigen.workflow_form_import import image_form_workflow, video_form_workflow
from aigen.workflow_sam_import import sam_form_workflow


from aigen.tui_form_fields import FieldRow, FormFields, FormModel, PathInput


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
VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
AUDIO_EXTENSIONS = VIDEO_EXTENSIONS | frozenset({".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"})
CONFIG_EXTENSIONS = frozenset({".json"})
SAM_SELECTION_EXTENSIONS = frozenset({".json"})
LORA_EXTENSIONS = frozenset({".safetensors"})
WORKFLOW_EXTENSIONS = frozenset({".json"})
TAB_ACTION_BUTTON_IDS = (
    "generation-action",
    "video-action",
    "sam-action",
    "postprocess-action",
)


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


def _generation_progress_from_payload(
    payload: Mapping[str, object],
) -> GenerationProgress:
    remaining = payload["remaining_seconds"]
    gpu = payload["gpu_percent"]
    vram_used = payload["vram_used_mb"]
    vram_total = payload["vram_total_mb"]
    return GenerationProgress(
        phase=str(payload["phase"]),
        completed=int(payload["completed"]),
        total=int(payload["total"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        remaining_seconds=None if remaining is None else float(remaining),
        final=bool(payload["final"]),
        cpu_percent=float(payload["cpu_percent"]),
        gpu_percent=None if gpu is None else int(gpu),
        vram_used_mb=None if vram_used is None else int(vram_used),
        vram_total_mb=None if vram_total is None else int(vram_total),
    )




class GenerationUpdated(Message):
    def __init__(self, progress: GenerationProgress) -> None:
        super().__init__()
        self.progress = progress


class GenerationFinished(Message):
    def __init__(
        self,
        output_dir: str,
        contact_sheets: tuple[Path, ...] = (),
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.contact_sheets = contact_sheets


class GenerationFailed(Message):
    def __init__(self, error: str, title: str = "Generation failed") -> None:
        super().__init__()
        self.error = error
        self.title = title


class GenerationCancelled(Message):
    pass


class WorkflowNodeUpdated(Message):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__()
        self.payload = payload
        self.node_id = str(payload["node_id"])
        self.status = str(payload["status"])
        progress = payload.get("progress")
        self.progress = _generation_progress_from_payload(progress) if progress is not None else None


class ImageGenerationApp(App[None]):
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("ctrl+c", "quit", show=False),
        Binding("shift+f5", "stop_generation", show=False, priority=True),
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

    .field-row.multiline {
        grid-size: 1 2;
        grid-columns: 1fr;
        grid-rows: 1 auto;
        height: auto;
    }

    .field-row MultilineInput.field-editor {
        height: 25vh;
        min-height: 4;
        max-height: 12;
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

    .dialog-actions Button {
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

    """

    def __init__(self) -> None:
        super().__init__()
        self.form = ImageEditForm()
        self.video_form = VideoForm()
        self.sam_form = SamEditForm()
        self.postprocess_form = PostprocessForm()
        self.workflow_buffer = WorkflowEditBuffer(
            image_style_workflow_template()
        )
        self.workflow_document_paths: list[Path] = []
        self.workflow_editor: WorkflowEditor | None = None
        self.workflow_run_state = WorkflowRunState()
        self.workflow_request_path: Path | None = None
        self.workflow_run_dir: Path | None = None
        self.configuration_path: Path | None = None
        self.sam_selection_path: Path | None = None
        self.sam_prompt_dialog: SAMPromptDialog | None = None
        self.selected_field: FieldSelection | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.generation_worker: Worker[None] | None = None
        self.cancel_requested = False
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
            with TabPane("SAM Edit", id="sam-edit"):
                yield FormFields(self.sam_form, id="sam-fields")
            with TabPane("Post-processing", id="postprocessing"):
                yield FormFields(self.postprocess_form, id="postprocess-fields")
            with TabPane("Workflows", id="workflows"):
                yield WorkflowDocumentList(
                    self.workflow_document_paths,
                    current_path=self.workflow_buffer.document_path,
                    current_name=self.workflow_buffer.document.name,
                    dirty=self.workflow_buffer.dirty,
                    id="workflow-documents",
                )
        yield ImageTUIFooter(id="action-footer")
        yield ProgressBar(id="generation-progress")
        yield Static("Ready.", id="status")

    def on_mount(self) -> None:
        if self.startup_error is not None:
            self._show_error("Cannot load saved form", self.startup_error)
        self._update_sam_prompt_canvas()
        self._update_video_actions()

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
        if self.selected_field is not None and self.selected_field.form is form:
            selected = self.selected_field.field
            replacement = next((field for field in form.fields if (field.name, field.slot_id) == (selected.name, selected.slot_id)), None)
            self.selected_field = FieldSelection(form, replacement) if replacement is not None else None
        await self.query_one(fields_id, FormFields).refresh_fields()
        self._refresh_selected_field()
        if form is self.video_form:
            self._update_video_actions()
        if form is self.sam_form:
            self._update_sam_prompt_canvas()

    def _update_video_actions(self) -> None:
        footer = self.query_one(ImageTUIFooter)
        for kind, command in (("keyframe", "add-keyframe"), ("seed", "add-video-seed"), ("image", "add-video-image")):
            footer.set_action_enabled(command, self.video_form.can_add_slot(kind))

    def _update_sam_prompt_canvas(self) -> None:
        form = self.sam_form
        self.query_one("#sam-workflow", Button).disabled = form.field("operation").value == "region-plan"
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
        path = self.sam_selection_path
        self.push_screen(
            FileBrowser(
                path.parent if path is not None else PROJECT_ROOT,
                title='Save SAM selection', directories_only=False,
                extensions=SAM_SELECTION_EXTENSIONS, select_label="Save",
                save_name=path.name if path is not None else 'sam-selection.json',
            ),
            self._save_sam_prompt_selection,
        )

    def _save_sam_prompt_selection(self, output: Path | None) -> None:
        if output is None:
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
        footer = self.query_one(ImageTUIFooter)
        selection = self.selected_field
        for form, browse, remove in ((self.form, "browse", "remove"), (self.video_form, "browse-video", "remove-video"),
                                     (self.sam_form, "browse-sam", None)):
            field = selection.field if selection is not None and selection.form is form else None
            footer.set_action_enabled(browse, field is not None and field.path_kind is not None)
            if remove is not None:
                footer.set_action_enabled(remove, field is not None and field.slot_id is not None)

    def _hovered_row(self) -> FieldRow | None:
        return next(
            (row for row in self.query(FieldRow) if row.has_class("hovered")),
            None,
        )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "stop_generation":
            return self.generation_worker is not None
        if action in {"remove_hovered_slot", "remove_selected_slot", "clear_selected_field"}:
            if self.screen is not self.screen_stack[0] or isinstance(self.focused, (Input, TextArea)):
                return False
        if action == "remove_hovered_slot":
            hovered = self._hovered_row()
            return hovered is not None and hovered.field.slot_id is not None
        if action == "remove_selected_slot":
            return self._hovered_row() is None
        return super().check_action(action, parameters)

    @on(Input.Changed)
    @on(TextArea.Changed)
    def input_changed(self, event: Input.Changed | TextArea.Changed) -> None:
        control = event.control
        value = control.text if isinstance(control, TextArea) else control.value
        row = control.parent
        if isinstance(row, FieldRow):
            if row.field.value == value:
                return
            row.form.set_value(row.field, value)
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
        if row.field.name in {
            "operation",
            "model",
            "sampling",
            "prompt_mode",
            "engine",
        }:
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
        self.query_one(ImageTUIFooter).show_tab(event.pane.id)
        self._refresh_selected_field()

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

        if button.name is not None:
            await self._perform_action(button.name)

    @on(ImageTUIFooter.CommandRequested)
    async def footer_command_requested(self, event: ImageTUIFooter.CommandRequested) -> None:
        await self._perform_action(event.command)

    async def _perform_action(self, action: str) -> None:
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
            if self.generation_worker is None:
                self._start_generation()
            else:
                self._cancel_generation()
        elif action == "postprocess":
            if self.generation_worker is None:
                self._start_postprocess()
            else:
                self._cancel_generation()
        elif action == "video-generate":
            if self.generation_worker is None:
                self._start_video()
            else:
                self._cancel_generation()
        elif action == "sam-segment":
            if self.generation_worker is None:
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
            self._choose_sam_selection_directory()
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
        elif action == "workflow-open":
            self._open_workflow_editor()
        elif action == "workflow-new":
            await self._new_workflow()
        elif action == "workflow-new-video":
            await self._new_workflow(keyframed_video_workflow_template())
        elif action == "workflow-new-character":
            await self._new_workflow(character_workflow_template())
        elif action in {"image-workflow", "video-workflow", "sam-workflow"}:
            try:
                document = (image_form_workflow(self.form) if action == "image-workflow"
                            else video_form_workflow(self.video_form) if action == "video-workflow"
                            else sam_form_workflow(self.sam_form))
            except ValueError as error:
                self._show_error("Cannot open workflow", str(error))
                return
            await self._new_workflow(document)
        elif action == "workflow-load":
            await self._load_workflow()
        elif action == "quit":
            await self.action_quit()

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
        if field.path_kind == "image":
            directories_only = False
            title = field.label
            select_label = "Select image"
            extensions = IMAGE_EXTENSIONS
        elif field.path_kind == "video":
            directories_only = False
            title = field.label
            select_label = "Select video"
            extensions = VIDEO_EXTENSIONS
        elif field.path_kind == "reference_pack":
            directories_only = False
            title = field.label
            select_label = "Select pack"
            extensions = CONFIG_EXTENSIONS
        elif field.path_kind == "config":
            directories_only = False
            title = field.label
            select_label = "Select JSON"
            extensions = CONFIG_EXTENSIONS
        elif field.path_kind == "lora":
            directories_only = False
            title = field.label
            select_label = "Select LoRA"
            extensions = LORA_EXTENSIONS
        elif field.path_kind == "directory":
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
        path = self.configuration_path
        self.push_screen(
            FileBrowser(
                path.parent if path is not None else PROJECT_ROOT,
                title='Save configuration', directories_only=False,
                extensions=CONFIG_EXTENSIONS, select_label="Save",
                save_name=path.name if path is not None else 'image-edit.json',
            ),
            self._save_configuration,
        )

    def _save_configuration(self, output: Path | None) -> None:
        if output is None:
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
        if self.generation_worker is not None:
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

    def _update_workflow_documents(self) -> None:
        self.query_one(WorkflowDocumentList).show_documents(
            self.workflow_document_paths,
            current_path=self.workflow_buffer.document_path,
            current_name=self.workflow_buffer.document.name,
            dirty=self.workflow_buffer.dirty,
        )

    def _remember_workflow_document(self, path: Path) -> None:
        if path not in self.workflow_document_paths:
            self.workflow_document_paths.append(path)

    def _open_workflow_editor(self) -> None:
        if self.workflow_editor is not None:
            return
        editor = WorkflowEditor(self.workflow_buffer, self.workflow_run_state, DEFAULT_WORKFLOW_RUNS_ROOT)
        self.workflow_editor = editor
        self.push_screen(editor, self._close_workflow_editor)
        self.call_after_refresh(
            editor.refresh_runtime_statuses,
        )
        self.call_after_refresh(
            editor.set_running,
            self.active_action_button_id == "workflow-run",
        )

    def _close_workflow_editor(
        self,
        _result: None,
    ) -> None:
        self.workflow_editor = None
        self._update_workflow_documents()

    async def _new_workflow(self, document: WorkflowGraph | None = None) -> None:
        if self.generation_worker is not None:
            self._set_status("Stop the active operation before creating a workflow.")
            return
        if not await self._commit_workflow_draft():
            return
        if document is None:
            document = image_style_workflow_template()
        if self.workflow_buffer.dirty:
            self.push_screen(
                ConfirmationDialog(
                    "Discard workflow changes?",
                    "Create a new workflow and discard the unsaved changes?",
                    confirm_label="Discard and create",
                ),
                lambda discard: self._new_workflow_confirmed(discard, document),
            )
            return
        self._replace_workflow_document(
            document,
            document_path=None,
            status="New workflow",
        )

    def _new_workflow_confirmed(self, discard: bool, document: WorkflowGraph) -> None:
        if not discard:
            return
        self._replace_workflow_document(
            document,
            document_path=None,
            status="New workflow",
        )

    async def _load_workflow(self) -> None:
        if not await self._prepare_workflow_load():
            return
        start = self.workflow_buffer.document_path or PROJECT_ROOT
        self.push_screen(
            FileBrowser(
                self._browser_start(start.as_posix()),
                title="Load workflow",
                directories_only=False,
                extensions=WORKFLOW_EXTENSIONS,
                select_label="Load",
            ),
            self._apply_workflow,
        )

    async def _prepare_workflow_load(self) -> bool:
        if self.generation_worker is not None:
            self._set_status("Stop the active operation before loading a workflow.")
            return False
        if not await self._commit_workflow_draft():
            return False
        return True

    @on(WorkflowDocumentList.DocumentActivated)
    async def workflow_document_activated(
        self,
        event: WorkflowDocumentList.DocumentActivated,
    ) -> None:
        if event.path == self.workflow_buffer.document_path:
            self._open_workflow_editor()
            return
        assert event.path is not None
        if await self._prepare_workflow_load():
            self._apply_workflow(event.path)

    def _apply_workflow(self, path: Path | None) -> None:
        if path is None:
            return
        if self.workflow_buffer.dirty:
            self.push_screen(
                ConfirmationDialog(
                    "Discard workflow changes?",
                    "Load the selected workflow and discard the unsaved changes?",
                    confirm_label="Discard and load",
                ),
                lambda discard: self._workflow_load_confirmed(
                    discard,
                    path,
                ),
            )
            return
        self._load_workflow_document(path)

    def _load_workflow_document(self, path: Path) -> None:
        try:
            document = load_workflow_document(path)
        except (OSError, ValueError) as error:
            self._show_error("Cannot load workflow", str(error))
            return
        self._replace_workflow_document(
            document,
            document_path=path,
            status=f"Loaded {path}",
        )

    def _workflow_load_confirmed(
        self,
        discard: bool,
        path: Path,
    ) -> None:
        if not discard:
            return
        self._load_workflow_document(path)

    def _replace_workflow_document(
        self,
        document: WorkflowGraph,
        *,
        document_path: Path | None,
        status: str,
    ) -> None:
        if document_path is None:
            self.workflow_buffer.replace_document(document)
        else:
            self._remember_workflow_document(document_path)
            self.workflow_buffer.load_document(document, document_path)
        self.workflow_run_state.clear()
        self._update_workflow_documents()
        if self.workflow_editor is None:
            self._open_workflow_editor()
            return
        self.run_worker(
            self.workflow_editor.show_replaced_document(status),
            group="workflow-document",
            exclusive=True,
        )

    async def _commit_workflow_draft(self) -> bool:
        if self.workflow_editor is None:
            return True
        return await self.workflow_editor.commit_pending_property()

    @on(WorkflowEditor.SaveRequested)
    def workflow_save_requested(
        self,
        _event: WorkflowEditor.SaveRequested,
    ) -> None:
        if self.workflow_buffer.document_path is not None:
            self._save_workflow_to(
                self.workflow_buffer.document_path
            )
            return
        self._choose_workflow_directory()

    def _choose_workflow_directory(self) -> None:
        path = self.workflow_buffer.document_path
        self.push_screen(
            FileBrowser(
                path.parent if path is not None else PROJECT_ROOT,
                title='Save workflow', directories_only=False,
                extensions=WORKFLOW_EXTENSIONS, select_label="Save",
                save_name=path.name if path is not None else 'workflow.json',
            ),
            self._save_workflow,
        )

    def _save_workflow(self, output: Path | None) -> None:
        if output is None:
            return
        if (
            output.exists()
            and output != self.workflow_buffer.document_path
        ):
            self._show_error(
                "Cannot save workflow",
                f"Workflow already exists: {output}",
            )
            return
        self._save_workflow_to(output)

    def _save_workflow_to(self, output: Path) -> None:
        try:
            save_workflow_document(
                self.workflow_buffer.document,
                output,
            )
        except OSError as error:
            self._show_error("Cannot save workflow", str(error))
            return
        self.workflow_buffer.mark_saved(output)
        self._remember_workflow_document(output)
        if self.workflow_editor is not None:
            self.workflow_editor.document_saved()
        self._update_workflow_documents()

    @on(WorkflowEditor.LoadRequested)
    async def workflow_load_requested(self) -> None:
        await self._load_workflow()

    @on(WorkflowEditor.BrowseRequested)
    def workflow_browse_requested(
        self,
        event: WorkflowEditor.BrowseRequested,
    ) -> None:
        editor = self.workflow_editor
        if editor is None:
            return
        node = self.workflow_buffer.document.node(event.node_id)
        if isinstance(node, ImageSourceNode):
            title = "Select image"
            extensions = IMAGE_EXTENSIONS
        elif isinstance(node, VideoSourceNode):
            title = "Select video"
            extensions = VIDEO_EXTENSIONS
        elif isinstance(node, AudioSourceNode):
            title = "Select audio or a video containing audio"
            extensions = AUDIO_EXTENSIONS
        elif isinstance(node, ReferencePackNode):
            title = "Select reference pack"
            extensions = CONFIG_EXTENSIONS
        elif isinstance(node, LoraSourceNode):
            title = "Select LoRA"
            extensions = LORA_EXTENSIONS
        else:
            raise RuntimeError(f"node {node.id!r} has no browsable path")
        self.push_screen(
            FileBrowser(
                self._browser_start(event.current_value),
                title=title,
                directories_only=False,
                extensions=extensions,
                select_label="Select",
            ),
            lambda path: self._apply_workflow_browsed_path(
                event.node_id,
                event.field_name,
                path,
            ),
        )

    def _apply_workflow_browsed_path(
        self,
        node_id: str,
        field_name: str,
        path: Path | None,
    ) -> None:
        if path is None or self.workflow_editor is None:
            return
        self.run_worker(
            self.workflow_editor.apply_browsed_path(
                node_id,
                field_name,
                Path(display_project_path(path)),
            ),
            group="workflow-property",
            exclusive=True,
        )

    @on(WorkflowEditor.RunRequested)
    def workflow_run_requested(
        self,
        event: WorkflowEditor.RunRequested,
    ) -> None:
        if self.generation_worker is not None:
            self._set_status("An operation is already running.")
            return
        document = event.document
        run_id = new_workflow_run_id()
        request_path = (
            DEFAULT_WORKFLOW_RUNS_ROOT
            / "requests"
            / f"request-{uuid4().hex}.json"
        )
        try:
            save_workflow_document(document, request_path)
        except OSError as error:
            self._show_error("Cannot start workflow", str(error))
            return
        self.workflow_run_state.start(document, target_node_ids=event.target_node_ids)
        if self.workflow_editor is not None:
            self.workflow_editor.refresh_runtime_statuses()
        self._start_command(
            [
                sys.executable,
                "-m",
                "aigen.cli",
                "workflow",
                "run",
                "--input",
                request_path.as_posix(),
                "--runs-root",
                DEFAULT_WORKFLOW_RUNS_ROOT.as_posix(),
                "--run-id", run_id,
                *(argument for target in event.target_node_ids or () for argument in ("--target", target)),
            ],
            display_project_path(DEFAULT_WORKFLOW_RUNS_ROOT / "runs"),
            action_button_id="workflow-run",
            idle_label="Run",
            error_title="Cannot start workflow",
            running_label=None,
        )
        self.workflow_request_path = request_path
        self.workflow_run_dir = workflow_run_path(DEFAULT_WORKFLOW_RUNS_ROOT, document.workflow_id, run_id)
        if self.workflow_editor is not None:
            self.workflow_editor.set_running(True)

    @on(WorkflowEditor.StopRequested)
    def action_stop_generation(self) -> None:
        self._cancel_generation()

    @on(WorkflowEditor.QuitRequested)
    async def workflow_quit_requested(self) -> None:
        await self.action_quit()

    def _start_generation(self) -> None:
        if self.generation_worker is not None:
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
        if self.generation_worker is not None:
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
        if self.generation_worker is not None:
            self._set_status("Video generation is already running.")
            return
        try:
            command, output_dir, outputs = self.video_form.generation_command()
        except ValueError as error:
            self._show_error("Cannot start video generation", str(error))
            return
        self._start_command(
            command,
            output_dir,
            action_button_id="video-action",
            idle_label="Generate",
            error_title="Cannot start video generation",
            contact_sheet_videos=outputs,
        )

    def _start_sam(self) -> None:
        if self.generation_worker is not None:
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
        contact_sheet_videos: tuple[Path, ...] = (),
        running_label: str | None = "Stop",
    ) -> None:
        self.cancel_requested = False
        self.workflow_run_dir = None
        self.active_action_button_id = action_button_id
        self.active_action_idle_label = idle_label
        self.generation_progress = None
        self._set_status("Starting...")
        for button in self._action_buttons():
            if button.id == action_button_id and running_label is not None:
                button.label = running_label
            button.disabled = button.id != action_button_id
        self.generation_worker = self.run_worker(
            partial(self._watch_generation, command, output_dir, contact_sheet_videos, error_title),
            name="image-generation",
        )

    async def _watch_generation(
        self,
        command: list[str],
        output_dir: str,
        contact_sheet_videos: tuple[Path, ...],
        error_title: str,
    ) -> None:
        environment = {**os.environ, "AIGEN_PROGRESS": "json"}
        try:
            if self.cancel_requested:
                raise asyncio.CancelledError
            async with command_process(command, cwd=PROJECT_ROOT, env=environment) as process:
                self.process = process
                if self.cancel_requested:
                    raise asyncio.CancelledError
                error_title = "Generation failed"
                completed_output_dir = await self._read_generation_output(process, output_dir)
            contact_sheets = []
            for video in contact_sheet_videos:
                error_title = "Contact sheet failed"
                output = contact_sheet_path(video).resolve()
                sheet_command = [sys.executable, "-m", "aigen.cli", "video-postprocess", "contact-sheet",
                                 "--input", str(video), "--output", str(output)]
                async with command_process(sheet_command, cwd=PROJECT_ROOT, env=environment) as process:
                    self.process = process
                    await self._read_generation_output(process, output_dir)
                contact_sheets.append(output)
        except asyncio.CancelledError:
            outcome = GenerationCancelled()
        except Exception as error:
            message = f"Video output: {output_dir}\n{error}" if error_title == "Contact sheet failed" else str(error)
            outcome = GenerationFailed(message, error_title)
        else:
            outcome = GenerationFinished(completed_output_dir, tuple(contact_sheets))
        if self.workflow_run_dir is not None and not isinstance(outcome, GenerationFinished):
            message = outcome.error if isinstance(outcome, GenerationFailed) else "Stopped by the user"
            try:
                published = await asyncio.to_thread(finalize_terminated_workflow, self.workflow_run_dir, message)
                for result in published:
                    self.workflow_run_state.update(result.node_id, result.status)
            except Exception as error:
                outcome = GenerationFailed(f"{message}\nCannot finalize {self.workflow_run_dir}: {error}")
        self.post_message(outcome)

    async def _read_generation_output(self, process: asyncio.subprocess.Process, output_dir: str) -> str:
        assert process.stdout is not None
        output_lines: deque[str] = deque(maxlen=200)
        completed_output_dir = output_dir
        async for raw_line in command_lines(process.stdout):
            line = raw_line.strip()
            if not line:
                continue
            try:
                if line.startswith(WORKFLOW_EVENT_PREFIX):
                    payload = json.loads(line[len(WORKFLOW_EVENT_PREFIX) :])
                    if payload.get("kind") == "workflow-run":
                        completed_output_dir = display_project_path(
                            Path(payload["run_dir"])
                        )
                    else:
                        self.post_message(WorkflowNodeUpdated(payload))
                    continue
                if line.startswith(JSON_PROGRESS_PREFIX):
                    payload = json.loads(line[len(JSON_PROGRESS_PREFIX) :])
                    self.post_message(GenerationUpdated(_generation_progress_from_payload(payload)))
                else:
                    output_lines.append(line)
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                raise ValueError(f"Invalid generation event: {error}\n{line}") from error
        returncode = await process.wait()
        if returncode:
            raise RuntimeError(self._error_message("\n".join(output_lines), returncode))
        return completed_output_dir

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
        status = self._progress_text(event.progress)
        if self.active_action_button_id == "workflow-run":
            # The editor shows node progress; the main screen shows the run.
            self.query_one("#status", Static).update(status)
        else:
            self._set_status(status)

    @on(WorkflowNodeUpdated)
    def workflow_node_updated(self, event: WorkflowNodeUpdated) -> None:
        node_id = event.node_id
        status = event.status
        self.workflow_run_state.update(node_id, status)
        if self.workflow_editor is not None:
            self.workflow_editor.refresh_runtime_status(node_id)
            if event.progress is not None:
                detail = self._progress_text(event.progress)
            else:
                detail = str(event.payload.get("message") or status)
            self.workflow_editor.set_status(f"{node_id}: {detail}")

    @on(GenerationFinished)
    def generation_finished(self, event: GenerationFinished) -> None:
        self._generation_stopped()
        status = f"Output: {event.output_dir}"
        if event.contact_sheets:
            sheets = ", ".join(
                display_project_path(path)
                for path in event.contact_sheets
            )
            status += f" | Contact sheet: {sheets}"
        self._set_status(status)

    @on(GenerationFailed)
    def generation_failed(self, event: GenerationFailed) -> None:
        self.workflow_run_state.finish("failed")
        self._generation_stopped()
        self._show_error(event.title, event.error)

    @on(GenerationCancelled)
    def generation_cancelled(self) -> None:
        self.workflow_run_state.finish("interrupted")
        self._generation_stopped()
        self._set_status("Stopped.")

    def _generation_stopped(self) -> None:
        self._discard_workflow_request()
        self.process = None
        self.generation_worker = None
        self.workflow_run_dir = None
        self.cancel_requested = False
        self.generation_progress = None
        self.query_one("#generation-progress", ProgressBar).display = False
        if self.active_action_button_id is not None:
            for button in self.query(f"#{self.active_action_button_id}"):
                assert isinstance(button, Button)
                button.label = self.active_action_idle_label
        for button in self._action_buttons():
            button.disabled = False
        if self.workflow_editor is not None:
            self.workflow_editor.set_running(False)
            self.workflow_editor.refresh_runtime_statuses()
        self.active_action_button_id = None
        self.active_action_idle_label = ""

    def _action_buttons(self) -> tuple[Button, ...]:
        return tuple(
            button
            for button_id in TAB_ACTION_BUTTON_IDS
            for button in self.query(f"#{button_id}")
            if isinstance(button, Button)
        )

    def _cancel_generation(self) -> None:
        if self.generation_worker is None:
            self._set_status("No generation is running.")
            return
        if self.cancel_requested:
            return
        self.cancel_requested = True
        if self.process is not None:
            self.generation_worker.cancel()
        self._set_status("Stopping generation...")

    async def action_quit(self) -> None:
        if not await self._commit_workflow_draft():
            return
        if self.workflow_buffer.dirty:
            self.push_screen(
                ConfirmationDialog(
                    "Discard workflow changes?",
                    "Quit and discard the unsaved workflow changes?",
                    confirm_label="Discard and quit",
                ),
                self._quit_confirmed,
            )
            return
        await self._quit()

    async def _quit_confirmed(self, discard: bool) -> None:
        if discard:
            await self._quit()

    async def _quit(self) -> None:
        try:
            self.form.save(STATE_PATH)
        except OSError as error:
            self._show_error("Cannot save form", str(error))
            return
        worker = self.generation_worker
        if worker is not None:
            self._cancel_generation()
            try:
                await worker.wait()
            except WorkerCancelled:
                pass
        exports = tuple(worker for worker in self.workers if worker.group == "artifact-export")
        if exports:
            self._set_status("Finishing exports before quitting…")
            await self.workers.wait_for_complete(exports)
        self._discard_workflow_request()
        self.exit()

    def _discard_workflow_request(self) -> None:
        if self.workflow_request_path is None:
            return
        self.workflow_request_path.unlink(missing_ok=True)
        self.workflow_request_path = None

    def _show_error(self, title: str, message: str) -> None:
        self.push_screen(MessageDialog(title, message))

    def _set_status(self, status: str) -> None:
        self.query_one("#status", Static).update(status)
        if self.workflow_editor is not None:
            self.workflow_editor.set_status(status)

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
            message = payload.get("message") if isinstance(payload, dict) else None
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
