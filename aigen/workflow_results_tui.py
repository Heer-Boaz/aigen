from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
from functools import partial

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ItemGrid, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Select, TextArea

from aigen.artifact_actions import export_artifact, open_artifact
from aigen.image_io import load_thumbnail
from aigen.manifest_io import ManifestIOError
from aigen.tui_dialogs import PromptDialog
from aigen.tui_action_button import ActionButton
from aigen.tui_result_records import ResultRecords
from aigen.workflow_result_widgets import ResultComparison, ResultPreview
from aigen.workflow_artifacts import (AudioArtifact, ImageArtifact, ImageCandidate, ImageCollectionArtifact, MaskArtifact,
    ImageSequenceArtifact, KeyframeArtifact, ReferencePackArtifact, VideoArtifact)
from aigen.workflow_media_results import (MediaCandidate, export_media, media_label, media_path,
    media_preview, resolve_media_result)
from aigen.workflow_graph import ImageResultReference, ImageSelectionNode, WorkflowGraph
from aigen.workflow_results import NodeResultManifest, load_node_result, node_result_history, resolve_image_result
from aigen.workflow_cache import WorkflowCacheError


class WorkflowResults(ModalScreen[tuple[str, ImageResultReference] | None]):
    """One saved result history and original-image comparison, without generation."""

    BINDINGS = [
        Binding("escape", "dismiss(None)", show=False),
        Binding("left", "move_candidate(-1)", show=False),
        Binding("right", "move_candidate(1)", show=False),
    ]
    DEFAULT_CSS = """
    WorkflowResults { width: 100%; height: 100%; }
    WorkflowResults > Container { width: 100%; height: 100%; padding: 0 1; background: #100d16; }
    WorkflowResults #result-gallery-scroll { height: 7; }
    WorkflowResults #result-gallery { height: auto; grid-gutter: 0 1; }
    WorkflowResults #result-selected { height: 1; text-overflow: ellipsis; color: #d8c5eb; }
    WorkflowResults #result-details { display: none; height: 1fr; }
    WorkflowResults.details-open ResultComparison { display: none; }
    WorkflowResults.details-open #result-details { display: block; }
    WorkflowResults #result-actions { height: 1; }
    WorkflowResults #result-actions Button { width: auto; min-width: 0; padding: 0 1; margin-right: 1; }
    WorkflowResults #result-target { height: auto; }
    """

    def __init__(self, document: WorkflowGraph, node_id: str, runs_root: Path, *, initial_path: Path | None = None) -> None:
        super().__init__()
        self.document = document
        self.node_id = node_id
        self.runs_root = runs_root
        self._initial_path = initial_path
        self.history_node_id = node_id
        node = document.node(node_id)
        self.saved_selection = node.config.selected if isinstance(node, ImageSelectionNode) else None
        self.targets: tuple[ImageSelectionNode, ...]
        if isinstance(node, ImageSelectionNode):
            self.targets = (node,)
            incoming = document.incoming_connections()[node_id].get("collection", ())
            if incoming:
                self.history_node_id = incoming[0].source.node_id
        else:
            consumers = {wire.target.node_id for wire in document.connections if wire.source.node_id == node_id}
            self.targets = tuple(item for item in document.nodes if isinstance(item, ImageSelectionNode) and item.id in consumers)
        self.candidates: tuple[ImageCandidate | MediaCandidate, ...] = ()
        self.highlighted_index: int | None = None
        self.manifest_path: Path | None = None
        self._shown_inputs = None
        self._view_id = 0
        self._previews = ()
        self._requested_index = 0
        self._inspected_result: NodeResultManifest | None = None

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(f"Results · {self.document.node(self.node_id).title}", markup=False)
            yield Select([], id="result-history", prompt="Run history", compact=True)
            yield Label("Loading saved results…", id="result-status", markup=False)
            yield Label("No candidate selected", id="result-selected", markup=False)
            yield ResultComparison()
            yield TextArea(read_only=True, id="result-details")
            with VerticalScroll(id="result-gallery-scroll"):
                yield ItemGrid(min_column_width=16, max_column_width=24, regular=True,
                               stretch_height=False, id="result-gallery")
            if self.targets:
                yield Select(((node.title, node.id) for node in self.targets), value=self.targets[0].id,
                             id="result-target", allow_blank=False, compact=True)
            with Horizontal(id="result-actions"):
                yield Button("Use output", id="result-select", disabled=True, variant="primary", compact=True)
                yield Button("Open original", id="result-open", disabled=True, compact=True)
                yield Button("Export", id="result-export", disabled=True, compact=True)
                yield ActionButton("Details", id="result-info", disabled=True, compact=True)
                yield Button("Records", id="result-log", disabled=True, compact=True)
                yield Button("Close", id="result-close", compact=True)

    async def on_mount(self) -> None:
        self.query_one("#result-select").display = bool(self.targets)
        if self.targets:
            self.query_one("#result-target").display = len(self.targets) > 1
        try:
            paths = await asyncio.to_thread(node_result_history, self.runs_root, self.document.workflow_id, self.history_node_id)
        except (OSError, ValueError, ManifestIOError) as error:
            self._status(str(error))
            return
        saved_path = Path(self.saved_selection.manifest_path) if self.saved_selection is not None else None
        if saved_path is not None:
            paths = tuple(dict.fromkeys((saved_path, *paths)))
        if not paths:
            self._status("No saved results. Use Run to here on this node or its image collection.")
            return
        history = self.query_one("#result-history", Select)
        history.set_options((("Saved selection · " if path == saved_path else "")
                             + path.parents[2].name.removeprefix("attempt-"), str(path)) for path in paths)
        history.value = str(self._initial_path if self._initial_path in paths else paths[0])

    @on(Select.Changed, "#result-history")
    def history_selected(self, event: Select.Changed) -> None:
        if isinstance(event.value, str):
            self.show_result(Path(event.value))

    @work(exclusive=True, group="result-history-load")
    async def show_result(self, path: Path) -> None:
        self.workers.cancel_group(self, "result-candidate-load")
        self._view_id += 1
        view_id = self._view_id
        self.candidates = ()
        self.manifest_path = None
        self.highlighted_index = None
        self._shown_inputs = None
        self._inspected_result = None
        self._enable_actions(False)
        self.query_one(TextArea).load_text("")
        self.query_one(ResultComparison).clear()
        self.query_one("#result-gallery-scroll").display = False
        self.query_one("#result-selected", Label).update("No candidate selected")
        gallery = self.query_one("#result-gallery", ItemGrid)
        gallery.disabled = True
        await gallery.remove_children()
        try:
            result, candidates, previews = await asyncio.to_thread(self._read_view, path)
        except (OSError, ValueError, ManifestIOError, WorkflowCacheError) as error:
            self._status(str(error))
            return
        await gallery.mount(*(ResultPreview(label, image, index, candidates[index], view_id)
                              for label, image, index in previews))
        self.manifest_path = path
        self.candidates = candidates
        self._previews = tuple(previews)
        self.query_one("#result-gallery-scroll").display = len(candidates) > 1
        gallery.disabled = False
        count = len(candidates)
        self._status(f"{result.status} · {count} output{'s' if count != 1 else ''} · originals retain their full resolution")
        if candidates:
            pinned = self.targets[0].config.selected if len(self.targets) == 1 else None
            index = next((index for index, candidate in enumerate(candidates)
                          if isinstance(candidate, ImageCandidate) and pinned is not None and candidate.reference.producer_signature == pinned.producer_signature
                          and candidate.reference.output_port == pinned.output_port
                          and candidate.image.identity == pinned.artifact_identity), 0)
            self.inspect_candidate(candidates[index], index, view_id)
        else:
            self.query_one(TextArea).load_text(result.model_dump_json(indent=2))

    def _read_view(self, path: Path):
        result = load_node_result(path)
        if self.saved_selection is not None and path == Path(self.saved_selection.manifest_path):
            selected = self.saved_selection
            image = result.outputs.get(selected.output_port)
            if (result.signature != selected.producer_signature or not isinstance(image, ImageArtifact)
                    or image.identity != selected.artifact_identity):
                raise ValueError("Saved selection no longer matches its recorded result")
        candidates = []
        for port, output in result.outputs.items():
            if isinstance(output, ImageCollectionArtifact):
                candidates.extend(output.candidates)
            elif isinstance(output, ImageArtifact):
                candidates.append(result.candidate(path, port))
            elif isinstance(output, (VideoArtifact, ImageSequenceArtifact, AudioArtifact, MaskArtifact)):
                candidates.append(MediaCandidate(result.title or result.node_id,
                    result.details.effective_config.get("seed") if result.details else None,
                    path, result.signature, port, output))
        previews = []
        for index, candidate in enumerate(candidates):
            label = f"{candidate.title} · seed {candidate.seed}" if candidate.seed is not None else candidate.title
            if isinstance(candidate, MediaCandidate):
                artifact = candidate.artifact
                if ((isinstance(artifact, VideoArtifact) and artifact.content_sha256 is None)
                        or (isinstance(artifact, ImageSequenceArtifact) and artifact.frame_sha256s is None)):
                    artifact = resolve_media_result(candidate)
                previews.append((f"{label}\n{media_label(artifact)}", media_preview(artifact, (160, 128)), index))
                continue
            image = candidate.image
            if image.content_sha256 is None:
                image = resolve_image_result(candidate.reference)
            previews.append((label, load_thumbnail(Path(image.path), (160, 128), expected_sha256=image.content_sha256), index))
        return result, tuple(candidates), previews

    @staticmethod
    def _input_previews(inputs):
        previews = []
        for port, artifacts in inputs.items():
            for artifact in artifacts:
                if isinstance(artifact, KeyframeArtifact):
                    artifact = artifact.image
                if isinstance(artifact, (VideoArtifact, ImageSequenceArtifact, AudioArtifact)):
                    try:
                        previews.append((f"{port.replace('_', ' ').capitalize()} · {media_label(artifact)}", media_preview(artifact, (160, 128))))
                    except (OSError, ValueError) as error:
                        previews.append((f"Input unavailable: {error}", None))
                    continue
                if isinstance(artifact, (ImageArtifact, MaskArtifact)):
                    files = ((artifact.path, artifact.content_sha256),)
                elif isinstance(artifact, ReferencePackArtifact):
                    files = zip(artifact.references, artifact.reference_sha256s or (None,) * len(artifact.references), strict=True)
                else:
                    continue
                for file_index, (input_path, expected_sha256) in enumerate(files):
                    label = port.replace("_", " ").capitalize()
                    if isinstance(artifact, ReferencePackArtifact):
                        label += f" · image {file_index + 1}"
                    if expected_sha256 is None:
                        previews.append((f"Original input unavailable: legacy record has no file checksum · {input_path}", None))
                        continue
                    try:
                        preview = load_thumbnail(Path(input_path), (160, 128), expected_sha256=expected_sha256)
                    except (OSError, ValueError) as error:
                        # Missing historical inputs do not make an existing output unreadable.
                        previews.append((f"Input unavailable: {input_path}: {error}", None))
                    else:
                        previews.append((label, preview))
        return previews

    @on(ResultPreview.Highlighted)
    def preview_highlighted(self, event: ResultPreview.Highlighted) -> None:
        if event.view_id == self._view_id:
            self.inspect_candidate(event.candidate, event.index, event.view_id)

    @work(exclusive=True, group="result-candidate-load")
    async def inspect_candidate(self, candidate: ImageCandidate | MediaCandidate, index: int, view_id: int) -> None:
        self.highlighted_index = None
        self._requested_index = index
        self._enable_actions(False)
        try:
            result = await asyncio.to_thread(load_node_result, candidate.manifest_path)
        except (OSError, ValueError, ManifestIOError) as error:
            self._status(str(error))
            return
        inputs = result.details.inputs if result.details else {}
        if inputs != self._shown_inputs:
            previews = await asyncio.to_thread(self._input_previews, inputs)
            if view_id != self._view_id:
                return
            self.query_one(ResultComparison).set_inputs(previews)
            self._shown_inputs = inputs
        if view_id != self._view_id:
            return
        self.highlighted_index = index
        self._inspected_result = result
        self._enable_actions(True)
        label, preview, _ = self._previews[index]
        self.query_one(ResultComparison).set_output(preview, label)
        backend = result.details.effective_config.get("backend", "") if result.details else ""
        target = next((node for node in self.targets if node.config.selected == getattr(candidate, "reference", None)), None)
        summary = f"Output {index + 1}/{len(self.candidates)} · {label.replace(chr(10), ' · ')}"
        if backend:
            summary += f" · {backend}"
        if target is not None:
            summary += f" · Chosen for {target.title}"
        self.query_one("#result-selected", Label).update(summary)
        self.query_one("#result-select", Button).tooltip = f"Use output {index + 1}: {candidate.title}"
        for tile in self.query(ResultPreview):
            tile.set_class(tile.index == index, "-highlighted")
            if tile.index == index:
                tile.scroll_visible(animate=False)
        if self.has_class("details-open"):
            self._render_details()

    def _render_details(self) -> None:
        assert self.highlighted_index is not None and self._inspected_result is not None
        candidate = self.candidates[self.highlighted_index]
        result = self._inspected_result
        self.query_one(TextArea).load_text(json.dumps({
            "output": candidate.image.path if isinstance(candidate, ImageCandidate) else candidate.artifact.model_dump(mode="json"),
            "output_id": candidate.output_id,
            "generation": result.details.model_dump(mode="json") if result.details else "Legacy result without execution details",
            "provenance": result.provenance.model_dump(mode="json") if result.provenance else None,
        }, indent=2, ensure_ascii=False))

    @on(Button.Pressed)
    def action_pressed(self, event: Button.Pressed) -> None:
        action = event.button.id
        if action == "result-close":
            self.dismiss(None)
        elif action == "result-info":
            self.toggle_class("details-open")
            event.button.label = "Preview" if self.has_class("details-open") else "Details"
            if self.has_class("details-open"):
                self._render_details()
        elif self.highlighted_index is not None:
            if action == "result-export":
                candidate = self.candidates[self.highlighted_index]
                filename = (f"image-seed-{candidate.seed}.png" if isinstance(candidate, ImageCandidate)
                            else "frames.zip" if isinstance(candidate.artifact, ImageSequenceArtifact) else media_path(candidate.artifact).name)
                self.app.push_screen(PromptDialog("Export original result", "Destination path", str(Path.cwd() / filename)),
                                     lambda path: self._export_destination(candidate, path))
            elif action in ("result-open", "result-select", "result-log"):
                self.apply_action(action, self.candidates[self.highlighted_index])

    def _export_destination(self, candidate: ImageCandidate | MediaCandidate, path: str | None) -> None:
        if path is not None:
            self._status(f"Exporting original result to {path}…")
            self.app.run_worker(partial(self._export_result, self.app, candidate, Path(path)),
                                name="Export original result", group="artifact-export")

    async def _export_result(self, app: App, candidate: ImageCandidate | MediaCandidate, destination: Path) -> None:
        def export() -> Path:
            if isinstance(candidate, MediaCandidate):
                return export_media(resolve_media_result(candidate, verify_contents=False), destination)
            image = resolve_image_result(candidate.reference, verify_contents=False)
            return export_artifact(Path(image.path), destination, expected_sha256=image.content_sha256)

        operation = asyncio.create_task(asyncio.to_thread(export))
        try:
            try:
                exported = await asyncio.shield(operation)
            except asyncio.CancelledError:
                # Filesystem work cannot be cancelled by cancelling its awaiter.
                exported = await operation
        except (OSError, ValueError, ManifestIOError, WorkflowCacheError) as error:
            if self.is_attached:
                self._status(f"Export failed: {error}")
            app.notify(str(error), title="Export failed", severity="error", timeout=15)
        else:
            if self.is_attached:
                self._status(f"Exported original result: {exported}")
            app.notify(f"Saved original result: {exported}", title="Export completed", timeout=10)

    @work(group="result-action", exclusive=True)
    async def apply_action(self, action: str, candidate: ImageCandidate | MediaCandidate) -> None:
        try:
            if action == "result-log":
                result = await asyncio.to_thread(load_node_result, candidate.manifest_path)
                record_dir = Path(result.details.record_dir) if result.details and result.details.record_dir else None
                self.app.push_screen(ResultRecords(candidate.manifest_path, record_dir))
                return
            if isinstance(candidate, MediaCandidate):
                artifact = await asyncio.to_thread(resolve_media_result, candidate)
                if action == "result-open":
                    await asyncio.to_thread(open_artifact, media_path(artifact))
                return
            image = await asyncio.to_thread(resolve_image_result, candidate.reference)
            if action == "result-open":
                await asyncio.to_thread(open_artifact, Path(image.path))
            elif action == "result-select":
                target = self.query_one("#result-target", Select).value
                if isinstance(target, str):
                    self.dismiss((target, candidate.reference))
        except (OSError, ValueError, ManifestIOError, WorkflowCacheError, subprocess.SubprocessError) as error:
            self._status(str(error))

    def _enable_actions(self, enabled: bool) -> None:
        for name in ("open", "export", "log", "info"):
            self.query_one(f"#result-{name}", Button).disabled = not enabled
        image_selected = self.highlighted_index is not None and isinstance(self.candidates[self.highlighted_index], ImageCandidate)
        self.query_one("#result-select", Button).disabled = not (enabled and self.targets and image_selected)

    def action_move_candidate(self, direction: int) -> None:
        if self.candidates:
            index = (self._requested_index + direction) % len(self.candidates)
            self.inspect_candidate(self.candidates[index], index, self._view_id)

    def _status(self, text: str) -> None:
        self.query_one("#result-status", Label).update(text)
