from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
from functools import partial

from PIL import Image
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, ItemGrid, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Select, TextArea
from textual_image.widget import HalfcellImage

from aigen.artifact_actions import export_artifact, open_artifact
from aigen.image_io import load_thumbnail
from aigen.manifest_io import ManifestIOError
from aigen.tui_dialogs import PromptDialog
from aigen.tui_result_records import ResultRecords
from aigen.workflow_artifacts import (AudioArtifact, ImageArtifact, ImageCandidate, ImageCollectionArtifact, MaskArtifact,
    ImageSequenceArtifact, KeyframeArtifact, ReferencePackArtifact, VideoArtifact)
from aigen.workflow_media_results import (MediaCandidate, export_media, media_label, media_path,
    media_preview, resolve_media_result)
from aigen.workflow_graph import ImageResultReference, ImageSelectionNode, WorkflowGraph
from aigen.workflow_results import load_node_result, node_result_history, resolve_image_result
from aigen.workflow_cache import WorkflowCacheError


class ResultPreview(Container):
    class Highlighted(Message):
        def __init__(self, candidate: ImageCandidate | MediaCandidate, index: int, view_id: int) -> None:
            super().__init__()
            self.candidate = candidate
            self.index = index
            self.view_id = view_id

    def __init__(self, label: str, preview: Image.Image | None, index: int | None,
                 candidate: ImageCandidate | MediaCandidate | None = None, view_id: int = 0) -> None:
        super().__init__(classes="result-preview")
        self.label = label
        self.preview = preview
        self.index = index
        self.candidate = candidate
        self.view_id = view_id

    def compose(self) -> ComposeResult:
        with Container(classes="result-preview-image"):
            if self.preview is not None:
                yield HalfcellImage(self.preview)
        yield Label(self.label, markup=False)
        if self.index is not None:
            yield Button("Inspect", id=f"candidate-{self.index}", compact=True)

    @on(Button.Pressed)
    def inspect(self, event: Button.Pressed) -> None:
        event.stop()
        if self.candidate is not None:
            assert self.index is not None
            self.post_message(self.Highlighted(self.candidate, self.index, self.view_id))


class WorkflowResults(ModalScreen[tuple[str, ImageResultReference] | None]):
    """One saved result history and original-image comparison, without generation."""

    DEFAULT_CSS = """
    WorkflowResults { width: 100%; height: 100%; }
    WorkflowResults > Container { width: 100%; height: 100%; padding: 0 1; background: #100d16; }
    WorkflowResults #result-gallery-scroll { height: 2fr; }
    WorkflowResults #result-gallery, WorkflowResults #result-inputs { height: auto; grid-gutter: 1; }
    WorkflowResults .result-preview { height: 24; padding: 0 1; border: solid #352944; }
    WorkflowResults .result-preview.-highlighted { border: solid #b791dd; }
    WorkflowResults .result-preview-image { height: 17; width: 1fr; align: center middle; }
    WorkflowResults HalfcellImage { width: auto; height: auto; }
    WorkflowResults .result-preview Label { height: 2; text-overflow: ellipsis; }
    WorkflowResults #result-details { height: 1fr; min-height: 5; }
    WorkflowResults #result-actions { height: auto; }
    WorkflowResults #result-target { height: auto; }
    """

    def __init__(self, document: WorkflowGraph, node_id: str, runs_root: Path) -> None:
        super().__init__()
        self.document = document
        self.node_id = node_id
        self.runs_root = runs_root
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

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(f"Results · {self.document.node(self.node_id).title}", markup=False)
            yield Select([], id="result-history", prompt="Run history", compact=True)
            yield Label("Loading saved results…", id="result-status", markup=False)
            with VerticalScroll(id="result-gallery-scroll"):
                yield ItemGrid(min_column_width=28, max_column_width=48, regular=True, stretch_height=False, id="result-inputs")
                yield ItemGrid(min_column_width=28, max_column_width=48, regular=True, stretch_height=False, id="result-gallery")
            yield TextArea(read_only=True, id="result-details")
            if self.targets:
                yield Select(((node.title, node.id) for node in self.targets), value=self.targets[0].id, id="result-target", allow_blank=False, compact=True)
            yield ItemGrid(
                Button("Select image", id="result-select", disabled=True, compact=True),
                Button("Open original", id="result-open", disabled=True, compact=True),
                Button("Export", id="result-export", disabled=True, compact=True),
                Button("Records / log", id="result-log", disabled=True, compact=True),
                Button("Close", id="result-close", compact=True),
                min_column_width=14, stretch_height=False, regular=False, id="result-actions",
            )

    async def on_mount(self) -> None:
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
        history.value = str(paths[0])

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
        self._enable_actions(False)
        self.query_one(TextArea).load_text("")
        gallery = self.query_one("#result-gallery", ItemGrid)
        gallery.disabled = True
        await gallery.remove_children()
        await self.query_one("#result-inputs", ItemGrid).remove_children()
        try:
            result, candidates, previews = await asyncio.to_thread(self._read_view, path)
        except (OSError, ValueError, ManifestIOError, WorkflowCacheError) as error:
            self._status(str(error))
            return
        await gallery.mount(*(ResultPreview(label, image, index, candidates[index], view_id)
                              for label, image, index in previews))
        self.manifest_path = path
        self.candidates = candidates
        gallery.disabled = False
        self._status(f"{result.status} · {len(candidates)} outputs · originals retain their full resolution")
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
                previews.append((f"{label}\n{media_label(artifact)}", media_preview(artifact, (80, 64)), index))
                continue
            image = candidate.image
            if image.content_sha256 is None:
                image = resolve_image_result(candidate.reference)
            previews.append((label, load_thumbnail(Path(image.path), (80, 64), expected_sha256=image.content_sha256), index))
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
                        previews.append((f"Original input · {port} · {media_label(artifact)}", media_preview(artifact, (80, 64))))
                    except (OSError, ValueError) as error:
                        previews.append((f"Input unavailable: {error}", None))
                    continue
                if isinstance(artifact, (ImageArtifact, MaskArtifact)):
                    files = ((artifact.path, artifact.content_sha256),)
                elif isinstance(artifact, ReferencePackArtifact):
                    files = zip(artifact.references, artifact.reference_sha256s or (None,) * len(artifact.references), strict=True)
                else:
                    continue
                for input_path, expected_sha256 in files:
                    label = f"Original input · {port} · {Path(input_path).name}"
                    if expected_sha256 is None:
                        previews.append((f"Original input unavailable: legacy record has no file checksum · {input_path}", None))
                        continue
                    try:
                        preview = load_thumbnail(Path(input_path), (80, 64), expected_sha256=expected_sha256)
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
        self._enable_actions(False)
        try:
            result = await asyncio.to_thread(load_node_result, candidate.manifest_path)
        except (OSError, ValueError, ManifestIOError) as error:
            self._status(str(error))
            return
        inputs = result.details.inputs if result.details else {}
        if inputs != self._shown_inputs:
            previews = await asyncio.to_thread(self._input_previews, inputs)
            input_gallery = self.query_one("#result-inputs", ItemGrid)
            await input_gallery.remove_children()
            await input_gallery.mount(*(ResultPreview(label, image, None) for label, image in previews))
            self._shown_inputs = inputs
        if view_id != self._view_id:
            return
        self.highlighted_index = index
        self._enable_actions(True)
        for tile in self.query(ResultPreview):
            tile.set_class(tile.index == index, "-highlighted")
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
        for name in ("open", "export", "log"):
            self.query_one(f"#result-{name}", Button).disabled = not enabled
        image_selected = self.highlighted_index is not None and isinstance(self.candidates[self.highlighted_index], ImageCandidate)
        self.query_one("#result-select", Button).disabled = not (enabled and self.targets and image_selected)

    def _status(self, text: str) -> None:
        self.query_one("#result-status", Label).update(text)
