from __future__ import annotations

import asyncio
from pathlib import Path

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Select, TextArea


class ResultRecords(ModalScreen[None]):
    """Inspect original manifests and logs, including unsuccessful batch records."""

    DEFAULT_CSS = """
    ResultRecords { width: 100%; height: 100%; }
    ResultRecords > Container { width: 100%; height: 100%; padding: 1; background: #100d16; }
    ResultRecords TextArea { height: 1fr; }
    ResultRecords Horizontal { height: auto; }
    """

    def __init__(self, manifest_path: Path, record_dir: Path | None) -> None:
        super().__init__()
        self.manifest_path = manifest_path
        self.record_dir = record_dir

    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Original result and execution records")
            yield Select([], id="result-record-file", prompt="Record file")
            yield TextArea(read_only=True, id="result-record-text")
            with Horizontal():
                yield Button("Close", id="result-record-close", compact=True)

    async def on_mount(self) -> None:
        paths = await asyncio.to_thread(self._record_paths)
        self.query_one(Select).set_options((str(path), str(path)) for path in paths)
        self.query_one(Select).value = str(self.manifest_path)

    def _record_paths(self) -> tuple[Path, ...]:
        files = [self.manifest_path]
        if self.record_dir is not None:
            files.extend(sorted(path for path in self.record_dir.rglob("*") if path.suffix in (".json", ".jsonl", ".log")))
        return tuple(files)

    @on(Select.Changed, "#result-record-file")
    def file_selected(self, event: Select.Changed) -> None:
        if isinstance(event.value, str):
            self.load_record(Path(event.value))

    @work(exclusive=True)
    async def load_record(self, path: Path) -> None:
        try:
            content = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        except OSError as error:
            self.notify(str(error), severity="error")
            return
        self.query_one(TextArea).load_text(content)

    @on(Button.Pressed, "#result-record-close")
    def close_records(self) -> None:
        self.dismiss(None)
