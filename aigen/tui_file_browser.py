from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ItemGrid, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Label, Select
from textual.worker import get_current_worker
from textual_image.widget import HalfcellImage


@dataclass(frozen=True, slots=True)
class BrowserEntry:
    path: Path
    name_key: str
    is_directory: bool
    modified_ns: int
    size: int
    modified_text: str
    size_text: str


def _load_scaled_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        source.draft("RGB", size)
        ImageOps.exif_transpose(source, in_place=True)
        source.thumbnail(size, Image.Resampling.LANCZOS)
        return source.convert("RGB")


class BrowserThumbnail(Widget):
    class Clicked(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def __init__(self, entry: BrowserEntry, index: int) -> None:
        super().__init__(classes="browser-thumbnail")
        self.entry = entry
        self.index = index

    def compose(self) -> ComposeResult:
        with Container(classes="browser-thumbnail-media"):
            yield HalfcellImage(
                classes="browser-thumbnail-image browser-view-hidden",
            )
            yield Label(
                "📁" if self.entry.is_directory else "…",
                classes="browser-thumbnail-placeholder",
            )
        yield Label(
            f"{self.entry.path.name}/"
            if self.entry.is_directory
            else self.entry.path.name,
            classes="browser-thumbnail-label",
            markup=False,
        )

    def set_thumbnail(self, image: Image.Image | None) -> None:
        image_widget = self.query_one(
            ".browser-thumbnail-image",
            HalfcellImage,
        )
        placeholder = self.query_one(
            ".browser-thumbnail-placeholder",
            Label,
        )
        if image is None:
            placeholder.update("Unreadable")
            return
        image_widget.image = image
        image_widget.remove_class("browser-view-hidden")
        placeholder.add_class("browser-view-hidden")

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.post_message(self.Clicked(self.index))


class BrowserThumbnailGrid(VerticalScroll, can_focus=True):
    THUMBNAIL_PIXEL_SIZE = (32, 20)

    BINDINGS = [
        Binding("left", "cursor_left", show=False),
        Binding("right", "cursor_right", show=False),
        Binding("up", "cursor_up", show=False),
        Binding("down", "cursor_down", show=False),
        Binding("home", "first", show=False),
        Binding("end", "last", show=False),
        Binding("enter", "select", show=False),
    ]

    class Highlighted(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    class Selected(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._tiles: tuple[BrowserThumbnail, ...] = ()
        self._thumbnail_cache: dict[
            tuple[Path, int],
            Image.Image | None,
        ] = {}
        self._load_generation = 0
        self.selected_index: int | None = None

    def compose(self) -> ComposeResult:
        yield ItemGrid(
            id="browser-thumbnail-items",
            min_column_width=20,
            max_column_width=32,
            stretch_height=False,
            regular=False,
        )

    async def set_entries(
        self,
        entries: tuple[BrowserEntry, ...],
        selected_index: int | None,
    ) -> None:
        self.stop_loading()
        item_grid = self.query_one("#browser-thumbnail-items", ItemGrid)
        await item_grid.remove_children()
        self.selected_index = None
        self._tiles = tuple(
            BrowserThumbnail(entry, index)
            for index, entry in enumerate(entries)
        )
        if self._tiles:
            await item_grid.mount(*self._tiles)
        self.highlight(selected_index, emit=False, scroll=False)

    def clear_cache(self) -> None:
        self.stop_loading()
        self._thumbnail_cache.clear()

    def stop_loading(self) -> None:
        self._load_generation += 1
        self.workers.cancel_group(self, "browser-thumbnails")

    def load_thumbnails(self) -> None:
        self._load_generation += 1
        generation = self._load_generation
        requests: list[tuple[int, BrowserEntry]] = []
        for index, tile in enumerate(self._tiles):
            entry = tile.entry
            if entry.is_directory:
                continue
            key = (entry.path, entry.modified_ns)
            if key in self._thumbnail_cache:
                tile.set_thumbnail(self._thumbnail_cache[key])
            else:
                requests.append((index, entry))
        if requests:
            self.run_worker(
                lambda: self._load_thumbnail_batch(generation, requests),
                group="browser-thumbnails",
                exclusive=True,
                thread=True,
                exit_on_error=False,
            )

    def _load_thumbnail_batch(
        self,
        generation: int,
        requests: list[tuple[int, BrowserEntry]],
    ) -> None:
        worker = get_current_worker()
        for index, entry in requests:
            if worker.is_cancelled:
                return
            try:
                thumbnail = _load_scaled_image(
                    entry.path,
                    self.THUMBNAIL_PIXEL_SIZE,
                )
            except OSError:
                thumbnail = None
            self.app.call_from_thread(
                self._thumbnail_loaded,
                generation,
                index,
                entry,
                thumbnail,
            )

    def _thumbnail_loaded(
        self,
        generation: int,
        index: int,
        entry: BrowserEntry,
        thumbnail: Image.Image | None,
    ) -> None:
        if generation != self._load_generation:
            return
        self._thumbnail_cache[(entry.path, entry.modified_ns)] = thumbnail
        if index >= len(self._tiles) or self._tiles[index].entry != entry:
            return
        self._tiles[index].set_thumbnail(thumbnail)

    def highlight(
        self,
        index: int | None,
        *,
        emit: bool = True,
        scroll: bool = True,
    ) -> None:
        if self.selected_index is not None:
            self._tiles[self.selected_index].remove_class("-selected")
        self.selected_index = index
        if index is None:
            return
        tile = self._tiles[index]
        tile.add_class("-selected")
        if scroll:
            self.scroll_to_widget(tile, animate=False)
        if emit:
            self.post_message(self.Highlighted(index))

    def action_cursor_left(self) -> None:
        self._move_linear(-1)

    def action_cursor_right(self) -> None:
        self._move_linear(1)

    def action_cursor_up(self) -> None:
        self._move_vertical(-1)

    def action_cursor_down(self) -> None:
        self._move_vertical(1)

    def action_first(self) -> None:
        if self._tiles:
            self.highlight(0)

    def action_last(self) -> None:
        if self._tiles:
            self.highlight(len(self._tiles) - 1)

    def action_select(self) -> None:
        if self.selected_index is not None:
            self.post_message(self.Selected(self.selected_index))

    def _move_linear(self, delta: int) -> None:
        if not self._tiles:
            return
        current = self.selected_index if self.selected_index is not None else 0
        self.highlight(max(0, min(len(self._tiles) - 1, current + delta)))

    def _move_vertical(self, direction: int) -> None:
        if not self._tiles:
            return
        if self.selected_index is None:
            self.highlight(0)
            return
        current = self._tiles[self.selected_index]
        current_x = current.region.center[0]
        current_y = current.region.y
        row_positions = sorted(
            {
                tile.region.y
                for tile in self._tiles
                if (tile.region.y - current_y) * direction > 0
            }
        )
        if not row_positions:
            return
        target_y = row_positions[0] if direction > 0 else row_positions[-1]
        target = min(
            (tile for tile in self._tiles if tile.region.y == target_y),
            key=lambda tile: abs(tile.region.center[0] - current_x),
        )
        self.highlight(target.index)

    @on(BrowserThumbnail.Clicked)
    def thumbnail_clicked(self, event: BrowserThumbnail.Clicked) -> None:
        self.focus()
        if event.index == self.selected_index:
            self.post_message(self.Selected(event.index))
        else:
            self.highlight(event.index)


class FileBrowser(ModalScreen[Path | None]):
    DEFAULT_CSS = """
    FileBrowser {
        align: center middle;
        background: #000000 55%;
    }

    FileBrowser .browser-dialog {
        width: 90%;
        height: 85%;
        padding: 1 2;
        background: #211a2d;
        border: solid #8c72aa;
    }

    FileBrowser .browser-title {
        height: 1;
        text-style: bold;
    }

    FileBrowser #browser-path {
        height: 1;
    }

    FileBrowser .browser-controls {
        height: 2;
    }

    FileBrowser .browser-control-row {
        height: 1;
    }

    FileBrowser .browser-filter-label {
        width: 7;
    }

    FileBrowser .browser-option-label {
        width: 5;
    }

    FileBrowser .browser-option-select {
        width: 1fr;
    }

    FileBrowser .browser-options-row {
        layout: grid;
        grid-size: 5 1;
        grid-columns: 5 1fr 3 5 1fr;
        grid-rows: 1;
    }

    FileBrowser .browser-controls Label {
        height: 1;
        content-align-vertical: middle;
    }

    FileBrowser #browser-filter,
    FileBrowser #browser-sort,
    FileBrowser #browser-view {
        width: 100%;
        height: 1;
        border: none;
        padding: 0;
    }

    FileBrowser #browser-sort-direction {
        width: 3;
        min-width: 3;
        height: 1;
        min-height: 1;
        border: none;
        padding: 0;
    }

    FileBrowser #browser-preview-layout,
    FileBrowser #browser-thumbnails {
        height: 1fr;
    }

    FileBrowser #browser-list {
        width: 2fr;
        height: 1fr;
        scrollbar-size-vertical: 1;
        overflow-x: hidden;
    }

    FileBrowser #browser-list-pane {
        width: 2fr;
        height: 1fr;
    }

    FileBrowser #browser-preview-pane {
        width: 3fr;
        height: 1fr;
        margin-left: 1;
        align: center middle;
    }

    FileBrowser #browser-preview-placeholder {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        color: #b9adc8;
    }

    FileBrowser #browser-preview-image {
        width: auto;
        height: auto;
    }

    FileBrowser #browser-thumbnails {
        scrollbar-size-vertical: 1;
    }

    FileBrowser #browser-thumbnail-items {
        width: 1fr;
        height: auto;
        grid-gutter: 0 1;
    }

    FileBrowser .browser-thumbnail {
        height: 11;
        padding: 0 1;
    }

    FileBrowser .browser-thumbnail:hover {
        background: #352944;
    }

    FileBrowser .browser-thumbnail.-selected {
        background: #725a91;
    }

    FileBrowser .browser-thumbnail-media {
        width: 1fr;
        height: 10;
        align: center middle;
    }

    FileBrowser .browser-thumbnail-image {
        width: auto;
        height: auto;
    }

    FileBrowser .browser-thumbnail-placeholder {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        color: #b9adc8;
    }

    FileBrowser .browser-thumbnail-label {
        width: 1fr;
        height: 1;
        text-align: center;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    FileBrowser .browser-view-hidden {
        display: none;
    }

    FileBrowser #browser-meta {
        height: 1;
        color: #b9adc8;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    FileBrowser .browser-actions {
        height: 1;
        align-horizontal: center;
    }

    FileBrowser .browser-actions Button {
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 1;
    }
    """

    SORT_OPTIONS = (
        ("Name", "name"),
        ("Modified", "modified"),
        ("Size", "size"),
    )
    VIEW_OPTIONS = (
        ("Preview", "preview"),
        ("Thumbnails", "thumbnails"),
    )

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
        self.directory = start
        self.title = title
        self.directories_only = directories_only
        self.extensions = extensions
        self.select_label = select_label
        self.sort_key = "name"
        self.sort_descending = False
        self.view_mode = "preview"
        self.entries: tuple[BrowserEntry, ...] = ()
        self.visible_entries: tuple[BrowserEntry, ...] = ()
        self.selected_index: int | None = None
        self._preview_cache: dict[
            tuple[Path, int, tuple[int, int]],
            Image.Image | None,
        ] = {}
        self._preview_generation = 0

    def compose(self) -> ComposeResult:
        with Container(classes="browser-dialog"):
            yield Label(self.title, classes="browser-title")
            yield Label(self.directory.as_posix(), id="browser-path")
            with Container(classes="browser-controls"):
                with Horizontal(classes="browser-control-row"):
                    yield Label("Filter", classes="browser-filter-label")
                    yield Input(
                        placeholder="Filename",
                        id="browser-filter",
                        compact=True,
                    )
                with Horizontal(classes="browser-control-row browser-options-row"):
                    yield Label("Sort", classes="browser-option-label")
                    yield Select(
                        self.SORT_OPTIONS,
                        allow_blank=False,
                        value=self.sort_key,
                        id="browser-sort",
                        classes="browser-option-select",
                        compact=True,
                    )
                    yield Button(
                        "↑",
                        id="browser-sort-direction",
                        compact=True,
                        tooltip="Ascending",
                    )
                    yield Label("View", classes="browser-option-label")
                    yield Select(
                        self.VIEW_OPTIONS,
                        allow_blank=False,
                        value=self.view_mode,
                        id="browser-view",
                        classes="browser-option-select",
                        compact=True,
                    )
            with Horizontal(id="browser-preview-layout"):
                with Container(id="browser-list-pane"):
                    yield DataTable(
                        cursor_type="row",
                        show_header=False,
                        zebra_stripes=True,
                        id="browser-list",
                    )
                with Container(id="browser-preview-pane"):
                    yield HalfcellImage(id="browser-preview-image")
                    yield Label(
                        "No image preview",
                        id="browser-preview-placeholder",
                    )
            yield BrowserThumbnailGrid(
                id="browser-thumbnails",
                classes="browser-view-hidden",
            )
            yield Label("", id="browser-meta")
            with Horizontal(classes="browser-actions"):
                yield Button("Up", id="browser-up", compact=True)
                yield Button(
                    self.select_label,
                    variant="primary",
                    id="browser-select",
                    compact=True,
                )
                yield Button("Cancel", id="browser-cancel", compact=True)

    async def on_mount(self) -> None:
        self.query_one("#browser-list", DataTable).add_column("Name")
        await self._load_directory(self.directory)
        self.query_one("#browser-list", DataTable).focus()

    async def _load_directory(self, directory: Path) -> None:
        try:
            with os.scandir(directory) as items:
                entries = []
                for item in items:
                    if item.name.startswith("."):
                        continue
                    is_directory = item.is_dir()
                    if (
                        not is_directory
                        and (
                            self.directories_only
                            or Path(item.name).suffix.casefold() not in self.extensions
                        )
                    ):
                        continue
                    stat = item.stat()
                    modified_text = datetime.fromtimestamp(
                        stat.st_mtime_ns / 1_000_000_000
                    ).strftime("%Y-%m-%d %H:%M")
                    size_text = (
                        ""
                        if is_directory
                        else self._format_size(stat.st_size)
                    )
                    entries.append(
                        BrowserEntry(
                            path=Path(item.path),
                            name_key=item.name.casefold(),
                            is_directory=is_directory,
                            modified_ns=stat.st_mtime_ns,
                            size=0 if is_directory else stat.st_size,
                            modified_text=modified_text,
                            size_text=size_text,
                        )
                    )
        except OSError as error:
            self.notify(str(error), severity="error")
            return
        if directory != self.directory:
            self.query_one(
                "#browser-thumbnails",
                BrowserThumbnailGrid,
            ).clear_cache()
            self._preview_cache.clear()
        self.directory = directory
        self.entries = tuple(entries)
        self.query_one("#browser-path", Label).update(directory.as_posix())
        await self._render_entries()

    async def _render_entries(self) -> None:
        selected_path = (
            self.visible_entries[self.selected_index].path
            if self.selected_index is not None
            else None
        )
        filter_text = self.query_one("#browser-filter", Input).value.casefold()
        matching = [
            entry
            for entry in self.entries
            if filter_text in entry.name_key
        ]
        directories = [entry for entry in matching if entry.is_directory]
        files = [entry for entry in matching if not entry.is_directory]
        directories.sort(
            key=self._directory_sort_value,
            reverse=self.sort_descending,
        )
        files.sort(
            key=self._sort_value,
            reverse=self.sort_descending,
        )
        self.visible_entries = tuple(directories + files)
        self.selected_index = next(
            (
                index
                for index, entry in enumerate(self.visible_entries)
                if entry.path == selected_path
            ),
            0 if self.visible_entries else None,
        )

        table = self.query_one("#browser-list", DataTable)
        table.clear()
        table.add_rows(
            (
                (
                    f"{entry.path.name}/"
                    if entry.is_directory
                    else entry.path.name
                ),
            )
            for entry in self.visible_entries
        )
        if self.selected_index is not None:
            table.move_cursor(row=self.selected_index)

        thumbnails = self.query_one(
            "#browser-thumbnails",
            BrowserThumbnailGrid,
        )
        await thumbnails.set_entries(
            self.visible_entries,
            self.selected_index,
        )
        if self.view_mode == "thumbnails":
            thumbnails.load_thumbnails()
        self._update_selection()

    def _sort_value(self, entry: BrowserEntry) -> str | int:
        if self.sort_key == "modified":
            return entry.modified_ns
        if self.sort_key == "size":
            return entry.size
        return entry.name_key

    def _directory_sort_value(self, entry: BrowserEntry) -> str | int:
        return (
            entry.modified_ns
            if self.sort_key == "modified"
            else entry.name_key
        )

    @staticmethod
    def _format_size(size: int) -> str:
        value = float(size)
        units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")
        unit_index = 0
        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024
            unit_index += 1
        unit = units[unit_index]
        return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"

    def _current_entry(self) -> BrowserEntry | None:
        return (
            self.visible_entries[self.selected_index]
            if self.selected_index is not None
            else None
        )

    def _update_selection(self) -> None:
        entry = self._current_entry()
        if entry is None:
            text = ""
        elif entry.is_directory:
            text = f"Selected: {entry.path.name}/  <DIR>  {entry.modified_text}"
        else:
            text = (
                f"Selected: {entry.path.name}  "
                f"{entry.size_text}  {entry.modified_text}"
            )
        self.query_one("#browser-meta", Label).update(text)
        preview = self.query_one("#browser-preview-image", HalfcellImage)
        placeholder = self.query_one("#browser-preview-placeholder", Label)
        self._preview_generation += 1
        self.workers.cancel_group(self, "browser-preview")
        preview.image = None
        if self.view_mode != "preview":
            return
        if entry is None or entry.is_directory:
            preview.add_class("browser-view-hidden")
            placeholder.update(
                "No image selected"
                if entry is None
                else f"Folder: {entry.path.name}"
            )
            placeholder.remove_class("browser-view-hidden")
            return
        preview.add_class("browser-view-hidden")
        placeholder.update("Loading preview…")
        placeholder.remove_class("browser-view-hidden")
        self._start_preview_load(entry)

    def _start_preview_load(self, entry: BrowserEntry) -> None:
        if self.view_mode != "preview" or entry != self._current_entry():
            return
        preview_pane = self.query_one("#browser-preview-pane", Container)
        size = (
            preview_pane.content_size.width,
            preview_pane.content_size.height * 2,
        )
        if not size[0] or not size[1]:
            return
        key = (entry.path, entry.modified_ns, size)
        self._preview_generation += 1
        generation = self._preview_generation
        if key in self._preview_cache:
            self._preview_loaded(
                generation,
                entry,
                size,
                self._preview_cache[key],
            )
            return
        self.run_worker(
            lambda: self._load_preview(generation, entry, size),
            group="browser-preview",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _load_preview(
        self,
        generation: int,
        entry: BrowserEntry,
        size: tuple[int, int],
    ) -> None:
        try:
            image = _load_scaled_image(entry.path, size)
        except OSError:
            image = None
        worker = get_current_worker()
        if not worker.is_cancelled:
            self.app.call_from_thread(
                self._preview_loaded,
                generation,
                entry,
                size,
                image,
            )

    def _preview_loaded(
        self,
        generation: int,
        entry: BrowserEntry,
        size: tuple[int, int],
        image: Image.Image | None,
    ) -> None:
        if (
            generation != self._preview_generation
            or self.view_mode != "preview"
            or entry != self._current_entry()
        ):
            return
        self._preview_cache[(entry.path, entry.modified_ns, size)] = image
        preview = self.query_one("#browser-preview-image", HalfcellImage)
        placeholder = self.query_one("#browser-preview-placeholder", Label)
        if image is None:
            placeholder.update("Unreadable image")
            return
        preview.image = image
        placeholder.add_class("browser-view-hidden")
        preview.remove_class("browser-view-hidden")

    @on(Input.Changed, "#browser-filter")
    async def filter_changed(self) -> None:
        await self._render_entries()

    @on(Select.Changed, "#browser-sort")
    async def sort_changed(self, event: Select.Changed) -> None:
        if event.value is Select.NULL:
            return
        self.sort_key = str(event.value)
        await self._render_entries()

    @on(Button.Pressed, "#browser-sort-direction")
    async def toggle_sort_direction(self) -> None:
        self.sort_descending = not self.sort_descending
        button = self.query_one("#browser-sort-direction", Button)
        button.label = "↓" if self.sort_descending else "↑"
        button.tooltip = "Descending" if self.sort_descending else "Ascending"
        await self._render_entries()

    @on(Select.Changed, "#browser-view")
    def view_changed(self, event: Select.Changed) -> None:
        if event.value is Select.NULL:
            return
        self.view_mode = str(event.value)
        preview_layout = self.query_one("#browser-preview-layout", Horizontal)
        thumbnails = self.query_one(
            "#browser-thumbnails",
            BrowserThumbnailGrid,
        )
        show_thumbnails = self.view_mode == "thumbnails"
        preview_layout.set_class(show_thumbnails, "browser-view-hidden")
        thumbnails.set_class(not show_thumbnails, "browser-view-hidden")
        if show_thumbnails:
            self._update_selection()
            thumbnails.load_thumbnails()
            thumbnails.focus()
        else:
            thumbnails.stop_loading()
            self.app.call_after_refresh(self._update_selection)
            self.query_one("#browser-list", DataTable).focus()

    @on(DataTable.RowHighlighted, "#browser-list")
    def row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self.selected_index = event.cursor_row
        self.query_one(
            "#browser-thumbnails",
            BrowserThumbnailGrid,
        ).highlight(event.cursor_row, emit=False, scroll=False)
        self._update_selection()

    @on(DataTable.RowSelected, "#browser-list")
    async def row_selected(self, event: DataTable.RowSelected) -> None:
        await self._open_entry(event.cursor_row)

    @on(BrowserThumbnailGrid.Highlighted)
    def thumbnail_highlighted(
        self,
        event: BrowserThumbnailGrid.Highlighted,
    ) -> None:
        self.selected_index = event.index
        self.query_one("#browser-list", DataTable).move_cursor(row=event.index)
        self._update_selection()

    @on(BrowserThumbnailGrid.Selected)
    async def thumbnail_selected(
        self,
        event: BrowserThumbnailGrid.Selected,
    ) -> None:
        await self._open_entry(event.index)

    async def _open_entry(self, index: int) -> None:
        entry = self.visible_entries[index]
        if entry.is_directory:
            await self._load_directory(entry.path)
        else:
            self.dismiss(entry.path)

    @on(Button.Pressed, "#browser-up")
    async def go_up(self) -> None:
        parent = self.directory.parent
        if parent != self.directory:
            await self._load_directory(parent)

    @on(Button.Pressed, "#browser-select")
    async def select_current(self) -> None:
        entry = self._current_entry()
        if self.directories_only:
            self.dismiss(
                entry.path
                if entry is not None and entry.is_directory
                else self.directory
            )
        elif entry is not None:
            if entry.is_directory:
                await self._load_directory(entry.path)
            else:
                self.dismiss(entry.path)

    @on(Button.Pressed, "#browser-cancel")
    def cancel(self) -> None:
        self.dismiss(None)
