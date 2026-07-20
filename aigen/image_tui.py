from __future__ import annotations

import curses
import json
import queue
import subprocess
import sys
import termios
import threading
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from aigen.image_edit_commands import IMAGE_EDIT_BACKENDS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LORA_ROOT = PROJECT_ROOT / "loras"
REFERENCE_PACK_ROOT = PROJECT_ROOT / "assets" / "reference-packs"
TABS = ("Images", "Videos", "SAM Edit")
IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
BROWSER_FOCUS_ORDER = ("list", "select", "cancel")


@dataclass
class FormField:
    name: str
    label: str
    value: str
    slot_kind: str | None = None
    slot_id: int | None = None


@dataclass(frozen=True)
class DropdownOption:
    label: str
    value: str


class ImageGenerationTUI:
    def __init__(self) -> None:
        self.active_tab = 0
        self.fields = [
            FormField("prompt", "Prompt", ""),
            FormField("model", "Model", "flux2-klein"),
            FormField("output_dir", "Output directory", "runs/image-tui"),
            FormField("seed", "Seed", "0"),
            FormField("image", "Image 1", "", "image", 1),
        ]
        self.next_slot_id = 2
        self.selected = 0
        self.scroll = 0
        self.status = "Ready."
        self.process: subprocess.Popen[str] | None = None
        self.events: queue.SimpleQueue[tuple[str, int | str]] = queue.SimpleQueue()
        self.visible_rows: dict[int, int] = {}
        self.tab_ranges: list[tuple[int, int]] = []
        self.button_ranges: list[tuple[int, int, int, str]] = []
        self.hover_tab: int | None = None
        self.hover_field: int | None = None
        self.hover_button: int | None = None
        self.last_field_press: int | None = None
        self.last_field_press_at = 0.0
        self.exit_requested = False

    def run(self, screen) -> None:
        old_termios = self._disable_xon_xoff()
        try:
            curses.curs_set(0)
            screen.keypad(True)
            screen.timeout(100)
            curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
            curses.mouseinterval(0)
            self._set_mouse_motion_tracking(True)
            while True:
                self._drain_events(screen)
                self._draw(screen)
                try:
                    key = screen.get_wch()
                except curses.error:
                    continue
                if self._handle_key(screen, key) or self.exit_requested:
                    return
        finally:
            self._set_mouse_motion_tracking(False)
            self._stop_generation()
            self._restore_termios(old_termios)

    def _handle_key(self, screen, key: int | str) -> bool:
        if key == curses.KEY_MOUSE:
            self._handle_mouse(screen)
            return False
        if key in (curses.KEY_LEFT, curses.KEY_BTAB):
            self.active_tab = (self.active_tab - 1) % len(TABS)
            return False
        if key in (curses.KEY_RIGHT, "\t"):
            self.active_tab = (self.active_tab + 1) % len(TABS)
            return False
        if self.active_tab != 0:
            return False
        if key == curses.KEY_UP:
            self.selected = max(0, self.selected - 1)
            self._reveal_selection(screen)
        elif key == curses.KEY_DOWN:
            self.selected = min(len(self.fields) - 1, self.selected + 1)
            self._reveal_selection(screen)
        elif key in ("\n", "\r", curses.KEY_ENTER):
            value = self._edit_field(screen, self.fields[self.selected])
            if value is not None:
                self.fields[self.selected].value = value
        elif key in ("\b", "\x7f", curses.KEY_BACKSPACE):
            self._remove_selected_slot()
        elif key == curses.KEY_DC:
            self.fields[self.selected].value = ""
        return False

    def _handle_mouse(self, screen) -> None:
        try:
            _, x, y, _, state = curses.getmouse()
        except curses.error:
            return
        self._update_hover(x, y)
        if state & curses.REPORT_MOUSE_POSITION:
            return
        if not state & (curses.BUTTON1_PRESSED | curses.BUTTON1_DOUBLE_CLICKED):
            return
        if y == 0:
            for index, (start, end) in enumerate(self.tab_ranges):
                if start <= x < end:
                    self.last_field_press = None
                    self.active_tab = index
                    return
        for button_y, start, end, action in self.button_ranges:
            if y == button_y and start <= x < end:
                self.last_field_press = None
                self._activate_button(screen, action)
                return
        field_index = self.visible_rows.get(y)
        if self.active_tab == 0 and field_index is not None:
            now = time.monotonic()
            double_clicked = bool(state & curses.BUTTON1_DOUBLE_CLICKED) or (
                self.last_field_press == field_index
                and now - self.last_field_press_at <= 0.35
            )
            self.last_field_press = field_index
            self.last_field_press_at = now
            self.selected = field_index
            field = self.fields[self.selected]
            if self._dropdown_options(field) is not None or double_clicked:
                value = self._edit_field(screen, field)
                if value is not None:
                    field.value = value
            return
        self.last_field_press = None

    def _update_hover(self, x: int, y: int) -> None:
        self.hover_tab = None
        self.hover_field = None
        self.hover_button = None
        if y == 0:
            for index, (start, end) in enumerate(self.tab_ranges):
                if start <= x < end:
                    self.hover_tab = index
                    return
        for index, (button_y, start, end, _action) in enumerate(self.button_ranges):
            if y == button_y and start <= x < end:
                self.hover_button = index
                return
        if self.active_tab == 0:
            self.hover_field = self.visible_rows.get(y)

    def _activate_button(self, screen, action: str) -> None:
        if action == "add_image":
            self._add_slot("image")
        elif action == "add_reference_pack":
            self._add_slot("reference_pack")
        elif action == "add_lora":
            self._add_slot("lora")
        elif action == "remove":
            self._remove_selected_slot()
        elif action == "browse":
            self._browse_selected_field(screen)
        elif action == "generate":
            self._start_generation()
        elif action == "stop":
            self._cancel_generation()
        elif action == "quit":
            self.exit_requested = True

    def _add_slot(self, slot_kind: str) -> None:
        slot_id = self.next_slot_id
        self.next_slot_id += 1
        if slot_kind == "lora":
            insert_at = next(
                index for index, item in enumerate(self.fields) if item.name == "output_dir"
            )
            new_fields = [
                FormField("lora", "", "", "lora", slot_id),
                FormField("lora_weight", "", "1.0", "lora", slot_id),
            ]
        elif slot_kind == "image":
            insert_at = next(
                (index for index, item in enumerate(self.fields) if item.slot_kind == "reference_pack"),
                len(self.fields),
            )
            new_fields = [FormField("image", "", "", "image", slot_id)]
        else:
            insert_at = len(self.fields)
            new_fields = [
                FormField("reference_pack", "", "", "reference_pack", slot_id)
            ]
        self.fields[insert_at:insert_at] = new_fields
        self._renumber_slots()
        self.selected = insert_at

    def _remove_selected_slot(self) -> None:
        if self.fields[self.selected].slot_kind is None:
            self.status = "Only image, reference-pack and LoRA slots can be removed."
            return
        selected = self.fields[self.selected]
        self.fields = [
            field
            for field in self.fields
            if field.slot_id != selected.slot_id
        ]
        self.selected = min(self.selected, len(self.fields) - 1)
        self._renumber_slots()

    def _browse_selected_field(self, screen) -> None:
        field = self.fields[self.selected]
        if field.slot_kind == "image":
            directories_only = False
        elif field.name == "output_dir":
            directories_only = True
        else:
            self.status = "Select an Image or Output directory field to browse."
            return
        selected = self._browse_path(
            screen,
            field,
            directories_only=directories_only,
        )
        if selected is not None:
            field.value = self._path_for_field(selected)

    def _renumber_slots(self) -> None:
        counts = {"lora": 0, "image": 0, "reference_pack": 0}
        slot_numbers: dict[tuple[str, int], int] = {}
        for field in self.fields:
            if field.slot_kind is not None:
                key = (field.slot_kind, field.slot_id)
                if key not in slot_numbers:
                    counts[field.slot_kind] += 1
                    slot_numbers[key] = counts[field.slot_kind]
                number = slot_numbers[key]
                if field.name == "lora_weight":
                    field.label = f"LoRA {number} weight"
                elif field.slot_kind == "reference_pack":
                    field.label = f"Reference pack {number}"
                elif field.slot_kind == "lora":
                    field.label = f"LoRA {number}"
                else:
                    field.label = f"{field.slot_kind.title()} {number}"

    def _start_generation(self) -> None:
        if self.process is not None:
            self.status = "Generation is already running."
            return
        try:
            command, output_dir = self._generation_command()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except (OSError, ValueError) as error:
            self.status = str(error)
            return
        self.process = process
        self.status = "Starting generation..."
        threading.Thread(
            target=self._watch_generation,
            args=(self.process, output_dir),
            daemon=True,
        ).start()

    def _generation_command(self) -> tuple[list[str], str]:
        prompt = self._value("prompt").strip()
        model = self._value("model").strip()
        output_dir = self._value("output_dir").strip()
        seed = self._value("seed").strip()
        if not prompt:
            raise ValueError("Prompt is required.")
        if not model:
            raise ValueError("Model is required.")
        if not output_dir:
            raise ValueError("Output directory is required.")
        if not seed:
            raise ValueError("Seed is required.")

        image_paths = [
            Path(field.value).expanduser()
            for field in self.fields
            if field.slot_kind == "image" and field.value.strip()
        ]
        reference_packs = [
            field.value
            for field in self.fields
            if field.slot_kind == "reference_pack" and field.value.strip()
        ]
        if not image_paths and not reference_packs:
            raise ValueError("At least one image or reference pack is required.")

        command = [
            sys.executable,
            "-m",
            "aigen.cli",
            "image-edit",
            "--backend",
            model,
            "--prompt",
            prompt,
            "--output-dir",
            output_dir,
            "--overwrite",
            "--seed",
            seed,
        ]
        for image_path in image_paths:
            command.extend(("--image", image_path.as_posix()))
        for reference_pack in reference_packs:
            command.extend(("--reference-pack", reference_pack))
        for field in self.fields:
            if field.name != "lora" or not field.value.strip():
                continue
            weight = next(
                item.value.strip()
                for item in self.fields
                if item.name == "lora_weight" and item.slot_id == field.slot_id
            )
            command.extend(("--lora", str(Path(field.value).expanduser())))
            command.extend(("--lora-weight", weight or "1.0"))
        return command, output_dir

    def _watch_generation(self, process: subprocess.Popen[str], output_dir: str) -> None:
        assert process.stdout is not None
        assert process.stderr is not None
        stderr_lines: list[str] = []
        stderr_thread = threading.Thread(
            target=stderr_lines.extend,
            args=(process.stderr,),
            daemon=True,
        )
        stderr_thread.start()
        last_line = ""
        for raw_line in process.stdout:
            line = raw_line.strip()
            if line:
                last_line = line
                self.events.put(("progress", line))
        returncode = process.wait()
        stderr_thread.join()
        if returncode == 0:
            self.events.put(("completed", output_dir))
            return
        stderr_text = "".join(stderr_lines).strip()
        self.events.put(("failed", self._error_message(stderr_text or last_line, returncode)))

    @staticmethod
    def _error_message(output: str, returncode: int) -> str:
        if output:
            try:
                payload = json.loads(output)
            except json.JSONDecodeError:
                return output
            message = payload.get("message")
            if isinstance(message, str):
                return message
            return output
        return f"Image generation exited with code {returncode}."

    def _drain_events(self, screen) -> None:
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                return
            if kind == "progress":
                self.status = str(value)
            elif kind == "completed":
                self.process = None
                self.status = f"Output: {value}"
            elif kind == "failed":
                self.process = None
                self.status = "Generation failed."
                self._show_dialog(screen, "Generation failed", str(value))

    def _browse_path(
        self,
        screen,
        field: FormField,
        *,
        directories_only: bool,
    ) -> Path | None:
        current = self._browser_start_directory(field)
        selected = 0
        scroll = 0
        focus = "list"
        hovered_button: str | None = None
        screen.timeout(-1)
        try:
            while True:
                try:
                    entries = self._browser_entries(current, directories_only=directories_only)
                except OSError as error:
                    self._show_dialog(screen, "Cannot open folder", str(error))
                    screen.timeout(-1)
                    current = current.parent
                    selected = 0
                    scroll = 0
                    continue

                height, width = screen.getmaxyx()
                box_width = min(120, width - 4)
                box_height = height - 4
                top = 2
                left = (width - box_width) // 2
                list_top = 2
                visible_count = box_height - 5
                selected = min(selected, max(0, len(entries) - 1))
                if selected < scroll:
                    scroll = selected
                elif selected >= scroll + visible_count:
                    scroll = selected - visible_count + 1
                scroll = min(scroll, max(0, len(entries) - visible_count))

                select_text = "[Select folder]" if directories_only else "[Select image]"
                cancel_text = "[Cancel]"
                button_gap = 2
                buttons_width = len(select_text) + button_gap + len(cancel_text)
                select_x = left + (box_width - buttons_width) // 2
                cancel_x = select_x + len(select_text) + button_gap
                button_y = top + box_height - 2

                window = screen.derwin(box_height, box_width, top, left)
                window.erase()
                window.box()
                title = "Select output folder" if directories_only else "Select image"
                window.addnstr(0, 2, f" {title} ", box_width - 4, curses.A_BOLD)
                current_text = current.as_posix()
                if len(current_text) > box_width - 4:
                    current_text = "…" + current_text[-(box_width - 5) :]
                window.addnstr(1, 2, current_text, box_width - 4, curses.A_BOLD)

                for row, path in enumerate(entries[scroll : scroll + visible_count], start=list_top):
                    index = scroll + row - list_top
                    label = self._browser_entry_label(current, path)
                    attr = (
                        curses.A_REVERSE
                        if index == selected and focus == "list"
                        else curses.A_BOLD if index == selected else 0
                    )
                    window.addnstr(row, 2, label, box_width - 4, attr)

                window.addstr(
                    box_height - 2,
                    select_x - left,
                    select_text,
                    curses.A_REVERSE
                    if hovered_button == "select" or focus == "select"
                    else curses.A_BOLD,
                )
                window.addstr(
                    box_height - 2,
                    cancel_x - left,
                    cancel_text,
                    curses.A_REVERSE
                    if hovered_button == "cancel" or focus == "cancel"
                    else curses.A_BOLD,
                )
                window.refresh()

                key = screen.get_wch()
                if key == "\x1b":
                    return None
                if key == "\t":
                    focus = BROWSER_FOCUS_ORDER[
                        (BROWSER_FOCUS_ORDER.index(focus) + 1) % len(BROWSER_FOCUS_ORDER)
                    ]
                elif key == curses.KEY_BTAB:
                    focus = BROWSER_FOCUS_ORDER[
                        (BROWSER_FOCUS_ORDER.index(focus) - 1) % len(BROWSER_FOCUS_ORDER)
                    ]
                elif key == curses.KEY_UP and entries:
                    focus = "list"
                    selected = (selected - 1) % len(entries)
                elif key == curses.KEY_DOWN and entries:
                    focus = "list"
                    selected = (selected + 1) % len(entries)
                elif key in ("\n", "\r", curses.KEY_ENTER):
                    if focus == "cancel":
                        return None
                    if focus == "select":
                        if directories_only:
                            return current
                        if entries and entries[selected].is_file():
                            return entries[selected]
                    elif entries:
                        target = entries[selected]
                        if target.is_dir():
                            current = target
                            selected = 0
                            scroll = 0
                        elif not directories_only:
                            return target
                elif key == curses.KEY_MOUSE:
                    try:
                        _, mouse_x, mouse_y, _, state = curses.getmouse()
                    except curses.error:
                        continue
                    if state & curses.REPORT_MOUSE_POSITION:
                        hovered_button = None
                        if mouse_y == button_y:
                            if select_x <= mouse_x < select_x + len(select_text):
                                hovered_button = "select"
                            elif cancel_x <= mouse_x < cancel_x + len(cancel_text):
                                hovered_button = "cancel"
                        row = mouse_y - top - list_top
                        index = scroll + row
                        if (
                            left < mouse_x < left + box_width - 1
                            and 0 <= row < visible_count
                            and index < len(entries)
                        ):
                            selected = index
                        continue
                    if state & curses.BUTTON4_PRESSED and entries:
                        focus = "list"
                        selected = max(0, selected - 3)
                        continue
                    if state & curses.BUTTON5_PRESSED and entries:
                        focus = "list"
                        selected = min(len(entries) - 1, selected + 3)
                        continue
                    if not state & (curses.BUTTON1_PRESSED | curses.BUTTON1_DOUBLE_CLICKED):
                        continue
                    if mouse_y == button_y:
                        if select_x <= mouse_x < select_x + len(select_text):
                            if directories_only:
                                return current
                            if entries and entries[selected].is_file():
                                return entries[selected]
                        elif cancel_x <= mouse_x < cancel_x + len(cancel_text):
                            return None
                        continue
                    row = mouse_y - top - list_top
                    index = scroll + row
                    if (
                        left < mouse_x < left + box_width - 1
                        and 0 <= row < visible_count
                        and index < len(entries)
                    ):
                        focus = "list"
                        selected = index
                        target = entries[index]
                        if target.is_dir():
                            current = target
                            selected = 0
                            scroll = 0
                        else:
                            return target
        finally:
            screen.timeout(100)

    @staticmethod
    def _browser_entries(current: Path, *, directories_only: bool) -> tuple[Path, ...]:
        children = (
            path
            for path in current.iterdir()
            if not path.name.startswith(".")
            and (
                path.is_dir()
                or (not directories_only and path.suffix.casefold() in IMAGE_EXTENSIONS)
            )
        )
        entries = sorted(children, key=lambda path: (not path.is_dir(), path.name.casefold()))
        if current != current.parent:
            entries.insert(0, current.parent)
        return tuple(entries)

    @staticmethod
    def _browser_entry_label(current: Path, path: Path) -> str:
        if path == current.parent:
            return "../"
        return path.name + ("/" if path.is_dir() else "")

    @staticmethod
    def _browser_start_directory(field: FormField) -> Path:
        raw_path = Path(field.value).expanduser() if field.value.strip() else PROJECT_ROOT
        path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
        if path.is_dir():
            return path.resolve()
        if path.parent.is_dir():
            return path.parent.resolve()
        return PROJECT_ROOT

    @staticmethod
    def _path_for_field(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return resolved.as_posix()

    def _show_dialog(self, screen, title: str, message: str) -> None:
        screen.timeout(-1)
        scroll: int | None = None
        close_hovered = False
        try:
            while True:
                height, width = screen.getmaxyx()
                box_width = min(100, width - 4)
                content_width = box_width - 4
                lines = [
                    line
                    for paragraph in message.splitlines() or [""]
                    for line in (textwrap.wrap(paragraph, content_width) or [""])
                ]
                visible_count = min(len(lines), height - 7)
                max_scroll = max(0, len(lines) - visible_count)
                scroll = max_scroll if scroll is None else min(scroll, max_scroll)
                box_height = visible_count + 4
                top = (height - box_height) // 2
                left = (width - box_width) // 2
                close_text = "[Close]"
                close_x = left + (box_width - len(close_text)) // 2
                close_y = top + box_height - 2

                window = screen.derwin(box_height, box_width, top, left)
                window.erase()
                window.box()
                window.addnstr(0, 2, f" {title} ", box_width - 4, curses.A_BOLD)
                for row, line in enumerate(lines[scroll : scroll + visible_count], start=1):
                    window.addnstr(row, 2, line, content_width)
                window.addstr(
                    box_height - 2,
                    (box_width - len(close_text)) // 2,
                    close_text,
                    curses.A_REVERSE if close_hovered else curses.A_BOLD,
                )
                window.refresh()

                key = screen.get_wch()
                if key in ("\x1b", "\n", "\r", curses.KEY_ENTER):
                    return
                if key == curses.KEY_UP:
                    scroll = max(0, scroll - 1)
                elif key == curses.KEY_DOWN:
                    scroll = min(max(0, len(lines) - visible_count), scroll + 1)
                elif key == curses.KEY_MOUSE:
                    try:
                        _, mouse_x, mouse_y, _, state = curses.getmouse()
                    except curses.error:
                        continue
                    over_close = (
                        mouse_y == close_y
                        and close_x <= mouse_x < close_x + len(close_text)
                    )
                    if state & curses.REPORT_MOUSE_POSITION:
                        close_hovered = over_close
                    elif state & curses.BUTTON4_PRESSED:
                        scroll = max(0, scroll - 1)
                    elif state & curses.BUTTON5_PRESSED:
                        scroll = min(max(0, len(lines) - visible_count), scroll + 1)
                    elif state & (curses.BUTTON1_PRESSED | curses.BUTTON1_DOUBLE_CLICKED):
                        if over_close:
                            return
        finally:
            screen.timeout(100)

    def _cancel_generation(self) -> None:
        if self.process is None:
            self.status = "No generation is running."
            return
        self.process.terminate()
        self.status = "Stopping generation..."

    def _stop_generation(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def _value(self, name: str) -> str:
        return next(field.value for field in self.fields if field.name == name)

    def _edit_field(self, screen, field: FormField) -> str | None:
        options = self._dropdown_options(field)
        if options is not None:
            return self._select_option(screen, field.label, options, field.value)
        return self._edit_line(screen, field)

    def _dropdown_options(self, field: FormField) -> tuple[DropdownOption, ...] | None:
        if field.name == "model":
            return tuple(DropdownOption(backend, backend) for backend in IMAGE_EDIT_BACKENDS)
        if field.name == "lora":
            weights = sorted(LORA_ROOT.glob("*.safetensors"), key=lambda path: path.name.casefold())
            return (
                DropdownOption("None", ""),
                *(DropdownOption(path.name, path.relative_to(PROJECT_ROOT).as_posix()) for path in weights),
            )
        if field.name == "reference_pack":
            packs = sorted(
                REFERENCE_PACK_ROOT.glob("*.json"),
                key=lambda path: path.stem.casefold(),
            )
            return (
                DropdownOption("None", ""),
                *(
                    DropdownOption(
                        path.stem,
                        path.relative_to(PROJECT_ROOT).as_posix(),
                    )
                    for path in packs
                ),
            )
        return None

    def _select_option(
        self,
        screen,
        title: str,
        options: tuple[DropdownOption, ...],
        current: str,
    ) -> str | None:
        selected = next(
            (index for index, option in enumerate(options) if option.value == current),
            0,
        )
        screen.timeout(-1)
        try:
            while True:
                height, width = screen.getmaxyx()
                visible_count = min(len(options), height - 4)
                box_width = min(
                    max(32, len(title) + 6, max(len(option.label) for option in options) + 4),
                    width - 4,
                )
                box_height = visible_count + 2
                top = (height - box_height) // 2
                left = (width - box_width) // 2
                scroll = min(
                    max(0, selected - visible_count + 1),
                    max(0, len(options) - visible_count),
                )
                window = screen.derwin(box_height, box_width, top, left)
                window.erase()
                window.box()
                window.addnstr(0, 2, f" {title} ", box_width - 4, curses.A_BOLD)
                for row, option in enumerate(options[scroll : scroll + visible_count], start=1):
                    index = scroll + row - 1
                    attr = curses.A_REVERSE if index == selected else 0
                    window.addnstr(row, 2, option.label, box_width - 4, attr)
                window.refresh()

                key = screen.get_wch()
                if key == "\x1b":
                    return None
                if key == curses.KEY_UP:
                    selected = (selected - 1) % len(options)
                elif key == curses.KEY_DOWN:
                    selected = (selected + 1) % len(options)
                elif key in ("\n", "\r", curses.KEY_ENTER):
                    return options[selected].value
                elif key == curses.KEY_MOUSE:
                    try:
                        _, mouse_x, mouse_y, _, state = curses.getmouse()
                    except curses.error:
                        continue
                    option_index = scroll + mouse_y - top - 1
                    inside = (
                        left < mouse_x < left + box_width - 1
                        and 0 <= mouse_y - top - 1 < visible_count
                        and option_index < len(options)
                    )
                    if state & curses.REPORT_MOUSE_POSITION:
                        if inside:
                            selected = option_index
                        continue
                    if state & (curses.BUTTON1_PRESSED | curses.BUTTON1_DOUBLE_CLICKED):
                        if inside:
                            return options[option_index].value
                        return None
        finally:
            screen.timeout(100)

    def _edit_line(self, screen, field: FormField) -> str | None:
        curses.curs_set(1)
        screen.timeout(-1)
        buffer = list(field.value)
        position = len(buffer)
        try:
            while True:
                height, width = screen.getmaxyx()
                y = height - 2
                prefix = f"{field.label}: "
                available = max(1, width - len(prefix) - 1)
                start = max(0, position - available + 1)
                visible = "".join(buffer[start : start + available])
                screen.move(y, 0)
                screen.clrtoeol()
                screen.addnstr(y, 0, prefix, width - 1, curses.A_BOLD)
                screen.addnstr(y, len(prefix), visible, available)
                screen.move(y, min(width - 1, len(prefix) + position - start))
                screen.refresh()
                key = screen.get_wch()
                if key == "\x1b":
                    return None
                if key in ("\n", "\r", curses.KEY_ENTER):
                    return "".join(buffer)
                if key in ("\b", "\x7f", curses.KEY_BACKSPACE):
                    if position:
                        del buffer[position - 1]
                        position -= 1
                elif key == curses.KEY_DC:
                    if position < len(buffer):
                        del buffer[position]
                elif key == curses.KEY_LEFT:
                    position = max(0, position - 1)
                elif key == curses.KEY_RIGHT:
                    position = min(len(buffer), position + 1)
                elif key == curses.KEY_HOME:
                    position = 0
                elif key == curses.KEY_END:
                    position = len(buffer)
                elif isinstance(key, str) and key.isprintable():
                    buffer.insert(position, key)
                    position += 1
        finally:
            screen.timeout(100)
            curses.curs_set(0)

    def _draw(self, screen) -> None:
        screen.erase()
        height, width = screen.getmaxyx()
        if height < 12 or width < 64:
            screen.addnstr(0, 0, "Terminal must be at least 64x12.", max(1, width - 1))
            screen.refresh()
            return

        self.tab_ranges = []
        x = 0
        for index, tab in enumerate(TABS):
            text = f" {tab} "
            attr = curses.A_REVERSE | curses.A_BOLD if index == self.active_tab else curses.A_BOLD
            if index == self.hover_tab and index != self.active_tab:
                attr |= curses.A_UNDERLINE
            screen.addnstr(0, x, text, width - x - 1, attr)
            self.tab_ranges.append((x, x + len(text)))
            x += len(text) + 1

        self.visible_rows.clear()
        self.button_ranges.clear()
        if self.active_tab == 0:
            self._draw_image_fields(screen, height, width)
            self._draw_buttons(screen, height - 2, width)
        else:
            self._draw_buttons(screen, height - 2, width, empty_tab=True)

        running = " [RUNNING]" if self.process is not None else ""
        screen.addnstr(height - 1, 0, (self.status + running).ljust(width - 1), width - 1, curses.A_REVERSE)
        screen.refresh()

    def _draw_image_fields(self, screen, height: int, width: int) -> None:
        first_row = 2
        available_rows = height - first_row - 3
        self.scroll = min(self.scroll, max(0, len(self.fields) - available_rows))
        label_width = min(20, max(len(field.label) for field in self.fields) + 1)
        value_width = max(1, width - label_width - 4)
        for row_offset, field_index in enumerate(
            range(self.scroll, min(len(self.fields), self.scroll + available_rows))
        ):
            y = first_row + row_offset
            field = self.fields[field_index]
            selected = field_index == self.selected
            hovered = field_index == self.hover_field and not selected
            attr = curses.A_REVERSE if selected else curses.A_UNDERLINE if hovered else 0
            label_attr = curses.A_BOLD | (curses.A_UNDERLINE if hovered else 0)
            screen.addnstr(y, 1, field.label.ljust(label_width), label_width, label_attr)
            display = self._display_value(field, value_width)
            screen.addnstr(y, label_width + 2, display.ljust(value_width), value_width, attr)
            self.visible_rows[y] = field_index

    def _draw_buttons(self, screen, y: int, width: int, *, empty_tab: bool = False) -> None:
        actions = [("Quit", "quit")]
        if not empty_tab:
            actions = [
                ("+ Image", "add_image"),
                ("+ Pack", "add_reference_pack"),
                ("+ LoRA", "add_lora"),
                ("Remove", "remove"),
                ("Browse", "browse"),
                (
                    "Stop" if self.process is not None else "Generate",
                    "stop" if self.process is not None else "generate",
                ),
                *actions,
            ]
        x = 0
        for index, (label, action) in enumerate(actions):
            text = f"[{label}]"
            if x + len(text) >= width:
                break
            attr = curses.A_BOLD
            if index == self.hover_button:
                attr |= curses.A_REVERSE
            screen.addstr(y, x, text, attr)
            self.button_ranges.append((y, x, x + len(text), action))
            x += len(text) + 1

    def _display_value(self, field: FormField, width: int) -> str:
        dropdown = field.name in ("model", "lora", "reference_pack")
        marker = " v" if dropdown else ""
        value_width = max(0, width - len(marker))
        if len(field.value) <= value_width:
            return field.value + marker
        if field.name == "prompt":
            return field.value[: max(0, value_width - 1)] + "…" + marker
        return "…" + field.value[-max(0, value_width - 1) :] + marker

    def _reveal_selection(self, screen) -> None:
        height, _ = screen.getmaxyx()
        available_rows = max(1, height - 6)
        if self.selected < self.scroll:
            self.scroll = self.selected
        elif self.selected >= self.scroll + available_rows:
            self.scroll = self.selected - available_rows + 1

    def _disable_xon_xoff(self):
        try:
            file_descriptor = sys.stdin.fileno()
            old = termios.tcgetattr(file_descriptor)
            new = termios.tcgetattr(file_descriptor)
            new[0] &= ~(termios.IXON | termios.IXOFF | termios.IXANY)
            termios.tcsetattr(file_descriptor, termios.TCSANOW, new)
            return old
        except (OSError, termios.error):
            return None

    def _set_mouse_motion_tracking(self, enabled: bool) -> None:
        sys.stdout.write("\x1b[?1003h" if enabled else "\x1b[?1003l")
        sys.stdout.flush()

    def _restore_termios(self, old) -> None:
        if old is None:
            return
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, old)


def main() -> None:
    curses.wrapper(ImageGenerationTUI().run)


if __name__ == "__main__":
    main()
