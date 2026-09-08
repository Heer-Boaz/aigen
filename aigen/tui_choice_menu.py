from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from rich.cells import cell_len
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.geometry import Offset
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList
from textual.widgets.option_list import Option


@dataclass(frozen=True, slots=True)
class MenuChoice:
    id: str
    label: str
    shortcut: str = ""
    disabled: bool = False


class ChoiceMenu(ModalScreen[str | None]):
    """Anchored, keyboard-accessible choices with optional filtering."""

    BINDINGS = [
        Binding("escape", "dismiss(None)", show=False),
        Binding("down", "focus_choices", show=False),
    ]
    DEFAULT_CSS = """
    ChoiceMenu { background: transparent; }
    ChoiceMenu #choice-menu-panel {
        position: absolute;
        border: solid #b681e6;
        background: #211a2d;
        padding: 0 1;
    }
    ChoiceMenu #choice-menu-title { height: 1; color: #d8c5eb; text-style: bold; }
    ChoiceMenu #choice-menu-search { height: 1; border: none; padding: 0; }
    ChoiceMenu #choice-menu-options { height: 1fr; border: none; padding: 0; background: #211a2d; }
    """

    def __init__(
        self,
        title: str,
        choices: Sequence[MenuChoice],
        *,
        anchor: Offset | None = None,
        searchable: bool = False,
    ) -> None:
        super().__init__()
        self._title = title
        self._choices = tuple(choices)
        self._filtered = self._choices
        self._anchor = anchor
        self._searchable = searchable
        self._label_width = max((cell_len(choice.label) for choice in choices), default=0)
        self._width = max(
            24, cell_len(title) + 4,
            self._label_width + max((cell_len(choice.shortcut) for choice in choices), default=0) + 6,
        )

    def compose(self) -> ComposeResult:
        with Container(id="choice-menu-panel"):
            yield Label(self._title, id="choice-menu-title", markup=False)
            if self._searchable:
                yield Input(placeholder="Search nodes…", compact=True, id="choice-menu-search")
            yield OptionList(*self._options(), id="choice-menu-options")

    def _options(self) -> list[Option]:
        return [
            Option(
                Text(choice.label + " " * (self._label_width - cell_len(choice.label))
                     + (f"  {choice.shortcut}" if choice.shortcut else "")),
                id=choice.id, disabled=choice.disabled,
            )
            for choice in self._filtered
        ]

    def on_mount(self) -> None:
        self._place()
        self.query_one(Input if self._searchable else OptionList).focus()

    def on_resize(self, event: events.Resize) -> None:
        self._place()

    def _place(self) -> None:
        width = min(self._width, self.size.width)
        height = min(len(self._filtered) + 3 + int(self._searchable), self.size.height)
        anchor = self._anchor or Offset((self.size.width - width) // 2, min(3, self.size.height - height))
        panel = self.query_one("#choice-menu-panel")
        panel.styles.width = width
        panel.styles.height = height
        panel.styles.offset = (
            max(0, min(anchor.x, self.size.width - width)),
            max(0, min(anchor.y, self.size.height - height)),
        )

    @on(Input.Changed, "#choice-menu-search")
    def filter_choices(self, event: Input.Changed) -> None:
        query = event.value.casefold().strip()
        self._filtered = tuple(choice for choice in self._choices if query in choice.label.casefold())
        options = self.query_one(OptionList)
        options.clear_options().add_options(self._options())
        options.highlighted = 0 if self._filtered else None
        self._place()

    def action_focus_choices(self) -> None:
        self.query_one(OptionList).focus()

    @on(Input.Submitted, "#choice-menu-search")
    def choose_search_result(self) -> None:
        self.query_one(OptionList).action_select()

    @on(OptionList.OptionSelected)
    def choose(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if not self.query_one("#choice-menu-panel").region.contains(*event.screen_offset):
            event.stop()
            self.dismiss(None)
