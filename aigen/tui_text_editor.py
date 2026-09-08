from __future__ import annotations

from textual.binding import Binding
from textual.widgets import TextArea


class MultilineInput(TextArea):
    """Native text editing with form navigation and a viewport-sized default."""

    BINDINGS = [Binding("ctrl+shift+z", "redo", show=False)]
    DEFAULT_CSS = """
    MultilineInput {
        height: 25vh;
        min-height: 4;
        max-height: 12;
        border: none;
        padding: 0 1;
        scrollbar-size-vertical: 1;
    }
    """

    def __init__(self, text: str = "", **kwargs) -> None:
        super().__init__(text, soft_wrap=True, tab_behavior="focus", compact=True,
                         highlight_cursor_line=False, **kwargs)
