from textual.content import Content, ContentText
from textual.reactive import reactive
from textual.widgets import Button


class ActionButton(Button):
    """Action labels participate in layout when the available operation changes."""

    label: reactive[ContentText] = reactive[ContentText](Content.empty(), layout=True)
