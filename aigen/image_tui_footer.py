from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.geometry import Offset
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, ContentSwitcher

from aigen.tui_choice_menu import ChoiceMenu, MenuChoice
from aigen.tui_action_button import ActionButton


class ImageTUIFooter(Widget):
    """Task actions stay visible; additional commands have readable menus."""

    class CommandRequested(Message):
        def __init__(self, command: str) -> None:
            super().__init__()
            self.command = command

    DEFAULT_CSS = """
    ImageTUIFooter { width: 100%; height: 1; }
    ImageTUIFooter .footer-row { width: 100%; height: 1; padding: 0 1; }
    ImageTUIFooter ContentSwitcher { width: 1fr; height: 1; }
    ImageTUIFooter .tab-actions { width: 100%; height: 1; }
    ImageTUIFooter Button {
        width: auto; min-width: 0; height: 1; min-height: 1;
        border: none; padding: 0 1; margin-right: 1;
    }
    """

    ACTION_IDS = {
        "images": "image-actions", "videos": "video-actions", "sam-edit": "sam-actions",
        "postprocessing": "postprocess-actions", "workflows": "workflow-actions",
    }
    MENUS = {
        "image-inputs": ("Add input", (
            ("add-image", "Image…"), ("add-reference-pack", "Reference pack…"),
            ("add-seed", "Seed"), ("add-lora", "LoRA…"),
        )),
        "image-more": ("Image actions", (
            ("browse", "Browse selected path…"), ("remove", "Remove selected input"),
            ("use-result", "Use a saved result…"), ("save-pack", "Save reference pack…"),
            ("save-config", "Save configuration…"), ("load-config", "Open configuration…"),
        )),
        "video-inputs": ("Add input", (
            ("add-keyframe", "Keyframe…"), ("add-video-image", "Image…"), ("add-video-seed", "Seed"),
        )),
        "video-more": ("Video actions", (
            ("browse-video", "Browse selected path…"), ("remove-video", "Remove selected input"),
        )),
        "sam-more": ("Region actions", (
            ("browse-sam", "Browse selected path…"), ("sam-clear", "Clear region selection"),
        )),
        "workflow-new-menu": ("New workflow", (
            ("workflow-new", "Image workflow"), ("workflow-new-character", "Character workflow"),
            ("workflow-new-video", "Video workflow"),
        )),
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._disabled_actions: set[str] = set()

    def compose(self) -> ComposeResult:
        with Horizontal(classes="footer-row"):
            with ContentSwitcher(initial="image-actions"):
                with Horizontal(id="image-actions", classes="tab-actions"):
                    yield ActionButton("Generate", name="generate", id="generation-action", variant="primary", compact=True)
                    yield Button("Open as workflow", name="image-workflow", compact=True)
                    yield Button("Add input…", name="image-inputs", compact=True)
                    yield Button("More…", name="image-more", compact=True)
                with Horizontal(id="video-actions", classes="tab-actions"):
                    yield ActionButton("Generate", name="video-generate", id="video-action", variant="primary", compact=True)
                    yield Button("Open as workflow", name="video-workflow", compact=True)
                    yield Button("Add input…", name="video-inputs", compact=True)
                    yield Button("More…", name="video-more", compact=True)
                with Horizontal(id="sam-actions", classes="tab-actions"):
                    yield ActionButton("Run", name="sam-segment", id="sam-action", variant="primary", compact=True)
                    yield Button("Select region", name="sam-edit", id="sam-edit-prompts", compact=True)
                    yield Button("Open as workflow", name="sam-workflow", id="sam-workflow", compact=True)
                    yield Button("More…", name="sam-more", compact=True)
                with Horizontal(id="postprocess-actions", classes="tab-actions"):
                    yield ActionButton("Process", name="postprocess", id="postprocess-action", variant="primary", compact=True)
                with Horizontal(id="workflow-actions", classes="tab-actions"):
                    yield Button("Open editor", name="workflow-open", variant="primary", compact=True)
                    yield Button("New workflow…", name="workflow-new-menu", compact=True)
                    yield Button("Open file…", name="workflow-load", compact=True)
            yield Button("Quit", name="quit", compact=True)

    def show_tab(self, tab_id: str) -> None:
        self.query_one(ContentSwitcher).current = self.ACTION_IDS[tab_id]

    def set_action_enabled(self, command: str, enabled: bool) -> None:
        if enabled:
            self._disabled_actions.discard(command)
        else:
            self._disabled_actions.add(command)

    @on(Button.Pressed)
    def open_menu(self, event: Button.Pressed) -> None:
        name = event.button.name
        if name not in self.MENUS:
            return
        event.stop()
        title, actions = self.MENUS[name]
        choices = tuple(MenuChoice(command, label, disabled=command in self._disabled_actions)
                        for command, label in actions)
        self.app.push_screen(ChoiceMenu(title, choices,
            anchor=Offset(event.button.region.x, event.button.region.y - len(choices) - 3)), self._chosen)

    def _chosen(self, command: str | None) -> None:
        if command is not None:
            self.post_message(self.CommandRequested(command))
