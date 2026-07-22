from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, ItemGrid
from textual.screen import ModalScreen
from textual.widgets import Button, Label

from aigen.sam_prompt_canvas import SAMPromptCanvas


class SAMPromptDialog(ModalScreen[None]):
    def __init__(
        self,
        *,
        image: str,
        prompt_mode: str,
        box: str,
        positive_points: str,
        negative_points: str,
    ) -> None:
        super().__init__(classes="sam-prompt-dialog-screen")
        self._state = {
            "image": image,
            "prompt_mode": prompt_mode,
            "box": box,
            "positive_points": positive_points,
            "negative_points": negative_points,
        }

    def compose(self) -> ComposeResult:
        with Container(classes="sam-prompt-dialog"):
            yield Label("SAM prompt editor", classes="dialog-title")
            yield SAMPromptCanvas(id="sam-dialog-canvas")
            yield ItemGrid(
                Button("Clear", name="sam-prompt-clear", id="sam-prompt-clear", compact=True),
                Button("Save selection", name="sam-prompt-save", id="sam-prompt-save", compact=True),
                Button("Load selection", name="sam-prompt-load", id="sam-prompt-load", compact=True),
                Button("Done", name="sam-prompt-close", id="sam-prompt-close", variant="primary", compact=True),
                min_column_width=16,
                stretch_height=False,
                id="sam-prompt-dialog-actions",
            )

    def on_mount(self) -> None:
        self._update_canvas()

    def set_state(
        self,
        *,
        image: str,
        prompt_mode: str,
        box: str,
        positive_points: str,
        negative_points: str,
    ) -> None:
        self._state = {
            "image": image,
            "prompt_mode": prompt_mode,
            "box": box,
            "positive_points": positive_points,
            "negative_points": negative_points,
        }
        if self.is_mounted:
            self._update_canvas()

    def clear_prompts(self) -> None:
        self.query_one("#sam-dialog-canvas", SAMPromptCanvas).clear_prompts()

    def close(self) -> None:
        self.dismiss()

    def _update_canvas(self) -> None:
        canvas = self.query_one("#sam-dialog-canvas", SAMPromptCanvas)
        canvas.set_state(
            self._state["image"],
            prompt_mode=self._state["prompt_mode"],
            box=self._state["box"],
            positive_points=self._state["positive_points"],
            negative_points=self._state["negative_points"],
        )
