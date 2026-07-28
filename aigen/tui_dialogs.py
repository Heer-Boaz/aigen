from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static


class MessageDialog(ModalScreen[None]):
    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self.title = title
        self.message = message

    def compose(self) -> ComposeResult:
        with Container(classes="dialog"):
            yield Label(self.title, classes="dialog-title")
            yield Static(self.message, classes="dialog-message")
            yield Button(
                "Close",
                variant="primary",
                id="close-dialog",
                compact=True,
            )

    @on(Button.Pressed, "#close-dialog")
    def close_dialog(self) -> None:
        self.dismiss()


class PromptDialog(ModalScreen[str | None]):
    def __init__(
        self,
        title: str,
        label: str,
        value: str = "",
    ) -> None:
        super().__init__()
        self.title = title
        self.label = label
        self.value = value

    def compose(self) -> ComposeResult:
        with Container(classes="dialog"):
            yield Label(self.title, classes="dialog-title")
            yield Label(self.label)
            yield Input(self.value, id="dialog-input", compact=True)
            with Horizontal(classes="dialog-actions"):
                yield Button(
                    "OK",
                    variant="primary",
                    id="dialog-ok",
                    compact=True,
                )
                yield Button(
                    "Cancel",
                    id="dialog-cancel",
                    compact=True,
                )

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    @on(Input.Submitted, "#dialog-input")
    @on(Button.Pressed, "#dialog-ok")
    def accept(self) -> None:
        self.dismiss(self.query_one(Input).value)

    @on(Button.Pressed, "#dialog-cancel")
    def cancel(self) -> None:
        self.dismiss(None)


class ConfirmationDialog(ModalScreen[bool]):
    def __init__(
        self,
        title: str,
        message: str,
        *,
        confirm_label: str,
    ) -> None:
        super().__init__()
        self.title = title
        self.message = message
        self.confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Container(classes="dialog"):
            yield Label(self.title, classes="dialog-title")
            yield Static(self.message, classes="dialog-message")
            with Horizontal(classes="dialog-actions"):
                yield Button(
                    "Cancel",
                    id="confirmation-cancel",
                    compact=True,
                )
                yield Button(
                    self.confirm_label,
                    variant="error",
                    id="confirmation-confirm",
                    compact=True,
                )

    @on(Button.Pressed, "#confirmation-cancel")
    def cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#confirmation-confirm")
    def confirm(self) -> None:
        self.dismiss(True)
