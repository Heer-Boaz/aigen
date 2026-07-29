from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from textual import on
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from aigen.runtime_profiles import display_project_path


_UNSAVED_OPTION_ID = "workflow-document:unsaved"
_PATH_OPTION_PREFIX = "workflow-document:path:"


class WorkflowDocumentList(OptionList):
    """Keyboard and mouse navigation for workflows seen in this session."""

    DEFAULT_CSS = """
    WorkflowDocumentList {
        width: 100%;
        height: 1fr;
        border: none;
        padding: 0 1;
    }
    """

    class DocumentActivated(Message):
        def __init__(self, path: Path | None) -> None:
            super().__init__()
            self.path = path

    def __init__(
        self,
        document_paths: Sequence[Path],
        *,
        current_path: Path | None,
        current_name: str,
        dirty: bool,
        id: str | None = None,
    ) -> None:
        self._paths_by_option_id: dict[str, Path] = {}
        super().__init__(
            *self._document_options(
                document_paths,
                current_path=current_path,
                current_name=current_name,
                dirty=dirty,
            ),
            id=id,
            markup=False,
            compact=True,
        )
        self.highlighted = self.get_option_index(
            self._current_option_id(current_path)
        )

    def show_documents(
        self,
        document_paths: Sequence[Path],
        *,
        current_path: Path | None,
        current_name: str,
        dirty: bool,
    ) -> None:
        highlighted_id = (
            self.highlighted_option.id
            if self.highlighted_option is not None
            else None
        )
        options = self._document_options(
            document_paths,
            current_path=current_path,
            current_name=current_name,
            dirty=dirty,
        )
        option_ids = {option.id for option in options}
        self.set_options(options)
        target_id = (
            highlighted_id
            if highlighted_id in option_ids
            else self._current_option_id(current_path)
        )
        self.highlighted = self.get_option_index(target_id)

    @on(OptionList.OptionSelected)
    def option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        option_id = event.option_id
        assert option_id is not None
        self.post_message(
            self.DocumentActivated(
                (
                    None
                    if option_id == _UNSAVED_OPTION_ID
                    else self._paths_by_option_id[option_id]
                )
            )
        )

    def _document_options(
        self,
        document_paths: Sequence[Path],
        *,
        current_path: Path | None,
        current_name: str,
        dirty: bool,
    ) -> tuple[Option, ...]:
        if current_path is not None:
            assert current_path in document_paths
        self._paths_by_option_id.clear()
        options: list[Option] = []
        if current_path is None:
            options.append(
                Option(
                    _document_prompt(
                        current=True,
                        dirty=dirty,
                        name=current_name,
                        location="Unsaved",
                    ),
                    id=_UNSAVED_OPTION_ID,
                )
            )
        for path in document_paths:
            option_id = _path_option_id(path)
            self._paths_by_option_id[option_id] = path
            current = path == current_path
            options.append(
                Option(
                    _document_prompt(
                        current=current,
                        dirty=dirty if current else False,
                        name=current_name if current else path.stem,
                        location=display_project_path(path),
                    ),
                    id=option_id,
                )
            )
        return tuple(options)

    @staticmethod
    def _current_option_id(current_path: Path | None) -> str:
        return (
            _UNSAVED_OPTION_ID
            if current_path is None
            else _path_option_id(current_path)
        )


def _path_option_id(path: Path) -> str:
    return f"{_PATH_OPTION_PREFIX}{path.as_posix()}"


def _document_prompt(
    *,
    current: bool,
    dirty: bool,
    name: str,
    location: str,
) -> str:
    return (
        f"{'▶' if current else ' '} "
        f"{'*' if dirty else ' '} "
        f"{name} | {location}"
    )
