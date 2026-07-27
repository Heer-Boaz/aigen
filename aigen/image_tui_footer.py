from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, ItemGrid
from textual.widget import Widget
from textual.widgets import Button, ContentSwitcher


class ImageTUIFooter(Widget):
    DEFAULT_CSS = """
    ImageTUIFooter {
        width: 100%;
        height: auto;
    }

    ImageTUIFooter .footer-row {
        width: 100%;
        height: auto;
        padding: 0 1;
    }

    ImageTUIFooter .context-actions {
        width: 1fr;
        height: auto;
    }

    ImageTUIFooter .tab-actions {
        width: 100%;
        height: auto;
        grid-gutter: 0;
    }

    ImageTUIFooter .global-actions {
        width: 8;
        min-width: 8;
        height: 1;
        grid-gutter: 0;
    }

    ImageTUIFooter Button {
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 1;
    }
    """

    ACTION_IDS = {
        "images": "image-actions",
        "videos": "video-actions",
        "sam-edit": "sam-actions",
        "postprocessing": "postprocess-actions",
    }

    def compose(self) -> ComposeResult:
        with Horizontal(classes="footer-row"):
            with ContentSwitcher(
                initial="image-actions",
                classes="context-actions",
            ):
                yield ItemGrid(
                    Button("+ Seed", name="add-seed", compact=True),
                    Button("+ Image", name="add-image", compact=True),
                    Button("+ Pack", name="add-reference-pack", compact=True),
                    Button("+ LoRA", name="add-lora", compact=True),
                    Button("Remove", name="remove", compact=True),
                    Button("Browse", name="browse", compact=True),
                    Button("Use Result", name="use-result", compact=True),
                    Button("Save Pack", name="save-pack", compact=True),
                    Button("Save Config", name="save-config", compact=True),
                    Button("Load Config", name="load-config", compact=True),
                    Button(
                        "Generate",
                        name="generate",
                        id="generation-action",
                        variant="primary",
                        compact=True,
                    ),
                    min_column_width=12,
                    stretch_height=False,
                    id="image-actions",
                    classes="tab-actions",
                )
                yield ItemGrid(
                    Button(
                        "+ Keyframe",
                        name="add-keyframe",
                        id="video-add-keyframe",
                        compact=True,
                    ),
                    Button(
                        "+ Seed",
                        name="add-video-seed",
                        id="video-add-seed",
                        compact=True,
                    ),
                    Button(
                        "+ Image",
                        name="add-video-image",
                        id="video-add-image",
                        compact=True,
                    ),
                    Button("Remove", name="remove-video", compact=True),
                    Button("Browse", name="browse-video", compact=True),
                    Button(
                        "Generate",
                        name="video-generate",
                        id="video-action",
                        variant="primary",
                        compact=True,
                    ),
                    min_column_width=12,
                    stretch_height=False,
                    id="video-actions",
                    classes="tab-actions",
                )
                yield ItemGrid(
                    Button("Browse", name="browse-sam", compact=True),
                    Button(
                        "Edit prompts",
                        name="sam-edit",
                        id="sam-edit-prompts",
                        compact=True,
                    ),
                    Button("Clear prompts", name="sam-clear", compact=True),
                    Button(
                        "Run",
                        name="sam-segment",
                        id="sam-action",
                        variant="primary",
                        compact=True,
                    ),
                    min_column_width=12,
                    stretch_height=False,
                    id="sam-actions",
                    classes="tab-actions",
                )
                yield ItemGrid(
                    Button(
                        "Process",
                        name="postprocess",
                        id="postprocess-action",
                        variant="primary",
                        compact=True,
                    ),
                    min_column_width=12,
                    stretch_height=False,
                    id="postprocess-actions",
                    classes="tab-actions",
                )
            yield ItemGrid(
                Button("Quit", name="quit", compact=True),
                min_column_width=8,
                stretch_height=False,
                classes="global-actions",
            )

    def show_tab(self, tab_id: str) -> None:
        self.query_one(ContentSwitcher).current = self.ACTION_IDS[tab_id]
