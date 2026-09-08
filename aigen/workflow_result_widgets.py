from __future__ import annotations

from PIL import Image
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Label, Select
from textual_image.widget import HalfcellImage

from aigen.workflow_artifacts import ImageCandidate
from aigen.workflow_media_results import MediaCandidate


class ResultPreview(Container, can_focus=True):
    """A whole candidate tile is one mouse/keyboard target."""

    BINDINGS = [Binding('enter,space', 'inspect', show=False)]
    DEFAULT_CSS = '''
    ResultPreview { height: 7; border: solid #352944; padding: 0 1; }
    ResultPreview.-highlighted, ResultPreview:focus { border: solid #b791dd; }
    ResultPreview .result-preview-image { height: 3; width: 1fr; align: center middle; }
    ResultPreview HalfcellImage { width: auto; height: auto; }
    ResultPreview Label { height: 2; text-overflow: ellipsis; }
    '''

    class Highlighted(Message):
        def __init__(self, candidate: ImageCandidate | MediaCandidate, index: int, view_id: int) -> None:
            super().__init__()
            self.candidate, self.index, self.view_id = candidate, index, view_id

    def __init__(self, label: str, preview: Image.Image | None, index: int,
                 candidate: ImageCandidate | MediaCandidate, view_id: int) -> None:
        super().__init__(id=f'candidate-{index}')
        self.label, self.preview, self.index = label, preview, index
        self.candidate, self.view_id = candidate, view_id
        self.tooltip = label

    def compose(self) -> ComposeResult:
        with Container(classes='result-preview-image'):
            if self.preview is not None:
                yield HalfcellImage(self.preview)
        yield Label(self.label, markup=False)

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.action_inspect()

    def action_inspect(self) -> None:
        self.post_message(self.Highlighted(self.candidate, self.index, self.view_id))


class ResultComparison(Container):
    """Keep the chosen output identifiable when viewing or comparing its inputs."""

    DEFAULT_CSS = '''
    ResultComparison { height: 1fr; min-height: 5; }
    ResultComparison #result-view-controls { height: 1; }
    ResultComparison #result-view-mode { width: 18; }
    ResultComparison #result-input-choice { width: 1fr; }
    ResultComparison #result-comparison-panes { height: 1fr; }
    ResultComparison .result-image-pane { width: 1fr; height: 1fr; }
    ResultComparison .result-image-frame { width: 1fr; height: 1fr; align: center middle; }
    ResultComparison .result-image-title { height: 1; text-overflow: ellipsis; }
    ResultComparison HalfcellImage { width: auto; height: auto; }
    ResultComparison .result-image-empty { width: 1fr; height: 1fr; content-align: center middle; }
    '''

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._inputs: list[tuple[str, Image.Image | None]] = []
        self._mode = 'output'
        self._wide: bool | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id='result-view-controls'):
            yield Select((('Output', 'output'), ('Input', 'input')), value='output', allow_blank=False,
                         compact=True, id='result-view-mode')
            yield Select([], prompt='Compare with input', compact=True, id='result-input-choice')
        with Horizontal(id='result-comparison-panes'):
            with Container(id='result-inputs', classes='result-image-pane'):
                yield Label('Input', classes='result-image-title', markup=False)
                with Container(classes='result-image-frame'):
                    yield HalfcellImage(id='result-input-image')
                    yield Label('No recorded input', id='result-input-empty', classes='result-image-empty', markup=False)
            with Container(id='result-output', classes='result-image-pane'):
                yield Label('Output', classes='result-image-title', markup=False)
                with Container(classes='result-image-frame'):
                    yield HalfcellImage(id='result-output-image')
                    yield Label('No output selected', id='result-output-empty', classes='result-image-empty', markup=False)

    def on_mount(self) -> None:
        self.clear()
        self._layout()

    def on_resize(self) -> None:
        self._layout()

    def _layout(self) -> None:
        wide = self.size.width >= 100
        selector = self.query_one('#result-view-mode', Select)
        if wide != self._wide:
            self._wide = wide
            choices = [('Output', 'output'), ('Input', 'input')]
            if wide:
                choices.insert(1, ('Compare', 'compare'))
            with selector.prevent(Select.Changed):
                selector.set_options(choices)
                selector.value = self._mode if wide or self._mode != 'compare' else 'output'
        mode = selector.value
        self.query_one('#result-inputs').display = mode in ('input', 'compare')
        self.query_one('#result-output').display = mode in ('output', 'compare')
        self.query_one('#result-input-choice').display = mode in ('input', 'compare') and bool(self._inputs)

    @on(Select.Changed, '#result-view-mode')
    def mode_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, str):
            self._mode = event.value
            self._layout()

    @on(Select.Changed, '#result-input-choice')
    def input_changed(self, event: Select.Changed) -> None:
        if isinstance(event.value, int) and event.value < len(self._inputs):
            label, preview = self._inputs[event.value]
            self._show_image('input', preview, label)

    def set_inputs(self, inputs: list[tuple[str, Image.Image | None]]) -> None:
        self._inputs = inputs
        selector = self.query_one('#result-input-choice', Select)
        previous = selector.value
        with selector.prevent(Select.Changed):
            selector.set_options((f'{index + 1}. {label}', index) for index, (label, _) in enumerate(inputs))
            selector.value = previous if isinstance(previous, int) and previous < len(inputs) else (0 if inputs else Select.NULL)
        if inputs:
            label, preview = inputs[selector.value]
            self._show_image('input', preview, label)
        else:
            self._show_image('input', None, 'No recorded input')
        self._layout()

    def set_output(self, preview: Image.Image | None, label: str) -> None:
        self._show_image('output', preview, label)

    def clear(self) -> None:
        self.set_inputs([])
        self._show_image('output', None, 'No output selected')

    def _show_image(self, role: str, preview: Image.Image | None, label: str) -> None:
        image = self.query_one(f'#result-{role}-image', HalfcellImage)
        image.image = preview
        image.display = preview is not None
        empty = self.query_one(f'#result-{role}-empty', Label)
        empty.update(label)
        empty.display = preview is None
