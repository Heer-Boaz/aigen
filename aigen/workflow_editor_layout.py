from __future__ import annotations

from textual import events, on
from textual.containers import Container
from textual.message import Message
from textual.widget import Widget

from aigen.workflow_inspector import WorkflowInspector
from aigen.workflow_layout import AUTO_LAYOUT_MIN_COLUMN_GAP, NODE_WIDTH


class InspectorDivider(Widget):
    DEFAULT_CSS = '''
    InspectorDivider { width: 1; height: 1fr; background: #5b496d; }
    InspectorDivider:hover { background: #b791dd; }
    '''

    def __init__(self) -> None:
        super().__init__()
        self.tooltip = 'Drag to resize properties'

    class Moved(Message):
        def __init__(self, x: int) -> None:
            super().__init__()
            self.x = x

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button == 1:
            event.stop()
            self.capture_mouse()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self.app.mouse_captured is self:
            event.stop()
            self.post_message(self.Moved(event.screen_x))

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self.app.mouse_captured is self:
            event.stop()
            self.post_message(self.Moved(event.screen_x))
            self.release_mouse()


class WorkflowEditorBody(Container):
    """Own pane geometry while the mounted editors retain their documents and history."""

    DEFAULT_CSS = '''
    WorkflowEditorBody { layout: horizontal; width: 100%; height: 1fr; }
    WorkflowEditorBody > #workflow-inspector-panel, WorkflowEditorBody > InspectorDivider { display: none; }
    WorkflowEditorBody.inspector-open > #workflow-inspector-panel { display: block; }
    WorkflowEditorBody.inspector-open > InspectorDivider { display: block; }
    WorkflowEditorBody.narrow.inspector-open > WorkflowCanvas,
    WorkflowEditorBody.expanded > WorkflowCanvas { display: none; }
    WorkflowEditorBody.narrow > InspectorDivider,
    WorkflowEditorBody.expanded > InspectorDivider { display: none; }
    WorkflowEditorBody.expanded PropertyTextArea.workflow-property-editor { height: 50vh; max-height: 50vh; }
    '''

    class LayoutChanged(Message):
        pass

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._requested: bool | None = None
        self._selected = False
        self._width: int | None = None

    @property
    def narrow(self) -> bool:
        return self.has_class('narrow')

    @property
    def expanded(self) -> bool:
        return self.has_class('expanded')

    def set_context(self, selected: bool) -> None:
        if selected != self._selected:
            self._selected = selected
            self._layout()

    def show_inspector(self, visible: bool) -> None:
        self._requested = visible
        if not visible:
            self.remove_class('expanded')
        self._layout()

    def expand_inspector(self, expanded: bool) -> None:
        self._requested = True
        self.set_class(expanded, 'expanded')
        self._layout()

    def on_resize(self) -> None:
        self._layout()

    @on(InspectorDivider.Moved)
    def resize_inspector(self, event: InspectorDivider.Moved) -> None:
        self._width = self.content_region.right - event.x
        self._layout()

    def _layout(self) -> None:
        minimum = self.query_one(WorkflowInspector).horizontal_minimum_width
        canvas_minimum = 2 * NODE_WIDTH + AUTO_LAYOUT_MIN_COLUMN_GAP
        width = self.content_size.width
        narrow = width < canvas_minimum + minimum + 1
        self.set_class(narrow, 'narrow')
        visible = self._requested if self._requested is not None else self._selected and not narrow
        self.set_class(visible, 'inspector-open')
        if narrow or self.expanded:
            panel_width = width
        else:
            preferred = self._width if self._width is not None else min(width // 3, width - canvas_minimum - 1)
            panel_width = max(minimum, min(preferred, width - NODE_WIDTH - 1))
        self.query_one('#workflow-inspector-panel').styles.width = panel_width
        self.post_message(self.LayoutChanged())
