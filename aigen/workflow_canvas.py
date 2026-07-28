from __future__ import annotations

from typing import Mapping

from textual import events
from textual.binding import Binding
from textual.geometry import Offset, Region, Size
from textual.message import Message
from textual.scroll_view import ScrollView
from textual.strip import Strip

from aigen.workflow_graph import (
    NodePortRef,
    WorkflowGraph,
)
from aigen.workflow_scene import (
    NodeGeometry,
    PortHit,
    WorkflowScene,
)


SCROLL_COLUMNS = 6
SCROLL_ROWS = 3


class WorkflowCanvas(ScrollView, can_focus=True):
    """Retained, scrollable ASCII projection of a workflow document."""

    BINDINGS = [
        Binding("left", "select_left", show=False),
        Binding("right", "select_right", show=False),
        Binding("up", "select_up", show=False),
        Binding("down", "select_down", show=False),
        Binding("shift+left", "move_left", show=False),
        Binding("shift+right", "move_right", show=False),
        Binding("shift+up", "move_up", show=False),
        Binding("shift+down", "move_down", show=False),
    ]

    DEFAULT_CSS = """
    WorkflowCanvas {
        width: 3fr;
        height: 1fr;
        background: #15111d;
        border: solid #5b496d;
        scrollbar-size: 1 1;
    }

    WorkflowCanvas:focus {
        border: solid #b681e6;
    }
    """

    class SelectionChanged(Message):
        def __init__(
            self,
            node_id: str | None,
            connection_id: str | None,
            previous_node_id: str | None,
            previous_connection_id: str | None,
        ) -> None:
            super().__init__()
            self.node_id = node_id
            self.connection_id = connection_id
            self.previous_node_id = previous_node_id
            self.previous_connection_id = previous_connection_id

    class NodeMoved(Message):
        def __init__(self, node_id: str, x: int, y: int) -> None:
            super().__init__()
            self.node_id = node_id
            self.x = x
            self.y = y

    class ConnectionRequested(Message):
        def __init__(self, source: NodePortRef, target: NodePortRef) -> None:
            super().__init__()
            self.source = source
            self.target = target

    def __init__(
        self,
        document: WorkflowGraph,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._scene = WorkflowScene(document)
        self._drag_node_id: str | None = None
        self._drag_node_origin = Offset(0, 0)
        self._drag_pointer_origin = Offset(0, 0)
        self._drag_layout: Offset | None = None
        self._pan_pointer_origin = Offset(0, 0)
        self._pan_scroll_origin = Offset(0, 0)
        self._panning = False
        self._connection_source: PortHit | None = None
        self._connection_pointer: Offset | None = None
        self._editable = True
        self.virtual_size = self._scene.content_size

    @property
    def document(self) -> WorkflowGraph:
        return self._scene.document

    @property
    def selected_node_id(self) -> str | None:
        return self._scene.selected_node_id

    @property
    def selected_connection_id(self) -> str | None:
        return self._scene.selected_connection_id

    @property
    def runtime_statuses(self) -> Mapping[str, str]:
        return self._scene.runtime_statuses

    def set_editable(self, editable: bool) -> None:
        if editable == self._editable:
            return
        self._editable = editable
        if not editable:
            self._cancel_gesture()

    def on_hide(self, event: events.Hide) -> None:
        self._cancel_gesture()

    def on_unmount(self) -> None:
        self._clear_gesture()

    def on_mouse_release(self, event: events.MouseRelease) -> None:
        self._cancel_gesture()

    def set_document(self, document: WorkflowGraph) -> None:
        self._cancel_gesture()
        selected_node_id = self.selected_node_id
        selected_connection_id = self.selected_connection_id
        if (
            selected_node_id is not None
            and not any(
                node.id == selected_node_id
                for node in document.nodes
            )
        ):
            selected_node_id = None
        if (
            selected_connection_id is not None
            and not any(
                connection.id == selected_connection_id
                for connection in document.connections
            )
        ):
            selected_connection_id = None
        self._scene.set_document(document)
        self._scene.set_selection(selected_node_id, selected_connection_id)
        self._sync_virtual_size()
        self.refresh()

    def set_selected_node(self, node_id: str | None) -> None:
        self.set_selection(node_id, None)

    def set_selection(
        self,
        node_id: str | None,
        connection_id: str | None,
    ) -> None:
        if (
            node_id == self.selected_node_id
            and connection_id == self.selected_connection_id
        ):
            return
        self._scene.set_selection(node_id, connection_id)
        self.refresh()

    def set_runtime_statuses(self, statuses: Mapping[str, str]) -> None:
        self._scene.set_runtime_statuses(statuses)
        self.refresh()

    def set_runtime_status(self, node_id: str, status: str) -> None:
        if not self._scene.set_runtime_status(node_id, status):
            return
        geometry = self._scene.node_geometries.get(node_id)
        if geometry is not None:
            self.refresh_line(geometry.y)

    def node_geometry(self, node_id: str) -> NodeGeometry:
        return self._scene.node_geometry(node_id)

    def action_select_left(self) -> None:
        self._select_in_direction(-1, 0)

    def action_select_right(self) -> None:
        self._select_in_direction(1, 0)

    def action_select_up(self) -> None:
        self._select_in_direction(0, -1)

    def action_select_down(self) -> None:
        self._select_in_direction(0, 1)

    def action_move_left(self) -> None:
        self._move_selected_node(-2, 0)

    def action_move_right(self) -> None:
        self._move_selected_node(2, 0)

    def action_move_up(self) -> None:
        self._move_selected_node(0, -1)

    def action_move_down(self) -> None:
        self._move_selected_node(0, 1)

    def render_line(self, y: int) -> Strip:
        scroll_x, scroll_y = self.scroll_offset
        return self._scene.render_line(
            scroll_y + y,
            scroll_x,
            self.scrollable_content_region.width,
        )

    def on_resize(self, event: events.Resize) -> None:
        self._sync_virtual_size()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button not in {1, 2}:
            return
        self.focus()
        point = self._virtual_pointer(event)
        if event.button == 1:
            if self._editable and (output := self._scene.output_at(point)):
                self._select(output.node_id, None)
                self._connection_source = output
                self._connection_pointer = point
                self._scene.set_preview_connection(output, point)
                self.capture_mouse()
                self.refresh()
                event.stop()
                return

            if geometry := self._scene.node_at(point):
                self._select(geometry.node_id, None)
                if self._editable and point.y == geometry.y:
                    self._drag_node_id = geometry.node_id
                    self._drag_node_origin = Offset(geometry.x, geometry.y)
                    self._drag_pointer_origin = Offset(
                        event.screen_x,
                        event.screen_y,
                    )
                    self._drag_layout = self._drag_node_origin
                    self.capture_mouse()
                event.stop()
                return

            if wire := self._scene.wire_at(point):
                assert wire.connection is not None
                self._select(None, wire.connection.id)
                event.stop()
                return

            self._select(None, None)

        self._panning = True
        self._pan_pointer_origin = Offset(event.screen_x, event.screen_y)
        self._pan_scroll_origin = self.scroll_offset
        self.capture_mouse()
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._drag_node_id is not None:
            delta = Offset(
                event.screen_x - self._drag_pointer_origin.x,
                event.screen_y - self._drag_pointer_origin.y,
            )
            self._drag_layout = Offset(
                max(0, self._drag_node_origin.x + delta.x),
                max(0, self._drag_node_origin.y + delta.y),
            )
            self._scene.move_node(
                self._drag_node_id,
                self._drag_layout.x,
                self._drag_layout.y,
            )
            self._sync_virtual_size()
            self.refresh()
            event.stop()
            return

        if self._connection_source is not None:
            self._connection_pointer = self._virtual_pointer(event)
            self._scene.set_preview_connection(
                self._connection_source,
                self._connection_pointer,
            )
            self.refresh()
            event.stop()
            return

        if self._panning:
            delta_x = event.screen_x - self._pan_pointer_origin.x
            delta_y = event.screen_y - self._pan_pointer_origin.y
            self.scroll_to(
                max(0, self._pan_scroll_origin.x - delta_x),
                max(0, self._pan_scroll_origin.y - delta_y),
                animate=False,
                immediate=True,
            )
            event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if event.button not in {1, 2}:
            return
        if self._drag_node_id is not None:
            assert self._drag_layout is not None
            node_id = self._drag_node_id
            layout = self._drag_layout
            self._clear_gesture()
            self.release_mouse()
            self.post_message(
                self.NodeMoved(
                    node_id,
                    layout.x,
                    layout.y,
                )
            )
            event.stop()
            return

        if self._connection_source is not None:
            source = self._connection_source
            target = self._scene.input_at(self._virtual_pointer(event))
            self._clear_gesture()
            self.release_mouse()
            self.refresh()
            if target is not None and self._scene.ports_compatible(source, target):
                self.post_message(
                    self.ConnectionRequested(
                        NodePortRef(node_id=source.node_id, port=source.port),
                        NodePortRef(node_id=target.node_id, port=target.port),
                    )
                )
            event.stop()
            return

        if self._panning:
            self._clear_gesture()
            self.release_mouse()
            event.stop()

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if event.shift or event.ctrl:
            self.scroll_relative(
                x=SCROLL_COLUMNS,
                animate=False,
                immediate=True,
            )
        else:
            self.scroll_relative(
                y=SCROLL_ROWS,
                animate=False,
                immediate=True,
            )
        event.stop()

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if event.shift or event.ctrl:
            self.scroll_relative(
                x=-SCROLL_COLUMNS,
                animate=False,
                immediate=True,
            )
        else:
            self.scroll_relative(
                y=-SCROLL_ROWS,
                animate=False,
                immediate=True,
            )
        event.stop()

    def _select(
        self,
        node_id: str | None,
        connection_id: str | None,
    ) -> None:
        if (
            node_id == self.selected_node_id
            and connection_id == self.selected_connection_id
        ):
            return
        previous_node_id = self.selected_node_id
        previous_connection_id = self.selected_connection_id
        self._scene.set_selection(node_id, connection_id)
        self.post_message(
            self.SelectionChanged(
                node_id,
                connection_id,
                previous_node_id,
                previous_connection_id,
            )
        )
        self.refresh()

    def _select_in_direction(self, delta_x: int, delta_y: int) -> None:
        geometries = self._scene.node_geometries
        if not geometries:
            return
        current = (
            geometries.get(self.selected_node_id)
            if self.selected_node_id is not None
            else None
        )
        if current is None:
            selected = min(
                geometries.values(),
                key=lambda geometry: (
                    geometry.y,
                    geometry.x,
                    geometry.node_id,
                ),
            )
        else:
            current_x = current.x + current.width // 2
            current_y = current.y + current.height // 2
            candidates = []
            for geometry in geometries.values():
                if geometry.node_id == current.node_id:
                    continue
                candidate_x = geometry.x + geometry.width // 2
                candidate_y = geometry.y + geometry.height // 2
                offset_x = candidate_x - current_x
                offset_y = candidate_y - current_y
                if (
                    delta_x != 0
                    and offset_x * delta_x <= 0
                ) or (
                    delta_y != 0
                    and offset_y * delta_y <= 0
                ):
                    continue
                candidates.append(
                    (
                        abs(offset_x) + abs(offset_y),
                        abs(offset_y if delta_x else offset_x),
                        geometry.y,
                        geometry.x,
                        geometry.node_id,
                    )
                )
            if not candidates:
                return
            selected = geometries[min(candidates)[-1]]
        self._select(selected.node_id, None)
        self.scroll_to_region(
            Region(
                selected.x,
                selected.y,
                selected.width,
                selected.height,
            ),
            animate=False,
            immediate=True,
        )

    def _move_selected_node(self, delta_x: int, delta_y: int) -> None:
        if not self._editable or self.selected_node_id is None:
            return
        geometry = self._scene.node_geometry(self.selected_node_id)
        self.post_message(
            self.NodeMoved(
                geometry.node_id,
                max(0, geometry.x + delta_x),
                max(0, geometry.y + delta_y),
            )
        )

    def _sync_virtual_size(self) -> None:
        content_size = self._scene.content_size
        self.virtual_size = Size(
            max(self.size.width, content_size.width),
            max(self.size.height, content_size.height),
        )

    def _cancel_gesture(self) -> None:
        if (
            self._drag_node_id is None
            and self._connection_source is None
            and not self._panning
        ):
            return
        if self._drag_node_id is not None:
            self._scene.move_node(
                self._drag_node_id,
                self._drag_node_origin.x,
                self._drag_node_origin.y,
            )
        self._clear_gesture()
        self.release_mouse()
        self._sync_virtual_size()
        self.refresh()

    def _clear_gesture(self) -> None:
        self._drag_node_id = None
        self._drag_layout = None
        self._connection_source = None
        self._connection_pointer = None
        self._panning = False
        self._scene.clear_preview_connection()

    def _virtual_pointer(self, event: events.MouseEvent) -> Offset:
        offset = event.get_content_offset_capture(self)
        return Offset(
            offset.x + self.scroll_offset.x,
            offset.y + self.scroll_offset.y,
        )
