from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.binding import Binding
from textual.geometry import Offset, Region, Size
from textual.message import Message
from textual.scroll_view import ScrollView
from textual.strip import Strip

from aigen.workflow_graph import (
    NodePortRef,
    PortDefinition,
    WorkflowConnection,
    WorkflowGraph,
    node_definition,
)


NODE_WIDTH = 34
CANVAS_MARGIN_X = 4
CANVAS_MARGIN_Y = 2
SCROLL_COLUMNS = 6
SCROLL_ROWS = 3

_BASE_STYLE = Style(color="#7f748b", bgcolor="#15111d")
_WIRE_STYLE = Style(color="#756486", bgcolor="#15111d")
_WIRE_ACTIVE_STYLE = Style(color="#d6a8ff", bgcolor="#15111d", bold=True)
_NODE_STYLE = Style(color="#b9adc8", bgcolor="#211a2d")
_NODE_SELECTED_STYLE = Style(
    color="#f2e7ff",
    bgcolor="#352944",
    bold=True,
)
_PORT_STYLE = Style(color="#7cc4ff", bgcolor="#211a2d", bold=True)
_PORT_SELECTED_STYLE = Style(color="#bde5ff", bgcolor="#352944", bold=True)
_STATUS_STYLES = {
    "queued": Style(color="#c8b9d8", bgcolor="#211a2d"),
    "running": Style(color="#ffd166", bgcolor="#211a2d", bold=True),
    "completed": Style(color="#86e1a8", bgcolor="#211a2d", bold=True),
    "reused": Style(color="#86e1a8", bgcolor="#211a2d"),
    "failed": Style(color="#ff7f8f", bgcolor="#211a2d", bold=True),
    "skipped": Style(color="#a99bb7", bgcolor="#211a2d"),
}

_NORTH = 1
_EAST = 2
_SOUTH = 4
_WEST = 8
_WIRE_CHARACTERS = {
    _NORTH: "│",
    _EAST: "─",
    _SOUTH: "│",
    _WEST: "─",
    _NORTH | _SOUTH: "│",
    _EAST | _WEST: "─",
    _EAST | _SOUTH: "┌",
    _WEST | _SOUTH: "┐",
    _EAST | _NORTH: "└",
    _WEST | _NORTH: "┘",
    _NORTH | _EAST | _SOUTH: "├",
    _NORTH | _WEST | _SOUTH: "┤",
    _EAST | _SOUTH | _WEST: "┬",
    _NORTH | _EAST | _WEST: "┴",
    _NORTH | _EAST | _SOUTH | _WEST: "┼",
}
_ARTIFACT_LABELS = {
    "image": "img",
    "reference-pack": "pack",
    "lora": "lora",
    "video": "vid",
    "image-sequence": "frames",
}


@dataclass(frozen=True)
class PortHit:
    node_id: str
    port: str
    x: int
    y: int


@dataclass(frozen=True)
class NodeGeometry:
    node_id: str
    x: int
    y: int
    width: int
    height: int
    inputs: tuple[PortHit, ...]
    outputs: tuple[PortHit, ...]

    @property
    def right(self) -> int:
        return self.x + self.width - 1

    @property
    def bottom(self) -> int:
        return self.y + self.height - 1

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x <= self.right and self.y <= y <= self.bottom


@dataclass(frozen=True)
class WireGeometry:
    connection: WorkflowConnection | None
    source: PortHit
    target_x: int
    target_y: int


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
        ) -> None:
            super().__init__()
            self.node_id = node_id
            self.connection_id = connection_id

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
        self.document = document
        self.selected_node_id: str | None = None
        self.selected_connection_id: str | None = None
        self.runtime_statuses: dict[str, str] = {}
        self._nodes_by_id = {
            node.id: node
            for node in document.nodes
        }
        self._connected_inputs: set[tuple[str, str]] = set()
        self._node_geometries: dict[str, NodeGeometry] = {}
        self._render_geometries: tuple[NodeGeometry, ...] = ()
        self._wire_geometries: tuple[WireGeometry, ...] = ()
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
        self._rebuild_geometry()

    def set_editable(self, editable: bool) -> None:
        if editable == self._editable:
            return
        self._editable = editable
        if not editable:
            had_edit_gesture = (
                self._drag_node_id is not None
                or self._connection_source is not None
            )
            self._drag_node_id = None
            self._drag_layout = None
            self._connection_source = None
            self._connection_pointer = None
            if had_edit_gesture:
                self.release_mouse()
            self._rebuild_geometry()
            self.refresh()

    def set_document(self, document: WorkflowGraph) -> None:
        self.document = document
        self._nodes_by_id = {
            node.id: node
            for node in document.nodes
        }
        self._drag_node_id = None
        self._drag_layout = None
        if (
            self.selected_node_id is not None
            and not any(
                node.id == self.selected_node_id
                for node in document.nodes
            )
        ):
            self.selected_node_id = None
        if (
            self.selected_connection_id is not None
            and not any(
                connection.id == self.selected_connection_id
                for connection in document.connections
            )
        ):
            self.selected_connection_id = None
        self._rebuild_geometry()
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
        self.selected_node_id = node_id
        self.selected_connection_id = connection_id
        self._rebuild_draw_order()
        self.refresh()

    def set_runtime_statuses(self, statuses: Mapping[str, str]) -> None:
        self.runtime_statuses = dict(statuses)
        self.refresh()

    def node_geometry(self, node_id: str) -> NodeGeometry:
        return self._node_geometries[node_id]

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
        width = self.virtual_size.width
        characters = [" "] * width
        styles = [_BASE_STYLE] * width
        wire_bits: dict[int, int] = {}

        for wire in self._wire_geometries:
            active = (
                wire.connection is not None
                and (
                    self.selected_connection_id == wire.connection.id
                    or (
                        self.selected_node_id is not None
                        and (
                            self.selected_node_id
                            == wire.connection.source.node_id
                            or self.selected_node_id
                            == wire.connection.target.node_id
                        )
                    )
                )
            )
            self._draw_wire_line(
                wire,
                y,
                wire_bits,
                styles,
                _WIRE_ACTIVE_STYLE if active else _WIRE_STYLE,
            )

        if (
            self._connection_source is not None
            and self._connection_pointer is not None
        ):
            self._draw_wire_line(
                WireGeometry(
                    connection=None,
                    source=self._connection_source,
                    target_x=self._connection_pointer.x,
                    target_y=self._connection_pointer.y,
                ),
                y,
                wire_bits,
                styles,
                _WIRE_ACTIVE_STYLE,
            )

        for x, bits in wire_bits.items():
            if 0 <= x < width:
                characters[x] = _WIRE_CHARACTERS.get(bits, "•")

        for geometry in self._render_geometries:
            if geometry.y <= y <= geometry.bottom:
                self._draw_node_line(geometry, y, characters, styles)

        segments: list[Segment] = []
        span_start = 0
        span_style = styles[0] if styles else _BASE_STYLE
        for index in range(1, width):
            if styles[index] != span_style:
                segments.append(
                    Segment("".join(characters[span_start:index]), span_style)
                )
                span_start = index
                span_style = styles[index]
        if width:
            segments.append(Segment("".join(characters[span_start:]), span_style))
        return Strip(segments, width)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button not in {1, 2}:
            return
        self.focus()
        point = self._virtual_pointer(event)
        if event.button == 1:
            if self._editable and (output := self._output_at(point)):
                self._select(output.node_id, None)
                self._connection_source = output
                self._connection_pointer = point
                self.capture_mouse()
                self.refresh()
                event.stop()
                return

            if geometry := self._node_at(point):
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

            if wire := self._wire_at(point):
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
            self._rebuild_geometry()
            self.refresh()
            event.stop()
            return

        if self._connection_source is not None:
            self._connection_pointer = self._virtual_pointer(event)
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
            self.post_message(
                self.NodeMoved(
                    self._drag_node_id,
                    self._drag_layout.x,
                    self._drag_layout.y,
                )
            )
            self._drag_node_id = None
            self.release_mouse()
            event.stop()
            return

        if self._connection_source is not None:
            source = self._connection_source
            target = self._input_at(self._virtual_pointer(event))
            self._connection_source = None
            self._connection_pointer = None
            self.release_mouse()
            self.refresh()
            if target is not None and self._ports_compatible(source, target):
                self.post_message(
                    self.ConnectionRequested(
                        NodePortRef(node_id=source.node_id, port=source.port),
                        NodePortRef(node_id=target.node_id, port=target.port),
                    )
                )
            event.stop()
            return

        if self._panning:
            self._panning = False
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
        self.selected_node_id = node_id
        self.selected_connection_id = connection_id
        self._rebuild_draw_order()
        self.post_message(self.SelectionChanged(node_id, connection_id))
        self.refresh()

    def _select_in_direction(self, delta_x: int, delta_y: int) -> None:
        if not self._node_geometries:
            return
        current = (
            self._node_geometries.get(self.selected_node_id)
            if self.selected_node_id is not None
            else None
        )
        if current is None:
            selected = min(
                self._node_geometries.values(),
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
            for geometry in self._node_geometries.values():
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
            selected = self._node_geometries[min(candidates)[-1]]
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
        geometry = self._node_geometries[self.selected_node_id]
        self.post_message(
            self.NodeMoved(
                geometry.node_id,
                max(0, geometry.x + delta_x),
                max(0, geometry.y + delta_y),
            )
        )

    def _rebuild_geometry(self) -> None:
        geometries: dict[str, NodeGeometry] = {}
        for node in self.document.nodes:
            definition = node_definition(node.kind)
            port_rows = max(len(definition.inputs), len(definition.outputs), 1)
            if node.id == self._drag_node_id and self._drag_layout is not None:
                x, y = self._drag_layout
            else:
                x, y = node.layout.x, node.layout.y
            geometries[node.id] = NodeGeometry(
                node_id=node.id,
                x=x,
                y=y,
                width=NODE_WIDTH,
                height=port_rows + 2,
                inputs=tuple(
                    PortHit(node.id, port.name, x, y + index + 1)
                    for index, port in enumerate(definition.inputs)
                ),
                outputs=tuple(
                    PortHit(
                        node.id,
                        port.name,
                        x + NODE_WIDTH - 1,
                        y + index + 1,
                    )
                    for index, port in enumerate(definition.outputs)
                ),
            )
        self._node_geometries = geometries
        self._connected_inputs = {
            (connection.target.node_id, connection.target.port)
            for connection in self.document.connections
        }
        self._rebuild_draw_order()
        wires: list[WireGeometry] = []
        for connection in self.document.connections:
            source_geometry = geometries[connection.source.node_id]
            target_geometry = geometries[connection.target.node_id]
            source = next(
                port
                for port in source_geometry.outputs
                if port.port == connection.source.port
            )
            target = next(
                port
                for port in target_geometry.inputs
                if port.port == connection.target.port
            )
            wires.append(
                WireGeometry(
                    connection=connection,
                    source=source,
                    target_x=target.x,
                    target_y=target.y,
                )
            )
        self._wire_geometries = tuple(wires)
        right = max(
            (geometry.right for geometry in geometries.values()),
            default=0,
        )
        bottom = max(
            (geometry.bottom for geometry in geometries.values()),
            default=0,
        )
        self.virtual_size = Size(
            max(self.size.width, right + CANVAS_MARGIN_X),
            max(self.size.height, bottom + CANVAS_MARGIN_Y),
        )

    def _draw_wire_line(
        self,
        wire: WireGeometry,
        y: int,
        bits: dict[int, int],
        styles: list[Style],
        style: Style,
    ) -> None:
        source_x = wire.source.x
        source_y = wire.source.y
        target_x = wire.target_x
        target_y = wire.target_y
        if target_x > source_x:
            elbow_x = (source_x + target_x) // 2
        else:
            elbow_x = max(source_x, target_x) + 4

        if y == source_y:
            self._add_horizontal(
                bits,
                styles,
                source_x,
                elbow_x,
                style,
            )
        if y == target_y:
            self._add_horizontal(
                bits,
                styles,
                elbow_x,
                target_x,
                style,
            )
        if min(source_y, target_y) <= y <= max(source_y, target_y):
            vertical = _NORTH | _SOUTH
            if y == source_y:
                vertical &= ~(
                    _NORTH if target_y >= source_y else _SOUTH
                )
            if y == target_y:
                vertical &= ~(
                    _SOUTH if target_y >= source_y else _NORTH
                )
            bits[elbow_x] = bits.get(elbow_x, 0) | vertical
            if 0 <= elbow_x < len(styles):
                styles[elbow_x] = style

    @staticmethod
    def _add_horizontal(
        bits: dict[int, int],
        styles: list[Style],
        start_x: int,
        end_x: int,
        style: Style,
    ) -> None:
        left = min(start_x, end_x)
        right = max(start_x, end_x)
        direction = _EAST | _WEST
        for x in range(left, right + 1):
            cell_bits = direction
            if x == start_x:
                cell_bits &= ~(_WEST if end_x >= start_x else _EAST)
            if x == end_x:
                cell_bits &= ~(_EAST if end_x >= start_x else _WEST)
            bits[x] = bits.get(x, 0) | cell_bits
            if 0 <= x < len(styles):
                styles[x] = style

    def _draw_node_line(
        self,
        geometry: NodeGeometry,
        y: int,
        characters: list[str],
        styles: list[Style],
    ) -> None:
        node = self._nodes_by_id[geometry.node_id]
        definition = node_definition(node.kind)
        selected = geometry.node_id == self.selected_node_id
        node_style = _NODE_SELECTED_STYLE if selected else _NODE_STYLE
        port_style = _PORT_SELECTED_STYLE if selected else _PORT_STYLE
        relative_y = y - geometry.y
        if relative_y == 0:
            status = self.runtime_statuses.get(node.id, "")
            suffix = f" [{status}]" if status else ""
            available = geometry.width - len(suffix) - 3
            title = f" {node.title[:available]}{suffix} "
            text = "┌" + title + "─" * (geometry.width - len(title) - 2) + "┐"
            status_style = _STATUS_STYLES.get(status, node_style)
            self._write(
                characters,
                styles,
                geometry.x,
                text,
                status_style if status else node_style,
            )
            return
        if relative_y == geometry.height - 1:
            self._write(
                characters,
                styles,
                geometry.x,
                "└" + "─" * (geometry.width - 2) + "┘",
                node_style,
            )
            return

        row = relative_y - 1
        self._write(
            characters,
            styles,
            geometry.x,
            "│" + " " * (geometry.width - 2) + "│",
            node_style,
        )
        if row < len(definition.inputs):
            port = definition.inputs[row]
            connected = (node.id, port.name) in self._connected_inputs
            input_text = self._port_text(port)
            characters[geometry.x] = "●" if connected else "○"
            styles[geometry.x] = port_style
            self._write(
                characters,
                styles,
                geometry.x + 2,
                input_text[: (geometry.width - 4) // 2],
                node_style,
            )
        if row < len(definition.outputs):
            port = definition.outputs[row]
            output_text = self._port_text(port)
            available = (geometry.width - 4) // 2
            output_text = output_text[:available]
            output_x = geometry.right - len(output_text) - 1
            self._write(
                characters,
                styles,
                output_x,
                output_text,
                node_style,
            )
            characters[geometry.right] = "●"
            styles[geometry.right] = port_style

    @staticmethod
    def _port_text(port: PortDefinition) -> str:
        artifact_types = "/".join(
            _ARTIFACT_LABELS[artifact_type.value]
            for artifact_type in port.artifact_types
        )
        port_name = (
            port.name
            if len(port.name) <= 6
            else f"{port.name[:3]}…"
        )
        return f"{port_name}:{artifact_types}"

    def _rebuild_draw_order(self) -> None:
        self._render_geometries = tuple(
            geometry
            for geometry in self._node_geometries.values()
            if geometry.node_id != self.selected_node_id
        )
        if (
            self.selected_node_id is not None
            and self.selected_node_id in self._node_geometries
        ):
            self._render_geometries += (
                self._node_geometries[self.selected_node_id],
            )

    @staticmethod
    def _write(
        characters: list[str],
        styles: list[Style],
        x: int,
        text: str,
        style: Style,
    ) -> None:
        left = max(0, x)
        right = min(len(characters), x + len(text))
        if left >= right:
            return
        source_start = left - x
        for destination, character in enumerate(
            text[source_start : source_start + right - left],
            start=left,
        ):
            characters[destination] = character
            styles[destination] = style

    def _virtual_pointer(self, event: events.MouseEvent) -> Offset:
        offset = event.get_content_offset_capture(self)
        return Offset(
            offset.x + self.scroll_offset.x,
            offset.y + self.scroll_offset.y,
        )

    def _node_at(self, point: Offset) -> NodeGeometry | None:
        geometries = tuple(self._node_geometries.values())
        if self.selected_node_id is not None:
            geometries = tuple(
                geometry
                for geometry in geometries
                if geometry.node_id != self.selected_node_id
            ) + (self._node_geometries[self.selected_node_id],)
        return next(
            (
                geometry
                for geometry in reversed(geometries)
                if geometry.contains(point.x, point.y)
            ),
            None,
        )

    def _output_at(self, point: Offset) -> PortHit | None:
        return next(
            (
                port
                for geometry in self._node_geometries.values()
                for port in geometry.outputs
                if (port.x, port.y) == (point.x, point.y)
            ),
            None,
        )

    def _input_at(self, point: Offset) -> PortHit | None:
        return next(
            (
                port
                for geometry in self._node_geometries.values()
                for port in geometry.inputs
                if (port.x, port.y) == (point.x, point.y)
            ),
            None,
        )

    def _wire_at(self, point: Offset) -> WireGeometry | None:
        wires = tuple(
            wire
            for wire in self._wire_geometries
            if wire.connection is not None
            and wire.connection.id != self.selected_connection_id
        )
        if self.selected_connection_id is not None:
            wires += tuple(
                wire
                for wire in self._wire_geometries
                if wire.connection is not None
                and wire.connection.id == self.selected_connection_id
            )
        return next(
            (
                wire
                for wire in reversed(wires)
                if self._wire_contains(wire, point)
            ),
            None,
        )

    @staticmethod
    def _wire_contains(wire: WireGeometry, point: Offset) -> bool:
        source_x = wire.source.x
        source_y = wire.source.y
        target_x = wire.target_x
        target_y = wire.target_y
        elbow_x = (
            (source_x + target_x) // 2
            if target_x > source_x
            else max(source_x, target_x) + 4
        )
        return (
            point.y == source_y
            and min(source_x, elbow_x) <= point.x <= max(source_x, elbow_x)
        ) or (
            point.x == elbow_x
            and min(source_y, target_y) <= point.y <= max(source_y, target_y)
        ) or (
            point.y == target_y
            and min(elbow_x, target_x) <= point.x <= max(elbow_x, target_x)
        )

    def _ports_compatible(self, source: PortHit, target: PortHit) -> bool:
        source_node = self.document.node(source.node_id)
        target_node = self.document.node(target.node_id)
        source_definition = node_definition(source_node.kind).output(source.port)
        target_definition = node_definition(target_node.kind).input(target.port)
        assert source_definition is not None
        assert target_definition is not None
        return bool(
            set(source_definition.artifact_types).intersection(
                target_definition.artifact_types
            )
        )
