from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Mapping

from rich.segment import Segment
from rich.style import Style
from textual.geometry import Offset, Size
from textual.strip import Strip

from aigen.workflow_graph import (
    NodeDefinition,
    PortDefinition,
    WorkflowConnection,
    WorkflowGraph,
    WorkflowNode,
    node_definition,
)


NODE_WIDTH = 34
CANVAS_MARGIN_X = 4
CANVAS_MARGIN_Y = 2

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
_ROW_CACHE_SIZE = 1024


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

    @property
    def elbow_x(self) -> int:
        if self.target_x > self.source.x:
            return (self.source.x + self.target_x) // 2
        return max(self.source.x, self.target_x) + 4

    @property
    def top(self) -> int:
        return min(self.source.y, self.target_y)

    @property
    def bottom(self) -> int:
        return max(self.source.y, self.target_y)

    @property
    def right(self) -> int:
        return max(self.source.x, self.target_x, self.elbow_x)

    def contains(self, x: int, y: int) -> bool:
        elbow_x = self.elbow_x
        return (
            y == self.source.y
            and min(self.source.x, elbow_x) <= x <= max(self.source.x, elbow_x)
        ) or (
            x == elbow_x
            and self.top <= y <= self.bottom
        ) or (
            y == self.target_y
            and min(elbow_x, self.target_x) <= x <= max(elbow_x, self.target_x)
        )

@dataclass(frozen=True)
class _CachedRow:
    scroll_x: int
    width: int
    strip: Strip


class WorkflowScene:
    """Retained geometry, hit indexes and ASCII rows for a workflow graph."""

    def __init__(self, document: WorkflowGraph) -> None:
        self.document = document
        self.selected_node_id: str | None = None
        self.selected_connection_id: str | None = None
        self.runtime_statuses: dict[str, str] = {}
        self._nodes_by_id: dict[str, WorkflowNode] = {}
        self._definitions_by_node_id: dict[str, NodeDefinition] = {}
        self._node_geometries: dict[str, NodeGeometry] = {}
        self._wires_by_id: dict[str, WireGeometry] = {}
        self._connections_by_id: dict[str, WorkflowConnection] = {}
        self._node_rank: dict[str, int] = {}
        self._wire_rank: dict[str, int] = {}
        self._incident_wires: dict[str, tuple[str, ...]] = {}
        self._node_rows: dict[int, list[str]] = {}
        self._wire_rows: dict[int, list[str]] = {}
        self._input_hits: dict[tuple[int, int], list[PortHit]] = {}
        self._output_hits: dict[tuple[int, int], list[PortHit]] = {}
        self._connected_inputs: set[tuple[str, str]] = set()
        self._right_edges: Counter[int] = Counter()
        self._wire_right_edges: Counter[int] = Counter()
        self._bottom_edges: Counter[int] = Counter()
        self._preview_wire: WireGeometry | None = None
        self._row_cache: OrderedDict[int, _CachedRow] = OrderedDict()
        self._build(document)

    @property
    def node_geometries(self) -> Mapping[str, NodeGeometry]:
        return self._node_geometries

    @property
    def content_size(self) -> Size:
        right = max(
            max(self._right_edges, default=0),
            max(self._wire_right_edges, default=0),
        )
        bottom = max(self._bottom_edges, default=0)
        return Size(
            right + CANVAS_MARGIN_X,
            bottom + CANVAS_MARGIN_Y,
        )

    def set_document(self, document: WorkflowGraph) -> None:
        self.document = document
        self._preview_wire = None
        self._build(document)

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
        self._invalidate_selection(
            self.selected_node_id,
            self.selected_connection_id,
        )
        self.selected_node_id = node_id
        self.selected_connection_id = connection_id
        self._invalidate_selection(node_id, connection_id)

    def set_runtime_statuses(self, statuses: Mapping[str, str]) -> None:
        updated = dict(statuses)
        changed_node_ids = {
            node_id
            for node_id in self.runtime_statuses.keys() | updated.keys()
            if self.runtime_statuses.get(node_id) != updated.get(node_id)
        }
        self.runtime_statuses = updated
        for node_id in changed_node_ids:
            geometry = self._node_geometries.get(node_id)
            if geometry is not None:
                self._row_cache.pop(geometry.y, None)

    def node_geometry(self, node_id: str) -> NodeGeometry:
        return self._node_geometries[node_id]

    def move_node(self, node_id: str, x: int, y: int) -> None:
        old_geometry = self._node_geometries[node_id]
        if old_geometry.x == x and old_geometry.y == y:
            return

        incident_wire_ids = self._incident_wires[node_id]
        for wire_id in incident_wire_ids:
            old_wire = self._wires_by_id[wire_id]
            self._remove_wire_indexes(old_wire)
            _remove_count(self._wire_right_edges, old_wire.right)
        self._remove_node_indexes(old_geometry)
        _remove_count(self._right_edges, old_geometry.right)
        _remove_count(self._bottom_edges, old_geometry.bottom)

        geometry = NodeGeometry(
            node_id=node_id,
            x=x,
            y=y,
            width=old_geometry.width,
            height=old_geometry.height,
            inputs=tuple(
                PortHit(node_id, port.port, x, y + index + 1)
                for index, port in enumerate(old_geometry.inputs)
            ),
            outputs=tuple(
                PortHit(
                    node_id,
                    port.port,
                    x + old_geometry.width - 1,
                    y + index + 1,
                )
                for index, port in enumerate(old_geometry.outputs)
            ),
        )
        self._node_geometries[node_id] = geometry
        self._right_edges[geometry.right] += 1
        self._bottom_edges[geometry.bottom] += 1
        self._add_node_indexes(geometry)

        for wire_id in incident_wire_ids:
            connection = self._connections_by_id[wire_id]
            wire = self._wire_geometry(connection)
            self._wires_by_id[wire_id] = wire
            self._wire_right_edges[wire.right] += 1
            self._add_wire_indexes(wire)

    def set_preview_connection(
        self,
        source: PortHit,
        target: Offset,
    ) -> None:
        preview = WireGeometry(
            connection=None,
            source=source,
            target_x=target.x,
            target_y=target.y,
        )
        if preview == self._preview_wire:
            return
        self._invalidate_wire_rows(self._preview_wire)
        self._preview_wire = preview
        self._invalidate_wire_rows(preview)

    def clear_preview_connection(self) -> None:
        self._invalidate_wire_rows(self._preview_wire)
        self._preview_wire = None

    def render_line(self, y: int, scroll_x: int, width: int) -> Strip:
        cached = self._row_cache.get(y)
        if (
            cached is not None
            and cached.scroll_x == scroll_x
            and cached.width == width
        ):
            self._row_cache.move_to_end(y)
            return cached.strip
        if width <= 0:
            return Strip.blank(0, _BASE_STYLE)

        characters = [" "] * width
        styles = [_BASE_STYLE] * width
        wire_bits = [0] * width

        wire_ids = self._wire_rows.get(y)
        if wire_ids is not None:
            for wire_id in wire_ids:
                if wire_id != self.selected_connection_id:
                    self._draw_wire_line(
                        self._wires_by_id[wire_id],
                        y,
                        scroll_x,
                        wire_bits,
                        styles,
                        self._wire_style(wire_id),
                    )
            if (
                self.selected_connection_id is not None
                and self.selected_connection_id in wire_ids
            ):
                self._draw_wire_line(
                    self._wires_by_id[self.selected_connection_id],
                    y,
                    scroll_x,
                    wire_bits,
                    styles,
                    _WIRE_ACTIVE_STYLE,
                )

        if (
            self._preview_wire is not None
            and self._preview_wire.top <= y <= self._preview_wire.bottom
        ):
            self._draw_wire_line(
                self._preview_wire,
                y,
                scroll_x,
                wire_bits,
                styles,
                _WIRE_ACTIVE_STYLE,
            )

        for x, bits in enumerate(wire_bits):
            if bits:
                characters[x] = _WIRE_CHARACTERS.get(bits, "•")

        node_ids = self._node_rows.get(y)
        if node_ids is not None:
            for node_id in node_ids:
                if node_id != self.selected_node_id:
                    self._draw_node_line(
                        self._node_geometries[node_id],
                        y,
                        scroll_x,
                        characters,
                        styles,
                    )
            if (
                self.selected_node_id is not None
                and self.selected_node_id in node_ids
            ):
                self._draw_node_line(
                    self._node_geometries[self.selected_node_id],
                    y,
                    scroll_x,
                    characters,
                    styles,
                )

        segments: list[Segment] = []
        span_start = 0
        span_style = styles[0]
        for index in range(1, width):
            if styles[index] != span_style:
                segments.append(
                    Segment("".join(characters[span_start:index]), span_style)
                )
                span_start = index
                span_style = styles[index]
        segments.append(Segment("".join(characters[span_start:]), span_style))
        strip = Strip(segments, width)
        self._row_cache[y] = _CachedRow(scroll_x, width, strip)
        self._row_cache.move_to_end(y)
        if len(self._row_cache) > _ROW_CACHE_SIZE:
            self._row_cache.popitem(last=False)
        return strip

    def node_at(self, point: Offset) -> NodeGeometry | None:
        node_ids = self._node_rows.get(point.y)
        if node_ids is None:
            return None
        if (
            self.selected_node_id is not None
            and self.selected_node_id in node_ids
        ):
            selected = self._node_geometries[self.selected_node_id]
            if selected.contains(point.x, point.y):
                return selected
        for node_id in reversed(node_ids):
            geometry = self._node_geometries[node_id]
            if geometry.contains(point.x, point.y):
                return geometry
        return None

    def output_at(self, point: Offset) -> PortHit | None:
        return self._port_at(self._output_hits.get((point.x, point.y)))

    def input_at(self, point: Offset) -> PortHit | None:
        return self._port_at(self._input_hits.get((point.x, point.y)))

    def wire_at(self, point: Offset) -> WireGeometry | None:
        wire_ids = self._wire_rows.get(point.y)
        if wire_ids is None:
            return None
        if (
            self.selected_connection_id is not None
            and self.selected_connection_id in wire_ids
        ):
            selected = self._wires_by_id[self.selected_connection_id]
            if selected.contains(point.x, point.y):
                return selected
        for wire_id in reversed(wire_ids):
            wire = self._wires_by_id[wire_id]
            if wire.contains(point.x, point.y):
                return wire
        return None

    def ports_compatible(self, source: PortHit, target: PortHit) -> bool:
        source_definition = self._definitions_by_node_id[source.node_id].output(
            source.port
        )
        target_definition = self._definitions_by_node_id[target.node_id].input(
            target.port
        )
        assert source_definition is not None
        assert target_definition is not None
        return any(
            artifact_type in target_definition.artifact_types
            for artifact_type in source_definition.artifact_types
        )

    def _build(self, document: WorkflowGraph) -> None:
        self._nodes_by_id = {node.id: node for node in document.nodes}
        self._definitions_by_node_id = {
            node.id: node_definition(node.kind)
            for node in document.nodes
        }
        self._node_geometries.clear()
        self._wires_by_id.clear()
        self._connections_by_id = {
            connection.id: connection
            for connection in document.connections
        }
        self._node_rank = {
            node.id: rank
            for rank, node in enumerate(document.nodes)
        }
        self._wire_rank = {
            connection.id: rank
            for rank, connection in enumerate(document.connections)
        }
        incident_wires: dict[str, list[str]] = {
            node.id: []
            for node in document.nodes
        }
        for connection in document.connections:
            incident_wires[connection.source.node_id].append(connection.id)
            incident_wires[connection.target.node_id].append(connection.id)
        self._incident_wires = {
            node_id: tuple(wire_ids)
            for node_id, wire_ids in incident_wires.items()
        }
        self._node_rows.clear()
        self._wire_rows.clear()
        self._input_hits.clear()
        self._output_hits.clear()
        self._connected_inputs = {
            (connection.target.node_id, connection.target.port)
            for connection in document.connections
        }
        self._right_edges.clear()
        self._wire_right_edges.clear()
        self._bottom_edges.clear()
        self._row_cache.clear()

        for node in document.nodes:
            definition = self._definitions_by_node_id[node.id]
            port_rows = max(len(definition.inputs), len(definition.outputs), 1)
            geometry = NodeGeometry(
                node_id=node.id,
                x=node.layout.x,
                y=node.layout.y,
                width=NODE_WIDTH,
                height=port_rows + 2,
                inputs=tuple(
                    PortHit(
                        node.id,
                        port.name,
                        node.layout.x,
                        node.layout.y + index + 1,
                    )
                    for index, port in enumerate(definition.inputs)
                ),
                outputs=tuple(
                    PortHit(
                        node.id,
                        port.name,
                        node.layout.x + NODE_WIDTH - 1,
                        node.layout.y + index + 1,
                    )
                    for index, port in enumerate(definition.outputs)
                ),
            )
            self._node_geometries[node.id] = geometry
            self._right_edges[geometry.right] += 1
            self._bottom_edges[geometry.bottom] += 1
            self._add_node_indexes(geometry)

        for connection in document.connections:
            wire = self._wire_geometry(connection)
            self._wires_by_id[connection.id] = wire
            self._wire_right_edges[wire.right] += 1
            self._add_wire_indexes(wire)

    def _wire_geometry(self, connection: WorkflowConnection) -> WireGeometry:
        source_geometry = self._node_geometries[connection.source.node_id]
        target_geometry = self._node_geometries[connection.target.node_id]
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
        return WireGeometry(
            connection=connection,
            source=source,
            target_x=target.x,
            target_y=target.y,
        )

    def _add_node_indexes(self, geometry: NodeGeometry) -> None:
        for y in range(geometry.y, geometry.bottom + 1):
            self._insert_node_id(self._node_rows.setdefault(y, []), geometry.node_id)
            self._row_cache.pop(y, None)
        for port in geometry.inputs:
            self._insert_port_hit(
                self._input_hits.setdefault((port.x, port.y), []),
                port,
            )
        for port in geometry.outputs:
            self._insert_port_hit(
                self._output_hits.setdefault((port.x, port.y), []),
                port,
            )

    def _remove_node_indexes(self, geometry: NodeGeometry) -> None:
        for y in range(geometry.y, geometry.bottom + 1):
            node_ids = self._node_rows[y]
            node_ids.remove(geometry.node_id)
            if not node_ids:
                del self._node_rows[y]
            self._row_cache.pop(y, None)
        for port in geometry.inputs:
            self._remove_port_hit(self._input_hits, port)
        for port in geometry.outputs:
            self._remove_port_hit(self._output_hits, port)

    def _add_wire_indexes(self, wire: WireGeometry) -> None:
        assert wire.connection is not None
        wire_id = wire.connection.id
        for y in range(wire.top, wire.bottom + 1):
            self._insert_wire_id(self._wire_rows.setdefault(y, []), wire_id)
            self._row_cache.pop(y, None)

    def _remove_wire_indexes(self, wire: WireGeometry) -> None:
        assert wire.connection is not None
        wire_id = wire.connection.id
        for y in range(wire.top, wire.bottom + 1):
            wire_ids = self._wire_rows[y]
            wire_ids.remove(wire_id)
            if not wire_ids:
                del self._wire_rows[y]
            self._row_cache.pop(y, None)

    def _insert_node_id(self, node_ids: list[str], node_id: str) -> None:
        rank = self._node_rank[node_id]
        for index, other_id in enumerate(node_ids):
            if rank < self._node_rank[other_id]:
                node_ids.insert(index, node_id)
                return
        node_ids.append(node_id)

    def _insert_wire_id(self, wire_ids: list[str], wire_id: str) -> None:
        rank = self._wire_rank[wire_id]
        for index, other_id in enumerate(wire_ids):
            if rank < self._wire_rank[other_id]:
                wire_ids.insert(index, wire_id)
                return
        wire_ids.append(wire_id)

    def _insert_port_hit(self, hits: list[PortHit], hit: PortHit) -> None:
        rank = self._node_rank[hit.node_id]
        for index, other in enumerate(hits):
            if rank < self._node_rank[other.node_id]:
                hits.insert(index, hit)
                return
        hits.append(hit)

    @staticmethod
    def _remove_port_hit(
        index: dict[tuple[int, int], list[PortHit]],
        hit: PortHit,
    ) -> None:
        coordinate = (hit.x, hit.y)
        hits = index[coordinate]
        hits.remove(hit)
        if not hits:
            del index[coordinate]

    def _port_at(self, hits: list[PortHit] | None) -> PortHit | None:
        if hits is None:
            return None
        if self.selected_node_id is not None:
            for hit in hits:
                if hit.node_id == self.selected_node_id:
                    return hit
        return hits[-1]

    def _invalidate_selection(
        self,
        node_id: str | None,
        connection_id: str | None,
    ) -> None:
        if node_id is not None and node_id in self._node_geometries:
            geometry = self._node_geometries[node_id]
            for y in range(geometry.y, geometry.bottom + 1):
                self._row_cache.pop(y, None)
            for wire_id in self._incident_wires[node_id]:
                self._invalidate_wire_rows(self._wires_by_id[wire_id])
        if connection_id is not None and connection_id in self._wires_by_id:
            self._invalidate_wire_rows(self._wires_by_id[connection_id])

    def _invalidate_wire_rows(self, wire: WireGeometry | None) -> None:
        if wire is None:
            return
        for y in range(wire.top, wire.bottom + 1):
            self._row_cache.pop(y, None)

    def _wire_style(self, wire_id: str) -> Style:
        if self.selected_node_id is None:
            return _WIRE_STYLE
        connection = self._connections_by_id[wire_id]
        if (
            self.selected_node_id == connection.source.node_id
            or self.selected_node_id == connection.target.node_id
        ):
            return _WIRE_ACTIVE_STYLE
        return _WIRE_STYLE

    @staticmethod
    def _draw_wire_line(
        wire: WireGeometry,
        y: int,
        scroll_x: int,
        bits: list[int],
        styles: list[Style],
        style: Style,
    ) -> None:
        elbow_x = wire.elbow_x
        if y == wire.source.y:
            _add_horizontal(
                bits,
                styles,
                scroll_x,
                wire.source.x,
                elbow_x,
                style,
            )
        if y == wire.target_y:
            _add_horizontal(
                bits,
                styles,
                scroll_x,
                elbow_x,
                wire.target_x,
                style,
            )
        if wire.top <= y <= wire.bottom:
            local_x = elbow_x - scroll_x
            if 0 <= local_x < len(bits):
                vertical = _NORTH | _SOUTH
                if y == wire.source.y:
                    vertical &= ~(
                        _NORTH if wire.target_y >= wire.source.y else _SOUTH
                    )
                if y == wire.target_y:
                    vertical &= ~(
                        _SOUTH if wire.target_y >= wire.source.y else _NORTH
                    )
                bits[local_x] |= vertical
                styles[local_x] = style

    def _draw_node_line(
        self,
        geometry: NodeGeometry,
        y: int,
        scroll_x: int,
        characters: list[str],
        styles: list[Style],
    ) -> None:
        node = self._nodes_by_id[geometry.node_id]
        definition = self._definitions_by_node_id[node.id]
        selected = geometry.node_id == self.selected_node_id
        node_style = _NODE_SELECTED_STYLE if selected else _NODE_STYLE
        port_style = _PORT_SELECTED_STYLE if selected else _PORT_STYLE
        local_x = geometry.x - scroll_x
        relative_y = y - geometry.y
        if relative_y == 0:
            status = self.runtime_statuses.get(node.id, "")
            suffix = f" [{status}]" if status else ""
            available = geometry.width - len(suffix) - 3
            title = f" {node.title[:available]}{suffix} "
            text = "┌" + title + "─" * (geometry.width - len(title) - 2) + "┐"
            status_style = _STATUS_STYLES.get(status, node_style)
            _write(
                characters,
                styles,
                local_x,
                text,
                status_style if status else node_style,
            )
            return
        if relative_y == geometry.height - 1:
            _write(
                characters,
                styles,
                local_x,
                "└" + "─" * (geometry.width - 2) + "┘",
                node_style,
            )
            return

        row = relative_y - 1
        _write(
            characters,
            styles,
            local_x,
            "│" + " " * (geometry.width - 2) + "│",
            node_style,
        )
        if row < len(definition.inputs):
            port = definition.inputs[row]
            connected = (node.id, port.name) in self._connected_inputs
            input_text = _port_text(port)
            if 0 <= local_x < len(characters):
                characters[local_x] = "●" if connected else "○"
                styles[local_x] = port_style
            _write(
                characters,
                styles,
                local_x + 2,
                input_text[: (geometry.width - 4) // 2],
                node_style,
            )
        if row < len(definition.outputs):
            port = definition.outputs[row]
            output_text = _port_text(port)
            available = (geometry.width - 4) // 2
            output_text = output_text[:available]
            output_x = geometry.right - len(output_text) - 1 - scroll_x
            _write(
                characters,
                styles,
                output_x,
                output_text,
                node_style,
            )
            local_right = geometry.right - scroll_x
            if 0 <= local_right < len(characters):
                characters[local_right] = "●"
                styles[local_right] = port_style


def _remove_count(counts: Counter[int], value: int) -> None:
    counts[value] -= 1
    if counts[value] == 0:
        del counts[value]


def _add_horizontal(
    bits: list[int],
    styles: list[Style],
    scroll_x: int,
    start_x: int,
    end_x: int,
    style: Style,
) -> None:
    left = max(min(start_x, end_x), scroll_x)
    right = min(max(start_x, end_x), scroll_x + len(bits) - 1)
    if left > right:
        return
    direction = _EAST | _WEST
    for x in range(left, right + 1):
        cell_bits = direction
        if x == start_x:
            cell_bits &= ~(_WEST if end_x >= start_x else _EAST)
        if x == end_x:
            cell_bits &= ~(_EAST if end_x >= start_x else _WEST)
        local_x = x - scroll_x
        bits[local_x] |= cell_bits
        styles[local_x] = style


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


def _port_text(port: PortDefinition) -> str:
    artifact_types = "/".join(
        _ARTIFACT_LABELS[artifact_type.value]
        for artifact_type in port.artifact_types
    )
    port_name = port.name if len(port.name) <= 6 else f"{port.name[:3]}…"
    return f"{port_name}:{artifact_types}"
