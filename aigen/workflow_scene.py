from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from collections import Counter, OrderedDict, defaultdict
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
    port_definitions_compatible,
)
from aigen.workflow_layout import (
    AUTO_LAYOUT_DETOUR_MARGIN_Y,
    AUTO_LAYOUT_DETOUR_ROW_GAP,
    NODE_WIDTH,
    connected_node_components,
    node_height,
)


CANVAS_MARGIN_X = 4
CANVAS_MARGIN_Y = 2

_BASE_STYLE = Style(color="#7f748b", bgcolor="#15111d")
_WIRE_STYLE = Style(color="#756486", bgcolor="#15111d")
_WIRE_ACTIVE_STYLE = Style(color="#d6a8ff", bgcolor="#15111d", bold=True)
_WIRE_TARGET_STYLE = Style(color="#86e1a8", bgcolor="#15111d", bold=True)
_NODE_STYLE = Style(color="#b9adc8", bgcolor="#211a2d")
_NODE_SELECTED_STYLE = Style(
    color="#f2e7ff",
    bgcolor="#352944",
    bold=True,
)
_PORT_STYLE = Style(color="#7cc4ff", bgcolor="#211a2d", bold=True)
_PORT_SELECTED_STYLE = Style(color="#bde5ff", bgcolor="#352944", bold=True)
_PORT_COMPATIBLE_STYLE = Style(
    color="#86e1a8",
    bgcolor="#211a2d",
    bold=True,
)
_PORT_TARGET_STYLE = Style(
    color="#ffd166",
    bgcolor="#352944",
    bold=True,
)
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


@dataclass(frozen=True, slots=True)
class PortHit:
    node_id: str
    port: str
    x: int
    y: int


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class WireGeometry:
    points: tuple[Offset, ...]

    @property
    def source_x(self) -> int:
        return self.points[0].x

    @property
    def source_y(self) -> int:
        return self.points[0].y

    @property
    def target_x(self) -> int:
        return self.points[-1].x

    @property
    def target_y(self) -> int:
        return self.points[-1].y

    @property
    def top(self) -> int:
        return min(point.y for point in self.points)

    @property
    def bottom(self) -> int:
        return max(point.y for point in self.points)

    @property
    def right(self) -> int:
        return max(point.x for point in self.points)

    def contains(self, x: int, y: int) -> bool:
        if len(self.points) == 1:
            point = self.points[0]
            return point.x == x and point.y == y
        return any(
            (
                start.y == end.y == y
                and min(start.x, end.x) <= x <= max(start.x, end.x)
            )
            or (
                start.x == end.x == x
                and min(start.y, end.y) <= y <= max(start.y, end.y)
            )
            for start, end in zip(self.points, self.points[1:])
        )


@dataclass(frozen=True, slots=True)
class _WireEndpoints:
    connection: WorkflowConnection
    source: PortHit
    target: PortHit
    component: int


@dataclass(frozen=True, slots=True)
class _ConnectionPreview:
    wire: WireGeometry
    fixed_port: PortHit
    moving_source: bool
    compatible_ports: frozenset[PortHit]
    target_port: PortHit | None
    suppressed_connection_id: str | None


@dataclass(frozen=True, slots=True)
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
        self._component_by_node: dict[str, int] = {}
        self._component_column_counts: list[Counter[int]] = []
        self._component_columns: list[list[int]] = []
        self._component_bottom_edges: list[Counter[int]] = []
        self._node_rows: dict[int, list[str]] = {}
        self._wire_rows: dict[int, list[str]] = {}
        self._input_hits: dict[tuple[int, int], list[PortHit]] = {}
        self._output_hits: dict[tuple[int, int], list[PortHit]] = {}
        self._connected_inputs: set[tuple[str, str]] = set()
        self._right_edges: Counter[int] = Counter()
        self._wire_right_edges: Counter[int] = Counter()
        self._bottom_edges: Counter[int] = Counter()
        self._wire_bottom_edges: Counter[int] = Counter()
        self._connection_preview: _ConnectionPreview | None = None
        self._row_cache: OrderedDict[int, _CachedRow] = OrderedDict()
        self._build(document)

    @property
    def node_geometries(self) -> Mapping[str, NodeGeometry]:
        return self._node_geometries

    @property
    def content_size(self) -> Size:
        preview = (
            self._connection_preview.wire
            if self._connection_preview is not None
            else None
        )
        right = max(
            max(self._right_edges, default=0),
            max(self._wire_right_edges, default=0),
            preview.right if preview is not None else 0,
        )
        bottom = max(
            max(self._bottom_edges, default=0),
            max(self._wire_bottom_edges, default=0),
            preview.bottom if preview is not None else 0,
        )
        return Size(
            right + CANVAS_MARGIN_X,
            bottom + CANVAS_MARGIN_Y,
        )

    def set_document(self, document: WorkflowGraph) -> None:
        self._connection_preview = None
        if not self._can_reconcile_nodes(document):
            self.document = document
            self._build(document)
            return

        previous_nodes = self._nodes_by_id
        previous_connections = self._connections_by_id
        nodes_by_id = {node.id: node for node in document.nodes}
        connections_by_id = {
            connection.id: connection
            for connection in document.connections
        }
        moved_node_ids = tuple(
            node.id
            for node in document.nodes
            if (
                self._node_geometries[node.id].x != node.layout.x
                or self._node_geometries[node.id].y != node.layout.y
            )
        )
        changed_connection_ids = {
            connection_id
            for connection_id in (
                previous_connections.keys()
                & connections_by_id.keys()
            )
            if previous_connections[connection_id]
            != connections_by_id[connection_id]
        }
        connection_geometry_changed = (
            previous_connections.keys() != connections_by_id.keys()
            or bool(changed_connection_ids)
        )
        if moved_node_ids and connection_geometry_changed:
            self.document = document
            self._build(document)
            return

        self.document = document
        self._nodes_by_id = nodes_by_id
        if connection_geometry_changed:
            self._reconcile_connections(connections_by_id)
        else:
            self._connections_by_id = connections_by_id
            self._reconcile_node_layouts(moved_node_ids)

        for node in document.nodes:
            previous = previous_nodes[node.id]
            if previous.title != node.title:
                self._row_cache.pop(
                    self._node_geometries[node.id].y,
                    None,
                )

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

    def set_runtime_status(self, node_id: str, status: str) -> bool:
        if self.runtime_statuses.get(node_id) == status:
            return False
        self.runtime_statuses[node_id] = status
        geometry = self._node_geometries.get(node_id)
        if geometry is not None:
            self._row_cache.pop(geometry.y, None)
        return True

    def node_geometry(self, node_id: str) -> NodeGeometry:
        return self._node_geometries[node_id]

    def move_node(
        self,
        node_id: str,
        x: int,
        y: int,
    ) -> tuple[int, int] | None:
        """Move retained live geometry and reroute only incident wires."""
        old_geometry = self._node_geometries[node_id]
        if old_geometry.x == x and old_geometry.y == y:
            return None

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
        top = min(old_geometry.y, geometry.y)
        bottom = max(old_geometry.bottom, geometry.bottom)
        self._invalidate_rows(old_geometry.y, old_geometry.bottom)
        self._remove_node_spatial_indexes(old_geometry)
        self._replace_component_geometry_indexes(
            old_geometry,
            geometry,
        )
        self._node_geometries[node_id] = geometry
        self._add_node_spatial_indexes(geometry)
        self._invalidate_rows(geometry.y, geometry.bottom)

        for wire_id, wire in self._route_wire_geometries(
            self._incident_wires[node_id]
        ).items():
            changed_span = self._replace_wire_geometry(wire_id, wire)
            if changed_span is not None:
                top = min(top, changed_span[0])
                bottom = max(bottom, changed_span[1])
        return top, bottom

    def finish_node_move(self) -> tuple[int, int] | None:
        """Settle globally dependent lanes after the live drag ends."""
        top: int | None = None
        bottom: int | None = None
        for wire_id, wire in self._route_wire_geometries(
            tuple(self._connections_by_id)
        ).items():
            changed_span = self._replace_wire_geometry(wire_id, wire)
            if changed_span is None:
                continue
            top = (
                changed_span[0]
                if top is None
                else min(top, changed_span[0])
            )
            bottom = (
                changed_span[1]
                if bottom is None
                else max(bottom, changed_span[1])
            )
        return None if top is None else (top, bottom)

    def begin_connection_preview(
        self,
        fixed_port: PortHit,
        pointer: Offset,
        *,
        moving_source: bool,
        suppressed_connection_id: str | None = None,
    ) -> tuple[int, int]:
        compatible_ports = frozenset(
            port
            for geometry in self._node_geometries.values()
            for port in (
                geometry.outputs if moving_source else geometry.inputs
            )
            if port.node_id != fixed_port.node_id
            and self._ports_compatible_for_drag(
                port if moving_source else fixed_port,
                fixed_port if moving_source else port,
            )
        )
        self._connection_preview = _ConnectionPreview(
            wire=self._preview_wire_geometry(
                fixed_port,
                pointer,
                moving_source=moving_source,
            ),
            fixed_port=fixed_port,
            moving_source=moving_source,
            compatible_ports=compatible_ports,
            target_port=None,
            suppressed_connection_id=suppressed_connection_id,
        )
        span = self._invalidate_connection_preview(
            self._connection_preview,
            include_compatible_ports=True,
        )
        return span

    def update_connection_preview(
        self,
        pointer: Offset,
    ) -> tuple[PortHit | None, tuple[int, int] | None]:
        preview = self._connection_preview
        assert preview is not None
        candidate = (
            self.output_near(pointer)
            if preview.moving_source
            else self.input_near(pointer)
        )
        target_port = (
            candidate
            if candidate is not None
            and candidate in preview.compatible_ports
            else None
        )
        endpoint = (
            Offset(target_port.x, target_port.y)
            if target_port is not None
            else pointer
        )
        updated = _ConnectionPreview(
            wire=self._preview_wire_geometry(
                preview.fixed_port,
                endpoint,
                moving_source=preview.moving_source,
            ),
            fixed_port=preview.fixed_port,
            moving_source=preview.moving_source,
            compatible_ports=preview.compatible_ports,
            target_port=target_port,
            suppressed_connection_id=preview.suppressed_connection_id,
        )
        if updated == preview:
            return target_port, None
        span = self._invalidate_connection_preview_change(preview, updated)
        self._connection_preview = updated
        return target_port, span

    def clear_connection_preview(self) -> tuple[int, int] | None:
        preview = self._connection_preview
        if preview is None:
            return None
        span = self._invalidate_connection_preview(
            preview,
            include_compatible_ports=True,
        )
        self._connection_preview = None
        return span

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
        suppressed_connection_id = (
            self._connection_preview.suppressed_connection_id
            if self._connection_preview is not None
            else None
        )
        if wire_ids is not None:
            for wire_id in wire_ids:
                if (
                    wire_id != self.selected_connection_id
                    and wire_id != suppressed_connection_id
                ):
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
                and self.selected_connection_id
                != suppressed_connection_id
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
            self._connection_preview is not None
            and self._connection_preview.wire.top
            <= y
            <= self._connection_preview.wire.bottom
        ):
            self._draw_wire_line(
                self._connection_preview.wire,
                y,
                scroll_x,
                wire_bits,
                styles,
                (
                    _WIRE_TARGET_STYLE
                    if self._connection_preview.target_port is not None
                    else _WIRE_ACTIVE_STYLE
                ),
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

    def output_near(self, point: Offset) -> PortHit | None:
        return self._near_port(self._output_hits, point)

    def input_near(self, point: Offset) -> PortHit | None:
        return self._near_port(self._input_hits, point)

    def wire_at(self, point: Offset) -> str | None:
        wire_ids = self._wire_rows.get(point.y)
        if wire_ids is None:
            return None
        if (
            self.selected_connection_id is not None
            and self.selected_connection_id in wire_ids
        ):
            selected = self._wires_by_id[self.selected_connection_id]
            if selected.contains(point.x, point.y):
                return self.selected_connection_id
        for wire_id in reversed(wire_ids):
            wire = self._wires_by_id[wire_id]
            if wire.contains(point.x, point.y):
                return wire_id
        return None

    def ports_compatible(self, source: PortHit, target: PortHit) -> bool:
        return self._ports_compatible_for_drag(source, target)

    def reconnectable_input_connection(
        self,
        port: PortHit,
        preferred_connection_id: str | None,
    ) -> str | None:
        incoming = tuple(
            connection_id
            for connection_id in self._incident_wires[port.node_id]
            if self._connections_by_id[connection_id].target.node_id
            == port.node_id
            and self._connections_by_id[connection_id].target.port
            == port.port
        )
        if (
            preferred_connection_id is not None
            and preferred_connection_id in incoming
        ):
            return preferred_connection_id
        return incoming[0] if len(incoming) == 1 else None

    def reconnectable_output_connection(
        self,
        port: PortHit,
        preferred_connection_id: str | None,
    ) -> str | None:
        if preferred_connection_id is None:
            return None
        connection = self._connections_by_id.get(preferred_connection_id)
        if (
            connection is None
            or connection.source.node_id != port.node_id
            or connection.source.port != port.port
        ):
            return None
        return preferred_connection_id

    def connection_source_port(self, connection_id: str) -> PortHit:
        connection = self._connections_by_id[connection_id]
        return next(
            port
            for port in self._node_geometries[
                connection.source.node_id
            ].outputs
            if port.port == connection.source.port
        )

    def _ports_compatible_for_drag(
        self,
        source: PortHit,
        target: PortHit,
    ) -> bool:
        source_definition = self._definitions_by_node_id[source.node_id].output(
            source.port
        )
        target_definition = self._definitions_by_node_id[target.node_id].input(
            target.port
        )
        assert source_definition is not None
        assert target_definition is not None
        return port_definitions_compatible(
            source_definition,
            target_definition,
        )

    def _build(self, document: WorkflowGraph) -> None:
        self._nodes_by_id = {node.id: node for node in document.nodes}
        self._node_rank = {
            node.id: rank
            for rank, node in enumerate(document.nodes)
        }
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
        self._wire_rank = {
            connection.id: rank
            for rank, connection in enumerate(document.connections)
        }

        for node in document.nodes:
            self._node_geometries[node.id] = self._node_geometry(node)

        self._rebuild_connection_indexes()
        self._rebuild_wire_geometries()
        self._rebuild_spatial_indexes()

    def _can_reconcile_nodes(self, document: WorkflowGraph) -> bool:
        if tuple(self._nodes_by_id) != tuple(
            node.id for node in document.nodes
        ):
            return False
        return not any(
            self._nodes_by_id[node.id].kind != node.kind
            for node in document.nodes
        )

    def _reconcile_connections(
        self,
        updated: dict[str, WorkflowConnection],
    ) -> None:
        self._connections_by_id = updated
        self._wire_rank = {
            connection.id: rank
            for rank, connection in enumerate(self.document.connections)
        }
        self._rebuild_connection_indexes()
        self._rebuild_wire_geometries()
        self._rebuild_spatial_indexes()

    def _reconcile_node_layouts(
        self,
        moved_node_ids: tuple[str, ...],
    ) -> None:
        if not moved_node_ids:
            return
        for node_id in moved_node_ids:
            layout = self._nodes_by_id[node_id].layout
            self.move_node(
                node_id,
                layout.x,
                layout.y,
            )
        self.finish_node_move()

    def _rebuild_spatial_indexes(self) -> None:
        self._node_rows.clear()
        self._wire_rows.clear()
        self._input_hits.clear()
        self._output_hits.clear()
        self._right_edges.clear()
        self._wire_right_edges.clear()
        self._bottom_edges.clear()
        self._wire_bottom_edges.clear()
        self._row_cache.clear()

        for node in self.document.nodes:
            geometry = self._node_geometries[node.id]
            self._add_node_spatial_indexes(geometry)
        for connection in self.document.connections:
            wire = self._wires_by_id[connection.id]
            self._add_wire_spatial_indexes(connection.id, wire)

    def _rebuild_connection_indexes(self) -> None:
        incident_wires: dict[str, list[str]] = {
            node.id: []
            for node in self.document.nodes
        }
        connected_inputs: set[tuple[str, str]] = set()
        for connection in self.document.connections:
            incident_wires[connection.source.node_id].append(connection.id)
            incident_wires[connection.target.node_id].append(connection.id)
            connected_inputs.add(
                (connection.target.node_id, connection.target.port)
            )
        self._incident_wires = {
            node_id: tuple(wire_ids)
            for node_id, wire_ids in incident_wires.items()
        }
        self._connected_inputs = connected_inputs
        components = connected_node_components(self.document)
        self._component_by_node = {
            node_id: component
            for component, node_ids in enumerate(components)
            for node_id in node_ids
        }
        self._component_column_counts = []
        self._component_columns = []
        self._component_bottom_edges = []
        for node_ids in components:
            columns = Counter(
                self._node_geometries[node_id].x
                for node_id in node_ids
            )
            self._component_column_counts.append(columns)
            self._component_columns.append(sorted(columns))
            self._component_bottom_edges.append(
                Counter(
                    self._node_geometries[node_id].bottom
                    for node_id in node_ids
                )
            )

    def _replace_component_geometry_indexes(
        self,
        previous: NodeGeometry,
        updated: NodeGeometry,
    ) -> None:
        component = self._component_by_node[previous.node_id]
        columns = self._component_column_counts[component]
        _decrement_counter(columns, previous.x)
        if previous.x not in columns:
            ordered_columns = self._component_columns[component]
            ordered_columns.pop(
                bisect_left(ordered_columns, previous.x)
            )
        if updated.x not in columns:
            insort(self._component_columns[component], updated.x)
        columns[updated.x] += 1

        bottom_edges = self._component_bottom_edges[component]
        _decrement_counter(bottom_edges, previous.bottom)
        bottom_edges[updated.bottom] += 1

    def _node_geometry(self, node: WorkflowNode) -> NodeGeometry:
        definition = self._definitions_by_node_id[node.id]
        return NodeGeometry(
            node_id=node.id,
            x=node.layout.x,
            y=node.layout.y,
            width=NODE_WIDTH,
            height=node_height(node.kind),
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

    def _rebuild_wire_geometries(self) -> None:
        self._wires_by_id = self._route_wire_geometries(
            tuple(self._connections_by_id)
        )

    def _route_wire_geometries(
        self,
        wire_ids: tuple[str, ...],
    ) -> dict[str, WireGeometry]:
        direct_groups: dict[
            tuple[int, int, int],
            list[_WireEndpoints],
        ] = defaultdict(list)
        detours: dict[int, list[_WireEndpoints]] = defaultdict(list)

        for wire_id in wire_ids:
            connection = self._connections_by_id[wire_id]
            source_geometry = self._node_geometries[
                connection.source.node_id
            ]
            target_geometry = self._node_geometries[
                connection.target.node_id
            ]
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
            component = self._component_by_node[
                connection.source.node_id
            ]
            endpoints = _WireEndpoints(
                connection=connection,
                source=source,
                target=target,
                component=component,
            )
            columns = self._component_columns[component]
            skips_column = (
                bisect_right(columns, source_geometry.x)
                < bisect_left(columns, target_geometry.x)
            )
            if (
                source_geometry.right < target_geometry.x
                and not skips_column
            ):
                direct_groups[
                    (
                        component,
                        source_geometry.x,
                        target_geometry.x,
                    )
                ].append(endpoints)
            else:
                detours[component].append(endpoints)

        routed: dict[str, WireGeometry] = {}
        for group in direct_groups.values():
            group.sort(key=self._wire_order)
            lane_left = group[0].source.x + 2
            lane_right = group[0].target.x - 2
            lane_count = lane_right - lane_left + 1
            if lane_count < len(group):
                detours[group[0].component].extend(group)
                continue
            lane_x = lane_left + (lane_count - len(group)) // 2
            for offset, endpoints in enumerate(group):
                source = endpoints.source
                target = endpoints.target
                lane = lane_x + offset
                routed[endpoints.connection.id] = _orthogonal_wire(
                    Offset(source.x, source.y),
                    Offset(lane, source.y),
                    Offset(lane, target.y),
                    Offset(target.x, target.y),
                )

        for component, edges in detours.items():
            edges.sort(key=self._wire_order)
            first_lane_y = (
                max(self._component_bottom_edges[component])
                + AUTO_LAYOUT_DETOUR_MARGIN_Y
            )
            for lane, endpoints in enumerate(edges):
                source = endpoints.source
                target = endpoints.target
                route_y = (
                    first_lane_y
                    + lane * AUTO_LAYOUT_DETOUR_ROW_GAP
                )
                routed[endpoints.connection.id] = _orthogonal_wire(
                    Offset(source.x, source.y),
                    Offset(source.x + 2, source.y),
                    Offset(source.x + 2, route_y),
                    Offset(max(0, target.x - 2), route_y),
                    Offset(max(0, target.x - 2), target.y),
                    Offset(target.x, target.y),
                )

        return routed

    def _wire_order(
        self,
        endpoints: _WireEndpoints,
    ) -> tuple[int, int, int, int, int]:
        return (
            endpoints.source.y + endpoints.target.y,
            endpoints.source.y,
            endpoints.target.y,
            endpoints.connection.order,
            self._wire_rank[endpoints.connection.id],
        )

    def _add_node_spatial_indexes(self, geometry: NodeGeometry) -> None:
        self._right_edges[geometry.right] += 1
        self._bottom_edges[geometry.bottom] += 1
        for y in range(geometry.y, geometry.bottom + 1):
            self._insert_node_id(
                self._node_rows.setdefault(y, []),
                geometry.node_id,
            )
        for port in geometry.inputs:
            self._add_port_hit(self._input_hits, port)
        for port in geometry.outputs:
            self._add_port_hit(self._output_hits, port)

    def _remove_node_spatial_indexes(
        self,
        geometry: NodeGeometry,
    ) -> None:
        _decrement_counter(self._right_edges, geometry.right)
        _decrement_counter(self._bottom_edges, geometry.bottom)
        for y in range(geometry.y, geometry.bottom + 1):
            node_ids = self._node_rows[y]
            node_ids.remove(geometry.node_id)
            if not node_ids:
                del self._node_rows[y]
        for port in geometry.inputs:
            self._remove_port_hit(self._input_hits, port)
        for port in geometry.outputs:
            self._remove_port_hit(self._output_hits, port)

    def _add_wire_spatial_indexes(
        self,
        wire_id: str,
        wire: WireGeometry,
    ) -> None:
        self._wire_right_edges[wire.right] += 1
        self._wire_bottom_edges[wire.bottom] += 1
        for y in range(wire.top, wire.bottom + 1):
            self._insert_wire_id(
                self._wire_rows.setdefault(y, []),
                wire_id,
            )

    def _remove_wire_spatial_indexes(
        self,
        wire_id: str,
        wire: WireGeometry,
    ) -> None:
        _decrement_counter(self._wire_right_edges, wire.right)
        _decrement_counter(self._wire_bottom_edges, wire.bottom)
        for y in range(wire.top, wire.bottom + 1):
            wire_ids = self._wire_rows[y]
            wire_ids.remove(wire_id)
            if not wire_ids:
                del self._wire_rows[y]

    def _replace_wire_geometry(
        self,
        wire_id: str,
        wire: WireGeometry,
    ) -> tuple[int, int] | None:
        previous = self._wires_by_id[wire_id]
        if previous == wire:
            return None
        self._invalidate_wire_rows(previous)
        self._remove_wire_spatial_indexes(wire_id, previous)
        self._wires_by_id[wire_id] = wire
        self._add_wire_spatial_indexes(wire_id, wire)
        self._invalidate_wire_rows(wire)
        return min(previous.top, wire.top), max(
            previous.bottom,
            wire.bottom,
        )

    def _add_port_hit(
        self,
        index: dict[tuple[int, int], list[PortHit]],
        port: PortHit,
    ) -> None:
        hits = index.setdefault((port.x, port.y), [])
        insert_at = bisect_left(
            hits,
            self._node_rank[port.node_id],
            key=lambda hit: self._node_rank[hit.node_id],
        )
        hits.insert(insert_at, port)

    def _insert_node_id(
        self,
        node_ids: list[str],
        node_id: str,
    ) -> None:
        insert_at = bisect_left(
            node_ids,
            self._node_rank[node_id],
            key=self._node_rank.__getitem__,
        )
        node_ids.insert(insert_at, node_id)

    def _insert_wire_id(
        self,
        wire_ids: list[str],
        wire_id: str,
    ) -> None:
        insert_at = bisect_left(
            wire_ids,
            self._wire_rank[wire_id],
            key=self._wire_rank.__getitem__,
        )
        wire_ids.insert(insert_at, wire_id)

    @staticmethod
    def _remove_port_hit(
        index: dict[tuple[int, int], list[PortHit]],
        port: PortHit,
    ) -> None:
        key = (port.x, port.y)
        hits = index[key]
        hits.remove(port)
        if not hits:
            del index[key]

    def _port_at(self, hits: list[PortHit] | None) -> PortHit | None:
        if hits is None:
            return None
        if self.selected_node_id is not None:
            for hit in hits:
                if hit.node_id == self.selected_node_id:
                    return hit
        return hits[-1]

    def _near_port(
        self,
        index: dict[tuple[int, int], list[PortHit]],
        point: Offset,
    ) -> PortHit | None:
        for x in (point.x, point.x - 1, point.x + 1):
            hit = self._port_at(index.get((x, point.y)))
            if hit is not None:
                return hit
        return None

    @staticmethod
    def _preview_wire_geometry(
        fixed_port: PortHit,
        endpoint: Offset,
        *,
        moving_source: bool,
    ) -> WireGeometry:
        if moving_source:
            return _dogleg_wire(
                Offset(endpoint.x, endpoint.y),
                Offset(fixed_port.x, fixed_port.y),
            )
        return _dogleg_wire(
            Offset(fixed_port.x, fixed_port.y),
            Offset(endpoint.x, endpoint.y),
        )

    def _invalidate_connection_preview(
        self,
        preview: _ConnectionPreview,
        *,
        include_compatible_ports: bool,
    ) -> tuple[int, int]:
        top = preview.wire.top
        bottom = preview.wire.bottom
        self._invalidate_rows(top, bottom)
        if preview.target_port is not None:
            self._row_cache.pop(preview.target_port.y, None)
            top = min(top, preview.target_port.y)
            bottom = max(bottom, preview.target_port.y)
        if include_compatible_ports:
            for port in preview.compatible_ports:
                self._row_cache.pop(port.y, None)
                top = min(top, port.y)
                bottom = max(bottom, port.y)
        if preview.suppressed_connection_id is not None:
            wire = self._wires_by_id[preview.suppressed_connection_id]
            self._invalidate_rows(wire.top, wire.bottom)
            top = min(top, wire.top)
            bottom = max(bottom, wire.bottom)
        return top, bottom

    def _invalidate_connection_preview_change(
        self,
        previous: _ConnectionPreview,
        updated: _ConnectionPreview,
    ) -> tuple[int, int]:
        top = min(previous.wire.top, updated.wire.top)
        bottom = max(previous.wire.bottom, updated.wire.bottom)
        self._invalidate_rows(previous.wire.top, previous.wire.bottom)
        self._invalidate_rows(updated.wire.top, updated.wire.bottom)
        if previous.target_port is not None:
            self._row_cache.pop(previous.target_port.y, None)
            top = min(top, previous.target_port.y)
            bottom = max(bottom, previous.target_port.y)
        if updated.target_port is not None:
            self._row_cache.pop(updated.target_port.y, None)
            top = min(top, updated.target_port.y)
            bottom = max(bottom, updated.target_port.y)
        return top, bottom

    def _invalidate_rows(self, top: int, bottom: int) -> None:
        for y in range(top, bottom + 1):
            self._row_cache.pop(y, None)

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
        for start, end in zip(wire.points, wire.points[1:]):
            if start.y == end.y:
                if y == start.y:
                    _add_horizontal(
                        bits,
                        styles,
                        scroll_x,
                        start.x,
                        end.x,
                        style,
                    )
                continue
            if min(start.y, end.y) <= y <= max(start.y, end.y):
                _add_vertical(
                    bits,
                    styles,
                    scroll_x,
                    start,
                    end,
                    y,
                    style,
                )

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
        default_port_style = (
            _PORT_SELECTED_STYLE if selected else _PORT_STYLE
        )
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
            port_hit = geometry.inputs[row]
            connected = (node.id, port.name) in self._connected_inputs
            input_text = _port_text(port)
            if 0 <= local_x < len(characters):
                characters[local_x] = "●" if connected else "○"
                styles[local_x] = self._preview_port_style(
                    port_hit,
                    default_port_style,
                )
            _write(
                characters,
                styles,
                local_x + 2,
                input_text[: (geometry.width - 4) // 2],
                node_style,
            )
        if row < len(definition.outputs):
            port = definition.outputs[row]
            port_hit = geometry.outputs[row]
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
                styles[local_right] = self._preview_port_style(
                    port_hit,
                    default_port_style,
                )

    def _preview_port_style(
        self,
        port: PortHit,
        default: Style,
    ) -> Style:
        preview = self._connection_preview
        if preview is None:
            return default
        if port == preview.target_port:
            return _PORT_TARGET_STYLE
        if port in preview.compatible_ports:
            return _PORT_COMPATIBLE_STYLE
        return default


def _dogleg_wire(source: Offset, target: Offset) -> WireGeometry:
    elbow_x = (
        (source.x + target.x) // 2
        if target.x > source.x
        else max(source.x, target.x) + 4
    )
    return _orthogonal_wire(
        source,
        Offset(elbow_x, source.y),
        Offset(elbow_x, target.y),
        target,
    )


def _orthogonal_wire(*points: Offset) -> WireGeometry:
    compact: list[Offset] = []
    for point in points:
        if compact and compact[-1] == point:
            continue
        compact.append(point)
        while len(compact) >= 3:
            first, middle, last = compact[-3:]
            if not (
                first.x == middle.x == last.x
                or first.y == middle.y == last.y
            ):
                break
            compact.pop(-2)
    return WireGeometry(points=tuple(compact))


def _decrement_counter(counter: Counter[int], value: int) -> None:
    counter[value] -= 1
    if counter[value] == 0:
        del counter[value]


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


def _add_vertical(
    bits: list[int],
    styles: list[Style],
    scroll_x: int,
    start: Offset,
    end: Offset,
    y: int,
    style: Style,
) -> None:
    local_x = start.x - scroll_x
    if not 0 <= local_x < len(bits):
        return
    vertical = _NORTH | _SOUTH
    if y == start.y:
        vertical &= ~(
            _NORTH if end.y >= start.y else _SOUTH
        )
    if y == end.y:
        vertical &= ~(
            _SOUTH if end.y >= start.y else _NORTH
        )
    bits[local_x] |= vertical
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
