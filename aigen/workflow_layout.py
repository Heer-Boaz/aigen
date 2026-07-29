from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from graphlib import TopologicalSorter

from aigen.workflow_graph import (
    NodeKind,
    NodeLayout,
    WorkflowConnection,
    WorkflowGraph,
    WorkflowNode,
    node_definition,
)


NODE_WIDTH = 34
AUTO_LAYOUT_ORIGIN_X = 4
AUTO_LAYOUT_ORIGIN_Y = 2
AUTO_LAYOUT_MIN_COLUMN_GAP = 12
AUTO_LAYOUT_ROW_GAP = 3
AUTO_LAYOUT_COMPONENT_GAP = 8
AUTO_LAYOUT_ROUTING_MARGIN = 6
AUTO_LAYOUT_DETOUR_MARGIN_Y = 3
AUTO_LAYOUT_DETOUR_ROW_GAP = 2
AUTO_LAYOUT_SWEEPS = 8
NEW_NODE_COLLISION_GAP_X = 1
NEW_NODE_OFFSET_X = 4
NEW_NODE_OFFSET_Y = 2


@dataclass(frozen=True, slots=True)
class _LayoutVertex:
    node_id: str | None
    layer: int
    height: int
    rank: int


@dataclass(frozen=True, slots=True)
class _LayerEdge:
    source: int
    target: int
    source_offset: int
    target_offset: int


@dataclass(slots=True)
class _IsotonicBlock:
    start: int
    end: int
    total: float
    count: int

    @property
    def mean(self) -> float:
        return self.total / self.count


def node_height(kind: NodeKind) -> int:
    definition = node_definition(kind)
    return max(
        len(definition.inputs),
        len(definition.outputs),
        1,
    ) + 2


def collision_free_node_layout(
    document: WorkflowGraph,
    kind: NodeKind,
    *,
    x: int,
    y: int,
) -> NodeLayout:
    height = node_height(kind)
    occupied = tuple(
        (
            node.layout.x,
            node.layout.y,
            node.layout.x + NODE_WIDTH,
            node.layout.y + node_height(node.kind),
        )
        for node in document.nodes
    )

    while any(
        x < right + NEW_NODE_COLLISION_GAP_X
        and x + NODE_WIDTH + NEW_NODE_COLLISION_GAP_X > left
        and y < bottom
        and y + height > top
        for left, top, right, bottom in occupied
    ):
        x += NEW_NODE_OFFSET_X
        y += NEW_NODE_OFFSET_Y
    return NodeLayout(x=x, y=y)


def auto_layout_positions(
    document: WorkflowGraph,
) -> dict[str, NodeLayout]:
    nodes_by_id = {
        node.id: node
        for node in document.nodes
    }
    node_rank = {
        node.id: rank
        for rank, node in enumerate(document.nodes)
    }
    components = connected_node_components(document)
    component_by_node = {
        node_id: component_index
        for component_index, component in enumerate(components)
        for node_id in component
    }
    component_connections: list[list[WorkflowConnection]] = [
        []
        for _ in components
    ]
    for connection in document.connections:
        component_connections[
            component_by_node[connection.source.node_id]
        ].append(connection)
    positions: dict[str, NodeLayout] = {}
    component_y = AUTO_LAYOUT_ORIGIN_Y

    for component, connections in zip(
        components,
        component_connections,
        strict=True,
    ):
        component_positions, component_height = _layout_component(
            component,
            connections,
            nodes_by_id,
            node_rank,
        )
        for node_id, position in component_positions.items():
            positions[node_id] = NodeLayout(
                x=position.x,
                y=position.y + component_y,
            )
        component_y += component_height + AUTO_LAYOUT_COMPONENT_GAP

    return positions


def connected_node_components(
    document: WorkflowGraph,
) -> tuple[tuple[str, ...], ...]:
    node_rank = {
        node.id: rank
        for rank, node in enumerate(document.nodes)
    }
    neighbours = {
        node.id: set()
        for node in document.nodes
    }
    for connection in document.connections:
        source_id = connection.source.node_id
        target_id = connection.target.node_id
        neighbours[source_id].add(target_id)
        neighbours[target_id].add(source_id)

    visited: set[str] = set()
    components: list[tuple[str, ...]] = []
    for root in document.nodes:
        if root.id in visited:
            continue
        stack = [root.id]
        visited.add(root.id)
        component: list[str] = []
        while stack:
            node_id = stack.pop()
            component.append(node_id)
            adjacent = sorted(
                neighbours[node_id],
                key=node_rank.__getitem__,
                reverse=True,
            )
            for neighbour_id in adjacent:
                if neighbour_id in visited:
                    continue
                visited.add(neighbour_id)
                stack.append(neighbour_id)
        component.sort(key=node_rank.__getitem__)
        components.append(tuple(component))
    return tuple(components)


def _layout_component(
    component: tuple[str, ...],
    component_connections: list[WorkflowConnection],
    nodes_by_id: dict[str, WorkflowNode],
    node_rank: dict[str, int],
) -> tuple[dict[str, NodeLayout], int]:
    connections = tuple(
        sorted(
            component_connections,
            key=lambda connection: _connection_layout_key(
                connection,
                nodes_by_id,
                node_rank,
            ),
        )
    )
    predecessors = {
        node_id: set()
        for node_id in component
    }
    for connection in connections:
        predecessors[connection.target.node_id].add(
            connection.source.node_id
        )

    ordered_predecessors = {
        node_id: tuple(
            sorted(
                predecessors[node_id],
                key=node_rank.__getitem__,
            )
        )
        for node_id in component
    }
    depths: dict[str, int] = {}
    for node_id in TopologicalSorter(ordered_predecessors).static_order():
        depths[node_id] = max(
            (
                depths[predecessor] + 1
                for predecessor in predecessors[node_id]
            ),
            default=0,
        )

    vertices: list[_LayoutVertex] = []
    real_vertex: dict[str, int] = {}
    layer_count = max(depths.values(), default=0) + 1
    layers: list[list[int]] = [
        []
        for _ in range(layer_count)
    ]
    for node_id in component:
        node = nodes_by_id[node_id]
        vertex_id = len(vertices)
        real_vertex[node_id] = vertex_id
        vertex = _LayoutVertex(
            node_id=node_id,
            layer=depths[node_id],
            height=node_height(node.kind),
            rank=node_rank[node_id],
        )
        vertices.append(vertex)
        layers[vertex.layer].append(vertex_id)

    layer_edges: list[list[_LayerEdge]] = [
        []
        for _ in range(max(layer_count - 1, 0))
    ]
    incoming: list[list[_LayerEdge]] = [
        []
        for _ in vertices
    ]
    outgoing: list[list[_LayerEdge]] = [
        []
        for _ in vertices
    ]
    next_dummy_rank = len(nodes_by_id)

    for connection in connections:
        source_id = connection.source.node_id
        target_id = connection.target.node_id
        source_layer = depths[source_id]
        target_layer = depths[target_id]
        previous = real_vertex[source_id]
        previous_offset = _output_port_offset(
            nodes_by_id[source_id].kind,
            connection.source.port,
        )

        for layer in range(source_layer + 1, target_layer):
            dummy = len(vertices)
            vertices.append(
                _LayoutVertex(
                    node_id=None,
                    layer=layer,
                    height=1,
                    rank=next_dummy_rank,
                )
            )
            next_dummy_rank += 1
            layers[layer].append(dummy)
            incoming.append([])
            outgoing.append([])
            edge = _LayerEdge(
                source=previous,
                target=dummy,
                source_offset=previous_offset,
                target_offset=0,
            )
            _append_layer_edge(
                edge,
                layer_edges,
                incoming,
                outgoing,
                vertices,
            )
            previous = dummy
            previous_offset = 0

        edge = _LayerEdge(
            source=previous,
            target=real_vertex[target_id],
            source_offset=previous_offset,
            target_offset=_input_port_offset(
                nodes_by_id[target_id].kind,
                connection.target.port,
            ),
        )
        _append_layer_edge(
            edge,
            layer_edges,
            incoming,
            outgoing,
            vertices,
        )

    _minimize_crossings(
        layers,
        vertices,
        layer_edges,
        incoming,
        outgoing,
    )
    vertical_positions = _place_layers(
        layers,
        vertices,
        incoming,
        outgoing,
    )
    layer_x = _layer_x_positions(layer_edges)

    component_positions = {
        node_id: NodeLayout(
            x=layer_x[depths[node_id]],
            y=vertical_positions[vertex_id],
        )
        for node_id, vertex_id in real_vertex.items()
    }
    component_height = max(
        (
            vertical_positions[real_vertex[node_id]]
            + node_height(nodes_by_id[node_id].kind)
            for node_id in component
        ),
        default=0,
    )
    occupied_layers = frozenset(depths.values())
    detour_count = sum(
        any(
            depths[connection.source.node_id] < layer
            < depths[connection.target.node_id]
            for layer in occupied_layers
        )
        for connection in connections
    )
    if detour_count:
        component_height += (
            AUTO_LAYOUT_DETOUR_MARGIN_Y
            + (detour_count - 1) * AUTO_LAYOUT_DETOUR_ROW_GAP
        )
    return component_positions, component_height


def _connection_layout_key(
    connection: WorkflowConnection,
    nodes_by_id: dict[str, WorkflowNode],
    node_rank: dict[str, int],
) -> tuple[int, int, int, int, int, str]:
    source = nodes_by_id[connection.source.node_id]
    target = nodes_by_id[connection.target.node_id]
    return (
        node_rank[connection.target.node_id],
        _input_port_offset(target.kind, connection.target.port),
        connection.order,
        node_rank[connection.source.node_id],
        _output_port_offset(source.kind, connection.source.port),
        connection.id,
    )


def _append_layer_edge(
    edge: _LayerEdge,
    layer_edges: list[list[_LayerEdge]],
    incoming: list[list[_LayerEdge]],
    outgoing: list[list[_LayerEdge]],
    vertices: list[_LayoutVertex],
) -> None:
    boundary = vertices[edge.source].layer
    layer_edges[boundary].append(edge)
    outgoing[edge.source].append(edge)
    incoming[edge.target].append(edge)


def _output_port_offset(kind: NodeKind, port_name: str) -> int:
    return next(
        index
        for index, port in enumerate(
            node_definition(kind).outputs,
            start=1,
        )
        if port.name == port_name
    )


def _input_port_offset(kind: NodeKind, port_name: str) -> int:
    return next(
        index
        for index, port in enumerate(
            node_definition(kind).inputs,
            start=1,
        )
        if port.name == port_name
    )


def _minimize_crossings(
    layers: list[list[int]],
    vertices: list[_LayoutVertex],
    layer_edges: list[list[_LayerEdge]],
    incoming: list[list[_LayerEdge]],
    outgoing: list[list[_LayerEdge]],
) -> None:
    # Graphviz dot's mincross pattern: alternating median sweeps, adjacent
    # transposes with reverse-pass plateau moves, and restoration of the
    # globally best crossing count.
    positions = _layer_order(layers)
    best_order = _layer_snapshot(layers)
    best_crossings = _crossing_count(
        layer_edges,
        positions,
        vertices,
    )
    if best_crossings == 0:
        return

    for sweep in range(AUTO_LAYOUT_SWEEPS):
        forward = sweep % 2 == 0
        free_layers = (
            layers[1:]
            if forward
            else reversed(layers[:-1])
        )
        adjacent = incoming if forward else outgoing
        for layer in free_layers:
            _sort_layer(
                layer,
                positions,
                vertices,
                adjacent,
                from_predecessors=forward,
            )
            _update_layer_order(layer, positions)

        _transpose_layers(
            layers,
            positions,
            vertices,
            incoming,
            outgoing,
            reverse_ties=sweep % 4 >= 2,
        )
        crossings = _crossing_count(
            layer_edges,
            positions,
            vertices,
        )
        if crossings < best_crossings:
            best_crossings = crossings
            best_order = _layer_snapshot(layers)
            if best_crossings == 0:
                break

    for layer, order in zip(layers, best_order, strict=True):
        layer[:] = order


def _sort_layer(
    layer: list[int],
    positions: dict[int, int],
    vertices: list[_LayoutVertex],
    adjacent: list[list[_LayerEdge]],
    *,
    from_predecessors: bool,
) -> None:
    previous_positions = {
        vertex_id: index
        for index, vertex_id in enumerate(layer)
    }

    def sort_key(vertex_id: int) -> tuple[float, int, int]:
        coordinates = [
            _edge_endpoint_coordinate(
                edge.source if from_predecessors else edge.target,
                (
                    edge.source_offset
                    if from_predecessors
                    else edge.target_offset
                ),
                positions,
                vertices,
            )
            for edge in adjacent[vertex_id]
        ]
        return (
            _median(coordinates)
            if coordinates
            else float(previous_positions[vertex_id]),
            previous_positions[vertex_id],
            vertices[vertex_id].rank,
        )

    layer.sort(key=sort_key)


def _transpose_layers(
    layers: list[list[int]],
    positions: dict[int, int],
    vertices: list[_LayoutVertex],
    incoming: list[list[_LayerEdge]],
    outgoing: list[list[_LayerEdge]],
    *,
    reverse_ties: bool,
) -> None:
    while True:
        improvement = 0
        for layer in layers:
            improvement += _transpose_layer(
                layer,
                positions,
                vertices,
                incoming,
                outgoing,
                reverse_ties=reverse_ties,
            )
        if improvement == 0:
            return


def _transpose_layer(
    layer: list[int],
    positions: dict[int, int],
    vertices: list[_LayoutVertex],
    incoming: list[list[_LayerEdge]],
    outgoing: list[list[_LayerEdge]],
    *,
    reverse_ties: bool,
) -> int:
    improvement = 0
    for index in range(len(layer) - 1):
        upper = layer[index]
        lower = layer[index + 1]
        current, swapped = _pair_crossings(
            upper,
            lower,
            positions,
            vertices,
            incoming,
            outgoing,
        )
        if not (
            swapped < current
            or (
                reverse_ties
                and current > 0
                and swapped == current
            )
        ):
            continue
        layer[index], layer[index + 1] = lower, upper
        positions[upper], positions[lower] = (
            positions[lower],
            positions[upper],
        )
        improvement += current - swapped
    return improvement


def _layer_snapshot(
    layers: list[list[int]],
) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(layer) for layer in layers)


def _crossing_count(
    layer_edges: list[list[_LayerEdge]],
    positions: dict[int, int],
    vertices: list[_LayoutVertex],
) -> int:
    crossings = 0
    for edges in layer_edges:
        if len(edges) < 2:
            continue
        endpoints = sorted(
            (
                (
                    _edge_endpoint_coordinate(
                        edge.source,
                        edge.source_offset,
                        positions,
                        vertices,
                    ),
                    _edge_endpoint_coordinate(
                        edge.target,
                        edge.target_offset,
                        positions,
                        vertices,
                    ),
                )
                for edge in edges
            ),
            key=lambda endpoint: (endpoint[0], endpoint[1]),
        )
        target_coordinates = sorted(
            {
                target
                for _, target in endpoints
            }
        )
        tree = [0] * (len(target_coordinates) + 1)
        processed = 0
        group_start = 0
        while group_start < len(endpoints):
            group_end = group_start + 1
            source_coordinate = endpoints[group_start][0]
            while (
                group_end < len(endpoints)
                and endpoints[group_end][0] == source_coordinate
            ):
                group_end += 1

            for _, target_coordinate in endpoints[group_start:group_end]:
                target_rank = (
                    bisect_left(
                        target_coordinates,
                        target_coordinate,
                    )
                    + 1
                )
                crossings += processed - _prefix_sum(
                    tree,
                    target_rank,
                )

            for _, target_coordinate in endpoints[group_start:group_end]:
                target_rank = (
                    bisect_left(
                        target_coordinates,
                        target_coordinate,
                    )
                    + 1
                )
                _add_to_tree(tree, target_rank)
                processed += 1
            group_start = group_end
    return crossings


def _prefix_sum(tree: list[int], index: int) -> int:
    total = 0
    while index:
        total += tree[index]
        index -= index & -index
    return total


def _add_to_tree(tree: list[int], index: int) -> None:
    while index < len(tree):
        tree[index] += 1
        index += index & -index


def _pair_crossings(
    upper: int,
    lower: int,
    positions: dict[int, int],
    vertices: list[_LayoutVertex],
    incoming: list[list[_LayerEdge]],
    outgoing: list[list[_LayerEdge]],
) -> tuple[int, int]:
    current = 0
    swapped = 0
    for upper_edges, lower_edges, source_side in (
        (incoming[upper], incoming[lower], True),
        (outgoing[upper], outgoing[lower], False),
    ):
        for upper_edge in upper_edges:
            upper_vertex = (
                upper_edge.source
                if source_side
                else upper_edge.target
            )
            upper_offset = (
                upper_edge.source_offset
                if source_side
                else upper_edge.target_offset
            )
            upper_coordinate = _edge_endpoint_coordinate(
                upper_vertex,
                upper_offset,
                positions,
                vertices,
            )
            for lower_edge in lower_edges:
                lower_vertex = (
                    lower_edge.source
                    if source_side
                    else lower_edge.target
                )
                if upper_vertex == lower_vertex:
                    continue
                lower_offset = (
                    lower_edge.source_offset
                    if source_side
                    else lower_edge.target_offset
                )
                lower_coordinate = _edge_endpoint_coordinate(
                    lower_vertex,
                    lower_offset,
                    positions,
                    vertices,
                )
                current += upper_coordinate > lower_coordinate
                swapped += upper_coordinate < lower_coordinate
    return current, swapped


def _edge_endpoint_coordinate(
    vertex_id: int,
    port_offset: int,
    positions: dict[int, int],
    vertices: list[_LayoutVertex],
) -> float:
    vertex = vertices[vertex_id]
    return positions[vertex_id] + port_offset / (vertex.height + 1)


def _layer_order(layers: list[list[int]]) -> dict[int, int]:
    return {
        vertex_id: index
        for layer in layers
        for index, vertex_id in enumerate(layer)
    }


def _update_layer_order(
    layer: list[int],
    positions: dict[int, int],
) -> None:
    for index, vertex_id in enumerate(layer):
        positions[vertex_id] = index


def _place_layers(
    layers: list[list[int]],
    vertices: list[_LayoutVertex],
    incoming: list[list[_LayerEdge]],
    outgoing: list[list[_LayerEdge]],
) -> list[int]:
    positions = [0] * len(vertices)
    for layer in layers:
        y = 0
        for vertex_id in layer:
            positions[vertex_id] = y
            y += vertices[vertex_id].height + AUTO_LAYOUT_ROW_GAP

    for _ in range(AUTO_LAYOUT_SWEEPS):
        for layer in layers[1:]:
            _align_layer(
                layer,
                positions,
                vertices,
                incoming,
                from_predecessors=True,
            )
        for layer in reversed(layers[:-1]):
            _align_layer(
                layer,
                positions,
                vertices,
                outgoing,
                from_predecessors=False,
            )

    minimum = min(positions, default=0)
    if minimum:
        positions = [
            position - minimum
            for position in positions
        ]
    return positions


def _align_layer(
    layer: list[int],
    positions: list[int],
    vertices: list[_LayoutVertex],
    adjacent: list[list[_LayerEdge]],
    *,
    from_predecessors: bool,
) -> None:
    desired: list[float] = []
    for vertex_id in layer:
        candidates: list[float] = []
        for edge in adjacent[vertex_id]:
            if from_predecessors:
                neighbour = edge.source
                neighbour_offset = edge.source_offset
                own_offset = edge.target_offset
            else:
                neighbour = edge.target
                neighbour_offset = edge.target_offset
                own_offset = edge.source_offset
            candidates.append(
                positions[neighbour]
                + neighbour_offset
                - own_offset
            )
        desired.append(
            _median(candidates)
            if candidates
            else float(positions[vertex_id])
        )

    aligned = _non_overlapping_positions(
        layer,
        desired,
        vertices,
    )
    for vertex_id, position in zip(layer, aligned, strict=True):
        positions[vertex_id] = position


def _non_overlapping_positions(
    layer: list[int],
    desired: list[float],
    vertices: list[_LayoutVertex],
) -> list[int]:
    offsets: list[int] = []
    offset = 0
    for vertex_id in layer:
        offsets.append(offset)
        offset += vertices[vertex_id].height + AUTO_LAYOUT_ROW_GAP

    blocks: list[_IsotonicBlock] = []
    for index, (target, minimum_offset) in enumerate(
        zip(desired, offsets, strict=True)
    ):
        blocks.append(
            _IsotonicBlock(
                start=index,
                end=index + 1,
                total=target - minimum_offset,
                count=1,
            )
        )
        while len(blocks) >= 2 and blocks[-2].mean > blocks[-1].mean:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                _IsotonicBlock(
                    start=left.start,
                    end=right.end,
                    total=left.total + right.total,
                    count=left.count + right.count,
                )
            )

    projected = [0] * len(layer)
    for block in blocks:
        value = round(block.mean)
        for index in range(block.start, block.end):
            projected[index] = value + offsets[index]
    return projected


def _layer_x_positions(
    layer_edges: list[list[_LayerEdge]],
) -> list[int]:
    layer_x = [AUTO_LAYOUT_ORIGIN_X]
    for edges in layer_edges:
        routing_gap = len(edges) + AUTO_LAYOUT_ROUTING_MARGIN
        column_gap = max(AUTO_LAYOUT_MIN_COLUMN_GAP, routing_gap)
        layer_x.append(
            layer_x[-1] + NODE_WIDTH + column_gap
        )
    return layer_x


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2
