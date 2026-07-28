from __future__ import annotations

from graphlib import TopologicalSorter

from aigen.workflow_graph import (
    NodeKind,
    NodeLayout,
    WorkflowGraph,
    node_definition,
)


NODE_WIDTH = 34
AUTO_LAYOUT_ORIGIN_X = 4
AUTO_LAYOUT_ORIGIN_Y = 2
AUTO_LAYOUT_COLUMN_GAP = 10
AUTO_LAYOUT_ROW_GAP = 2
NEW_NODE_COLLISION_GAP_X = 1
NEW_NODE_OFFSET_X = 4
NEW_NODE_OFFSET_Y = 2


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
    nodes_by_id = {node.id: node for node in document.nodes}
    predecessors = {node.id: set() for node in document.nodes}
    for connection in document.connections:
        predecessors[connection.target.node_id].add(
            connection.source.node_id
        )

    depths: dict[str, int] = {}
    ordered_predecessors = {
        node_id: tuple(sorted(node_predecessors))
        for node_id, node_predecessors in predecessors.items()
    }
    for node_id in TopologicalSorter(ordered_predecessors).static_order():
        depths[node_id] = max(
            (
                depths[predecessor] + 1
                for predecessor in predecessors[node_id]
            ),
            default=0,
        )

    layers: dict[int, list[str]] = {}
    for node in document.nodes:
        layers.setdefault(depths[node.id], []).append(node.id)

    positions: dict[str, NodeLayout] = {}
    for depth, node_ids in layers.items():
        y = AUTO_LAYOUT_ORIGIN_Y
        for node_id in node_ids:
            node = nodes_by_id[node_id]
            positions[node_id] = NodeLayout(
                x=(
                    AUTO_LAYOUT_ORIGIN_X
                    + depth * (NODE_WIDTH + AUTO_LAYOUT_COLUMN_GAP)
                ),
                y=y,
            )
            y += node_height(node.kind) + AUTO_LAYOUT_ROW_GAP
    return positions
