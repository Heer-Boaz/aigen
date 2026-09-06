from __future__ import annotations

from collections.abc import Sequence

from aigen.workflow_graph import NodePortRef, WorkflowGraph


class WorkflowRunState:
    """Last requested run and the applicability of its statuses to the edited graph."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._statuses: dict[str, str] = {}
        self._executed: WorkflowGraph | None = None
        self._document: WorkflowGraph | None = None
        self._node_ids: set[str] = set()
        self._outdated: set[str] = set()
        self._executed_inputs: dict[str, dict[str, tuple[NodePortRef, ...]]] = {}

    def start(self, document: WorkflowGraph, *, target_node_ids: Sequence[str] | None = None) -> None:
        self._executed = document
        self._document = document
        self._node_ids = set(document.execution_scope(target_node_ids))
        self._statuses = {node_id: "queued" for node_id in self._node_ids}
        self._outdated = set()
        self._executed_inputs = _input_routes(document)

    def update(self, node_id: str, status: str) -> None:
        self._statuses[node_id] = status

    def status(self, node_id: str) -> str:
        return "outdated" if node_id in self._outdated else self._statuses[node_id]

    def project(self, document: WorkflowGraph) -> dict[str, str]:
        if document is not self._document:
            self._reconcile(document)
        return {
            node_id: self.status(node_id)
            for node_id in self._statuses
            if node_id in self._node_ids
        }

    def _reconcile(self, document: WorkflowGraph) -> None:
        self._document = document
        self._node_ids = {node.id for node in document.nodes}
        if self._executed is None:
            return
        executed_nodes = {node.id: node for node in self._executed.nodes}
        inputs = _input_routes(document)
        outdated = {
            node.id for node in document.nodes
            if (
                node.id not in executed_nodes
                or node.kind != executed_nodes[node.id].kind
                or node.config != executed_nodes[node.id].config
                or inputs[node.id] != self._executed_inputs[node.id]
            )
        }
        children: dict[str, list[str]] = {node.id: [] for node in document.nodes}
        for ports in document.execution_inputs().values():
            for connections in ports.values():
                for connection in connections:
                    children[connection.source.node_id].append(connection.target.node_id)
        work = list(outdated)
        while work:
            for child in children[work.pop()]:
                if child not in outdated:
                    outdated.add(child)
                    work.append(child)
        self._outdated = outdated


def _input_routes(document: WorkflowGraph) -> dict[str, dict[str, tuple[NodePortRef, ...]]]:
    return {
        node_id: {
            port: tuple(connection.source for connection in connections)
            for port, connections in ports.items()
        }
        for node_id, ports in document.execution_inputs().items()
    }
