from __future__ import annotations

from typing import Literal
from dataclasses import dataclass

from collections.abc import Sequence

from aigen.workflow_graph import NodePortRef, WorkflowGraph, WorkflowNode


@dataclass(frozen=True)
class WorkflowExecutionSnapshot:
    """Execution-relevant graph state; visual layout and titles never invalidate results."""

    nodes: dict[str, WorkflowNode]
    inputs: dict[str, dict[str, tuple[NodePortRef, ...]]]

    @classmethod
    def capture(cls, document: WorkflowGraph) -> WorkflowExecutionSnapshot:
        return cls({node.id: node for node in document.nodes}, _input_routes(document))

    def outdated_from(self, previous: WorkflowExecutionSnapshot) -> set[str]:
        outdated = {
            node_id for node_id, node in self.nodes.items()
            if node_id not in previous.nodes or node.kind != previous.nodes[node_id].kind
            or node.config != previous.nodes[node_id].config
            or self.inputs[node_id] != previous.inputs[node_id]
        }
        children: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for target, ports in self.inputs.items():
            for sources in ports.values():
                for source in sources:
                    children[source.node_id].append(target)
        work = list(outdated)
        while work:
            for child in children[work.pop()]:
                if child not in outdated:
                    outdated.add(child)
                    work.append(child)
        return outdated


class WorkflowRunState:
    """Last requested run and the applicability of its statuses to the edited graph."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._statuses: dict[str, str] = {}
        self._executed: WorkflowExecutionSnapshot | None = None
        self._document: WorkflowGraph | None = None
        self._node_ids: set[str] = set()
        self._outdated: set[str] = set()

    def start(self, document: WorkflowGraph, *, target_node_ids: Sequence[str] | None = None) -> None:
        self._executed = WorkflowExecutionSnapshot.capture(document)
        self._document = document
        self._node_ids = set(document.execution_scope(target_node_ids))
        self._statuses = {node_id: "queued" for node_id in self._node_ids}
        self._outdated = set()

    def update(self, node_id: str, status: str) -> None:
        self._statuses[node_id] = status

    def finish(self, status: Literal["failed", "interrupted"]) -> None:
        for node_id, current in self._statuses.items():
            if current in ("queued", "running"):
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
        self._outdated = WorkflowExecutionSnapshot.capture(document).outdated_from(self._executed)


def _input_routes(document: WorkflowGraph) -> dict[str, dict[str, tuple[NodePortRef, ...]]]:
    return {
        node_id: {
            port: tuple(connection.source for connection in connections)
            for port, connections in ports.items()
        }
        for node_id, ports in document.execution_inputs().items()
    }
