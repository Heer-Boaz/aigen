from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Select

from aigen.workflow_graph import (
    ArtifactType,
    NodePortRef,
    PortDefinition,
    WorkflowGraph,
    node_definition,
)


@dataclass(frozen=True, slots=True)
class _SourceEndpoint:
    reference: NodePortRef
    definition: PortDefinition


@dataclass(frozen=True, slots=True)
class _TargetEndpoint:
    reference: NodePortRef
    rank: int


class WorkflowConnectionDialog(
    ModalScreen[tuple[NodePortRef, NodePortRef] | None]
):
    DEFAULT_CSS = """
    WorkflowConnectionDialog {
        align: center middle;
        background: #000000 55%;
    }

    WorkflowConnectionDialog #workflow-connect-dialog {
        width: 92%;
        height: auto;
        border: solid #8a66a3;
        background: #1c1724;
        padding: 1 2;
    }

    WorkflowConnectionDialog .workflow-connect-label {
        width: 10;
        height: 1;
    }

    WorkflowConnectionDialog .workflow-connect-select {
        width: 1fr;
        height: 1;
        border: none;
        padding: 0;
    }

    WorkflowConnectionDialog .workflow-connect-row {
        width: 100%;
        height: 1;
    }

    WorkflowConnectionDialog #workflow-connect-actions {
        width: 100%;
        height: 1;
        margin-top: 1;
        align-horizontal: right;
    }

    WorkflowConnectionDialog Button {
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        document: WorkflowGraph,
        selected_node_id: str | None,
    ) -> None:
        super().__init__()
        self._node_titles = {
            node.id: node.title
            for node in document.nodes
        }
        target_buckets: dict[ArtifactType, list[_TargetEndpoint]] = {
            artifact_type: []
            for artifact_type in ArtifactType
        }
        target_node_counts: dict[ArtifactType, Counter[str]] = {
            artifact_type: Counter()
            for artifact_type in ArtifactType
        }
        target_rank = 0
        for node in document.nodes:
            for port in node_definition(node.kind).inputs:
                endpoint = _TargetEndpoint(
                    reference=NodePortRef(
                        node_id=node.id,
                        port=port.name,
                    ),
                    rank=target_rank,
                )
                target_rank += 1
                for artifact_type in port.artifact_types:
                    target_buckets[artifact_type].append(endpoint)
                    target_node_counts[artifact_type][node.id] += 1
        self._targets_by_artifact = {
            artifact_type: tuple(endpoints)
            for artifact_type, endpoints in target_buckets.items()
        }

        self._sources: dict[str, _SourceEndpoint] = {}
        for node in document.nodes:
            for port in node_definition(node.kind).outputs:
                if not any(
                    len(self._targets_by_artifact[artifact_type])
                    > target_node_counts[artifact_type][node.id]
                    for artifact_type in port.artifact_types
                ):
                    continue
                source = _SourceEndpoint(
                    reference=NodePortRef(
                        node_id=node.id,
                        port=port.name,
                    ),
                    definition=port,
                )
                self._sources[_endpoint_key(source.reference)] = source
        preferred = next(
            (
                key
                for key, source in self._sources.items()
                if source.reference.node_id == selected_node_id
            ),
            None,
        )
        self._selected_source = preferred or next(
            iter(self._sources),
            None,
        )
        self._selected_targets = (
            self._compatible_targets(self._selected_source)
            if self._selected_source is not None
            else {}
        )

    @property
    def can_connect(self) -> bool:
        return self._selected_source is not None

    def compose(self) -> ComposeResult:
        assert self._selected_source is not None
        targets = tuple(self._selected_targets.values())
        with Container(id="workflow-connect-dialog"):
            yield Label("Connect nodes")
            with Horizontal(classes="workflow-connect-row"):
                yield Label("From", classes="workflow-connect-label")
                yield Select(
                    tuple(
                        (
                            self._endpoint_label(source.reference),
                            key,
                        )
                        for key, source in self._sources.items()
                    ),
                    value=self._selected_source,
                    allow_blank=False,
                    compact=True,
                    id="workflow-connect-source",
                    classes="workflow-connect-select",
                )
            with Horizontal(classes="workflow-connect-row"):
                yield Label("To", classes="workflow-connect-label")
                yield Select(
                    tuple(
                        (
                            self._endpoint_label(target),
                            _endpoint_key(target),
                        )
                        for target in targets
                    ),
                    value=_endpoint_key(targets[0]),
                    allow_blank=False,
                    compact=True,
                    id="workflow-connect-target",
                    classes="workflow-connect-select",
                )
            with Horizontal(id="workflow-connect-actions"):
                yield Button("Cancel", id="workflow-connect-cancel")
                yield Button(
                    "Connect",
                    id="workflow-connect-confirm",
                    variant="primary",
                )

    @on(Select.Changed, "#workflow-connect-source")
    def source_changed(self, event: Select.Changed) -> None:
        if not isinstance(event.value, str):
            return
        self._selected_source = event.value
        self._selected_targets = self._compatible_targets(event.value)
        targets = tuple(self._selected_targets.values())
        target_select = self.query_one(
            "#workflow-connect-target",
            Select,
        )
        target_select.set_options(
            tuple(
                (
                    self._endpoint_label(target),
                    _endpoint_key(target),
                )
                for target in targets
            )
        )
        target_select.value = _endpoint_key(targets[0])

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "workflow-connect-cancel":
            self.dismiss(None)
            return
        if event.button.id != "workflow-connect-confirm":
            return
        source_key = self.query_one(
            "#workflow-connect-source",
            Select,
        ).value
        target_key = self.query_one(
            "#workflow-connect-target",
            Select,
        ).value
        assert isinstance(source_key, str)
        assert isinstance(target_key, str)
        self.dismiss(
            (
                self._sources[source_key].reference,
                self._selected_targets[target_key],
            )
        )

    def _compatible_targets(
        self,
        source_key: str,
    ) -> dict[str, NodePortRef]:
        source = self._sources[source_key]
        compatible: dict[str, _TargetEndpoint] = {}
        for artifact_type in source.definition.artifact_types:
            for target in self._targets_by_artifact[artifact_type]:
                if target.reference.node_id == source.reference.node_id:
                    continue
                compatible.setdefault(
                    _endpoint_key(target.reference),
                    target,
                )
        return {
            _endpoint_key(target.reference): target.reference
            for target in sorted(
                compatible.values(),
                key=lambda endpoint: endpoint.rank,
            )
        }

    def _endpoint_label(self, endpoint: NodePortRef) -> str:
        return (
            f"{self._node_titles[endpoint.node_id]}."
            f"{endpoint.port}"
        )


def _endpoint_key(endpoint: NodePortRef) -> str:
    return f"{endpoint.node_id}:{endpoint.port}"
