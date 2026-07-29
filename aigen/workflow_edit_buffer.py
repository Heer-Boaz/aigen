from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Union, cast, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError

from aigen.generation.animegen_i2v import animegen_sampling_profile
from aigen.generation.image_edit import image_edit_backend_settings
from aigen.workflow_graph import (
    AnimeGenI2VNode,
    FramePostprocessNode,
    ImageEditNode,
    ImagePostprocessNode,
    NodeKind,
    NodeLayout,
    NodePortRef,
    WorkflowConnection,
    WorkflowGraph,
    WorkflowNode,
    node_definition,
)
from aigen.workflow_layout import (
    auto_layout_positions,
    collision_free_node_layout,
)
from aigen.workflow_templates import (
    create_workflow_node,
    postprocess_config_for_model,
)


DEFAULT_WORKFLOW_HISTORY_LIMIT = 128


@dataclass(frozen=True, slots=True)
class _WorkflowEditRecord:
    before: WorkflowGraph
    after: WorkflowGraph
    before_revision: int
    after_revision: int
    label: str


@dataclass(frozen=True, slots=True)
class WorkflowPropertyEdit:
    node_id: str | None
    field_name: str
    raw_value: object


class WorkflowPropertyEditError(ValueError):
    def __init__(
        self,
        edit: WorkflowPropertyEdit,
        error: ValidationError | ValueError,
    ) -> None:
        super().__init__(str(error))
        self.edit = edit


class WorkflowEditBuffer:
    """Owns one workflow document, its save revision and edit history."""

    def __init__(
        self,
        document: WorkflowGraph,
        *,
        document_path: Path | None = None,
        history_limit: int = DEFAULT_WORKFLOW_HISTORY_LIMIT,
    ) -> None:
        self._undo: deque[_WorkflowEditRecord] = deque(
            maxlen=history_limit
        )
        self._redo: deque[_WorkflowEditRecord] = deque(
            maxlen=history_limit
        )
        self._document = document
        self._document_path = document_path
        self._revision = 0
        self._next_revision = 1
        self._saved_revision = 0

    @property
    def document(self) -> WorkflowGraph:
        return self._document

    @property
    def document_path(self) -> Path | None:
        return self._document_path

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def dirty(self) -> bool:
        return self._revision != self._saved_revision

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> str | None:
        return self._undo[-1].label if self._undo else None

    @property
    def redo_label(self) -> str | None:
        return self._redo[-1].label if self._redo else None

    def replace_document(self, document: WorkflowGraph) -> None:
        self._reset(document, document_path=None)

    def load_document(
        self,
        document: WorkflowGraph,
        document_path: Path,
    ) -> None:
        self._reset(document, document_path=document_path)

    def mark_saved(self, document_path: Path) -> None:
        self._document_path = document_path
        self._saved_revision = self._revision

    def undo(self) -> bool:
        if not self._undo:
            return False
        record = self._undo.pop()
        self._redo.append(record)
        self._document = record.before
        self._revision = record.before_revision
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        record = self._redo.pop()
        self._undo.append(record)
        self._document = record.after
        self._revision = record.after_revision
        return True

    def rename_graph(self, name: str) -> bool:
        edit = WorkflowPropertyEdit(
            node_id=None,
            field_name="name",
            raw_value=name,
        )
        return self._commit(
            _document_with_property_edit(self._document, edit),
            _property_edit_label(self._document, edit),
        )

    def add_node(
        self,
        kind: NodeKind,
        *,
        x: int,
        y: int,
    ) -> WorkflowNode:
        layout = collision_free_node_layout(
            self._document,
            kind,
            x=x,
            y=y,
        )
        node = create_workflow_node(
            kind,
            x=layout.x,
            y=layout.y,
        )
        document = _rebuild_graph(
            self._document,
            nodes=[*self._document.nodes, node],
        )
        self._commit(document, f"Add {node.title}")
        return node

    def delete_node(self, node_id: str) -> bool:
        removed = self._document.node(node_id)
        removed_connections = [
            connection
            for connection in self._document.connections
            if connection.source.node_id == node_id
            or connection.target.node_id == node_id
        ]
        affected_targets = {
            (connection.target.node_id, connection.target.port)
            for connection in removed_connections
            if connection.target.node_id != node_id
        }
        removed_connection_ids = {
            connection.id for connection in removed_connections
        }
        document = _rebuild_graph(
            self._document,
            nodes=[
                node
                for node in self._document.nodes
                if node.id != node_id
            ],
            connections=_normalized_connection_orders(
                [
                    connection
                    for connection in self._document.connections
                    if connection.id not in removed_connection_ids
                ],
                affected_targets,
            ),
        )
        return self._commit(document, f"Delete {removed.title}")

    def delete_connection(self, connection_id: str) -> bool:
        connection = _connection_by_id(
            self._document,
            connection_id,
        )
        document = _rebuild_graph(
            self._document,
            connections=_normalized_connection_orders(
                [
                    retained
                    for retained in self._document.connections
                    if retained.id != connection_id
                ],
                {
                    (
                        connection.target.node_id,
                        connection.target.port,
                    )
                },
            ),
        )
        return self._commit(
            document,
            (
                f"Disconnect {connection.source.node_id}."
                f"{connection.source.port}"
            ),
        )

    def connect_ports(
        self,
        source: NodePortRef,
        target: NodePortRef,
    ) -> WorkflowConnection:
        existing = next(
            (
                connection
                for connection in self._document.connections
                if connection.source == source
                and connection.target == target
            ),
            None,
        )
        if existing is not None:
            return existing

        target_node = self._document.node(target.node_id)
        target_port = node_definition(target_node.kind).input(target.port)
        if target_port is None:
            raise ValueError(
                f"unknown input {target.node_id}.{target.port}"
            )

        retained = [
            connection
            for connection in self._document.connections
            if target_port.multiple
            or connection.target != target
        ]
        order = (
            max(
                (
                    connection.order
                    for connection in retained
                    if connection.target == target
                ),
                default=-1,
            )
            + 1
            if target_port.multiple
            else 0
        )
        connection = WorkflowConnection(
            id=f"connection-{uuid.uuid4().hex}",
            source=source,
            target=target,
            order=order,
        )
        document = _rebuild_graph(
            self._document,
            connections=[*retained, connection],
        )
        self._commit(
            document,
            (
                f"Connect {source.node_id}.{source.port} to "
                f"{target.node_id}.{target.port}"
            ),
        )
        return connection

    def reconnect_connection(
        self,
        connection_id: str,
        source: NodePortRef,
        target: NodePortRef,
    ) -> WorkflowConnection:
        existing = _connection_by_id(self._document, connection_id)
        if existing.source == source and existing.target == target:
            return existing

        target_node = self._document.node(target.node_id)
        target_port = node_definition(target_node.kind).input(target.port)
        if target_port is None:
            raise ValueError(
                f"unknown input {target.node_id}.{target.port}"
            )

        retained = [
            connection
            for connection in self._document.connections
            if connection.id != connection_id
            and (
                target_port.multiple
                or connection.target != target
            )
        ]
        order = (
            existing.order
            if target_port.multiple and target == existing.target
            else (
                max(
                    (
                        connection.order
                        for connection in retained
                        if connection.target == target
                    ),
                    default=-1,
                )
                + 1
                if target_port.multiple
                else 0
            )
        )
        changed = WorkflowConnection(
            id=existing.id,
            source=source,
            target=target,
            order=order,
        )
        retained_ids = {
            connection.id
            for connection in retained
        }
        connections: list[WorkflowConnection] = []
        for connection in self._document.connections:
            if connection.id == connection_id:
                connections.append(changed)
            elif connection.id in retained_ids:
                connections.append(connection)
        document = _rebuild_graph(
            self._document,
            connections=_normalized_connection_orders(
                connections,
                {
                    (
                        existing.target.node_id,
                        existing.target.port,
                    ),
                    (target.node_id, target.port),
                },
            ),
        )
        self._commit(
            document,
            (
                f"Reconnect {existing.source.node_id}."
                f"{existing.source.port}"
            ),
        )
        return _connection_by_id(self._document, connection_id)

    def connection_move_capabilities(
        self,
        connection_id: str,
    ) -> tuple[bool, bool]:
        selected = _connection_by_id(self._document, connection_id)
        siblings = _ordered_connection_siblings(
            self._document,
            selected,
        )
        index = siblings.index(selected)
        return index > 0, index < len(siblings) - 1

    def move_connection(
        self,
        connection_id: str,
        direction: int,
    ) -> bool:
        selected = _connection_by_id(self._document, connection_id)
        siblings = _ordered_connection_siblings(
            self._document,
            selected,
        )
        source_index = siblings.index(selected)
        target_index = source_index + direction
        if not 0 <= target_index < len(siblings):
            return False

        other = siblings[target_index]
        connections = [
            _connection_with_order(connection, other.order)
            if connection.id == selected.id
            else (
                _connection_with_order(connection, selected.order)
                if connection.id == other.id
                else connection
            )
            for connection in self._document.connections
        ]
        return self._commit(
            _rebuild_graph(
                self._document,
                connections=connections,
            ),
            "Move connection earlier"
            if direction < 0
            else "Move connection later",
        )

    def move_node(
        self,
        node_id: str,
        *,
        x: int,
        y: int,
    ) -> bool:
        node = self._document.node(node_id)
        changed = _replace_node(
            node,
            layout=NodeLayout(x=x, y=y),
        )
        return self._replace_node(changed, f"Move {node.title}")

    def auto_layout(self) -> bool:
        positions = auto_layout_positions(self._document)
        nodes = [
            _replace_node(node, layout=positions[node.id])
            for node in self._document.nodes
        ]
        return self._commit(
            _rebuild_graph(self._document, nodes=nodes),
            "Auto layout",
        )

    def update_node_title(self, node_id: str, title: str) -> bool:
        edit = WorkflowPropertyEdit(
            node_id=node_id,
            field_name="title",
            raw_value=title,
        )
        return self._commit(
            _document_with_property_edit(self._document, edit),
            _property_edit_label(self._document, edit),
        )

    def update_node_config(
        self,
        node_id: str,
        field_name: str,
        raw_value: object,
    ) -> bool:
        edit = WorkflowPropertyEdit(
            node_id=node_id,
            field_name=field_name,
            raw_value=raw_value,
        )
        return self._commit(
            _document_with_property_edit(self._document, edit),
            _property_edit_label(self._document, edit),
        )

    def update_properties(
        self,
        edits: tuple[WorkflowPropertyEdit, ...],
    ) -> bool:
        document = self._document
        for edit in edits:
            try:
                document = _document_with_property_edit(document, edit)
            except (ValidationError, ValueError) as error:
                raise WorkflowPropertyEditError(edit, error) from error
        return self._commit(
            document,
            (
                _property_edit_label(self._document, edits[0])
                if len(edits) == 1
                else "Edit workflow properties"
            ),
        )

    def _replace_node(
        self,
        changed: WorkflowNode,
        label: str,
    ) -> bool:
        document = _rebuild_graph(
            self._document,
            nodes=[
                changed if node.id == changed.id else node
                for node in self._document.nodes
            ],
        )
        return self._commit(document, label)

    def _commit(
        self,
        document: WorkflowGraph,
        label: str,
    ) -> bool:
        if document == self._document:
            return False

        revision = self._next_revision
        self._next_revision += 1
        self._undo.append(
            _WorkflowEditRecord(
                before=self._document,
                after=document,
                before_revision=self._revision,
                after_revision=revision,
                label=label,
            )
        )
        self._redo.clear()
        self._document = document
        self._revision = revision
        return True

    def _reset(
        self,
        document: WorkflowGraph,
        *,
        document_path: Path | None,
    ) -> None:
        self._document = document
        self._document_path = document_path
        self._revision = 0
        self._next_revision = 1
        self._saved_revision = 0
        self._undo.clear()
        self._redo.clear()


def _document_with_property_edit(
    document: WorkflowGraph,
    edit: WorkflowPropertyEdit,
) -> WorkflowGraph:
    if edit.node_id is None:
        return WorkflowGraph(
            version=document.version,
            name=cast(str, edit.raw_value),
            nodes=document.nodes,
            connections=document.connections,
        )

    node = document.node(edit.node_id)
    if edit.field_name == "title":
        changed = _replace_node(
            node,
            title=cast(str, edit.raw_value),
        )
    else:
        model_field = type(node.config).model_fields[edit.field_name]
        value = _parse_config_value(
            edit.raw_value,
            model_field.annotation,
        )
        if (
            isinstance(
                node,
                (ImagePostprocessNode, FramePostprocessNode),
            )
            and edit.field_name == "model"
        ):
            config = postprocess_config_for_model(str(value))
        else:
            config_payload = node.config.model_dump(mode="python")
            config_payload[edit.field_name] = value
            _apply_backend_defaults(
                node,
                edit.field_name,
                value,
                config_payload,
            )
            config = type(node.config).model_validate(config_payload)
        changed = _replace_node(node, config=config)

    return _rebuild_graph(
        document,
        nodes=[
            changed if retained.id == changed.id else retained
            for retained in document.nodes
        ],
    )


def _property_edit_label(
    document: WorkflowGraph,
    edit: WorkflowPropertyEdit,
) -> str:
    if edit.node_id is None:
        return "Rename workflow"
    node = document.node(edit.node_id)
    if edit.field_name == "title":
        return f"Rename {node.title}"
    return f"Edit {node.title} {edit.field_name}"


def _rebuild_graph(
    document: WorkflowGraph,
    *,
    nodes: list[WorkflowNode] | None = None,
    connections: list[WorkflowConnection] | None = None,
) -> WorkflowGraph:
    return WorkflowGraph(
        version=document.version,
        name=document.name,
        nodes=document.nodes if nodes is None else nodes,
        connections=(
            document.connections
            if connections is None
            else connections
        ),
    )


def _replace_node(
    node: WorkflowNode,
    *,
    title: str | None = None,
    layout: NodeLayout | None = None,
    config: BaseModel | None = None,
) -> WorkflowNode:
    payload = node.model_dump(mode="python")
    if title is not None:
        payload["title"] = title
    if layout is not None:
        payload["layout"] = layout
    if config is not None:
        payload["config"] = config
    return type(node).model_validate(payload)


def _normalized_connection_orders(
    connections: list[WorkflowConnection],
    targets: set[tuple[str, str]],
) -> list[WorkflowConnection]:
    groups: dict[tuple[str, str], list[WorkflowConnection]] = {
        target: [] for target in targets
    }
    for connection in connections:
        target = (connection.target.node_id, connection.target.port)
        if target in groups:
            groups[target].append(connection)

    replacements: dict[str, WorkflowConnection] = {}
    for siblings in groups.values():
        siblings.sort(key=lambda connection: connection.order)
        for order, connection in enumerate(siblings):
            if connection.order != order:
                replacements[connection.id] = _connection_with_order(
                    connection,
                    order,
                )
    if not replacements:
        return connections
    return [
        replacements.get(connection.id, connection)
        for connection in connections
    ]


def _connection_with_order(
    connection: WorkflowConnection,
    order: int,
) -> WorkflowConnection:
    payload = connection.model_dump(mode="python")
    payload["order"] = order
    return WorkflowConnection.model_validate(payload)


def _connection_by_id(
    document: WorkflowGraph,
    connection_id: str,
) -> WorkflowConnection:
    return next(
        connection
        for connection in document.connections
        if connection.id == connection_id
    )


def _ordered_connection_siblings(
    document: WorkflowGraph,
    connection: WorkflowConnection,
) -> list[WorkflowConnection]:
    return sorted(
        (
            sibling
            for sibling in document.connections
            if sibling.target == connection.target
        ),
        key=lambda sibling: sibling.order,
    )


def _apply_backend_defaults(
    node: WorkflowNode,
    field_name: str,
    value: object,
    config_payload: dict[str, object],
) -> None:
    if isinstance(node, ImageEditNode) and field_name == "backend":
        settings = image_edit_backend_settings(str(value))
        config_payload.update(
            steps=settings.steps,
            guidance=settings.guidance,
            strength=settings.strength,
            sampler=settings.sampler,
            scheduler=settings.scheduler,
        )
    elif (
        isinstance(node, AnimeGenI2VNode)
        and field_name == "sampling"
    ):
        config_payload["steps"] = animegen_sampling_profile(
            str(value)
        ).steps


def _parse_config_value(
    raw_value: object,
    annotation: object,
) -> object:
    if (
        raw_value == ""
        and get_origin(annotation) in {Union, UnionType}
        and type(None) in get_args(annotation)
    ):
        return None
    return TypeAdapter(annotation).validate_python(raw_value)
