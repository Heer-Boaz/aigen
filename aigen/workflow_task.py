from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aigen.manifest_io import read_json
from aigen.workflow_document_io import load_workflow_document
from aigen.workflow_graph import ImageSelectionNode, WorkflowGraph
from aigen.workflow_results import has_viewable_output
from aigen.workflow_run_state import WorkflowExecutionSnapshot


@dataclass(frozen=True)
class RecordedWorkflowRun:
    snapshot: WorkflowExecutionSnapshot
    results: dict[str, Path]


def read_workflow_runs(root: Path, workflow_id: str) -> tuple[RecordedWorkflowRun, ...]:
    """Read once at workspace entry/refresh/run completion, never during rendering."""
    records = []
    for path in sorted((root / 'runs' / workflow_id).glob('*/run.json'), reverse=True):
        state = read_json(path, label='workflow run')
        if not state['nodes']:
            continue
        document = load_workflow_document(Path(state['workflow_snapshot']))
        records.append(RecordedWorkflowRun(WorkflowExecutionSnapshot.capture(document),
                                          {node_id: Path(path) for node_id, path in state['nodes'].items()}))
    return tuple(records)


@dataclass(frozen=True)
class WorkflowStep:
    action: Literal['run', 'choose', 'connect', 'results']
    label: str
    description: str
    targets: tuple[str, ...] | None = None
    selection_id: str | None = None
    result_path: Path | None = None


def choice_frontier(document: WorkflowGraph, targets: tuple[str, ...] | None = None) -> tuple[tuple[str, str | None], ...]:
    """Earliest unresolved human choices; saved choices retain their artifact boundary."""
    nodes = {node.id: node for node in document.nodes}
    authored = document.incoming_connections()
    effective = document.execution_inputs()
    visited: dict[str, bool] = {}
    frontier: dict[str, str | None] = {}

    def visit(node_id: str) -> bool:
        if node_id in visited:
            return visited[node_id]
        node = nodes[node_id]
        unresolved = isinstance(node, ImageSelectionNode) and node.config.selected is None
        routes = authored[node_id].get('collection', ()) if unresolved else (
            wire for wires in effective[node_id].values() for wire in wires)
        preceding_choice = False
        for wire in routes:
            preceding_choice = visit(wire.source.node_id) or preceding_choice
        if unresolved and not preceding_choice:
            collection = authored[node_id].get('collection', ())
            frontier[node_id] = collection[0].source.node_id if collection else None
        visited[node_id] = unresolved or preceding_choice
        return visited[node_id]

    # Use the graph's target/scoping authority, then unfold only unresolved choices.
    for node_id in document.execution_scope(targets):
        visit(node_id)
    return tuple(frontier.items())


def project_results(document: WorkflowGraph, records: tuple[RecordedWorkflowRun, ...]) -> tuple[dict[str, Path], dict[str, Path]]:
    current = WorkflowExecutionSnapshot.capture(document)
    latest: dict[str, Path] = {}
    applicable: dict[str, Path] = {}
    for record in records:
        outdated = current.outdated_from(record.snapshot)
        for node_id, path in record.results.items():
            if node_id in current.nodes:
                latest.setdefault(node_id, path)
                if node_id not in outdated:
                    applicable.setdefault(node_id, path)
    return latest, applicable


def next_workflow_step(document: WorkflowGraph, applicable: dict[str, Path],
                       targets: tuple[str, ...] | None = None) -> WorkflowStep:
    frontier = choice_frontier(document, targets)
    for selection_id, collection_id in frontier:
        title = document.node(selection_id).title
        if collection_id is None:
            return WorkflowStep('connect', 'Connect input', f'Connect an image collection to {title}.',
                                selection_id=selection_id)
        if collection_id in applicable:
            return WorkflowStep('choose', 'Choose image', f'Choose an output for {title}.',
                                selection_id=selection_id, result_path=applicable[collection_id])
    if frontier:
        collections = tuple(dict.fromkeys(collection for _, collection in frontier))
        titles = ', '.join(document.node(node_id).title for node_id in collections)
        return WorkflowStep('run', 'Generate choices', f'Run to {titles}; then choose an output.', targets=collections)
    terminal = document.execution_targets(targets)
    viewable = tuple(node_id for node_id in terminal if has_viewable_output(document.node(node_id)))
    if viewable and all(node_id in applicable for node_id in terminal):
        return WorkflowStep('results', 'View results', 'View completed outputs for these settings.', targets=viewable)
    has_choice = any(isinstance(node, ImageSelectionNode) and node.config.selected is not None for node in document.nodes)
    return WorkflowStep('run', 'Continue' if has_choice else 'Run',
                        'Run with the chosen outputs.' if has_choice else 'Run workflow.', targets=targets)
