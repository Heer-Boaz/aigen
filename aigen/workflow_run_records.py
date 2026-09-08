from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
import re
from uuid import uuid4

from aigen.manifest_io import atomic_write_json, read_json
from aigen.workflow_results import NodeResultManifest, load_node_result


def new_workflow_run_id() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:12]}"


def workflow_run_path(root: Path, workflow_id: str, run_id: str) -> Path:
    if re.fullmatch(r"\d{8}T\d{12}Z-[0-9a-f]{12}", run_id) is None:
        raise ValueError("invalid workflow run id")
    return root / "runs" / workflow_id / f"attempt-{run_id}"


def write_workflow_interruption(run_dir: Path, *, workflow_digest: str, node_ids: Sequence[str],
                                node_manifests: Mapping[str, Path], message: str) -> Path:
    path = run_dir / "interrupted.json"
    atomic_write_json(path, {
        "status": "interrupted", "workflow_digest": workflow_digest,
        "node_ids": list(node_ids), "message": message,
        "completed_nodes": sorted(node_manifests),
    })
    return path


def published_workflow_nodes(run_dir: Path) -> dict[Path, NodeResultManifest]:
    return {path: load_node_result(path) for path in (run_dir / "nodes").glob("*/result.json")}


def finalize_terminated_workflow(run_dir: Path, message: str) -> tuple[NodeResultManifest, ...]:
    """Finalize the known run only after its command and descendants have exited."""
    state_path = run_dir / "run.json"
    if not state_path.exists():
        # Compilation or cancellation before execution never registered a run.
        return ()
    state = read_json(state_path, label="workflow run")
    # A node can publish its result just before the process dies, before run.json
    # has indexed it. Atomic node manifests remain the completion authority.
    published = published_workflow_nodes(run_dir)
    if state["status"] == "running":
        manifests = {result.node_id: path for path, result in published.items()}
        execution = read_json(run_dir / "execution.json", label="workflow execution")
        interruption = write_workflow_interruption(
            run_dir, workflow_digest=state["workflow_digest"],
            node_ids=tuple(node_id for node_id in execution["execution_order"] if node_id not in manifests),
            node_manifests=manifests, message=message,
        )
        state.update(status="interrupted", failure=str(interruption),
                     nodes={node_id: str(path) for node_id, path in manifests.items()})
        atomic_write_json(state_path, state)
    return tuple(published.values())
