from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from graphlib import TopologicalSorter
from hashlib import sha256
from pathlib import Path
from typing import Literal, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from aigen.generation.animegen_i2v import generate_animegen_i2v
from aigen.generation.image_batch_postprocess import (
    ImageBatchPostprocessResult,
    postprocess_image_batch,
)
from aigen.generation.image_edit import (
    FLUX2_KLEIN_BACKEND,
    QWEN_2511_BASE_BACKEND,
    QWEN_2511_LIGHTNING_BACKEND,
    ImageEditRequest,
    resolve_image_edit_canvas_size,
    run_image_edit,
)
from aigen.generation.image_edit_batch import (
    ImageEditBatchCase,
    ImageEditBatchLora,
    ImageEditBatchRequest,
    run_image_edit_batch,
)
from aigen.generation.video_postprocess import (
    create_video_contact_sheet,
    extract_video_frames,
)
from aigen.manifest_io import atomic_write_json, sha256_file
from aigen.progress import (
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    SILENT_STATUS,
    RuntimeStatus,
    StatusReporter,
)
from aigen.system_telemetry import SystemTelemetrySampler
from aigen.workflow_compilation import (
    CompiledAnimeGenConfig,
    CompiledImageEditConfig,
    CompiledImageSourceConfig,
    CompiledIllustrationUpscaleConfig,
    CompiledLoraSourceConfig,
    CompiledNode,
    CompiledPixelArtFixerConfig,
    CompiledPostprocessConfig,
    CompiledReferencePackConfig,
    CompiledVosrLongSideConfig,
    CompiledVosrScaleConfig,
    CompiledWorkflow,
    CompiledWuPixelizationConfig,
    execution_config_payload,
)
from aigen.workflow_artifacts import (
    ImageArtifact,
    ImageSequenceArtifact,
    LoraArtifact,
    ReferencePackArtifact,
    VideoArtifact,
    WorkflowArtifact,
)
from aigen.workflow_document_io import save_workflow_document
from aigen.workflow_graph import (
    AnimeGenI2VNode,
    ArtifactType,
    ExtractVideoFramesNode,
    FramePostprocessNode,
    ImageEditNode,
    ImagePostprocessNode,
    ImageSourceNode,
    LoraSourceNode,
    NodeKind,
    ReferencePackNode,
    VideoContactSheetNode,
    WorkflowGraph,
    WorkflowNode,
    WorkflowConnection,
    node_definition,
)
from aigen.workflow_cache import (
    GeneratedNodeOutput,
    NodeCacheHit,
    NodeExecutionProvenance,
    NodeInputIdentity,
    WorkflowNodeCache,
    build_node_signature,
)
from aigen.workflow_provenance import workflow_node_provenance


WORKFLOW_EVENT_PREFIX = "AIGEN_WORKFLOW "
WORKFLOW_RUN_VERSION = 1


class WorkflowExecutionError(RuntimeError):
    pass


class WorkflowInterrupted(WorkflowExecutionError):
    pass


class RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NodeResultManifest(RuntimeModel):
    version: Literal[WORKFLOW_RUN_VERSION] = WORKFLOW_RUN_VERSION
    node_id: str
    node_kind: NodeKind
    signature: str
    outputs: dict[str, WorkflowArtifact]


@dataclass(frozen=True)
class WorkflowRunResult:
    run_dir: Path
    result_path: Path
    workflow_digest: str
    node_manifests: Mapping[str, Path]
    terminal_outputs: Mapping[str, Mapping[str, WorkflowArtifact]]

    def to_json(self) -> dict[str, object]:
        return {
            "status": "completed",
            "kind": "workflow-run",
            "workflow_digest": self.workflow_digest,
            "run_dir": self.run_dir.as_posix(),
            "result": self.result_path.as_posix(),
            "nodes": {
                node_id: manifest.as_posix()
                for node_id, manifest in self.node_manifests.items()
            },
            "outputs": {
                node_id: {
                    port: artifact.model_dump(mode="json")
                    for port, artifact in outputs.items()
                }
                for node_id, outputs in self.terminal_outputs.items()
            },
        }


WorkflowEventSink = Callable[[dict[str, object]], None]
NodeProgressSink = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class _ResolvedImageEditPlan:
    backend: str
    prompt: str
    seed: int
    references: tuple[Path, ...]
    loras: tuple[ImageEditBatchLora, ...]
    width: int
    height: int
    steps: int
    guidance: float | None
    strength: float | None
    sampler: str
    scheduler: str


@dataclass(frozen=True)
class _PendingNodeExecution:
    compiled_node: CompiledNode
    node: WorkflowNode
    inputs: Mapping[str, Sequence[WorkflowArtifact]]
    signature: str
    provenance: NodeExecutionProvenance
    image_edit_plan: _ResolvedImageEditPlan | None = None


@dataclass(frozen=True)
class _NodeOutcome:
    outputs: Mapping[str, WorkflowArtifact]


def execute_workflow(
    workflow: CompiledWorkflow,
    *,
    runs_root: Path,
    progress: StatusReporter,
    event_sink: WorkflowEventSink | None = None,
    node_progress_sink: NodeProgressSink | None = None,
) -> WorkflowRunResult:
    graph = workflow.document
    execution_order = workflow.execution_order
    workflow_digest = workflow.digest
    workflow_root = runs_root.expanduser().resolve()
    run_dir = _create_run_dir(workflow_root, workflow_digest)
    node_cache = WorkflowNodeCache(workflow_root / "cache")
    snapshot_path = _save_snapshot(graph, run_dir)
    outputs_by_node: dict[str, dict[str, WorkflowArtifact]] = {}
    node_manifests: dict[str, Path] = {}
    order_index = {
        node_id: index
        for index, node_id in enumerate(execution_order)
    }
    sorter = TopologicalSorter(workflow.predecessors)
    sorter.prepare()
    ready = tuple(sorter.get_ready())
    pending: dict[str, _PendingNodeExecution] = {}
    progress.begin(len(execution_order), f"workflow: {graph.name}")
    _write_run_state(
        run_dir,
        graph,
        workflow_digest,
        snapshot_path,
        status="running",
        node_manifests=node_manifests,
    )

    active_node_ids: tuple[str, ...] = ()
    try:
        while sorter.is_active():
            run_state_changed = False
            while ready:
                for node_id in sorted(
                    ready,
                    key=order_index.__getitem__,
                ):
                    active_node_ids = (node_id,)
                    compiled_node = workflow.node(node_id)
                    node = compiled_node.node
                    inputs = _resolve_inputs(
                        node_id,
                        compiled_node.incoming,
                        outputs_by_node,
                    )
                    source_outputs = _source_outputs(compiled_node)
                    provenance = workflow_node_provenance(node)
                    signature = _node_signature(
                        compiled_node,
                        inputs,
                        source_outputs,
                        provenance,
                    )
                    outcome: _NodeOutcome | None = None
                    status: Literal["completed", "reused"]
                    if source_outputs is not None:
                        _validate_output_contract(
                            node,
                            source_outputs,
                        )
                        outcome = _NodeOutcome(
                            outputs=dict(source_outputs),
                        )
                        status = "completed"
                    else:
                        cache_hit = node_cache.lookup(
                            signature,
                            node_kind=node.kind,
                            provenance=provenance,
                        )
                        if cache_hit is not None:
                            outcome = _outcome_from_cache_hit(
                                node,
                                cache_hit,
                            )
                            status = "reused"

                    if outcome is None:
                        pending[node_id] = _PendingNodeExecution(
                            compiled_node=compiled_node,
                            node=node,
                            inputs=inputs,
                            signature=signature,
                            provenance=provenance,
                            image_edit_plan=(
                                _resolve_image_edit_plan(
                                    compiled_node,
                                    inputs,
                                )
                                if isinstance(node, ImageEditNode)
                                else None
                            ),
                        )
                        active_node_ids = ()
                        continue

                    outputs_by_node[node_id] = dict(outcome.outputs)
                    node_manifests[node_id] = _write_node_manifest(
                        run_dir,
                        node,
                        signature,
                        outcome.outputs,
                    )
                    _emit(
                        event_sink,
                        node_id=node_id,
                        node_kind=node.kind,
                        status=status,
                    )
                    progress.step(f"{status} {node.title}")
                    sorter.done(node_id)
                    active_node_ids = ()
                    run_state_changed = True
                ready = tuple(sorter.get_ready())

            if run_state_changed:
                _write_run_state(
                    run_dir,
                    graph,
                    workflow_digest,
                    snapshot_path,
                    status="running",
                    node_manifests=node_manifests,
                )

            if not pending:
                if sorter.is_active():
                    raise WorkflowExecutionError(
                        "workflow scheduler has no ready or pending nodes"
                    )
                break

            group = _next_execution_group(
                tuple(pending.values()),
                order_index,
            )
            active_node_ids = tuple(item.node.id for item in group)
            for item in group:
                _emit(
                    event_sink,
                    node_id=item.node.id,
                    node_kind=item.node.kind,
                    status="running",
                )
            outcomes = _execute_group(
                group,
                node_cache=node_cache,
                node_progress_sink=node_progress_sink,
            )
            for item, outcome in zip(group, outcomes, strict=True):
                node_id = item.node.id
                outputs_by_node[node_id] = dict(outcome.outputs)
                node_manifests[node_id] = _write_node_manifest(
                    run_dir,
                    item.node,
                    item.signature,
                    outcome.outputs,
                )
                pending.pop(node_id)
                sorter.done(node_id)
                _emit(
                    event_sink,
                    node_id=node_id,
                    node_kind=item.node.kind,
                    status="completed",
                )
                progress.step(f"completed {item.node.title}")
            active_node_ids = ()
            _write_run_state(
                run_dir,
                graph,
                workflow_digest,
                snapshot_path,
                status="running",
                node_manifests=node_manifests,
            )
            ready = tuple(sorter.get_ready())
    except WorkflowInterrupted as error:
        for node_id in active_node_ids:
            node = workflow.node(node_id).node
            _emit(
                event_sink,
                node_id=node_id,
                node_kind=node.kind,
                status="interrupted",
                message=str(error),
            )
        interruption_path = run_dir / "interrupted.json"
        atomic_write_json(
            interruption_path,
            {
                "status": "interrupted",
                "workflow_digest": workflow_digest,
                "node_ids": list(active_node_ids),
                "message": str(error),
                "completed_nodes": sorted(node_manifests),
            },
        )
        _write_run_state(
            run_dir,
            graph,
            workflow_digest,
            snapshot_path,
            status="interrupted",
            node_manifests=node_manifests,
            failure=interruption_path,
        )
        raise
    except Exception as error:
        for node_id in active_node_ids:
            node = workflow.node(node_id).node
            _emit(
                event_sink,
                node_id=node_id,
                node_kind=node.kind,
                status="failed",
                message=str(error),
            )
        failure_path = run_dir / "failure.json"
        atomic_write_json(
            failure_path,
            {
                "status": "failed",
                "workflow_digest": workflow_digest,
                "node_ids": list(active_node_ids),
                "message": str(error),
                "completed_nodes": sorted(node_manifests),
            },
        )
        _write_run_state(
            run_dir,
            graph,
            workflow_digest,
            snapshot_path,
            status="failed",
            node_manifests=node_manifests,
            failure=failure_path,
        )
        if isinstance(error, WorkflowExecutionError):
            raise
        raise WorkflowExecutionError(str(error)) from error

    terminal_outputs = {
        node_id: outputs_by_node[node_id]
        for node_id in workflow.terminal_node_ids
    }
    result_path = run_dir / "result.json"
    result = WorkflowRunResult(
        run_dir=run_dir,
        result_path=result_path,
        workflow_digest=workflow_digest,
        node_manifests=node_manifests,
        terminal_outputs=terminal_outputs,
    )
    atomic_write_json(result_path, result.to_json())
    _write_run_state(
        run_dir,
        graph,
        workflow_digest,
        snapshot_path,
        status="completed",
        node_manifests=node_manifests,
        result=result_path,
    )
    return result


def _next_execution_group(
    pending: Sequence[_PendingNodeExecution],
    order_index: Mapping[str, int],
) -> tuple[_PendingNodeExecution, ...]:
    ordered = sorted(
        pending,
        key=lambda item: order_index[item.node.id],
    )
    first = ordered[0]
    if _is_batchable_image_edit(first.image_edit_plan):
        key = (
            "image-edit",
            _image_edit_batch_key(
                cast(_ResolvedImageEditPlan, first.image_edit_plan)
            ),
        )
    elif isinstance(first.node, ImagePostprocessNode):
        key = (
            "image-postprocess",
            _digest(execution_config_payload(first.compiled_node.config)),
        )
    else:
        return (first,)
    return tuple(
        item
        for item in ordered
        if (
            (
                "image-edit",
                _image_edit_batch_key(
                    cast(_ResolvedImageEditPlan, item.image_edit_plan)
                ),
            )
            if _is_batchable_image_edit(item.image_edit_plan)
            else (
                "image-postprocess",
                _digest(
                    execution_config_payload(item.compiled_node.config)
                ),
            )
            if isinstance(item.node, ImagePostprocessNode)
            else ("node", item.node.id)
        )
        == key
    )


def _is_batchable_image_edit(
    plan: _ResolvedImageEditPlan | None,
) -> bool:
    return plan is not None and plan.backend in {
        FLUX2_KLEIN_BACKEND,
        QWEN_2511_LIGHTNING_BACKEND,
        QWEN_2511_BASE_BACKEND,
    }


def _resolve_image_edit_plan(
    compiled_node: CompiledNode,
    inputs: Mapping[str, Sequence[WorkflowArtifact]],
) -> _ResolvedImageEditPlan:
    node = cast(ImageEditNode, compiled_node.node)
    config = cast(CompiledImageEditConfig, compiled_node.config)
    references = _image_edit_reference_paths(node, inputs)
    settings = config.settings
    width, height = resolve_image_edit_canvas_size(
        backend=config.backend,
        first_reference=references[0],
        settings=settings,
    )
    return _ResolvedImageEditPlan(
        backend=config.backend,
        prompt=config.prompt,
        seed=config.seed,
        references=references,
        loras=tuple(
            ImageEditBatchLora(
                path=artifact.path,
                weight=artifact.weight,
            )
            for artifact in _image_edit_loras(node, inputs)
        ),
        width=width,
        height=height,
        steps=settings.steps,
        guidance=settings.guidance,
        strength=settings.strength,
        sampler=settings.sampler,
        scheduler=settings.scheduler,
    )


def _image_edit_batch_key(plan: _ResolvedImageEditPlan) -> str:
    session: dict[str, object] = {
        "backend": plan.backend,
        "loras": [
            lora.model_dump(mode="json")
            for lora in plan.loras
        ],
        "steps": plan.steps,
        "guidance": plan.guidance,
        "strength": plan.strength,
        "sampler": plan.sampler,
        "scheduler": plan.scheduler,
    }
    if plan.backend in {
        QWEN_2511_LIGHTNING_BACKEND,
        QWEN_2511_BASE_BACKEND,
    }:
        session["canvas"] = (plan.width, plan.height)
    return _digest(session)


def _execute_group(
    group: Sequence[_PendingNodeExecution],
    *,
    node_cache: WorkflowNodeCache,
    node_progress_sink: NodeProgressSink | None,
) -> tuple[_NodeOutcome, ...]:
    if all(_is_batchable_image_edit(item.image_edit_plan) for item in group):
        return _execute_image_edit_group(
            group,
            node_cache=node_cache,
            node_progress_sink=node_progress_sink,
        )
    if len(group) > 1 and all(
        isinstance(item.node, ImagePostprocessNode) for item in group
    ):
        return _execute_image_postprocess_group(
            group,
            node_cache=node_cache,
            node_progress_sink=node_progress_sink,
        )
    return tuple(
        _execute_node(
            item,
            node_cache=node_cache,
            node_progress_sink=node_progress_sink,
        )
        for item in group
    )


def _execute_image_edit_group(
    group: Sequence[_PendingNodeExecution],
    *,
    node_cache: WorkflowNodeCache,
    node_progress_sink: NodeProgressSink | None,
) -> tuple[_NodeOutcome, ...]:
    plans = tuple(
        cast(_ResolvedImageEditPlan, item.image_edit_plan)
        for item in group
    )
    with ExitStack() as stack:
        writes = tuple(
            stack.enter_context(
                node_cache.begin(
                    item.signature,
                    node_kind=item.node.kind,
                    provenance=item.provenance,
                )
            )
            for item in group
        )
        request = ImageEditBatchRequest(
            backend=plans[0].backend,
            cases=tuple(
                ImageEditBatchCase(
                    id=item.node.id,
                    prompt=plan.prompt,
                    image_paths=plan.references,
                    width=plan.width,
                    height=plan.height,
                    seed=plan.seed,
                    output_path=write.output_dir / "image.png",
                )
                for item, plan, write in zip(
                    group,
                    plans,
                    writes,
                    strict=True,
                )
            ),
            loras=plans[0].loras,
            steps=plans[0].steps,
            guidance=plans[0].guidance,
            strength=plans[0].strength,
            sampler=plans[0].sampler,
            scheduler=plans[0].scheduler,
        )
        with _node_progress(
            tuple(item.node.id for item in group),
            node_progress_sink,
        ) as node_progress:
            result = run_image_edit_batch(
                request,
                progress=node_progress,
            )
        outputs_by_case = {
            output.case_id: output
            for output in result.outputs
        }
        expected_case_ids = {item.node.id for item in group}
        if set(outputs_by_case) != expected_case_ids:
            raise WorkflowExecutionError(
                "image-edit batch returned cases "
                f"{sorted(outputs_by_case)}; expected "
                f"{sorted(expected_case_ids)}"
            )
        validated_outputs = tuple(
            _require_file(
                outputs_by_case[item.node.id].path,
                item.node,
            )
            for item in group
        )
        outcomes = []
        for item, write, output_path in zip(
            group,
            writes,
            validated_outputs,
            strict=True,
        ):
            output = outputs_by_case[item.node.id]
            expected = write.output_dir / "image.png"
            if output.path.resolve() != expected.resolve():
                raise WorkflowExecutionError(
                    f"image-edit batch returned the wrong output for "
                    f"{item.node.id!r}: {output.path}"
                )
            cache_hit = write.publish(
                {
                    "image": GeneratedNodeOutput(
                        artifact_type=ArtifactType.IMAGE,
                        paths=(output_path,),
                    )
                }
            )
            outcomes.append(
                _outcome_from_cache_hit(item.node, cache_hit)
            )
        return tuple(outcomes)


def format_workflow_event(payload: dict[str, object]) -> str:
    return WORKFLOW_EVENT_PREFIX + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _create_run_dir(workflow_root: Path, workflow_digest: str) -> Path:
    attempts_root = workflow_root / "runs" / workflow_digest
    attempts_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = attempts_root / f"attempt-{timestamp}-{uuid4().hex[:12]}"
    run_dir.mkdir()
    return run_dir


def _save_snapshot(graph: WorkflowGraph, run_dir: Path) -> Path:
    encoded = graph.model_dump_json(exclude_none=False).encode("utf-8")
    document_digest = sha256(encoded).hexdigest()
    snapshot_path = run_dir / "snapshots" / f"{document_digest}.json"
    if not snapshot_path.exists():
        save_workflow_document(graph, snapshot_path)
    return snapshot_path


def _write_run_state(
    run_dir: Path,
    graph: WorkflowGraph,
    workflow_digest: str,
    snapshot_path: Path,
    *,
    status: Literal["running", "completed", "failed", "interrupted"],
    node_manifests: Mapping[str, Path],
    failure: Path | None = None,
    result: Path | None = None,
) -> None:
    payload: dict[str, object] = {
        "version": WORKFLOW_RUN_VERSION,
        "status": status,
        "workflow_name": graph.name,
        "workflow_digest": workflow_digest,
        "workflow_snapshot": snapshot_path.as_posix(),
        "nodes": {
            node_id: path.as_posix()
            for node_id, path in node_manifests.items()
        },
    }
    if failure is not None:
        payload["failure"] = failure.as_posix()
    if result is not None:
        payload["result"] = result.as_posix()
    atomic_write_json(run_dir / "run.json", payload)


def _resolve_inputs(
    node_id: str,
    incoming: Mapping[str, Sequence[WorkflowConnection]],
    outputs_by_node: Mapping[str, Mapping[str, WorkflowArtifact]],
) -> dict[str, tuple[WorkflowArtifact, ...]]:
    resolved: dict[str, tuple[WorkflowArtifact, ...]] = {}
    for port, connections in incoming.items():
        artifacts: list[WorkflowArtifact] = []
        for connection in connections:
            source = connection.source
            try:
                artifact = outputs_by_node[source.node_id][source.port]
            except KeyError as error:
                raise WorkflowExecutionError(
                    f"node {node_id!r} depends on unavailable artifact "
                    f"{source.node_id}.{source.port}"
                ) from error
            artifacts.append(artifact)
        resolved[port] = tuple(artifacts)
    return resolved


def _source_outputs(
    compiled_node: CompiledNode,
) -> dict[str, WorkflowArtifact] | None:
    node = compiled_node.node
    if isinstance(node, ImageSourceNode):
        path = cast(CompiledImageSourceConfig, compiled_node.config).path
        identity = sha256_file(path)
        return {
            "image": ImageArtifact(
                path=path.as_posix(),
                identity=identity,
            )
        }
    if isinstance(node, ReferencePackNode):
        pack = cast(
            CompiledReferencePackConfig,
            compiled_node.config,
        ).pack
        references = tuple(pack.references.values())
        identity_payload = {
            "pack": sha256_file(pack.path),
            "references": [
                {
                    "name": name,
                    "path": reference.as_posix(),
                    "sha256": sha256_file(reference),
                }
                for name, reference in pack.references.items()
            ],
        }
        identity = _digest(identity_payload)
        return {
            "pack": ReferencePackArtifact(
                path=pack.path.as_posix(),
                references=tuple(path.as_posix() for path in references),
                identity=identity,
            )
        }
    if isinstance(node, LoraSourceNode):
        config = cast(CompiledLoraSourceConfig, compiled_node.config)
        path = config.path
        identity = _digest(
            {
                "sha256": sha256_file(path),
                "weight": config.weight,
            }
        )
        return {
            "lora": LoraArtifact(
                path=path.as_posix(),
                weight=config.weight,
                identity=identity,
            )
        }
    return None


def _node_signature(
    compiled_node: CompiledNode,
    inputs: Mapping[str, Sequence[WorkflowArtifact]],
    source_outputs: Mapping[str, WorkflowArtifact] | None,
    provenance: NodeExecutionProvenance,
) -> str:
    node = compiled_node.node
    return build_node_signature(
        node_kind=node.kind,
        execution_config=execution_config_payload(compiled_node.config),
        inputs={
            port: tuple(
                NodeInputIdentity(
                    artifact_type=artifact.type,
                    identity=artifact.identity,
                )
                for artifact in artifacts
            )
            for port, artifacts in inputs.items()
        },
        source_outputs=(
            {
                port: NodeInputIdentity(
                    artifact_type=artifact.type,
                    identity=artifact.identity,
                )
                for port, artifact in source_outputs.items()
            }
            if source_outputs is not None
            else None
        ),
        provenance=provenance,
    )


def _write_node_manifest(
    run_dir: Path,
    node: WorkflowNode,
    signature: str,
    outputs: Mapping[str, WorkflowArtifact],
) -> Path:
    final_outputs = dict(outputs)
    _validate_output_contract(node, final_outputs)
    manifest_path = run_dir / "nodes" / node.id / "result.json"
    manifest = NodeResultManifest(
        node_id=node.id,
        node_kind=node.kind,
        signature=signature,
        outputs=final_outputs,
    )
    atomic_write_json(
        manifest_path,
        manifest.model_dump(mode="json"),
    )
    return manifest_path


def _outcome_from_cache_hit(
    node: WorkflowNode,
    hit: NodeCacheHit,
) -> _NodeOutcome:
    _validate_output_contract(node, hit.outputs)
    return _NodeOutcome(
        outputs=hit.outputs,
    )


def _execute_node(
    pending: _PendingNodeExecution,
    *,
    node_cache: WorkflowNodeCache,
    node_progress_sink: NodeProgressSink | None,
) -> _NodeOutcome:
    with node_cache.begin(
        pending.signature,
        node_kind=pending.node.kind,
        provenance=pending.provenance,
    ) as write:
        cache_hit = write.publish(
            _run_generated_node(
                pending,
                write.output_dir,
                node_progress_sink=node_progress_sink,
            )
        )
    return _outcome_from_cache_hit(pending.node, cache_hit)


def _execute_image_postprocess_group(
    group: Sequence[_PendingNodeExecution],
    *,
    node_cache: WorkflowNodeCache,
    node_progress_sink: NodeProgressSink | None,
) -> tuple[_NodeOutcome, ...]:
    first_config = cast(
        CompiledPostprocessConfig,
        group[0].compiled_node.config,
    )
    output_names = tuple(f"{item.node.id}.png" for item in group)
    sources = tuple(
        _one_artifact(item.inputs, "image", ImageArtifact)
        for item in group
    )
    with ExitStack() as stack:
        writes = tuple(
            stack.enter_context(
                node_cache.begin(
                    item.signature,
                    node_kind=item.node.kind,
                    provenance=item.provenance,
                )
            )
            for item in group
        )
        batch_output = writes[0].output_dir / "batch"
        with _node_progress(
            tuple(item.node.id for item in group),
            node_progress_sink,
        ) as node_progress:
            result = _postprocess_images(
                first_config,
                tuple(Path(source.path) for source in sources),
                batch_output,
                output_names=output_names,
                progress=node_progress,
            )
        staged_outputs = []
        for item, write, generated in zip(
            group,
            writes,
            result.outputs,
            strict=True,
        ):
            output = write.output_dir / "image.png"
            generated.replace(output)
            staged_outputs.append((item, write, output))
        outcomes = []
        for item, write, output in staged_outputs:
            cache_hit = write.publish(
                {
                    "image": GeneratedNodeOutput(
                        artifact_type=ArtifactType.IMAGE,
                        paths=(output,),
                    )
                }
            )
            outcomes.append(
                _outcome_from_cache_hit(item.node, cache_hit)
            )
        return tuple(outcomes)


def _run_generated_node(
    pending: _PendingNodeExecution,
    staging: Path,
    *,
    node_progress_sink: NodeProgressSink | None,
) -> dict[str, GeneratedNodeOutput]:
    node = pending.node
    inputs = pending.inputs
    with _node_progress((node.id,), node_progress_sink) as node_progress:
        if isinstance(node, ImageEditNode):
            plan = cast(
                _ResolvedImageEditPlan,
                pending.image_edit_plan,
            )
            result = run_image_edit(
                ImageEditRequest(
                    backend=plan.backend,
                    prompt=plan.prompt,
                    output_dir=staging,
                    images=plan.references,
                    seeds=(plan.seed,),
                    width=plan.width,
                    height=plan.height,
                    steps=plan.steps,
                    guidance=plan.guidance,
                    strength=plan.strength,
                    sampler=plan.sampler,
                    scheduler=plan.scheduler,
                    loras=tuple(lora.path for lora in plan.loras),
                    lora_weights=tuple(lora.weight for lora in plan.loras),
                ),
                progress=node_progress,
            )
            if len(result.outputs) != 1:
                raise WorkflowExecutionError(
                    f"node {node.id!r} produced {len(result.outputs)} images"
                )
            return {
                "image": GeneratedNodeOutput(
                    artifact_type=ArtifactType.IMAGE,
                    paths=(
                        _require_file(result.outputs[0].path, node),
                    ),
                )
            }

        if isinstance(node, ImagePostprocessNode):
            source = _one_artifact(inputs, "image", ImageArtifact)
            result = _postprocess_images(
                cast(
                    CompiledPostprocessConfig,
                    pending.compiled_node.config,
                ),
                (Path(source.path),),
                staging / "image",
                output_names=("image.png",),
                progress=node_progress,
            )
            return {
                "image": GeneratedNodeOutput(
                    artifact_type=ArtifactType.IMAGE,
                    paths=(_require_file(result.outputs[0], node),),
                )
            }

        if isinstance(node, AnimeGenI2VNode):
            start = _one_artifact(inputs, "start", ImageArtifact)
            end_artifacts = inputs.get("end", ())
            end = (
                cast(ImageArtifact, end_artifacts[0])
                if end_artifacts
                else None
            )
            config = cast(
                CompiledAnimeGenConfig,
                pending.compiled_node.config,
            )
            settings = config.settings
            output = staging / "video.mp4"
            result = generate_animegen_i2v(
                prompt=config.prompt,
                image=Path(start.path),
                last_image=Path(end.path) if end is not None else None,
                output=output,
                frames=settings.frames,
                fps=settings.fps,
                sampling=settings.sampling,
                steps=settings.steps,
                precision=settings.precision,
                seed=config.seed,
                progress=node_progress,
            )
            return {
                "video": GeneratedNodeOutput(
                    artifact_type=ArtifactType.VIDEO,
                    paths=(_require_file(result.output, node),),
                )
            }

        if isinstance(node, VideoContactSheetNode):
            video = _one_artifact(inputs, "video", VideoArtifact)
            node_progress.phase("create video contact sheet")
            output = create_video_contact_sheet(
                Path(video.path),
                staging / "contact-sheet.png",
            )
            node_progress.phase("video contact sheet completed")
            return {
                "image": GeneratedNodeOutput(
                    artifact_type=ArtifactType.IMAGE,
                    paths=(_require_file(output, node),),
                )
            }

        if isinstance(node, ExtractVideoFramesNode):
            video = _one_artifact(inputs, "video", VideoArtifact)
            result = extract_video_frames(
                Path(video.path),
                staging / "frames",
                progress=node_progress,
            )
            paths = tuple(
                result.output_dir / f"frame-{index:06d}.png"
                for index in range(result.frames)
            )
            return {
                "images": GeneratedNodeOutput(
                    artifact_type=ArtifactType.IMAGE_SEQUENCE,
                    paths=paths,
                )
            }

        if isinstance(node, FramePostprocessNode):
            source = _one_artifact(
                inputs,
                "images",
                ImageSequenceArtifact,
            )
            source_paths = tuple(Path(path) for path in source.paths)
            result = _postprocess_images(
                cast(
                    CompiledPostprocessConfig,
                    pending.compiled_node.config,
                ),
                source_paths,
                staging / "frames",
                output_names=tuple(path.name for path in source_paths),
                progress=node_progress,
            )
            return {
                "images": GeneratedNodeOutput(
                    artifact_type=ArtifactType.IMAGE_SEQUENCE,
                    paths=result.outputs,
                )
            }

    raise WorkflowExecutionError(
        f"unsupported generated workflow node: {node.kind}"
    )


def _image_edit_reference_paths(
    node: ImageEditNode,
    inputs: Mapping[str, Sequence[WorkflowArtifact]],
) -> tuple[Path, ...]:
    references: list[Path] = []
    for artifact in inputs["references"]:
        if isinstance(artifact, ImageArtifact):
            references.append(Path(artifact.path))
        elif isinstance(artifact, ReferencePackArtifact):
            references.extend(Path(path) for path in artifact.references)
        else:
            raise WorkflowExecutionError(
                f"node {node.id!r} received {artifact.type} as an image reference"
            )
    return tuple(references)


def _image_edit_loras(
    node: ImageEditNode,
    inputs: Mapping[str, Sequence[WorkflowArtifact]],
) -> tuple[LoraArtifact, ...]:
    artifacts = inputs.get("loras", ())
    invalid = next(
        (
            artifact
            for artifact in artifacts
            if not isinstance(artifact, LoraArtifact)
        ),
        None,
    )
    if invalid is not None:
        raise WorkflowExecutionError(
            f"node {node.id!r} received {invalid.type} as a LoRA"
        )
    return cast(tuple[LoraArtifact, ...], tuple(artifacts))


def _postprocess_images(
    config: CompiledPostprocessConfig,
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    output_names: Sequence[str] | None = None,
    progress: StatusReporter,
) -> ImageBatchPostprocessResult:
    if isinstance(config, CompiledVosrLongSideConfig):
        return postprocess_image_batch(
            inputs,
            output_dir,
            model=config.model,
            progress=progress,
            output_names=output_names,
            long_side=config.long_side,
            infer_steps=config.infer_steps,
            cfg_scale=config.cfg_scale,
            weak_cond_strength_aelq=config.weak_cond_strength_aelq,
            align_method=config.align_method,
            tile_size=config.tile_size,
            seed=config.seed,
        )
    if isinstance(config, CompiledVosrScaleConfig):
        return postprocess_image_batch(
            inputs,
            output_dir,
            model=config.model,
            progress=progress,
            output_names=output_names,
            scale=config.scale,
            infer_steps=config.infer_steps,
            cfg_scale=config.cfg_scale,
            weak_cond_strength_aelq=config.weak_cond_strength_aelq,
            align_method=config.align_method,
            tile_size=config.tile_size,
            seed=config.seed,
        )
    if isinstance(config, CompiledIllustrationUpscaleConfig):
        return postprocess_image_batch(
            inputs,
            output_dir,
            model=config.model,
            progress=progress,
            output_names=output_names,
            long_side=config.long_side,
        )
    if isinstance(config, CompiledWuPixelizationConfig):
        return postprocess_image_batch(
            inputs,
            output_dir,
            model=config.model,
            progress=progress,
            output_names=output_names,
            cell_size=config.cell_size,
        )
    if isinstance(config, CompiledPixelArtFixerConfig):
        return postprocess_image_batch(
            inputs,
            output_dir,
            model=config.model,
            progress=progress,
            output_names=output_names,
            mode=config.mode,
            low_memory=config.low_memory,
            force_step=config.force_step,
        )
    raise WorkflowExecutionError(
        f"unsupported postprocessing config: {type(config).__name__}"
    )


def _node_progress(
    node_ids: Sequence[str],
    sink: NodeProgressSink | None,
) -> StatusReporter:
    if sink is None:
        return SILENT_STATUS

    def forward(payload: dict[str, object]) -> None:
        for node_id in node_ids:
            sink(node_id, payload)

    return RuntimeStatus.callback(
        interval_seconds=DEFAULT_PROGRESS_INTERVAL_SECONDS,
        callback=forward,
        telemetry=SystemTelemetrySampler(),
    )


ArtifactT = TypeVar("ArtifactT", bound=RuntimeModel)


def _one_artifact(
    inputs: Mapping[str, Sequence[WorkflowArtifact]],
    port: str,
    artifact_class: type[ArtifactT],
) -> ArtifactT:
    artifacts = inputs[port]
    if len(artifacts) != 1:
        raise WorkflowExecutionError(
            f"workflow input {port!r} does not contain one "
            f"{artifact_class.__name__}"
        )
    artifact = artifacts[0]
    if not isinstance(artifact, artifact_class):
        raise WorkflowExecutionError(
            f"workflow input {port!r} does not contain one "
            f"{artifact_class.__name__}"
        )
    return artifact


def _validate_output_contract(
    node: WorkflowNode,
    outputs: Mapping[str, WorkflowArtifact],
) -> None:
    definitions = node_definition(node.kind).outputs
    expected_ports = {port.name for port in definitions}
    if set(outputs) != expected_ports:
        raise WorkflowExecutionError(
            f"node {node.id!r} produced ports {sorted(outputs)}; "
            f"expected {sorted(expected_ports)}"
        )
    for port in definitions:
        artifact = outputs[port.name]
        if artifact.type not in port.artifact_types:
            accepted = ", ".join(port.artifact_types)
            raise WorkflowExecutionError(
                f"node {node.id!r} output {port.name!r} produced "
                f"{artifact.type}; expected {accepted}"
            )


def _require_file(path: Path, node: WorkflowNode) -> Path:
    if not path.is_file():
        raise WorkflowExecutionError(
            f"node {node.id!r} completed without its declared output: {path}"
        )
    return path


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _emit(
    sink: WorkflowEventSink | None,
    *,
    node_id: str,
    node_kind: NodeKind,
    status: Literal[
        "running",
        "completed",
        "reused",
        "failed",
        "interrupted",
    ],
    message: str | None = None,
) -> None:
    if sink is None:
        return
    payload: dict[str, object] = {
        "node_id": node_id,
        "node_kind": node_kind,
        "status": status,
    }
    if message is not None:
        payload["message"] = message
    sink(payload)
