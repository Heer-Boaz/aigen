from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict

from aigen.manifest_io import atomic_write_json, sha256_file
from aigen.progress import JSON_PROGRESS_PREFIX, StatusReporter
from aigen.runtime_profiles import PROJECT_ROOT
from aigen.generation.video_postprocess import probe_video
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
    NodeCacheWrite,
    NodeExecutionProvenance,
    NodeInputIdentity,
    WorkflowNodeCache,
    build_node_signature,
)
from aigen.workflow_provenance import workflow_node_provenance
from aigen.generation.image_edit_batch import (
    ImageEditBatchCase,
    ImageEditBatchLora,
    ImageEditBatchRequest,
    ImageEditBatchResult,
)
from aigen.generation.image_edit import (
    FLUX2_KLEIN_BACKEND,
    QWEN_2511_BASE_BACKEND,
    QWEN_2511_LIGHTNING_BACKEND,
    resolve_image_edit_canvas_size,
)


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
    source_outputs: Mapping[str, WorkflowArtifact] | None
    signature: str
    manifest_path: Path
    provenance: NodeExecutionProvenance
    image_edit_plan: _ResolvedImageEditPlan | None = None


@dataclass(frozen=True)
class _NodeStaging:
    pending: _PendingNodeExecution
    output_dir: Path
    staging: Path
    log_path: Path
    cache_write: NodeCacheWrite | None = None


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
    run_dir = workflow_root / "runs" / workflow_digest
    node_cache = WorkflowNodeCache(workflow_root / "cache")
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = _save_snapshot(graph, run_dir)
    outputs_by_node: dict[str, dict[str, WorkflowArtifact]] = {}
    node_manifests: dict[str, Path] = {}
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
        for layer in workflow.execution_layers:
            pending: list[_PendingNodeExecution] = []
            for node_id in layer:
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
                if source_outputs is not None:
                    manifest_path = _node_manifest_path(run_dir, node, signature)
                    manifest = _load_reusable_manifest(
                        manifest_path,
                        node,
                        signature,
                    )
                else:
                    cache_hit = node_cache.lookup(
                        signature,
                        node_kind=node.kind,
                        provenance=provenance,
                    )
                    manifest_path = (
                        node_cache.entry_dir(signature) / "result.json"
                    )
                    manifest = (
                        _manifest_from_cache_hit(node, cache_hit)
                        if cache_hit is not None
                        else None
                    )
                if manifest is not None:
                    outputs_by_node[node_id] = manifest.outputs
                    node_manifests[node_id] = manifest_path
                    _emit(
                        event_sink,
                        node_id=node_id,
                        node_kind=node.kind,
                        status="reused",
                    )
                    progress.step(f"reused {node.title}")
                    continue
                pending.append(
                    _PendingNodeExecution(
                        compiled_node=compiled_node,
                        node=node,
                        inputs=inputs,
                        source_outputs=source_outputs,
                        signature=signature,
                        manifest_path=manifest_path,
                        provenance=provenance,
                        image_edit_plan=(
                            _resolve_image_edit_plan(compiled_node, inputs)
                            if isinstance(node, ImageEditNode)
                            else None
                        ),
                    )
                )

            for group in _execution_groups(pending):
                active_node_ids = tuple(item.node.id for item in group)
                for item in group:
                    _emit(
                        event_sink,
                        node_id=item.node.id,
                        node_kind=item.node.kind,
                        status="running",
                    )
                manifests = _execute_group(
                    group,
                    node_cache=node_cache,
                    node_progress_sink=node_progress_sink,
                )
                for item, manifest in zip(group, manifests, strict=True):
                    node_id = item.node.id
                    outputs_by_node[node_id] = manifest.outputs
                    node_manifests[node_id] = item.manifest_path
                    _emit(
                        event_sink,
                        node_id=node_id,
                        node_kind=item.node.kind,
                        status="completed",
                    )
                    progress.step(f"completed {item.node.title}")
                _write_run_state(
                    run_dir,
                    graph,
                    workflow_digest,
                    snapshot_path,
                    status="running",
                    node_manifests=node_manifests,
                )
                active_node_ids = ()
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


def _execution_groups(
    pending: Sequence[_PendingNodeExecution],
) -> tuple[tuple[_PendingNodeExecution, ...], ...]:
    grouped: dict[tuple[str, str], list[_PendingNodeExecution]] = {}
    order: list[tuple[str, str]] = []
    for item in pending:
        if _is_batchable_image_edit(item.image_edit_plan):
            key = (
                "image-edit",
                _image_edit_batch_key(item.image_edit_plan),
            )
        elif isinstance(item.node, ImagePostprocessNode):
            key = (
                "image-postprocess",
                _digest(
                    execution_config_payload(item.compiled_node.config)
                ),
            )
        else:
            key = ("node", item.node.id)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(item)
    return tuple(tuple(grouped[key]) for key in order)


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
) -> tuple[NodeResultManifest, ...]:
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
) -> tuple[NodeResultManifest, ...]:
    created_contexts: list[_NodeStaging] = []
    try:
        for item in group:
            created_contexts.append(
                _start_node_staging(
                    item,
                    item.manifest_path.parent,
                    node_cache=node_cache,
                )
            )
    except Exception:
        for context in created_contexts:
            _preserve_failed_staging(context)
        raise
    contexts = tuple(created_contexts)
    plans = tuple(
        cast(_ResolvedImageEditPlan, item.image_edit_plan)
        for item in group
    )
    request_path = contexts[0].staging / "batch-request.json"
    response_path = contexts[0].staging / "batch-response.json"
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
                output_path=context.staging / "image" / "image.png",
            )
            for item, plan, context in zip(
                group,
                plans,
                contexts,
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
    atomic_write_json(
        request_path,
        request.model_dump(mode="json"),
    )

    def fan_out_progress(
        _node_id: str,
        payload: dict[str, object],
    ) -> None:
        if node_progress_sink is None:
            return
        for item in group:
            node_progress_sink(item.node.id, payload)

    command = [
        sys.executable,
        "-m",
        "aigen.generation.image_edit_batch_worker",
        request_path.as_posix(),
        response_path.as_posix(),
    ]
    try:
        _run_subcommand(
            command,
            contexts[0].log_path,
            group[0].node.id,
            fan_out_progress,
        )
        result = ImageEditBatchResult.model_validate_json(
            response_path.read_text(encoding="utf-8")
        )
        outputs_by_case = {
            output.case_id: output
            for output in result.outputs
        }
        staged_outputs = []
        for item, context in zip(group, contexts, strict=True):
            output = outputs_by_case[item.node.id]
            expected = context.staging / "image" / "image.png"
            if output.path.resolve() != expected.resolve():
                raise WorkflowExecutionError(
                    f"image-edit batch returned the wrong output for "
                    f"{item.node.id!r}: {output.path}"
                )
            staged_outputs.append(
                {
                    "image": GeneratedNodeOutput(
                        artifact_type=ArtifactType.IMAGE,
                        paths=(_require_file(expected, item.node),),
                    )
                }
            )
        return tuple(
            _publish_generated_node_staging(context, outputs)
            for context, outputs in zip(contexts, staged_outputs, strict=True)
        )
    except WorkflowInterrupted:
        for context in contexts:
            _preserve_failed_staging(context, label="interrupted")
        raise
    except Exception:
        for context in contexts:
            _preserve_failed_staging(context, label="failed")
        raise


def format_workflow_event(payload: dict[str, object]) -> str:
    return WORKFLOW_EVENT_PREFIX + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


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


def _node_manifest_path(
    run_dir: Path,
    node: WorkflowNode,
    signature: str,
) -> Path:
    return run_dir / "nodes" / node.id / signature / "result.json"


def _load_reusable_manifest(
    manifest_path: Path,
    node: WorkflowNode,
    signature: str,
) -> NodeResultManifest | None:
    if not manifest_path.is_file():
        return None
    try:
        manifest = NodeResultManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise WorkflowExecutionError(
            f"invalid workflow node manifest: {manifest_path}"
        ) from error
    if manifest.signature != signature:
        return None
    if manifest.node_id != node.id or manifest.node_kind != node.kind:
        raise WorkflowExecutionError(
            f"cached workflow manifest does not belong to node {node.id!r}: "
            f"{manifest_path}"
        )
    _validate_output_contract(node, manifest.outputs)
    missing = next(
        (
            path
            for artifact in manifest.outputs.values()
            for path in _artifact_paths(artifact)
            if not path.exists()
        ),
        None,
    )
    if missing is not None:
        raise WorkflowExecutionError(
            f"cached workflow artifact is missing: {missing}"
        )
    return manifest


def _manifest_from_cache_hit(
    node: WorkflowNode,
    hit: NodeCacheHit,
) -> NodeResultManifest:
    outputs = dict(hit.outputs)
    _validate_output_contract(node, outputs)
    return NodeResultManifest(
        node_id=node.id,
        node_kind=node.kind,
        signature=hit.signature,
        outputs=outputs,
    )


def _execute_node(
    pending: _PendingNodeExecution,
    *,
    node_cache: WorkflowNodeCache,
    node_progress_sink: NodeProgressSink | None,
) -> NodeResultManifest:
    context = _start_node_staging(
        pending,
        pending.manifest_path.parent,
        node_cache=(
            node_cache
            if pending.source_outputs is None
            else None
        ),
    )
    try:
        if pending.source_outputs is not None:
            return _publish_source_node_staging(
                context,
                pending.source_outputs,
            )
        return _publish_generated_node_staging(
            context,
            _run_generated_node(
                pending,
                context.staging,
                context.log_path,
                node_progress_sink=node_progress_sink,
            ),
        )
    except WorkflowInterrupted:
        _preserve_failed_staging(context, label="interrupted")
        raise
    except Exception:
        _preserve_failed_staging(context, label="failed")
        raise


def _start_node_staging(
    pending: _PendingNodeExecution,
    output_dir: Path,
    *,
    node_cache: WorkflowNodeCache | None,
) -> _NodeStaging:
    if node_cache is not None:
        cache_write = node_cache.begin(
            pending.signature,
            node_kind=pending.node.kind,
            provenance=pending.provenance,
        )
        cache_write.__enter__()
        return _NodeStaging(
            pending=pending,
            output_dir=node_cache.entry_dir(pending.signature),
            staging=cache_write.output_dir,
            log_path=cache_write.staging_dir / "node.log",
            cache_write=cache_write,
        )
    if output_dir.exists():
        raise WorkflowExecutionError(
            f"workflow node output is incomplete or corrupt: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.",
        )
    )
    return _NodeStaging(
        pending=pending,
        output_dir=output_dir,
        staging=staging,
        log_path=staging / "node.log",
    )


def _publish_generated_node_staging(
    context: _NodeStaging,
    staged_outputs: Mapping[str, GeneratedNodeOutput],
) -> NodeResultManifest:
    pending = context.pending
    cache_write = cast(NodeCacheWrite, context.cache_write)
    cache_hit = cache_write.publish(staged_outputs)
    cache_write.__exit__(None, None, None)
    return _manifest_from_cache_hit(pending.node, cache_hit)


def _publish_source_node_staging(
    context: _NodeStaging,
    outputs: Mapping[str, WorkflowArtifact],
) -> NodeResultManifest:
    pending = context.pending
    final_outputs = dict(outputs)
    _validate_output_contract(pending.node, final_outputs)
    manifest = NodeResultManifest(
        node_id=pending.node.id,
        node_kind=pending.node.kind,
        signature=pending.signature,
        outputs=final_outputs,
    )
    atomic_write_json(
        context.staging / "result.json",
        manifest.model_dump(mode="json"),
    )
    context.staging.replace(context.output_dir)
    return manifest


def _preserve_failed_staging(
    context: _NodeStaging,
    *,
    label: Literal["failed", "interrupted"] = "failed",
) -> None:
    if context.cache_write is not None:
        context.cache_write.__exit__(None, None, None)
        return
    if not context.staging.exists():
        return
    failed_dir = context.output_dir.parent / (
        f"{label}-{context.pending.signature[:12]}-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    context.staging.replace(failed_dir)


def _execute_image_postprocess_group(
    group: Sequence[_PendingNodeExecution],
    *,
    node_cache: WorkflowNodeCache,
    node_progress_sink: NodeProgressSink | None,
) -> tuple[NodeResultManifest, ...]:
    created_contexts: list[_NodeStaging] = []
    try:
        for item in group:
            created_contexts.append(
                _start_node_staging(
                    item,
                    item.manifest_path.parent,
                    node_cache=node_cache,
                )
            )
    except Exception:
        for context in created_contexts:
            _preserve_failed_staging(context)
        raise
    contexts = tuple(created_contexts)
    first_config = cast(
        CompiledPostprocessConfig,
        group[0].compiled_node.config,
    )
    batch_output = contexts[0].staging / "batch"
    output_names = tuple(f"{item.node.id}.png" for item in group)
    sources = tuple(
        _one_artifact(item.inputs, "image", ImageArtifact)
        for item in group
    )
    command = _batch_postprocess_command(
        first_config,
        tuple(Path(source.path) for source in sources),
        batch_output,
        output_names=output_names,
    )

    def fan_out_progress(
        _node_id: str,
        payload: dict[str, object],
    ) -> None:
        if node_progress_sink is None:
            return
        for item in group:
            node_progress_sink(item.node.id, payload)

    try:
        _run_subcommand(
            command,
            contexts[0].log_path,
            group[0].node.id,
            fan_out_progress,
        )
        staged_outputs: list[dict[str, GeneratedNodeOutput]] = []
        for context, output_name in zip(contexts, output_names, strict=True):
            generated = _require_file(batch_output / output_name, context.pending.node)
            output = context.staging / "image" / "image.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            generated.replace(output)
            staged_outputs.append(
                {
                    "image": GeneratedNodeOutput(
                        artifact_type=ArtifactType.IMAGE,
                        paths=(output,),
                    )
                }
            )
        return tuple(
            _publish_generated_node_staging(context, outputs)
            for context, outputs in zip(contexts, staged_outputs, strict=True)
        )
    except WorkflowInterrupted:
        for context in contexts:
            _preserve_failed_staging(context, label="interrupted")
        raise
    except Exception:
        for context in contexts:
            _preserve_failed_staging(context, label="failed")
        raise


def _run_generated_node(
    pending: _PendingNodeExecution,
    staging: Path,
    log_path: Path,
    *,
    node_progress_sink: NodeProgressSink | None,
) -> dict[str, GeneratedNodeOutput]:
    node = pending.node
    inputs = pending.inputs
    if isinstance(node, ImageEditNode):
        output_dir = staging / "image"
        command = _image_edit_command(
            cast(CompiledImageEditConfig, pending.compiled_node.config),
            node,
            inputs,
            output_dir,
        )
        _run_subcommand(command, log_path, node.id, node_progress_sink)
        output = _require_file(output_dir / "image.png", node)
        return {
            "image": GeneratedNodeOutput(
                artifact_type=ArtifactType.IMAGE,
                paths=(output,),
            )
        }

    if isinstance(node, ImagePostprocessNode):
        source = _one_artifact(inputs, "image", ImageArtifact)
        output_dir = staging / "image"
        command = _batch_postprocess_command(
            cast(CompiledPostprocessConfig, pending.compiled_node.config),
            (Path(source.path),),
            output_dir,
        )
        _run_subcommand(command, log_path, node.id, node_progress_sink)
        output = _require_file(output_dir / Path(source.path).name, node)
        return {
            "image": GeneratedNodeOutput(
                artifact_type=ArtifactType.IMAGE,
                paths=(output,),
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
        output = staging / "video.mp4"
        command = _animegen_command(
            cast(CompiledAnimeGenConfig, pending.compiled_node.config),
            start,
            end,
            output,
        )
        _run_subcommand(command, log_path, node.id, node_progress_sink)
        return {
            "video": GeneratedNodeOutput(
                artifact_type=ArtifactType.VIDEO,
                paths=(_require_file(output, node),),
            )
        }

    if isinstance(node, VideoContactSheetNode):
        video = _one_artifact(inputs, "video", VideoArtifact)
        output = staging / "contact-sheet.png"
        command = [
            sys.executable,
            "-m",
            "aigen.cli",
            "video-postprocess",
            "contact-sheet",
            "--input",
            video.path,
            "--output",
            output.as_posix(),
        ]
        _run_subcommand(command, log_path, node.id, node_progress_sink)
        return {
            "image": GeneratedNodeOutput(
                artifact_type=ArtifactType.IMAGE,
                paths=(_require_file(output, node),),
            )
        }

    if isinstance(node, ExtractVideoFramesNode):
        video = _one_artifact(inputs, "video", VideoArtifact)
        frame_count = probe_video(Path(video.path)).frames
        output_dir = staging / "frames"
        command = [
            sys.executable,
            "-m",
            "aigen.cli",
            "video-postprocess",
            "extract-frames",
            "--input",
            video.path,
            "--output-dir",
            output_dir.as_posix(),
        ]
        _run_subcommand(command, log_path, node.id, node_progress_sink)
        paths = tuple(
            output_dir / f"frame-{index:06d}.png"
            for index in range(frame_count)
        )
        missing = next((path for path in paths if not path.is_file()), None)
        if missing is not None:
            raise WorkflowExecutionError(
                f"node {node.id!r} did not extract frame: {missing}"
            )
        return {
            "images": GeneratedNodeOutput(
                artifact_type=ArtifactType.IMAGE_SEQUENCE,
                paths=paths,
            )
        }

    if isinstance(node, FramePostprocessNode):
        source = _one_artifact(inputs, "images", ImageSequenceArtifact)
        output_dir = staging / "frames"
        command = _batch_postprocess_command(
            cast(CompiledPostprocessConfig, pending.compiled_node.config),
            tuple(Path(path) for path in source.paths),
            output_dir,
        )
        _run_subcommand(command, log_path, node.id, node_progress_sink)
        paths = tuple(output_dir / Path(path).name for path in source.paths)
        missing = next((path for path in paths if not path.is_file()), None)
        if missing is not None:
            raise WorkflowExecutionError(
                f"node {node.id!r} did not produce frame: {missing}"
            )
        return {
            "images": GeneratedNodeOutput(
                artifact_type=ArtifactType.IMAGE_SEQUENCE,
                paths=paths,
            )
        }

    raise WorkflowExecutionError(f"unsupported generated workflow node: {node.kind}")


def _image_edit_command(
    config: CompiledImageEditConfig,
    node: ImageEditNode,
    inputs: Mapping[str, Sequence[WorkflowArtifact]],
    output_dir: Path,
) -> list[str]:
    settings = config.settings
    command = [
        sys.executable,
        "-m",
        "aigen.cli",
        "image-edit",
        "--backend",
        config.backend,
        "--prompt",
        config.prompt,
        "--output-dir",
        output_dir.as_posix(),
        "--seed",
        str(config.seed),
    ]
    for reference in _image_edit_reference_paths(node, inputs):
        command.extend(("--image", reference.as_posix()))
    for artifact in _image_edit_loras(node, inputs):
        command.extend(
            (
                "--lora",
                artifact.path,
                "--lora-weight",
                str(artifact.weight),
            )
        )
    if settings.aspect_ratio is not None:
        command.extend(
            (
                "--aspect-ratio",
                f"{settings.aspect_ratio[0]}:{settings.aspect_ratio[1]}",
            )
        )
    if settings.width is not None:
        command.extend(("--width", str(settings.width)))
    if settings.height is not None:
        command.extend(("--height", str(settings.height)))
    for flag, value in (
        ("--steps", settings.steps),
        ("--guidance", settings.guidance),
        ("--strength", settings.strength),
        ("--sampler", settings.sampler),
        ("--scheduler", settings.scheduler),
    ):
        if value is not None and value != "":
            command.extend((flag, str(value)))
    return command


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


def _animegen_command(
    config: CompiledAnimeGenConfig,
    start: ImageArtifact,
    end: ImageArtifact | None,
    output: Path,
) -> list[str]:
    settings = config.settings
    command = [
        sys.executable,
        "-m",
        "aigen.cli",
        "animegen-i2v",
        "--image",
        start.path,
        "--prompt",
        config.prompt,
        "--output",
        output.as_posix(),
        "--frames",
        str(settings.frames),
        "--fps",
        str(settings.fps),
        "--sampling",
        settings.sampling,
        "--steps",
        str(settings.steps),
        "--precision",
        settings.precision,
        "--seed",
        str(config.seed),
    ]
    if end is not None:
        command.extend(("--last-image", end.path))
    return command


def _batch_postprocess_command(
    config: CompiledPostprocessConfig,
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    output_names: Sequence[str] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "aigen.cli",
        "image-postprocess-batch",
        "--model",
        config.model,
        "--output-dir",
        output_dir.as_posix(),
    ]
    for input_path in inputs:
        command.extend(("--input", input_path.as_posix()))
    for output_name in output_names or ():
        command.extend(("--output-name", output_name))
    if isinstance(
        config,
        (CompiledVosrLongSideConfig, CompiledVosrScaleConfig),
    ):
        if isinstance(config, CompiledVosrLongSideConfig):
            command.extend(("--long-side", str(config.long_side)))
        else:
            command.extend(("--scale", str(config.scale)))
        for name, value in (
            ("infer-steps", config.infer_steps),
            ("cfg-scale", config.cfg_scale),
            ("weak-cond-strength-aelq", config.weak_cond_strength_aelq),
            ("align-method", config.align_method),
            ("tile-size", config.tile_size),
            ("seed", config.seed),
        ):
            command.extend((f"--{name}", str(value)))
    elif isinstance(config, CompiledIllustrationUpscaleConfig):
        if config.long_side is not None:
            command.extend(("--long-side", str(config.long_side)))
    elif isinstance(config, CompiledWuPixelizationConfig):
        command.extend(("--cell-size", str(config.cell_size)))
    elif isinstance(config, CompiledPixelArtFixerConfig):
        command.extend(("--mode", config.mode))
        if config.low_memory:
            command.append("--low-memory")
        if config.force_step is not None:
            command.extend(("--force-step", str(config.force_step)))
    return command


def _run_subcommand(
    command: Sequence[str],
    log_path: Path,
    node_id: str,
    node_progress_sink: NodeProgressSink | None,
) -> None:
    environment = os.environ.copy()
    environment["AIGEN_PROGRESS"] = "json"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    tail: deque[str] = deque(maxlen=200)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                if line.startswith(JSON_PROGRESS_PREFIX):
                    if node_progress_sink is not None:
                        node_progress_sink(
                            node_id,
                            json.loads(line[len(JSON_PROGRESS_PREFIX) :]),
                        )
                elif line.strip():
                    tail.append(line.rstrip())
        returncode = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    if returncode != 0:
        detail = "\n".join(tail)
        raise WorkflowExecutionError(
            detail
            or f"workflow node command exited with code {returncode}: "
            f"{' '.join(command)}"
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


def _artifact_paths(artifact: WorkflowArtifact) -> tuple[Path, ...]:
    if isinstance(artifact, ImageSequenceArtifact):
        return tuple(Path(path) for path in artifact.paths)
    if isinstance(artifact, ReferencePackArtifact):
        return (
            Path(artifact.path),
            *(Path(path) for path in artifact.references),
        )
    return (Path(artifact.path),)


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
