from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from graphlib import TopologicalSorter
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from PIL import Image

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
    ImageEditBatchOutput,
    run_image_edit_batch,
)
from aigen.manifest_io import atomic_write_json, read_json, sha256_file
from aigen.progress import (
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    SILENT_STATUS,
    RuntimeStatus,
    StatusReporter,
)
from aigen.system_telemetry import SystemTelemetrySampler
from aigen.workflow_compilation import (
    CompiledImageEditConfig,
    CompiledCharacterEditConfig,
    CompiledCharacterRefineConfig,
    CompiledImageSourceConfig,
    CompiledImageSelectionConfig,
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
    CompiledVideoSourceConfig, CompiledAudioSourceConfig,
)
from aigen.workflow_artifacts import (
    ImageArtifact,
    ImageCollectionArtifact,
    ImageSequenceArtifact,
    LoraArtifact,
    ReferencePackArtifact,
    VideoArtifact,
    WorkflowArtifact,
    AudioArtifact, KeyframeArtifact, one_artifact, with_sequence_timing,
)
from aigen.workflow_document_io import save_workflow_document
from aigen.workflow_graph import (
    ArtifactType,
    ExtractVideoFramesNode,
    FramePostprocessNode,
    ImageEditNode,
    CharacterEditNode,
    BindMaskNode, SamSegmentNode, CharacterRefineNode,
    ImageCollectionNode,
    ImageSelectionNode,
    ImageResultReference,
    ImagePostprocessNode,
    ImageSourceNode,
    LoraSourceNode,
    NodeKind,
    ReferencePackNode,
    WorkflowGraph,
    WorkflowNode,
    WorkflowConnection,
    node_definition,
    VideoSourceNode, AudioSourceNode, PositionedKeyframeNode,
)
from aigen.workflow_cache import (
    GeneratedNodeOutput,
    NodeCacheHit,
    NodeExecutionProvenance,
    NodeExecutionDetails,
    NodeCacheWrite,
    NodeInputIdentity,
    WorkflowNodeCache,
    build_node_signature,
)
from aigen.workflow_provenance import workflow_node_provenance
from aigen.workflow_character_execution import execute_character_node
from aigen.workflow_results import NodeResultManifest, load_node_result
from aigen.workflow_video_execution import (
    VIDEO_EXECUTION_NODES, VIDEO_SEED_SWEEP_NODES, execute_video_node, execute_video_seed_sweep,
)
from aigen.media_timing import load_audio_track, video_audio_track


WORKFLOW_EVENT_PREFIX = "AIGEN_WORKFLOW "
WORKFLOW_RUN_VERSION = 1


class WorkflowExecutionError(RuntimeError):
    pass


class WorkflowInterrupted(WorkflowExecutionError):
    pass


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
    cache_hit: NodeCacheHit | None = None


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
    run_dir = _create_run_dir(workflow_root, graph.workflow_id)
    node_cache = WorkflowNodeCache(workflow_root / "cache")
    snapshot_path = _save_snapshot(graph, run_dir)
    atomic_write_json(run_dir / "execution.json", {
        "workflow_digest": workflow_digest,
        "targets": workflow.terminal_node_ids,
        "execution_order": execution_order,
        "effective_configs": {node_id: execution_config_payload(workflow.node(node_id).config) for node_id in execution_order},
        "snapshot": snapshot_path.as_posix(),
    })
    outputs_by_node: dict[str, dict[str, WorkflowArtifact]] = {}
    node_manifests: dict[str, Path] = {}

    def forward_node_progress(node_id: str, payload: dict[str, object]) -> None:
        if node_progress_sink is not None and node_id not in node_manifests:
            node_progress_sink(node_id, payload)
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
                    source_outputs = _source_outputs(compiled_node, node_cache)
                    if isinstance(node, ImageCollectionNode):
                        candidates = tuple(
                            load_node_result(node_manifests[wire.source.node_id]).candidate(
                                node_manifests[wire.source.node_id], wire.source.port,
                            )
                            for wire in compiled_node.incoming["images"]
                        )
                        source_outputs = {"collection": ImageCollectionArtifact(
                            candidates=candidates,
                            identity=_digest([candidate.output_id for candidate in candidates]),
                        )}
                    elif isinstance(node, PositionedKeyframeNode):
                        image = one_artifact(inputs, "image", ImageArtifact)
                        source_outputs = {"keyframe": KeyframeArtifact(
                            image=image, frame=node.config.frame, identity=_digest({"image": image.identity, "frame": node.config.frame}))}
                    elif isinstance(node, BindMaskNode):
                        from aigen.workflow_mask_execution import bind_mask
                        source_outputs = {"mask": bind_mask(one_artifact(inputs, "source", ImageArtifact),
                                                            Path(one_artifact(inputs, "image", ImageArtifact).path), node.config)}
                    provenance = workflow_node_provenance(node, compiled_node.config)
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
                                inputs,
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
                        outcome,
                        compiled_node=compiled_node,
                        inputs=inputs,
                        provenance=provenance,
                        status=status,
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
            def completed(item: _PendingNodeExecution, outcome: _NodeOutcome) -> None:
                nonlocal active_node_ids
                node_id = item.node.id
                outputs_by_node[node_id] = dict(outcome.outputs)
                node_manifests[node_id] = _write_node_manifest(
                    run_dir,
                    item.node,
                    item.signature,
                    outcome,
                    compiled_node=item.compiled_node,
                    inputs=item.inputs,
                    provenance=item.provenance,
                    status="completed",
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
                active_node_ids = tuple(active for active in active_node_ids if active != node_id)
                _write_run_state(run_dir, graph, workflow_digest, snapshot_path,
                                 status="running", node_manifests=node_manifests)

            record_dir = run_dir / "executions" / uuid4().hex
            record_dir.mkdir(parents=True)
            _execute_group(
                group,
                node_cache=node_cache,
                node_progress_sink=forward_node_progress,
                record_dir=record_dir,
                on_completed=completed,
            )
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
    key = _execution_group_key(ordered[0])
    return (ordered[0], *(item for item in ordered[1:] if _execution_group_key(item) == key))


def _execution_group_key(item: _PendingNodeExecution) -> tuple[str, str]:
    if _is_batchable_image_edit(item.image_edit_plan):
        return "image-edit", _image_edit_batch_key(cast(_ResolvedImageEditPlan, item.image_edit_plan))
    if isinstance(item.node, ImagePostprocessNode):
        return "image-postprocess", _digest(execution_config_payload(item.compiled_node.config))
    if isinstance(item.node, VIDEO_SEED_SWEEP_NODES):
        config = execution_config_payload(item.compiled_node.config)
        del config["seed"], config["seed_mode"]
        return item.node.kind, _digest({
            "config": config,
            "inputs": {port: [artifact.identity for artifact in artifacts] for port, artifacts in item.inputs.items()},
            "provenance": item.provenance.model_dump(mode="json"),
        })
    return "node", item.node.id


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
    record_dir: Path,
    on_completed: Callable[[_PendingNodeExecution, _NodeOutcome], None],
) -> None:
    if all(_is_batchable_image_edit(item.image_edit_plan) for item in group):
        _execute_image_edit_group(
            group, node_cache=node_cache, node_progress_sink=node_progress_sink,
            record_dir=record_dir, on_completed=on_completed,
        )
    elif isinstance(group[0].node, VIDEO_SEED_SWEEP_NODES):
        _execute_video_group(
            group, node_cache=node_cache, node_progress_sink=node_progress_sink,
            record_dir=record_dir, on_completed=on_completed,
        )
    elif len(group) > 1 and all(isinstance(item.node, ImagePostprocessNode) for item in group):
        outcomes = _execute_image_postprocess_group(
            group, node_cache=node_cache, node_progress_sink=node_progress_sink,
            record_dir=record_dir,
        )
        for item, outcome in zip(group, outcomes, strict=True):
            on_completed(item, outcome)
    else:
        for item in group:
            on_completed(item, _execute_node(
                item, node_cache=node_cache, node_progress_sink=node_progress_sink,
                record_dir=record_dir,
            ))


def _execute_image_edit_group(
    group: Sequence[_PendingNodeExecution],
    *,
    node_cache: WorkflowNodeCache,
    node_progress_sink: NodeProgressSink | None,
    record_dir: Path,
    on_completed: Callable[[_PendingNodeExecution, _NodeOutcome], None],
) -> None:
    plans = tuple(cast(_ResolvedImageEditPlan, item.image_edit_plan) for item in group)
    with ExitStack() as stack:
        writes = {
            item.node.id: stack.enter_context(node_cache.begin(
                item.signature, node_kind=item.node.kind, provenance=item.provenance,
            ))
            for item in group
        }
        request = ImageEditBatchRequest(
            backend=plans[0].backend,
            cases=tuple(
                ImageEditBatchCase(
                    id=item.node.id, prompt=plan.prompt, image_paths=plan.references,
                    width=plan.width, height=plan.height, seed=plan.seed,
                    output_path=record_dir / "images" / f"{item.node.id}.png",
                )
                for item, plan in zip(group, plans, strict=True)
            ),
            loras=plans[0].loras, steps=plans[0].steps, guidance=plans[0].guidance,
            strength=plans[0].strength, sampler=plans[0].sampler, scheduler=plans[0].scheduler,
        )
        items = {item.node.id: item for item in group}
        completed: set[str] = set()

        def output_completed(output: ImageEditBatchOutput) -> None:
            if output.case_id not in items or output.case_id in completed:
                raise WorkflowExecutionError(f"unexpected or duplicate image-edit output: {output.case_id}")
            item = items[output.case_id]
            expected = record_dir / "images" / f"{item.node.id}.png"
            if output.path.resolve() != expected:
                raise WorkflowExecutionError(f"image-edit returned the wrong output for {item.node.id}: {output.path}")
            generated = {"image": GeneratedNodeOutput(ArtifactType.IMAGE, (_require_file(expected, item.node),))}
            outcome = _publish_generated(writes[item.node.id], item, generated, record_dir)
            completed.add(item.node.id)
            on_completed(item, outcome)

        with _node_progress(tuple(items), node_progress_sink, record_dir) as node_progress:
            result = run_image_edit_batch(
                request, progress=node_progress, record_dir=record_dir / "backend",
                on_output=output_completed,
            )
        if completed != set(items) or {output.case_id for output in result.outputs} != completed:
            raise WorkflowExecutionError("image-edit batch did not publish every requested case")


def format_workflow_event(payload: dict[str, object]) -> str:
    return WORKFLOW_EVENT_PREFIX + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _execute_video_group(
    group: Sequence[_PendingNodeExecution],
    *,
    node_cache: WorkflowNodeCache,
    node_progress_sink: NodeProgressSink | None,
    record_dir: Path,
    on_completed: Callable[[_PendingNodeExecution, _NodeOutcome], None],
) -> None:
    by_seed: dict[int, list[_PendingNodeExecution]] = {}
    for item in group:
        seed = item.compiled_node.config.seed
        by_seed.setdefault(seed, []).append(item)
    seeds = tuple(by_seed)

    def completed(seed: int, generated: dict[str, GeneratedNodeOutput]) -> None:
        items = by_seed.pop(seed, None)
        if items is None:
            raise WorkflowExecutionError(f"unexpected or duplicate video seed-sweep output: {seed}")
        for item in items:
            with node_cache.begin(item.signature, node_kind=item.node.kind, provenance=item.provenance) as write:
                on_completed(item, _publish_generated(write, item, generated, record_dir))

    first = group[0]
    with _node_progress(tuple(item.node.id for item in group), node_progress_sink, record_dir) as progress:
        execute_video_seed_sweep(first.compiled_node, first.inputs, seeds, record_dir / "outputs",
                                 progress=progress, on_output=completed)
    if by_seed:
        raise WorkflowExecutionError("video seed sweep did not publish every requested output")


def _create_run_dir(workflow_root: Path, workflow_id: str) -> Path:
    attempts_root = workflow_root / "runs" / workflow_id
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
        "workflow_id": graph.workflow_id,
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
    node_cache: WorkflowNodeCache,
) -> dict[str, WorkflowArtifact] | None:
    node = compiled_node.node
    if isinstance(node, VideoSourceNode):
        path = cast(CompiledVideoSourceConfig, compiled_node.config).path
        identity = sha256_file(path)
        return {"video": VideoArtifact(path=path.as_posix(), identity=identity, content_sha256=identity,
                                       info=node_cache.video_info(path, identity))}
    if isinstance(node, AudioSourceNode):
        config = cast(CompiledAudioSourceConfig, compiled_node.config)
        track = load_audio_track(config.path, stream_index=config.stream_index)
        return {"audio": AudioArtifact(track=track, identity=_digest(track.model_dump(mode="json", exclude={"path"})))}
    if isinstance(node, ImageSelectionNode):
        return {"image": cast(CompiledImageSelectionConfig, compiled_node.config).image}
    if isinstance(node, ImageSourceNode):
        path = cast(CompiledImageSourceConfig, compiled_node.config).path
        identity = sha256_file(path)
        return {
            "image": ImageArtifact(
                path=path.as_posix(),
                identity=identity,
                content_sha256=identity,
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
                reference_sha256s=tuple(item["sha256"] for item in identity_payload["references"]),
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
    if isinstance(node, FramePostprocessNode) and one_artifact(inputs, "images", ImageSequenceArtifact).pixel_identity is None:
        raise WorkflowExecutionError("frame pixel identity is missing; extract the source video again")
    return build_node_signature(
        node_kind=node.kind,
        execution_config=execution_config_payload(compiled_node.config),
        inputs={
            port: tuple(
                NodeInputIdentity(
                    artifact_type=artifact.type,
                    identity=artifact.pixel_identity if isinstance(node, FramePostprocessNode) and isinstance(artifact, ImageSequenceArtifact) else artifact.identity,
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
    outcome: _NodeOutcome,
    *,
    compiled_node: CompiledNode,
    inputs: Mapping[str, Sequence[WorkflowArtifact]],
    provenance: NodeExecutionProvenance,
    status: Literal["completed", "reused"],
) -> Path:
    final_outputs = dict(outcome.outputs)
    _validate_output_contract(node, final_outputs)
    manifest_path = run_dir / "nodes" / node.id / "result.json"
    manifest = NodeResultManifest(
        node_id=node.id,
        node_kind=node.kind,
        signature=signature,
        outputs=final_outputs,
        title=node.title,
        status=status,
        provenance=provenance,
        cache_manifest=outcome.cache_hit.manifest_path.as_posix() if outcome.cache_hit else None,
        details=(outcome.cache_hit.details if outcome.cache_hit else NodeExecutionDetails(
            completed_at=datetime.now(UTC).isoformat(),
            effective_config=execution_config_payload(compiled_node.config),
            inputs={port: tuple(artifacts) for port, artifacts in inputs.items()},
            measured_outputs=_measure_outputs(final_outputs),
        )),
        image_origins=({"image": cast(CompiledImageSelectionConfig, compiled_node.config).reference}
                       if isinstance(node, ImageSelectionNode) else {}),
    )
    atomic_write_json(
        manifest_path,
        manifest.model_dump(mode="json"),
    )
    return manifest_path


def _outcome_from_cache_hit(
    node: WorkflowNode,
    hit: NodeCacheHit,
    inputs: Mapping[str, Sequence[WorkflowArtifact]],
) -> _NodeOutcome:
    _validate_output_contract(node, hit.outputs)
    outputs = hit.outputs
    if isinstance(node, FramePostprocessNode):
        source = one_artifact(inputs, "images", ImageSequenceArtifact)
        outputs = {"images": with_sequence_timing(cast(ImageSequenceArtifact, hit.outputs["images"]),
                                                  timeline=source.timeline, audio=source.audio)}
    elif isinstance(node, ExtractVideoFramesNode):
        video = one_artifact(inputs, "video", VideoArtifact)
        assert video.info is not None and video.content_sha256 is not None
        outputs = {"images": with_sequence_timing(cast(ImageSequenceArtifact, hit.outputs["images"]),
            timeline=video.info.timeline, audio=video_audio_track(Path(video.path), video.info, video.content_sha256))}
    return _NodeOutcome(
        outputs=outputs,
        cache_hit=hit,
    )


def _execute_node(
    pending: _PendingNodeExecution,
    *,
    node_cache: WorkflowNodeCache,
    node_progress_sink: NodeProgressSink | None,
    record_dir: Path,
) -> _NodeOutcome:
    with node_cache.begin(
        pending.signature, node_kind=pending.node.kind, provenance=pending.provenance,
    ) as write:
        generated = _run_generated_node(
            pending, record_dir / "outputs", node_progress_sink=node_progress_sink,
        )
        return _publish_generated(write, pending, generated, record_dir)


def _execute_image_postprocess_group(
    group: Sequence[_PendingNodeExecution],
    *,
    node_cache: WorkflowNodeCache,
    node_progress_sink: NodeProgressSink | None,
    record_dir: Path,
) -> tuple[_NodeOutcome, ...]:
    first_config = cast(CompiledPostprocessConfig, group[0].compiled_node.config)
    sources = tuple(one_artifact(item.inputs, "image", ImageArtifact) for item in group)
    with _node_progress(tuple(item.node.id for item in group), node_progress_sink, record_dir) as node_progress:
        result = _postprocess_images(
            first_config, tuple(Path(source.path) for source in sources), record_dir / "outputs",
            output_names=tuple(f"{item.node.id}.png" for item in group), progress=node_progress,
        )
    outcomes = []
    for item, output in zip(group, result.outputs, strict=True):
        with node_cache.begin(item.signature, node_kind=item.node.kind, provenance=item.provenance) as write:
            outcomes.append(_publish_generated(
                write, item, {"image": GeneratedNodeOutput(ArtifactType.IMAGE, (output,))}, record_dir,
            ))
    return tuple(outcomes)


def _publish_generated(
    write: NodeCacheWrite,
    item: _PendingNodeExecution,
    generated: Mapping[str, GeneratedNodeOutput],
    record_dir: Path,
) -> _NodeOutcome:
    config = execution_config_payload(item.compiled_node.config)
    if item.image_edit_plan is not None:
        config.update(width=item.image_edit_plan.width, height=item.image_edit_plan.height)
    measured = {}
    for port, output in generated.items():
        if output.artifact_type in (ArtifactType.IMAGE, ArtifactType.IMAGE_SEQUENCE, ArtifactType.MASK):
            with Image.open(output.paths[0]) as image:
                measured[port] = {"width": image.width, "height": image.height, "images": len(output.paths)}
        elif output.artifact_type == ArtifactType.VIDEO:
            assert output.video_info is not None
            info = output.video_info
            measured[port] = {**info.model_dump(mode="json"), "frames": info.frames, "fps": str(info.fps)}
    if isinstance(item.node, (CharacterEditNode, CharacterRefineNode)):
        result = read_json(record_dir / "outputs" / "backend-result.json", label="character result")
        selected = result["outputs"][0]
        measured["image"].update(
            seed=selected["raw"]["seed"], candidate_identity=selected["candidate_identity"],
            audit_report=selected["audit_report"], raw=selected["raw"],
        )
    details = NodeExecutionDetails(
        completed_at=datetime.now(UTC).isoformat(), effective_config=config,
        inputs={port: tuple(artifacts) for port, artifacts in item.inputs.items()},
        measured_outputs=measured, record_dir=record_dir.as_posix(),
        case_record=(record_dir / "backend" / "cases" / f"{item.node.id}.json").as_posix()
        if _is_batchable_image_edit(item.image_edit_plan)
        else (record_dir / "outputs" / f"seed-{config['seed']}" / "backend-result.json").as_posix()
        if isinstance(item.node, VIDEO_SEED_SWEEP_NODES)
        else (record_dir / "outputs" / "backend-result.json").as_posix()
        if isinstance(item.node, (CharacterEditNode, CharacterRefineNode, SamSegmentNode)) else None,
    )
    hit = write.publish(write.import_outputs(generated), details=details)
    return _outcome_from_cache_hit(item.node, hit, item.inputs)


def _measure_outputs(outputs: Mapping[str, WorkflowArtifact]) -> dict[str, dict[str, object]]:
    measured = {}
    for port, artifact in outputs.items():
        if isinstance(artifact, ImageArtifact):
            with Image.open(artifact.path) as image:
                measured[port] = {"width": image.width, "height": image.height}
        elif isinstance(artifact, ImageCollectionArtifact):
            measured[port] = {"candidates": len(artifact.candidates)}
        elif isinstance(artifact, VideoArtifact) and artifact.info is not None:
            measured[port] = {**artifact.info.model_dump(mode="json"), "frames": artifact.info.frames, "fps": str(artifact.info.fps)}
    return measured


def _run_generated_node(
    pending: _PendingNodeExecution,
    staging: Path,
    *,
    node_progress_sink: NodeProgressSink | None,
) -> dict[str, GeneratedNodeOutput]:
    staging.mkdir(parents=True)
    node = pending.node
    inputs = pending.inputs
    with _node_progress((node.id,), node_progress_sink, staging.parent) as node_progress:
        if isinstance(node, SamSegmentNode):
            from aigen.workflow_mask_execution import execute_sam_node
            return execute_sam_node(node.config, inputs, staging, progress=node_progress)
        if isinstance(node, CharacterRefineNode):
            from aigen.workflow_mask_execution import execute_character_refine_node
            return execute_character_refine_node(
                cast(CompiledCharacterRefineConfig, pending.compiled_node.config), inputs,
                _image_edit_reference_paths(node, inputs), _image_edit_loras(node, inputs), staging, progress=node_progress,
            )
        if isinstance(node, CharacterEditNode):
            return execute_character_node(
                cast(CompiledCharacterEditConfig, pending.compiled_node.config),
                _image_edit_reference_paths(node, inputs), _image_edit_loras(node, inputs),
                inputs, staging, progress=node_progress,
            )
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
            source = one_artifact(inputs, "image", ImageArtifact)
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

        if isinstance(node, VIDEO_EXECUTION_NODES):
            return execute_video_node(pending.compiled_node, inputs, staging, progress=node_progress)

        if isinstance(node, FramePostprocessNode):
            source = one_artifact(
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
    node: ImageEditNode | CharacterEditNode | CharacterRefineNode,
    inputs: Mapping[str, Sequence[WorkflowArtifact]],
) -> tuple[Path, ...]:
    references: list[Path] = []
    for artifact in inputs.get("references", ()):
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
    node: ImageEditNode | CharacterEditNode | CharacterRefineNode,
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


@contextmanager
def _node_progress(
    node_ids: Sequence[str],
    sink: NodeProgressSink | None,
    record_dir: Path,
):
    with (record_dir / "progress.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        def forward(payload: dict[str, object]) -> None:
            log.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            if sink is not None:
                for node_id in node_ids:
                    sink(node_id, payload)

        with RuntimeStatus.callback(
            interval_seconds=DEFAULT_PROGRESS_INTERVAL_SECONDS,
            callback=forward, telemetry=SystemTelemetrySampler(),
        ) as progress:
            yield progress


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
