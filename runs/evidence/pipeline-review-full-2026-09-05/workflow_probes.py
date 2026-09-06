"""Snapshot audit: real compiler, scheduler, disk cache and FFmpeg; no model runs.

Generation is doubled at the backend boundary. Synthetic outputs vary with source,
seed and LoRA bytes so downstream invalidation can actually be observed.
"""
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from PIL import Image
from aigen.progress import SILENT_STATUS
from aigen import workflow_execution as execution
from aigen.workflow_cache import NodeExecutionProvenance, RevisionedComponent
from aigen.workflow_compilation import compile_workflow_run
from aigen.workflow_document_io import load_workflow_document, save_workflow_document
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_graph import (
    AnimeGenI2VNode, ExtractVideoFramesNode, FramePostprocessNode,
    ImageEditConfig, ImageEditNode, ImageSourceConfig, ImageSourceNode,
    LoraSourceConfig, LoraSourceNode, NodePortRef, PixelArtFixerConfig,
    VideoContactSheetNode, WorkflowConnection, WorkflowGraph,
)
from aigen.workflow_templates import default_animegen_config

EVIDENCE = Path(__file__).resolve().parent


def make_graph(source: Path, lora: Path) -> WorkflowGraph:
    def wire(source_id, source_port, target_id, target_port):
        return WorkflowConnection(
            id=f"wire-{source_id}-{target_id}",
            source=NodePortRef(node_id=source_id, port=source_port),
            target=NodePortRef(node_id=target_id, port=target_port),
        )
    edit = ImageEditConfig(backend="flux2-klein", prompt="Keep the image unchanged.", width=512, height=512)
    anime = default_animegen_config().model_copy(update={"prompt": "The camera remains still.", "frames": 5, "fps": 12})
    return WorkflowGraph(name="CPU flow audit", nodes=(
        ImageSourceNode(id="source", title="Synthetic source", config=ImageSourceConfig(path=str(source))),
        LoraSourceNode(id="lora", title="Synthetic LoRA", config=LoraSourceConfig(path=str(lora))),
        ImageEditNode(id="start", title="Start", config=edit.model_copy(update={"seed": 11})),
        ImageEditNode(id="end", title="End", config=edit.model_copy(update={"seed": 12})),
        AnimeGenI2VNode(id="video", title="Video", config=anime),
        VideoContactSheetNode(id="sheet", title="Contact sheet"),
        ExtractVideoFramesNode(id="extract", title="Extract"),
        FramePostprocessNode(id="frames", title="Processed frames", config=PixelArtFixerConfig(mode="fast", low_memory=False, force_step=2.0)),
    ), connections=(
        wire("source", "image", "start", "references"), wire("source", "image", "end", "references"),
        wire("lora", "lora", "start", "loras"), wire("lora", "lora", "end", "loras"),
        wire("start", "image", "video", "start"), wire("end", "image", "video", "end"),
        wire("video", "video", "sheet", "video"), wire("video", "video", "extract", "video"),
        wire("extract", "images", "frames", "images"),
    ))


def main():
    workspace = EVIDENCE / f"workflow-data-{uuid4().hex[:8]}"
    workspace.mkdir()
    source, lora = workspace / "source.png", workspace / "lora.safetensors"
    Image.new("RGB", (64, 64), (80, 120, 160)).save(source)
    lora.write_bytes(b"synthetic-lora-v1; never loaded by a model")
    graph = make_graph(source, lora)
    document = workspace / "workflow.json"
    save_workflow_document(graph, document)
    graph = load_workflow_document(document)
    calls = []
    mode = {"value": "ok"}

    def batch(request, *, progress):
        calls.append({"kind": "image-batch", "cases": [c.id for c in request.cases]})
        if mode["value"] == "fail":
            raise RuntimeError("deliberate CPU backend failure")
        if mode["value"] == "interrupt":
            raise execution.WorkflowInterrupted("deliberate CPU interruption")
        outputs = []
        for case in request.cases:
            data = b"".join(p.read_bytes() for p in case.image_paths)
            data += b"".join(x.path.read_bytes() + str(x.weight).encode() for x in request.loras)
            color = tuple(sha256(data + str(case.seed).encode()).digest()[:3])
            case.output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), color).save(case.output_path)
            outputs.append(SimpleNamespace(case_id=case.id, path=case.output_path))
        return SimpleNamespace(outputs=tuple(outputs))

    def video(**kwargs):
        calls.append({"kind": "video", "frames": kwargs["frames"], "fps": kwargs["fps"]})
        output = kwargs["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        # Both keyframes influence bytes. Encoding/decoding and frame postprocessing are real.
        color = tuple(sha256(kwargs["image"].read_bytes() + kwargs["last_image"].read_bytes()).digest()[:3])
        rgba = "0x" + bytes(color).hex()
        subprocess.run(("ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c={rgba}:s=64x64:r={kwargs['fps']}",
                        "-frames:v", str(kwargs["frames"]), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)), check=True)
        return SimpleNamespace(output=output)

    provenance = NodeExecutionProvenance(
        executor=RevisionedComponent(name="audit-executor", revision="1"),
        backend=RevisionedComponent(name="CPU-doubles", revision="1"),
    )
    records = {}

    def run(label, current_graph, *, error=None):
        events, before = [], len(calls)
        try:
            result = execution.execute_workflow(compile_workflow_run(current_graph), runs_root=workspace / "runs", progress=SILENT_STATUS, event_sink=events.append)
        except (RuntimeError, execution.WorkflowInterrupted) as failure:
            if error is None or not isinstance(failure, error):
                raise
            record = {"error": str(failure)}
        else:
            assert error is None
            record = {"run_dir": str(result.run_dir), "terminal_outputs": result.to_json()["outputs"]}
        record.update(events=events, calls=calls[before:])
        records[label] = record
        return {str(e["node_id"]): e["status"] for e in events}

    with ExitStack() as patches:
        patches.enter_context(patch.object(execution, "run_image_edit_batch", batch))
        patches.enter_context(patch.object(execution, "generate_animegen_i2v", video))
        patches.enter_context(patch.object(execution, "workflow_node_provenance", return_value=provenance))
        first = run("saved_and_reloaded", graph)
        assert set(first.values()) == {"completed"}
        repeated = run("cache_hit", graph)
        assert not records["cache_hit"]["calls"]
        assert all(repeated[n] == "reused" for n in ("start", "end", "video", "sheet", "extract", "frames"))
        buffer = WorkflowEditBuffer(graph)
        buffer.update_node_config("start", "seed", 15)
        changed = run("one_seed_changed", buffer.document)
        assert changed["end"] == "reused" and changed["start"] == changed["video"] == "completed"
        lora.write_bytes(b"synthetic-lora-v2; never loaded by a model")
        changed = run("lora_bytes_changed", graph)
        assert changed["start"] == changed["end"] == "completed"
        Image.new("RGB", (64, 64), (160, 40, 20)).save(source)
        changed = run("source_bytes_changed", graph)
        assert changed["start"] == changed["end"] == "completed"
        buffer = WorkflowEditBuffer(graph)
        buffer.update_node_config("start", "seed", 91)
        mode["value"] = "fail"
        failed = run("backend_failure", buffer.document, error=RuntimeError)
        assert failed["start"] == "failed" and "video" not in failed
        mode["value"] = "interrupt"
        interrupted = run("interrupted", buffer.document, error=execution.WorkflowInterrupted)
        assert interrupted["start"] == "interrupted" and "video" not in interrupted
        mode["value"] = "ok"
        restarted = run("restart", buffer.document)
        assert restarted["start"] == "completed" and restarted["end"] == "reused"

    # Record actual manifest paths independently of the date-based run naming convention.
    records["state_manifests"] = [str(p) for p in (workspace / "runs").rglob("run.json")]
    records["failure_manifests"] = [str(p) for p in (workspace / "runs").rglob("failure.json")]
    records["interruption_manifests"] = [str(p) for p in (workspace / "runs").rglob("interrupted.json")]
    records["scope"] = "Real workflow/compiler/cache/FFmpeg/pixel-art-fixer; image and video generators and backend provenance doubled; no CUDA or model quality evidence."
    (EVIDENCE / "workflow-results.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"status": "passed", "scenarios": len([v for v in records.values() if isinstance(v, dict)]), "workspace": str(workspace)}))


if __name__ == "__main__":
    main()
