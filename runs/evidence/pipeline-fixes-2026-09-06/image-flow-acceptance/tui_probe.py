"""Actual TUI -> CLI -> native models -> cached variants -> saved image choice.

Run `variants` first, inspect the exported chosen image, then run `continue`.
No neural doubles. All character/reference content comes from the reviewed manifest.
"""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import psutil
from textual.widgets import Button, Input

from aigen import image_tui
from aigen.gpu_status import nvidia_smi_memory_snapshot
from aigen.manifest_io import atomic_write_json, sha256_file
from aigen.workflow_canvas import WorkflowCanvas
from aigen.workflow_document_io import load_workflow_document, save_workflow_document
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_graph import ImageEditConfig, ImageEditNode, ImageSourceConfig, ImageSourceNode, NodePortRef, WorkflowConnection, WorkflowGraph
from aigen.workflow_inspector import PropertyInput, WorkflowInspector
from aigen.workflow_results import load_node_result, node_result_history
from aigen.workflow_results_tui import WorkflowResults


def make_document(request, continuation):
    nodes = [ImageSourceNode(id=f"ref{index}", title=f"Reference {index + 1}", config=ImageSourceConfig(path=path))
             for index, path in enumerate(request["references"])]
    config = ImageEditConfig(backend=request["backend"], prompt=request["prompt"],
                            width=request["width"], height=request["height"], steps=request["steps"], seed=request["seeds"][0])
    nodes.extend((ImageEditNode(id="edit", title="Style variants", config=config),
                  ImageEditNode(id="continue", title="Continue with selected image", config=config.model_copy(update=continuation))))
    wires = [WorkflowConnection(id=f"wire{index}", source=NodePortRef(node_id=f"ref{index}", port="image"),
                                target=NodePortRef(node_id="edit", port="references"), order=index)
             for index in range(len(request["references"]))]
    wires.append(WorkflowConnection(id="next", source=NodePortRef(node_id="edit", port="image"),
                                     target=NodePortRef(node_id="continue", port="references")))
    buffer = WorkflowEditBuffer(WorkflowGraph(name=f"{request['backend']} image flow", nodes=nodes, connections=wires))
    buffer.auto_layout()
    return buffer.document


async def select_node(app, editor, pilot, node_id):
    editor.query_one(WorkflowCanvas).set_selected_node(node_id)
    await editor.query_one(WorkflowInspector).show(app.workflow_buffer.document, node_id, None)
    await pilot.pause()


async def run_action(app, editor, pilot, action, directory, label, minimum):
    wait_started = time.monotonic()
    available_since = None
    with (directory / f"{label}-gpu-wait.jsonl").open("a", buffering=1) as log:
        while True:
            gpu = nvidia_smi_memory_snapshot()
            free = gpu["nvidia_smi_device_total_mb"] - gpu["nvidia_smi_used_mb"]
            now = time.monotonic()
            log.write(json.dumps({"elapsed_seconds": now-wait_started, "free_mib": free}) + "\n")
            if free >= minimum:
                if available_since is None:
                    available_since = now
                if now - available_since >= 8:
                    break
            else:
                available_since = None
            assert now - wait_started < 1800, "GPU stayed occupied for 30 minutes"
            await pilot.pause(2)
    atomic_write_json(directory / f"{label}-preflight.json", {**gpu, "free_mib": free})
    print(json.dumps({"label":label, "gpu_free_mib":free}), flush=True)
    assert free >= minimum, f"GPU occupied: {free} MiB free, need {minimum}"
    started = time.monotonic()
    assert await pilot.click(action)
    await pilot.pause()
    process = app.process
    assert process is not None
    previous = None
    with (directory / f"{label}-telemetry.jsonl").open("w", buffering=1) as log:
        while app.process is not None:
            assert time.monotonic() - started < 1800, "Workflow exceeded 30 minutes"
            await pilot.pause(1)
            state = app.generation_progress
            swap = psutil.swap_memory()
            record = {"elapsed_seconds":time.monotonic()-started, "swap_used_mib":swap.used/1024**2,
                      "swap_in_bytes":swap.sin,"swap_out_bytes":swap.sout,
                      "host_available_mib":psutil.virtual_memory().available/1024**2,
                      "progress":asdict(state) if state is not None else None}
            log.write(json.dumps(record)+"\n")
            if state is not None and state.phase != previous:
                print(json.dumps({"label":label, "phase":state.phase}), flush=True)
                previous = state.phase
    record = {"returncode":process.returncode,"stdout_closed":process.stdout.closed,
              "elapsed_seconds":time.monotonic()-started,
              "statuses":dict(editor.query_one(WorkflowCanvas).runtime_statuses)}
    atomic_write_json(directory / f"{label}-result.json", record)
    assert process.returncode == 0, record
    assert process.stdout.closed
    return record


async def run(args):
    config = json.loads(args.manifest.read_text())
    request = config[args.backend]
    directory = args.manifest.resolve().parent / (args.attempt or args.backend)
    if args.stage == "variants":
        directory.mkdir()
        save_workflow_document(make_document(request, config["continuation"]), directory / "workflow.json")
    document_path = directory / "workflow.json"
    root = directory / "workflows"
    image_tui.STATE_PATH = directory / "form.json"
    image_tui.DEFAULT_WORKFLOW_RUNS_ROOT = root
    app = image_tui.ImageGenerationApp()
    app.workflow_buffer = WorkflowEditBuffer(load_workflow_document(document_path), document_path=document_path)
    app.workflow_document_paths = [document_path]
    async with app.run_test(size=(160, 54)) as pilot:
        app._open_workflow_editor()
        await pilot.pause()
        editor = app.workflow_editor
        if args.stage == "variants":
            await select_node(app, editor, pilot, "edit")
            assert await pilot.click("#workflow-variants")
            await pilot.pause()
            app.screen.query_one(Input).value = ", ".join(map(str,request["seeds"]))
            assert await pilot.click("#dialog-ok")
            await pilot.pause()
            await editor.workers.wait_for_complete()
            collection_id = editor.query_one(WorkflowCanvas).selected_node_id
            assert await pilot.click("#workflow-save")
            await pilot.pause()
            # Reopen from the persisted document before starting the model.
            assert await pilot.click("#workflow-close")
            await pilot.pause()
            app.workflow_buffer = WorkflowEditBuffer(load_workflow_document(document_path), document_path=document_path)
            app._open_workflow_editor()
            await pilot.pause()
            editor = app.workflow_editor
            await select_node(app, editor, pilot, collection_id)
            await run_action(app, editor, pilot, "#workflow-run-target", directory, "generation", config["min_free_mib"])
            cached = await run_action(app, editor, pilot, "#workflow-run-target", directory, "cache", config["min_free_mib"])
            assert all(status == "reused" for node_id,status in cached["statuses"].items() if node_id not in {collection_id,*(f"ref{i}" for i in range(len(request["references"])) )})
            await select_node(app, editor, pilot, "edit")
            next(widget for widget in editor.query(PropertyInput) if widget.field_name == "seed").value = "74"
            assert await editor.commit_pending_property()
            await select_node(app, editor, pilot, collection_id)
            changed = await run_action(app, editor, pilot, "#workflow-run-target", directory, "changed-seed", config["min_free_mib"])
            assert changed["statuses"]["edit"] == "completed"
            other = next(node.id for node in app.workflow_buffer.document.nodes if isinstance(node,ImageEditNode) and node.id not in ("edit","continue"))
            assert changed["statuses"][other] == "reused"
            assert await pilot.click("#workflow-results")
            await pilot.pause()
            results = app.screen
            assert isinstance(results,WorkflowResults)
            await results.workers.wait_for_complete()
            results.query_one("#candidate-1",Button).scroll_visible(animate=False)
            await pilot.pause()
            assert await pilot.click("#candidate-1")
            await results.workers.wait_for_complete()
            chosen = results.candidates[1]
            (directory/"comparison.svg").write_text(app.export_screenshot())
            assert await pilot.click("#result-export")
            await pilot.pause()
            app.screen.query_one(Input).value = str(directory/"selected.png")
            assert await pilot.click("#dialog-ok")
            await pilot.pause()
            await results.workers.wait_for_complete()
            assert sha256_file(directory/"selected.png") == sha256_file(Path(chosen.image.path))
            assert await pilot.click("#result-select")
            await pilot.pause()
            await editor.workers.wait_for_complete()
            saved = load_workflow_document(document_path)
            assert any(getattr(node.config,"selected",None) == chosen.reference for node in saved.nodes)
            atomic_write_json(directory/"selected.json", chosen.model_dump(mode="json"))
            print(json.dumps({"status":"variants-and-selection-completed", "inspect_before_continuing":str(directory/"selected.png")}),flush=True)
        else:
            result = await run_action(app, editor, pilot, "#workflow-run", directory, "continuation", config["min_free_mib"])
            assert len(result["statuses"]) == 2, result
            assert result["statuses"]["continue"] == "completed"
            await select_node(app,editor,pilot,"continue")
            assert await pilot.click("#workflow-results")
            await pilot.pause()
            results=app.screen
            await results.workers.wait_for_complete()
            (directory/"continuation.svg").write_text(app.export_screenshot())
            assert await pilot.click("#result-export")
            await pilot.pause()
            app.screen.query_one(Input).value = str(directory/"continued.png")
            assert await pilot.click("#dialog-ok")
            await pilot.pause()
            await results.workers.wait_for_complete()
            assert (directory/"continued.png").is_file()
            print(json.dumps({"status":"continuation-completed","image":str(directory/"continued.png")}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("backend",choices=("klein","qwen"))
    parser.add_argument("stage",choices=("variants","continue"))
    parser.add_argument("--attempt",help="Separate retained evidence directory for a retry")
    asyncio.run(run(parser.parse_args()))
