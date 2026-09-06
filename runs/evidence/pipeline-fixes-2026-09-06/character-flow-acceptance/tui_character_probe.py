"""Reviewed TUI character edit -> raw audit -> optional VOSR -> cache/export."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

from textual.widgets import Input

from aigen import image_tui
from aigen.manifest_io import atomic_write_json, sha256_file
from aigen.workflow_document_io import load_workflow_document, save_workflow_document
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_graph import CharacterEditConfig, CharacterEditNode, ImageSourceConfig, ImageSourceNode, NodePortRef, WorkflowConnection, WorkflowGraph
from aigen.workflow_results import load_node_result, node_result_history

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "image-flow-acceptance"))
from tui_probe import run_action, select_node


async def run(args):
    root = Path(__file__).resolve().parent
    request = json.loads((root / "manifest.json").read_text())
    assert request["review_status"] == "approved"
    source = request["source"]
    assert sha256_file(Path(source["path"])) == source["sha256"]
    config = CharacterEditConfig(**request["routes"][args.backend], prompt=request["prompt"],
        seed=request["seeds"][0], candidates=len(request["seeds"]), max_iterations=request["max_iterations"])
    graph = WorkflowGraph(name=f"{args.backend} audited character flow", nodes=(
        ImageSourceNode(id="reference", title="Reference", config=ImageSourceConfig(path=source["path"])),
        CharacterEditNode(id="edit", title="Character edit", config=config),
    ), connections=(WorkflowConnection(id="ref", source=NodePortRef(node_id="reference", port="image"),
        target=NodePortRef(node_id="edit", port="references")),))
    buffer = WorkflowEditBuffer(graph)
    buffer.auto_layout()
    directory = root / args.attempt
    directory.mkdir()
    path = directory / "workflow.json"
    save_workflow_document(buffer.document, path)
    image_tui.STATE_PATH = directory / "form.json"
    image_tui.DEFAULT_WORKFLOW_RUNS_ROOT = directory / "workflows"
    app = image_tui.ImageGenerationApp()
    app.workflow_buffer = WorkflowEditBuffer(load_workflow_document(path), document_path=path)
    app.workflow_document_paths = [path]
    async with app.run_test(size=(160, 54)) as pilot:
        app._open_workflow_editor()
        await pilot.pause()
        editor = app.workflow_editor
        await select_node(app, editor, pilot, "edit")
        generated = await run_action(app, editor, pilot, "#workflow-run-target", directory, "generation", request["min_free_mib"])
        cached = await run_action(app, editor, pilot, "#workflow-run-target", directory, "cache", request["min_free_mib"])
        assert cached["statuses"]["edit"] == "reused"
        assert await pilot.click("#workflow-results")
        await pilot.pause()
        results = app.screen
        await results.workers.wait_for_complete()
        (directory / "result.svg").write_text(app.export_screenshot())
        assert await pilot.click("#result-export")
        await pilot.pause()
        app.screen.query_one(Input).value = str(directory / "accepted.png")
        assert await pilot.click("#dialog-ok")
        await pilot.pause()
        await results.workers.wait_for_complete()
        manifest = node_result_history(directory / "workflows", graph.workflow_id, "edit")[0]
        result = load_node_result(manifest)
        candidate = result.candidate(manifest, "image")
        assert sha256_file(directory / "accepted.png") == sha256_file(Path(candidate.image.path))
        atomic_write_json(directory / "acceptance.json", {
            "backend": args.backend, "generation": generated, "cache": cached,
            "selected_seed": candidate.seed, "result": result.model_dump(mode="json"),
        })
        print(json.dumps({"status": "completed", "output": str(directory / "accepted.png")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("klein", "qwen"))
    parser.add_argument("attempt")
    asyncio.run(run(parser.parse_args()))
