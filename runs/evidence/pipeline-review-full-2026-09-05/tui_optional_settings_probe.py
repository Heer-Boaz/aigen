"""A compiler-valid graph with implicit sampling defaults crashes its inspector."""
import asyncio
from contextlib import redirect_stderr
import json
from pathlib import Path
import sys
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from PIL import Image
from textual.widgets._select import InvalidSelectValueError
from aigen import image_tui
from aigen.workflow_canvas import WorkflowCanvas
from aigen.workflow_compilation import compile_workflow
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_graph import ImageEditConfig, ImageEditNode, ImageSourceConfig, ImageSourceNode, NodePortRef, WorkflowConnection, WorkflowGraph
from aigen.workflow_inspector import WorkflowInspector

EVIDENCE = Path(__file__).resolve().parent


async def main():
    workspace = EVIDENCE / f"optional-data-{uuid4().hex[:8]}"
    workspace.mkdir()
    source = workspace / "source.png"
    Image.new("RGB", (64, 64), "blue").save(source)
    graph = WorkflowGraph(name="Implicit sampling audit", nodes=(
        ImageSourceNode(id="source", title="Source", config=ImageSourceConfig(path=str(source))),
        ImageEditNode(id="edit", title="Edit", config=ImageEditConfig(backend="qwen-image-edit-2511-base", prompt="Keep the image unchanged.")),
    ), connections=(WorkflowConnection(id="wire", source=NodePortRef(node_id="source", port="image"), target=NodePortRef(node_id="edit", port="references")),))
    compiled = compile_workflow(graph)
    result = {"compiled": True, "sampler_before": graph.node("edit").config.sampler, "resolved_sampler": compiled.node("edit").config.settings.sampler}
    with patch.object(image_tui, "STATE_PATH", workspace / "form.json"), (workspace / "expected-crash.log").open("w") as log, redirect_stderr(log):
        app = image_tui.ImageGenerationApp()
        app.workflow_buffer = WorkflowEditBuffer(graph)
        try:
            async with app.run_test(size=(120, 40)) as pilot:
                app._open_workflow_editor()
                await pilot.pause()
                editor = app.workflow_editor
                editor.query_one(WorkflowCanvas).set_selected_node("edit")
                await editor.query_one(WorkflowInspector).show(graph, "edit", None)
                await pilot.pause(0.2)
        except InvalidSelectValueError as error:
            result["inspector_error"] = str(error)
        else:
            raise AssertionError("Expected the current implicit-default inspector bug")
    result["scope"] = "Real compiler and Textual inspector; no model execution."
    result["log"] = str(workspace / "expected-crash.log")
    (EVIDENCE / "tui-optional-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    asyncio.run(main())
