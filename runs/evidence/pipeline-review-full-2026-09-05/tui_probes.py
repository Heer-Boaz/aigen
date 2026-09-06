"""Headless real TUI -> CLI -> CPU pixel-art-fixer -> workflow cache audit."""
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from PIL import Image
from textual.widgets import Button, Label, Static
from aigen import image_tui
from aigen.generation.image_edit import image_edit_backend_settings
from aigen.workflow_canvas import WorkflowCanvas
from aigen.workflow_document_io import load_workflow_document, save_workflow_document
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_graph import (
    AnimeGenI2VNode, ImageEditConfig, ImageEditNode,
    ImageSourceConfig, ImageSourceNode, ImagePostprocessNode, NodeLayout,
    NodePortRef, PixelArtFixerConfig, WorkflowConnection, WorkflowGraph,
)
from aigen.workflow_inspector import PropertyInput, WorkflowInspector
from aigen.workflow_templates import default_animegen_config

EVIDENCE = Path(__file__).resolve().parent


async def main():
    workspace = EVIDENCE / f"tui-data-{uuid4().hex[:8]}"
    workspace.mkdir()
    source = workspace / "source.png"
    pixels = Image.new("RGBA", (16, 16), (30, 80, 120, 0))
    for y in range(4, 12):
        for x in range(4, 12):
            pixels.putpixel((x, y), (200, 90, 20, 255))
    pixels.resize((64, 64), Image.Resampling.NEAREST).save(source)
    graph = WorkflowGraph(name="CPU TUI audit", nodes=(
        ImageSourceNode(id="source", title="Synthetic source", config=ImageSourceConfig(path=str(source)), layout=NodeLayout(x=2, y=2)),
        ImagePostprocessNode(id="fix", title="Native grid", config=PixelArtFixerConfig(mode="fast", low_memory=False, force_step=4.0), layout=NodeLayout(x=34, y=2)),
    ), connections=(WorkflowConnection(id="wire", source=NodePortRef(node_id="source", port="image"), target=NodePortRef(node_id="fix", port="image")),))
    document = workspace / "workflow.json"
    save_workflow_document(graph, document)
    records = {"workspace": str(workspace)}
    with ExitStack() as patches:
        patches.enter_context(patch.object(image_tui, "STATE_PATH", workspace / "form.json"))
        patches.enter_context(patch.object(image_tui, "DEFAULT_WORKFLOW_RUNS_ROOT", workspace / "runs"))
        app = image_tui.ImageGenerationApp()
        app.workflow_buffer = WorkflowEditBuffer(load_workflow_document(document), document_path=document)
        app.workflow_document_paths = [document]
        async with app.run_test(size=(120, 40)) as pilot:
            app._open_workflow_editor()
            await pilot.pause()
            editor = app.workflow_editor
            assert editor is not None
            canvas = editor.query_one(WorkflowCanvas)

            async def select_fix():
                canvas.set_selected_node("fix")
                await editor.query_one(WorkflowInspector).show(app.workflow_buffer.document, "fix", None)
                await pilot.pause()

            await select_fix()
            records["after_selecting_node"] = app.workflow_buffer.document.node("fix").config.model_dump(mode="json")
            assert app.workflow_buffer.document.node("fix").config.mode == "full"
            assert app.workflow_buffer.document.node("fix").config.force_step is None
            force = next(w for w in editor.query(PropertyInput) if w.field_name == "force_step")
            force.value = "2"
            assert await pilot.click("#workflow-save")
            await pilot.pause(0.2)
            assert load_workflow_document(document).node("fix").config.force_step == 2.0
            records["after_save"] = {"dirty": app.workflow_buffer.dirty, "saved": load_workflow_document(document).node("fix").config.model_dump(mode="json"), "live": app.workflow_buffer.document.node("fix").config.model_dump(mode="json")}
            assert app.workflow_buffer.dirty
            assert app.workflow_buffer.document.node("fix").config.force_step is None
            assert await pilot.click("#workflow-undo")
            await pilot.pause(0.2)
            records["after_undo"] = {"config": app.workflow_buffer.document.node("fix").config.model_dump(mode="json"), "can_redo": app.workflow_buffer.can_redo, "undo_label": app.workflow_buffer.undo_label}

            async def run(label):
                assert await pilot.click("#workflow-run")
                await pilot.pause()
                assert app.process is not None
                assert editor.query_one("#workflow-run", Button).disabled
                for _ in range(600):
                    await pilot.pause(0.1)
                    if app.process is None:
                        break
                assert app.process is None, "CPU workflow failed to finish within 60 seconds"
                records[label] = {
                    "statuses": dict(canvas.runtime_statuses),
                    "editor_status": str(editor.query_one("#workflow-editor-status", Label).render()),
                    "main_status": str(app.query_one("#status", Static).render()),
                    "buttons": [str(b.label) for b in editor.query(Button)],
                }
                assert records[label]["statuses"]["fix"] in ("completed", "reused")

            await run("first_execution")
            await run("second_execution")
            assert records["second_execution"]["statuses"]["fix"] == "reused"
            (workspace / "completed-120x40.svg").write_text(app.export_screenshot())
            replacement = workspace / "replacement.png"
            pixels.resize((96, 96), Image.Resampling.NEAREST).save(replacement)
            canvas.set_selected_node("source")
            await editor.query_one(WorkflowInspector).show(app.workflow_buffer.document, "source", None)
            await pilot.pause()
            path_field = next(w for w in editor.query(PropertyInput) if w.field_name == "path")
            path_field.value = str(replacement)
            assert await editor.commit_pending_property()
            await pilot.pause()
            records["after_edit"] = {"source": app.workflow_buffer.document.node("source").config.path, "statuses": dict(canvas.runtime_statuses)}
            # This assertion documents the current bug, not desired behavior.
            assert canvas.runtime_statuses["fix"] == "reused"

            await select_fix()
            force = next(w for w in editor.query(PropertyInput) if w.field_name == "force_step")
            force.value = "invalid"
            previous = app.workflow_buffer.document
            assert not await editor.commit_pending_property()
            assert app.workflow_buffer.document == previous
            force.value = ""
            assert await editor.commit_pending_property()
            records["invalid_draft_is_atomic"] = "passed"
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            (workspace / "editor-80x24.svg").write_text(app.export_screenshot())
            records["80x24"] = {b.id: {"region": list(b.region), "visible": b.visible} for b in editor.query(Button)}
            await pilot.resize_terminal(120, 40)
            qwen_settings = image_edit_backend_settings("qwen-image-edit-2511-base")
            for node in (
                ImageEditNode(id="edit", title="Image settings", config=ImageEditConfig(backend="qwen-image-edit-2511-base", prompt="Keep the image unchanged.", steps=37, guidance=5.25, sampler=qwen_settings.sampler, scheduler=qwen_settings.scheduler)),
                AnimeGenI2VNode(id="anime", title="Video settings", config=default_animegen_config().model_copy(update={"steps": 9})),
            ):
                current = WorkflowGraph(name="Inspector settings audit", nodes=(node,), connections=())
                app._replace_workflow_document(current, document_path=None, status="Inspector audit")
                await pilot.pause(0.2)
                canvas.set_selected_node(node.id)
                await editor.query_one(WorkflowInspector).show(current, node.id, None)
                await pilot.pause(0.2)
                after = app.workflow_buffer.document.node(node.id).config
                records[f"select_{node.id}"] = {"before": node.config.model_dump(mode="json"), "after": after.model_dump(mode="json")}
                assert after.steps != node.config.steps

            app._start_command(
                [sys.executable, "-c", "import signal; print('CPU cancellation probe ready', flush=True); signal.pause()"],
                str(workspace), action_button_id="workflow-run", idle_label="Run", error_title="CPU probe start", running_label=None,
            )
            editor.set_running(True)
            process = app.process
            assert process is not None
            process_group = os.getpgid(process.pid)
            assert process_group == process.pid
            assert await pilot.click("#workflow-stop")
            for _ in range(30):
                await pilot.pause(0.1)
                if app.process is None:
                    break
            assert app.process is None and process.returncode == -15
            records["cancel_owned_process"] = {"pid": process.pid, "process_group": process_group, "returncode": process.returncode}
            records["scope"] = "Real Textual Pilot clicks, save, undo, child CLI, pixel-art-fixer, cache, edits, invalid draft, resize and SIGTERM of an owned CPU subprocess; neural models never run. Current setting-reset bugs are asserted as audit findings."
    (EVIDENCE / "tui-results.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"status": "passed", "workspace": str(workspace), "after_edit": records["after_edit"]}))


if __name__ == "__main__":
    asyncio.run(main())
