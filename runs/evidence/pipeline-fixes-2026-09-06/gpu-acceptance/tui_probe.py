"""Real Textual clicks, CLI process and GPU workflow; only evidence paths are isolated."""
import time
from dataclasses import asdict
import json

from aigen import image_tui
from aigen.manifest_io import atomic_write_json
from aigen.workflow_canvas import WorkflowCanvas
from aigen.workflow_document_io import load_workflow_document
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_inspector import WorkflowInspector


async def run_tui_workflow(document, directory):
    image_tui.STATE_PATH = directory / "form-state.json"
    image_tui.DEFAULT_WORKFLOW_RUNS_ROOT = directory / "workflows"
    app = image_tui.ImageGenerationApp()
    app.workflow_buffer = WorkflowEditBuffer(load_workflow_document(document), document_path=document)
    app.workflow_document_paths = [document]
    authored = app.workflow_buffer.document.model_dump(mode="json")
    records = {}
    async with app.run_test(size=(140, 46)) as pilot:
        app._open_workflow_editor()
        await pilot.pause()
        editor = app.workflow_editor
        canvas = editor.query_one(WorkflowCanvas)
        canvas.set_selected_node("video")
        await editor.query_one(WorkflowInspector).show(app.workflow_buffer.document, "video", None)
        await pilot.pause()
        assert app.workflow_buffer.document.model_dump(mode="json") == authored
        for label in ("generation", "cache"):
            assert await pilot.click("#workflow-run")
            await pilot.pause()
            process = app.process
            assert process is not None
            deadline = time.monotonic() + 1800
            previous_progress = None
            while app.process is not None and time.monotonic() < deadline:
                await pilot.pause(1)
                state = app.generation_progress
                if state is not None and state != previous_progress:
                    with (directory / "tui-progress.jsonl").open("a") as log:
                        log.write(json.dumps({"run": label, **asdict(state)}) + "\n")
                    previous_progress = state
            assert app.process is None, "Workflow exceeded the 30-minute acceptance limit"
            records[label] = {"returncode": process.returncode, "statuses": dict(canvas.runtime_statuses),
                              "stdout_closed": process.stdout.closed}
            atomic_write_json(directory / "tui-results.json", records)
            assert process.returncode == 0, records[label]
            assert canvas.runtime_statuses["video"] == ("completed" if label == "generation" else "reused")
            assert process.stdout.closed
            assert app.workflow_buffer.document.model_dump(mode="json") == authored
        (directory / "tui.svg").write_text(app.export_screenshot())
    return records
