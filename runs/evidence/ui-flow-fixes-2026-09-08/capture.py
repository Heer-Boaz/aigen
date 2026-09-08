"""CPU acceptance/screenshots: env -u NO_COLOR COLORTERM=truecolor PYTHONPATH=tests .venv/bin/python <this file>."""
import asyncio
from pathlib import Path
from unittest.mock import patch

from textual import events
from textual.widgets import Button, Label, Select, TabbedContent

from aigen import image_tui
from aigen.manifest_io import atomic_write_json
from aigen.workflow_editor_layout import WorkflowEditorBody
from aigen.workflow_property_widgets import PropertyTextArea
from aigen.workflow_result_widgets import ResultComparison, ResultPreview
from aigen.workflow_results_tui import WorkflowResults
from test_tui_editing import open_app, prompt_row
from test_workflow_interactions import interaction_graph
from test_workflow_properties import open_editor, property_widget, select_node
from test_workflow_results_tui import cpu_graph, finish_process
from aigen.tui_text_editor import MultilineInput


ROOT = Path(__file__).resolve().parent


def capture(app, name):
    (ROOT / f"{name}.svg").write_text(app.export_screenshot(), encoding="utf-8")


async def main():
    metrics = []
    state = ROOT / "forms"
    state.mkdir(exist_ok=True)
    async with open_app(state) as (app, pilot):
        text = prompt_row(app, app.form).query_one(MultilineInput)
        text.load_text("")
        text.focus()
        await pilot.pause()
        app.post_message(events.Paste("Multiline editing example\n\nSecond paragraph with preserved spacing."))
        for width, height in ((80, 24), (120, 40), (160, 50)):
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            capture(app, f"images-{width}x{height}")
        app.query_one(TabbedContent).active = "videos"
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        capture(app, "videos-80x24")

    async with open_editor(interaction_graph(), state, size=(160, 50)) as (app, editor, pilot, _):
        await editor.workers.wait_for_complete()
        capture(app, "canvas-160x50")
        await select_node(app, editor, pilot, "edit")
        text = property_widget(editor, PropertyTextArea, "prompt")
        text.focus()
        app.post_message(events.Paste("Multiline editing example\n\nSecond paragraph with preserved spacing."))
        await pilot.pause()
        body = editor.query_one(WorkflowEditorBody)
        for width, height in ((160, 50), (120, 40), (80, 24)):
            await pilot.resize_terminal(width, height)
            body.show_inspector(True)
            await pilot.pause()
            capture(app, f"inspector-{width}x{height}")
            metrics.append({"screen": "inspector", "terminal": [width, height],
                            "pane": list(editor.query_one("#workflow-inspector-panel").region),
                            "text": list(text.region), "draft": text.text})
        await pilot.resize_terminal(160, 50)
        body.expand_inspector(True)
        await pilot.pause()
        capture(app, "inspector-expanded-160x50")

    for width, height in ((80, 24), (120, 40), (160, 50)):
        directory = ROOT / "cpu-flow" / f"{width}x{height}"
        directory.mkdir(parents=True, exist_ok=True)
        graph = cpu_graph(directory)
        with patch.object(image_tui, "DEFAULT_WORKFLOW_RUNS_ROOT", directory / "runs"):
            async with open_editor(graph, directory, size=(width, height)) as (app, editor, pilot, _):
                await editor.workers.wait_for_complete()
                await pilot.pause()
                stages = [str(editor.query_one("#workflow-run", Button).label)]
                assert stages[-1] == "Generate choices"
                await pilot.click("#workflow-run")
                await pilot.pause()
                await finish_process(app, pilot)
                await editor.workers.wait_for_complete()
                await pilot.pause()
                stages.append(str(editor.query_one("#workflow-run", Button).label))
                assert stages[-1] == "Choose image"
                await pilot.click("#workflow-run")
                await pilot.pause()
                results = app.screen
                assert isinstance(results, WorkflowResults)
                await results.workers.wait_for_complete()
                await pilot.pause()
                capture(app, f"results-{width}x{height}")
                tile = results.query_one("#candidate-1", ResultPreview)
                await pilot.click(offset=(tile.region.x + 3, tile.region.y + 2))
                await results.workers.wait_for_complete()
                assert results.highlighted_index == 1
                if width >= 120:
                    results.query_one("#result-view-mode", Select).value = "compare"
                    await pilot.pause()
                    capture(app, f"comparison-{width}x{height}")
                metrics.append({"screen": "results", "terminal": [width, height],
                                "comparison": list(results.query_one(ResultComparison).region),
                                "candidate": list(tile.region), "identity": str(results.query_one("#result-selected", Label).content),
                                "actions": list(results.query_one("#result-actions").region)})
                await pilot.click("#result-select")
                await pilot.pause()
                await editor.workers.wait_for_complete()
                await pilot.pause()
                stages.append(str(editor.query_one("#workflow-run", Button).label))
                assert stages[-1] == "Continue"
                await pilot.click("#workflow-run")
                await pilot.pause()
                await finish_process(app, pilot)
                await editor.workers.wait_for_complete()
                await pilot.pause()
                stages.append(str(editor.query_one("#workflow-run", Button).label))
                assert stages[-1] == "View results"
                await pilot.click("#workflow-run")
                await pilot.pause()
                assert app.screen.node_id == "next"
                await app.screen.workers.wait_for_complete()
                capture(app, f"completed-{width}x{height}")
                metrics.append({"screen": "execution", "terminal": [width, height], "stages": stages,
                                "chosen": app.workflow_buffer.document.node("choice").config.selected.model_dump(mode="json")})
    atomic_write_json(ROOT / "metrics.json", metrics)


if __name__ == "__main__":
    asyncio.run(main())
