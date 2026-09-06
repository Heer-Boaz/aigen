"""Reviewed real video routes through the saved TUI -> CLI graph, no model doubles."""
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
from aigen.workflow_graph import (
    AnimeGenI2VConfig, AnimeGenI2VNode, AssembleVideoNode, ExtractVideoFramesNode,
    FramePostprocessNode, ImageSelectionConfig, ImageSelectionNode, Ltx23Config, Ltx23Node,
    NodePortRef, PixelArtFixerConfig, PositionedKeyframeConfig, PositionedKeyframeNode,
    WorkflowConnection, WorkflowGraph,
)
from aigen.workflow_results import load_node_result, node_result_history

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "image-flow-acceptance"))
from tui_probe import run_action, select_node


def document(request, backend, selected, second_seed=None):
    config = request[backend]
    nodes = [ImageSelectionNode(id="selected", title="Saved image choice",
                               config=ImageSelectionConfig(selected=selected["reference"]))]
    wires = []
    def connect(start, output, end, port, order=0):
        wires.append(WorkflowConnection(id=f"wire{len(wires)}", source=NodePortRef(node_id=start, port=output),
                                        target=NodePortRef(node_id=end, port=port), order=order))
    if backend == "anime":
        nodes.append(AnimeGenI2VNode(id="generate", title="AnimeGen", config=AnimeGenI2VConfig(
            prompt=request["prompt"], **{key:value for key,value in config.items() if key in AnimeGenI2VConfig.model_fields})))
        connect("selected", "image", "generate", "start")
        connect("selected", "image", "generate", "end")
    else:
        nodes.append(Ltx23Node(id="generate", title="LTX", config=Ltx23Config(
            prompt=request["prompt"], **{key:value for key,value in config.items() if key in Ltx23Config.model_fields})))
        for index, frame in enumerate(config["positions"]):
            node_id=f"keyframe{index}"
            nodes.append(PositionedKeyframeNode(id=node_id,title=f"Frame {frame}",config=PositionedKeyframeConfig(frame=frame)))
            connect("selected", "image", node_id, "image")
            connect(node_id, "keyframe", "generate", "keyframes", index)
    if second_seed is not None:
        original = next(node for node in nodes if node.id == "generate")
        nodes.append(original.model_copy(update={"id":"generate2", "title":f"{original.title} second seed",
            "config":original.config.model_copy(update={"seed":second_seed})}))
        for wire in tuple(wires):
            if wire.target.node_id == "generate":
                connect(wire.source.node_id, wire.source.port, "generate2", wire.target.port, wire.order)
    nodes.extend((ExtractVideoFramesNode(id="extract", title="Extract frames"),
        FramePostprocessNode(id="process", title="CPU frame processing",config=PixelArtFixerConfig(mode="fast",low_memory=False,force_step=2)),
        AssembleVideoNode(id="assemble",title="Finished video")))
    connect("generate", "video", "extract", "video")
    connect("extract", "images", "process", "images")
    connect("process", "images", "assemble", "images")
    buffer=WorkflowEditBuffer(WorkflowGraph(name=f"{backend} video flow acceptance",nodes=nodes,connections=wires))
    buffer.auto_layout()
    return buffer.document


async def run(args):
    base = Path(__file__).resolve().parent
    request = json.loads((base / "manifest.json").read_text())
    assert sha256_file(Path(request["input"])) == request["sha256"]
    selected = json.loads((base.parent / "image-flow-acceptance/klein-v2/selected.json").read_text())
    directory = base / args.attempt
    directory.mkdir()
    path = directory / "workflow.json"
    save_workflow_document(document(request,args.backend,selected,args.second_seed),path)
    atomic_write_json(directory / "request.json", {**request, "second_seed":args.second_seed})
    image_tui.STATE_PATH = directory / "form.json"
    image_tui.DEFAULT_WORKFLOW_RUNS_ROOT = directory / "workflows"
    app = image_tui.ImageGenerationApp()
    app.workflow_buffer = WorkflowEditBuffer(load_workflow_document(path),document_path=path)
    app.workflow_document_paths = [path]
    async with app.run_test(size=(160,54)) as pilot:
        app._open_workflow_editor()
        await pilot.pause()
        editor = app.workflow_editor
        await select_node(app,editor,pilot,"generate")
        generated = await run_action(app,editor,pilot,"#workflow-run" if args.second_seed is not None else "#workflow-run-target",
                                     directory,"generation",request["min_free_mib"])
        assert generated["statuses"]["generate"] == "completed"
        await select_node(app,editor,pilot,"assemble")
        assembled = await run_action(app,editor,pilot,"#workflow-run-target",directory,"assembly",request["min_free_mib"])
        assert assembled["statuses"]["generate"] == "reused", assembled
        assert await pilot.click("#workflow-results")
        await pilot.pause()
        results=app.screen
        await results.workers.wait_for_complete()
        (directory/"result.svg").write_text(app.export_screenshot())
        assert await pilot.click("#result-export")
        await pilot.pause()
        app.screen.query_one(Input).value = str(directory/"export.mp4")
        assert await pilot.click("#dialog-ok")
        await pilot.pause()
        await results.workers.wait_for_complete()
        assert (directory/"export.mp4").is_file()
        node_ids = ("generate", "extract", "process", "assemble") + (("generate2",) if args.second_seed is not None else ())
        histories = {node:node_result_history(directory/"workflows",app.workflow_buffer.document.workflow_id,node)[0]
                     for node in node_ids}
        outputs = {node:load_node_result(manifest).model_dump(mode="json") for node,manifest in histories.items()}
        if args.second_seed is not None:
            assert generated["statuses"]["generate2"] == "completed"
            assert outputs["generate"]["details"]["record_dir"] == outputs["generate2"]["details"]["record_dir"]
        atomic_write_json(directory/"acceptance.json",{"backend":args.backend,"generation":generated,"assembly":assembled,"outputs":outputs})
        print(json.dumps({"status":"completed","backend":args.backend,"output":str(directory/"export.mp4")}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("backend",choices=("anime","ltx"))
    parser.add_argument("attempt")
    parser.add_argument("--second-seed", type=int)
    asyncio.run(run(parser.parse_args()))
