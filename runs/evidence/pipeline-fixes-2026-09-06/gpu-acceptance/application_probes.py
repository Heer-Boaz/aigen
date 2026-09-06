"""Acceptance through the installed application backends; no neural doubles."""
from pathlib import Path

from aigen.image_assets import image_asset_json
from aigen.manifest_io import atomic_write_json


def run_image_edit(config, job, directory, progress):
    from aigen.generation.image_edit import ImageEditRequest, run_image_edit

    refs = tuple(Path(path) for path in job.get("references", config["klein"]["references"]))
    atomic_write_json(directory / "inputs.json", {"images": [image_asset_json(p) for p in refs]})
    result = run_image_edit(ImageEditRequest(
        backend=job["model"], prompt=job["prompt"], images=refs,
        output_dir=directory / "output", width=job["width"], height=job["height"],
        seeds=tuple(job.get("seeds", [job["seed"]])),
    ), progress=progress)
    return result.to_json()


def run_upscale(config, job, directory, progress):
    from PIL import Image
    from aigen.generation.image_upscale import IllustrationUpscaler, upscale_model_path
    from aigen.image_io import image_alpha, open_image

    files = tuple((Path(path), directory / f"image-{index}.png")
                  for index, path in enumerate(config["upscale_inputs"]))
    atomic_write_json(directory / "inputs.json", {"images": [image_asset_json(p) for p, _ in files]})
    upscaler = IllustrationUpscaler(model_path=upscale_model_path(job["model"]))
    result = upscaler.upscale_files(files, long_side=1024, progress=progress)
    records = []
    for source, target in files:
        with open_image(source) as original, open_image(target) as output:
            alpha = image_alpha(original)
            if alpha is not None:
                with alpha, alpha.resize(output.size, Image.Resampling.LANCZOS) as expected, output.getchannel("A") as actual:
                    assert actual.tobytes() == expected.tobytes(), "Alpha differs from the deterministic resample"
        records.append(image_asset_json(target))
    return {"model": job["model"], "images": records, "outputs": [
        {"device": r.device, "elapsed_ms": r.elapsed_ms, "scale": r.scale,
         "natural_size": [r.natural_width, r.natural_height], "target_size": [r.target_width, r.target_height]}
        for r in result]}


def run_ltx(config, job, directory, progress):
    from aigen.generation.ltx23_keyframes import Ltx23Keyframe, generate_ltx23_keyframes

    source = Path(config["video"]["inputs"][job["source"]])
    atomic_write_json(directory / "inputs.json", {"images": [image_asset_json(source)]})
    result = generate_ltx23_keyframes(
        prompt=config["video"]["prompt"], keyframes=(Ltx23Keyframe(source, 0), Ltx23Keyframe(source, 32)),
        output=directory / "video.mp4", resolution="768x1024", frames=33, fps=24, steps=8,
        phases=1, solver="distilled_8_steps", negative_prompt=job["negative_prompt"], conditioning_strength=1.0,
        model="nvfp4", seed=71, progress=progress,
    )
    return result.to_json()


def run_anime_tui(config, job, directory, progress):
    import asyncio
    from aigen.workflow_compilation import compile_workflow_run
    from aigen.workflow_document_io import save_workflow_document
    from aigen.workflow_graph import (
        AnimeGenI2VConfig, AnimeGenI2VNode, ExtractVideoFramesConfig, ExtractVideoFramesNode,
        ImageSourceConfig, ImageSourceNode, NodeLayout, NodePortRef, VideoContactSheetConfig,
        VideoContactSheetNode, WorkflowConnection, WorkflowGraph,
    )
    from tui_probe import run_tui_workflow

    source = Path(config["video"]["inputs"][job["source"]])
    atomic_write_json(directory / "inputs.json", {"images": [image_asset_json(source)]})
    nodes = (
        ImageSourceNode(id="source", title="Original reference", config=ImageSourceConfig(path=str(source)), layout=NodeLayout(x=2,y=2)),
        AnimeGenI2VNode(id="video", title="AnimeGen idle loop", layout=NodeLayout(x=34,y=2), config=AnimeGenI2VConfig(
            prompt=config["video"]["prompt"], seed=71, frames=33, fps=16, sampling="lightning-8", steps=8, precision="fp8")),
        VideoContactSheetNode(id="sheet", title="Contact sheet", config=VideoContactSheetConfig(), layout=NodeLayout(x=66,y=2)),
        ExtractVideoFramesNode(id="frames", title="Decoded frames", config=ExtractVideoFramesConfig(), layout=NodeLayout(x=66,y=12)),
    )
    routes = (("source", "image", "video", "start"), ("source", "image", "video", "end"),
              ("video", "video", "sheet", "video"), ("video", "video", "frames", "video"))
    graph = WorkflowGraph(name=f"AnimeGen GPU {job['source']}", nodes=nodes, connections=tuple(
        WorkflowConnection(id=f"wire{index}", source=NodePortRef(node_id=source_id,port=source_port),
                           target=NodePortRef(node_id=target_id,port=target_port))
        for index,(source_id,source_port,target_id,target_port) in enumerate(routes)))
    compile_workflow_run(graph)
    document = directory / "graph.json"
    save_workflow_document(graph, document)
    progress.phase("TUI runs AnimeGen workflow")
    return asyncio.run(run_tui_workflow(document, directory))
