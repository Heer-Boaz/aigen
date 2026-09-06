from __future__ import annotations

from typing import TYPE_CHECKING

from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_graph import (
    ImageEditConfig, ImageEditNode, ImageSourceConfig, ImageSourceNode,
    LoraSourceConfig, LoraSourceNode, NodeLayout, NodePortRef, ReferencePackConfig,
    ReferencePackNode, WorkflowConnection, WorkflowGraph,
    AnimeGenI2VConfig, AnimeGenI2VNode, Ltx23Config, Ltx23Node,
    HunyuanI2VConfig, HunyuanI2VNode, PositionedKeyframeConfig, PositionedKeyframeNode,
)

if TYPE_CHECKING:
    from aigen.image_tui_model import ImageEditForm
    from aigen.video_tui_model import VideoForm


def image_form_workflow(form: ImageEditForm) -> WorkflowGraph:
    """Import the form's settings into the same authored graph used by the editor."""
    values = {field.name: field.value.strip() for field in form.fields if field.slot_kind is None}
    config = ImageEditConfig.model_validate({
        "backend": values["model"],
        **{name: value for name, value in values.items() if name in ImageEditConfig.model_fields and value.strip()},
    })
    edit = ImageEditNode(id="edit", title="Image edit", config=config, layout=NodeLayout(x=42, y=2))
    nodes = [edit]
    wires = []
    references = [field for kind in ("image", "reference_pack") for field in form.fields if field.slot_kind == kind and field.value.strip()]
    for index, field in enumerate(references):
        node_id = f"reference{index + 1}"
        common = dict(id=node_id, title=f"Reference {index + 1}", layout=NodeLayout(x=2, y=2 + index * 12))
        if field.slot_kind == "image":
            node = ImageSourceNode(config=ImageSourceConfig(path=field.value.strip()), **common)
            port = "image"
        else:
            node = ReferencePackNode(config=ReferencePackConfig(path=field.value.strip()), **common)
            port = "pack"
        nodes.append(node)
        wires.append(WorkflowConnection(id=f"wire-{node_id}", source=NodePortRef(node_id=node_id, port=port),
                                        target=NodePortRef(node_id="edit", port="references"), order=index))
    for index, field in enumerate(item for item in form.fields if item.name == "lora" and item.value.strip()):
        weight = next(item.value for item in form.fields if item.name == "lora_weight" and item.slot_id == field.slot_id)
        node_id = f"lora{index + 1}"
        nodes.append(LoraSourceNode(id=node_id, title=f"LoRA {index + 1}", layout=NodeLayout(x=2, y=2 + (len(references) + index) * 12),
                                    config=LoraSourceConfig(path=field.value.strip(), weight=float(weight) if weight.strip() else 1.0)))
        wires.append(WorkflowConnection(id=f"wire-{node_id}", source=NodePortRef(node_id=node_id, port="lora"),
                                        target=NodePortRef(node_id="edit", port="loras"), order=index))
    buffer = WorkflowEditBuffer(WorkflowGraph(name="Image edit variants", nodes=tuple(nodes), connections=tuple(wires)))
    seeds = tuple(int(field.value) for field in form.fields if field.slot_kind == "seed" and field.value.strip()) or (0,)
    buffer.create_image_variants("edit", seeds)
    return buffer.document


def video_form_workflow(form: VideoForm) -> WorkflowGraph:
    from aigen.video_tui_model import ANIMEGEN_BACKEND, HUNYUANVIDEO15_BACKEND, LTX23_BACKEND

    values = {field.name: field.value.strip() for field in form.fields if field.slot_kind is None}
    config_type, node_type = {
        ANIMEGEN_BACKEND: (AnimeGenI2VConfig, AnimeGenI2VNode),
        LTX23_BACKEND: (Ltx23Config, Ltx23Node),
        HUNYUANVIDEO15_BACKEND: (HunyuanI2VConfig, HunyuanI2VNode),
    }[values["backend"]]
    settings = {name: value for name, value in values.items() if name in config_type.model_fields and value}
    seeds = (int(values["seed"]),) if values["backend"] == HUNYUANVIDEO15_BACKEND else tuple(
        int(field.value) for field in form.fields if field.slot_kind == "seed" and field.value.strip())
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Video seeds must be distinct.")
    nodes = []
    wires = []
    sources = []
    fields = [field for field in form.fields if field.name in ("image", "keyframe")]
    for index, field in enumerate(fields):
        # Empty AnimeGen end slots have the same optional meaning as in the CLI.
        if field.name == "image" and not field.value.strip():
            continue
        image_id = f"image{index + 1}"
        nodes.append(ImageSourceNode(id=image_id, title=field.label,
            config=ImageSourceConfig(path=field.value.strip()), layout=NodeLayout(x=2, y=2 + index * 12)))
        if field.name == "keyframe":
            frame = next(item.value for item in form.fields if item.name == "frame" and item.slot_id == field.slot_id)
            keyframe_id = f"keyframe{index + 1}"
            nodes.append(PositionedKeyframeNode(id=keyframe_id, title=f"Frame {frame.strip()}",
                config=PositionedKeyframeConfig(frame=int(frame)), layout=NodeLayout(x=42, y=2 + index * 12)))
            wires.append(WorkflowConnection(id=f"wire-{keyframe_id}", source=NodePortRef(node_id=image_id, port="image"),
                                            target=NodePortRef(node_id=keyframe_id, port="image")))
            sources.append((keyframe_id, "keyframe"))
        else:
            sources.append((image_id, "image"))
    for index, seed in enumerate(seeds):
        node_id = f"video{index + 1}"
        nodes.append(node_type(id=node_id, title=f"{form.field('backend').value} · seed {seed}",
            config=config_type.model_validate({**settings, "seed": seed}), layout=NodeLayout(x=82, y=2 + index * 16)))
        for order, (source_id, port) in enumerate(sources):
            target_port = "keyframes" if values["backend"] == LTX23_BACKEND else "start" if order == 0 else "end"
            wires.append(WorkflowConnection(id=f"wire-{source_id}-{node_id}", source=NodePortRef(node_id=source_id, port=port),
                target=NodePortRef(node_id=node_id, port=target_port), order=order if target_port == "keyframes" else 0))
    return WorkflowGraph(name="Video flow", nodes=tuple(nodes), connections=tuple(wires))
