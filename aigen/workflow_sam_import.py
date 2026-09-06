"""Import the SAM selection form into the existing typed workflow graph."""
from __future__ import annotations

from pathlib import Path

from aigen.sam_tui_model import SamEditForm
from aigen.workflow_graph import (
    BindMaskConfig, BindMaskNode, CharacterRefineConfig, CharacterRefineNode,
    ImageSourceConfig, ImageSourceNode, NodeLayout, NodePortRef, ReferencePackConfig,
    ReferencePackNode, SamSegmentConfig, SamSegmentNode, WorkflowConnection, WorkflowGraph,
)


def sam_form_workflow(form: SamEditForm) -> WorkflowGraph:
    def value(name: str) -> str:
        return form.field(name).value.strip()

    source = ImageSourceNode(id="source", title="Source image", layout=NodeLayout(x=2, y=2),
                             config=ImageSourceConfig(path=value("input")))
    nodes = [source]
    wires = []

    def connect(source_id: str, port: str, target: str, target_port: str) -> None:
        wires.append(WorkflowConnection(id=f"wire-{len(wires)}", source=NodePortRef(node_id=source_id, port=port),
                                        target=NodePortRef(node_id=target, port=target_port)))

    operation = value("operation")
    if operation == "segment":
        settings = {name: value(name) for name in SamSegmentConfig.model_fields if name != "mask_candidate"}
        settings["mask_candidate"] = None if value("mask_candidate") == "auto" else int(value("mask_candidate")) - 1
        nodes.append(SamSegmentNode(id="segment", title="SAM selection", layout=NodeLayout(x=42, y=2),
                                    config=SamSegmentConfig.model_validate(settings)))
        connect("source", "image", "segment", "source")
        return WorkflowGraph(name="SAM selection", nodes=tuple(nodes), connections=tuple(wires))
    if operation != "qwen-edit":
        raise ValueError("Open a segmentation or a masked Qwen edit as a workflow")
    from aigen.generation.qwen_image_edit_identity import DEFAULT_QWEN_IDENTITY_PROFILE
    if value("profile") != DEFAULT_QWEN_IDENTITY_PROFILE:
        raise ValueError("masked workflows require the explicit Qwen-2511 Lightning profile")
    if value("padding_mask_crop") or value("nunchaku_blocks_on_gpu"):
        raise ValueError("Qwen-2511 masked workflows use the source canvas and native block offload")
    binding = BindMaskConfig()
    mask_path = value("mask")
    if value("mask_source") == "region-plan":
        from aigen.character_qwen_refine import _resolve_refine_mask
        from aigen.manifest_io import read_json
        path, metadata = _resolve_refine_mask(mask_path=None, region_plan_path=Path(value("region_plan")),
                                              region_name=value("region"))
        mask_path = str(path)
        source_record = read_json(Path(value("region_plan")), label="character region plan")["image"]
        binding = BindMaskConfig(source_sha256=source_record["sha256"],
                                 mask_sha256=metadata["region"]["segmentation"]["mask"]["sha256"])
    nodes.extend((
        ImageSourceNode(id="mask-image", title="Repaint mask", layout=NodeLayout(x=2, y=14),
                        config=ImageSourceConfig(path=mask_path)),
        BindMaskNode(id="mask", title="Source-bound mask", config=binding, layout=NodeLayout(x=42, y=2)),
    ))
    if value("reference_pack"):
        nodes.append(ReferencePackNode(id="references", title="References", layout=NodeLayout(x=42, y=24),
                                      config=ReferencePackConfig(path=value("reference_pack"))))
        connect("references", "pack", "refine", "references")
    settings = {name: value(name) for name in ("seed", "steps", "strength", "max_side", "max_sequence_length", "candidates", "max_iterations") if value(name)}
    settings["prompt"] = value("instruction")
    for name, field in (("guidance", "true_cfg_scale"), ("guidance_scale", "guidance_scale")):
        if value(field):
            settings[name] = float(value(field))
    nodes.append(CharacterRefineNode(id="refine", title="Qwen regional edit", layout=NodeLayout(x=82, y=2),
                                     config=CharacterRefineConfig.model_validate(settings)))
    connect("source", "image", "mask", "source")
    connect("mask-image", "image", "mask", "image")
    connect("source", "image", "refine", "source")
    connect("mask", "mask", "refine", "mask")
    return WorkflowGraph(name="Regional character edit", nodes=tuple(nodes), connections=tuple(wires))
