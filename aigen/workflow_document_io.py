from __future__ import annotations

from pathlib import Path
from typing import Any

from aigen.manifest_io import atomic_write_json, read_json
from aigen.workflow_graph import WORKFLOW_DOCUMENT_VERSION, WorkflowGraph


def load_workflow_document(path: Path) -> WorkflowGraph:
    payload = read_json(path, label="workflow document")
    if payload.get("version") == 1:
        payload = _upgrade_version_one(payload)
    return WorkflowGraph.model_validate(payload)


def save_workflow_document(document: WorkflowGraph, path: Path) -> None:
    atomic_write_json(
        path,
        document.model_dump(mode="json"),
        sort_keys=False,
    )


def _upgrade_version_one(payload: dict[str, Any]) -> dict[str, Any]:
    upgraded = dict(payload)
    upgraded["version"] = WORKFLOW_DOCUMENT_VERSION
    upgraded["nodes"] = [
        _upgrade_version_one_node(node)
        for node in payload.get("nodes", ())
    ]
    return upgraded


def _upgrade_version_one_node(node: dict[str, Any]) -> dict[str, Any]:
    if node.get("kind") not in (
        "image-postprocess",
        "frame-postprocess",
    ):
        return node
    upgraded = dict(node)
    config = node["config"]
    model = config["model"]
    if model == "vosr-1.4b-ms-upscale":
        fields = (
            "model",
            "sizing",
            "long_side",
            "scale",
            "infer_steps",
            "cfg_scale",
            "weak_cond_strength_aelq",
            "align_method",
            "tile_size",
            "seed",
        )
    elif model in (
        "illustrationjanai-dat2",
        "illustrationjanai-esrgan",
        "animesharp-x4",
    ):
        fields = ("model", "long_side")
    elif model == "wu-pixelization":
        fields = ("model", "cell_size")
    elif model == "pixel-art-fixer":
        fields = ("model", "mode", "low_memory", "force_step")
    else:
        return node
    upgraded["config"] = {
        field: config[field]
        for field in fields
    }
    return upgraded
