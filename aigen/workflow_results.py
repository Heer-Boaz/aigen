from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aigen.manifest_io import read_json, sha256_file
from aigen.workflow_artifacts import ImageArtifact, ImageCandidate, WorkflowArtifact
from aigen.workflow_cache import NodeExecutionDetails, NodeExecutionProvenance, WorkflowNodeCache
from aigen.workflow_graph import ArtifactType, ImageResultReference, NodeKind, WorkflowNode, node_definition


RESULT_DISPLAY_TYPES = frozenset((
    ArtifactType.IMAGE, ArtifactType.IMAGE_COLLECTION, ArtifactType.MASK,
    ArtifactType.VIDEO, ArtifactType.AUDIO, ArtifactType.IMAGE_SEQUENCE,
))


def has_viewable_output(node: WorkflowNode) -> bool:
    return any(kind in RESULT_DISPLAY_TYPES for port in node_definition(node.kind).outputs for kind in port.artifact_types)


class NodeResultManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1, 2] = 2
    node_id: str
    node_kind: NodeKind
    signature: str
    outputs: dict[str, WorkflowArtifact]
    title: str = ""
    status: Literal["completed", "reused"] = "completed"
    cache_manifest: str | None = None
    provenance: NodeExecutionProvenance | None = None
    details: NodeExecutionDetails | None = None
    image_origins: dict[str, ImageResultReference] = Field(default_factory=dict)

    def candidate(self, manifest_path: Path, port: str) -> ImageCandidate:
        image = self.outputs[port]
        if not isinstance(image, ImageArtifact):
            raise ValueError(f"{self.node_id}.{port} is not an image")
        reference = self.image_origins.get(port)
        if reference is None:
            reference = ImageResultReference(
                manifest_path=manifest_path.as_posix(),
                producer_signature=self.signature,
                output_port=port,
                artifact_identity=image.identity,
            )
        return ImageCandidate(
            node_id=self.node_id,
            title=self.title or self.node_id,
            image=image,
            reference=reference,
            seed=self.details.measured_outputs.get(port, {}).get("seed", self.details.effective_config.get("seed"))
            if self.details else None,
        )


def load_node_result(path: Path) -> NodeResultManifest:
    result = NodeResultManifest.model_validate(read_json(path, label="workflow node result"))
    if result.version == 1 and result.node_kind not in (NodeKind.IMAGE_SOURCE, NodeKind.REFERENCE_PACK, NodeKind.LORA_SOURCE):
        output = next(iter(result.outputs.values()))
        output_path = Path(output.paths[0] if hasattr(output, "paths") else output.path)
        entry = next((directory for directory in output_path.parents if directory.name == result.signature), None)
        if entry is None:
            raise ValueError(f"legacy generated result has no signature directory: {path}")
        result = result.model_copy(update={"cache_manifest": str(entry / "result.json")})
    return result


def resolve_image_result(reference: ImageResultReference, *, verify_contents: bool = True) -> ImageArtifact:
    """Resolve saved identity; exporters verify content while copying the stream."""
    result = load_node_result(Path(reference.manifest_path))
    if result.signature != reference.producer_signature:
        raise ValueError("selected image producer signature no longer matches its result")
    if reference.output_port not in result.outputs:
        raise ValueError(f"selected output {reference.output_port!r} is absent from its result")
    image = result.outputs[reference.output_port]
    if not isinstance(image, ImageArtifact) or image.identity != reference.artifact_identity:
        raise ValueError("selected image identity no longer matches its result")
    if result.cache_manifest is not None:
        cached = WorkflowNodeCache.read_result(Path(result.cache_manifest), verify_contents=verify_contents)
        cached_image = cached.outputs.get(reference.output_port)
        if (cached.signature != result.signature or not isinstance(cached_image, ImageArtifact)
                or cached_image.path != image.path or cached_image.identity != image.identity
                or (image.content_sha256 is not None and cached_image.content_sha256 != image.content_sha256)):
            raise ValueError("selected image no longer matches its immutable cache entry")
        return cached_image
    if verify_contents and sha256_file(Path(image.path)) != image.identity:
        raise ValueError(f"selected source image changed: {image.path}")
    return image.model_copy(update={"content_sha256": image.identity})


def node_result_history(runs_root: Path, workflow_id: str, node_id: str) -> tuple[Path, ...]:
    """Newest first; read only this workflow's manifests, outside the render loop."""
    history = []
    for state_path in sorted((runs_root / "runs" / workflow_id).glob("*/run.json"), reverse=True):
        state = read_json(state_path, label="workflow run")
        if node_id in state["nodes"]:
            history.append(Path(state["nodes"][node_id]))
    return tuple(history)
