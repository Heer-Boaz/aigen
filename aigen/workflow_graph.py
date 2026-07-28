from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import Annotated, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


WORKFLOW_DOCUMENT_VERSION = 2
_STABLE_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]*$"
StableId: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=_STABLE_ID_PATTERN),
]


class ArtifactType(StrEnum):
    IMAGE = "image"
    REFERENCE_PACK = "reference-pack"
    LORA = "lora"
    VIDEO = "video"
    IMAGE_SEQUENCE = "image-sequence"


class NodeKind(StrEnum):
    IMAGE_SOURCE = "image-source"
    REFERENCE_PACK = "reference-pack"
    LORA_SOURCE = "lora-source"
    IMAGE_EDIT = "image-edit"
    IMAGE_POSTPROCESS = "image-postprocess"
    ANIMEGEN_I2V = "animegen-i2v"
    VIDEO_CONTACT_SHEET = "video-contact-sheet"
    EXTRACT_VIDEO_FRAMES = "extract-video-frames"
    FRAME_POSTPROCESS = "frame-postprocess"


class WorkflowModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NodeLayout(WorkflowModel):
    x: int = Field(default=0, ge=0)
    y: int = Field(default=0, ge=0)


class ImageSourceConfig(WorkflowModel):
    path: str = ""


class ReferencePackConfig(WorkflowModel):
    path: str = ""


class LoraSourceConfig(WorkflowModel):
    path: str = ""
    weight: float = 1.0


class ImageEditConfig(WorkflowModel):
    backend: str
    prompt: str = ""
    seed: int = 0
    aspect_ratio: str = ""
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    steps: int | None = Field(default=None, gt=0)
    guidance: float | None = None
    strength: float | None = None
    sampler: str | None = None
    scheduler: str | None = None


class VosrPostprocessConfig(WorkflowModel):
    model: Literal["vosr-1.4b-ms-upscale"] = "vosr-1.4b-ms-upscale"
    sizing: Literal["long-side", "scale"]
    long_side: int | None = Field(gt=0)
    scale: int = Field(gt=0)
    infer_steps: int = Field(gt=0)
    cfg_scale: float
    weak_cond_strength_aelq: float
    align_method: Literal["wavelet", "adain", "nofix"]
    tile_size: int = Field(gt=0)
    seed: int


class IllustrationUpscaleConfig(WorkflowModel):
    model: Literal[
        "illustrationjanai-dat2",
        "illustrationjanai-esrgan",
        "animesharp-x4",
    ] = "illustrationjanai-dat2"
    long_side: int | None = Field(gt=0)


class WuPixelizationConfig(WorkflowModel):
    model: Literal["wu-pixelization"] = "wu-pixelization"
    cell_size: int = Field(gt=0)


class PixelArtFixerConfig(WorkflowModel):
    model: Literal["pixel-art-fixer"] = "pixel-art-fixer"
    mode: Literal["full", "fast"]
    low_memory: bool
    force_step: float | None = Field(gt=0)


ImagePostprocessConfig: TypeAlias = Annotated[
    VosrPostprocessConfig
    | IllustrationUpscaleConfig
    | WuPixelizationConfig
    | PixelArtFixerConfig,
    Field(discriminator="model"),
]


class AnimeGenI2VConfig(WorkflowModel):
    prompt: str = ""
    seed: int = 0
    frames: int = Field(gt=0)
    fps: int = Field(gt=0)
    sampling: str
    steps: int | None = Field(default=None, gt=0)
    precision: str


class VideoContactSheetConfig(WorkflowModel):
    pass


class ExtractVideoFramesConfig(WorkflowModel):
    pass


FramePostprocessConfig: TypeAlias = ImagePostprocessConfig


class WorkflowNodeBase(WorkflowModel):
    id: StableId
    title: str = Field(min_length=1, max_length=160)
    layout: NodeLayout = Field(default_factory=NodeLayout)


class ImageSourceNode(WorkflowNodeBase):
    kind: Literal[NodeKind.IMAGE_SOURCE] = NodeKind.IMAGE_SOURCE
    config: ImageSourceConfig = Field(default_factory=ImageSourceConfig)


class ReferencePackNode(WorkflowNodeBase):
    kind: Literal[NodeKind.REFERENCE_PACK] = NodeKind.REFERENCE_PACK
    config: ReferencePackConfig = Field(default_factory=ReferencePackConfig)


class LoraSourceNode(WorkflowNodeBase):
    kind: Literal[NodeKind.LORA_SOURCE] = NodeKind.LORA_SOURCE
    config: LoraSourceConfig = Field(default_factory=LoraSourceConfig)


class ImageEditNode(WorkflowNodeBase):
    kind: Literal[NodeKind.IMAGE_EDIT] = NodeKind.IMAGE_EDIT
    config: ImageEditConfig


class ImagePostprocessNode(WorkflowNodeBase):
    kind: Literal[NodeKind.IMAGE_POSTPROCESS] = NodeKind.IMAGE_POSTPROCESS
    config: ImagePostprocessConfig


class AnimeGenI2VNode(WorkflowNodeBase):
    kind: Literal[NodeKind.ANIMEGEN_I2V] = NodeKind.ANIMEGEN_I2V
    config: AnimeGenI2VConfig


class VideoContactSheetNode(WorkflowNodeBase):
    kind: Literal[NodeKind.VIDEO_CONTACT_SHEET] = NodeKind.VIDEO_CONTACT_SHEET
    config: VideoContactSheetConfig = Field(default_factory=VideoContactSheetConfig)


class ExtractVideoFramesNode(WorkflowNodeBase):
    kind: Literal[NodeKind.EXTRACT_VIDEO_FRAMES] = NodeKind.EXTRACT_VIDEO_FRAMES
    config: ExtractVideoFramesConfig = Field(default_factory=ExtractVideoFramesConfig)


class FramePostprocessNode(WorkflowNodeBase):
    kind: Literal[NodeKind.FRAME_POSTPROCESS] = NodeKind.FRAME_POSTPROCESS
    config: FramePostprocessConfig


WorkflowNode: TypeAlias = Annotated[
    ImageSourceNode
    | ReferencePackNode
    | LoraSourceNode
    | ImageEditNode
    | ImagePostprocessNode
    | AnimeGenI2VNode
    | VideoContactSheetNode
    | ExtractVideoFramesNode
    | FramePostprocessNode,
    Field(discriminator="kind"),
]


class NodePortRef(WorkflowModel):
    node_id: StableId
    port: str = Field(min_length=1, max_length=64, pattern=_STABLE_ID_PATTERN)


class WorkflowConnection(WorkflowModel):
    id: StableId
    source: NodePortRef
    target: NodePortRef
    order: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class PortDefinition:
    name: str
    artifact_types: tuple[ArtifactType, ...]
    required: bool = False
    multiple: bool = False
    label: str = ""


@dataclass(frozen=True)
class NodeDefinition:
    kind: NodeKind
    label: str
    inputs: tuple[PortDefinition, ...] = ()
    outputs: tuple[PortDefinition, ...] = ()

    def input(self, name: str) -> PortDefinition | None:
        return next((port for port in self.inputs if port.name == name), None)

    def output(self, name: str) -> PortDefinition | None:
        return next((port for port in self.outputs if port.name == name), None)


def port_definitions_compatible(
    source: PortDefinition,
    target: PortDefinition,
) -> bool:
    return any(
        artifact_type in target.artifact_types
        for artifact_type in source.artifact_types
    )


_IMAGE_OUTPUT = PortDefinition("image", (ArtifactType.IMAGE,), label="Image")
_VIDEO_OUTPUT = PortDefinition("video", (ArtifactType.VIDEO,), label="Video")
_SEQUENCE_OUTPUT = PortDefinition(
    "images",
    (ArtifactType.IMAGE_SEQUENCE,),
    label="Images",
)

NODE_DEFINITIONS = MappingProxyType(
    {
        NodeKind.IMAGE_SOURCE: NodeDefinition(
            kind=NodeKind.IMAGE_SOURCE,
            label="Image",
            outputs=(_IMAGE_OUTPUT,),
        ),
        NodeKind.REFERENCE_PACK: NodeDefinition(
            kind=NodeKind.REFERENCE_PACK,
            label="Reference pack",
            outputs=(
                PortDefinition(
                    "pack",
                    (ArtifactType.REFERENCE_PACK,),
                    label="Pack",
                ),
            ),
        ),
        NodeKind.LORA_SOURCE: NodeDefinition(
            kind=NodeKind.LORA_SOURCE,
            label="LoRA",
            outputs=(
                PortDefinition(
                    "lora",
                    (ArtifactType.LORA,),
                    label="LoRA",
                ),
            ),
        ),
        NodeKind.IMAGE_EDIT: NodeDefinition(
            kind=NodeKind.IMAGE_EDIT,
            label="Image edit",
            inputs=(
                PortDefinition(
                    "references",
                    (ArtifactType.IMAGE, ArtifactType.REFERENCE_PACK),
                    required=True,
                    multiple=True,
                    label="References",
                ),
                PortDefinition(
                    "loras",
                    (ArtifactType.LORA,),
                    multiple=True,
                    label="LoRAs",
                ),
            ),
            outputs=(_IMAGE_OUTPUT,),
        ),
        NodeKind.IMAGE_POSTPROCESS: NodeDefinition(
            kind=NodeKind.IMAGE_POSTPROCESS,
            label="Image postprocess",
            inputs=(
                PortDefinition(
                    "image",
                    (ArtifactType.IMAGE,),
                    required=True,
                    label="Image",
                ),
            ),
            outputs=(_IMAGE_OUTPUT,),
        ),
        NodeKind.ANIMEGEN_I2V: NodeDefinition(
            kind=NodeKind.ANIMEGEN_I2V,
            label="AnimeGen-I2V",
            inputs=(
                PortDefinition(
                    "start",
                    (ArtifactType.IMAGE,),
                    required=True,
                    label="Start",
                ),
                PortDefinition(
                    "end",
                    (ArtifactType.IMAGE,),
                    label="End",
                ),
            ),
            outputs=(_VIDEO_OUTPUT,),
        ),
        NodeKind.VIDEO_CONTACT_SHEET: NodeDefinition(
            kind=NodeKind.VIDEO_CONTACT_SHEET,
            label="Video contact sheet",
            inputs=(
                PortDefinition(
                    "video",
                    (ArtifactType.VIDEO,),
                    required=True,
                    label="Video",
                ),
            ),
            outputs=(_IMAGE_OUTPUT,),
        ),
        NodeKind.EXTRACT_VIDEO_FRAMES: NodeDefinition(
            kind=NodeKind.EXTRACT_VIDEO_FRAMES,
            label="Extract video frames",
            inputs=(
                PortDefinition(
                    "video",
                    (ArtifactType.VIDEO,),
                    required=True,
                    label="Video",
                ),
            ),
            outputs=(_SEQUENCE_OUTPUT,),
        ),
        NodeKind.FRAME_POSTPROCESS: NodeDefinition(
            kind=NodeKind.FRAME_POSTPROCESS,
            label="Frame postprocess",
            inputs=(
                PortDefinition(
                    "images",
                    (ArtifactType.IMAGE_SEQUENCE,),
                    required=True,
                    label="Images",
                ),
            ),
            outputs=(_SEQUENCE_OUTPUT,),
        ),
    }
)


def node_definition(kind: NodeKind) -> NodeDefinition:
    return NODE_DEFINITIONS[kind]


class WorkflowGraph(WorkflowModel):
    version: Literal[WORKFLOW_DOCUMENT_VERSION] = WORKFLOW_DOCUMENT_VERSION
    name: str = Field(min_length=1, max_length=160)
    nodes: tuple[WorkflowNode, ...] = ()
    connections: tuple[WorkflowConnection, ...] = ()

    @model_validator(mode="after")
    def validate_graph(self) -> WorkflowGraph:
        nodes_by_id = _unique_by_id(self.nodes, "node")
        _unique_by_id(self.connections, "connection")
        connected_inputs: dict[tuple[str, str], int] = {}
        input_orders: dict[tuple[str, str], set[int]] = {}
        seen_routes: set[tuple[str, str, str, str]] = set()
        predecessors = {node.id: set() for node in self.nodes}

        for connection in self.connections:
            source_node = nodes_by_id.get(connection.source.node_id)
            target_node = nodes_by_id.get(connection.target.node_id)
            if source_node is None:
                raise ValueError(
                    f"connection {connection.id!r} references unknown source node "
                    f"{connection.source.node_id!r}"
                )
            if target_node is None:
                raise ValueError(
                    f"connection {connection.id!r} references unknown target node "
                    f"{connection.target.node_id!r}"
                )
            source_port = node_definition(source_node.kind).output(
                connection.source.port
            )
            if source_port is None:
                raise ValueError(
                    f"connection {connection.id!r} references unknown output "
                    f"{connection.source.node_id}.{connection.source.port}"
                )
            target_port = node_definition(target_node.kind).input(
                connection.target.port
            )
            if target_port is None:
                raise ValueError(
                    f"connection {connection.id!r} references unknown input "
                    f"{connection.target.node_id}.{connection.target.port}"
                )
            if not port_definitions_compatible(source_port, target_port):
                source_types = ", ".join(source_port.artifact_types)
                target_types = ", ".join(target_port.artifact_types)
                raise ValueError(
                    f"connection {connection.id!r} is incompatible: "
                    f"{connection.source.node_id}.{connection.source.port} "
                    f"produces {source_types}; "
                    f"{connection.target.node_id}.{connection.target.port} "
                    f"accepts {target_types}"
                )

            route = (
                connection.source.node_id,
                connection.source.port,
                connection.target.node_id,
                connection.target.port,
            )
            if route in seen_routes:
                raise ValueError(
                    f"duplicate connection route to "
                    f"{connection.target.node_id}.{connection.target.port}"
                )
            seen_routes.add(route)
            input_key = (connection.target.node_id, connection.target.port)
            input_count = connected_inputs.get(input_key, 0) + 1
            if input_count > 1 and not target_port.multiple:
                raise ValueError(
                    f"input {connection.target.node_id}.{connection.target.port} "
                    "accepts one connection"
                )
            if not target_port.multiple and connection.order != 0:
                raise ValueError(
                    f"single input {connection.target.node_id}."
                    f"{connection.target.port} requires connection order 0"
                )
            port_orders = input_orders.setdefault(input_key, set())
            if connection.order in port_orders:
                raise ValueError(
                    f"input {connection.target.node_id}.{connection.target.port} "
                    f"has duplicate connection order {connection.order}"
                )
            port_orders.add(connection.order)
            connected_inputs[input_key] = input_count
            predecessors[target_node.id].add(source_node.id)

        try:
            tuple(TopologicalSorter(predecessors).static_order())
        except CycleError as error:
            cycle = " -> ".join(error.args[1]) if len(error.args) > 1 else "unknown"
            raise ValueError(f"workflow graph contains a cycle: {cycle}") from error
        return self

    def node(self, node_id: str) -> WorkflowNode:
        return next(node for node in self.nodes if node.id == node_id)


ItemT = TypeVar("ItemT", WorkflowNodeBase, WorkflowConnection)


def _unique_by_id(
    items: Sequence[ItemT],
    label: str,
) -> dict[str, ItemT]:
    indexed: dict[str, ItemT] = {}
    for item in items:
        if item.id in indexed:
            raise ValueError(f"duplicate {label} id: {item.id!r}")
        indexed[item.id] = item
    return indexed
