from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import Annotated, Literal, TypeAlias, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


WORKFLOW_DOCUMENT_VERSION = 4
_STABLE_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]*$"
StableId: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=_STABLE_ID_PATTERN),
]


class ArtifactType(StrEnum):
    MASK = "mask"
    IMAGE = "image"
    REFERENCE_PACK = "reference-pack"
    LORA = "lora"
    VIDEO = "video"
    IMAGE_SEQUENCE = "image-sequence"
    IMAGE_COLLECTION = "image-collection"
    AUDIO = "audio"
    KEYFRAME = "keyframe"


class NodeKind(StrEnum):
    BIND_MASK = "bind-mask"
    SAM_SEGMENT = "sam-segment"
    CHARACTER_REFINE = "character-refine"
    IMAGE_SOURCE = "image-source"
    REFERENCE_PACK = "reference-pack"
    LORA_SOURCE = "lora-source"
    IMAGE_EDIT = "image-edit"
    CHARACTER_EDIT = "character-edit"
    IMAGE_COLLECTION = "image-collection"
    IMAGE_SELECTION = "image-selection"
    IMAGE_POSTPROCESS = "image-postprocess"
    ANIMEGEN_I2V = "animegen-i2v"
    VIDEO_CONTACT_SHEET = "video-contact-sheet"
    EXTRACT_VIDEO_FRAMES = "extract-video-frames"
    FRAME_POSTPROCESS = "frame-postprocess"
    VIDEO_SOURCE = "video-source"
    AUDIO_SOURCE = "audio-source"
    POSITIONED_KEYFRAME = "positioned-keyframe"
    LTX23 = "ltx23-keyframes"
    HUNYUAN_I2V = "hunyuanvideo15-i2v"
    ASSEMBLE_VIDEO = "assemble-video"


class WorkflowModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NodeLayout(WorkflowModel):
    x: int = Field(default=0, ge=0)
    y: int = Field(default=0, ge=0)


class ImageSourceConfig(WorkflowModel):
    path: str = ""


class VideoSourceConfig(WorkflowModel):
    path: str = ""


class AudioSourceConfig(WorkflowModel):
    path: str = ""
    stream_index: int | None = Field(default=None, ge=0)


class PositionedKeyframeConfig(WorkflowModel):
    frame: int = Field(default=0, ge=0)


class ReferencePackConfig(WorkflowModel):
    path: str = ""


class LoraSourceConfig(WorkflowModel):
    path: str = ""
    weight: float = 1.0


class ImageEditConfig(WorkflowModel):
    backend: str
    prompt: str = ""
    seed_mode: Literal["fixed", "random"] = "fixed"
    seed: int = 0
    aspect_ratio: str = ""
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    steps: int | None = Field(default=None, gt=0)
    guidance: float | None = None
    strength: float | None = None
    sampler: str | None = None
    scheduler: str | None = None


class CharacterEditConfig(ImageEditConfig):
    backend: Literal["flux2-klein", "qwen-image-edit-2511-lightning"] = "flux2-klein"
    candidates: int = Field(default=2, ge=1)
    max_iterations: int = Field(default=2, ge=1)
    upscale_long_side: int | None = Field(default=2048, gt=0)
    pose_mode: Literal["native", "keypoint"] = "native"
    structure_control: Literal["depth", "edge"] = "depth"
    max_sequence_length: int | None = Field(default=None, ge=1, le=1024)
    guidance_scale: float | None = None


class BindMaskConfig(WorkflowModel):
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mask_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class SamSegmentConfig(WorkflowModel):
    engine: Literal["sam2", "sam1", "anime"] = "sam2"
    device: Literal["cuda", "cpu"] = "cuda"
    prompt_mode: Literal["auto", "box", "points", "box+points"] = "auto"
    mask_candidate: int | None = Field(default=None, ge=0, le=2)
    box: str = ""
    positive_points: str = ""
    negative_points: str = ""
    threshold: float = 28.0
    grow: int = 0
    feather: int = Field(default=0, ge=0)
    fill_holes: bool = True
    largest_component: bool = True
    invert: bool = False


class CharacterRefineConfig(WorkflowModel):
    backend: Literal["qwen-image-edit-2511-lightning"] = "qwen-image-edit-2511-lightning"
    prompt: str = ""
    seed_mode: Literal["fixed", "random"] = "fixed"
    seed: int = 0
    steps: int = Field(default=8, ge=1)
    guidance: Literal[1.0] = 1.0
    guidance_scale: Literal[1.0] = 1.0
    strength: float = Field(default=1.0, gt=0, le=1)
    max_side: int | None = Field(default=None, ge=16)
    max_sequence_length: int = Field(default=512, ge=1, le=1024)
    sampler: Literal["flowmatch-euler", "euler-ancestral"] = "flowmatch-euler"
    scheduler: Literal["flowmatch-dynamic-shift", "simple"] = "flowmatch-dynamic-shift"
    candidates: int = Field(default=2, ge=1)
    max_iterations: int = Field(default=2, ge=1)


class ImageResultReference(WorkflowModel):
    manifest_path: str = Field(min_length=1)
    producer_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_port: str = Field(min_length=1)
    artifact_identity: str = Field(min_length=1)


class ImageCollectionConfig(WorkflowModel):
    pass


class ImageSelectionConfig(WorkflowModel):
    selected: ImageResultReference | None = None


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
    seed_mode: Literal["fixed", "random"] = "fixed"
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
    seed_mode: Literal["fixed", "random"] = "fixed"
    seed: int = 0
    frames: int = Field(gt=0)
    fps: int = Field(gt=0)
    sampling: str
    steps: int | None = Field(default=None, gt=0)
    precision: str
    keyframe_fit: Literal["crop", "pad", "stretch"] = "stretch"


class Ltx23Config(WorkflowModel):
    prompt: str = ""
    negative_prompt: str = ""
    seed_mode: Literal["fixed", "random"] = "fixed"
    seed: int = 0
    resolution: str = "640x640"
    frames: int = Field(default=121, gt=0)
    fps: int = Field(default=24, gt=0)
    steps: int = Field(default=15, gt=0)
    phases: int = 1
    solver: str = "res2s"
    conditioning_strength: float = 1.0
    model: str = "nvfp4"
    keyframe_fit: Literal["crop", "pad", "stretch"] = "crop"


class HunyuanI2VConfig(WorkflowModel):
    prompt: str = ""
    seed_mode: Literal["fixed", "random"] = "fixed"
    seed: int = 0
    frames: int = Field(default=49, gt=0)
    steps: int = 8
    overlap_group_offloading: bool = True


class AssembleVideoConfig(WorkflowModel):
    audio_policy: Literal["preserve", "remove", "replace"] = "preserve"
    background: str = "black"


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


class VideoSourceNode(WorkflowNodeBase):
    kind: Literal[NodeKind.VIDEO_SOURCE] = NodeKind.VIDEO_SOURCE
    config: VideoSourceConfig = Field(default_factory=VideoSourceConfig)


class AudioSourceNode(WorkflowNodeBase):
    kind: Literal[NodeKind.AUDIO_SOURCE] = NodeKind.AUDIO_SOURCE
    config: AudioSourceConfig = Field(default_factory=AudioSourceConfig)


class PositionedKeyframeNode(WorkflowNodeBase):
    kind: Literal[NodeKind.POSITIONED_KEYFRAME] = NodeKind.POSITIONED_KEYFRAME
    config: PositionedKeyframeConfig = Field(default_factory=PositionedKeyframeConfig)


class ReferencePackNode(WorkflowNodeBase):
    kind: Literal[NodeKind.REFERENCE_PACK] = NodeKind.REFERENCE_PACK
    config: ReferencePackConfig = Field(default_factory=ReferencePackConfig)


class LoraSourceNode(WorkflowNodeBase):
    kind: Literal[NodeKind.LORA_SOURCE] = NodeKind.LORA_SOURCE
    config: LoraSourceConfig = Field(default_factory=LoraSourceConfig)


class ImageEditNode(WorkflowNodeBase):
    kind: Literal[NodeKind.IMAGE_EDIT] = NodeKind.IMAGE_EDIT
    config: ImageEditConfig


class CharacterEditNode(WorkflowNodeBase):
    kind: Literal[NodeKind.CHARACTER_EDIT] = NodeKind.CHARACTER_EDIT
    config: CharacterEditConfig = Field(default_factory=CharacterEditConfig)


class BindMaskNode(WorkflowNodeBase):
    kind: Literal[NodeKind.BIND_MASK] = NodeKind.BIND_MASK
    config: BindMaskConfig = Field(default_factory=BindMaskConfig)


class SamSegmentNode(WorkflowNodeBase):
    kind: Literal[NodeKind.SAM_SEGMENT] = NodeKind.SAM_SEGMENT
    config: SamSegmentConfig = Field(default_factory=SamSegmentConfig)


class CharacterRefineNode(WorkflowNodeBase):
    kind: Literal[NodeKind.CHARACTER_REFINE] = NodeKind.CHARACTER_REFINE
    config: CharacterRefineConfig = Field(default_factory=CharacterRefineConfig)


class ImageCollectionNode(WorkflowNodeBase):
    kind: Literal[NodeKind.IMAGE_COLLECTION] = NodeKind.IMAGE_COLLECTION
    config: ImageCollectionConfig = Field(default_factory=ImageCollectionConfig)


class ImageSelectionNode(WorkflowNodeBase):
    kind: Literal[NodeKind.IMAGE_SELECTION] = NodeKind.IMAGE_SELECTION
    config: ImageSelectionConfig = Field(default_factory=ImageSelectionConfig)


class ImagePostprocessNode(WorkflowNodeBase):
    kind: Literal[NodeKind.IMAGE_POSTPROCESS] = NodeKind.IMAGE_POSTPROCESS
    config: ImagePostprocessConfig


class AnimeGenI2VNode(WorkflowNodeBase):
    kind: Literal[NodeKind.ANIMEGEN_I2V] = NodeKind.ANIMEGEN_I2V
    config: AnimeGenI2VConfig


class Ltx23Node(WorkflowNodeBase):
    kind: Literal[NodeKind.LTX23] = NodeKind.LTX23
    config: Ltx23Config = Field(default_factory=Ltx23Config)


class HunyuanI2VNode(WorkflowNodeBase):
    kind: Literal[NodeKind.HUNYUAN_I2V] = NodeKind.HUNYUAN_I2V
    config: HunyuanI2VConfig = Field(default_factory=HunyuanI2VConfig)


class AssembleVideoNode(WorkflowNodeBase):
    kind: Literal[NodeKind.ASSEMBLE_VIDEO] = NodeKind.ASSEMBLE_VIDEO
    config: AssembleVideoConfig = Field(default_factory=AssembleVideoConfig)


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
    | CharacterEditNode
    | BindMaskNode
    | SamSegmentNode
    | CharacterRefineNode
    | ImageCollectionNode
    | ImageSelectionNode
    | ImagePostprocessNode
    | AnimeGenI2VNode
    | VideoContactSheetNode
    | ExtractVideoFramesNode
    | FramePostprocessNode
    | VideoSourceNode
    | AudioSourceNode
    | PositionedKeyframeNode
    | Ltx23Node
    | HunyuanI2VNode
    | AssembleVideoNode,
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
        NodeKind.CHARACTER_EDIT: NodeDefinition(
            kind=NodeKind.CHARACTER_EDIT, label="Character edit",
            inputs=(
                PortDefinition("references", (ArtifactType.IMAGE, ArtifactType.REFERENCE_PACK), required=True, multiple=True, label="References"),
                PortDefinition("loras", (ArtifactType.LORA,), multiple=True, label="LoRAs"),
                PortDefinition("pose", (ArtifactType.IMAGE,), label="Pose source"),
                PortDefinition("scene", (ArtifactType.IMAGE,), label="Scene source"),
            ),
            outputs=(_IMAGE_OUTPUT,),
        ),
        NodeKind.VIDEO_SOURCE: NodeDefinition(kind=NodeKind.VIDEO_SOURCE, label="Video file", outputs=(_VIDEO_OUTPUT,)),
        NodeKind.AUDIO_SOURCE: NodeDefinition(kind=NodeKind.AUDIO_SOURCE, label="Audio track",
            outputs=(PortDefinition("audio", (ArtifactType.AUDIO,), label="Audio"),)),
        NodeKind.POSITIONED_KEYFRAME: NodeDefinition(kind=NodeKind.POSITIONED_KEYFRAME, label="Positioned keyframe",
            inputs=(PortDefinition("image", (ArtifactType.IMAGE,), required=True, label="Image"),),
            outputs=(PortDefinition("keyframe", (ArtifactType.KEYFRAME,), label="Keyframe"),)),
        NodeKind.LTX23: NodeDefinition(kind=NodeKind.LTX23, label="LTX-2.3",
            inputs=(PortDefinition("keyframes", (ArtifactType.KEYFRAME,), required=True, multiple=True, label="Keyframes"),),
            outputs=(_VIDEO_OUTPUT,)),
        NodeKind.HUNYUAN_I2V: NodeDefinition(kind=NodeKind.HUNYUAN_I2V, label="HunyuanVideo-1.5",
            inputs=(PortDefinition("start", (ArtifactType.IMAGE,), required=True, label="Start"),), outputs=(_VIDEO_OUTPUT,)),
        NodeKind.ASSEMBLE_VIDEO: NodeDefinition(kind=NodeKind.ASSEMBLE_VIDEO, label="Assemble video",
            inputs=(PortDefinition("images", (ArtifactType.IMAGE_SEQUENCE,), required=True, label="Frames"),
                    PortDefinition("audio", (ArtifactType.AUDIO,), label="Replacement audio")), outputs=(_VIDEO_OUTPUT,)),
        NodeKind.IMAGE_SOURCE: NodeDefinition(
            kind=NodeKind.IMAGE_SOURCE,
            label="Image",
            outputs=(_IMAGE_OUTPUT,),
        ),
        NodeKind.BIND_MASK: NodeDefinition(kind=NodeKind.BIND_MASK, label="Bind mask to source",
            inputs=(PortDefinition("source", (ArtifactType.IMAGE,), required=True, label="Source"),
                    PortDefinition("image", (ArtifactType.IMAGE,), required=True, label="Mask image")),
            outputs=(PortDefinition("mask", (ArtifactType.MASK,), label="Mask"),)),
        NodeKind.SAM_SEGMENT: NodeDefinition(kind=NodeKind.SAM_SEGMENT, label="SAM segmentation",
            inputs=(PortDefinition("source", (ArtifactType.IMAGE,), required=True, label="Source"),),
            outputs=(PortDefinition("mask", (ArtifactType.MASK,), label="Mask"),
                     PortDefinition("cutout", (ArtifactType.IMAGE,), label="Cutout"),
                     PortDefinition("preview", (ArtifactType.IMAGE,), label="Preview"))),
        NodeKind.CHARACTER_REFINE: NodeDefinition(kind=NodeKind.CHARACTER_REFINE, label="Qwen regional edit",
            inputs=(PortDefinition("source", (ArtifactType.IMAGE,), required=True, label="Source"),
                    PortDefinition("mask", (ArtifactType.MASK,), required=True, label="Mask"),
                    PortDefinition("references", (ArtifactType.IMAGE, ArtifactType.REFERENCE_PACK), multiple=True, label="References"),
                    PortDefinition("loras", (ArtifactType.LORA,), multiple=True, label="LoRAs")),
            outputs=(_IMAGE_OUTPUT,)),
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
        NodeKind.IMAGE_COLLECTION: NodeDefinition(
            kind=NodeKind.IMAGE_COLLECTION,
            label="Image variants",
            inputs=(PortDefinition("images", (ArtifactType.IMAGE,), required=True, multiple=True, label="Candidates"),),
            outputs=(PortDefinition("collection", (ArtifactType.IMAGE_COLLECTION,), label="Variants"),),
        ),
        NodeKind.IMAGE_SELECTION: NodeDefinition(
            kind=NodeKind.IMAGE_SELECTION,
            label="Selected image",
            inputs=(PortDefinition("collection", (ArtifactType.IMAGE_COLLECTION,), label="Choose from"),),
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
    workflow_id: StableId = Field(default_factory=lambda: f"workflow-{uuid4().hex}")
    name: str = Field(min_length=1, max_length=160)
    nodes: tuple[WorkflowNode, ...] = ()
    connections: tuple[WorkflowConnection, ...] = ()

    def execution_inputs(self) -> dict[str, dict[str, tuple[WorkflowConnection, ...]]]:
        """A saved image choice is an artifact boundary, not a fresh generation."""
        incoming = self.incoming_connections()
        for node in self.nodes:
            if isinstance(node, ImageSelectionNode):
                incoming[node.id] = {}
            elif isinstance(node, AssembleVideoNode) and node.config.audio_policy != "replace":
                incoming[node.id].pop("audio", None)
        return incoming

    def execution_targets(self, targets: Sequence[str] | None = None) -> tuple[str, ...]:
        nodes = {node.id for node in self.nodes}
        if targets is None:
            connected = {wire.source.node_id for wire in self.connections}
            targets = tuple(node.id for node in self.nodes if node.id not in connected)
        unknown = set(targets) - nodes
        if unknown:
            raise ValueError(f"unknown workflow targets: {', '.join(sorted(unknown))}")
        return tuple(targets)

    def execution_scope(self, targets: Sequence[str] | None = None) -> tuple[str, ...]:
        incoming = self.execution_inputs()
        scope: set[str] = set()
        work = list(self.execution_targets(targets))
        while work:
            node_id = work.pop()
            if node_id in scope:
                continue
            scope.add(node_id)
            work.extend(wire.source.node_id for wires in incoming[node_id].values() for wire in wires)
        return tuple(node.id for node in self.nodes if node.id in scope)

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

    def incoming_connections(self) -> dict[str, dict[str, tuple[WorkflowConnection, ...]]]:
        incoming: dict[str, dict[str, list[WorkflowConnection]]] = {
            node.id: {} for node in self.nodes
        }
        for connection in self.connections:
            incoming[connection.target.node_id].setdefault(connection.target.port, []).append(connection)
        return {
            node_id: {
                port: tuple(sorted(connections, key=lambda connection: connection.order))
                for port, connections in ports.items()
            }
            for node_id, ports in incoming.items()
        }


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
