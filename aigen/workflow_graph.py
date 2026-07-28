from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from graphlib import CycleError, TopologicalSorter
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, TypeAlias, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aigen.character_reference_models import CharacterReferenceError
from aigen.character_reference_pack import load_character_reference_pack
from aigen.generation.animegen_i2v import (
    ANIMEGEN_DEFAULT_FPS,
    ANIMEGEN_DEFAULT_FRAMES,
    ANIMEGEN_DEFAULT_PRECISION,
    ANIMEGEN_DEFAULT_SAMPLING,
    AnimeGenI2VError,
    animegen_sampling_profile,
    resolve_animegen_settings,
)
from aigen.generation.image_batch_postprocess import (
    IMAGE_BATCH_DEFAULT_CELL_SIZE,
    IMAGE_BATCH_DEFAULT_FIXER_MODE,
    IMAGE_BATCH_DEFAULT_LOW_MEMORY,
    PIXEL_ART_FIXER_MODEL,
    WU_PIXELIZATION_MODEL,
    image_batch_postprocess_model_names,
)
from aigen.generation.image_upscale import upscale_model_names
from aigen.generation.vosr_backend import (
    VOSR_DEFAULT_ALIGN_METHOD,
    VOSR_DEFAULT_CFG_SCALE,
    VOSR_DEFAULT_INFER_STEPS,
    VOSR_DEFAULT_SCALE,
    VOSR_DEFAULT_SEED,
    VOSR_DEFAULT_TILE_SIZE,
    VOSR_DEFAULT_WEAK_COND_STRENGTH_AELQ,
    VOSR_POSTPROCESS_NAME,
)
from aigen.image_edit_commands import (
    IMAGE_EDIT_BACKENDS,
    ImageEditCommandError,
    image_edit_backend_settings,
    resolve_image_edit_settings,
)
from aigen.image_dimensions import parse_aspect_ratio
from aigen.manifest_io import ManifestIOError, atomic_write_json
from aigen.runtime_profiles import resolve_project_path


WORKFLOW_DOCUMENT_VERSION = 1
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
    model_config = ConfigDict(extra="forbid")


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
    backend: str = IMAGE_EDIT_BACKENDS[0]
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

    @model_validator(mode="before")
    @classmethod
    def apply_backend_defaults(
        cls,
        value: object,
    ) -> object:
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        backend = str(payload.get("backend", IMAGE_EDIT_BACKENDS[0]))
        try:
            settings = image_edit_backend_settings(backend)
        except ImageEditCommandError as error:
            raise ValueError(str(error)) from error
        payload.setdefault("steps", settings.steps)
        payload.setdefault("guidance", settings.guidance)
        payload.setdefault("strength", settings.strength)
        payload.setdefault("sampler", settings.sampler)
        payload.setdefault("scheduler", settings.scheduler)
        return payload


class ImagePostprocessConfig(WorkflowModel):
    model: str = VOSR_POSTPROCESS_NAME
    sizing: Literal["long-side", "scale"] = "long-side"
    long_side: int | None = Field(default=2048, gt=0)
    scale: int = Field(default=VOSR_DEFAULT_SCALE, gt=0)
    infer_steps: int = Field(default=VOSR_DEFAULT_INFER_STEPS, gt=0)
    cfg_scale: float = VOSR_DEFAULT_CFG_SCALE
    weak_cond_strength_aelq: float = VOSR_DEFAULT_WEAK_COND_STRENGTH_AELQ
    align_method: Literal["wavelet", "adain", "nofix"] = VOSR_DEFAULT_ALIGN_METHOD
    tile_size: int = Field(default=VOSR_DEFAULT_TILE_SIZE, gt=0)
    seed: int = VOSR_DEFAULT_SEED
    cell_size: int = Field(default=IMAGE_BATCH_DEFAULT_CELL_SIZE, gt=0)
    mode: Literal["full", "fast"] = IMAGE_BATCH_DEFAULT_FIXER_MODE
    low_memory: bool = IMAGE_BATCH_DEFAULT_LOW_MEMORY
    force_step: float | None = Field(default=None, gt=0)


class AnimeGenI2VConfig(WorkflowModel):
    prompt: str = ""
    seed: int = 0
    frames: int = Field(default=ANIMEGEN_DEFAULT_FRAMES, gt=0)
    fps: int = Field(default=ANIMEGEN_DEFAULT_FPS, gt=0)
    sampling: str = ANIMEGEN_DEFAULT_SAMPLING
    steps: int = Field(gt=0)
    precision: str = ANIMEGEN_DEFAULT_PRECISION

    @model_validator(mode="before")
    @classmethod
    def apply_sampling_defaults(
        cls,
        value: object,
    ) -> object:
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        sampling = str(
            payload.get("sampling", ANIMEGEN_DEFAULT_SAMPLING)
        )
        try:
            profile = animegen_sampling_profile(sampling)
        except AnimeGenI2VError as error:
            raise ValueError(str(error)) from error
        payload.setdefault("steps", profile.steps)
        return payload


class VideoContactSheetConfig(WorkflowModel):
    pass


class ExtractVideoFramesConfig(WorkflowModel):
    pass


class FramePostprocessConfig(ImagePostprocessConfig):
    pass


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
    config: ImageEditConfig = Field(default_factory=ImageEditConfig)


class ImagePostprocessNode(WorkflowNodeBase):
    kind: Literal[NodeKind.IMAGE_POSTPROCESS] = NodeKind.IMAGE_POSTPROCESS
    config: ImagePostprocessConfig = Field(default_factory=ImagePostprocessConfig)


class AnimeGenI2VNode(WorkflowNodeBase):
    kind: Literal[NodeKind.ANIMEGEN_I2V] = NodeKind.ANIMEGEN_I2V
    config: AnimeGenI2VConfig = Field(default_factory=AnimeGenI2VConfig)


class VideoContactSheetNode(WorkflowNodeBase):
    kind: Literal[NodeKind.VIDEO_CONTACT_SHEET] = NodeKind.VIDEO_CONTACT_SHEET
    config: VideoContactSheetConfig = Field(default_factory=VideoContactSheetConfig)


class ExtractVideoFramesNode(WorkflowNodeBase):
    kind: Literal[NodeKind.EXTRACT_VIDEO_FRAMES] = NodeKind.EXTRACT_VIDEO_FRAMES
    config: ExtractVideoFramesConfig = Field(default_factory=ExtractVideoFramesConfig)


class FramePostprocessNode(WorkflowNodeBase):
    kind: Literal[NodeKind.FRAME_POSTPROCESS] = NodeKind.FRAME_POSTPROCESS
    config: FramePostprocessConfig = Field(default_factory=FramePostprocessConfig)


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

_NODE_CLASSES: Mapping[NodeKind, type[WorkflowNodeBase]] = MappingProxyType(
    {
        NodeKind.IMAGE_SOURCE: ImageSourceNode,
        NodeKind.REFERENCE_PACK: ReferencePackNode,
        NodeKind.LORA_SOURCE: LoraSourceNode,
        NodeKind.IMAGE_EDIT: ImageEditNode,
        NodeKind.IMAGE_POSTPROCESS: ImagePostprocessNode,
        NodeKind.ANIMEGEN_I2V: AnimeGenI2VNode,
        NodeKind.VIDEO_CONTACT_SHEET: VideoContactSheetNode,
        NodeKind.EXTRACT_VIDEO_FRAMES: ExtractVideoFramesNode,
        NodeKind.FRAME_POSTPROCESS: FramePostprocessNode,
    }
)


def node_definition(kind: NodeKind) -> NodeDefinition:
    return NODE_DEFINITIONS[kind]


def create_node(
    kind: NodeKind,
    *,
    x: int = 0,
    y: int = 0,
    node_id: str | None = None,
    title: str | None = None,
) -> WorkflowNode:
    definition = node_definition(kind)
    node_class = _NODE_CLASSES[kind]
    return cast(
        WorkflowNode,
        node_class(
            id=node_id or f"node-{uuid.uuid4().hex}",
            title=title or definition.label,
            layout=NodeLayout(x=x, y=y),
        ),
    )


class WorkflowGraph(WorkflowModel):
    version: Literal[WORKFLOW_DOCUMENT_VERSION] = WORKFLOW_DOCUMENT_VERSION
    name: str = Field(min_length=1, max_length=160)
    nodes: list[WorkflowNode] = Field(default_factory=list)
    connections: list[WorkflowConnection] = Field(default_factory=list)

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
            if not set(source_port.artifact_types).intersection(
                target_port.artifact_types
            ):
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

    @classmethod
    def load(cls, path: Path) -> WorkflowGraph:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        validated = type(self).model_validate(self.model_dump(mode="python"))
        atomic_write_json(
            path,
            validated.model_dump(mode="json"),
            sort_keys=False,
        )

    def node(self, node_id: str) -> WorkflowNode:
        return next(node for node in self.nodes if node.id == node_id)

    def validate_executable(self) -> WorkflowGraph:
        connected_inputs = {
            (connection.target.node_id, connection.target.port)
            for connection in self.connections
        }
        for node in self.nodes:
            for port in node_definition(node.kind).inputs:
                if port.required and (node.id, port.name) not in connected_inputs:
                    raise ValueError(f"required input {node.id}.{port.name} is not connected")
            if isinstance(node, (ImageSourceNode, ReferencePackNode, LoraSourceNode)):
                if not node.config.path.strip():
                    raise ValueError(f"source path is required for node {node.id}")
                source_path = resolve_project_path(node.config.path)
                if not source_path.is_file():
                    raise ValueError(
                        f"source file does not exist for node {node.id}: "
                        f"{source_path}"
                    )
                if isinstance(node, ReferencePackNode):
                    try:
                        load_character_reference_pack(source_path)
                    except (CharacterReferenceError, ManifestIOError) as error:
                        raise ValueError(
                            f"invalid reference pack for node {node.id}: {error}"
                        ) from error
            elif isinstance(node, ImageEditNode):
                try:
                    settings = image_edit_backend_settings(node.config.backend)
                    resolve_image_edit_settings(
                        backend=node.config.backend,
                        width=node.config.width,
                        height=node.config.height,
                        aspect_ratio=(
                            parse_aspect_ratio(node.config.aspect_ratio)
                            if node.config.aspect_ratio.strip()
                            else None
                        ),
                        steps=node.config.steps,
                        guidance=node.config.guidance,
                        strength=node.config.strength,
                        sampler=node.config.sampler,
                        scheduler=node.config.scheduler,
                    )
                except (ImageEditCommandError, ValueError) as error:
                    raise ValueError(
                        f"invalid image-edit settings for node {node.id}: {error}"
                    ) from error
                if not node.config.prompt.strip() and not settings.supports_empty_prompt:
                    raise ValueError(f"image-edit prompt is required for node {node.id}")
            elif isinstance(node, (ImagePostprocessNode, FramePostprocessNode)):
                if node.config.model not in image_batch_postprocess_model_names():
                    raise ValueError(
                        f"unknown postprocess model for node {node.id}: "
                        f"{node.config.model!r}"
                    )
                if (
                    node.config.model == VOSR_POSTPROCESS_NAME
                    and node.config.sizing == "long-side"
                    and node.config.long_side is None
                ):
                    raise ValueError(
                        f"VOSR long-side sizing requires a long side for node "
                        f"{node.id}"
                    )
            elif isinstance(node, AnimeGenI2VNode):
                if not node.config.prompt.strip():
                    raise ValueError(f"AnimeGen motion prompt is required for node {node.id}")
                try:
                    resolve_animegen_settings(
                        frames=node.config.frames,
                        fps=node.config.fps,
                        sampling=node.config.sampling,
                        steps=node.config.steps,
                        precision=node.config.precision,
                    )
                except AnimeGenI2VError as error:
                    raise ValueError(
                        f"invalid AnimeGen settings for node {node.id}: {error}"
                    ) from error
        return self

    def execution_order(self) -> tuple[str, ...]:
        self.validate_executable()
        predecessors = {node.id: set() for node in self.nodes}
        for connection in self.connections:
            predecessors[connection.target.node_id].add(connection.source.node_id)
        return tuple(TopologicalSorter(predecessors).static_order())

    def incoming_connections(
        self,
        node_id: str,
        port: str | None = None,
    ) -> tuple[WorkflowConnection, ...]:
        return tuple(
            sorted(
                (
                    connection
                    for connection in self.connections
                    if connection.target.node_id == node_id
                    and (port is None or connection.target.port == port)
                ),
                key=lambda connection: (
                    connection.order,
                    connection.target.port,
                ),
            )
        )

    def validated_incoming_connections(
        self,
    ) -> dict[str, dict[str, tuple[WorkflowConnection, ...]]]:
        self.validate_executable()
        return self._incoming_connections()

    def validated_execution_plan(
        self,
    ) -> tuple[
        tuple[str, ...],
        dict[str, dict[str, tuple[WorkflowConnection, ...]]],
    ]:
        self.validate_executable()
        predecessors = {node.id: set() for node in self.nodes}
        for connection in self.connections:
            predecessors[connection.target.node_id].add(
                connection.source.node_id
            )
        return (
            tuple(TopologicalSorter(predecessors).static_order()),
            self._incoming_connections(),
        )

    def _incoming_connections(
        self,
    ) -> dict[str, dict[str, tuple[WorkflowConnection, ...]]]:
        incoming: dict[str, dict[str, list[WorkflowConnection]]] = {
            node.id: {} for node in self.nodes
        }
        for connection in self.connections:
            ports = incoming[connection.target.node_id]
            ports.setdefault(connection.target.port, []).append(connection)
        return {
            node_id: {
                port: tuple(
                    sorted(connections, key=lambda connection: connection.order)
                )
                for port, connections in ports.items()
            }
            for node_id, ports in incoming.items()
        }

    def execution_digest(self) -> str:
        payload = {
            "version": self.version,
            "nodes": [
                {
                    "id": node.id,
                    "kind": node.kind,
                    "config": node_execution_config(node),
                }
                for node in sorted(self.nodes, key=lambda candidate: candidate.id)
            ],
            "connections": [
                {
                    "source": connection.source.model_dump(mode="json"),
                    "target": connection.target.model_dump(mode="json"),
                    "order": connection.order,
                }
                for connection in sorted(
                    self.connections,
                    key=lambda candidate: (
                        candidate.source.node_id,
                        candidate.source.port,
                        candidate.target.node_id,
                        candidate.target.port,
                        candidate.order,
                    ),
                )
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def topological_node_ids(self) -> tuple[str, ...]:
        return self.execution_order()


def node_execution_config(node: WorkflowNode) -> dict[str, object]:
    if isinstance(node, (ImagePostprocessNode, FramePostprocessNode)):
        return image_postprocess_execution_config(node.config)
    return node.config.model_dump(mode="json")


def image_postprocess_execution_config(
    config: ImagePostprocessConfig,
) -> dict[str, object]:
    if config.model == VOSR_POSTPROCESS_NAME:
        output_size: dict[str, object] = (
            {"long_side": config.long_side}
            if config.sizing == "long-side"
            else {"scale": config.scale}
        )
        return {
            "model": config.model,
            **output_size,
            "infer_steps": config.infer_steps,
            "cfg_scale": config.cfg_scale,
            "weak_cond_strength_aelq": config.weak_cond_strength_aelq,
            "align_method": config.align_method,
            "tile_size": config.tile_size,
            "seed": config.seed,
        }
    if config.model in upscale_model_names():
        return {
            "model": config.model,
            "long_side": config.long_side,
        }
    if config.model == WU_PIXELIZATION_MODEL:
        return {
            "model": config.model,
            "cell_size": config.cell_size,
        }
    if config.model == PIXEL_ART_FIXER_MODEL:
        return {
            "model": config.model,
            "mode": config.mode,
            "low_memory": config.low_memory,
            "force_step": config.force_step,
        }
    raise ValueError(f"unknown postprocess model: {config.model!r}")


def keyframed_video_workflow_template() -> WorkflowGraph:
    image_config = ImageEditConfig()
    nodes: list[WorkflowNode] = [
        ReferencePackNode(
            id="references",
            title="Visual references",
            layout=NodeLayout(x=2, y=8),
        ),
        ImageEditNode(
            id="first-keyframe",
            title="First keyframe",
            layout=NodeLayout(x=42, y=2),
            config=image_config.model_copy(deep=True),
        ),
        ImageEditNode(
            id="last-keyframe",
            title="Last keyframe",
            layout=NodeLayout(x=42, y=14),
            config=image_config.model_copy(deep=True),
        ),
        ImagePostprocessNode(
            id="first-keyframe-postprocess",
            title="Postprocess first keyframe",
            layout=NodeLayout(x=82, y=2),
        ),
        ImagePostprocessNode(
            id="last-keyframe-postprocess",
            title="Postprocess last keyframe",
            layout=NodeLayout(x=82, y=14),
        ),
        AnimeGenI2VNode(
            id="video",
            title="Generate video",
            layout=NodeLayout(x=122, y=8),
        ),
        VideoContactSheetNode(
            id="contact-sheet",
            title="Contact sheet",
            layout=NodeLayout(x=162, y=2),
        ),
        ExtractVideoFramesNode(
            id="extract-frames",
            title="Extract frames",
            layout=NodeLayout(x=162, y=14),
        ),
        FramePostprocessNode(
            id="postprocess-frames",
            title="Postprocess frames",
            layout=NodeLayout(x=202, y=14),
        ),
    ]
    connections = [
        _connection("references-first", "references", "pack", "first-keyframe", "references"),
        _connection("references-last", "references", "pack", "last-keyframe", "references"),
        _connection(
            "first-postprocess",
            "first-keyframe",
            "image",
            "first-keyframe-postprocess",
            "image",
        ),
        _connection(
            "last-postprocess",
            "last-keyframe",
            "image",
            "last-keyframe-postprocess",
            "image",
        ),
        _connection(
            "first-video",
            "first-keyframe-postprocess",
            "image",
            "video",
            "start",
        ),
        _connection(
            "last-video",
            "last-keyframe-postprocess",
            "image",
            "video",
            "end",
        ),
        _connection("video-contact", "video", "video", "contact-sheet", "video"),
        _connection("video-frames", "video", "video", "extract-frames", "video"),
        _connection(
            "frames-postprocess",
            "extract-frames",
            "images",
            "postprocess-frames",
            "images",
        ),
    ]
    return WorkflowGraph(
        name="Keyframed video",
        nodes=nodes,
        connections=connections,
    )


def _connection(
    connection_id: str,
    source_node: str,
    source_port: str,
    target_node: str,
    target_port: str,
    order: int = 0,
) -> WorkflowConnection:
    return WorkflowConnection(
        id=connection_id,
        source=NodePortRef(node_id=source_node, port=source_port),
        target=NodePortRef(node_id=target_node, port=target_port),
        order=order,
    )


ItemT = TypeVar("ItemT", WorkflowNodeBase, WorkflowConnection)


def _unique_by_id(
    items: list[ItemT],
    label: str,
) -> dict[str, ItemT]:
    indexed: dict[str, ItemT] = {}
    for item in items:
        if item.id in indexed:
            raise ValueError(f"duplicate {label} id: {item.id!r}")
        indexed[item.id] = item
    return indexed
