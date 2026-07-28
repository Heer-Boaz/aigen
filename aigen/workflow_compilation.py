from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from graphlib import TopologicalSorter
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

from aigen.character_reference_models import CharacterReferenceError
from aigen.character_reference_pack import (
    LoadedCharacterReferencePack,
    load_character_reference_pack,
)
from aigen.generation.animegen_i2v import (
    AnimeGenI2VError,
    ResolvedAnimeGenSettings,
    resolve_animegen_settings,
)
from aigen.generation.image_batch_postprocess import (
    image_batch_postprocess_model_names,
)
from aigen.image_dimensions import parse_aspect_ratio
from aigen.generation.image_edit import (
    ImageEditError,
    ResolvedImageEditSettings,
    image_edit_backend_settings,
    resolve_image_edit_settings,
)
from aigen.manifest_io import ManifestIOError
from aigen.runtime_profiles import resolve_project_path
from aigen.workflow_graph import (
    AnimeGenI2VNode,
    ExtractVideoFramesConfig,
    ExtractVideoFramesNode,
    FramePostprocessNode,
    IllustrationUpscaleConfig,
    ImageEditNode,
    ImagePostprocessNode,
    ImageSourceNode,
    LoraSourceNode,
    PixelArtFixerConfig,
    ReferencePackNode,
    VideoContactSheetConfig,
    VideoContactSheetNode,
    VosrPostprocessConfig,
    WorkflowConnection,
    WorkflowGraph,
    WorkflowNode,
    WuPixelizationConfig,
    node_definition,
)


@dataclass(frozen=True)
class CompiledImageSourceConfig:
    path: Path


@dataclass(frozen=True)
class CompiledReferencePackConfig:
    pack: LoadedCharacterReferencePack


@dataclass(frozen=True)
class CompiledLoraSourceConfig:
    path: Path
    weight: float


@dataclass(frozen=True)
class CompiledImageEditConfig:
    backend: str
    prompt: str
    seed: int
    settings: ResolvedImageEditSettings


@dataclass(frozen=True)
class CompiledAnimeGenConfig:
    prompt: str
    seed: int
    settings: ResolvedAnimeGenSettings


@dataclass(frozen=True)
class CompiledVosrLongSideConfig:
    model: str
    long_side: int
    infer_steps: int
    cfg_scale: float
    weak_cond_strength_aelq: float
    align_method: str
    tile_size: int
    seed: int


@dataclass(frozen=True)
class CompiledVosrScaleConfig:
    model: str
    scale: int
    infer_steps: int
    cfg_scale: float
    weak_cond_strength_aelq: float
    align_method: str
    tile_size: int
    seed: int


@dataclass(frozen=True)
class CompiledIllustrationUpscaleConfig:
    model: str
    long_side: int | None


@dataclass(frozen=True)
class CompiledWuPixelizationConfig:
    model: str
    cell_size: int


@dataclass(frozen=True)
class CompiledPixelArtFixerConfig:
    model: str
    mode: str
    low_memory: bool
    force_step: float | None


CompiledPostprocessConfig: TypeAlias = (
    CompiledVosrLongSideConfig
    | CompiledVosrScaleConfig
    | CompiledIllustrationUpscaleConfig
    | CompiledWuPixelizationConfig
    | CompiledPixelArtFixerConfig
)
CompiledNodeConfig: TypeAlias = (
    CompiledImageSourceConfig
    | CompiledReferencePackConfig
    | CompiledLoraSourceConfig
    | CompiledImageEditConfig
    | CompiledPostprocessConfig
    | CompiledAnimeGenConfig
    | VideoContactSheetConfig
    | ExtractVideoFramesConfig
)


@dataclass(frozen=True)
class CompiledNode:
    node: WorkflowNode
    incoming: MappingProxyType[
        str,
        tuple[WorkflowConnection, ...],
    ]
    config: CompiledNodeConfig

    @property
    def id(self) -> str:
        return self.node.id


@dataclass(frozen=True)
class CompiledWorkflow:
    document: WorkflowGraph
    nodes: MappingProxyType[str, CompiledNode]
    predecessors: MappingProxyType[str, tuple[str, ...]]
    execution_order: tuple[str, ...]
    terminal_node_ids: tuple[str, ...]
    digest: str

    def node(self, node_id: str) -> CompiledNode:
        return self.nodes[node_id]


def compile_workflow(document: WorkflowGraph) -> CompiledWorkflow:
    nodes_by_id = {node.id: node for node in document.nodes}
    incoming = _incoming_connections(document)
    predecessors = {node.id: set() for node in document.nodes}
    for connection in document.connections:
        predecessors[connection.target.node_id].add(connection.source.node_id)

    compiled_nodes: dict[str, CompiledNode] = {}
    for node in document.nodes:
        node_incoming = incoming[node.id]
        for port in node_definition(node.kind).inputs:
            if port.required and port.name not in node_incoming:
                raise ValueError(
                    f"required input {node.id}.{port.name} is not connected"
                )
        compiled_nodes[node.id] = CompiledNode(
            node=node,
            incoming=MappingProxyType(node_incoming),
            config=_compile_node_config(node),
        )

    frozen_predecessors = MappingProxyType(
        {
            node_id: tuple(sorted(node_predecessors))
            for node_id, node_predecessors in predecessors.items()
        }
    )
    execution_order = tuple(
        TopologicalSorter(frozen_predecessors).static_order()
    )
    connected_sources = {
        connection.source.node_id
        for connection in document.connections
    }
    frozen_nodes = MappingProxyType(compiled_nodes)
    return CompiledWorkflow(
        document=document,
        nodes=frozen_nodes,
        predecessors=frozen_predecessors,
        execution_order=execution_order,
        terminal_node_ids=tuple(
            node_id
            for node_id in execution_order
            if node_id not in connected_sources
        ),
        digest=_execution_digest(document, frozen_nodes),
    )


def execution_config_payload(config: CompiledNodeConfig) -> dict[str, object]:
    if isinstance(config, CompiledImageSourceConfig):
        return {"path": config.path.as_posix()}
    if isinstance(config, CompiledReferencePackConfig):
        return {"path": config.pack.path.as_posix()}
    if isinstance(config, CompiledLoraSourceConfig):
        return {
            "path": config.path.as_posix(),
            "weight": config.weight,
        }
    if isinstance(config, CompiledImageEditConfig):
        settings = asdict(config.settings)
        return {
            "backend": config.backend,
            "prompt": config.prompt,
            "seed": config.seed,
            **settings,
        }
    if isinstance(
        config,
        (
            CompiledVosrLongSideConfig,
            CompiledVosrScaleConfig,
            CompiledIllustrationUpscaleConfig,
            CompiledWuPixelizationConfig,
            CompiledPixelArtFixerConfig,
        ),
    ):
        return asdict(config)
    if isinstance(config, CompiledAnimeGenConfig):
        return {
            "prompt": config.prompt,
            "seed": config.seed,
            "frames": config.settings.frames,
            "fps": config.settings.fps,
            "sampling": config.settings.sampling,
            "steps": config.settings.steps,
            "precision": config.settings.precision,
        }
    return {}


def _compile_node_config(node: WorkflowNode) -> CompiledNodeConfig:
    if isinstance(node, ImageSourceNode):
        return CompiledImageSourceConfig(
            path=_source_path(node.id, node.config.path)
        )
    if isinstance(node, ReferencePackNode):
        path = _source_path(node.id, node.config.path)
        try:
            pack = load_character_reference_pack(path)
        except (CharacterReferenceError, ManifestIOError, OSError) as error:
            raise ValueError(
                f"invalid reference pack for node {node.id}: {error}"
            ) from error
        return CompiledReferencePackConfig(pack=pack)
    if isinstance(node, LoraSourceNode):
        return CompiledLoraSourceConfig(
            path=_source_path(node.id, node.config.path),
            weight=node.config.weight,
        )
    if isinstance(node, ImageEditNode):
        try:
            backend = image_edit_backend_settings(node.config.backend)
            settings = resolve_image_edit_settings(
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
        except (ImageEditError, ValueError) as error:
            raise ValueError(
                f"invalid image-edit settings for node {node.id}: {error}"
            ) from error
        prompt = node.config.prompt.strip()
        if not prompt and not backend.supports_empty_prompt:
            raise ValueError(
                f"image-edit prompt is required for node {node.id}"
            )
        return CompiledImageEditConfig(
            backend=node.config.backend,
            prompt=prompt,
            seed=node.config.seed,
            settings=settings,
        )
    if isinstance(node, (ImagePostprocessNode, FramePostprocessNode)):
        config = node.config
        if config.model not in image_batch_postprocess_model_names():
            raise ValueError(
                f"unknown postprocess model for node {node.id}: "
                f"{config.model!r}"
            )
        if isinstance(config, VosrPostprocessConfig):
            common = {
                "model": config.model,
                "infer_steps": config.infer_steps,
                "cfg_scale": config.cfg_scale,
                "weak_cond_strength_aelq": (
                    config.weak_cond_strength_aelq
                ),
                "align_method": config.align_method,
                "tile_size": config.tile_size,
                "seed": config.seed,
            }
            if config.sizing == "long-side":
                if config.long_side is None:
                    raise ValueError(
                        f"VOSR long-side sizing requires a long side for "
                        f"node {node.id}"
                    )
                return CompiledVosrLongSideConfig(
                    long_side=config.long_side,
                    **common,
                )
            return CompiledVosrScaleConfig(
                scale=config.scale,
                **common,
            )
        if isinstance(config, IllustrationUpscaleConfig):
            return CompiledIllustrationUpscaleConfig(
                model=config.model,
                long_side=config.long_side,
            )
        if isinstance(config, WuPixelizationConfig):
            return CompiledWuPixelizationConfig(
                model=config.model,
                cell_size=config.cell_size,
            )
        return CompiledPixelArtFixerConfig(
            model=config.model,
            mode=config.mode,
            low_memory=config.low_memory,
            force_step=config.force_step,
        )
    if isinstance(node, AnimeGenI2VNode):
        prompt = node.config.prompt.strip()
        if not prompt:
            raise ValueError(
                f"AnimeGen motion prompt is required for node {node.id}"
            )
        try:
            settings = resolve_animegen_settings(
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
        return CompiledAnimeGenConfig(
            prompt=prompt,
            seed=node.config.seed,
            settings=settings,
        )
    if isinstance(node, VideoContactSheetNode):
        return node.config
    if isinstance(node, ExtractVideoFramesNode):
        return node.config
    raise TypeError(f"unsupported workflow node: {node.kind}")


def _source_path(node_id: str, raw_path: str) -> Path:
    if not raw_path.strip():
        raise ValueError(f"source path is required for node {node_id}")
    path = resolve_project_path(raw_path)
    if not path.is_file():
        raise ValueError(
            f"source file does not exist for node {node_id}: {path}"
        )
    return path


def _incoming_connections(
    document: WorkflowGraph,
) -> dict[str, dict[str, tuple[WorkflowConnection, ...]]]:
    incoming: dict[str, dict[str, list[WorkflowConnection]]] = {
        node.id: {} for node in document.nodes
    }
    for connection in document.connections:
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


def _execution_digest(
    document: WorkflowGraph,
    compiled_nodes: MappingProxyType[str, CompiledNode],
) -> str:
    payload = {
        "version": document.version,
        "nodes": [
            {
                "id": node.id,
                "kind": node.kind,
                "config": execution_config_payload(
                    compiled_nodes[node.id].config
                ),
            }
            for node in sorted(
                document.nodes,
                key=lambda candidate: candidate.id,
            )
        ],
        "connections": [
            {
                "source": connection.source.model_dump(mode="json"),
                "target": connection.target.model_dump(mode="json"),
                "order": connection.order,
            }
            for connection in sorted(
                document.connections,
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
