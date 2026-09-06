from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, replace
from collections.abc import Collection, Sequence
from graphlib import TopologicalSorter
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAlias

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
    validate_image_edit_inputs,
)
from aigen.manifest_io import ManifestIOError
from aigen.runtime_profiles import resolve_project_path
from aigen.workflow_graph import (
    AnimeGenI2VNode,
    AnimeGenI2VConfig,
    ExtractVideoFramesConfig,
    ExtractVideoFramesNode,
    FramePostprocessNode,
    IllustrationUpscaleConfig,
    ImageEditConfig,
    ImageEditNode,
    CharacterEditNode,
    BindMaskNode, BindMaskConfig, SamSegmentNode, SamSegmentConfig,
    CharacterRefineNode, CharacterRefineConfig,
    ImageCollectionConfig,
    ImageCollectionNode,
    ImageSelectionNode,
    ImageResultReference,
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
from aigen.workflow_artifacts import ImageArtifact
from aigen.character_edit_audit import default_character_audit_config
from aigen.vlm_qwen import QwenVlmConfig, qwen_vlm_config_json
from aigen.workflow_results import resolve_image_result
from aigen.lora_weights import inspect_lora_weights
from aigen.generation.ltx23_settings import ResolvedLtx23Settings, resolve_ltx23_settings, validate_ltx23_positions, Ltx23KeyframesError
from aigen.generation.ltx23_runtime import Ltx23Installation, resolve_ltx23_installation
from aigen.generation.animegen_i2v import AnimeGenInstallation, resolve_animegen_installation
from aigen.generation.hunyuanvideo15 import (HunyuanVideo15Installation, HunyuanVideo15Error,
    HUNYUANVIDEO15_FPS, resolve_hunyuanvideo15_installation, validate_hunyuanvideo15_settings)
from aigen.workflow_graph import (VideoSourceNode, AudioSourceNode, PositionedKeyframeNode, PositionedKeyframeConfig,
    Ltx23Node, Ltx23Config, HunyuanI2VNode, HunyuanI2VConfig, AssembleVideoNode, AssembleVideoConfig)


@dataclass(frozen=True)
class CompiledImageSourceConfig:
    path: Path


@dataclass(frozen=True)
class CompiledVideoSourceConfig:
    path: Path


@dataclass(frozen=True)
class CompiledAudioSourceConfig:
    path: Path
    stream_index: int | None


@dataclass(frozen=True)
class CompiledLtx23Config:
    prompt: str
    negative_prompt: str
    seed_mode: Literal["fixed", "random"]
    seed: int
    settings: ResolvedLtx23Settings
    installation: Ltx23Installation


@dataclass(frozen=True)
class CompiledHunyuanConfig:
    settings: HunyuanI2VConfig
    installation: HunyuanVideo15Installation


@dataclass(frozen=True)
class CompiledImageSelectionConfig:
    reference: ImageResultReference
    image: ImageArtifact


@dataclass(frozen=True)
class CompiledReferencePackConfig:
    pack: LoadedCharacterReferencePack


@dataclass(frozen=True)
class CompiledLoraSourceConfig:
    path: Path
    weight: float
    architecture: str


@dataclass(frozen=True)
class CompiledImageEditConfig:
    backend: str
    prompt: str
    seed_mode: Literal["fixed", "random"]
    seed: int
    settings: ResolvedImageEditSettings


@dataclass(frozen=True)
class CompiledCharacterEditConfig(CompiledImageEditConfig):
    candidates: int
    max_iterations: int
    upscale_long_side: int | None
    pose_mode: str
    structure_control: str
    max_sequence_length: int | None
    guidance_scale: float | None
    audit: QwenVlmConfig
    conditioning_models: tuple[Path, ...]
    conditioning_runtime: tuple[str, ...]


@dataclass(frozen=True)
class CompiledCharacterRefineConfig:
    settings: CharacterRefineConfig
    audit: QwenVlmConfig


@dataclass(frozen=True)
class CompiledAnimeGenConfig:
    prompt: str
    seed_mode: Literal["fixed", "random"]
    seed: int
    settings: ResolvedAnimeGenSettings
    installation: AnimeGenInstallation


@dataclass(frozen=True)
class CompiledVosrLongSideConfig:
    model: str
    long_side: int
    infer_steps: int
    cfg_scale: float
    weak_cond_strength_aelq: float
    align_method: str
    tile_size: int
    seed_mode: Literal["fixed", "random"]
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
    seed_mode: Literal["fixed", "random"]
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
    | CompiledCharacterRefineConfig
    | BindMaskConfig
    | SamSegmentConfig
    | CompiledPostprocessConfig
    | CompiledAnimeGenConfig
    | VideoContactSheetConfig
    | ExtractVideoFramesConfig
    | ImageCollectionConfig
    | CompiledImageSelectionConfig
    | CompiledVideoSourceConfig
    | CompiledAudioSourceConfig
    | CompiledLtx23Config
    | CompiledHunyuanConfig
    | PositionedKeyframeConfig
    | AssembleVideoConfig
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


def compile_workflow(
    document: WorkflowGraph, *, target_node_ids: Sequence[str] | None = None,
) -> CompiledWorkflow:
    scope = document.execution_scope(target_node_ids)
    incoming = document.execution_inputs()
    predecessors = {
        node_id: {wire.source.node_id for wires in incoming[node_id].values() for wire in wires}
        for node_id in scope
    }

    nodes_by_id = {node.id: node for node in document.nodes}
    execution_order = tuple(TopologicalSorter(predecessors).static_order())
    compiled_nodes: dict[str, CompiledNode] = {}
    for node_id in execution_order:
        node = nodes_by_id[node_id]
        node_incoming = incoming[node.id]
        for port in node_definition(node.kind).inputs:
            if port.required and port.name not in node_incoming:
                raise ValueError(
                    f"required input {node.id}.{port.name} is not connected"
                )
        if isinstance(node, Ltx23Node):
            try:
                positions = tuple(compiled_nodes[wire.source.node_id].config.frame for wire in node_incoming["keyframes"])
                validate_ltx23_positions(positions, node.config.frames)
            except Ltx23KeyframesError as error:
                raise ValueError(f"invalid LTX keyframes for node {node.id}: {error}") from error
        if isinstance(node, AssembleVideoNode) and node.config.audio_policy == "replace" and "audio" not in node_incoming:
            raise ValueError(f"replacement audio input is required for node {node.id}")
        if isinstance(node, (ImageEditNode, CharacterEditNode, CharacterRefineNode)):
            references = tuple(compiled_nodes[wire.source.node_id].config for wire in node_incoming.get("references", ()))
            loras = tuple(compiled_nodes[wire.source.node_id].config for wire in node_incoming.get("loras", ()))
            try:
                validate_image_edit_inputs(
                    node.config.backend,
                    sum(len(config.pack.references) if isinstance(config, CompiledReferencePackConfig) else 1 for config in references)
                    + int(isinstance(node, CharacterRefineNode)),
                    tuple(config.architecture for config in loras),
                )
            except ImageEditError as error:
                raise ValueError(f"invalid inputs for {node.id}: {error}") from error
            if isinstance(node, CharacterEditNode) and node.config.backend == "flux2-klein":
                if "scene" in node_incoming or ("pose" in node_incoming and node.config.pose_mode == "keypoint"):
                    raise ValueError(f"node {node.id}: structural depth/edge/keypoint controls require the explicit Qwen character route")
        compiled_nodes[node.id] = CompiledNode(
            node=node,
            incoming=MappingProxyType(node_incoming),
            config=_compile_node_config(node, input_ports=node_incoming),
        )

    frozen_predecessors = MappingProxyType(
        {
            node_id: tuple(sorted(node_predecessors))
            for node_id, node_predecessors in predecessors.items()
        }
    )
    connected_sources = {
        predecessor for values in predecessors.values() for predecessor in values
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


def compile_workflow_run(
    document: WorkflowGraph, *, target_node_ids: Sequence[str] | None = None,
) -> CompiledWorkflow:
    scope = document.execution_scope(target_node_ids)
    compiled = compile_workflow(_resolve_random_seeds(document, scope), target_node_ids=target_node_ids)
    return replace(compiled, document=document)


def execution_config_payload(config: CompiledNodeConfig) -> dict[str, object]:
    if isinstance(config, (BindMaskConfig, SamSegmentConfig)):
        return config.model_dump(mode="json")
    if isinstance(config, CompiledCharacterRefineConfig):
        settings = config.settings
        return {**settings.model_dump(mode="json"), "seed": _execution_seed(settings.seed_mode, settings.seed),
                "audit": qwen_vlm_config_json(config.audit)}
    if isinstance(config, (PositionedKeyframeConfig, AssembleVideoConfig)):
        return config.model_dump(mode="json")
    if isinstance(config, CompiledVideoSourceConfig):
        return {"path": config.path.as_posix()}
    if isinstance(config, CompiledAudioSourceConfig):
        return {"path": config.path.as_posix(), "stream_index": config.stream_index}
    if isinstance(config, CompiledLtx23Config):
        return {"prompt": config.prompt, "negative_prompt": config.negative_prompt,
                "seed_mode": config.seed_mode, "seed": _execution_seed(config.seed_mode, config.seed),
                **asdict(config.settings)}
    if isinstance(config, CompiledHunyuanConfig):
        settings = config.settings
        return {**settings.model_dump(mode="json"), "seed": _execution_seed(settings.seed_mode, settings.seed), "fps": HUNYUANVIDEO15_FPS}
    if isinstance(config, CompiledImageSelectionConfig):
        return {
            "producer_signature": config.reference.producer_signature,
            "output_port": config.reference.output_port,
            "artifact_identity": config.reference.artifact_identity,
        }
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
        payload = {
            "backend": config.backend,
            "prompt": config.prompt,
            "seed_mode": config.seed_mode,
            "seed": _execution_seed(config.seed_mode, config.seed),
            **settings,
        }
        if isinstance(config, CompiledCharacterEditConfig):
            payload.update(
                candidates=config.candidates, max_iterations=config.max_iterations,
                upscale_long_side=config.upscale_long_side, pose_mode=config.pose_mode,
                structure_control=config.structure_control, max_sequence_length=config.max_sequence_length,
                guidance_scale=config.guidance_scale, audit=qwen_vlm_config_json(config.audit),
            )
        return payload
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
        payload = asdict(config)
        if isinstance(
            config,
            (CompiledVosrLongSideConfig, CompiledVosrScaleConfig),
        ):
            payload["seed"] = _execution_seed(
                config.seed_mode,
                config.seed,
            )
        return payload
    if isinstance(config, CompiledAnimeGenConfig):
        return {
            "prompt": config.prompt,
            "seed_mode": config.seed_mode,
            "seed": _execution_seed(config.seed_mode, config.seed),
            "frames": config.settings.frames,
            "fps": config.settings.fps,
            "sampling": config.settings.sampling,
            "steps": config.settings.steps,
            "precision": config.settings.precision,
            "keyframe_fit": config.settings.keyframe_fit,
        }
    return {}


def _execution_seed(
    mode: Literal["fixed", "random"],
    seed: int,
) -> int | None:
    return seed if mode == "fixed" else None


def _compile_node_config(node: WorkflowNode, *, input_ports: Collection[str]) -> CompiledNodeConfig:
    if isinstance(node, BindMaskNode):
        return node.config
    if isinstance(node, SamSegmentNode):
        from aigen.workflow_mask_execution import sam_segmentation_arguments
        sam_segmentation_arguments(node.config)
        return node.config
    if isinstance(node, CharacterRefineNode):
        if not node.config.prompt.strip():
            raise ValueError(f"regional edit instruction is required for node {node.id}")
        if round(node.config.steps * node.config.strength) < 1:
            raise ValueError(f"regional edit strength selects zero steps for node {node.id}")
        return CompiledCharacterRefineConfig(node.config, default_character_audit_config())
    if isinstance(node, VideoSourceNode):
        return CompiledVideoSourceConfig(_source_path(node.id, node.config.path))
    if isinstance(node, AudioSourceNode):
        return CompiledAudioSourceConfig(_source_path(node.id, node.config.path), node.config.stream_index)
    if isinstance(node, PositionedKeyframeNode):
        return node.config
    if isinstance(node, AssembleVideoNode):
        from PIL import ImageColor
        ImageColor.getrgb(node.config.background)
        return node.config
    if isinstance(node, Ltx23Node):
        values = node.config
        if not values.prompt.strip() or not values.negative_prompt.strip():
            raise ValueError(f"LTX motion and negative prompts are required for node {node.id}")
        try:
            settings = resolve_ltx23_settings(
                resolution=values.resolution, frames=values.frames, fps=values.fps,
                steps=values.steps, phases=values.phases, solver=values.solver,
                conditioning_strength=values.conditioning_strength, model=values.model, keyframe_fit=values.keyframe_fit)
            installation = resolve_ltx23_installation(settings)
        except Ltx23KeyframesError as error:
            raise ValueError(f"invalid LTX settings for node {node.id}: {error}") from error
        return CompiledLtx23Config(values.prompt.strip(), values.negative_prompt.strip(), values.seed_mode, values.seed, settings, installation)
    if isinstance(node, HunyuanI2VNode):
        if not node.config.prompt.strip():
            raise ValueError(f"Hunyuan motion prompt is required for node {node.id}")
        try:
            validate_hunyuanvideo15_settings(steps=node.config.steps, frames=node.config.frames)
            installation = resolve_hunyuanvideo15_installation()
        except HunyuanVideo15Error as error:
            raise ValueError(f"Hunyuan node {node.id}: {error}") from error
        return CompiledHunyuanConfig(node.config.model_copy(update={"prompt": node.config.prompt.strip()}), installation)
    if isinstance(node, ImageCollectionNode):
        return node.config
    if isinstance(node, ImageSelectionNode):
        if node.config.selected is None:
            raise ValueError(f"choose and save an image for {node.title!r} before continuing; run its collection to make variants")
        return CompiledImageSelectionConfig(
            reference=node.config.selected,
            image=resolve_image_result(node.config.selected),
        )
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
        path = _source_path(node.id, node.config.path)
        return CompiledLoraSourceConfig(
            path=path,
            weight=node.config.weight,
            architecture=inspect_lora_weights(path).architecture,
        )
    if isinstance(node, (ImageEditNode, CharacterEditNode)):
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
        common = dict(
            backend=node.config.backend,
            prompt=prompt,
            seed_mode=node.config.seed_mode,
            seed=node.config.seed,
            settings=settings,
        )
        if isinstance(node, CharacterEditNode):
            config = node.config
            if config.backend == "flux2-klein" and (config.max_sequence_length is not None or config.guidance_scale is not None):
                raise ValueError(f"node {node.id}: max_sequence_length and guidance_scale are Qwen-only settings")
            conditioning_models = []
            conditioning_runtime = []
            if "pose" in input_ports and config.pose_mode == "keypoint":
                from aigen.dwpose_control import DEFAULT_DWPOSE_DET_MODEL, DEFAULT_DWPOSE_POSE_MODEL
                conditioning_models.extend((DEFAULT_DWPOSE_DET_MODEL, DEFAULT_DWPOSE_POSE_MODEL))
                conditioning_runtime.extend(("controlnet-dwpose", "onnxruntime-gpu"))
            if "scene" in input_ports:
                if config.structure_control == "depth":
                    from aigen.depth_v2_control import DEFAULT_DEPTH_V2_MODEL
                    conditioning_models.append(DEFAULT_DEPTH_V2_MODEL)
                else:
                    conditioning_runtime.append("opencv-python")
            return CompiledCharacterEditConfig(
                **common, candidates=config.candidates, max_iterations=config.max_iterations,
                upscale_long_side=config.upscale_long_side, pose_mode=config.pose_mode,
                structure_control=config.structure_control, max_sequence_length=config.max_sequence_length,
                guidance_scale=config.guidance_scale, audit=default_character_audit_config(),
                conditioning_models=tuple(conditioning_models), conditioning_runtime=tuple(conditioning_runtime),
            )
        return CompiledImageEditConfig(**common)
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
                "seed_mode": config.seed_mode,
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
                keyframe_fit=node.config.keyframe_fit,
            )
        except AnimeGenI2VError as error:
            raise ValueError(
                f"invalid AnimeGen settings for node {node.id}: {error}"
            ) from error
        return CompiledAnimeGenConfig(
            prompt=prompt,
            seed_mode=node.config.seed_mode,
            seed=node.config.seed,
            settings=settings,
            installation=resolve_animegen_installation(lightning_required=settings.profile.lightning),
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


def _resolve_random_seeds(document: WorkflowGraph, scope: Sequence[str]) -> WorkflowGraph:
    changed_nodes: list[WorkflowNode] | None = None
    active = set(scope)
    for index, node in enumerate(document.nodes):
        if node.id not in active:
            continue
        config = node.config
        if (
            not isinstance(
                config,
                (
                    ImageEditConfig,
                    CharacterRefineConfig,
                    VosrPostprocessConfig,
                    AnimeGenI2VConfig,
                    Ltx23Config,
                    HunyuanI2VConfig,
                ),
            )
            or config.seed_mode != "random"
        ):
            continue
        if changed_nodes is None:
            changed_nodes = list(document.nodes)
        changed_config = config.model_copy(
            update={
                "seed_mode": "fixed",
                "seed": secrets.randbits(63),
            }
        )
        changed_nodes[index] = node.model_copy(
            update={"config": changed_config}
        )
    if changed_nodes is None:
        return document
    return document.model_copy(update={"nodes": tuple(changed_nodes)})


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
                (compiled.node for compiled in compiled_nodes.values()),
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
                (wire for compiled in compiled_nodes.values() for wires in compiled.incoming.values() for wire in wires),
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
