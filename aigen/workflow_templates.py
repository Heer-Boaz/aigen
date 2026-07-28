from __future__ import annotations

import uuid

from aigen.generation.animegen_i2v import (
    ANIMEGEN_DEFAULT_FPS,
    ANIMEGEN_DEFAULT_FRAMES,
    ANIMEGEN_DEFAULT_PRECISION,
    ANIMEGEN_DEFAULT_SAMPLING,
    animegen_sampling_profile,
)
from aigen.generation.image_batch_postprocess import (
    IMAGE_BATCH_DEFAULT_CELL_SIZE,
    IMAGE_BATCH_DEFAULT_FIXER_MODE,
    IMAGE_BATCH_DEFAULT_LOW_MEMORY,
    PIXEL_ART_FIXER_MODEL,
    WU_PIXELIZATION_MODEL,
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
from aigen.generation.image_edit import (
    IMAGE_EDIT_BACKENDS,
    image_edit_backend_settings,
)
from aigen.workflow_graph import (
    AnimeGenI2VConfig,
    AnimeGenI2VNode,
    ExtractVideoFramesNode,
    FramePostprocessNode,
    IllustrationUpscaleConfig,
    ImageEditConfig,
    ImageEditNode,
    ImagePostprocessConfig,
    ImagePostprocessNode,
    ImageSourceNode,
    LoraSourceNode,
    NodeKind,
    NodeLayout,
    NodePortRef,
    PixelArtFixerConfig,
    ReferencePackNode,
    VideoContactSheetNode,
    VosrPostprocessConfig,
    WorkflowConnection,
    WorkflowGraph,
    WorkflowNode,
    WuPixelizationConfig,
    node_definition,
)


def create_workflow_node(
    kind: NodeKind,
    *,
    x: int = 0,
    y: int = 0,
    node_id: str | None = None,
    title: str | None = None,
) -> WorkflowNode:
    common = {
        "id": node_id or f"node-{uuid.uuid4().hex}",
        "title": title or node_definition(kind).label,
        "layout": NodeLayout(x=x, y=y),
    }
    match kind:
        case NodeKind.IMAGE_SOURCE:
            return ImageSourceNode(**common)
        case NodeKind.REFERENCE_PACK:
            return ReferencePackNode(**common)
        case NodeKind.LORA_SOURCE:
            return LoraSourceNode(**common)
        case NodeKind.IMAGE_EDIT:
            return ImageEditNode(config=default_image_edit_config(), **common)
        case NodeKind.IMAGE_POSTPROCESS:
            return ImagePostprocessNode(
                config=default_postprocess_config(),
                **common,
            )
        case NodeKind.ANIMEGEN_I2V:
            return AnimeGenI2VNode(config=default_animegen_config(), **common)
        case NodeKind.VIDEO_CONTACT_SHEET:
            return VideoContactSheetNode(**common)
        case NodeKind.EXTRACT_VIDEO_FRAMES:
            return ExtractVideoFramesNode(**common)
        case NodeKind.FRAME_POSTPROCESS:
            return FramePostprocessNode(
                config=default_postprocess_config(),
                **common,
            )


def default_image_edit_config() -> ImageEditConfig:
    backend = IMAGE_EDIT_BACKENDS[0]
    settings = image_edit_backend_settings(backend)
    return ImageEditConfig(
        backend=backend,
        steps=settings.steps,
        guidance=settings.guidance,
        strength=settings.strength,
        sampler=settings.sampler,
        scheduler=settings.scheduler,
    )


def default_animegen_config() -> AnimeGenI2VConfig:
    profile = animegen_sampling_profile(ANIMEGEN_DEFAULT_SAMPLING)
    return AnimeGenI2VConfig(
        frames=ANIMEGEN_DEFAULT_FRAMES,
        fps=ANIMEGEN_DEFAULT_FPS,
        sampling=ANIMEGEN_DEFAULT_SAMPLING,
        steps=profile.steps,
        precision=ANIMEGEN_DEFAULT_PRECISION,
    )


def postprocess_config_for_model(model: str) -> ImagePostprocessConfig:
    if model == VOSR_POSTPROCESS_NAME:
        return default_postprocess_config()
    if model in upscale_model_names():
        return IllustrationUpscaleConfig(model=model, long_side=2048)
    if model == WU_PIXELIZATION_MODEL:
        return WuPixelizationConfig(
            cell_size=IMAGE_BATCH_DEFAULT_CELL_SIZE
        )
    if model == PIXEL_ART_FIXER_MODEL:
        return PixelArtFixerConfig(
            mode=IMAGE_BATCH_DEFAULT_FIXER_MODE,
            low_memory=IMAGE_BATCH_DEFAULT_LOW_MEMORY,
            force_step=None,
        )
    raise ValueError(f"unknown postprocess model: {model!r}")


def default_postprocess_config() -> VosrPostprocessConfig:
    return VosrPostprocessConfig(
        sizing="long-side",
        long_side=2048,
        scale=VOSR_DEFAULT_SCALE,
        infer_steps=VOSR_DEFAULT_INFER_STEPS,
        cfg_scale=VOSR_DEFAULT_CFG_SCALE,
        weak_cond_strength_aelq=VOSR_DEFAULT_WEAK_COND_STRENGTH_AELQ,
        align_method=VOSR_DEFAULT_ALIGN_METHOD,
        tile_size=VOSR_DEFAULT_TILE_SIZE,
        seed=VOSR_DEFAULT_SEED,
    )


def keyframed_video_workflow_template() -> WorkflowGraph:
    image_config = default_image_edit_config()
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
            config=default_postprocess_config(),
        ),
        ImagePostprocessNode(
            id="last-keyframe-postprocess",
            title="Postprocess last keyframe",
            layout=NodeLayout(x=82, y=14),
            config=default_postprocess_config(),
        ),
        AnimeGenI2VNode(
            id="video",
            title="Generate video",
            layout=NodeLayout(x=122, y=8),
            config=default_animegen_config(),
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
            config=default_postprocess_config(),
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
