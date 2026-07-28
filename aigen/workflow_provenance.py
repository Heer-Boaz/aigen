from __future__ import annotations

import json
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from aigen.generation.image_edit import (
    BOOGU_IMAGE_EDIT_BACKEND,
    FLUX2_DEV_BACKEND,
    FLUX2_KLEIN_BACKEND,
    HIDREAM_O1_BACKEND,
    QWEN_2511_BASE_BACKEND,
    QWEN_2511_LIGHTNING_BACKEND,
    USO_FLUX1_BACKEND,
)
from aigen.workflow_cache import NodeExecutionProvenance, RevisionedComponent
from aigen.workflow_graph import (
    AnimeGenI2VNode,
    FramePostprocessNode,
    ImageEditNode,
    ImagePostprocessNode,
    NodeKind,
    WorkflowNode,
)


WORKFLOW_EXECUTOR_REVISION = "3"
IMAGE_EDIT_API_REVISION = "1"
IMAGE_POSTPROCESS_BATCH_REVISION = "1"
VIDEO_POSTPROCESS_REVISION = "1"


def workflow_node_provenance(node: WorkflowNode) -> NodeExecutionProvenance:
    if isinstance(node, ImageEditNode):
        return _image_edit_provenance(node)
    if isinstance(node, (ImagePostprocessNode, FramePostprocessNode)):
        return _postprocess_provenance(node)
    if isinstance(node, AnimeGenI2VNode):
        from aigen.generation.animegen_i2v import (
            ANIMEGEN_BASE_MODEL_REVISION,
            ANIMEGEN_LIGHTNING_REVISION,
            ANIMEGEN_MODEL_REVISION,
            animegen_sampling_profile,
        )

        models = [
            _component("AnimeGen-I2V", ANIMEGEN_MODEL_REVISION),
            _component("Wan2.2 I2V base", ANIMEGEN_BASE_MODEL_REVISION),
        ]
        if animegen_sampling_profile(node.config.sampling).lightning:
            models.append(
                _component("Wan2.2 Lightning LoRAs", ANIMEGEN_LIGHTNING_REVISION)
            )
        return _provenance(
            backend=_component("AnimeGen-I2V", "1"),
            models=tuple(models),
        )
    if node.kind in {
        NodeKind.VIDEO_CONTACT_SHEET,
        NodeKind.EXTRACT_VIDEO_FRAMES,
    }:
        return _provenance(
            backend=_component("video-postprocess", VIDEO_POSTPROCESS_REVISION)
        )
    return _provenance(
        backend=_component("workflow-source", "sha256-v1")
    )


def _image_edit_provenance(
    node: ImageEditNode,
) -> NodeExecutionProvenance:
    from aigen.generation.image_edit_batch import IMAGE_EDIT_BATCH_VERSION

    backend = node.config.backend
    batch_backend = backend in {
        FLUX2_KLEIN_BACKEND,
        QWEN_2511_LIGHTNING_BACKEND,
        QWEN_2511_BASE_BACKEND,
    }
    implementation = _component(
        "image-edit-batch" if batch_backend else "image-edit-api",
        (
            str(IMAGE_EDIT_BATCH_VERSION)
            if batch_backend
            else IMAGE_EDIT_API_REVISION
        ),
    )
    if backend == FLUX2_KLEIN_BACKEND:
        from aigen.generation.flux2_klein import (
            FLUX2_KLEIN_MODEL_ROOT,
            FLUX2_KLEIN_TEXT_ENCODER,
            FLUX2_KLEIN_TRANSFORMER,
        )

        models = (
            _component(
                "FLUX.2 Klein scaled-FP8 transformer",
                _path_inventory_revision(FLUX2_KLEIN_TRANSFORMER),
            ),
            _component(
                "FLUX.2 Klein VAE and scheduler",
                _path_inventory_revision(FLUX2_KLEIN_MODEL_ROOT),
            ),
            _component(
                "Qwen3-8B-FP8 conditioner",
                _path_inventory_revision(FLUX2_KLEIN_TEXT_ENCODER),
            ),
        )
    elif backend in {
        QWEN_2511_LIGHTNING_BACKEND,
        QWEN_2511_BASE_BACKEND,
    }:
        from aigen.generation.qwen_image_edit_lightx2v import (
            LIGHTX2V_REVISION,
            QWEN_EDIT_2511_LIGHTNING_REVISION,
            QWEN_EDIT_2511_REVISION,
        )

        models = (
            _component("LightX2V", LIGHTX2V_REVISION),
            _component("Qwen-Image-Edit-2511", QWEN_EDIT_2511_REVISION),
            _component(
                "Qwen-Image-Edit-2511 transformer",
                QWEN_EDIT_2511_LIGHTNING_REVISION,
            ),
        )
    elif backend == FLUX2_DEV_BACKEND:
        from aigen.generation.flux2_dev_wangp import (
            FLUX2_DEV_MODEL_TYPE,
            FLUX2_DEV_WANGP_REVISION,
        )

        models = (
            _component("WanGP", FLUX2_DEV_WANGP_REVISION),
            _component("FLUX.2 dev", FLUX2_DEV_MODEL_TYPE),
        )
    elif backend == HIDREAM_O1_BACKEND:
        from aigen.generation.hidream_o1_comfy import (
            COMFY_REVISION,
            HIDREAM_CHECKPOINT_REVISION,
        )

        models = (
            _component("ComfyUI image runtime", COMFY_REVISION),
            _component("HiDream-O1 checkpoint", HIDREAM_CHECKPOINT_REVISION),
        )
    elif backend == BOOGU_IMAGE_EDIT_BACKEND:
        from aigen.generation.boogu_image_edit import (
            BOOGU_MODEL_REVISION,
            BOOGU_SOURCE_REVISION,
        )

        models = (
            _component("Boogu source", BOOGU_SOURCE_REVISION),
            _component("Boogu model", BOOGU_MODEL_REVISION),
        )
    elif backend == USO_FLUX1_BACKEND:
        from aigen.generation.uso_flux1 import (
            USO_MODEL_TYPE,
            USO_SOURCE_REVISION,
        )

        models = (
            _component("USO source", USO_SOURCE_REVISION),
            _component("USO model", USO_MODEL_TYPE),
        )
    else:
        raise ValueError(f"unsupported image-edit backend: {backend}")
    return _provenance(backend=implementation, models=models)


def _postprocess_provenance(
    node: ImagePostprocessNode | FramePostprocessNode,
) -> NodeExecutionProvenance:
    from aigen.generation.vosr_backend import VOSR_POSTPROCESS_NAME

    model = node.config.model
    if model == VOSR_POSTPROCESS_NAME:
        from aigen.generation.vosr_backend import (
            VOSR_MODEL_REVISION,
            VOSR_SOURCE_REVISION,
        )

        models = (
            _component("VOSR source", VOSR_SOURCE_REVISION),
            _component("VOSR model", VOSR_MODEL_REVISION),
        )
    elif model == "wu-pixelization":
        from aigen.generation.wu_pixelization import (
            WU_PIXELIZATION_MODEL_ROOT,
            WU_PIXELIZATION_REVISION,
        )

        models = (
            _component("Wu pixelization source", WU_PIXELIZATION_REVISION),
            _component(
                "Wu pixelization models",
                _path_inventory_revision(WU_PIXELIZATION_MODEL_ROOT),
            ),
        )
    elif model == "pixel-art-fixer":
        from aigen.generation.pixel_art_fixer import (
            PIXEL_ART_FIXER_UPSTREAM_REVISION,
        )

        models = (
            _component(
                "Pixel Art Fixer",
                PIXEL_ART_FIXER_UPSTREAM_REVISION,
            ),
        )
    else:
        from aigen.generation.image_upscale import upscale_model_path

        models = (
            _component(
                model,
                _path_inventory_revision(upscale_model_path(model)),
            ),
        )
    return _provenance(
        backend=_component(
            "image-postprocess-batch",
            IMAGE_POSTPROCESS_BATCH_REVISION,
        ),
        models=models,
    )


def _provenance(
    *,
    backend: RevisionedComponent,
    models: tuple[RevisionedComponent, ...] = (),
) -> NodeExecutionProvenance:
    return NodeExecutionProvenance(
        executor=_component(
            "aigen-workflow-executor",
            WORKFLOW_EXECUTOR_REVISION,
        ),
        backend=backend,
        models=models,
    )


def _component(name: str, revision: str) -> RevisionedComponent:
    return RevisionedComponent(name=name, revision=revision)


@lru_cache(maxsize=None)
def _path_inventory_revision(path: Path) -> str:
    resolved = path.expanduser().resolve()
    paths = (
        tuple(
            candidate
            for candidate in sorted(resolved.rglob("*"))
            if candidate.is_file()
        )
        if resolved.is_dir()
        else (resolved,)
    )
    inventory = []
    for candidate in paths:
        file_stat = candidate.stat()
        inventory.append(
            (
                candidate.relative_to(resolved).as_posix()
                if resolved.is_dir()
                else candidate.name,
                file_stat.st_size,
                file_stat.st_mtime_ns,
                file_stat.st_ctime_ns,
            )
        )
    encoded = json.dumps(
        inventory,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
