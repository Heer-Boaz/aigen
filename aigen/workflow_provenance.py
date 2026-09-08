from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

from aigen.generation.image_edit import (
    BOOGU_IMAGE_EDIT_BACKEND,
    FLUX2_DEV_BACKEND,
    FLUX2_KLEIN_BACKEND,
    HIDREAM_O1_BACKEND,
    QWEN_2511_BASE_BACKEND,
    QWEN_2511_LIGHTNING_BACKEND,
    USO_FLUX1_BACKEND,
)
from aigen.model_artifacts import ModelArtifactComponent, model_artifact_stat_revision
from aigen.media_timing import ffmpeg_runtime_revision, media_runtime_revision
from aigen.workflow_cache import NodeExecutionProvenance, RevisionedComponent
from aigen.workflow_graph import (
    AnimeGenI2VNode,
    FramePostprocessNode,
    ImageEditNode,
    CharacterEditNode,
    CharacterRefineNode, SamSegmentNode,
    ImagePostprocessNode,
    Ltx23Node,
    HunyuanI2VNode,
    NodeKind,
    WorkflowNode,
)


WORKFLOW_EXECUTOR_REVISION = "5"
IMAGE_EDIT_API_REVISION = "2"
IMAGE_POSTPROCESS_BATCH_REVISION = "1"
VIDEO_POSTPROCESS_REVISION = "2"

if TYPE_CHECKING:
    from aigen.workflow_compilation import CompiledNodeConfig


def workflow_node_provenance(node: WorkflowNode, config: CompiledNodeConfig | None = None) -> NodeExecutionProvenance:
    if isinstance(node, (CharacterEditNode, CharacterRefineNode)):
        return _character_edit_provenance(node, config)
    if isinstance(node, SamSegmentNode):
        from aigen.keyframe_segmentation import (
            SEGMENTATION_IMPLEMENTATION_REVISION, DEFAULT_SAM_CHECKPOINT,
            DEFAULT_SAM2_MODEL, DEFAULT_ANIME_SEGMENTATION_MODEL,
        )
        from aigen.runtime_provenance import build_python_runtime_provenance
        path, runtime_packages = {
            "sam1": (DEFAULT_SAM_CHECKPOINT, ("torch", "torchvision", "segment-anything")),
            "sam2": (DEFAULT_SAM2_MODEL, ("torch", "torchvision", "transformers", "safetensors")),
            "anime": (DEFAULT_ANIME_SEGMENTATION_MODEL, ("onnxruntime-gpu", "opencv-python-headless")),
        }[node.config.engine]
        return _provenance(backend=_component("SAM segmentation", SEGMENTATION_IMPLEMENTATION_REVISION), models=(
            _component(node.config.engine, _path_inventory_revision(path)),
            _component("segmentation runtime", str(build_python_runtime_provenance(
                (*runtime_packages, "numpy", "scipy", "pillow"))["fingerprint"])),
        ))
    if isinstance(node, ImageEditNode):
        return _image_edit_provenance(node)
    if isinstance(node, (ImagePostprocessNode, FramePostprocessNode)):
        return _postprocess_provenance(node.config.model)
    if isinstance(node, AnimeGenI2VNode):
        from aigen.generation.animegen_i2v import (
            ANIMEGEN_BASE_MODEL_REVISION,
            ANIMEGEN_LIGHTNING_REVISION,
            ANIMEGEN_MODEL_REVISION,
            ANIMEGEN_IMPLEMENTATION_REVISION,
            animegen_sampling_profile,
            resolve_animegen_installation,
        )
        from aigen.workflow_compilation import CompiledAnimeGenConfig
        from aigen.runtime_provenance import build_python_runtime_provenance

        installation = config.installation if isinstance(config, CompiledAnimeGenConfig) else resolve_animegen_installation(
            lightning_required=animegen_sampling_profile(node.config.sampling).lightning)

        models = [
            _component("AnimeGen-I2V", ANIMEGEN_MODEL_REVISION),
            _component("Wan2.2 I2V base", ANIMEGEN_BASE_MODEL_REVISION),
            _component("AnimeGen local model files", model_artifact_stat_revision(
                ModelArtifactComponent("AnimeGen", installation.model.parent.parent, installation.files))),
            _component("AnimeGen Python runtime", str(build_python_runtime_provenance(
                ("torch", "diffusers", "transformers", "accelerate", "safetensors", "imageio", "imageio-ffmpeg"))["fingerprint"])),
            *_media_components(),
        ]
        if animegen_sampling_profile(node.config.sampling).lightning:
            models.append(
                _component("Wan2.2 Lightning LoRAs", ANIMEGEN_LIGHTNING_REVISION)
            )
        return _provenance(
            backend=_component("AnimeGen-I2V", ANIMEGEN_IMPLEMENTATION_REVISION),
            models=tuple(models),
        )
    if isinstance(node, Ltx23Node):
        from aigen.generation.ltx23_keyframes import LTX23_IMPLEMENTATION_REVISION
        from aigen.workflow_compilation import CompiledLtx23Config

        assert isinstance(config, CompiledLtx23Config)
        return _provenance(backend=_component("LTX-2.3", LTX23_IMPLEMENTATION_REVISION), models=(
            *(_component(name, revision) for name, revision in config.installation.provenance.items()),
            _component("decoded presentation timeline", media_runtime_revision()),
        ))
    if isinstance(node, HunyuanI2VNode):
        from aigen.generation.hunyuanvideo15 import (HUNYUANVIDEO15_IMPLEMENTATION_REVISION,
            HUNYUANVIDEO15_SOURCE_REVISION, HUNYUANVIDEO15_MODEL_REVISION, HUNYUANVIDEO15_RUNTIME_PATCH)
        from aigen.workflow_compilation import CompiledHunyuanConfig
        from aigen.runtime_provenance import build_python_runtime_provenance_for_interpreter

        assert isinstance(config, CompiledHunyuanConfig)
        installation = config.installation
        runtime = build_python_runtime_provenance_for_interpreter(installation.torchrun.parent / "python",
            ("torch", "transformers", "diffusers", "safetensors"))
        return _provenance(backend=_component("HunyuanVideo-1.5", HUNYUANVIDEO15_IMPLEMENTATION_REVISION), models=(
            _component("Hunyuan source", HUNYUANVIDEO15_SOURCE_REVISION),
            _component("Hunyuan model", HUNYUANVIDEO15_MODEL_REVISION),
            _component("Hunyuan local models", _path_inventory_revision(installation.model)),
            _component("Hunyuan runtime patch", _path_inventory_revision(HUNYUANVIDEO15_RUNTIME_PATCH)),
            _component("Hunyuan Python runtime", str(runtime["fingerprint"])), *_media_components(),
        ))
    if node.kind in {
        NodeKind.VIDEO_CONTACT_SHEET,
        NodeKind.EXTRACT_VIDEO_FRAMES,
        NodeKind.ASSEMBLE_VIDEO,
    }:
        return _provenance(
            backend=_component("video-postprocess", VIDEO_POSTPROCESS_REVISION), models=_media_components(),
        )
    return _provenance(
        backend=_component("workflow-source", "sha256-v1")
    )


def _media_components() -> tuple[RevisionedComponent, ...]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError("video workflows require an installed FFmpeg executable")
    return (_component("PyAV and decoded presentation timeline", media_runtime_revision()),
            _component("FFmpeg executable", _path_inventory_revision(Path(ffmpeg))),
            _component("FFmpeg and linked codec libraries", ffmpeg_runtime_revision(Path(ffmpeg))))


def _image_edit_provenance(
    node: ImageEditNode | CharacterEditNode | CharacterRefineNode,
) -> NodeExecutionProvenance:
    from aigen.generation.image_edit_batch import IMAGE_EDIT_BATCH_VERSION

    backend = node.config.backend
    batch_backend = backend in {
        FLUX2_KLEIN_BACKEND,
        QWEN_2511_LIGHTNING_BACKEND,
        QWEN_2511_BASE_BACKEND,
    }
    implementation_name = "image-edit-batch" if batch_backend else "image-edit-api"
    implementation_revision = str(IMAGE_EDIT_BATCH_VERSION) if batch_backend else IMAGE_EDIT_API_REVISION
    if backend == FLUX2_KLEIN_BACKEND:
        from aigen.generation.flux2_klein_artifacts import (
            FLUX2_KLEIN_IMPLEMENTATION_REVISION,
            flux2_klein_model_artifacts,
        )

        implementation_name = "image-edit-batch/flux2-klein"
        implementation_revision = f"{IMAGE_EDIT_BATCH_VERSION}/{FLUX2_KLEIN_IMPLEMENTATION_REVISION}"
        models = tuple(
            _component(
                component.name,
                model_artifact_stat_revision(component),
            )
            for component in flux2_klein_model_artifacts()
        )
    elif backend in {
        QWEN_2511_LIGHTNING_BACKEND,
        QWEN_2511_BASE_BACKEND,
    }:
        from aigen.generation.qwen_image_edit_lightx2v import (
            LIGHTX2V_REVISION,
            LIGHTX2V_QWEN_EDIT_2511_PROFILE,
            LIGHTX2V_QWEN_EDIT_2511_BASE_PROFILE,
            QWEN_IMAGE_EDIT_LIGHTX2V_PROFILES,
            QWEN_EDIT_IMPLEMENTATION_REVISION,
            QWEN_EDIT_2511_LIGHTNING_REVISION,
            QWEN_EDIT_2511_REVISION,
        )
        from aigen.generation.qwen_image_edit_artifacts import qwen_2511_model_artifacts

        profile_name = (LIGHTX2V_QWEN_EDIT_2511_PROFILE if backend == QWEN_2511_LIGHTNING_BACKEND
                        else LIGHTX2V_QWEN_EDIT_2511_BASE_PROFILE)

        implementation_name = "image-edit-batch/qwen-2511"
        implementation_revision = f"{IMAGE_EDIT_BATCH_VERSION}/{QWEN_EDIT_IMPLEMENTATION_REVISION}"
        models = (
            _component("LightX2V", LIGHTX2V_REVISION),
            _component("Qwen-Image-Edit-2511", QWEN_EDIT_2511_REVISION),
            _component(
                "Qwen-Image-Edit-2511 transformer",
                QWEN_EDIT_2511_LIGHTNING_REVISION,
            ),
            *(_component(component.name, model_artifact_stat_revision(component))
              for component in qwen_2511_model_artifacts(QWEN_IMAGE_EDIT_LIGHTX2V_PROFILES[profile_name])),
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
            USO_IMPLEMENTATION_REVISION,
            USO_MODEL_TYPE,
            USO_SOURCE_REVISION,
            uso_flux1_model_paths,
        )

        implementation_name = "image-edit-api/uso"
        implementation_revision = f"{IMAGE_EDIT_API_REVISION}/{USO_IMPLEMENTATION_REVISION}"
        models = (
            _component("USO source", USO_SOURCE_REVISION),
            _component("USO model", USO_MODEL_TYPE),
            *(
                _component(f"USO {name}", _path_inventory_revision(path))
                for name, path in uso_flux1_model_paths().items()
            ),
        )
    else:
        raise ValueError(f"unsupported image-edit backend: {backend}")
    return _provenance(
        backend=_component(implementation_name, implementation_revision), models=models,
    )


def _character_edit_provenance(node: CharacterEditNode | CharacterRefineNode, config: CompiledNodeConfig | None) -> NodeExecutionProvenance:
    from aigen.character_edit import CHARACTER_EDIT_REVISION
    from aigen.character_edit_audit import CHARACTER_AUDIT_REVISION
    from aigen.workflow_compilation import CompiledCharacterEditConfig, CompiledCharacterRefineConfig
    from aigen.runtime_provenance import build_python_runtime_provenance

    assert isinstance(config, (CompiledCharacterEditConfig, CompiledCharacterRefineConfig))
    generation = _image_edit_provenance(node)
    components = [
        generation.backend, *generation.models,
        _component("character raw audit and retry", CHARACTER_EDIT_REVISION),
        _component("character audit prompt", CHARACTER_AUDIT_REVISION),
        _component("character audit model and processor", _path_inventory_revision(config.audit.model)),
        _component("character audit runtime", str(build_python_runtime_provenance(
            ("torch", "transformers", "accelerate", "bitsandbytes", "qwen-vl-utils", "pillow", "numpy"))["fingerprint"])),
    ]
    if isinstance(config, CompiledCharacterRefineConfig):
        from aigen.generation.qwen_image_edit_masked import QWEN_MASKED_EDIT_REVISION
        components.append(_component("native masked edit and exact preservation", QWEN_MASKED_EDIT_REVISION))
        return _provenance(backend=_component("character-refine", QWEN_MASKED_EDIT_REVISION), models=tuple(components))
    components.extend(_component(f"character control {path.name}", _path_inventory_revision(path)) for path in config.conditioning_models)
    if config.conditioning_runtime:
        components.append(_component("character control runtime", str(
            build_python_runtime_provenance(config.conditioning_runtime)["fingerprint"])))
    if config.upscale_long_side is not None:
        from aigen.generation.vosr_backend import VOSR_POSTPROCESS_NAME
        upscale = _postprocess_provenance(VOSR_POSTPROCESS_NAME)
        components.extend((upscale.backend, *upscale.models))
    return _provenance(backend=_component("character-edit", CHARACTER_EDIT_REVISION), models=tuple(components))


def _postprocess_provenance(model: str) -> NodeExecutionProvenance:
    from aigen.generation.vosr_backend import VOSR_POSTPROCESS_NAME

    implementation_name = "image-postprocess-batch"
    implementation_revision = IMAGE_POSTPROCESS_BATCH_REVISION
    if model == VOSR_POSTPROCESS_NAME:
        from aigen.generation.vosr_backend import (
            VOSR_IMPLEMENTATION_REVISION,
            VOSR_MODEL_REVISION,
            VOSR_SOURCE_REVISION,
            VOSR_CHECKPOINT,
            VOSR_VAE,
            VOSR_DINOV2_WEIGHTS,
        )

        implementation_name = "image-postprocess-batch/vosr"
        implementation_revision = f"{IMAGE_POSTPROCESS_BATCH_REVISION}/{VOSR_IMPLEMENTATION_REVISION}"
        models = (
            _component("VOSR source", VOSR_SOURCE_REVISION),
            _component("VOSR model", VOSR_MODEL_REVISION),
            *(_component(f"VOSR {name}", _path_inventory_revision(path)) for name, path in (
                ("checkpoint", VOSR_CHECKPOINT), ("VAE", VOSR_VAE), ("DINOv2", VOSR_DINOV2_WEIGHTS),
            )),
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
        from aigen.generation.image_upscale import (
            IMAGE_UPSCALE_IMPLEMENTATION_REVISION,
            upscale_model_path,
        )

        implementation_name = "image-postprocess-batch/illustration-upscale"
        implementation_revision = f"{IMAGE_POSTPROCESS_BATCH_REVISION}/{IMAGE_UPSCALE_IMPLEMENTATION_REVISION}"
        models = (
            _component(
                model,
                _path_inventory_revision(upscale_model_path(model)),
            ),
        )
    return _provenance(
        backend=_component(implementation_name, implementation_revision),
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
