from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from aigen.generation.qwen_image_edit_lightx2v import (
    QWEN25_VL_BF16_MODEL,
    QWEN_IMAGE_EDIT_2511_LIGHTNING_MODEL,
    QWEN_IMAGE_EDIT_2511_LOCAL_MODEL,
    lightx2v_runtime_root,
)
from aigen.model_artifacts import (
    ModelArtifactComponent,
    build_model_artifact_provenance,
)
from aigen.runtime_provenance import (
    build_python_runtime_provenance_for_interpreter,
)


QWEN_IMAGE_EDIT_LIGHTX2V_IMPLEMENTATION_REVISION = "1"
QWEN_IMAGE_EDIT_LIGHTX2V_RUNTIME_DISTRIBUTIONS = (
    "accelerate",
    "diffusers",
    "flash-attn",
    "lightx2v",
    "numpy",
    "Pillow",
    "safetensors",
    "torch",
    "transformers",
    "triton",
)


@lru_cache(maxsize=1)
def qwen_2511_lightning_model_artifacts() -> tuple[ModelArtifactComponent, ...]:
    model_root = Path(QWEN_IMAGE_EDIT_2511_LOCAL_MODEL).resolve()
    transformer = QWEN_IMAGE_EDIT_2511_LIGHTNING_MODEL.resolve()
    return (
        _directory_component(
            "Qwen-Image-Edit-2511 conditioner",
            QWEN25_VL_BF16_MODEL.resolve(),
        ),
        _directory_component(
            "Qwen-Image-Edit-2511 processor",
            (model_root / "processor").resolve(),
        ),
        _directory_component(
            "Qwen-Image-Edit-2511 scheduler",
            (model_root / "scheduler").resolve(),
        ),
        ModelArtifactComponent(
            name="Qwen-Image-Edit-2511 scaled-FP8 Lightning transformer",
            root=transformer.parent,
            files=(transformer,),
        ),
        _directory_component(
            "Qwen-Image-Edit-2511 tokenizer",
            (model_root / "tokenizer").resolve(),
        ),
        _directory_component(
            "Qwen-Image-Edit-2511 VAE",
            (model_root / "vae").resolve(),
        ),
        ModelArtifactComponent(
            name="Qwen-Image-Edit-2511 worker metadata",
            root=model_root,
            files=(
                model_root / "model_index.json",
                model_root / "transformer/config.json",
            ),
        ),
    )


def qwen_2511_lightning_model_provenance() -> dict[str, object]:
    return build_model_artifact_provenance(
        qwen_2511_lightning_model_artifacts()
    )


def qwen_2511_lightx2v_runtime_provenance() -> dict[str, object]:
    return build_python_runtime_provenance_for_interpreter(
        lightx2v_runtime_root() / "venv/bin/python",
        QWEN_IMAGE_EDIT_LIGHTX2V_RUNTIME_DISTRIBUTIONS,
    )


def _directory_component(name: str, root: Path) -> ModelArtifactComponent:
    return ModelArtifactComponent(
        name=name,
        root=root,
        files=tuple(
            path
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ),
    )
