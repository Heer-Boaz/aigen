from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from aigen.model_artifacts import (
    ModelArtifactComponent,
    build_model_artifact_provenance,
)
from aigen.runtime_profiles import MODELS_ROOT
from aigen.runtime_provenance import build_python_runtime_provenance


FLUX2_KLEIN_IMPLEMENTATION_REVISION = "1"
FLUX2_KLEIN_MODEL_ROOT = MODELS_ROOT / "flux2/black-forest-labs/FLUX.2-klein-9B"
FLUX2_KLEIN_TRANSFORMER = (
    MODELS_ROOT
    / "flux2/black-forest-labs/FLUX.2-klein-9b-fp8/flux-2-klein-9b-fp8.safetensors"
)
FLUX2_KLEIN_TEXT_ENCODER = MODELS_ROOT / "flux2/Qwen/Qwen3-8B-FP8"
FLUX2_KLEIN_RUNTIME_DISTRIBUTIONS = (
    "accelerate",
    "comfy-kitchen",
    "diffusers",
    "einops",
    "flux",
    "numpy",
    "Pillow",
    "safetensors",
    "torch",
    "tokenizers",
    "transformers",
)


@lru_cache(maxsize=1)
def flux2_klein_model_artifacts() -> tuple[ModelArtifactComponent, ...]:
    transformer = FLUX2_KLEIN_TRANSFORMER.resolve()
    vae_scheduler_files = _files_below(
        FLUX2_KLEIN_MODEL_ROOT,
        ("vae", "scheduler"),
    )
    conditioner_files = tuple(
        path
        for path in sorted(FLUX2_KLEIN_TEXT_ENCODER.resolve().iterdir())
        if path.is_file()
    )
    return (
        ModelArtifactComponent(
            name="FLUX.2 Klein scaled-FP8 transformer",
            root=transformer.parent,
            files=(transformer,),
        ),
        ModelArtifactComponent(
            name="FLUX.2 Klein VAE and scheduler",
            root=FLUX2_KLEIN_MODEL_ROOT.resolve(),
            files=vae_scheduler_files,
        ),
        ModelArtifactComponent(
            name="Qwen3-8B-FP8 conditioner",
            root=FLUX2_KLEIN_TEXT_ENCODER.resolve(),
            files=conditioner_files,
        ),
    )


def flux2_klein_model_provenance() -> dict[str, object]:
    return build_model_artifact_provenance(flux2_klein_model_artifacts())


def flux2_klein_runtime_provenance() -> dict[str, object]:
    return build_python_runtime_provenance(FLUX2_KLEIN_RUNTIME_DISTRIBUTIONS)


def _files_below(root: Path, directories: tuple[str, ...]) -> tuple[Path, ...]:
    return tuple(
        path
        for directory in directories
        for path in sorted((root / directory).resolve().rglob("*"))
        if path.is_file()
    )
