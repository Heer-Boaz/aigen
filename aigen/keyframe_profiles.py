from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KeyframeProfile:
    name: str
    model: str
    controlnet_model: str
    nunchaku_transformer_model: Path
    attention_impl: str
    dtype: str
    pipeline_cpu_offload: bool
    nunchaku_layer_offload: bool
    vae_tiling: bool
    model_revisions: dict[str, dict[str, str]]


@dataclass(frozen=True)
class KeyframeRefineProfile:
    name: str
    model: str
    nunchaku_transformer_model: Path
    attention_impl: str
    dtype: str
    pipeline_cpu_offload: bool
    vae_tiling: bool
    model_revisions: dict[str, dict[str, str]]
