from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from safetensors import SafetensorError, safe_open


QWEN_IMAGE_ARCHITECTURE = "qwen-image"
FLUX2_KLEIN_ARCHITECTURE = "flux2-klein"

AI_TOOLKIT_LORA_FORMAT = "ai-toolkit-lora"
DIFFUSERS_PEFT_LORA_FORMAT = "diffusers-peft-lora"
LOKR_FORMAT = "lokr"

_A_SUFFIX = ".lora_A.weight"
_B_SUFFIX = ".lora_B.weight"
_ALPHA_SUFFIX = ".alpha"
_LOKR_SUFFIXES = (".alpha", ".lokr_w1", ".lokr_w2")

_QWEN_TARGET = re.compile(
    r"^diffusion_model\.transformer_blocks\.\d+\."
    r"(?:"
    r"attn\.(?:add_[kqv]_proj|to_add_out|to_[kqv]|to_out\.0)"
    r"|(?:img|txt)_mlp\.net\.(?:0\.proj|2)"
    r"|(?:img|txt)_mod\.1"
    r")$"
)
_FLUX2_NATIVE_TARGET = re.compile(
    r"^diffusion_model\."
    r"(?:"
    r"double_blocks\.\d+\."
    r"(?:"
    r"(?:img|txt)_attn\.(?:proj|qkv)"
    r"|(?:img|txt)_mlp\.(?:0|2)"
    r")"
    r"|single_blocks\.\d+\.linear[12]"
    r")$"
)
_FLUX2_PEFT_TARGET = re.compile(
    r"^transformer\."
    r"(?:"
    r"single_transformer_blocks\.\d+\.attn\.(?:to_out|to_qkv_mlp_proj)"
    r"|transformer_blocks\.\d+\.attn\.(?:to_[kqv]|to_out\.0)"
    r")$"
)


class LoraWeightsError(ValueError):
    pass


@dataclass(frozen=True)
class LoraWeightsInfo:
    path: Path
    architecture: str
    format: str
    target_count: int


@dataclass(frozen=True)
class LoraLoadSpec:
    path: Path
    weight: float

    def to_json(self) -> dict[str, str | float]:
        return {"path": self.path.as_posix(), "weight": self.weight}


def inspect_lora_weights(path: Path) -> LoraWeightsInfo:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise LoraWeightsError(f"LoRA weights do not exist: {resolved}")
    stat = resolved.stat()
    return _inspect_lora_weights(resolved, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=16)
def _inspect_lora_weights(
    path: Path,
    _modified_ns: int,
    _size: int,
) -> LoraWeightsInfo:
    try:
        with safe_open(path, framework="pt", device="cpu") as weights:
            keys = tuple(weights.keys())
            metadata = weights.metadata() or {}
            if not keys:
                raise LoraWeightsError(f"LoRA contains no tensors: {path}")
            if any(key.endswith((".lora_A.weight", ".lora_B.weight")) for key in keys):
                architecture, format_name, targets = _inspect_lora_pairs(
                    weights,
                    keys,
                    metadata,
                    path,
                )
            elif any(key.endswith((".lokr_w1", ".lokr_w2")) for key in keys):
                architecture = FLUX2_KLEIN_ARCHITECTURE
                format_name = LOKR_FORMAT
                targets = _inspect_lokr(weights, keys, path)
            else:
                raise LoraWeightsError(f"Unsupported LoRA tensor format: {path}")
    except SafetensorError as error:
        raise LoraWeightsError(f"Invalid SafeTensors LoRA: {path}: {error}") from error

    metadata_architecture = _metadata_architecture(metadata.get("ss_base_model_version"))
    if metadata_architecture is not None and metadata_architecture != architecture:
        raise LoraWeightsError(
            f"LoRA metadata targets {metadata_architecture}, but its tensors target "
            f"{architecture}: {path}"
        )
    return LoraWeightsInfo(
        path=path,
        architecture=architecture,
        format=format_name,
        target_count=len(targets),
    )


def _inspect_lora_pairs(
    weights,
    keys: tuple[str, ...],
    metadata: dict[str, str],
    path: Path,
) -> tuple[str, str, tuple[str, ...]]:
    targets = tuple(sorted(key.removesuffix(_A_SUFFIX) for key in keys if key.endswith(_A_SUFFIX)))
    if not targets:
        raise LoraWeightsError(f"LoRA contains B factors without A factors: {path}")
    expected = {
        f"{target}{suffix}"
        for target in targets
        for suffix in (_A_SUFFIX, _B_SUFFIX)
    }
    expected.update(key for key in keys if key.endswith(_ALPHA_SUFFIX))
    if set(keys) != expected:
        raise LoraWeightsError(f"LoRA has incomplete or mixed A/B tensors: {path}")

    alpha_targets = {
        key.removesuffix(_ALPHA_SUFFIX)
        for key in keys
        if key.endswith(_ALPHA_SUFFIX)
    }
    if not alpha_targets.issubset(targets):
        raise LoraWeightsError(f"LoRA has alpha tensors without matching A/B factors: {path}")

    for target in targets:
        a_shape = tuple(weights.get_slice(f"{target}{_A_SUFFIX}").get_shape())
        b_shape = tuple(weights.get_slice(f"{target}{_B_SUFFIX}").get_shape())
        if len(a_shape) != 2 or len(b_shape) != 2 or b_shape[1] != a_shape[0]:
            raise LoraWeightsError(
                f"LoRA target {target} has incompatible A/B shapes {a_shape} and {b_shape}: {path}"
            )
        alpha_key = f"{target}{_ALPHA_SUFFIX}"
        if alpha_key in alpha_targets and weights.get_slice(alpha_key).get_shape():
            raise LoraWeightsError(f"LoRA alpha must be scalar for target {target}: {path}")

    if all(_QWEN_TARGET.fullmatch(target) for target in targets):
        return QWEN_IMAGE_ARCHITECTURE, AI_TOOLKIT_LORA_FORMAT, targets
    if all(_FLUX2_NATIVE_TARGET.fullmatch(target) for target in targets):
        return FLUX2_KLEIN_ARCHITECTURE, AI_TOOLKIT_LORA_FORMAT, targets
    if metadata.get("lora_adapter_metadata") is not None and all(
        _FLUX2_PEFT_TARGET.fullmatch(target) for target in targets
    ):
        _inspect_peft_metadata(metadata["lora_adapter_metadata"], path)
        return FLUX2_KLEIN_ARCHITECTURE, DIFFUSERS_PEFT_LORA_FORMAT, targets
    raise LoraWeightsError(f"LoRA uses an unknown or mixed model keyspace: {path}")


def _inspect_lokr(weights, keys: tuple[str, ...], path: Path) -> tuple[str, ...]:
    targets = tuple(sorted(key.removesuffix(".lokr_w1") for key in keys if key.endswith(".lokr_w1")))
    expected = {f"{target}{suffix}" for target in targets for suffix in _LOKR_SUFFIXES}
    if not targets or set(keys) != expected:
        raise LoraWeightsError(f"LoKr has incomplete or mixed factor tensors: {path}")
    if not all(_FLUX2_NATIVE_TARGET.fullmatch(target) for target in targets):
        raise LoraWeightsError(f"LoKr uses an unknown or mixed model keyspace: {path}")
    for target in targets:
        w1_shape = tuple(weights.get_slice(f"{target}.lokr_w1").get_shape())
        w2_shape = tuple(weights.get_slice(f"{target}.lokr_w2").get_shape())
        alpha_shape = tuple(weights.get_slice(f"{target}.alpha").get_shape())
        if len(w1_shape) != 2 or len(w2_shape) != 2 or alpha_shape:
            raise LoraWeightsError(f"LoKr target {target} has invalid factor shapes: {path}")
    return targets


def _inspect_peft_metadata(value: str, path: Path) -> None:
    try:
        metadata = json.loads(value)
        rank = int(metadata["transformer.r"])
        alpha = float(metadata["transformer.lora_alpha"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise LoraWeightsError(f"Invalid PEFT adapter metadata: {path}") from error
    if rank < 1 or not math.isfinite(alpha):
        raise LoraWeightsError(f"Invalid PEFT rank or alpha: {path}")


def _metadata_architecture(value: str | None) -> str | None:
    if value is None:
        return None
    if value.startswith("qwen_image"):
        return QWEN_IMAGE_ARCHITECTURE
    if value.startswith("flux2_klein"):
        return FLUX2_KLEIN_ARCHITECTURE
    return None
