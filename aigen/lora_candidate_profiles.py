from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aigen.runtime_profiles import KEYFRAME_MODEL_REVISIONS, MODELS_ROOT


class LoraCandidateProfileError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoraCandidateProfile:
    name: str
    model: str
    nunchaku_transformer_model: Path
    attention_impl: str
    dtype: str
    vae_tiling: bool
    model_revisions: dict[str, dict[str, str]]


LORA_CANDIDATE_PROFILE = LoraCandidateProfile(
    name="nunchaku-kontext-identity-candidates",
    model=(MODELS_ROOT / "diffusers/eramth/flux-kontext-4bit-fp4").as_posix(),
    nunchaku_transformer_model=(
        MODELS_ROOT
        / "nunchaku/nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-fp4_r32-flux.1-kontext-dev.safetensors"
    ),
    attention_impl="nunchaku-fp16",
    dtype="bfloat16",
    vae_tiling=False,
    model_revisions={
        "kontext": KEYFRAME_MODEL_REVISIONS["kontext"],
        "nunchaku_transformer": KEYFRAME_MODEL_REVISIONS["nunchaku_transformer"],
    },
)


def lora_candidate_profile_for_name(profile_name: str) -> LoraCandidateProfile:
    if profile_name != LORA_CANDIDATE_PROFILE.name:
        raise LoraCandidateProfileError(f"Unknown LoRA candidate profile: {profile_name}")
    return LORA_CANDIDATE_PROFILE
