from __future__ import annotations

from pathlib import Path

from aigen.keyframe_job_models import KeyframeJobError
from aigen.keyframe_profiles import KeyframeProfile, KeyframeRefineProfile
from aigen.keyframe_refine_models import KeyframeRefineError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "aigen" / "models"

KEYFRAME_MODEL_REVISIONS = {
    "kontext": {
        "repo_id": "eramth/flux-kontext-4bit-fp4",
        "revision": "499964b43d54eda6ca7e21c346efe18e2a1cdad8",
    },
    "controlnet": {
        "repo_id": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        "revision": "5d700aaad96c5ddcdf8a38ef9b22a82aac2c38e5",
    },
    "nunchaku_transformer": {
        "repo_id": "nunchaku-tech/nunchaku-flux.1-kontext-dev",
        "revision": "70dff7728491f3016e256137e8f7d87812af0b4f",
    },
}


KEYFRAME_PROFILE = KeyframeProfile(
    name="nunchaku-kontext-pose-quality",
    model=(MODELS_ROOT / "diffusers/eramth/flux-kontext-4bit-fp4").as_posix(),
    controlnet_model=(
        MODELS_ROOT / "diffusers/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
    ).as_posix(),
    nunchaku_transformer_model=(
        MODELS_ROOT
        / "nunchaku/nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-fp4_r32-flux.1-kontext-dev.safetensors"
    ),
    attention_impl="nunchaku-fp16",
    dtype="bfloat16",
    pipeline_cpu_offload=True,
    nunchaku_layer_offload=False,
    vae_tiling=False,
    model_revisions=KEYFRAME_MODEL_REVISIONS,
)


KEYFRAME_REFINE_PROFILES = {
    "kontext-inpaint-local": KeyframeRefineProfile(
        name="kontext-inpaint-local",
        model=(MODELS_ROOT / "diffusers/eramth/flux-kontext-4bit-fp4").as_posix(),
        nunchaku_transformer_model=(
            MODELS_ROOT
            / "nunchaku/nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-fp4_r32-flux.1-kontext-dev.safetensors"
        ),
        attention_impl="nunchaku-fp16",
        dtype="bfloat16",
        pipeline_cpu_offload=False,
        vae_tiling=False,
        model_revisions={
            "kontext": KEYFRAME_MODEL_REVISIONS["kontext"],
            "nunchaku_transformer": KEYFRAME_MODEL_REVISIONS["nunchaku_transformer"],
        },
    ),
}


def keyframe_profile_for_name(profile_name: str) -> KeyframeProfile:
    if profile_name != KEYFRAME_PROFILE.name:
        raise KeyframeJobError(f"Unknown keyframe profile: {profile_name}")
    return KEYFRAME_PROFILE


def keyframe_refine_profile_for_name(profile_name: str) -> KeyframeRefineProfile:
    try:
        return KEYFRAME_REFINE_PROFILES[profile_name]
    except KeyError as error:
        raise KeyframeRefineError(f"Unknown keyframe refine profile: {profile_name}") from error
