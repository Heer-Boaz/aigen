from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.generation.image_generation_requests import ImageGenerationCaseRequest
from aigen.image_edit_defaults import (
    QWEN_2511_BASE_DEFAULT_GUIDANCE,
    QWEN_2511_BASE_DEFAULT_STEPS,
    QWEN_2511_LIGHTNING_DEFAULT_GUIDANCE,
    QWEN_2511_LIGHTNING_DEFAULT_STEPS,
)
from aigen.lora_weights import LoraLoadSpec
from aigen.progress import StatusReporter
from aigen.runtime_profiles import MODELS_ROOT, PROJECT_ROOT


LIGHTX2V_QWEN_EDIT_2511_PROFILE = "lightx2v-qwen-edit-2511-fp8-lightning-8step"
LIGHTX2V_QWEN_EDIT_2511_BASE_PROFILE = "lightx2v-qwen-edit-2511-fp8-base-40step"
LIGHTX2V_REVISION = "b96309e82899145aebd8ecf95c387894aba66b1e"
QWEN_EDIT_2511_REVISION = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
QWEN_EDIT_2511_LIGHTNING_REVISION = "d74eba145674fd7e31b949324e148e21e7118abd"


class QwenImageEditLightX2VError(RuntimeError):
    pass


@dataclass(frozen=True)
class QwenImageEditLightX2VProfile:
    name: str
    base_model: str
    base_repo_id: str
    base_revision: str
    transformer_model: Path
    transformer_repo_id: str
    transformer_revision: str
    transformer_variant: str
    conditioner_model: Path
    conditioner_repo_id: str
    conditioner_revision: str
    dtype: str
    default_steps: int
    default_true_cfg_scale: float
    default_guidance_scale: float
    scheduler: str
    local_files_only: bool = True


@dataclass(frozen=True)
class LightX2VQwenResult:
    outputs: tuple[dict[str, Any], ...]
    timings_ms: dict[str, Any]
    memory: dict[str, Any]
    environment: dict[str, Any]


QWEN_IMAGE_EDIT_2511_LOCAL_MODEL = (
    MODELS_ROOT / "lightx2v/Qwen/Qwen-Image-Edit-2511"
).as_posix()
QWEN_IMAGE_EDIT_2511_LIGHTNING_MODEL = (
    MODELS_ROOT
    / "lightx2v/Qwen/Qwen-Image-Edit-2511-Lightning"
    / "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors"
)
QWEN_IMAGE_EDIT_2511_BASE_MODEL = (
    MODELS_ROOT
    / "lightx2v/Qwen/Qwen-Image-Edit-2511-Lightning"
    / "qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors"
)
QWEN25_VL_BF16_MODEL = (
    MODELS_ROOT / "lightx2v/Qwen/Qwen-Image-Edit-2511/text_encoder"
)

QWEN_IMAGE_EDIT_LIGHTX2V_PROFILES = {
    LIGHTX2V_QWEN_EDIT_2511_PROFILE: QwenImageEditLightX2VProfile(
        name=LIGHTX2V_QWEN_EDIT_2511_PROFILE,
        base_model=QWEN_IMAGE_EDIT_2511_LOCAL_MODEL,
        base_repo_id="Qwen/Qwen-Image-Edit-2511",
        base_revision=QWEN_EDIT_2511_REVISION,
        transformer_model=QWEN_IMAGE_EDIT_2511_LIGHTNING_MODEL,
        transformer_repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        transformer_revision=QWEN_EDIT_2511_LIGHTNING_REVISION,
        transformer_variant="fp8_e4m3fn_scaled_lightning_8steps_v1.0",
        conditioner_model=QWEN25_VL_BF16_MODEL,
        conditioner_repo_id="Qwen/Qwen-Image-Edit-2511",
        conditioner_revision=QWEN_EDIT_2511_REVISION,
        dtype="bfloat16",
        default_steps=QWEN_2511_LIGHTNING_DEFAULT_STEPS,
        default_true_cfg_scale=QWEN_2511_LIGHTNING_DEFAULT_GUIDANCE,
        default_guidance_scale=1.0,
        scheduler="lightning",
    ),
    LIGHTX2V_QWEN_EDIT_2511_BASE_PROFILE: QwenImageEditLightX2VProfile(
        name=LIGHTX2V_QWEN_EDIT_2511_BASE_PROFILE,
        base_model=QWEN_IMAGE_EDIT_2511_LOCAL_MODEL,
        base_repo_id="Qwen/Qwen-Image-Edit-2511",
        base_revision=QWEN_EDIT_2511_REVISION,
        transformer_model=QWEN_IMAGE_EDIT_2511_BASE_MODEL,
        transformer_repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        transformer_revision=QWEN_EDIT_2511_LIGHTNING_REVISION,
        transformer_variant="fp8_e4m3fn_scaled",
        conditioner_model=QWEN25_VL_BF16_MODEL,
        conditioner_repo_id="Qwen/Qwen-Image-Edit-2511",
        conditioner_revision=QWEN_EDIT_2511_REVISION,
        dtype="bfloat16",
        default_steps=QWEN_2511_BASE_DEFAULT_STEPS,
        default_true_cfg_scale=QWEN_2511_BASE_DEFAULT_GUIDANCE,
        default_guidance_scale=1.0,
        scheduler="flow-match-euler",
    ),
}


def run_lightx2v_qwen_image_edit(
    *,
    profile: QwenImageEditLightX2VProfile,
    cases: Sequence[ImageGenerationCaseRequest],
    steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    max_sequence_length: int,
    loras: tuple[LoraLoadSpec, ...],
    progress: StatusReporter,
) -> LightX2VQwenResult:
    if guidance_scale != 1.0:
        raise QwenImageEditLightX2VError(
            "Qwen-Image-Edit-2511 LightX2V does not use guidance embeddings; guidance_scale must be 1"
        )
    if profile.scheduler == "lightning" and true_cfg_scale != 1.0:
        raise QwenImageEditLightX2VError(
            "Qwen-Image-Edit-2511 Lightning FP8 runs without CFG; true_cfg_scale must be 1"
        )

    runtime_root = _lightx2v_runtime_root()
    python = runtime_root / "venv/bin/python"
    source = runtime_root / "LightX2V"
    required_paths = (
        python,
        source / "lightx2v/__init__.py",
        Path(profile.base_model) / "model_index.json",
        profile.transformer_model,
        profile.conditioner_model / "config.json",
        *(lora.path for lora in loras),
    )
    missing = [path.as_posix() for path in required_paths if not path.exists()]
    if missing:
        raise QwenImageEditLightX2VError(
            "Qwen-Image-Edit-2511 LightX2V runtime is incomplete: " + ", ".join(missing)
        )

    with tempfile.TemporaryDirectory(prefix="aigen-lightx2v-job-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        request_path = temporary_path / "request.json"
        response_path = temporary_path / "response.json"
        worker_log_path = temporary_path / "worker.log"
        worker_profile: dict[str, Any] = {
            "name": profile.name,
            "base_model": profile.base_model,
            "transformer_model": profile.transformer_model.as_posix(),
            "conditioner_model": profile.conditioner_model.as_posix(),
            "steps": steps,
            "true_cfg_scale": true_cfg_scale,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
        }
        if loras:
            worker_profile["loras"] = [
                {"path": lora.path.as_posix(), "strength": lora.weight}
                for lora in loras
            ]
        request_path.write_text(
            json.dumps(
                {
                    "kind": "aigen-qwen-image-edit-lightx2v-job",
                    "profile": worker_profile,
                    "cases": [
                        {
                            "name": case.name,
                            "prompt": case.prompt,
                            "image_paths": [path.as_posix() for path in case.image_paths],
                            "width": case.width,
                            "height": case.height,
                            "outputs": [
                                {
                                    "name": output.name,
                                    "seed": output.seed,
                                    "path": output.path.as_posix(),
                                }
                                for output in case.outputs
                            ],
                        }
                        for case in cases
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        environment = os.environ.copy()
        python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            PROJECT_ROOT.as_posix()
            if not python_path
            else os.pathsep.join((PROJECT_ROOT.as_posix(), python_path))
        )
        progress.phase("starting Qwen worker")
        with worker_log_path.open("w", encoding="utf-8") as worker_log:
            with subprocess.Popen(
                [
                    python.as_posix(),
                    "-m",
                    "aigen.generation.qwen_image_edit_lightx2v_worker",
                    request_path.as_posix(),
                    response_path.as_posix(),
                ],
                cwd=source,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=worker_log,
                text=True,
                encoding="utf-8",
                bufsize=1,
            ) as worker:
                try:
                    for line in worker.stdout:
                        _apply_worker_progress(line, progress)
                    returncode = worker.wait()
                except BaseException:
                    if worker.poll() is None:
                        worker.terminate()
                    raise
        response = _read_worker_response(response_path)
        if returncode != 0 or response.get("status") != "completed":
            message = response.get("message")
            if not message:
                worker_output = worker_log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
                message = worker_output[-8192:] or f"worker exited with status {returncode}"
            raise QwenImageEditLightX2VError(f"Qwen-Image-Edit-2511 LightX2V failed: {message}")
        progress.phase("generation completed")
        return LightX2VQwenResult(
            outputs=tuple(response["outputs"]),
            timings_ms=dict(response["timings_ms"]),
            memory=dict(response["memory"]),
            environment=dict(response["environment"]),
        )


def lightx2v_profile_json(profile: QwenImageEditLightX2VProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "dtype": profile.dtype,
        "load_strategy": "lightx2v-qwen-image-edit-2511-fp8-block-offload",
        "scheduler": profile.scheduler,
        "prompt_conditioning": "qwen25-vl-fp8-w8a8-language-bf16-vision-native-multimodal",
        "default_steps": profile.default_steps,
        "default_true_cfg_scale": profile.default_true_cfg_scale,
        "default_guidance_scale": profile.default_guidance_scale,
        "local_files_only": profile.local_files_only,
        "models": {
            "base": {
                "repo_id": profile.base_repo_id,
                "revision": profile.base_revision,
                "path": profile.base_model,
            },
            "lightx2v_transformer": {
                "repo_id": profile.transformer_repo_id,
                "revision": profile.transformer_revision,
                "variant": profile.transformer_variant,
                "path": profile.transformer_model.as_posix(),
            },
            "conditioner": {
                "repo_id": profile.conditioner_repo_id,
                "revision": profile.conditioner_revision,
                "path": profile.conditioner_model.as_posix(),
            },
            "engine": {
                "repo_id": "ModelTC/LightX2V",
                "revision": LIGHTX2V_REVISION,
                "path": (_lightx2v_runtime_root() / "LightX2V").as_posix(),
            },
        },
    }


def _lightx2v_runtime_root() -> Path:
    configured = os.environ.get("AIGEN_LIGHTX2V_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path.home() / ".cache/aigen-lightx2v"


def _read_worker_response(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QwenImageEditLightX2VError(f"Invalid LightX2V worker response {path}: {error}") from error


def _apply_worker_progress(line: str, progress: StatusReporter) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as error:
        raise QwenImageEditLightX2VError("Invalid LightX2V worker progress event") from error
    match event["kind"]:
        case "phase":
            progress.phase(event["text"])
        case "begin":
            progress.begin(event["total"], event["text"])
        case "step":
            progress.step(event["text"])
        case kind:
            raise QwenImageEditLightX2VError(f"Unknown LightX2V worker progress event: {kind}")
