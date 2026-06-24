from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aigen.generation.character_concept import DEFAULT_NEGATIVE_PROMPT, DTYPES


@dataclass(frozen=True)
class CharacterPoseResult:
    output_path: str
    base_model: str
    controlnet_model: str
    pose_image: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    true_cfg_scale: float
    controlnet_conditioning_scale: float
    control_guidance_start: float
    control_guidance_end: float
    control_mode: int | None
    dtype: str
    device: str
    cpu_offload: bool
    seed: int

    def to_json(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "base_model": self.base_model,
            "controlnet_model": self.controlnet_model,
            "pose_image": self.pose_image,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "true_cfg_scale": self.true_cfg_scale,
            "controlnet_conditioning_scale": self.controlnet_conditioning_scale,
            "control_guidance_start": self.control_guidance_start,
            "control_guidance_end": self.control_guidance_end,
            "control_mode": self.control_mode,
            "dtype": self.dtype,
            "device": self.device,
            "cpu_offload": self.cpu_offload,
            "seed": self.seed,
        }


class CharacterPoseError(RuntimeError):
    pass


class CharacterPoseDependencyError(CharacterPoseError):
    pass


def run_character_pose_control(
    base_model: str,
    controlnet_model: str,
    pose_image: Path,
    output_path: Path,
    prompt: str,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    steps: int = 30,
    guidance_scale: float = 3.5,
    true_cfg_scale: float = 1.0,
    width: int = 768,
    height: int = 1152,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 1,
    cpu_offload: bool = True,
    controlnet_conditioning_scale: float = 0.9,
    control_guidance_start: float = 0.0,
    control_guidance_end: float = 0.65,
    control_mode: int | None = None,
) -> CharacterPoseResult:
    torch, pipeline_class, controlnet_class, load_image = _load_flux_controlnet()
    torch_dtype = _torch_dtype(torch, dtype)
    pipeline = _build_pose_pipeline(
        pipeline_class,
        controlnet_class,
        base_model,
        controlnet_model,
        torch_dtype,
        device,
        cpu_offload,
    )

    image = pipeline(
        **_pose_args(
            torch,
            load_image,
            pose_image,
            prompt,
            negative_prompt,
            true_cfg_scale,
            width,
            height,
            steps,
            guidance_scale,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            control_mode,
            seed,
            device,
        )
    ).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return CharacterPoseResult(
        output_path=output_path.resolve().as_posix(),
        base_model=base_model,
        controlnet_model=controlnet_model,
        pose_image=pose_image.resolve().as_posix(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_start=control_guidance_start,
        control_guidance_end=control_guidance_end,
        control_mode=control_mode,
        dtype=dtype,
        device=device,
        cpu_offload=cpu_offload,
        seed=seed,
    )


def _build_pose_pipeline(
    pipeline_class: Any,
    controlnet_class: Any,
    base_model: str,
    controlnet_model: str,
    torch_dtype: Any,
    device: str,
    cpu_offload: bool,
) -> Any:
    controlnet = controlnet_class.from_pretrained(
        controlnet_model,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    pipeline = pipeline_class.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    pipeline.vae.enable_tiling()
    pipeline.vae.enable_slicing()
    if cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)
    return pipeline


def _pose_args(
    torch_module: Any,
    load_image: Any,
    pose_image: Path,
    prompt: str,
    negative_prompt: str,
    true_cfg_scale: float,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    control_guidance_start: float,
    control_guidance_end: float,
    control_mode: int | None,
    seed: int,
    device: str,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "true_cfg_scale": true_cfg_scale,
        "control_image": load_image(pose_image.resolve().as_posix()),
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
        "control_guidance_start": control_guidance_start,
        "control_guidance_end": control_guidance_end,
        "control_mode": control_mode,
        "generator": torch_module.Generator(device=device).manual_seed(seed),
    }


def _load_flux_controlnet() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from diffusers import FluxControlNetModel, FluxControlNetPipeline
        from diffusers.utils import load_image
    except ImportError as exc:
        raise CharacterPoseDependencyError(
            "pose-controlled character generation requires `pip install -e .[generation]`"
        ) from exc
    return torch, FluxControlNetPipeline, FluxControlNetModel, load_image


def _torch_dtype(torch_module: Any, dtype: str) -> Any:
    dtype_name = DTYPES[dtype]
    if dtype == "auto":
        return None
    return getattr(torch_module, dtype_name)
