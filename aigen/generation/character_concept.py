from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_NEGATIVE_PROMPT = (
    "cropped head, cropped feet, duplicate character, extra arms, extra legs, "
    "extra fingers, malformed hands, broken anatomy, blurry, low detail, "
    "watermark, logo, text"
)

FRAMING_PROMPTS = {
    "full-body": (
        "Full-body vertical character concept, complete figure and feet visible, "
        "simple background."
    ),
    "portrait": (
        "Polished character portrait, centered bust framing, clear face identity, "
        "readable hair shape, eyes, costume collar and upper-body materials."
    ),
}

DTYPES = {
    "auto": None,
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}


@dataclass(frozen=True)
class CharacterConceptResult:
    output_path: str
    model: str
    reference_image: str
    prompt: str
    pipeline_prompt: str
    negative_prompt: str
    width: int
    height: int
    framing: str
    steps: int
    guidance_scale: float
    true_cfg_scale: float
    dtype: str
    device: str
    cpu_offload: bool
    transformer_single_file: str | None
    seed: int

    def to_json(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "model": self.model,
            "reference_image": self.reference_image,
            "prompt": self.prompt,
            "pipeline_prompt": self.pipeline_prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "framing": self.framing,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "true_cfg_scale": self.true_cfg_scale,
            "dtype": self.dtype,
            "device": self.device,
            "cpu_offload": self.cpu_offload,
            "transformer_single_file": self.transformer_single_file,
            "seed": self.seed,
        }


class CharacterConceptError(RuntimeError):
    pass


class CharacterConceptDependencyError(CharacterConceptError):
    pass


def run_character_concept(
    model: str,
    reference_image: Path,
    output_path: Path,
    prompt: str,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    steps: int = 32,
    guidance_scale: float = 2.5,
    true_cfg_scale: float = 1.0,
    width: int = 1024,
    height: int = 1536,
    framing: str = "full-body",
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 1,
    cpu_offload: bool = True,
    transformer_single_file: Path | None = None,
) -> CharacterConceptResult:
    torch, pipeline_class, transformer_class, load_image = _load_flux_kontext()
    transformer = None
    if transformer_single_file:
        transformer = transformer_class.from_single_file(
            transformer_single_file.resolve().as_posix(),
            config=model,
            subfolder="transformer",
            torch_dtype=_torch_dtype(torch, dtype),
            local_files_only=True,
        )
    pipeline_kwargs: dict[str, Any] = {
        "torch_dtype": _torch_dtype(torch, dtype),
        "local_files_only": True,
    }
    if transformer:
        pipeline_kwargs["transformer"] = transformer
    pipeline = pipeline_class.from_pretrained(model, **pipeline_kwargs)
    pipeline.vae.enable_tiling()
    pipeline.vae.enable_slicing()
    if cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)

    pipeline_prompt = compose_character_prompt(prompt, framing)
    args: dict[str, Any] = {
        "image": load_image(reference_image.resolve().as_posix()),
        "prompt": pipeline_prompt,
        "negative_prompt": negative_prompt,
        "true_cfg_scale": true_cfg_scale,
        "width": width,
        "height": height,
        "max_area": width * height,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": torch.Generator(device=device).manual_seed(seed),
    }

    image = pipeline(**args).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return CharacterConceptResult(
        output_path=output_path.resolve().as_posix(),
        model=model,
        reference_image=reference_image.resolve().as_posix(),
        prompt=prompt,
        pipeline_prompt=pipeline_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        framing=framing,
        steps=steps,
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
        dtype=dtype,
        device=device,
        cpu_offload=cpu_offload,
        transformer_single_file=transformer_single_file.resolve().as_posix() if transformer_single_file else None,
        seed=seed,
    )


def _load_flux_kontext() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from diffusers import FluxKontextPipeline, FluxTransformer2DModel
        from diffusers.utils import load_image
    except ImportError as exc:
        raise CharacterConceptDependencyError(
            "character concept generation requires `pip install -e .[generation]`"
        ) from exc
    return torch, FluxKontextPipeline, FluxTransformer2DModel, load_image


def compose_character_prompt(prompt: str, framing: str) -> str:
    return f"{FRAMING_PROMPTS[framing]}\n\n{prompt}"


def _torch_dtype(torch_module: Any, dtype: str) -> Any:
    dtype_name = DTYPES[dtype]
    if dtype == "auto":
        return None
    return getattr(torch_module, dtype_name)
