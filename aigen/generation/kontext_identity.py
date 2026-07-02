from __future__ import annotations

from pathlib import Path
from typing import Any

from aigen.generation.flux_components import FLUX_TEXT_COMPONENTS_DISABLED, flux_text_component_device_reports
from aigen.generation.flux_prompt_encoding import FluxPromptEmbedding
from aigen.generation.runtime_diagnostics import elapsed_ms, module_device_report, synchronized_time
from aigen.generation.runtime_types import resolve_torch_dtype


class CharacterKontextIdentityError(RuntimeError):
    pass


class CharacterKontextIdentityDependencyError(CharacterKontextIdentityError):
    pass


class CharacterKontextIdentitySession:
    def __init__(
        self,
        model: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        nunchaku_transformer_model: Path,
        attention_impl: str,
        vae_tiling: bool,
    ) -> None:
        torch, pipeline_class, load_image = _load_flux_kontext_identity()
        self.torch = torch
        self.load_image = load_image
        self.device = device
        model_load_start = synchronized_time(torch)
        self.pipeline = _build_kontext_identity_pipeline(
            pipeline_class,
            model,
            nunchaku_transformer_model,
            attention_impl,
            _torch_dtype(torch, dtype),
            device,
            vae_tiling,
        )
        self.model_load_ms = elapsed_ms(model_load_start, synchronized_time(torch))

    def generate(
        self,
        *,
        reference_image: Path,
        prompt_embedding: FluxPromptEmbedding,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        max_sequence_length: int,
    ) -> tuple[Any, dict[str, Any]]:
        start = synchronized_time(self.torch)
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        execution_device = self.pipeline._execution_device
        output = self.pipeline(
            image=self.load_image(reference_image.resolve().as_posix()),
            prompt_embeds=prompt_embedding.prompt_embeds.to(device=execution_device),
            pooled_prompt_embeds=prompt_embedding.pooled_prompt_embeds.to(device=execution_device),
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            max_sequence_length=max_sequence_length,
            max_area=width * height,
            output_type="pil",
        )
        return output.images[0], {"pipeline_ms": elapsed_ms(start, synchronized_time(self.torch))}

    def environment(self) -> dict[str, Any]:
        return {
            "device_report": _pipeline_device_report(self.pipeline),
            "prompt_encoding": "precomputed_prompt_embeds",
        }

    def close(self) -> None:
        del self.pipeline
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _build_kontext_identity_pipeline(
    pipeline_class: Any,
    model: str,
    nunchaku_transformer_model: Path,
    attention_impl: str,
    torch_dtype: Any,
    device: str,
    vae_tiling: bool,
) -> Any:
    from nunchaku import NunchakuFluxTransformer2dModel

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        nunchaku_transformer_model.resolve().as_posix(),
        torch_dtype=torch_dtype,
        offload=False,
    )
    transformer.set_attention_impl(attention_impl)
    pipeline = pipeline_class.from_pretrained(
        model,
        transformer=transformer,
        **FLUX_TEXT_COMPONENTS_DISABLED,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    pipeline.set_progress_bar_config(disable=True)
    if vae_tiling:
        pipeline.vae.enable_tiling()
    else:
        pipeline.vae.disable_tiling()
    pipeline.vae.disable_slicing()
    pipeline.to(device)
    return pipeline


def _load_flux_kontext_identity() -> tuple[Any, Any, Any]:
    try:
        import torch
        from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
        from diffusers.utils import logging as diffusers_logging
        from diffusers.utils import load_image
    except ImportError as exc:
        raise CharacterKontextIdentityDependencyError(
            "Kontext identity generation requires `pip install -e .[generation]`"
        ) from exc

    diffusers_logging.disable_progress_bar()
    return torch, FluxKontextPipeline, load_image


def _pipeline_device_report(pipeline: Any) -> dict[str, Any]:
    components = {
        "transformer": module_device_report(pipeline.transformer),
        "vae": module_device_report(pipeline.vae),
    }
    components.update(flux_text_component_device_reports(pipeline))
    return {
        "pipeline_class": type(pipeline).__qualname__,
        "model_cpu_offload_seq": pipeline.model_cpu_offload_seq,
        "components": components,
    }


def _torch_dtype(torch_module: Any, dtype: str) -> Any:
    return resolve_torch_dtype(torch_module, dtype, auto_value=None)
