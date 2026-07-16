from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as functional
from comfy_kitchen import quantize_per_tensor_fp8
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout
from flux2.model import Flux2, Klein9BParams
from safetensors import safe_open
from safetensors.torch import load_file


class ScaledFP8Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(
                (out_features, in_features),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(out_features, dtype=torch.bfloat16, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)
        self.register_buffer("weight_scale", torch.empty((), dtype=torch.float32, device=device))
        self.register_buffer("input_scale", torch.empty((), dtype=torch.float32, device=device))
        self.register_buffer("lokr_w1", None, persistent=False)
        self.register_buffer("lokr_w2", None, persistent=False)
        self.lokr_weight = 1.0
        self._lora_adapters: list[tuple[str, str, slice | None, float]] = []

    def set_lokr(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        weight: float,
    ) -> None:
        if (w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1]) != (
            self.out_features,
            self.in_features,
        ):
            raise ValueError(
                f"LoKr factors {tuple(w1.shape)} x {tuple(w2.shape)} do not match "
                f"linear weight {(self.out_features, self.in_features)}"
            )
        self.lokr_w1 = w1.to(torch.bfloat16).contiguous()
        self.lokr_w2 = w2.to(torch.bfloat16).contiguous()
        self.lokr_weight = weight

    def add_lora(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        output_slice: slice | None,
        scale: float,
    ) -> None:
        output_width = self.out_features if output_slice is None else output_slice.stop - output_slice.start
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != self.in_features:
            raise ValueError(
                f"LoRA factors {tuple(a.shape)} x {tuple(b.shape)} do not match "
                f"linear input width {self.in_features}"
            )
        if b.shape != (output_width, a.shape[0]):
            raise ValueError(
                f"LoRA factors {tuple(a.shape)} x {tuple(b.shape)} do not match "
                f"linear output width {output_width}"
            )
        index = len(self._lora_adapters)
        a_name = f"lora_a_{index}"
        b_name = f"lora_b_{index}"
        self.register_buffer(a_name, a.to(torch.bfloat16).contiguous(), persistent=False)
        self.register_buffer(b_name, b.to(torch.bfloat16).contiguous(), persistent=False)
        self._lora_adapters.append((a_name, b_name, output_slice, scale))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original_shape = value.shape
        value_2d = value.reshape(-1, original_shape[-1])
        quantized_value = quantize_per_tensor_fp8(
            value_2d,
            self.input_scale,
            torch.float8_e4m3fn,
        )
        value_params = TensorCoreFP8Layout.Params(
            scale=self.input_scale,
            orig_dtype=value.dtype,
            orig_shape=tuple(value_2d.shape),
        )
        weight_params = TensorCoreFP8Layout.Params(
            scale=self.weight_scale,
            orig_dtype=value.dtype,
            orig_shape=tuple(self.weight.shape),
        )
        output = functional.linear(
            QuantizedTensor(quantized_value, "TensorCoreFP8Layout", value_params),
            QuantizedTensor(self.weight, "TensorCoreFP8Layout", weight_params),
            self.bias,
        ).reshape(*original_shape[:-1], self.out_features)
        if self.lokr_w1 is not None:
            grouped = value.reshape(
                *original_shape[:-1],
                self.lokr_w1.shape[1],
                self.lokr_w2.shape[1],
            )
            w2_output = functional.linear(grouped, self.lokr_w2)
            lokr_output = functional.linear(
                w2_output.transpose(-1, -2),
                self.lokr_w1,
            ).transpose(-1, -2).reshape(
                *original_shape[:-1],
                self.out_features,
            )
            lokr_output.mul_(self.lokr_weight)
            output.add_(lokr_output)
        for a_name, b_name, output_slice, scale in self._lora_adapters:
            a = getattr(self, a_name)
            b = getattr(self, b_name)
            lora_output = functional.linear(functional.linear(value, a), b)
            lora_output.mul_(scale)
            if output_slice is None:
                output.add_(lora_output)
            else:
                output[..., output_slice].add_(lora_output)
        return output


class Flux2KleinTransformerAdapter(torch.nn.Module):
    def __init__(self, model: Flux2) -> None:
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(in_channels=model.in_channels)

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def cache_context(self, _name: str):
        return nullcontext()

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        guidance: torch.Tensor | None,
        return_dict: bool,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor]:
        output = self.model(
            x=hidden_states,
            x_ids=img_ids,
            timesteps=timestep,
            ctx=encoder_hidden_states,
            ctx_ids=txt_ids,
            guidance=guidance,
        )
        if return_dict:
            raise ValueError("Flux2KleinTransformerAdapter returns tuples")
        return (output,)


def load_flux2_klein_scaled_fp8(
    checkpoint: Path,
    *,
    lora: Path | None = None,
    lora_weight: float = 1.0,
) -> Flux2KleinTransformerAdapter:
    with torch.device("meta"):
        model = Flux2(Klein9BParams()).to(torch.bfloat16)

    state_dict = load_file(checkpoint, device="cpu")
    quantized_modules = {
        key.removesuffix(".weight_scale")
        for key in state_dict
        if key.endswith(".weight_scale")
    }
    for name in quantized_modules:
        source = model.get_submodule(name)
        replacement = ScaledFP8Linear(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device=torch.device("meta"),
        )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, replacement)

    model.load_state_dict(state_dict, strict=True, assign=True)
    if lora is not None:
        lora_keys = _safetensor_keys(lora)
        if any(key.endswith(".lora_A.weight") for key in lora_keys):
            _load_peft_lora(model, lora, lora_weight)
        else:
            _load_lokr(model, lora, lora_weight)
    model.requires_grad_(False)
    model.eval()
    return Flux2KleinTransformerAdapter(model)


def _safetensor_keys(path: Path) -> list[str]:
    with safe_open(path, framework="pt", device="cpu") as weights:
        return list(weights.keys())


def _load_peft_lora(model: Flux2, path: Path, weight: float) -> None:
    state_dict = load_file(path, device="cpu")
    with safe_open(path, framework="pt", device="cpu") as weights:
        metadata = weights.metadata() or {}
    adapter_metadata = metadata.get("lora_adapter_metadata")
    if adapter_metadata is None:
        raise ValueError(f"Missing PEFT adapter metadata in {path}")
    adapter_config = json.loads(adapter_metadata)
    rank = int(adapter_config["transformer.r"])
    alpha = float(adapter_config["transformer.lora_alpha"])
    scale = weight * alpha / rank

    a_suffix = ".lora_A.weight"
    targets = sorted(key.removesuffix(a_suffix) for key in state_dict if key.endswith(a_suffix))
    expected_keys = {
        f"{target}.{suffix}"
        for target in targets
        for suffix in ("lora_A.weight", "lora_B.weight")
    }
    if not targets or set(state_dict) != expected_keys:
        raise ValueError(f"Unsupported PEFT LoRA weights in {path}")

    for target in targets:
        module_name, output_slice = _peft_target(target.removeprefix("transformer."), model.hidden_size)
        module = model.get_submodule(module_name)
        if not isinstance(module, ScaledFP8Linear):
            raise ValueError(f"PEFT LoRA target is not a scaled-FP8 linear: {module_name}")
        module.add_lora(
            state_dict[f"{target}.lora_A.weight"],
            state_dict[f"{target}.lora_B.weight"],
            output_slice,
            scale,
        )


def _peft_target(target: str, hidden_size: int) -> tuple[str, slice | None]:
    parts = target.split(".")
    if len(parts) == 4 and parts[0] == "single_transformer_blocks" and parts[2:] == ["attn", "to_out"]:
        return f"single_blocks.{parts[1]}.linear2", None
    if (
        len(parts) == 4
        and parts[0] == "single_transformer_blocks"
        and parts[2:] == ["attn", "to_qkv_mlp_proj"]
    ):
        return f"single_blocks.{parts[1]}.linear1", None
    if len(parts) == 5 and parts[0] == "transformer_blocks" and parts[2:4] == ["attn", "to_out"]:
        if parts[4] == "0":
            return f"double_blocks.{parts[1]}.img_attn.proj", None
    if len(parts) == 4 and parts[0] == "transformer_blocks" and parts[2] == "attn":
        qkv_index = {"to_q": 0, "to_k": 1, "to_v": 2}.get(parts[3])
        if qkv_index is not None:
            start = qkv_index * hidden_size
            return f"double_blocks.{parts[1]}.img_attn.qkv", slice(start, start + hidden_size)
    raise ValueError(f"Unsupported PEFT LoRA target: {target}")


def _load_lokr(model: Flux2, path: Path, weight: float) -> None:
    state_dict = load_file(path, device="cpu")
    targets = sorted(
        key.removesuffix(".lokr_w1")
        for key in state_dict
        if key.endswith(".lokr_w1")
    )
    expected_keys = {
        f"{target}.{suffix}"
        for target in targets
        for suffix in ("alpha", "lokr_w1", "lokr_w2")
    }
    if not targets or set(state_dict) != expected_keys:
        raise ValueError(f"Unsupported LoKr weights in {path}")

    for target in targets:
        module_name = target.removeprefix("diffusion_model.")
        module = model.get_submodule(module_name)
        if not isinstance(module, ScaledFP8Linear):
            raise ValueError(f"LoKr target is not a scaled-FP8 linear: {module_name}")
        module.set_lokr(
            state_dict[f"{target}.lokr_w1"],
            state_dict[f"{target}.lokr_w2"],
            weight,
        )
