from __future__ import annotations

from typing import Any


def euler_ancestral_step(
    sample: Any,
    model_output: Any,
    *,
    sigma: Any,
    sigma_next: Any,
    final: bool,
    generator: Any,
    torch: Any,
) -> Any:
    dtype = model_output.dtype
    sample = sample.float()
    model_output = model_output.float()
    sigma = sigma.float()
    sigma_next = sigma_next.float()
    denoised = sample - sigma * model_output
    if final:
        return denoised.to(dtype)

    sigma_down = sigma_next * (sigma_next / sigma)
    alpha_next = 1 - sigma_next
    alpha_down = 1 - sigma_down
    renoise = torch.sqrt(
        torch.clamp(
            sigma_next.square()
            - sigma_down.square() * alpha_next.square() / alpha_down.square(),
            min=0,
        )
    )
    down_ratio = sigma_down / sigma
    sample = down_ratio * sample + (1 - down_ratio) * denoised
    noise = torch.randn(
        sample.shape,
        generator=generator,
        device=sample.device,
        dtype=sample.dtype,
    )
    return (alpha_next / alpha_down * sample + noise * renoise).to(dtype)
