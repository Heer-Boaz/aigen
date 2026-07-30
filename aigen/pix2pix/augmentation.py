from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


TRANSLATION_RATIO = 0.125


def maximum_translation(image_size: int) -> int:
    return int(image_size * TRANSLATION_RATIO + 0.5)


class PairedIntegerTranslation:
    """Shared, interpolation-free translation for conditional GAN inputs."""

    def __init__(
        self,
        image_size: int,
        *,
        device: torch.device,
    ) -> None:
        self._maximum_shift = maximum_translation(image_size)
        self._rows = torch.arange(
            image_size,
            dtype=torch.long,
            device=device,
        ).view(1, image_size, 1)
        self._columns = torch.arange(
            image_size,
            dtype=torch.long,
            device=device,
        ).view(1, 1, image_size)

    def __call__(
        self,
        source: Tensor,
        *paired: Tensor,
    ) -> tuple[Tensor, ...]:
        batch_size = source.shape[0]
        shift_rows = torch.randint(
            -self._maximum_shift,
            self._maximum_shift + 1,
            (batch_size, 1, 1),
            device=source.device,
        )
        shift_columns = torch.randint(
            -self._maximum_shift,
            self._maximum_shift + 1,
            (batch_size, 1, 1),
            device=source.device,
        )
        rows = (self._rows + shift_rows + 1).clamp_(0, source.shape[-2] + 1)
        columns = (
            self._columns + shift_columns + 1
        ).clamp_(0, source.shape[-1] + 1)
        batch = torch.arange(
            batch_size,
            dtype=torch.long,
            device=source.device,
        ).view(batch_size, 1, 1)
        channels = source.shape[1]
        combined = torch.cat((source, *paired), dim=1)
        padded = functional.pad(combined, (1, 1, 1, 1), value=1.0)
        translated = (
            padded.permute(0, 2, 3, 1)[batch, rows, columns]
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return translated.split(channels, dim=1)
