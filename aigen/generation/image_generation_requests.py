from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageGenerationOutputRequest:
    name: str
    seed: int
    path: Path


@dataclass(frozen=True)
class ImageGenerationCaseRequest:
    name: str
    prompt: str
    image_paths: tuple[Path, ...]
    width: int | None
    height: int | None
    outputs: tuple[ImageGenerationOutputRequest, ...]
