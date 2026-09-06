from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import file_digest
from pathlib import Path

from PIL import Image, ImageOps


@contextmanager
def open_image(path: Path) -> Iterator[Image.Image]:
    """Decode pixels in display orientation, retaining color mode and metadata."""
    with Image.open(path) as image:
        ImageOps.exif_transpose(image, in_place=True)
        yield image


def oriented_image_size(image: Image.Image) -> tuple[int, int]:
    """Read display dimensions without decoding image pixels."""
    if image.getexif().get(274) in (5, 6, 7, 8):
        return image.height, image.width
    return image.size


def image_alpha(image: Image.Image) -> Image.Image | None:
    """Extract explicit alpha or PNG color-key/palette transparency."""
    if "A" in image.getbands():
        return image.getchannel("A")
    if "transparency" in image.info:
        with image.convert("RGBA") as rgba:
            return rgba.getchannel("A")
    return None


def load_thumbnail(path: Path, size: tuple[int, int], *, expected_sha256: str | None = None) -> Image.Image:
    """Decode a display-only preview; never used for model conditioning."""
    with path.open("rb") as stream:
        if expected_sha256 is not None:
            if file_digest(stream, "sha256").hexdigest() != expected_sha256:
                raise ValueError(f"Image contents changed since this result was recorded: {path}")
            stream.seek(0)
        with Image.open(stream) as source:
            source.draft("RGB", size)
            ImageOps.exif_transpose(source, in_place=True)
            source.thumbnail(size, Image.Resampling.LANCZOS)
            return source.convert("RGB")


def fit_image_canvas(image: Image.Image, size: tuple[int, int], mode: str) -> Image.Image:
    """Fit visual conditioning on a resolved canvas; pad uses a white background."""
    if image.size == size:
        return image
    if mode == "crop":
        return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    if mode == "pad":
        return ImageOps.pad(image, size, method=Image.Resampling.LANCZOS, color="white")
    if mode == "stretch":
        return image.resize(size)
    raise ValueError(f"unknown image canvas fit: {mode}")
