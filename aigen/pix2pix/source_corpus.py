from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from aigen.manifest_io import sha256_file
from aigen.pix2pix.corpus_io import corpus_member, require_exact_keys
from aigen.pix2pix.errors import Pix2PixError


SOURCE_OUTPUT_KEYS = {
    "id",
    "path",
    "sha256",
    "size_bytes",
    "mode",
    "width",
    "height",
    "seed",
}


def expected_source_shards(
    selected: tuple[dict[str, Any], ...],
    shard_size: int,
) -> tuple[tuple[int, tuple[dict[str, Any], ...]], ...]:
    return tuple(
        (
            start // shard_size,
            selected[start : start + shard_size],
        )
        for start in range(0, len(selected), shard_size)
    )


def validate_source_output(
    shard_dir: Path,
    output: object,
    *,
    pair_id: str,
    expected_seed: int,
    expected_size: tuple[int, int],
    label: str,
) -> Path:
    if not isinstance(output, dict):
        raise Pix2PixError(f"invalid {label} output for {pair_id}")
    require_exact_keys(output, SOURCE_OUTPUT_KEYS, f"{label} output for {pair_id}")
    if output["id"] != pair_id:
        raise Pix2PixError(f"{label} output id mismatch: {pair_id}")
    if output["seed"] != expected_seed:
        raise Pix2PixError(f"{label} source seed mismatch: {pair_id}")
    output_path = corpus_member(
        shard_dir,
        str(output["path"]),
        label=f"{label} source for {pair_id}",
    )
    if not output_path.is_file():
        raise Pix2PixError(f"missing {label} source: {output_path}")
    if output_path.stat().st_size != output["size_bytes"]:
        raise Pix2PixError(f"{label} source size mismatch: {pair_id}")
    if sha256_file(output_path) != output["sha256"]:
        raise Pix2PixError(f"{label} source checksum mismatch: {pair_id}")
    raster = inspect_source_image(
        output_path,
        expected_size=expected_size,
        label=label,
    )
    if raster != (output["mode"], output["width"], output["height"]):
        raise Pix2PixError(f"{label} source raster mismatch: {pair_id}")
    return output_path


def inspect_source_image(
    path: Path,
    *,
    expected_size: tuple[int, int],
    label: str,
) -> tuple[str, int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode not in {"RGB", "RGBA"}:
                raise Pix2PixError(
                    f"{label} source must be RGB or RGBA: {path.as_posix()}"
                )
            if image.size != expected_size:
                raise Pix2PixError(
                    f"{label} source must be "
                    f"{expected_size[0]}x{expected_size[1]}: {path.as_posix()}"
                )
            return image.mode, image.width, image.height
    except OSError as error:
        raise Pix2PixError(
            f"cannot decode {label} source {path}: {error}"
        ) from error
