from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from aigen.keyframe_polish_models import KeyframePolishError, KeyframePolishJobSpec
from aigen.manifest_io import read_json, resolve_existing_path, resolve_output_path


@dataclass(frozen=True)
class PolishContext:
    base_dir: Path
    result: dict[str, Any]
    candidate: str
    base_path: Path
    base_image: Image.Image
    identity_primer_path: Path
    identity_primer: Image.Image


def load_polish_context(spec: KeyframePolishJobSpec, job_path: Path) -> PolishContext:
    base_dir = resolve_output_path(spec.base.run_dir, job_path.parent)
    result = read_json(base_dir / "result.json", label="keyframe polish base result")
    base_path = Path(base_output(result, spec.base.candidate)["path"]).resolve()
    identity_primer_path = resolve_existing_path(spec.character.identity_primer.path, job_path.parent)
    with Image.open(base_path) as base_image:
        base_rgb = base_image.convert("RGB")
    with Image.open(identity_primer_path) as identity_primer:
        identity_rgb = identity_primer.convert("RGB")
    return PolishContext(
        base_dir=base_dir,
        result=result,
        candidate=spec.base.candidate,
        base_path=base_path,
        base_image=base_rgb,
        identity_primer_path=identity_primer_path,
        identity_primer=identity_rgb,
    )


def base_output(result: dict[str, Any], candidate: str) -> dict[str, Any]:
    for output in result["outputs"]:
        if output["name"] == candidate:
            return output
    raise KeyframePolishError(f"Base run has no candidate named {candidate}")

