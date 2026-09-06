from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from aigen.image_io import open_image
from aigen.manifest_io import atomic_write_json, sha256_file
from aigen.progress import StatusReporter
from aigen.workflow_artifacts import ImageArtifact, MaskArtifact, mask_identity, one_artifact
from aigen.workflow_cache import GeneratedNodeOutput
from aigen.workflow_graph import ArtifactType, BindMaskConfig, SamSegmentConfig

if TYPE_CHECKING:
    from aigen.workflow_compilation import CompiledCharacterRefineConfig


def bind_mask(source: ImageArtifact, mask_path: Path, binding: BindMaskConfig | None = None) -> MaskArtifact:
    with open_image(Path(source.path)) as image, open_image(mask_path) as mask:
        if image.size != mask.size:
            raise ValueError("mask dimensions must match the source image in display orientation")
        size = image.size
    if source.content_sha256 is None:
        raise ValueError("mask source has no recorded pixel-file checksum")
    checksum = sha256_file(mask_path)
    if binding is not None:
        if binding.source_sha256 is not None and binding.source_sha256 != source.content_sha256:
            raise ValueError("region-plan mask belongs to a different source image")
        if binding.mask_sha256 is not None and binding.mask_sha256 != checksum:
            raise ValueError("region-plan mask contents changed since the selection was recorded")
    return MaskArtifact(path=str(mask_path), content_sha256=checksum,
                        source_sha256=source.content_sha256, source_size=size,
                        identity=mask_identity(checksum, source.content_sha256, size))


def sam_segmentation_arguments(config: SamSegmentConfig) -> dict[str, object]:
    from aigen.sam_commands import _parse_box, _parse_points, _validate_prompt

    try:
        box = _parse_box(config.box) if config.box.strip() else None
        positive = _parse_points(config.positive_points) if config.positive_points.strip() else ()
        negative = _parse_points(config.negative_points) if config.negative_points.strip() else ()
    except argparse.ArgumentTypeError as error:
        raise ValueError(str(error)) from error
    _validate_prompt(config.prompt_mode, box, positive)
    if config.engine == "anime" and (config.device != "cuda" or config.prompt_mode != "auto"):
        raise ValueError("anime segmentation requires CUDA and automatic foreground selection")
    return {**config.model_dump(), "box": box, "positive_points": positive, "negative_points": negative}


def execute_sam_node(config: SamSegmentConfig, inputs, directory: Path, *, progress: StatusReporter):
    from aigen.sam_commands import segment_image

    source = one_artifact(inputs, "source", ImageArtifact)
    result = segment_image(Path(source.path), **sam_segmentation_arguments(config),
                           output_mode="all", output_dir=directory / "sam", overwrite=False, progress=progress)
    outputs = result["outputs"]
    mask = bind_mask(source, Path(outputs["mask"]))
    atomic_write_json(directory / "backend-result.json", {**result, "source_binding": mask.model_dump(mode="json")})
    return {
        "mask": GeneratedNodeOutput(ArtifactType.MASK, (Path(mask.path),),
                                    mask_source_sha256=mask.source_sha256, mask_source_size=mask.source_size),
        **{port: GeneratedNodeOutput(ArtifactType.IMAGE, (Path(outputs[port]),)) for port in ("cutout", "preview")},
    }


def execute_character_refine_node(config: CompiledCharacterRefineConfig, inputs, references, loras,
                                  directory: Path, *, progress: StatusReporter):
    from aigen.character_qwen_refine import run_character_masked_edit
    from aigen.generation.image_edit_batch import ImageEditBatchLora

    source = one_artifact(inputs, "source", ImageArtifact)
    mask = one_artifact(inputs, "mask", MaskArtifact)
    with open_image(Path(source.path)) as image:
        source_size = image.size
    if mask.source_sha256 != source.content_sha256 or mask.source_size != source_size:
        raise ValueError("edit mask belongs to a different source image; bind or segment this source first")
    settings = config.settings
    result = run_character_masked_edit(
        source_image=Path(source.path), mask_image=Path(mask.path), references=references,
        instruction=settings.prompt, output_dir=directory / "character", progress=progress,
        strength=settings.strength, candidates=settings.candidates, seed=settings.seed,
        max_side=settings.max_side, steps=settings.steps, true_cfg_scale=settings.guidance,
        guidance_scale=settings.guidance_scale, max_sequence_length=settings.max_sequence_length,
        max_iterations=settings.max_iterations, audit_config=config.audit,
        sampler=settings.sampler, scheduler=settings.scheduler,
        loras=tuple(ImageEditBatchLora(path=Path(lora.path), weight=lora.weight) for lora in loras),
    )
    atomic_write_json(directory / "backend-result.json", result)
    return {"image": GeneratedNodeOutput(ArtifactType.IMAGE, (Path(result["outputs"][0]["image"]["path"]),))}
