from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from aigen.character_edit import run_character_edit
from aigen.character_edit_audit import AuditContextImage, CharacterAuditGroup
from aigen.character_qwen_edit import run_qwen_character_edit
from aigen.generation.image_edit import FLUX2_KLEIN_BACKEND, resolve_image_edit_canvas_size
from aigen.generation.image_edit_batch import ImageEditBatchCase, ImageEditBatchLora, ImageEditBatchRequest
from aigen.generation.qwen_image_edit_identity import (
    DEFAULT_QWEN_IDENTITY_MAX_SIDE, DEFAULT_QWEN_IDENTITY_MAX_SEQUENCE_LENGTH,
    DEFAULT_QWEN_IDENTITY_PROFILE, qwen_image_edit_identity_profile_for_name,
)
from aigen.manifest_io import atomic_write_json
from aigen.progress import StatusReporter
from aigen.workflow_artifacts import ImageArtifact, LoraArtifact, WorkflowArtifact, one_artifact
from aigen.workflow_cache import GeneratedNodeOutput
from aigen.workflow_compilation import CompiledCharacterEditConfig
from aigen.workflow_graph import ArtifactType


def execute_character_node(
    config: CompiledCharacterEditConfig,
    references: tuple[Path, ...],
    loras: tuple[LoraArtifact, ...],
    inputs: Mapping[str, Sequence[WorkflowArtifact]],
    directory: Path,
    *,
    progress: StatusReporter,
) -> dict[str, GeneratedNodeOutput]:
    pose = Path(one_artifact(inputs, "pose", ImageArtifact).path) if "pose" in inputs else None
    scene = Path(one_artifact(inputs, "scene", ImageArtifact).path) if "scene" in inputs else None
    settings = config.settings
    adapters = tuple(ImageEditBatchLora(path=Path(lora.path), weight=lora.weight) for lora in loras)
    if config.backend == FLUX2_KLEIN_BACKEND:
        width, height = resolve_image_edit_canvas_size(
            backend=config.backend, first_reference=references[0], settings=settings,
        )
        ids = tuple(f"candidate-{index}" for index in range(config.candidates))
        context = (AuditContextImage("pose source", pose),) if pose is not None else ()
        cases = tuple(ImageEditBatchCase(
            id=candidate_id, prompt=config.prompt,
            image_paths=references + ((pose,) if pose is not None else ()),
            width=width, height=height, seed=config.seed + index,
            output_path=directory / f"{candidate_id}.png",
        ) for index, candidate_id in enumerate(ids))
        result = run_character_edit(
            ImageEditBatchRequest(
                backend=config.backend, cases=cases, loras=adapters, steps=settings.steps,
                guidance=settings.guidance, strength=settings.strength,
                sampler=settings.sampler, scheduler=settings.scheduler,
            ),
            audit_groups=(CharacterAuditGroup(
                id="edit", instruction=config.prompt,
                route="pose_transfer" if pose is not None else "unknown_reference_edit",
                references=references, context_images=context, candidate_ids=ids,
            ),),
            output_dir=directory / "character", progress=progress, audit_config=config.audit,
            max_iterations=config.max_iterations, upscale_long_side=config.upscale_long_side,
        ).to_json()
    else:
        result = run_qwen_character_edit(
            pack_path=None, source_image_paths=references, instruction=config.prompt,
            output_dir=directory / "character",
            profile=qwen_image_edit_identity_profile_for_name(DEFAULT_QWEN_IDENTITY_PROFILE),
            max_side=DEFAULT_QWEN_IDENTITY_MAX_SIDE,
            steps=settings.steps, true_cfg_scale=settings.guidance, guidance_scale=config.guidance_scale,
            seed=config.seed, candidates_per_case=config.candidates,
            max_sequence_length=DEFAULT_QWEN_IDENTITY_MAX_SEQUENCE_LENGTH if config.max_sequence_length is None else config.max_sequence_length,
            aspect_ratio=settings.aspect_ratio,
            canvas_size=(settings.width, settings.height) if settings.width is not None else None,
            upscale_long_side=2048 if config.upscale_long_side is None else config.upscale_long_side,
            postprocess="none" if config.upscale_long_side is None else "vosr",
            overwrite=False, nunchaku_blocks_on_gpu=None,
            pose_source_path=pose, pose_mode=config.pose_mode,
            structure_source_path=scene, structure_control=config.structure_control if scene is not None else None,
            max_iterations=config.max_iterations, audit_config=config.audit, loras=adapters,
            sampler=settings.sampler, scheduler=settings.scheduler, progress=progress,
        )
    atomic_write_json(directory / "backend-result.json", result)
    return {"image": GeneratedNodeOutput(
        artifact_type=ArtifactType.IMAGE, paths=(Path(result["outputs"][0]["image"]["path"]),),
    )}
