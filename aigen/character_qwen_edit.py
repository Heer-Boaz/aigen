from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from PIL import Image

from aigen.canny_control import CannyControl, CannyControlError, render_canny_control
from aigen.character_conditioning_models import CharacterConditioningPlanError, CharacterConditioningPlanSpec
from aigen.character_conditioning_planner import CharacterConditioningPlanner
from aigen.character_reference_models import CharacterReferenceError
from aigen.character_reference_pack import load_character_reference_pack
from aigen.depth_v2_control import DepthV2Control, DepthV2ControlError, render_depth_v2_control
from aigen.dwpose_control import DWPoseControl, DWPoseControlError, render_dwpose_control
from aigen.generation.qwen_image_edit_identity import (
    QwenControlImage,
    QwenIdentityCase,
    QwenImageEditProfile,
    run_qwen_image_edit_cases,
)
from aigen.image_assets import image_asset_json
from aigen.manifest_io import resolve_existing_path, write_json
from aigen.progress import StatusReporter


class QwenCharacterEditError(RuntimeError):
    pass


QWEN_DEFAULT_POSE_SOURCE_NAME = "default"
QWEN_POSE_MODES = ("native", "keypoint")
DEFAULT_QWEN_POSE_MODE = "native"
QWEN_DEPTH_CONTROL_NAME = "depth"
QWEN_EDGE_CONTROL_NAME = "edge"
QWEN_STRUCTURE_CONTROL_NAMES = ("depth", "edge")
QWEN_STRUCTURE_MODE_BY_CONTROL = {
    "depth": "depth",
    "edge": "edge_or_sketch",
}
QWEN_CONTROL_NAME_BY_MODE = {
    "depth": QWEN_DEPTH_CONTROL_NAME,
    "edge_or_sketch": QWEN_EDGE_CONTROL_NAME,
}
QWEN_POSE_CONDITIONING_MODE = {
    "native": "pose_reference",
    "keypoint": "pose_keypoint",
}
QWEN_EDIT_REQUEST_NAME = "edit"


@dataclass(frozen=True)
class QwenCharacterPoseSource:
    name: str
    path: Path
    mode: str


@dataclass(frozen=True)
class PlannedQwenCharacterEdit:
    edit_cases: tuple[QwenIdentityCase, ...]
    reference_paths: dict[str, Path]
    source_images: dict[str, Path] = field(default_factory=dict)


def run_qwen_character_edit(
    *,
    pack_path: Path | None,
    output_dir: Path,
    profile: QwenImageEditProfile,
    instruction: str,
    source_image_paths: Sequence[Path],
    max_side: int,
    steps: int | None,
    true_cfg_scale: float | None,
    guidance_scale: float | None,
    seed: int,
    max_sequence_length: int,
    candidates_per_case: int,
    aspect_ratio: tuple[int, int] | None,
    upscale_long_side: int,
    overwrite: bool,
    nunchaku_blocks_on_gpu: int | None,
    pose_source_path: Path | None = None,
    pose_mode: str = DEFAULT_QWEN_POSE_MODE,
    structure_source_path: Path | None = None,
    structure_control: str | None = None,
    progress: StatusReporter,
) -> dict[str, Any]:
    instruction = instruction.strip()
    if not instruction:
        raise QwenCharacterEditError("qwen-edit-run requires a non-empty --instruction")
    if candidates_per_case < 1:
        raise QwenCharacterEditError("--candidates must be at least 1")
    if (pack_path is None) == (not source_image_paths):
        raise QwenCharacterEditError("qwen-edit-run requires either --pack or --image inputs")
    if (structure_source_path is None) != (structure_control is None):
        raise QwenCharacterEditError("--structure-source and --structure-control must be provided together")
    if pose_source_path is not None and structure_source_path is not None:
        raise QwenCharacterEditError("--pose-source and --structure-source cannot be combined in one edit")
    if structure_control is not None and structure_control not in QWEN_STRUCTURE_MODE_BY_CONTROL:
        allowed = ", ".join(QWEN_STRUCTURE_CONTROL_NAMES)
        raise QwenCharacterEditError(f"Unknown structure control {structure_control}; expected one of: {allowed}")
    if pose_mode not in QWEN_POSE_CONDITIONING_MODE:
        allowed = ", ".join(QWEN_POSE_MODES)
        raise QwenCharacterEditError(f"Unknown pose mode {pose_mode}; expected one of: {allowed}")
    planned = _build_qwen_character_edit_request(
        pack_path=pack_path,
        instruction=instruction,
        source_image_paths=source_image_paths,
        pose_source_present=pose_source_path is not None,
        structure_source_present=structure_source_path is not None,
        progress=progress,
    )
    pose_sources: dict[str, QwenCharacterPoseSource] = {}
    if pose_source_path is not None:
        pose_source = QwenCharacterPoseSource(
            name=QWEN_DEFAULT_POSE_SOURCE_NAME,
            path=pose_source_path,
            mode=pose_mode,
        )
        pose_sources = {case.name: pose_source for case in planned.edit_cases}
    return run_planned_qwen_character_edit(
        planned=planned,
        pose_sources=pose_sources,
        structure_source_path=structure_source_path,
        structure_control=structure_control,
        output_dir=output_dir,
        profile=profile,
        max_side=max_side,
        steps=steps,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale,
        seed=seed,
        max_sequence_length=max_sequence_length,
        candidates_per_case=candidates_per_case,
        overwrite=overwrite,
        nunchaku_blocks_on_gpu=nunchaku_blocks_on_gpu,
        aspect_ratio=aspect_ratio,
        upscale_long_side=upscale_long_side,
        result_kind="qwen-character-edit-result",
        manifest_context=None,
        progress=progress,
    )


def run_planned_qwen_character_edit(
    *,
    planned: PlannedQwenCharacterEdit,
    pose_sources: Mapping[str, QwenCharacterPoseSource],
    structure_source_path: Path | None,
    structure_control: str | None,
    output_dir: Path,
    profile: QwenImageEditProfile,
    max_side: int,
    steps: int | None,
    true_cfg_scale: float | None,
    guidance_scale: float | None,
    seed: int,
    max_sequence_length: int,
    candidates_per_case: int,
    overwrite: bool,
    nunchaku_blocks_on_gpu: int | None,
    aspect_ratio: tuple[int, int] | None,
    upscale_long_side: int,
    result_kind: str,
    manifest_context: dict[str, Any] | None,
    progress: StatusReporter,
) -> dict[str, Any]:
    available_modes_by_case = {
        case.name: tuple(
            [QWEN_POSE_CONDITIONING_MODE[pose_sources[case.name].mode]]
            if case.name in pose_sources
            else []
        )
        + (
            (QWEN_STRUCTURE_MODE_BY_CONTROL[structure_control],)
            if structure_control is not None
            else ()
        )
        for case in planned.edit_cases
    }
    conditioning_plans = _plan_route_conditioning(
        planned.edit_cases,
        available_modes_by_case=available_modes_by_case,
    )
    guides, controls, conditioning_result = _prepare_pose_inputs(
        pose_sources,
        progress=progress,
    )
    structure_conditioning = (
        _render_structure_conditioning(structure_source_path, structure_control, progress)
        if structure_source_path is not None and structure_control is not None
        else None
    )
    if structure_conditioning is not None:
        source_path, structure_control_result = structure_conditioning
        control_name = QWEN_CONTROL_NAME_BY_MODE[QWEN_STRUCTURE_MODE_BY_CONTROL[structure_control]]
        controls[control_name] = QwenControlImage(image=structure_control_result.image)
        preprocessor = structure_control_result.metadata | {
            "tool": "depth_anything_v2_map" if structure_control == "depth" else "canny_edge_map",
        }
        if isinstance(structure_control_result, DepthV2Control):
            preprocessor |= {
                "device": structure_control_result.device,
                "model": structure_control_result.model.as_posix(),
            }
        conditioning_result[control_name] = {
            "source_image": image_asset_json(source_path),
            "preprocessor": preprocessor,
        }
    edit_cases = _apply_route_conditioning(
        conditioning_plans,
        pose_sources=pose_sources,
    )
    output_dir = output_dir.resolve()
    result = run_qwen_image_edit_cases(
        source_images=planned.source_images,
        references=planned.reference_paths,
        guides=guides,
        controls=controls,
        output_dir=output_dir,
        profile=profile,
        edit_cases=edit_cases,
        max_side=max_side,
        steps=steps,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale,
        seed=seed,
        max_sequence_length=max_sequence_length,
        candidates_per_case=candidates_per_case,
        overwrite=overwrite,
        nunchaku_blocks_on_gpu=nunchaku_blocks_on_gpu,
        aspect_ratio=aspect_ratio,
        upscale_long_side=upscale_long_side,
        result_kind=result_kind,
        manifest_context=manifest_context,
        progress=progress,
    )
    for control_name in controls:
        conditioning_result[control_name]["control_image"] = result["controls"][control_name]
    if conditioning_result:
        result["conditioning"] = conditioning_result
    write_json(output_dir / "result.json", result)
    return result


def _build_qwen_character_edit_request(
    *,
    pack_path: Path | None,
    instruction: str,
    source_image_paths: Sequence[Path],
    pose_source_present: bool,
    structure_source_present: bool,
    progress: StatusReporter,
) -> PlannedQwenCharacterEdit:
    route_kind, output_mode = _direct_request_route(
        pose_source_present=pose_source_present,
        structure_source_present=structure_source_present,
    )
    if source_image_paths:
        source_names = tuple(f"image_{index}" for index in range(1, len(source_image_paths) + 1))
        source_images = {
            name: resolve_existing_path(path.as_posix(), Path.cwd())
            for name, path in zip(source_names, source_image_paths)
        }
        reference_paths: dict[str, Path] = {}
        selected_refs: tuple[str, ...] = ()
    else:
        progress.phase("load qwen character edit reference pack")
        try:
            context = load_character_reference_pack(pack_path)
        except CharacterReferenceError as error:
            raise QwenCharacterEditError(str(error)) from error
        source_names = ()
        source_images = {}
        reference_paths = context.references
        selected_refs = tuple(context.spec.references)
    edit_case = QwenIdentityCase(
        name=QWEN_EDIT_REQUEST_NAME,
        source_images=source_names,
        references=selected_refs,
        prompt=instruction,
        edit_context={
            "task": "character_edit",
            "input_source": "explicit_images" if source_names else "reference_pack",
            "refs_used": list(selected_refs),
            "prompt_source": "user_instruction",
            "route_kind": route_kind,
            "output_mode": output_mode,
        },
    )
    return PlannedQwenCharacterEdit(
        edit_cases=(edit_case,),
        reference_paths=reference_paths,
        source_images=source_images,
    )


def _direct_request_route(*, pose_source_present: bool, structure_source_present: bool) -> tuple[str, str]:
    if pose_source_present:
        return "pose_transfer", "single_image_pose"
    if structure_source_present:
        return "scene_insertion", "single_image_scene"
    return "unknown_reference_edit", "single_image_reference_edit"


def _plan_route_conditioning(
    edit_cases: Sequence[QwenIdentityCase],
    *,
    available_modes_by_case: Mapping[str, Sequence[str]],
) -> tuple[tuple[QwenIdentityCase, CharacterConditioningPlanSpec], ...]:
    planner = CharacterConditioningPlanner()

    planned_cases = []
    for case in edit_cases:
        route_kind = _case_route_kind(case)
        try:
            conditioning_plan = planner.plan(
                route_kind=route_kind,
                available_modes=available_modes_by_case[case.name],
            )
        except CharacterConditioningPlanError as error:
            if route_kind == "pose_transfer":
                raise QwenCharacterEditError(
                    f"Qwen edit case {case.name} uses pose_transfer and requires --pose-source with a source image"
                ) from error
            if route_kind == "local_repair_or_inpaint":
                raise QwenCharacterEditError(
                    f"Qwen edit case {case.name} uses local_repair_or_inpaint and requires a source image and region mask"
                ) from error
            raise QwenCharacterEditError(f"Qwen edit case {case.name}: {error}") from error
        planned_cases.append((case, conditioning_plan))
    return tuple(planned_cases)


def _apply_route_conditioning(
    planned_cases: Sequence[tuple[QwenIdentityCase, CharacterConditioningPlanSpec]],
    *,
    pose_sources: Mapping[str, QwenCharacterPoseSource],
) -> tuple[QwenIdentityCase, ...]:
    conditioned_cases = []
    for case, conditioning_plan in planned_cases:
        modes = list(conditioning_plan.conditioning_modes)
        pose_source = pose_sources.get(case.name)
        case_guides = tuple(
            _pose_guide_name(pose_source.name)
            for mode in modes
            if mode == "pose_reference" and pose_source is not None
        )
        case_controls = tuple(
            _pose_control_name(pose_source.name)
            if mode == "pose_keypoint" and pose_source is not None
            else QWEN_CONTROL_NAME_BY_MODE[mode]
            for mode in modes
            if mode == "pose_keypoint" or mode in QWEN_CONTROL_NAME_BY_MODE
        )
        if case.edit_context is None:
            raise QwenCharacterEditError(f"Character edit case {case.name} has no route context")
        conditioned_cases.append(
            replace(
                case,
                guides=case_guides,
                controls=case_controls,
                edit_context=case.edit_context
                | {
                    "conditioning_modes": modes,
                    "conditioning_tools": list(conditioning_plan.planned_tools),
                    "guides_used": list(case_guides),
                    "controls_used": list(case_controls),
                },
            )
        )
    return tuple(conditioned_cases)


def _prepare_pose_inputs(
    pose_sources: Mapping[str, QwenCharacterPoseSource],
    *,
    progress: StatusReporter,
) -> tuple[dict[str, Path], dict[str, QwenControlImage], dict[str, Any]]:
    guides: dict[str, Path] = {}
    controls: dict[str, QwenControlImage] = {}
    conditioning_result: dict[str, Any] = {}
    prepared_sources: set[tuple[str, str]] = set()
    for pose_source in pose_sources.values():
        source_key = (pose_source.name, pose_source.mode)
        if source_key in prepared_sources:
            continue
        prepared_sources.add(source_key)
        if pose_source.mode == "native":
            source_path = resolve_existing_path(pose_source.path.as_posix(), Path.cwd())
            guide_name = _pose_guide_name(pose_source.name)
            guides[guide_name] = source_path
            conditioning_result[guide_name] = {
                "source_image": image_asset_json(source_path),
                "preprocessor": {
                    "tool": "qwen_native_pose_reference",
                },
            }
            continue
        source_path, pose_control = _render_pose_conditioning(pose_source.path, progress)
        control_name = _pose_control_name(pose_source.name)
        controls[control_name] = QwenControlImage(
            image=pose_control.image,
            content_box=pose_control.content_box,
        )
        conditioning_result[control_name] = {
            "source_image": image_asset_json(source_path),
            "preprocessor": pose_control.metadata
            | {
                "tool": "dwpose_keypoint_map",
                "device": pose_control.device,
                "det_model": pose_control.det_model.as_posix(),
                "pose_model": pose_control.pose_model.as_posix(),
            },
        }
    return guides, controls, conditioning_result


def _pose_guide_name(source_name: str) -> str:
    return f"pose_reference_{source_name}"


def _pose_control_name(source_name: str) -> str:
    return f"pose_keypoint_{source_name}"


def _render_pose_conditioning(
    pose_source_path: Path,
    progress: StatusReporter,
) -> tuple[Path, DWPoseControl]:
    source_path = resolve_existing_path(pose_source_path.as_posix(), Path.cwd())
    progress.phase("build DWPose keypoint control")
    try:
        with Image.open(source_path) as source_image:
            control = render_dwpose_control(
                source_image,
                source_label=source_path.as_posix(),
            )
    except (OSError, DWPoseControlError) as error:
        raise QwenCharacterEditError(str(error)) from error
    return source_path, control


def _render_structure_conditioning(
    structure_source_path: Path,
    structure_control: str,
    progress: StatusReporter,
) -> tuple[Path, DepthV2Control | CannyControl]:
    source_path = resolve_existing_path(structure_source_path.as_posix(), Path.cwd())
    progress.phase(f"build {structure_control} scene control")
    try:
        with Image.open(source_path) as source_image:
            if structure_control == "depth":
                control = render_depth_v2_control(
                    source_image,
                    source_label=source_path.as_posix(),
                )
            elif structure_control == "edge":
                control = render_canny_control(
                    source_image,
                    source_label=source_path.as_posix(),
                )
            else:
                raise QwenCharacterEditError(f"Unknown structure control {structure_control}")
    except (OSError, DepthV2ControlError, CannyControlError) as error:
        raise QwenCharacterEditError(str(error)) from error
    return source_path, control


def _case_route_kind(case: QwenIdentityCase) -> str:
    if case.edit_context is None:
        raise QwenCharacterEditError(f"Character edit case {case.name} has no route context")
    return str(case.edit_context["route_kind"])
