from __future__ import annotations

from pathlib import Path
from typing import Any

from aigen.character_qwen_edit import (
    PlannedQwenCharacterEdit,
    QwenCharacterPoseSource,
    load_qwen_character_reference_context,
    run_planned_qwen_character_edit,
)
from aigen.character_verification_models import (
    CharacterVerificationMatrixSpec,
    load_character_verification_matrix,
)
from aigen.generation.qwen_image_edit_identity import (
    QwenIdentityCase,
    QwenImageEditProfile,
)
from aigen.image_assets import image_asset_json
from aigen.manifest_io import resolve_existing_path, sha256_bytes, write_json
from aigen.progress import StatusReporter


CHARACTER_VERIFICATION_RESULT_KIND = "character-verification-matrix-result"


def run_character_verification_matrix(
    *,
    matrix_path: Path,
    output_dir: Path,
    profile: QwenImageEditProfile,
    overwrite: bool,
    progress: StatusReporter,
) -> dict[str, Any]:
    matrix_path = matrix_path.resolve()
    spec = load_character_verification_matrix(matrix_path)
    matrix_sha256 = sha256_bytes(matrix_path.read_bytes())
    pack_path = resolve_existing_path(spec.reference_pack, matrix_path.parent)
    context = load_qwen_character_reference_context(
        pack_path=pack_path,
        progress=progress,
        phase="load verification matrix reference pack",
    )
    pose_sources = {
        name: resolve_existing_path(path, matrix_path.parent)
        for name, path in spec.pose_sources.items()
    }
    case_pose_sources = {
        case.name: QwenCharacterPoseSource(
            name=case.pose.source,
            path=pose_sources[case.pose.source],
            mode=case.pose.mode,
        )
        for case in spec.cases
        if case.pose is not None
    }
    pack_reference_names = tuple(context.pack.references)
    selected_reference_names = tuple(pack_reference_names[index - 1] for index in spec.reference_indices)
    cases = _matrix_cases(
        spec,
        reference_names=selected_reference_names,
    )
    result = run_planned_qwen_character_edit(
        planned=PlannedQwenCharacterEdit(context=context, edit_cases=cases),
        pose_sources=case_pose_sources,
        structure_source_path=None,
        structure_control=None,
        output_dir=output_dir,
        profile=profile,
        max_side=spec.canvas.max_side,
        steps=None,
        true_cfg_scale=None,
        guidance_scale=None,
        seed=spec.cases[0].seeds[0],
        max_sequence_length=spec.canvas.max_sequence_length,
        candidates_per_case=len(spec.cases[0].seeds),
        overwrite=overwrite,
        nunchaku_blocks_on_gpu=None,
        output_format=spec.canvas.output_format,
        resolution=None,
        result_kind=CHARACTER_VERIFICATION_RESULT_KIND,
        manifest_context={
            "kind": spec.kind,
            "id": spec.id,
            "path": matrix_path.as_posix(),
            "sha256": matrix_sha256,
        },
        postprocess=False,
        progress=progress,
    )
    raw_contact_sheet = Path(result["output"]["contact_sheet"])
    result["matrix"] = {
        "id": spec.id,
        "path": matrix_path.as_posix(),
        "sha256": matrix_sha256,
        "judgment": "manual_visual_raw_only",
        "pose_sources": {
            name: image_asset_json(path)
            for name, path in pose_sources.items()
        },
        "raw_contact_sheet": image_asset_json(raw_contact_sheet),
    }
    result["output"]["raw_contact_sheet"] = raw_contact_sheet.as_posix()
    write_json(Path(result["output"]["result"]), result)
    return result


def _matrix_cases(
    spec: CharacterVerificationMatrixSpec,
    *,
    reference_names: tuple[str, ...],
) -> tuple[QwenIdentityCase, ...]:
    cases = []
    for case in spec.cases:
        cases.append(
            QwenIdentityCase(
                name=case.name,
                references=reference_names,
                prompt=case.instruction,
                edit_context={
                    "task": "verification_matrix",
                    "case": case.name,
                    "route_kind": case.route_kind,
                    "reference_selector": "matrix_fixed_indices",
                    "selected_ref_indices": list(spec.reference_indices),
                    "refs_used": list(reference_names),
                    "prompt_source": "matrix_user_instruction",
                },
                seeds=tuple(case.seeds),
            )
        )
    return tuple(cases)
