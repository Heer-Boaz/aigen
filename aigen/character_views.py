from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from aigen.character_view_models import (
    CHARACTER_VIEW_JOB_SCHEMA,
    CHARACTER_VIEW_SCHEMA_VERSION,
    VIEW_BANK_KIND,
    VIEW_BANK_SCHEMA_VERSION,
    CharacterViewError,
    CharacterViewBankSpec,
    CharacterViewJobSpec,
    ProfileViewName,
    ViewBankCharacterSpec,
    ViewBankEntryAcceptanceSpec,
    ViewBankEntryAssetsSpec,
    ViewBankEntrySpec,
    ViewBankViewSpec,
    ImageAssetSpec,
    load_character_view_bank,
    load_character_view_job,
)
from aigen.image_assets import image_asset_json
from aigen.manifest_io import (
    read_json,
    resolve_existing_path,
    resolve_output_path,
    sha256_bytes,
    write_json,
)
from aigen.keyframe_job_models import (
    AcceptanceSpec as KeyframeAcceptanceSpec,
    AssetSpec,
    CanvasSpec as KeyframeCanvasSpec,
    CharacterSpec,
    ControlConditionSpec as KeyframeControlConditionSpec,
    IdentityPrimerSpec,
    KeyframeJobSpec,
    KeyframeSpec,
    OutputSpec as KeyframeOutputSpec,
    PathSpec as KeyframePathSpec,
    PipelineSpec as KeyframePipelineSpec,
    PromptSpec as KeyframePromptSpec,
    SamplingSpec as KeyframeSamplingSpec,
    VariantSpec as KeyframeVariantSpec,
)
from aigen.keyframe_profiles import KeyframeProfile
from aigen.keyframes import (
    run_keyframe_spec,
)


def validate_character_view_job(job_path: Path, *, project_root: Path) -> dict[str, Any]:
    resolved = resolve_character_view_job(job_path, project_root=project_root, check_outputs=False)
    return {
        "status": "valid",
        "job_id": resolved["job_id"],
        "view": resolved["view"],
        "outputs": resolved["output"]["files"],
        "canonical_path": resolved["output"]["canonical_path"],
        "bank_path": resolved["output"]["bank_path"],
    }


def plan_character_view_job(job_path: Path, *, project_root: Path) -> dict[str, Any]:
    return resolve_character_view_job(job_path, project_root=project_root, check_outputs=True)


def run_character_view_job(job_path: Path, profile: KeyframeProfile, *, project_root: Path) -> dict[str, Any]:
    spec = load_character_view_job(job_path)
    keyframe_spec = _keyframe_spec_for_view(spec)
    return run_keyframe_spec(keyframe_spec, job_path, profile, project_root=project_root)


def resolve_character_view_job(
    job_path: Path,
    *,
    project_root: Path,
    check_outputs: bool,
) -> dict[str, Any]:
    spec = load_character_view_job(job_path)
    base_dir = job_path.parent
    assets = {
        "source_reference": image_asset_json(resolve_existing_path(spec.character.source_reference.path, base_dir)),
        "pose": image_asset_json(resolve_existing_path(spec.assets.pose.path, base_dir)),
        "contour": image_asset_json(resolve_existing_path(spec.assets.contour.path, base_dir)),
        "boundary_mask": image_asset_json(resolve_existing_path(spec.assets.boundary_mask.path, base_dir)),
    }
    for name in ("pose", "contour", "boundary_mask"):
        asset = assets[name]
        if asset["width"] != spec.canvas.width or asset["height"] != spec.canvas.height:
            raise CharacterViewError(
                f"Asset {name} must be {spec.canvas.width}x{spec.canvas.height}, "
                f"got {asset['width']}x{asset['height']}"
            )

    output_dir = resolve_output_path(spec.output.directory, base_dir)
    canonical_path = resolve_output_path(spec.output.canonical_path, base_dir)
    bank_path = resolve_output_path(spec.output.bank_path, base_dir)
    outputs = _planned_outputs(spec, output_dir)
    if check_outputs and not spec.output.overwrite:
        existing = [Path(output["path"]) for output in outputs if Path(output["path"]).exists()]
        if canonical_path.exists():
            existing.append(canonical_path)
        if existing:
            raise CharacterViewError(f"Output exists and overwrite=false: {existing[0].as_posix()}")

    return {
        "schema_version": CHARACTER_VIEW_SCHEMA_VERSION,
        "kind": "resolved-character-view",
        "job_path": job_path.resolve().as_posix(),
        "job_id": spec.id,
        "pipeline": spec.pipeline.model_dump(mode="json"),
        "character": {
            "id": spec.character.id,
            "source_reference": assets["source_reference"],
        },
        "view": spec.view.model_dump(mode="json"),
        "assets": {name: asset for name, asset in assets.items() if name != "source_reference"},
        "prompt": spec.prompt.model_dump(mode="json", exclude_none=True),
        "canvas": spec.canvas.model_dump(mode="json"),
        "sampling": spec.sampling.model_dump(mode="json"),
        "conditions": [condition.model_dump(mode="json", exclude_none=True) for condition in spec.conditions],
        "variants": [variant.model_dump(mode="json") for variant in spec.variants],
        "output": {
            **spec.output.model_dump(mode="json"),
            "directory": output_dir.as_posix(),
            "canonical_path": canonical_path.as_posix(),
            "bank_path": bank_path.as_posix(),
            "files": outputs,
        },
        "acceptance": spec.acceptance.model_dump(mode="json"),
        "git_commit": _git_commit(project_root),
        "spec_sha256": sha256_bytes(job_path.read_bytes()),
    }


def accept_character_view(
    job_path: Path,
    *,
    run_dir: Path,
    candidate: str,
    project_root: Path,
) -> dict[str, Any]:
    resolved = resolve_character_view_job(job_path, project_root=project_root, check_outputs=False)
    result = read_json(run_dir.resolve() / "result.json", label="character view result")
    output = _candidate_output(result, candidate)
    source_image = Path(output["path"]).resolve()
    canonical_path = Path(resolved["output"]["canonical_path"])
    if canonical_path.exists() and not resolved["output"]["overwrite"]:
        raise CharacterViewError(f"Canonical view already exists and overwrite=false: {canonical_path.as_posix()}")

    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, canonical_path)
    canonical_asset = image_asset_json(canonical_path)
    entry = ViewBankEntrySpec(
        view=ViewBankViewSpec(**resolved["view"]),
        image=ImageAssetSpec(**canonical_asset),
        accepted_candidate=candidate,
        accepted_seed=output.get("seed"),
        assets=ViewBankEntryAssetsSpec(**resolved["assets"]),
        acceptance=ViewBankEntryAcceptanceSpec(**resolved["acceptance"]),
    )
    bank_path = Path(resolved["output"]["bank_path"])
    bank = _load_or_create_bank(bank_path, resolved)
    bank.views[resolved["view"]["name"]] = entry
    write_json(bank_path, bank.model_dump(mode="json", exclude_none=True))
    return {
        "schema_version": CHARACTER_VIEW_SCHEMA_VERSION,
        "status": "accepted",
        "character": resolved["character"]["id"],
        "view": resolved["view"]["name"],
        "canonical_path": canonical_path.as_posix(),
        "canonical_sha256": canonical_asset["sha256"],
        "bank_path": bank_path.as_posix(),
    }


def _load_or_create_bank(path: Path, resolved: dict[str, Any]) -> CharacterViewBankSpec:
    if path.exists():
        return load_character_view_bank(path)
    return CharacterViewBankSpec(
        schema_version=VIEW_BANK_SCHEMA_VERSION,
        kind=VIEW_BANK_KIND,
        character=ViewBankCharacterSpec(
            id=resolved["character"]["id"],
            source_reference=ImageAssetSpec(**resolved["character"]["source_reference"]),
        ),
        views={},
    )


def _candidate_output(result: dict[str, Any], candidate: str) -> dict[str, Any]:
    for output in result["outputs"]:
        if output["name"] == candidate:
            return output
    raise CharacterViewError(f"Run has no candidate named {candidate}")


def _keyframe_spec_for_view(spec: CharacterViewJobSpec) -> KeyframeJobSpec:
    direction = _direction_for_view(spec.view.name)
    return KeyframeJobSpec(
        **{
            "$schema": CHARACTER_VIEW_JOB_SCHEMA,
            "schema_version": CHARACTER_VIEW_SCHEMA_VERSION,
            "kind": "character-keyframe",
            "id": spec.id,
            "pipeline": KeyframePipelineSpec(profile=spec.pipeline.profile).model_dump(mode="json"),
            "character": CharacterSpec(
                id=spec.character.id,
                identity_primer=IdentityPrimerSpec(
                    view="front",
                    path=spec.character.source_reference.path,
                ),
            ).model_dump(mode="json"),
            "keyframe": KeyframeSpec(
                action="turnaround",
                phase=spec.view.pose,
                direction=direction,
                camera="orthographic-side",
            ).model_dump(mode="json"),
            "assets": AssetSpec(
                pose=KeyframePathSpec(path=spec.assets.pose.path),
                contour=KeyframePathSpec(path=spec.assets.contour.path),
                boundary_mask=KeyframePathSpec(path=spec.assets.boundary_mask.path),
            ).model_dump(mode="json", exclude_none=True),
            "prompt": KeyframePromptSpec(**spec.prompt.model_dump(mode="json", exclude_none=True)).model_dump(
                mode="json",
                exclude_none=True,
            ),
            "canvas": KeyframeCanvasSpec(**spec.canvas.model_dump(mode="json")).model_dump(mode="json"),
            "sampling": KeyframeSamplingSpec(**spec.sampling.model_dump(mode="json")).model_dump(mode="json"),
            "conditions": [
                KeyframeControlConditionSpec(**condition.model_dump(mode="json", exclude_none=True)).model_dump(
                    mode="json",
                    exclude_none=True,
                )
                for condition in spec.conditions
            ],
            "variants": [
                KeyframeVariantSpec(**variant.model_dump(mode="json")).model_dump(mode="json")
                for variant in spec.variants
            ],
            "output": KeyframeOutputSpec(
                directory=spec.output.directory,
                filename=spec.output.filename,
                overwrite=spec.output.overwrite,
                save_conditions=spec.output.save_conditions,
                save_contact_sheet=spec.output.save_contact_sheet,
            ).model_dump(mode="json"),
            "acceptance": KeyframeAcceptanceSpec(**spec.acceptance.model_dump(mode="json")).model_dump(mode="json"),
        }
    )


def _direction_for_view(view: ProfileViewName) -> Literal["left", "right"]:
    if view == "left_profile":
        return "left"
    return "right"


def _planned_outputs(spec: CharacterViewJobSpec, output_dir: Path) -> list[dict[str, str | int]]:
    return [
        {
            "name": variant.name,
            "seed": variant.seed,
            "path": (output_dir / spec.output.filename.format(id=spec.id, variant=variant.name)).as_posix(),
        }
        for variant in spec.variants
    ]


def _git_commit(project_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
