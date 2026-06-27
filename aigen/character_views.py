from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aigen.keyframes import (
    AcceptanceSpec as KeyframeAcceptanceSpec,
    AssetSpec,
    CanvasSpec as KeyframeCanvasSpec,
    CharacterSpec,
    ControlConditionSpec as KeyframeControlConditionSpec,
    IdentityPrimerSpec,
    KeyframeJobSpec,
    KeyframeProfile,
    KeyframeSpec,
    OutputSpec as KeyframeOutputSpec,
    PathSpec as KeyframePathSpec,
    PipelineSpec as KeyframePipelineSpec,
    PromptSpec as KeyframePromptSpec,
    SamplingSpec as KeyframeSamplingSpec,
    VariantSpec as KeyframeVariantSpec,
    run_keyframe_spec,
)


CHARACTER_VIEW_JOB_SCHEMA = "schemas/character-view-job.schema.json"
CHARACTER_VIEW_BANK_SCHEMA = "schemas/character-view-bank.schema.json"
CHARACTER_VIEW_KIND = "character-view"
CHARACTER_VIEW_SCHEMA_VERSION = 1
VIEW_BANK_KIND = "character-view-bank"
VIEW_BANK_SCHEMA_VERSION = 1
ViewName = Literal["front", "left_profile", "right_profile", "back"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathSpec(StrictModel):
    path: str


class PipelineSpec(StrictModel):
    profile: str


class SourceCharacterSpec(StrictModel):
    id: str
    source_reference: PathSpec


class CharacterViewSpec(StrictModel):
    name: ViewName
    camera: Literal["orthographic-front", "orthographic-side", "orthographic-back"]
    pose: str


class ViewAssetsSpec(StrictModel):
    pose: PathSpec
    contour: PathSpec
    boundary_mask: PathSpec


class PromptSpec(StrictModel):
    clip: str
    t5: str
    negative: str | None = None
    true_cfg_scale: float


class CanvasSpec(StrictModel):
    width: int
    height: int
    reference_max_area: int
    max_sequence_length: int


class SamplingSpec(StrictModel):
    steps: int
    guidance_scale: float


class ControlConditionSpec(StrictModel):
    name: str
    type: Literal["pose", "canny", "softedge", "depth"]
    image: str
    scale: float
    start: float
    end: float
    residual_mask: str | None = None


class VariantSpec(StrictModel):
    name: str
    seed: int


class ViewOutputSpec(StrictModel):
    directory: str
    filename: str
    canonical_path: str
    bank_path: str
    overwrite: bool
    save_conditions: bool
    save_contact_sheet: bool


class AcceptanceSpec(StrictModel):
    manual: list[str]
    minimum_passing_variants: int


class CharacterViewJobSpec(StrictModel):
    schema_path: str = Field(alias="$schema")
    schema_version: Literal[1]
    kind: Literal["character-view"]
    id: str
    pipeline: PipelineSpec
    character: SourceCharacterSpec
    view: CharacterViewSpec
    assets: ViewAssetsSpec
    prompt: PromptSpec
    canvas: CanvasSpec
    sampling: SamplingSpec
    conditions: list[ControlConditionSpec]
    variants: list[VariantSpec]
    output: ViewOutputSpec
    acceptance: AcceptanceSpec


class CharacterViewError(RuntimeError):
    pass


def character_view_job_schema() -> dict[str, Any]:
    schema = CharacterViewJobSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def character_view_bank_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "kind", "character", "views"],
        "properties": {
            "schema_version": {"const": VIEW_BANK_SCHEMA_VERSION},
            "kind": {"const": VIEW_BANK_KIND},
            "character": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "source_reference"],
                "properties": {
                    "id": {"type": "string"},
                    "source_reference": _asset_schema(),
                },
            },
            "views": {
                "type": "object",
                "additionalProperties": _view_entry_schema(),
            },
        },
    }


def left_profile_view_template() -> dict[str, Any]:
    spec = CharacterViewJobSpec(
        **{
            "$schema": "../../schemas/character-view-job.schema.json",
            "schema_version": CHARACTER_VIEW_SCHEMA_VERSION,
            "kind": CHARACTER_VIEW_KIND,
            "id": "ai46.left_profile.neutral.v1",
            "pipeline": {"profile": "nunchaku-kontext-pose-quality"},
            "character": {
                "id": "ai46",
                "source_reference": {
                    "path": "../../assets/characters/ai46/views/front_v1.png",
                },
            },
            "view": {
                "name": "left_profile",
                "camera": "orthographic-side",
                "pose": "neutral-standing",
            },
            "assets": {
                "pose": {"path": "../../assets/views/ai46/left_profile_neutral_pose_512x768.png"},
                "contour": {"path": "../../assets/views/ai46/left_profile_neutral_canny_512x768.png"},
                "boundary_mask": {"path": "../../assets/views/ai46/left_profile_neutral_boundary_512x768.png"},
            },
            "prompt": {
                "clip": (
                    "Same anime girl, neutral standing model sheet, strict left-facing side profile, "
                    "exactly one eye visible, short light-brown bob, brown leather jacket, white shirt, "
                    "blue tie, brown shorts, blue thigh-high socks, brown boots."
                ),
                "t5": (
                    "Full-body neutral-standing character turnaround view. Generate the approved left-profile "
                    "identity primer for the same anime girl. Strict orthographic side view, relaxed neutral "
                    "standing pose, arms down, feet grounded, one visible eye, profile nose and chin, short "
                    "light-brown bob, white shirt, blue tie, brown leather jacket, brown shorts, gloves, blue "
                    "thigh-high socks and brown boots. Clean neutral background."
                ),
                "negative": "walking pose, action pose, punch, open hand gesture, front view, three-quarter view, long hair, ponytail, cropped feet",
                "true_cfg_scale": 1.25,
            },
            "canvas": {
                "width": 512,
                "height": 768,
                "reference_max_area": 294912,
                "max_sequence_length": 128,
            },
            "sampling": {"steps": 28, "guidance_scale": 2.5},
            "conditions": [
                {
                    "name": "pose",
                    "type": "pose",
                    "image": "pose",
                    "scale": 0.50,
                    "start": 0.0,
                    "end": 0.55,
                },
                {
                    "name": "profile_contour",
                    "type": "canny",
                    "image": "contour",
                    "residual_mask": "boundary_mask",
                    "scale": 0.35,
                    "start": 0.0,
                    "end": 0.40,
                },
            ],
            "variants": [
                {"name": "seed_001", "seed": 1},
                {"name": "seed_002", "seed": 2},
                {"name": "seed_003", "seed": 3},
                {"name": "seed_004", "seed": 4},
            ],
            "output": {
                "directory": "../../runs/characters/ai46/views/left_profile_neutral_v1",
                "filename": "{id}__{variant}.png",
                "canonical_path": "../../assets/characters/ai46/views/left_profile_v1.png",
                "bank_path": "../../assets/characters/ai46/view_bank.json",
                "overwrite": False,
                "save_conditions": True,
                "save_contact_sheet": True,
            },
            "acceptance": {
                "manual": [
                    "neutral standing pose",
                    "strict left profile",
                    "exactly one visible eye",
                    "short bob preserved",
                    "jacket, tie, shorts, socks and boots preserved",
                    "feet fully visible",
                    "no action pose bias",
                ],
                "minimum_passing_variants": 1,
            },
        }
    )
    return spec.model_dump(mode="json", by_alias=True, exclude_none=True)


def load_character_view_job(path: Path) -> CharacterViewJobSpec:
    try:
        return CharacterViewJobSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise CharacterViewError(f"Invalid character view job {path}: {error}") from error


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
        "source_reference": _asset_json(_resolve_path(spec.character.source_reference.path, base_dir)),
        "pose": _asset_json(_resolve_path(spec.assets.pose.path, base_dir)),
        "contour": _asset_json(_resolve_path(spec.assets.contour.path, base_dir)),
        "boundary_mask": _asset_json(_resolve_path(spec.assets.boundary_mask.path, base_dir)),
    }
    for name in ("pose", "contour", "boundary_mask"):
        asset = assets[name]
        if asset["width"] != spec.canvas.width or asset["height"] != spec.canvas.height:
            raise CharacterViewError(
                f"Asset {name} must be {spec.canvas.width}x{spec.canvas.height}, "
                f"got {asset['width']}x{asset['height']}"
            )

    output_dir = _resolve_output_dir(spec.output.directory, base_dir)
    canonical_path = _resolve_output_path(spec.output.canonical_path, base_dir)
    bank_path = _resolve_output_path(spec.output.bank_path, base_dir)
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
        "spec_sha256": _sha256_bytes(job_path.read_bytes()),
    }


def accept_character_view(
    job_path: Path,
    *,
    run_dir: Path,
    candidate: str,
    project_root: Path,
) -> dict[str, Any]:
    resolved = resolve_character_view_job(job_path, project_root=project_root, check_outputs=False)
    result = _read_json(run_dir.resolve() / "result.json")
    output = _candidate_output(result, candidate)
    source_image = Path(output["path"]).resolve()
    canonical_path = Path(resolved["output"]["canonical_path"])
    if canonical_path.exists() and not resolved["output"]["overwrite"]:
        raise CharacterViewError(f"Canonical view already exists and overwrite=false: {canonical_path.as_posix()}")

    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, canonical_path)
    canonical_asset = _asset_json(canonical_path)
    entry = {
        "view": resolved["view"],
        "image": canonical_asset,
        "accepted_candidate": candidate,
        "accepted_seed": output.get("seed"),
        "source_job": {
            "view_job_id": resolved["job_id"],
            "view_job_path": resolved["job_path"],
            "view_job_sha256": resolved["spec_sha256"],
            "run_dir": run_dir.resolve().as_posix(),
            "run_job_id": result["job_id"],
            "result_sha256": _sha256_bytes((run_dir.resolve() / "result.json").read_bytes()),
            "candidate_path": source_image.as_posix(),
            "candidate_sha256": _sha256_bytes(source_image.read_bytes()),
        },
        "prompt": resolved["prompt"],
        "assets": resolved["assets"],
        "acceptance": resolved["acceptance"],
    }
    bank_path = Path(resolved["output"]["bank_path"])
    bank = _load_or_create_bank(bank_path, resolved)
    bank["views"][resolved["view"]["name"]] = entry
    _write_json(bank_path, bank)
    return {
        "schema_version": CHARACTER_VIEW_SCHEMA_VERSION,
        "status": "accepted",
        "character": resolved["character"]["id"],
        "view": resolved["view"]["name"],
        "canonical_path": canonical_path.as_posix(),
        "canonical_sha256": canonical_asset["sha256"],
        "bank_path": bank_path.as_posix(),
    }


def _load_or_create_bank(path: Path, resolved: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        return _read_json(path)
    return {
        "schema_version": VIEW_BANK_SCHEMA_VERSION,
        "kind": VIEW_BANK_KIND,
        "character": {
            "id": resolved["character"]["id"],
            "source_reference": resolved["character"]["source_reference"],
        },
        "views": {},
    }


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


def _direction_for_view(view: ViewName) -> Literal["left", "right"]:
    if view == "left_profile":
        return "left"
    if view == "right_profile":
        return "right"
    raise CharacterViewError(f"View-run supports profile views, not {view}")


def _planned_outputs(spec: CharacterViewJobSpec, output_dir: Path) -> list[dict[str, str | int]]:
    return [
        {
            "name": variant.name,
            "seed": variant.seed,
            "path": (output_dir / spec.output.filename.format(id=spec.id, variant=variant.name)).as_posix(),
        }
        for variant in spec.variants
    ]


def _asset_json(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        mode = image.mode
        width, height = image.size
    return {
        "path": path.as_posix(),
        "sha256": _sha256_bytes(path.read_bytes()),
        "mode": mode,
        "width": width,
        "height": height,
    }


def _asset_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "sha256", "mode", "width", "height"],
        "properties": {
            "path": {"type": "string"},
            "sha256": {"type": "string"},
            "mode": {"type": "string"},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
        },
    }


def _view_entry_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "required": ["view", "image", "accepted_candidate", "source_job"],
        "properties": {
            "view": {"type": "object"},
            "image": _asset_schema(),
            "accepted_candidate": {"type": "string"},
            "accepted_seed": {"type": ["integer", "null"]},
            "source_job": {"type": "object"},
            "prompt": {"type": "object"},
            "assets": {"type": "object"},
            "acceptance": {"type": "object"},
        },
    }


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise CharacterViewError(f"Missing path: {path.as_posix()}")
    return path


def _resolve_output_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_output_dir(value: str, base_dir: Path) -> Path:
    return _resolve_output_path(value, base_dir)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CharacterViewError(f"Cannot read character view JSON {path.as_posix()}: {error}") from error


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_commit(project_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
