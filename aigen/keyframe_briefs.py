from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from aigen.keyframe_examples import KeyframeExampleExtractionConfig, extract_keyframe_example
from aigen.keyframe_judge import KeyframeJudgeConfig, QwenKeyframeJudge
from aigen.keyframes import (
    AcceptanceSpec,
    AssetSpec,
    CanvasSpec,
    CharacterSpec,
    ControlConditionSpec,
    IdentityPrimerSpec,
    KeyframeJobSpec,
    KeyframeSpec,
    OutputSpec,
    PathSpec,
    PipelineSpec,
    PromptSpec,
    SamplingSpec,
    VariantSpec,
    plan_keyframe_job,
    run_keyframe_job,
)
from aigen.keyframe_polish import (
    diagnose_keyframe_polish,
    run_keyframe_polish_job,
    select_keyframe_polish,
)
from aigen.keyframe_score import KeyframeScoreConfig, score_keyframe_run, select_scored_keyframe_run
from aigen.runtime_profiles import keyframe_profile_for_name, keyframe_refine_profile_for_name
from aigen.vlm_json import VlmJsonError, json_object_from_vlm_response


KEYFRAME_BRIEF_SCHEMA = "schemas/keyframe-brief.schema.json"
KEYFRAME_BRIEF_PLAN_SCHEMA = "schemas/keyframe-brief-plan.schema.json"
KEYFRAME_BRIEF_KIND = "keyframe-brief"
KEYFRAME_BRIEF_PLAN_KIND = "keyframe-brief-plan"
KEYFRAME_BRIEF_SCHEMA_VERSION = 1
KEYFRAME_BRIEF_PLAN_SCHEMA_VERSION = 1


class KeyframeBriefError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BriefCharacterSpec(StrictModel):
    id: str
    view_bank: PathSpec


class BriefRequestSpec(StrictModel):
    action: str
    phase: str
    direction: Literal["left", "right"]
    camera: Literal["platformer-side-view"]
    description: str


class BriefExampleSpec(StrictModel):
    path: str
    name: str
    width: int
    height: int
    mirror_x: bool


class BriefGenerationSpec(StrictModel):
    seed_start: int
    seed_count: int
    output_directory: str
    filename: str
    overwrite: bool
    save_conditions: bool
    save_contact_sheet: bool


class BriefOutputSpec(StrictModel):
    assets_directory: str
    plan_path: str
    job_path: str


class KeyframeBriefSpec(StrictModel):
    schema_path: str = Field(alias="$schema")
    schema_version: Literal[1]
    kind: Literal["keyframe-brief"]
    id: str
    pipeline: PipelineSpec
    character: BriefCharacterSpec
    request: BriefRequestSpec
    example: BriefExampleSpec
    generation: BriefGenerationSpec
    output: BriefOutputSpec


class BriefControlPlanSpec(StrictModel):
    name: str
    type: Literal["pose", "canny", "softedge", "depth"]
    source: Literal["example_pose", "example_contour"]
    scale: float
    start: float
    end: float
    residual_mask_source: Literal["example_boundary_mask"] | None = None


class BriefScoringPlanSpec(StrictModel):
    top_k: int
    priorities: list[str]
    checks: list[str]

    @model_validator(mode="after")
    def top_k_selects_candidates(self) -> BriefScoringPlanSpec:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        return self

    @field_validator("priorities", "checks")
    @classmethod
    def no_placeholder_items(cls, values: list[str]) -> list[str]:
        placeholders = {"", ".", "..."}
        if any(value.strip() in placeholders for value in values):
            raise ValueError("scoring priorities and checks must be concrete")
        return values


class BriefPolishPlanSpec(StrictModel):
    enabled: bool
    policy: str
    max_regions: int


class KeyframeBriefPlanSpec(StrictModel):
    schema_path: str = Field(alias="$schema")
    schema_version: Literal[1]
    kind: Literal["keyframe-brief-plan"]
    brief_id: str
    planner_id: str
    source_brief_sha256: str
    planner_prompt_sha256: str
    identity_description: str
    pose_description: str
    platformer_camera_description: str
    identity_primer: IdentityPrimerSpec
    prompt: PromptSpec
    canvas: CanvasSpec
    sampling: SamplingSpec
    controls: list[BriefControlPlanSpec]
    scoring: BriefScoringPlanSpec
    polish: BriefPolishPlanSpec
    rationale: list[str]

    @model_validator(mode="after")
    def negative_prompt_requires_active_true_cfg(self) -> KeyframeBriefPlanSpec:
        if self.prompt.negative is not None and self.prompt.true_cfg_scale <= 1.0:
            raise ValueError("prompt.negative requires true_cfg_scale > 1.0")
        return self


def keyframe_brief_schema() -> dict[str, Any]:
    schema = KeyframeBriefSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def keyframe_brief_plan_schema() -> dict[str, Any]:
    schema = KeyframeBriefPlanSpec.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_keyframe_brief(path: Path) -> KeyframeBriefSpec:
    try:
        return KeyframeBriefSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise KeyframeBriefError(f"Invalid keyframe brief {path}: {error}") from error


def load_keyframe_brief_plan(path: Path) -> KeyframeBriefPlanSpec:
    try:
        return KeyframeBriefPlanSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise KeyframeBriefError(f"Invalid keyframe brief plan {path}: {error}") from error


def plan_keyframe_brief(
    brief_path: Path,
    config: KeyframeJudgeConfig,
    *,
    project_root: Path,
    runner: Any | None = None,
) -> dict[str, Any]:
    spec = load_keyframe_brief(brief_path)
    view_bank_path = _resolve_path(spec.character.view_bank.path, brief_path.parent)
    example_path = _resolve_path(spec.example.path, brief_path.parent)
    view_bank = _read_json(view_bank_path)
    view_options = _view_options(view_bank)
    prompt = _planner_prompt(spec, view_options)
    raw_text = (runner if runner else QwenKeyframeJudge(config)).judge_candidate(
        prompt,
        [example_path, *[Path(option["path"]) for option in view_options]],
    )
    generated = _json_object(raw_text)
    plan_path = _resolve_output_path(spec.output.plan_path, brief_path.parent)
    raw_path = plan_path.with_suffix(".raw.txt")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(raw_text + "\n", encoding="utf-8")
    payload = {
        "$schema": _schema_reference(plan_path, project_root / KEYFRAME_BRIEF_PLAN_SCHEMA),
        "schema_version": KEYFRAME_BRIEF_PLAN_SCHEMA_VERSION,
        "kind": KEYFRAME_BRIEF_PLAN_KIND,
        "brief_id": spec.id,
        "planner_id": config.judge_id,
        "source_brief_sha256": _sha256_file(brief_path),
        "planner_prompt_sha256": _sha256_text(prompt),
        **generated,
    }
    try:
        plan = KeyframeBriefPlanSpec.model_validate(payload)
    except ValidationError as error:
        raise KeyframeBriefError(f"Invalid generated brief plan {plan_path}: {error}") from error
    _write_json(plan_path, plan.model_dump(mode="json", by_alias=True, exclude_none=True))
    return {
        "status": "planned",
        "brief_id": spec.id,
        "plan_path": plan_path.as_posix(),
        "raw_response": raw_path.as_posix(),
        "identity_primer": plan.identity_primer.model_dump(mode="json"),
        "controls": [control.model_dump(mode="json", exclude_none=True) for control in plan.controls],
        "scoring": plan.scoring.model_dump(mode="json"),
        "polish": plan.polish.model_dump(mode="json"),
    }


def materialize_keyframe_brief(
    brief_path: Path,
    *,
    project_root: Path,
    pose_device: str = "cpu",
) -> dict[str, Any]:
    spec = load_keyframe_brief(brief_path)
    plan_path = _resolve_path(spec.output.plan_path, brief_path.parent)
    plan = load_keyframe_brief_plan(plan_path)
    profile = keyframe_profile_for_name(spec.pipeline.profile)
    extraction = extract_keyframe_example(
        KeyframeExampleExtractionConfig(
            source=_resolve_path(spec.example.path, brief_path.parent),
            output_dir=_resolve_output_path(spec.output.assets_directory, brief_path.parent),
            name=spec.example.name,
            width=plan.canvas.width,
            height=plan.canvas.height,
            mirror_x=spec.example.mirror_x,
            pose_device=pose_device,
        )
    )
    job_path = _resolve_output_path(spec.output.job_path, brief_path.parent)
    job = _keyframe_job_from_brief(spec, plan, extraction, job_path, project_root)
    _write_json(job_path, job.model_dump(mode="json", by_alias=True, exclude_none=True))
    resolved = plan_keyframe_job(job_path, profile, project_root=project_root)
    return {
        "status": "materialized",
        "brief_id": spec.id,
        "plan_path": plan_path.as_posix(),
        "job_path": job_path.as_posix(),
        "extraction": extraction,
        "resolved": {
            "job_id": resolved["job_id"],
            "tokens": resolved["tokens"],
            "condition_plan": resolved["condition_plan"],
            "vram_plan": resolved["vram_plan"],
            "outputs": resolved["output"]["files"],
        },
    }


def run_keyframe_brief(
    brief_path: Path,
    *,
    project_root: Path,
    pose_device: str = "cpu",
) -> dict[str, Any]:
    materialized = materialize_keyframe_brief(brief_path, project_root=project_root, pose_device=pose_device)
    job_path = Path(materialized["job_path"])
    profile = keyframe_profile_for_name(load_keyframe_brief(brief_path).pipeline.profile)
    result = run_keyframe_job(job_path, profile, project_root=project_root)
    return {
        "status": "completed",
        "brief_id": materialized["brief_id"],
        "job_path": job_path.as_posix(),
        "run_dir": result["effective_config"]["output"]["directory"],
        "result": result,
    }


def execute_keyframe_brief(
    brief_path: Path,
    config: KeyframeJudgeConfig,
    *,
    project_root: Path,
    pose_device: str = "cpu",
    runner: Any | None = None,
) -> dict[str, Any]:
    planned = plan_keyframe_brief(brief_path, config, project_root=project_root, runner=runner)
    generated = run_keyframe_brief(brief_path, project_root=project_root, pose_device=pose_device)
    run_dir = Path(generated["run_dir"])
    score = score_keyframe_run(run_dir, KeyframeScoreConfig(), project_root=project_root)
    selection = select_scored_keyframe_run(run_dir, top_k=planned["scoring"]["top_k"])
    polish = _polish_selected_candidates(
        brief_path,
        config,
        project_root=project_root,
        selected=selection["selected"],
    )
    return {
        "status": "completed",
        "brief_id": planned["brief_id"],
        "plan_path": planned["plan_path"],
        "job_path": generated["job_path"],
        "run_dir": generated["run_dir"],
        "score": score,
        "selection": selection,
        "polish": polish,
        "result": generated["result"],
    }


def _polish_selected_candidates(
    brief_path: Path,
    config: KeyframeJudgeConfig,
    *,
    project_root: Path,
    selected: list[str],
) -> list[dict[str, Any]]:
    spec = load_keyframe_brief(brief_path)
    plan = load_keyframe_brief_plan(_resolve_path(spec.output.plan_path, brief_path.parent))
    if not plan.polish.enabled:
        return []
    profile = keyframe_refine_profile_for_name("kontext-inpaint-local")
    outputs = []
    for candidate in selected:
        job_path = _write_polish_job(spec, plan, brief_path, candidate, project_root)
        diagnose = diagnose_keyframe_polish(job_path, config=config, project_root=project_root)
        result = run_keyframe_polish_job(job_path, profile, project_root=project_root)
        selection = select_keyframe_polish(job_path, config=config, project_root=project_root)
        outputs.append(
            {
                "candidate": candidate,
                "job_path": job_path.as_posix(),
                "plan_path": diagnose["plan_path"],
                "run_dir": result["effective_config"]["output"]["directory"],
                "final_composite": selection["final_composite"],
            }
        )
    return outputs


def _write_polish_job(
    spec: KeyframeBriefSpec,
    plan: KeyframeBriefPlanSpec,
    brief_path: Path,
    candidate: str,
    project_root: Path,
) -> Path:
    base_output_dir = _resolve_output_path(spec.output.plan_path, brief_path.parent).parent / "polish" / candidate
    job_path = base_output_dir / "job.json"
    payload = {
        "$schema": _schema_reference(job_path, project_root / "schemas/keyframe-polish-job.schema.json"),
        "schema_version": 1,
        "kind": "keyframe-polish",
        "id": f"{spec.id}.polish.{candidate}",
        "pipeline": {"profile": "kontext-inpaint-local"},
        "base": {
            "run_dir": _relative_path(_resolve_output_path(spec.generation.output_directory, brief_path.parent), job_path.parent),
            "candidate": candidate,
        },
        "character": {
            "id": spec.character.id,
            "identity_primer": plan.identity_primer.model_dump(mode="json"),
        },
        "plan": {"path": "plan.json"},
        "planner": {"max_regions": plan.polish.max_regions},
        "micro_sweep": {"strength_offsets": [-0.06, 0.0, 0.06], "seed_offsets": [0, 1]},
        "output": {
            "directory": "outputs",
            "overwrite": True,
            "save_debug_images": True,
            "save_contact_sheet": True,
        },
        "acceptance": {
            "manual": [
                *plan.scoring.checks,
                "local polish preserves pose and silhouette",
                "outside feathered mask stays unchanged",
            ]
        },
    }
    _write_json(job_path, payload)
    return job_path


def _keyframe_job_from_brief(
    spec: KeyframeBriefSpec,
    plan: KeyframeBriefPlanSpec,
    extraction: dict[str, Any],
    job_path: Path,
    project_root: Path,
) -> KeyframeJobSpec:
    assets = _assets_from_plan(plan, extraction, job_path.parent)
    conditions = [_condition_from_plan(control) for control in plan.controls]
    job = KeyframeJobSpec(
        **{
            "$schema": _schema_reference(job_path, project_root / "schemas/keyframe-job.schema.json"),
            "schema_version": 1,
            "kind": "character-keyframe",
            "id": spec.id,
            "pipeline": spec.pipeline.model_dump(mode="json"),
            "character": CharacterSpec(
                id=spec.character.id,
                identity_primer=plan.identity_primer,
            ).model_dump(mode="json"),
            "keyframe": KeyframeSpec(
                action=spec.request.action,
                phase=spec.request.phase,
                direction=spec.request.direction,
                camera="orthographic-side",
            ).model_dump(mode="json"),
            "assets": assets.model_dump(mode="json", exclude_none=True),
            "prompt": plan.prompt.model_dump(mode="json", exclude_none=True),
            "canvas": plan.canvas.model_dump(mode="json"),
            "sampling": plan.sampling.model_dump(mode="json"),
            "conditions": [condition.model_dump(mode="json", exclude_none=True) for condition in conditions],
            "variants": [
                VariantSpec(name=_seed_name(seed), seed=seed).model_dump(mode="json")
                for seed in range(spec.generation.seed_start, spec.generation.seed_start + spec.generation.seed_count)
            ],
            "output": OutputSpec(
                directory=spec.generation.output_directory,
                filename=spec.generation.filename,
                overwrite=spec.generation.overwrite,
                save_conditions=spec.generation.save_conditions,
                save_contact_sheet=spec.generation.save_contact_sheet,
            ).model_dump(mode="json"),
            "acceptance": AcceptanceSpec(
                manual=plan.scoring.checks,
                minimum_passing_variants=min(plan.scoring.top_k, spec.generation.seed_count),
            ).model_dump(mode="json"),
        }
    )
    return job


def _assets_from_plan(plan: KeyframeBriefPlanSpec, extraction: dict[str, Any], base_dir: Path) -> AssetSpec:
    extracted_assets = extraction["assets"]
    pose_path = _relative_path(Path(extracted_assets["pose"]["path"]), base_dir)
    contour_path = None
    boundary_path = None
    if any(control.source == "example_contour" for control in plan.controls):
        contour_path = _relative_path(Path(extracted_assets["contour"]["path"]), base_dir)
    if any(control.residual_mask_source == "example_boundary_mask" for control in plan.controls):
        boundary_path = _relative_path(Path(extracted_assets["boundary_mask"]["path"]), base_dir)
    return AssetSpec(
        pose=PathSpec(path=pose_path),
        contour=PathSpec(path=contour_path) if contour_path else None,
        boundary_mask=PathSpec(path=boundary_path) if boundary_path else None,
    )


def _condition_from_plan(control: BriefControlPlanSpec) -> ControlConditionSpec:
    image = "pose" if control.source == "example_pose" else "contour"
    residual_mask = "boundary_mask" if control.residual_mask_source == "example_boundary_mask" else None
    return ControlConditionSpec(
        name=control.name,
        type=control.type,
        image=image,
        scale=control.scale,
        start=control.start,
        end=control.end,
        residual_mask=residual_mask,
    )


def _planner_prompt(spec: KeyframeBriefSpec, view_options: list[dict[str, Any]]) -> str:
    view_lines = "\n".join(_view_option_line(option) for option in view_options)
    return f"""You are planning a production AI keyframe-generation job.

The user supplies an identity view bank and one example platformer sprite. Plan the generation job.
Do not select the prettiest view. Choose the identity primer that best supports the requested gameplay camera and action readability.
The identity view bank is the source of truth for character identity, clothing, hair, colors and style.
The example sprite is only the source of truth for action pose, silhouette intent and gameplay readability.
Platformer side-view animation may cheat toward the camera when that improves readability. Do not require an exact mathematical 90-degree profile unless the request explicitly says so.

Request:
- character: {spec.character.id}
- action: {spec.request.action}
- phase: {spec.request.phase}
- direction: {spec.request.direction}
- camera: {spec.request.camera}
- description: {spec.request.description}
- output canvas requested by example extraction: {spec.example.width}x{spec.example.height}

Available identity-primer views:
{view_lines}

Return JSON only with these keys:
{{
  "identity_description": "concise visual identity description from the identity images",
  "pose_description": "concise pose/action description from the example sprite",
  "platformer_camera_description": "camera/readability interpretation for a platformer sprite",
  "identity_primer": {{"view": "front|left_profile|right_profile|back", "path": "exact path from the available views"}},
  "prompt": {{"clip": "...", "t5": "...", "true_cfg_scale": 1.0}},
  "canvas": {{"width": {spec.example.width}, "height": {spec.example.height}, "reference_max_area": 294912, "max_sequence_length": 128}},
  "sampling": {{"steps": 28, "guidance_scale": 2.5}},
  "controls": [
    {{"name": "source_pose", "type": "pose", "source": "example_pose", "scale": 0.72, "start": 0.0, "end": 0.65}},
    {{"name": "source_contour", "type": "canny", "source": "example_contour", "scale": 0.25, "start": 0.0, "end": 0.35, "residual_mask_source": "example_boundary_mask"}}
  ],
  "scoring": {{"top_k": 3, "priorities": ["condition adherence", "action readability", "identity preservation"], "checks": ["source pose is readable", "identity outfit is preserved", "top candidates are suitable for local polish"]}},
  "polish": {{"enabled": true, "policy": "model-planned local polish on top candidates only", "max_regions": 4}},
  "rationale": ["..."]
}}

Keep prompts specific to the approved identity primer and the example action. Use separate CLIP and T5 prompt text. Do not mention internal filenames in prompts.
Never return placeholder strings such as "...".
prompt.true_cfg_scale is required.
Omit prompt.negative when true_cfg_scale is 1.0."""


def _view_options(view_bank: dict[str, Any]) -> list[dict[str, Any]]:
    ordered_names = ("front", "left_profile", "right_profile", "back")
    views = view_bank["views"]
    options = []
    for name in ordered_names:
        if name in views:
            entry = views[name]
            options.append(
                {
                    "view": name,
                    "path": entry["image"]["path"],
                    "camera": entry["view"]["camera"],
                    "pose": entry["view"]["pose"],
                    "design_notes": _view_design_notes(entry),
                }
            )
    for name, entry in views.items():
        if name not in ordered_names:
            options.append(
                {
                    "view": name,
                    "path": entry["image"]["path"],
                    "camera": entry["view"]["camera"],
                    "pose": entry["view"]["pose"],
                    "design_notes": _view_design_notes(entry),
                }
            )
    return options


def _view_option_line(option: dict[str, Any]) -> str:
    notes = "; ".join(option["design_notes"])
    suffix = f" design notes: {notes}" if notes else ""
    return f"- {option['view']}: {option['path']} ({option['camera']}, {option['pose']}).{suffix}"


def _view_design_notes(entry: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if "acceptance" in entry:
        notes.extend(entry["acceptance"].get("manual", []))
    if "prompt" in entry:
        prompt = entry["prompt"]
        notes.extend([prompt.get("clip", ""), prompt.get("t5", "")])
    return [note for note in notes if note]


def _json_object(raw_text: str) -> dict[str, Any]:
    try:
        return json_object_from_vlm_response(raw_text)
    except VlmJsonError as error:
        raise KeyframeBriefError(str(error)) from error


def _seed_name(seed: int) -> str:
    return f"seed_{seed:03d}"


def _relative_path(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), base_dir.resolve())).as_posix()


def _schema_reference(target_path: Path, schema_path: Path) -> str:
    try:
        return _relative_path(schema_path, target_path.parent)
    except ValueError:
        return schema_path.resolve().as_posix()


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise KeyframeBriefError(f"Missing path: {path.as_posix()}")
    return path


def _resolve_output_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise KeyframeBriefError(f"Cannot read JSON {path}: {error}") from error


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
