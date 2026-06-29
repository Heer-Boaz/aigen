from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import closing
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aigen.manifest_io import read_json, sha256_file, write_json
from aigen.progress import StatusReporter
from aigen.vlm_json import VlmJsonError, json_object_from_vlm_response
from aigen.vlm_qwen import QwenVlm, QwenVlmConfig, qwen_vlm_config_json


class KeyframeJudgeError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HardRejects(StrictModel):
    front_or_three_quarter_view: bool
    two_eyes_visible: bool
    wrong_direction: bool
    cropped_feet: bool
    footwear_changed: bool
    hairstyle_changed: bool
    outfit_changed: bool
    pose_not_requested_action: bool
    severe_limb_error: bool


class JudgeScores(StrictModel):
    condition_adherence: float
    side_profile: float
    pose_match: float
    contour_match: float
    identity_preservation: float
    outfit_preservation: float
    artifact_quality: float
    overall: float


class JudgeEvidence(StrictModel):
    condition_match: str
    identity_match: str
    concerns: list[str]


class CandidateJudgment(StrictModel):
    candidate: str
    passes: bool = Field(alias="pass")
    rank_recommendation: int
    hard_rejects: HardRejects
    scores: JudgeScores
    evidence: JudgeEvidence


def judge_keyframe_run(
    run_dir: Path,
    config: QwenVlmConfig,
    *,
    project_root: Path,
    progress: StatusReporter,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.resolve()
    result = read_json(resolved_run_dir / "result.json", label="keyframe run result")
    effective_config = result["effective_config"]
    assets = result["assets"]
    judge_dir = resolved_run_dir / "judge" / config.judge_id
    prompts_dir = judge_dir / "prompts"
    raw_dir = judge_dir / "raw"
    overlay_dir = judge_dir / "overlays"
    if judge_dir.exists():
        shutil.rmtree(judge_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    with closing(QwenVlm(config)) as active_runner:
        candidate_results = []
        progress.begin(len(result["outputs"]), "judge candidates")
        for output in result["outputs"]:
            candidate_name = output["name"]
            candidate_path = Path(output["path"]).resolve()
            overlay_path = overlay_dir / f"{candidate_name}__contour_overlay.png"
            _save_contour_overlay(candidate_path, Path(assets["contour"]["path"]), overlay_path)
            prompt = _candidate_prompt(candidate_name, effective_config)
            prompt_sha256 = _sha256_text(prompt)
            (prompts_dir / f"{candidate_name}.txt").write_text(prompt, encoding="utf-8")
            image_paths = _candidate_image_paths(assets, candidate_path, overlay_path)
            raw_text = active_runner.judge_candidate(prompt, image_paths)
            (raw_dir / f"{candidate_name}.json").write_text(raw_text + "\n", encoding="utf-8")
            judgment = _parse_candidate_judgment(raw_text)
            if judgment.candidate != candidate_name:
                raise KeyframeJudgeError(
                    f"Judge returned candidate {judgment.candidate}, expected {candidate_name}"
                )
            candidate_results.append(
                {
                    "candidate": candidate_name,
                    "image": candidate_path.as_posix(),
                    "overlay": overlay_path.as_posix(),
                    "prompt_sha256": prompt_sha256,
                    "raw_response": (raw_dir / f"{candidate_name}.json").as_posix(),
                    "judgment": judgment.model_dump(mode="json", by_alias=True),
                }
            )
            progress.step(f"judged {candidate_name}")

        payload = {
            "status": "completed",
            "run_dir": resolved_run_dir.as_posix(),
            "job_id": result["job_id"],
            "git_commit": _git_commit(project_root),
            "source_result_sha256": sha256_file(resolved_run_dir / "result.json"),
            "judge": qwen_vlm_config_json(config) | {"device_report": active_runner.device_report},
            "candidates": candidate_results,
            "semantic_gate": _semantic_gate(candidate_results),
        }
        write_json(judge_dir / "judge.json", payload)
        return payload


def _candidate_image_paths(assets: dict[str, Any], candidate_path: Path, overlay_path: Path) -> list[Path]:
    paths = [
        Path(assets["identity_primer"]["path"]),
        candidate_path,
        Path(assets["pose"]["path"]),
        Path(assets["contour"]["path"]),
    ]
    if "boundary_mask" in assets:
        paths.append(Path(assets["boundary_mask"]["path"]))
    paths.append(overlay_path)
    return paths


def _candidate_prompt(candidate_name: str, effective_config: dict[str, Any]) -> str:
    keyframe = effective_config["keyframe"]
    prompt = effective_config["prompt"]
    acceptance = effective_config["acceptance"]["manual"]
    conditions = effective_config["condition_plan"]
    return f"""You are a strict visual QA judge for platformer character keyframes.

You will receive these images in order:
1. Approved character identity primer.
2. Generated candidate image named {candidate_name}.
3. Target pose condition.
4. Target contour condition.
5. Boundary mask if present.
6. Candidate image with the target contour overlaid in red.

Do not choose the prettiest image and do not rank this candidate against other candidates.
Judge whether this candidate follows the control conditions while preserving the approved character primer.

Priority order:
1. Condition adherence.
2. Strict side-profile camera.
3. Character identity and outfit preservation against the primer.
4. Artifact and aesthetic quality.

Hard reject if:
- The character is front view or three-quarter view.
- Both eyes are clearly visible when the requested camera is a side/profile view.
- The character faces or moves in the wrong direction.
- The feet are cropped.
- Distinct footwear from the identity primer is missing or replaced.
- The hairstyle no longer matches the identity primer.
- The outfit no longer matches the identity primer.
- The pose no longer reads as the requested action.
- There is a severe hand, arm, leg, or boot error that breaks the keyframe.

Camera/readability definition:
- Judge the camera against the requested keyframe camera and the supplied target contour.
- Platformer side-view animation may cheat toward the camera when that improves readability, but the candidate must still read as the requested direction and action.
- A visible full chest, centered outfit details, both shoulders, or a wide front-facing torso is a three-quarter/front-view failure when the requested camera is side-view.
- Exactly one eye can help a side-view read, but it is not sufficient if the torso, hips, legs or outfit face the camera.
- The best candidate is the one whose body mass, head, torso, hips, limbs and feet follow the red contour overlay most closely while preserving the identity primer.
- Penalize candidates that are prettier but stand outside the target contour or read less like the requested action.
- Do not mark all candidates equal; use the full 0-to-10 score range.

Target keyframe:
- action: {keyframe["action"]}
- phase: {keyframe["phase"]}
- direction: {keyframe["direction"]}
- camera: {keyframe["camera"]}

Positive CLIP prompt:
{prompt["clip"]}

Detailed T5 prompt:
{prompt["t5"]}

Manual acceptance criteria:
{json.dumps(acceptance, ensure_ascii=False)}

Active control conditions:
{json.dumps(conditions, ensure_ascii=False)}

Important scoring rules:
- Every score is an integer or decimal from 0 to 10.
- 10 means excellent, 7 means usable with concerns, 5 means weak, 0 means unusable.
- Do not assign score 1 when your evidence says the candidate is good.
- rank_recommendation is a local quality band where 1 means top-tier, 2 means usable, 3 means weak, 4 means reject-tier.

Return valid JSON only. The JSON object must contain:
- candidate: exactly "{candidate_name}"
- pass: boolean
- rank_recommendation: integer 1 to 4
- hard_rejects: object with booleans for front_or_three_quarter_view, two_eyes_visible, wrong_direction, cropped_feet, footwear_changed, hairstyle_changed, outfit_changed, pose_not_requested_action, severe_limb_error
- scores: object with numeric 0-to-10 values for condition_adherence, side_profile, pose_match, contour_match, identity_preservation, outfit_preservation, artifact_quality, overall
- evidence: object with condition_match string, identity_match string, concerns string array
Do not wrap the JSON in Markdown code fences.
The evidence.condition_match and evidence.identity_match fields must be strings, not arrays.
"""


def _parse_candidate_judgment(raw_text: str) -> CandidateJudgment:
    data = _json_from_vlm_response(raw_text)
    try:
        return CandidateJudgment.model_validate(data)
    except ValidationError as error:
        raise KeyframeJudgeError(f"Judge returned invalid candidate JSON: {error}") from error


def _json_from_vlm_response(raw_text: str) -> dict[str, Any]:
    try:
        return json_object_from_vlm_response(raw_text)
    except VlmJsonError as error:
        raise KeyframeJudgeError(str(error)) from error


def _save_contour_overlay(candidate_path: Path, contour_path: Path, output_path: Path) -> None:
    with Image.open(candidate_path) as candidate_image, Image.open(contour_path) as contour_image:
        candidate = candidate_image.convert("RGB")
        contour = contour_image.convert("L").resize(candidate.size, Image.Resampling.NEAREST)
        overlay = Image.new("RGBA", candidate.size, (255, 0, 0, 0))
        alpha = contour.point(lambda value: 210 if value > 32 else 0)
        overlay.putalpha(alpha)
        composed = Image.alpha_composite(candidate.convert("RGBA"), overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path)


def _semantic_gate(candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = []
    blocked = []
    for candidate in candidate_results:
        judgment = candidate["judgment"]
        hard_rejects = [
            name
            for name, rejected in judgment["hard_rejects"].items()
            if rejected
        ]
        if judgment["pass"] and not hard_rejects:
            passed.append(candidate["candidate"])
        else:
            blocked.append(
                {
                    "candidate": candidate["candidate"],
                    "pass": judgment["pass"],
                    "hard_rejects": hard_rejects,
                }
            )
    return {
        "passed": passed,
        "blocked": blocked,
        "usable_for_auto_select": False,
        "selection_owner": "condition_score",
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()




def _git_commit(project_root: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
    ).strip()
