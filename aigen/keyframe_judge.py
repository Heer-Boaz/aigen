from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, ValidationError


DEFAULT_JUDGE_ID = "qwen2.5-vl-7b"
DEFAULT_JUDGE_REPO_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_JUDGE_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
DEFAULT_JUDGE_QUANTIZATION = "bitsandbytes-8bit"
DEFAULT_MAX_PIXELS = 512 * 28 * 28
DEFAULT_MIN_PIXELS = 256 * 28 * 28
JUDGE_SCHEMA_VERSION = 1
DEFAULT_CALIBRATION_FIXTURE = Path("judge_fixtures/ai46_walk_contact_640x960_ref384_seed_sweep.json")
MAX_FULL_PAIRWISE_CANDIDATES = 16


class KeyframeJudgeError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class HardRejects(StrictModel):
    front_or_three_quarter_view: bool
    two_eyes_visible: bool
    wrong_direction: bool
    cropped_feet: bool
    missing_boots: bool
    long_hair: bool
    wrong_outfit: bool
    pose_not_walk_contact: bool
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


class PairwiseComparison(StrictModel):
    candidate_a: str
    candidate_b: str
    winner: str
    evidence: str


class PairwiseRanking(StrictModel):
    ordered_candidates: list[str]
    winner: str
    evidence: str
    comparisons: list[PairwiseComparison] = Field(default_factory=list)


class OrderConstraint(StrictModel):
    better: str
    worse: str


class CalibrationFixture(StrictModel):
    schema_version: int
    id: str
    expected_order_constraints: list[OrderConstraint]
    expected_rejects: dict[str, list[str]]


@dataclass(frozen=True)
class KeyframeJudgeConfig:
    judge_id: str
    model: Path
    repo_id: str
    revision: str
    dtype: str
    attention_impl: str
    quantization: str
    min_pixels: int
    max_pixels: int
    max_new_tokens: int
    temperature: float
    pairwise_top_k: int


class QwenKeyframeJudge:
    def __init__(self, config: KeyframeJudgeConfig) -> None:
        _validate_local_qwen_model(config)

        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = _torch_dtype(torch, config.dtype)
        quantization_config = _quantization_config(torch, config)
        device_map = {"": 0} if quantization_config else "auto"
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.model.as_posix(),
                torch_dtype=dtype,
                attn_implementation=config.attention_impl,
                device_map=device_map,
                quantization_config=quantization_config,
                local_files_only=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                config.model.as_posix(),
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                local_files_only=True,
            )
        except OSError as error:
            raise KeyframeJudgeError(f"Failed to load local judge model from {config.model.as_posix()}: {error}") from error
        self.process_vision_info = process_vision_info
        self.config = config
        self.torch = torch

    def judge_candidate(self, prompt: str, image_paths: list[Path]) -> str:
        return self._generate(prompt, image_paths)

    def rank_candidates(self, prompt: str, image_paths: list[Path]) -> str:
        return self._generate(prompt, image_paths)

    def _generate(self, prompt: str, image_paths: list[Path]) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": path.as_posix(),
                        "min_pixels": self.config.min_pixels,
                        "max_pixels": self.config.max_pixels,
                    }
                    for path in image_paths
                ]
                + [{"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(next(self.model.parameters()).device)
        generate_kwargs: dict[str, Any] = {"max_new_tokens": self.config.max_new_tokens}
        if self.config.temperature > 0.0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.config.temperature
        else:
            generate_kwargs["do_sample"] = False
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
        trimmed_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        return self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def judge_keyframe_run(
    run_dir: Path,
    config: KeyframeJudgeConfig,
    *,
    project_root: Path,
    runner: Any | None = None,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.resolve()
    result = _read_json(resolved_run_dir / "result.json")
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

    active_runner = runner if runner else QwenKeyframeJudge(config)
    candidate_results = []
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

    initial_order = _rank_candidate_names(candidate_results)
    pairwise = _run_pairwise_ranking(
        active_runner,
        config,
        effective_config,
        assets,
        candidate_results,
        initial_order,
        prompts_dir,
        raw_dir,
    )
    final_order = _final_order(initial_order, pairwise)
    payload = {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "status": "completed",
        "run_dir": resolved_run_dir.as_posix(),
        "job_id": result["job_id"],
        "git_commit": _git_commit(project_root),
        "source_result_sha256": _sha256_file(resolved_run_dir / "result.json"),
        "judge": _judge_config_json(config),
        "candidates": candidate_results,
        "ranking": {
            "initial": initial_order,
            "final": final_order,
            "pairwise": pairwise.model_dump(mode="json") if pairwise else None,
        },
    }
    _write_json(judge_dir / "judge.json", payload)
    return payload


def calibrate_keyframe_judge(run_dir: Path, *, judge_id: str, fixture_path: Path) -> dict[str, Any]:
    judge_dir = run_dir.resolve() / "judge" / judge_id
    judge_result = _read_json(judge_dir / "judge.json")
    fixture = _load_calibration_fixture(fixture_path)
    final_order = judge_result["ranking"]["final"]
    candidates_by_name = {candidate["candidate"]: candidate for candidate in judge_result["candidates"]}
    checks = []
    for constraint in fixture.expected_order_constraints:
        passed = final_order.index(constraint.better) < final_order.index(constraint.worse)
        checks.append(
            {
                "type": "order",
                "better": constraint.better,
                "worse": constraint.worse,
                "passed": passed,
            }
        )
    for candidate_name, reject_names in fixture.expected_rejects.items():
        hard_rejects = candidates_by_name[candidate_name]["judgment"]["hard_rejects"]
        for reject_name in reject_names:
            checks.append(
                {
                    "type": "hard_reject",
                    "candidate": candidate_name,
                    "reject": reject_name,
                    "passed": bool(hard_rejects[reject_name]),
                }
            )
    passed = all(check["passed"] for check in checks)
    payload = {
        "schema_version": 1,
        "status": "completed",
        "judge_status": "calibrated" if passed else "uncalibrated",
        "usable_for_auto_select": passed,
        "fixture": fixture.model_dump(mode="json"),
        "judge": judge_id,
        "run_dir": run_dir.resolve().as_posix(),
        "checks": checks,
    }
    _write_json(judge_dir / "calibration.json", payload)
    return payload


def select_keyframe_run(
    run_dir: Path,
    *,
    judge_id: str,
    top: int,
    allow_uncalibrated: bool,
) -> dict[str, Any]:
    if top < 1:
        raise KeyframeJudgeError("select --top must be at least 1")
    judge_dir = run_dir.resolve() / "judge" / judge_id
    judge_result = _read_json(judge_dir / "judge.json")
    calibration = _read_calibration(judge_dir)
    if not allow_uncalibrated and not calibration.get("usable_for_auto_select", False):
        raise KeyframeJudgeError(
            "Refusing automatic selection: judge is uncalibrated on golden fixture. "
            "Run `aigen keyframes judge-calibrate` or pass --allow-uncalibrated."
        )
    candidates_by_name = {candidate["candidate"]: candidate for candidate in judge_result["candidates"]}
    ordered = [candidates_by_name[name] for name in judge_result["ranking"]["final"]]
    if top > len(ordered):
        raise KeyframeJudgeError(f"select --top {top} exceeds candidate count {len(ordered)}")
    selected = ordered[:top]
    rejected = ordered[top:]
    _write_json(judge_dir / "selected.json", {"schema_version": 1, "selected": selected})
    _write_json(judge_dir / "rejected.json", {"schema_version": 1, "rejected": rejected})
    _save_ranked_sheet(ordered, judge_dir / "ranked_contact_sheet.png", image_key="image")
    _save_ranked_sheet(ordered, judge_dir / "condition_overlay_ranked.png", image_key="overlay")
    payload = {
        "schema_version": 1,
        "status": "completed",
        "judge": judge_id,
        "run_dir": run_dir.resolve().as_posix(),
        "calibration": calibration,
        "top": top,
        "selected": [candidate["candidate"] for candidate in selected],
        "rejected": [candidate["candidate"] for candidate in rejected],
        "outputs": {
            "selected": (judge_dir / "selected.json").as_posix(),
            "rejected": (judge_dir / "rejected.json").as_posix(),
            "ranked_contact_sheet": (judge_dir / "ranked_contact_sheet.png").as_posix(),
            "condition_overlay_ranked": (judge_dir / "condition_overlay_ranked.png").as_posix(),
        },
    }
    if selected:
        payload["best"] = selected[0]["candidate"]
    return payload


def _candidate_image_paths(assets: dict[str, Any], candidate_path: Path, overlay_path: Path) -> list[Path]:
    paths = [
        Path(assets["reference"]["path"]),
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
1. Original character reference.
2. Generated candidate image named {candidate_name}.
3. Target pose condition.
4. Target contour condition.
5. Boundary mask if present.
6. Candidate image with the target contour overlaid in red.

Do not choose the prettiest image. Judge whether the candidate follows the control conditions while preserving the character.

Priority order:
1. Condition adherence.
2. Strict side-profile camera.
3. Character identity and outfit preservation.
4. Artifact and aesthetic quality.

Hard reject if:
- The character is front view or three-quarter view.
- Both eyes are clearly visible.
- The character faces or moves in the wrong direction.
- The feet are cropped.
- Boots are missing or replaced by different footwear.
- The short bob becomes long hair or a ponytail.
- The outfit no longer matches the reference.
- The pose no longer reads as the requested action.
- There is a severe hand, arm, leg, or boot error that breaks the keyframe.

Strict side-profile definition:
- The head, torso, hips, legs and boots must read as a left-facing platformer side view.
- A visible full chest, full shirt front, centered tie, both jacket lapels, both shoulders, or a wide front-facing torso is a three-quarter/front-view failure.
- Exactly one eye is necessary but not sufficient. A one-eye image can still fail if the torso or outfit faces the camera.
- The best candidate is the one whose body mass, head, torso, hips, legs and boots follow the red contour overlay most closely while preserving the AI46 outfit.
- Penalize candidates that are prettier but stand outside the target contour or read less like the target walk-contact pose.
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
- hard_rejects: object with booleans for front_or_three_quarter_view, two_eyes_visible, wrong_direction, cropped_feet, missing_boots, long_hair, wrong_outfit, pose_not_walk_contact, severe_limb_error
- scores: object with numeric 0-to-10 values for condition_adherence, side_profile, pose_match, contour_match, identity_preservation, outfit_preservation, artifact_quality, overall
- evidence: object with condition_match string, identity_match string, concerns string array
Do not wrap the JSON in Markdown code fences.
The evidence.condition_match and evidence.identity_match fields must be strings, not arrays.
"""


def _pairwise_prompt(candidate_a: str, candidate_b: str, effective_config: dict[str, Any]) -> str:
    keyframe = effective_config["keyframe"]
    return f"""You are doing a strict A/B comparison for platformer keyframe QA.

You will receive these images in order:
1. Target contour condition.
2. A side-by-side red-contour overlay sheet. #1 is {candidate_a}; #2 is {candidate_b}.
3. A side-by-side full candidate sheet with the same labels. #1 is {candidate_a}; #2 is {candidate_b}.
4. Original character reference for identity tiebreak only.

Choose exactly one winner. Judge only by condition adherence first, then strict side profile, then identity and outfit preservation. Do not reward an image merely because it is prettier.

Strict side-profile means the torso is narrow and sideways. Penalize a candidate if the full shirt front, centered tie, both jacket lapels, both shoulders, or a broad chest face the camera, even when only one eye is visible.
Use the red contour overlays as the main evidence. Rank the candidate whose head, torso, hips, legs and boots best fit the contour and walk-contact silhouette above a prettier but less controlled image.
Do not choose a tie.

Target:
- action: {keyframe["action"]}
- phase: {keyframe["phase"]}
- direction: {keyframe["direction"]}
- camera: {keyframe["camera"]}

Candidates:
{json.dumps([candidate_a, candidate_b], ensure_ascii=False)}

Return valid JSON only with exactly this shape:
{{
  "candidate_a": "{candidate_a}",
  "candidate_b": "{candidate_b}",
  "winner": "<candidate_a or candidate_b>",
  "evidence": "short factual reason"
}}
Do not wrap the JSON in Markdown code fences."""


def _run_pairwise_ranking(
    runner: Any,
    config: KeyframeJudgeConfig,
    effective_config: dict[str, Any],
    assets: dict[str, Any],
    candidate_results: list[dict[str, Any]],
    initial_order: list[str],
    prompts_dir: Path,
    raw_dir: Path,
) -> PairwiseRanking | None:
    top_names = _pairwise_candidate_names(candidate_results, initial_order, config.pairwise_top_k)
    if len(top_names) < 2:
        return None
    candidates_by_name = {candidate["candidate"]: candidate for candidate in candidate_results}
    pairwise_dir = prompts_dir.parent / "pairwise"
    pairwise_dir.mkdir(parents=True, exist_ok=True)
    wins = {name: 0 for name in top_names}
    comparisons = []
    for index, candidate_a in enumerate(top_names):
        for candidate_b in top_names[index + 1 :]:
            pair = [candidates_by_name[candidate_a], candidates_by_name[candidate_b]]
            slug = f"{candidate_a}__vs__{candidate_b}"
            pairwise_candidate_sheet = pairwise_dir / f"{slug}_candidates.png"
            pairwise_overlay_sheet = pairwise_dir / f"{slug}_overlays.png"
            _save_ranked_sheet(pair, pairwise_candidate_sheet, image_key="image")
            _save_ranked_sheet(pair, pairwise_overlay_sheet, image_key="overlay")
            image_paths = [
                Path(assets["contour"]["path"]),
                pairwise_overlay_sheet,
                pairwise_candidate_sheet,
                Path(assets["reference"]["path"]),
            ]
            prompt = _pairwise_prompt(candidate_a, candidate_b, effective_config)
            (prompts_dir / f"{slug}.txt").write_text(prompt, encoding="utf-8")
            raw_text = runner.rank_candidates(prompt, image_paths)
            (raw_dir / f"{slug}.json").write_text(raw_text + "\n", encoding="utf-8")
            comparison = _parse_pairwise_comparison(raw_text)
            if comparison.candidate_a != candidate_a or comparison.candidate_b != candidate_b:
                raise KeyframeJudgeError(f"Pairwise judge changed candidate labels for {slug}")
            if comparison.winner not in (candidate_a, candidate_b):
                raise KeyframeJudgeError(f"Pairwise judge chose an unknown winner for {slug}: {comparison.winner}")
            wins[comparison.winner] += 1
            comparisons.append(comparison)
    ordered = sorted(
        top_names,
        key=lambda name: (
            -wins[name],
            _ranking_key(candidates_by_name[name]["judgment"]),
        ),
    )
    return PairwiseRanking(
        ordered_candidates=ordered,
        winner=ordered[0],
        evidence=f"Pairwise tournament wins: {wins}",
        comparisons=comparisons,
    )


def _parse_candidate_judgment(raw_text: str) -> CandidateJudgment:
    data = _json_from_vlm_response(raw_text)
    _canonicalize_evidence(data)
    try:
        return CandidateJudgment.model_validate(data)
    except ValidationError as error:
        raise KeyframeJudgeError(f"Judge returned invalid candidate JSON: {error}") from error


def _parse_pairwise_ranking(raw_text: str) -> PairwiseRanking:
    data = _json_from_vlm_response(raw_text)
    try:
        return PairwiseRanking.model_validate(data)
    except ValidationError as error:
        raise KeyframeJudgeError(f"Judge returned invalid pairwise JSON: {error}") from error


def _parse_pairwise_comparison(raw_text: str) -> PairwiseComparison:
    data = _json_from_vlm_response(raw_text)
    try:
        return PairwiseComparison.model_validate(data)
    except ValidationError as error:
        raise KeyframeJudgeError(f"Judge returned invalid pairwise comparison JSON: {error}") from error


def _pairwise_candidate_names(
    candidate_results: list[dict[str, Any]],
    initial_order: list[str],
    pairwise_top_k: int,
) -> list[str]:
    candidates_by_name = {candidate["candidate"]: candidate for candidate in candidate_results}
    eligible = [
        name
        for name in initial_order
        if not any(candidates_by_name[name]["judgment"]["hard_rejects"].values())
    ]
    if len(eligible) <= MAX_FULL_PAIRWISE_CANDIDATES:
        return eligible
    return eligible[:pairwise_top_k]


def _json_from_vlm_response(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise KeyframeJudgeError("Judge returned an unterminated Markdown JSON block")
        text = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise KeyframeJudgeError(f"Judge returned non-JSON output: {error}") from error
    if not isinstance(data, dict):
        raise KeyframeJudgeError("Judge returned JSON that is not an object")
    return data


def _canonicalize_evidence(data: dict[str, Any]) -> None:
    evidence = data.get("evidence")
    if not isinstance(evidence, dict):
        return
    for key in ("condition_match", "identity_match"):
        value = evidence.get(key)
        if isinstance(value, list):
            evidence[key] = " ".join(str(item) for item in value)


def _load_calibration_fixture(path: Path) -> CalibrationFixture:
    try:
        return CalibrationFixture.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as error:
        raise KeyframeJudgeError(f"Invalid judge calibration fixture {path.as_posix()}: {error}") from error


def _read_calibration(judge_dir: Path) -> dict[str, Any]:
    calibration_path = judge_dir / "calibration.json"
    if not calibration_path.exists():
        return {
            "judge_status": "uncalibrated",
            "usable_for_auto_select": False,
            "reason": "missing calibration.json",
        }
    return _read_json(calibration_path)


def _rank_candidate_names(candidate_results: list[dict[str, Any]]) -> list[str]:
    return [
        candidate["candidate"]
        for candidate in sorted(
            candidate_results,
            key=lambda candidate: _ranking_key(candidate["judgment"]),
        )
    ]


def _ranking_key(judgment: dict[str, Any]) -> tuple[Any, ...]:
    scores = judgment["scores"]
    return (
        not judgment["pass"],
        sum(1 for rejected in judgment["hard_rejects"].values() if rejected),
        judgment["rank_recommendation"],
        -scores["condition_adherence"],
        -scores["side_profile"],
        -scores["pose_match"],
        -scores["contour_match"],
        -scores["identity_preservation"],
        -scores["outfit_preservation"],
        -scores["artifact_quality"],
        -scores["overall"],
    )


def _final_order(initial_order: list[str], pairwise: PairwiseRanking | None) -> list[str]:
    if not pairwise:
        return initial_order
    ranked = list(pairwise.ordered_candidates)
    ranked.extend(name for name in initial_order if name not in ranked)
    return ranked


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


def _save_ranked_sheet(candidates: list[dict[str, Any]], output_path: Path, *, image_key: str) -> None:
    images = []
    for rank, candidate in enumerate(candidates, start=1):
        with Image.open(candidate[image_key]) as image:
            images.append((rank, candidate, image.convert("RGB").copy()))
    thumb_w = 256
    label_h = 44
    thumb_h = max(1, int(thumb_w * images[0][2].height / images[0][2].width))
    sheet = Image.new("RGB", (thumb_w * len(images), thumb_h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (rank, candidate, image) in enumerate(images):
        x = index * thumb_w
        sheet.paste(image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, label_h))
        score = candidate["judgment"]["scores"]["condition_adherence"]
        draw.text((x + 8, 8), f"#{rank} {candidate['candidate']}", fill="black")
        draw.text((x + 8, 24), f"condition {score:g}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _judge_config_json(config: KeyframeJudgeConfig) -> dict[str, Any]:
    return {
        "id": config.judge_id,
        "model": config.model.resolve().as_posix(),
        "repo_id": config.repo_id,
        "revision": config.revision,
        "dtype": config.dtype,
        "attention_impl": config.attention_impl,
        "quantization": config.quantization,
        "min_pixels": config.min_pixels,
        "max_pixels": config.max_pixels,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "pairwise_top_k": config.pairwise_top_k,
    }


def _validate_local_qwen_model(config: KeyframeJudgeConfig) -> None:
    if not config.model.exists():
        raise KeyframeJudgeError(
            "Missing local judge model. Download "
            f"{config.repo_id} to {config.model.as_posix()} before running keyframe judging."
        )
    config_path = config.model / "config.json"
    if not config_path.exists():
        raise KeyframeJudgeError(f"Local judge model is incomplete; missing {config_path.as_posix()}")
    if not any(config.model.glob("*.safetensors")):
        raise KeyframeJudgeError(
            "Local judge model is incomplete; missing safetensors weights in "
            f"{config.model.as_posix()}"
        )


def _quantization_config(torch: Any, config: KeyframeJudgeConfig) -> Any | None:
    if config.quantization == "none":
        return None
    from transformers import BitsAndBytesConfig

    if config.quantization == "bitsandbytes-8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if config.quantization != "bitsandbytes-4bit":
        raise KeyframeJudgeError(f"Unknown judge quantization: {config.quantization}")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=_torch_dtype(torch, config.dtype),
    )


def _torch_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    return getattr(torch, dtype)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise KeyframeJudgeError(f"Missing keyframe run artifact: {path.as_posix()}") from error
    except json.JSONDecodeError as error:
        raise KeyframeJudgeError(f"Invalid JSON artifact: {path.as_posix()}: {error}") from error


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(project_root: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
    ).strip()
