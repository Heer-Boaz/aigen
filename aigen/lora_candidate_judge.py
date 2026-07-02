from __future__ import annotations

import hashlib
import shutil
from contextlib import closing
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aigen.lora_candidates import CANDIDATE_EVIDENCE_DIR, CANDIDATES_MANIFEST
from aigen.lora_quality import lora_quality_contract
from aigen.manifest_io import read_json, sha256_file, write_json
from aigen.progress import StatusReporter
from aigen.vlm_json import VlmJsonError, json_object_from_vlm_response
from aigen.vlm_qwen import QwenVlm, QwenVlmConfig, qwen_vlm_config_json


class LoraCandidateJudgeError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoraCandidateHardRejects(StrictModel):
    wrong_face: bool
    wrong_hair_length_or_color: bool
    childlike_or_chibi_proportions: bool
    wrong_visible_outfit: bool
    missing_visible_required_identity_detail: bool
    deformed_visible_anatomy: bool
    broken_visible_hands_or_feet: bool
    framing_mismatch: bool
    dirty_or_distracting_background: bool
    style_drift: bool
    view_label_mismatch: bool


class LoraCandidateScores(StrictModel):
    identity_preservation: float
    visible_outfit_preservation: float
    visible_anatomy_quality: float
    petite_proportion_preservation: float
    framing_quality: float
    background_quality: float
    style_match: float
    view_pose_match: float
    training_usability: float


class LoraCandidateEvidence(StrictModel):
    identity_match: str
    quality_assessment: str
    concerns: list[str]


class LoraCandidateJudgment(StrictModel):
    candidate: str
    passes: bool = Field(alias="pass")
    hard_rejects: LoraCandidateHardRejects
    scores: LoraCandidateScores
    evidence: LoraCandidateEvidence


class LoraFreeCandidateHardRejects(StrictModel):
    wrong_face: bool
    wrong_hair_length_or_color: bool
    childlike_or_chibi_proportions: bool
    wrong_visible_outfit: bool
    missing_visible_required_identity_detail: bool
    deformed_visible_anatomy: bool
    broken_visible_hands_or_feet: bool
    awkward_crop_or_cutoff: bool
    dirty_or_distracting_background: bool
    style_drift: bool


class LoraFreeCandidateScores(StrictModel):
    identity_preservation: float
    visible_outfit_preservation: float
    visible_anatomy_quality: float
    petite_proportion_preservation: float
    framing_quality: float
    background_quality: float
    style_match: float
    training_usability: float


class LoraFreeCandidateJudgment(StrictModel):
    candidate: str
    passes: bool = Field(alias="pass")
    hard_rejects: LoraFreeCandidateHardRejects
    scores: LoraFreeCandidateScores
    evidence: LoraCandidateEvidence


def judge_lora_candidate_evidence(
    candidate_dir: Path,
    config: QwenVlmConfig,
    *,
    overwrite: bool,
    progress: StatusReporter,
) -> dict[str, Any]:
    candidate_dir = candidate_dir.resolve()
    evidence_dir = candidate_dir / CANDIDATE_EVIDENCE_DIR
    review_items_path = evidence_dir / "review_items.json"
    review_items = read_json(review_items_path, label="LoRA candidate review items")["items"]
    if not review_items:
        raise LoraCandidateJudgeError(f"No LoRA candidates are ready for judging: {review_items_path.as_posix()}")

    prompts_dir = evidence_dir / "judge_prompts"
    raw_dir = evidence_dir / "judge_raw"
    output_files = [
        evidence_dir / "judge.json",
        evidence_dir / "passed.json",
        evidence_dir / "blocked.json",
    ]
    if any(path.exists() for path in output_files + [prompts_dir, raw_dir]):
        if not overwrite:
            raise LoraCandidateJudgeError(f"LoRA candidate judge output exists in {evidence_dir.as_posix()}")
        for path in output_files:
            if path.exists():
                path.unlink()
        if prompts_dir.exists():
            shutil.rmtree(prompts_dir)
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
    prompts_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    with closing(QwenVlm(config)) as active_runner:
        judged = []
        progress.begin(len(review_items), "judge LoRA candidates")
        for item in review_items:
            candidate_name = item["name"]
            prompt = _candidate_prompt(item)
            prompt_path = prompts_dir / f"{candidate_name}.txt"
            raw_path = raw_dir / f"{candidate_name}.json"
            prompt_path.write_text(prompt, encoding="utf-8")
            raw_text = active_runner.judge_candidate(prompt, _candidate_image_paths(item))
            raw_path.write_text(raw_text + "\n", encoding="utf-8")
            judgment = _parse_candidate_judgment(raw_text, free=_is_free_item(item))
            if judgment.candidate != candidate_name:
                raise LoraCandidateJudgeError(f"Judge returned candidate {judgment.candidate}, expected {candidate_name}")
            judged.append(
                {
                    "candidate": candidate_name,
                    "image": item["image"]["path"],
                    "identity_primer": item["identity_primer"],
                    "prompt_sha256": _sha256_text(prompt),
                    "prompt": prompt_path.as_posix(),
                    "raw_response": raw_path.as_posix(),
                    "judgment": judgment.model_dump(mode="json", by_alias=True),
                    "selection": _selection_status(judgment),
                    "source": item,
                }
            )
            progress.step(f"judged {candidate_name}")

        passed = [item for item in judged if item["selection"]["passed"]]
        blocked = [item for item in judged if not item["selection"]["passed"]]
        payload = {
            "status": "completed",
            "kind": "lora-candidate-judge",
            "candidate_manifest": (candidate_dir / CANDIDATES_MANIFEST).as_posix(),
            "candidate_evidence": evidence_dir.as_posix(),
            "quality_contract": lora_quality_contract(),
            "judge": qwen_vlm_config_json(config) | {"device_report": active_runner.device_report},
            "counts": {
                "review_items": len(review_items),
                "passed": len(passed),
                "blocked": len(blocked),
            },
            "candidates": judged,
            "selection_gate": {
                "passed": [item["candidate"] for item in passed],
                "blocked": [
                    {
                        "candidate": item["candidate"],
                        "blockers": item["selection"]["blockers"],
                    }
                    for item in blocked
                ],
                "selection_owner": "human_review_after_model_gate",
            },
            "output": {
                "judge": (evidence_dir / "judge.json").as_posix(),
                "passed": (evidence_dir / "passed.json").as_posix(),
                "blocked": (evidence_dir / "blocked.json").as_posix(),
                "prompts": prompts_dir.as_posix(),
                "raw": raw_dir.as_posix(),
            },
        }
        write_json(evidence_dir / "judge.json", payload)
        write_json(evidence_dir / "passed.json", {"items": [item["source"] | {"model_judgment": item["judgment"]} for item in passed]})
        write_json(
            evidence_dir / "blocked.json",
            {"items": [item["source"] | {"model_judgment": item["judgment"], "blockers": item["selection"]["blockers"]} for item in blocked]},
        )
        return payload


def _candidate_image_paths(item: dict[str, Any]) -> list[Path]:
    return [
        Path(item["identity_primer"]["path"]),
        Path(item["image"]["path"]),
    ]


def _is_free_item(item: dict[str, Any]) -> bool:
    return item["candidate"].get("mode") == "free"


def _candidate_prompt(item: dict[str, Any]) -> str:
    if _is_free_item(item):
        return _free_candidate_prompt(item)
    hard_rejects = lora_quality_contract()["hard_rejects"]
    candidate = item["candidate"]
    return f"""You are a strict QA judge for LoRA training images.

You will receive these images in order:
1. Approved identity primer for the character.
2. Generated candidate image named {item["name"]}.

Decide whether the candidate is canon-worthy training data for the same character.
Do not choose the prettiest image. Do not accept an image just because the pose is useful.
Accept only if the visible candidate preserves identity, visible outfit details, petite proportions, visual style, requested framing and clean background.
Partial-body candidates are valid LoRA training data when their requested framing is partial-body. Do not reject a waist-up, thigh-up or portrait image just because boots, socks or lower legs are outside the frame.
Judge only details that should be visible in the requested framing, but always enforce face, hair color/length, petite balanced proportions, visible outfit accuracy, clean background and style.

Candidate label:
- template: {candidate["name"]}
- requested view: {candidate["view"]}
- requested pose: {candidate["pose"]}
- requested framing: {candidate["framing"]}

Generation prompt:
{item["generation_prompt"]}

Hard reject if any of these are true:
{hard_rejects}

Scoring rules:
- Every score is a number from 0 to 10.
- 10 is excellent, 7 is usable with minor concerns, 5 is weak, 0 is unusable.
- training_usability must be 7 or higher only for images you would actually train a character LoRA on.

Return valid JSON only. The JSON object must contain:
- candidate: exactly "{item["name"]}"
- pass: boolean
- hard_rejects: object with booleans for wrong_face, wrong_hair_length_or_color, childlike_or_chibi_proportions, wrong_visible_outfit, missing_visible_required_identity_detail, deformed_visible_anatomy, broken_visible_hands_or_feet, framing_mismatch, dirty_or_distracting_background, style_drift, view_label_mismatch
- scores: object with numeric values for identity_preservation, visible_outfit_preservation, visible_anatomy_quality, petite_proportion_preservation, framing_quality, background_quality, style_match, view_pose_match, training_usability
- evidence: object with identity_match string, quality_assessment string, concerns string array
Do not wrap the JSON in Markdown code fences.
"""


def _free_candidate_prompt(item: dict[str, Any]) -> str:
    hard_reject_fields = list(LoraFreeCandidateHardRejects.model_fields)
    score_fields = list(LoraFreeCandidateScores.model_fields)
    hard_reject_lines = "\n".join(f"- {field.replace('_', ' ')}" for field in hard_reject_fields)
    return f"""You are a strict QA judge for LoRA training images.

You will receive these images in order:
1. Approved identity primer for the character.
2. Generated candidate image named {item["name"]}.

The candidate was generated without a requested view, pose or framing. Any natural camera angle, pose,
expression and body coverage is acceptable; the image will be captioned from what it actually shows.
Judge only identity and quality. Do not reject an image for its view, pose, expression or for showing
a partial body, as long as the framing looks intentional rather than accidentally cut off.
Judge only details that should be visible in the actual framing, but always enforce face, hair color/length,
petite balanced adult proportions, visible outfit accuracy, clean background and style.
Accept only images you would train a character identity LoRA on.

Generation prompt:
{item["generation_prompt"]}

Hard reject if any of these are true:
{hard_reject_lines}

Scoring rules:
- Every score is a number from 0 to 10.
- 10 is excellent, 7 is usable with minor concerns, 5 is weak, 0 is unusable.
- training_usability must be 7 or higher only for images you would actually train a character LoRA on.

Return valid JSON only. The JSON object must contain:
- candidate: exactly "{item["name"]}"
- pass: boolean
- hard_rejects: object with booleans for {", ".join(hard_reject_fields)}
- scores: object with numeric values for {", ".join(score_fields)}
- evidence: object with identity_match string, quality_assessment string, concerns string array
Do not wrap the JSON in Markdown code fences.
"""


def _parse_candidate_judgment(raw_text: str, *, free: bool) -> LoraCandidateJudgment | LoraFreeCandidateJudgment:
    try:
        data = json_object_from_vlm_response(raw_text)
    except VlmJsonError as error:
        raise LoraCandidateJudgeError(str(error)) from error
    model = LoraFreeCandidateJudgment if free else LoraCandidateJudgment
    try:
        return model.model_validate(data)
    except ValidationError as error:
        raise LoraCandidateJudgeError(f"Judge returned invalid LoRA candidate JSON: {error}") from error


def _selection_status(judgment: LoraCandidateJudgment | LoraFreeCandidateJudgment) -> dict[str, Any]:
    dumped = judgment.model_dump(mode="json", by_alias=True)
    hard_rejects = [name for name, rejected in dumped["hard_rejects"].items() if rejected]
    score_blockers = [name for name, score in dumped["scores"].items() if score < 7.0]
    blockers = hard_rejects + score_blockers
    if not dumped["pass"]:
        blockers.append("judge_failed_candidate")
    return {
        "passed": not blockers,
        "blockers": blockers,
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
