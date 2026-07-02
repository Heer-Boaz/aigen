from __future__ import annotations

import shutil
from contextlib import closing
from pathlib import Path
from typing import Any

from aigen.lora_candidates import CANDIDATE_EVIDENCE_DIR, CANDIDATES_MANIFEST
from aigen.lora_text import caption_contains_token, join_prompt_parts
from aigen.manifest_io import read_json, write_json
from aigen.progress import StatusReporter
from aigen.vlm_qwen import QwenVlm, QwenVlmConfig, qwen_vlm_config_json


CAPTIONS_DIR = "captions"


class LoraCandidateCaptionError(RuntimeError):
    pass


def caption_lora_candidates(
    candidate_dir: Path,
    config: QwenVlmConfig,
    *,
    overwrite: bool,
    progress: StatusReporter,
) -> dict[str, Any]:
    candidate_dir = candidate_dir.resolve()
    plan = read_json(candidate_dir / CANDIDATES_MANIFEST, label="LoRA candidate manifest")
    if plan.get("kind") != "lora-candidate-plan":
        raise LoraCandidateCaptionError(f"Not a LoRA candidate manifest: {(candidate_dir / CANDIDATES_MANIFEST).as_posix()}")
    trigger_token = plan["character"]["trigger_token"]
    evidence_dir = candidate_dir / CANDIDATE_EVIDENCE_DIR
    passed_path = evidence_dir / "passed.json"
    items = read_json(passed_path, label="model-passed LoRA candidate items")["items"]
    if not items:
        raise LoraCandidateCaptionError(f"No model-passed LoRA candidates are ready for captioning: {passed_path.as_posix()}")

    captions_dir = candidate_dir / CAPTIONS_DIR
    captioned_path = evidence_dir / "captioned.json"
    report_path = evidence_dir / "caption_report.json"
    if captions_dir.exists() or captioned_path.exists() or report_path.exists():
        if not overwrite:
            raise LoraCandidateCaptionError(f"Caption output exists in {candidate_dir.as_posix()}")
        if captions_dir.exists():
            shutil.rmtree(captions_dir)
        for path in (captioned_path, report_path):
            if path.exists():
                path.unlink()
    captions_dir.mkdir(parents=True)

    with closing(QwenVlm(config)) as active_runner:
        captioned = []
        progress.begin(len(items), "caption LoRA candidates")
        for item in items:
            name = item["name"]
            raw_text = active_runner.describe_image(_caption_prompt(), [Path(item["image"]["path"])])
            caption = _normalize_caption(raw_text)
            _validate_caption(caption, trigger_token=trigger_token, name=name)
            training_caption = join_prompt_parts(trigger_token, caption)
            caption_path = captions_dir / f"{name}.txt"
            caption_path.write_text(training_caption + "\n", encoding="utf-8")
            captioned.append(
                item
                | {
                    "training_caption": training_caption,
                    "caption": {
                        "model_caption": caption,
                        "file": caption_path.as_posix(),
                    },
                }
            )
            progress.step(name)

        payload = {
            "status": "completed",
            "kind": "lora-candidate-captions",
            "candidate_manifest": (candidate_dir / CANDIDATES_MANIFEST).as_posix(),
            "captioner": qwen_vlm_config_json(config) | {"device_report": active_runner.device_report},
            "counts": {
                "captioned": len(captioned),
            },
            "items": captioned,
            "output": {
                "captions": captions_dir.as_posix(),
                "captioned": captioned_path.as_posix(),
                "report": report_path.as_posix(),
            },
        }
    write_json(captioned_path, {"items": captioned})
    write_json(report_path, payload)
    return payload


def _caption_prompt() -> str:
    return """You are writing a truthful training caption for one character image in a LoRA dataset.

Look only at the supplied image. Write exactly one line of comma-separated descriptive phrases covering:
- the subject's visible appearance: hair, eyes when visible, clothing items actually visible in frame
- camera view and angle, body pose, and expression when the face is visible
- how much of the body the frame shows, described naturally, such as full body, thigh-up, waist-up or close portrait
- the background
- the art style or medium

Rules:
- Describe only what is visible. Never mention body parts or clothing that are outside the frame.
- Plain text only: no JSON, no markdown, no quotes, no line breaks, no numbering.
- Do not start with filler such as "The image shows"; start directly with the first descriptive phrase.
- No quality boosters such as high quality, masterpiece or 8k.
- No character names and no invented trigger words.
"""


def _normalize_caption(raw_text: str) -> str:
    caption = " ".join(raw_text.split()).strip().strip("`\"'")
    return caption.rstrip(" ,.;:")


def _validate_caption(caption: str, *, trigger_token: str, name: str) -> None:
    if len(caption.split()) < 3:
        raise LoraCandidateCaptionError(f"Captioner returned an empty or unusable caption for {name}: {caption!r}")
    if caption_contains_token(caption, trigger_token):
        raise LoraCandidateCaptionError(f"Captioner leaked the trigger token {trigger_token} into the caption for {name}")
