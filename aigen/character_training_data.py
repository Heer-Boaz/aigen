from __future__ import annotations

import shutil
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aigen.character_view_models import (
    CharacterViewError,
    CharacterViewBankSpec,
    ImageAssetSpec,
    ViewBankCharacterSpec,
    ViewBankEntryAcceptanceSpec,
    ViewBankEntrySpec,
    ViewBankViewSpec,
    load_character_view_bank,
)
from aigen.character_view_training_validation import validate_lora_training_image
from aigen.generation.kontext_pose_control import CharacterKontextPoseSession
from aigen.generation.runtime_diagnostics import cuda_memory_stats, synchronized_time
from aigen.image_assets import image_asset_json
from aigen.keyframe_image_ops import save_contact_sheet
from aigen.keyframe_memory import (
    NvidiaSmiMemorySampler,
    keyframe_vram_plan,
    nvidia_smi_keyframe_preflight,
    planned_token_metadata,
)
from aigen.keyframe_profiles import KeyframeProfile
from aigen.lora_datasets import _perceptual_hash, _save_training_image, _write_metadata
from aigen.manifest_io import resolve_existing_path, write_json, write_json_line
from aigen.progress import StatusReporter
from aigen.vlm_json import VlmJsonError, json_object_from_vlm_response
from aigen.vlm_qwen import QwenVlm, QwenVlmConfig, qwen_vlm_config_json


DEFAULT_VIEW_INTENTS = [
    "front neutral full-body view",
    "left profile neutral full-body view",
    "right profile neutral full-body view",
    "back neutral full-body view",
    "left three-quarter neutral full-body view",
    "right three-quarter neutral full-body view",
    "rear left three-quarter neutral full-body view",
    "rear right three-quarter neutral full-body view",
    "slight left quarter neutral full-body view",
    "slight right quarter neutral full-body view",
    "high-angle front neutral full-body view",
    "low-angle front neutral full-body view",
    "top-down neutral full-body view",
    "bottom-up neutral full-body view",
    "random neutral model-sheet full-body view A",
    "random neutral model-sheet full-body view B",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlannedTrainingView(StrictModel):
    name: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    identity_primer_view: str = Field(min_length=1)
    camera: str = Field(min_length=1)
    pose: str = Field(min_length=1)
    clip: str = Field(min_length=1)
    t5: str = Field(min_length=1)
    acceptance_checks: list[str] = Field(min_length=1)


class TrainingViewPlan(StrictModel):
    identity_caption: str = Field(min_length=1)
    identity_details: dict[str, str]
    views: list[PlannedTrainingView] = Field(min_length=1)


IDENTITY_DETAIL_SLOTS = [
    "subject",
    "proportions",
    "hair",
    "face",
    "upper_clothing",
    "waist_garment",
    "legwear",
    "footwear",
    "accessories",
    "materials",
    "palette",
    "style",
]

TRAINING_VALIDATION_MIN_PIXELS = 128 * 28 * 28
TRAINING_VALIDATION_MAX_PIXELS = 256 * 28 * 28
TRAINING_VALIDATION_MAX_NEW_TOKENS = 320


def run_character_training_data(
    bank_path: Path,
    *,
    output_dir: Path,
    profile: KeyframeProfile,
    judge_config: QwenVlmConfig,
    project_root: Path,
    progress: StatusReporter,
    view_intents: list[str] | None = None,
    seeds_per_view: int = 2,
    seed_start: int = 1,
    width: int = 576,
    height: int = 864,
    steps: int = 24,
    guidance_scale: float = 2.5,
    reference_max_area: int = 294912,
    max_sequence_length: int = 128,
    trigger_token: str = "ai51char",
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_dir.exists():
        if not overwrite:
            raise CharacterViewError(f"Output exists and overwrite=false: {output_dir.as_posix()}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    progress.phase("load view bank")
    bank = load_character_view_bank(bank_path)
    bank_dir = bank_path.parent
    intents = view_intents or DEFAULT_VIEW_INTENTS
    progress.phase("plan training views")
    plan = _plan_training_views(bank, bank_dir, intents, judge_config, output_dir)
    write_json(output_dir / "training_plan.json", plan.model_dump(mode="json"))

    progress.phase("write blank control")
    blank_control_path = output_dir / "blank_control.png"
    Image.new("RGB", (width, height), (255, 255, 255)).save(blank_control_path)
    vram_preflight = _generation_preflight(bank, plan, width, height, reference_max_area, max_sequence_length)
    memory_sampler = NvidiaSmiMemorySampler(nvidia_smi_keyframe_preflight(vram_preflight))
    candidate_outputs: list[dict[str, Any]] = []
    generation_memory: dict[str, Any] = {}
    generation_timings: dict[str, Any] = {}
    memory_sampler.start()
    try:
        progress.phase("load generation models")
        session = CharacterKontextPoseSession(
            profile.model,
            profile.controlnet_model,
            dtype=profile.dtype,
            nunchaku_transformer_model=profile.nunchaku_transformer_model,
            attention_impl=profile.attention_impl,
            pipeline_cpu_offload=profile.pipeline_cpu_offload,
            nunchaku_layer_offload=profile.nunchaku_layer_offload,
            vae_tiling=profile.vae_tiling,
        )
        try:
            torch = session.torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats("cuda")
            total_start = synchronized_time(torch)
            for view_index, planned_view in enumerate(plan.views, start=1):
                progress.phase(f"prepare {planned_view.name} ({view_index}/{len(plan.views)})")
                primer = _view_entry(bank, planned_view.identity_primer_view)
                view_seed_start = seed_start + (view_index - 1) * seeds_per_view
                prepared = session.prepare(
                    reference_image=resolve_existing_path(primer.image.path, bank_dir),
                    pose_image=blank_control_path,
                    prompt=planned_view.clip,
                    t5_prompt=planned_view.t5,
                    negative_prompt=None,
                    true_cfg_scale=1.0,
                    width=width,
                    height=height,
                    reference_max_area=reference_max_area,
                    max_sequence_length=max_sequence_length,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=view_seed_start,
                )
                session.pipeline.maybe_free_model_hooks()
                denoised = []
                for seed_offset in range(seeds_per_view):
                    seed = view_seed_start + seed_offset
                    name = f"{_slug(planned_view.name)}__seed_{seed:03d}"
                    progress.phase(f"denoise {name}")
                    result = session.pipeline.denoise_prepared(
                        prepared,
                        name=name,
                        seed=seed,
                        controlnet_conditioning_scale=0.0,
                        control_guidance_start=0.0,
                        control_guidance_end=0.0,
                        control_conditions=[],
                        show_progress=progress.renders_live,
                    )
                    denoised.append(replace(result, latents=result.latents.detach().cpu()))
                    del result
                session.pipeline.maybe_free_model_hooks()
                progress.phase(f"decode {planned_view.name}")
                images, decode_ms = session.decode_many(prepared, denoised, chunk_size=1)
                for image, result in zip(images, denoised, strict=True):
                    candidate_path = output_dir / "candidates" / _slug(planned_view.name) / f"{result.name}.png"
                    candidate_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(candidate_path)
                    candidate_outputs.append(
                        {
                            "name": result.name,
                            "view": planned_view.name,
                            "intent": planned_view.intent,
                            "seed": result.seed,
                            "path": candidate_path.as_posix(),
                            "decode_ms": decode_ms,
                            "timings_ms": result.timings_ms,
                        }
                    )
                del images, denoised, prepared
            generation_memory = cuda_memory_stats(torch, "cuda") | memory_sampler.stop()
            generation_timings = {"total_ms": (synchronized_time(torch) - total_start) * 1000}
        finally:
            session.close()
    finally:
        sampled_memory = memory_sampler.stop()

    if not generation_memory:
        generation_memory = sampled_memory

    if candidate_outputs:
        save_contact_sheet(candidate_outputs, output_dir / "contact_sheet.png", thumb_width=192, label_x=8)

    progress.phase("validate candidates")
    validation_result = _validate_training_candidates(
        bank,
        plan,
        candidate_outputs,
        judge_config,
        output_dir,
        bank_dir,
        trigger_token,
        progress,
    )

    result = {
        "status": "completed",
        "kind": "character-training-data-result",
        "character": bank.character.id,
        "plan": (output_dir / "training_plan.json").as_posix(),
        "candidate_count": len(candidate_outputs),
        "accepted_count": len(validation_result["accepted"]),
        "rejected_count": len(validation_result["rejected"]),
        "output": {
            "directory": output_dir.as_posix(),
            "contact_sheet": (output_dir / "contact_sheet.png").as_posix() if candidate_outputs else None,
            "accepted_contact_sheet": validation_result["accepted_contact_sheet"],
            "dataset": validation_result["dataset_directory"],
            "view_bank": validation_result["view_bank"],
            "manifest": (output_dir / "result.json").as_posix(),
        },
        "generation": {
            "profile": profile.name,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "reference_max_area": reference_max_area,
            "max_sequence_length": max_sequence_length,
            "timings_ms": generation_timings,
            "memory": generation_memory,
            "git_commit": _git_commit(project_root),
        },
        "planner": {
            "judge": qwen_vlm_config_json(judge_config),
            "identity_caption": plan.identity_caption,
            "identity_details": plan.identity_details,
        },
        "candidates": candidate_outputs,
        "accepted": validation_result["accepted"],
        "rejected": validation_result["rejected"],
    }
    write_json(output_dir / "result.json", result)
    return result


def _plan_training_views(
    bank: CharacterViewBankSpec,
    bank_dir: Path,
    intents: list[str],
    config: QwenVlmConfig,
    output_dir: Path,
) -> TrainingViewPlan:
    evidence_dir = output_dir / "planner_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    source_reference = resolve_existing_path(bank.character.source_reference.path, bank_dir)
    evidence_paths = [_white_background_copy(source_reference, evidence_dir / "source_reference.png")]
    for name, entry in bank.views.items():
        view_path = resolve_existing_path(entry.image.path, bank_dir)
        if name != "front":
            evidence_paths.append(_white_background_copy(view_path, evidence_dir / f"{_slug(name)}.png"))
        evidence_paths.extend(_identity_detail_crops(view_path, evidence_dir, name).values())
    prompt = _training_plan_prompt(bank, intents, evidence_paths)
    raw_path = output_dir / "training_plan.raw.txt"
    with closing(QwenVlm(config)) as planner:
        raw_text = planner.judge_candidate(prompt, evidence_paths)
    raw_path.write_text(raw_text + "\n", encoding="utf-8")
    try:
        payload = json_object_from_vlm_response(raw_text)
        plan = TrainingViewPlan.model_validate(payload)
    except (VlmJsonError, ValidationError) as error:
        raise CharacterViewError(f"Invalid character training-data plan: {error}") from error
    _validate_training_plan(bank, plan, intents)
    return plan


def _training_plan_prompt(bank: CharacterViewBankSpec, intents: list[str], image_paths: list[Path]) -> str:
    view_lines = "\n".join(f"- {name}: {entry.image.path}" for name, entry in bank.views.items())
    intent_lines = "\n".join(f"- {intent}" for intent in intents)
    image_lines = "\n".join(f"{index}. {path}" for index, path in enumerate(image_paths, start=1))
    identity_note_lines = "\n".join(f"- {note}" for note in bank.character.identity_notes) or "- none"
    identity_slot_lines = ", ".join(IDENTITY_DETAIL_SLOTS)
    return f"""You are planning neutral character identity training images for a FLUX LoRA.

Images are supplied in this order:
{image_lines}

The images are the source of truth for character identity, proportions, clothing, colors and art style.
The output images must teach the same character, not an action pose.
Inspect the images yourself and build all prompts from visible evidence.
Do not invent a different outfit, hairstyle, body type, art style or background.
Keep every planned image full-body, clean, neutral, high quality and suitable for character identity LoRA training.
Every generated view must use a plain light neutral studio background; never plan black, dark, colored, vignetted or atmospheric backgrounds.
Hidden views such as back, top-down and bottom-up are design extrapolations; keep them conservative and consistent with the visible character.

Character id: {bank.character.id}
Accepted identity views:
{view_lines}

Human-approved identity notes:
{identity_note_lines}

Requested view intents:
{intent_lines}

Return JSON only with exactly these top-level keys:
identity_caption, identity_details, views.

identity_caption:
- one identity-only caption for LoRA training;
- describe subject, proportions, hair, face, clothing, colors, footwear, materials and art style;
- describe waist garment, legwear and footwear as separate visible parts;
- include every human-approved identity note as a preserved identity fact;
- if the waist/upper-leg clothing includes a skirt, shorts, skirt-like panel, or bodysuit bottom, name that garment; do not report only a belt when a waist garment is visible;
- do not include action poses, seeds, filenames or trigger tokens.

identity_details:
- object with concrete visual slots derived from the images;
- include exactly these slots: {identity_slot_lines}.
- do not replace a skirt, shorts, socks, stockings or boots with generic pants, belt or lower body.
- a belt is an accessory and belongs in accessories. It must not replace waist_garment when a waist garment is visible.
- if human-approved identity notes mention a waist garment or accessory, the matching identity_details slot must preserve it literally.

views:
- array with exactly one object for every requested view intent and in the same order;
- each object has exactly these keys:
  name, intent, identity_primer_view, camera, pose, clip, t5, acceptance_checks.

For each view object:
- name is a stable snake_case name derived from the intent;
- intent is the original requested intent string;
- identity_primer_view must be one of the accepted identity view names. Use an accepted profile view for requested profile or quarter views when one exists; do not default to front for side/profile requests.
- camera and pose describe the requested neutral view and must explicitly include the directional view words from the intent;
- clip is a short prompt built from the observed identity plus the requested view;
- t5 is a detailed prompt built from the observed identity plus the requested view;
- acceptance_checks lists concrete checks for identity, proportions, outfit, background, quality and requested view.
- clip and t5 must explicitly include the requested camera/view, such as right profile, back view, top-down view, low-angle view or three-quarter view.
- clip and t5 must explicitly require a plain light neutral background.
- clip and t5 must preserve each visible garment category from identity_details.
- clip, t5 and acceptance_checks must include every human-approved identity note as concrete visible identity criteria.

Do not return generic placeholder identity text.
Do not mention internal filenames.
Do not add action, punch, jump, combat, walk cycle or dynamic motion.
Do not wrap the JSON in Markdown code fences."""


def _validate_training_plan(bank: CharacterViewBankSpec, plan: TrainingViewPlan, intents: list[str]) -> None:
    if [view.intent for view in plan.views] != intents:
        raise CharacterViewError("Character training-data plan does not preserve requested view intent order")
    missing_slots = [slot for slot in IDENTITY_DETAIL_SLOTS if slot not in plan.identity_details]
    if missing_slots:
        raise CharacterViewError(f"Character training-data plan omits identity slots: {', '.join(missing_slots)}")
    _validate_identity_notes(bank, plan)
    for view in plan.views:
        if view.identity_primer_view not in bank.views:
            raise CharacterViewError(f"Unknown identity primer view in training-data plan: {view.identity_primer_view}")
        missing_terms = _missing_view_terms(view)
        if missing_terms:
            raise CharacterViewError(
                f"Character training-data plan for {view.intent!r} omits view terms: {', '.join(missing_terms)}"
            )
        if _is_profile_or_quarter(view.intent) and "left_profile" in bank.views and view.identity_primer_view == "front":
            raise CharacterViewError(
                f"Character training-data plan for {view.intent!r} must use an accepted profile primer, not front"
            )


def _validate_identity_notes(bank: CharacterViewBankSpec, plan: TrainingViewPlan) -> None:
    note_terms = [_identity_note_terms(note) for note in bank.character.identity_notes]
    if not note_terms:
        return
    details_text = " ".join([plan.identity_caption, *plan.identity_details.values()])
    for note, terms in zip(bank.character.identity_notes, note_terms, strict=True):
        missing = _missing_terms(details_text, terms)
        if missing:
            raise CharacterViewError(
                f"Character training-data plan omits identity note {note!r}: {', '.join(missing)}"
            )
    for view in plan.views:
        view_text = " ".join([view.clip, view.t5, " ".join(view.acceptance_checks)])
        for note, terms in zip(bank.character.identity_notes, note_terms, strict=True):
            missing = _missing_terms(view_text, terms)
            if missing:
                raise CharacterViewError(
                    f"Character training-data plan for {view.intent!r} omits identity note "
                    f"{note!r}: {', '.join(missing)}"
                )


def _identity_note_terms(note: str) -> list[str]:
    return [
        word
        for word in _view_words(note)
        if word
        not in {
            "a",
            "an",
            "and",
            "as",
            "at",
            "is",
            "of",
            "the",
            "waist",
            "with",
            "wears",
            "garment",
        }
    ]


def _missing_terms(value: str, terms: list[str]) -> list[str]:
    words = set(_view_words(value))
    return [term for term in terms if term not in words]


def _missing_view_terms(view: PlannedTrainingView) -> list[str]:
    required = [
        word
        for word in _view_words(view.intent)
        if word not in {"neutral", "full", "body", "view", "model", "sheet"}
    ]
    haystack = " ".join(
        [
            view.camera,
            view.pose,
            view.clip,
            view.t5,
            " ".join(view.acceptance_checks),
        ]
    ).lower()
    return [word for word in required if word not in haystack]


def _is_profile_or_quarter(intent: str) -> bool:
    words = set(_view_words(intent))
    return bool(words & {"profile", "quarter"})


def _view_words(value: str) -> list[str]:
    return [
        word
        for word in value.lower().replace("-", " ").replace("_", " ").split()
        if word.isalpha()
    ]


def _generation_preflight(
    bank: CharacterViewBankSpec,
    plan: TrainingViewPlan,
    width: int,
    height: int,
    reference_max_area: int,
    max_sequence_length: int,
) -> dict[str, Any]:
    largest_primer = max(
        (_view_entry(bank, view.identity_primer_view).image for view in plan.views),
        key=lambda image: image.width * image.height,
    )
    token_metadata = planned_token_metadata(
        identity_width=largest_primer.width,
        identity_height=largest_primer.height,
        canvas_width=width,
        canvas_height=height,
        reference_max_area=reference_max_area,
        max_sequence_length=max_sequence_length,
    )
    return keyframe_vram_plan(
        canvas_width=width,
        canvas_height=height,
        true_cfg_scale=1.0,
        token_metadata=token_metadata,
    )


def _validate_training_candidates(
    bank: CharacterViewBankSpec,
    plan: TrainingViewPlan,
    candidates: list[dict[str, Any]],
    config: QwenVlmConfig,
    output_dir: Path,
    bank_dir: Path,
    trigger_token: str,
    progress: StatusReporter,
) -> dict[str, Any]:
    accepted = []
    rejected = []
    accepted_bank = CharacterViewBankSpec(
        kind="character-view-bank",
        character=ViewBankCharacterSpec(
            id=bank.character.id,
            source_reference=bank.character.source_reference,
            identity_notes=bank.character.identity_notes,
        ),
        views={},
    )
    source_path = _white_background_copy(
        resolve_existing_path(bank.character.source_reference.path, bank_dir),
        output_dir / "validation_evidence" / "source_reference.png",
    )
    view_by_name = {view.name: view for view in plan.views}
    validator_config = _training_validation_config(config)
    with closing(QwenVlm(validator_config)) as judge:
        for index, candidate in enumerate(candidates, start=1):
            progress.phase(f"validate candidate {index}/{len(candidates)}")
            planned_view = view_by_name[candidate["view"]]
            validation = validate_lora_training_image(
                character_id=bank.character.id,
                source_path=source_path,
                candidate_path=Path(candidate["path"]),
                view_name=candidate["view"],
                view={
                    "name": planned_view.name,
                    "camera": planned_view.camera,
                    "pose": planned_view.pose,
                    "intent": planned_view.intent,
                    "identity_notes": bank.character.identity_notes,
                    "acceptance_checks": planned_view.acceptance_checks,
                },
                judge=judge,
            )
            record = {
                **candidate,
                "training_validation": validation.model_dump(mode="json"),
                "deterministic_quality": _deterministic_training_quality(Path(candidate["path"])),
            }
            if _training_validation_passes(record["training_validation"], record["deterministic_quality"]):
                accepted.append(record)
                entry_name = candidate["name"]
                accepted_bank.views[entry_name] = ViewBankEntrySpec(
                    view=ViewBankViewSpec(name=planned_view.name, camera=planned_view.camera, pose=planned_view.pose),
                    image=ImageAssetSpec(**image_asset_json(Path(candidate["path"]))),
                    accepted_candidate=candidate["name"],
                    accepted_seed=candidate["seed"],
                    acceptance=ViewBankEntryAcceptanceSpec(
                        manual=planned_view.acceptance_checks,
                        minimum_passing_variants=1,
                    ),
                    training_validation=validation,
                )
            else:
                rejected.append(record)

    view_bank_path = output_dir / "accepted_view_bank.json"
    write_json(view_bank_path, accepted_bank.model_dump(mode="json", exclude_none=True))
    accepted_sheet = output_dir / "accepted_contact_sheet.png"
    if accepted:
        save_contact_sheet(accepted, accepted_sheet, thumb_width=192, label_x=8)
    dataset_dir = output_dir / "dataset"
    _write_validated_dataset(
        accepted,
        dataset_dir,
        character_id=bank.character.id,
        trigger_token=trigger_token,
        identity_caption=plan.identity_caption,
    )
    validations_path = output_dir / "training_validation.jsonl"
    with validations_path.open("w", encoding="utf-8") as stream:
        for record in accepted + rejected:
            write_json_line(stream, record)
    return {
        "accepted": accepted,
        "rejected": rejected,
        "accepted_contact_sheet": accepted_sheet.as_posix() if accepted else None,
        "dataset_directory": dataset_dir.as_posix(),
        "view_bank": view_bank_path.as_posix(),
    }


def _training_validation_config(config: QwenVlmConfig) -> QwenVlmConfig:
    max_pixels = min(config.max_pixels, TRAINING_VALIDATION_MAX_PIXELS)
    return replace(
        config,
        min_pixels=min(config.min_pixels, max_pixels, TRAINING_VALIDATION_MIN_PIXELS),
        max_pixels=max_pixels,
        max_new_tokens=min(config.max_new_tokens, TRAINING_VALIDATION_MAX_NEW_TOKENS),
    )


def _training_validation_passes(validation: dict[str, Any], deterministic_quality: dict[str, Any]) -> bool:
    if deterministic_quality["hard_rejects"]:
        return False
    if not validation["usable_for_lora_training"]:
        return False
    if any(bool(value) for value in validation["hard_rejects"].values()):
        return False
    scores = validation["scores"]
    return all(float(scores[name]) >= 8.0 for name in (
        "identity_preservation",
        "outfit_preservation",
        "hairstyle_preservation",
        "anatomy_quality",
        "background_quality",
        "style_consistency",
        "overall",
    ))


def _deterministic_training_quality(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        border = max(4, min(width, height) // 32)
        strips = [
            rgb.crop((0, 0, width, border)),
            rgb.crop((0, height - border, width, height)),
            rgb.crop((0, 0, border, height)),
            rgb.crop((width - border, 0, width, height)),
        ]
        means = [sum(ImageStat.Stat(strip).mean) / 3.0 for strip in strips]
    border_mean = sum(means) / len(means)
    border_spread = max(means) - min(means)
    hard_rejects = []
    if border_mean < 180.0:
        hard_rejects.append("background_not_light_neutral")
    if border_spread > 35.0:
        hard_rejects.append("background_not_uniform")
    return {
        "border_luma_mean": round(border_mean, 3),
        "border_luma_spread": round(border_spread, 3),
        "hard_rejects": hard_rejects,
    }


def _identity_detail_crops(path: Path, output_dir: Path, view_name: str) -> dict[str, Path]:
    with Image.open(path) as image:
        rgb = _rgb_on_white(image)
        width, height = rgb.size
        crops = {
            "head": rgb.crop((0, 0, width, round(height * 0.34))),
            "torso_waist": rgb.crop((0, round(height * 0.22), width, round(height * 0.68))),
            "lower_body": rgb.crop((0, round(height * 0.42), width, height)),
        }
    outputs = {}
    for label, crop in crops.items():
        target = output_dir / f"{_slug(view_name)}_{label}.png"
        crop.save(target)
        outputs[label] = target
    return outputs


def _white_background_copy(source_path: Path, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        _rgb_on_white(image).save(target_path)
    return target_path


def _rgb_on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    return background.convert("RGB")


def _write_validated_dataset(
    accepted: list[dict[str, Any]],
    output_dir: Path,
    *,
    character_id: str,
    trigger_token: str,
    identity_caption: str,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    records = []
    seen_sha256: set[str] = set()
    for index, record in enumerate(accepted, start=1):
        source_path = Path(record["path"])
        source_sha256 = image_asset_json(source_path)["sha256"]
        if source_sha256 in seen_sha256:
            continue
        seen_sha256.add(source_sha256)
        file_name = f"images/train/{index:04d}_{_slug(character_id)}_{_slug(record['name'])}.png"
        target_path = output_dir / file_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _save_training_image(source_path, target_path)
        caption = _caption(trigger_token, identity_caption, record["view"], record["intent"])
        caption_path = target_path.with_suffix(".txt")
        caption_path.write_text(caption + "\n", encoding="utf-8")
        records.append(
            {
                "file_name": file_name,
                "caption_file": caption_path.relative_to(output_dir).as_posix(),
                "prompt": caption,
                "name": record["name"],
                "split": "train",
                "source_kind": "character_training_data",
                "source_path": source_path.resolve().as_posix(),
                "source_sha256": source_sha256,
                "perceptual_hash": _perceptual_hash(source_path),
                "tags": ["model_validated_identity_view"],
                "image": image_asset_json(target_path),
                "source_metadata": {
                    "view": record["view"],
                    "intent": record["intent"],
                    "seed": record["seed"],
                    "training_validation": record["training_validation"],
                },
            }
        )
    _write_metadata(records, output_dir)
    if records:
        save_contact_sheet(
            [
                {"name": record["name"], "path": (output_dir / record["file_name"]).as_posix()}
                for record in records
            ],
            output_dir / "contact_sheet.png",
            thumb_width=192,
            max_label_chars=24,
        )
    report = {
        "status": "completed",
        "kind": "lora-dataset-result",
        "dataset_id": f"{character_id}.identity.training",
        "character": {"id": character_id, "trigger_token": trigger_token},
        "accepted_image_count": len(records),
        "split_counts": {"train": len(records), "val": 0},
        "output": {
            "directory": output_dir.as_posix(),
            "images": (output_dir / "images").as_posix(),
            "metadata": (output_dir / "metadata.jsonl").as_posix(),
            "captions": (output_dir / "captions.txt").as_posix(),
            "contact_sheet": (output_dir / "contact_sheet.png").as_posix() if records else None,
            "report": (output_dir / "dataset_report.json").as_posix(),
        },
        "records": records,
    }
    write_json(output_dir / "dataset_report.json", report)


def _view_entry(bank: CharacterViewBankSpec, name: str) -> ViewBankEntrySpec:
    try:
        return bank.views[name]
    except KeyError as error:
        raise CharacterViewError(f"View bank has no view: {name}") from error


def _caption(trigger_token: str, identity_caption: str, view: str, intent: str) -> str:
    return ", ".join(_dedupe([trigger_token, identity_caption, _words(view), _words(intent)]))


def _dedupe(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        cleaned = " ".join(value.split())
        if cleaned and cleaned.lower() not in seen:
            result.append(cleaned)
            seen.add(cleaned.lower())
    return result


def _words(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def _slug(value: str) -> str:
    chars = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    return "".join(chars).strip("-") or "image"


def _git_commit(project_root: Path) -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
