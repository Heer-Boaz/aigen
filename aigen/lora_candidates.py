from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from aigen.generation.flux_prompt_encoding import encode_flux_prompts
from aigen.generation.kontext_identity import CharacterKontextIdentitySession
from aigen.generation.runtime_diagnostics import cuda_memory_stats, synchronized_time
from aigen.image_assets import image_asset_json
from aigen.keyframe_image_ops import save_contact_sheet
from aigen.keyframe_memory import NVIDIA_SMI_PREFLIGHT_LIMIT_MB, NvidiaSmiMemorySampler, nvidia_smi_preflight_limit
from aigen.lora_candidate_models import load_lora_candidate_brief
from aigen.lora_candidate_profiles import LoraCandidateProfile
from aigen.lora_canon import (
    CANON_MANIFEST,
    load_lora_canon_manifest,
    lora_canon_images_by_name,
)
from aigen.lora_quality import lora_quality_contract
from aigen.lora_text import caption_contains_token, join_prompt_parts
from aigen.manifest_io import read_json, resolve_output_path, sha256_file, write_json
from aigen.progress import StatusReporter


CANDIDATES_MANIFEST = "candidates.json"
CANDIDATE_EVIDENCE_DIR = "evidence"
REVIEW_DIR = "review"


class LoraCandidateError(RuntimeError):
    pass


def plan_lora_candidates(
    *,
    brief_path: Path,
    progress: StatusReporter,
) -> dict[str, Any]:
    brief_path = brief_path.resolve()
    brief = load_lora_candidate_brief(brief_path)
    canon_dir = resolve_output_path(brief.character.canon, brief_path.parent)
    manifest = load_lora_canon_manifest(canon_dir)
    character = manifest["character"]
    _validate_candidate_prompts_for_generation(brief.candidates, character["trigger_token"])

    output_dir = resolve_output_path(brief.output.directory, brief_path.parent)
    if output_dir.exists():
        if not brief.output.overwrite:
            raise LoraCandidateError(f"Output exists and overwrite=false: {output_dir.as_posix()}")
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "generation_prompts").mkdir(parents=True)

    canon_images = lora_canon_images_by_name(manifest, canon_dir)
    generation = brief.generation
    progress.begin(len(brief.candidates) * generation.seeds_per_candidate, "plan generation prompts")
    candidates = []
    for template in brief.candidates:
        identity_primer = _identity_primer_for_template(template.identity_primer, canon_images)
        for offset in range(generation.seeds_per_candidate):
            seed = generation.seed_start + offset
            name = f"{template.name}_seed_{seed:04d}"
            generation_prompt = _generation_prompt(
                character["identity_prompt"],
                template,
            )
            training_caption = _training_caption(
                character["trigger_token"],
                character["identity_prompt"],
                template,
            )
            prompt_path = output_dir / "generation_prompts" / f"{name}.txt"
            prompt_path.write_text(generation_prompt + "\n", encoding="utf-8")
            candidates.append(
                {
                    "name": name,
                    "status": "planned",
                    "candidate": {
                        "name": template.name,
                        "view": template.view,
                        "pose": template.pose,
                        "framing": template.framing,
                    },
                    "seed": seed,
                    "generation_prompt": generation_prompt,
                    "training_caption": training_caption,
                    "generation_prompt_file": prompt_path.relative_to(output_dir).as_posix(),
                    "identity_primer": identity_primer,
                    "image": {
                        "path": (output_dir / "images" / f"{name}.png").as_posix(),
                        "width": generation.width,
                        "height": generation.height,
                    },
                    "generation": {
                        "screening": True,
                        "width": generation.width,
                        "height": generation.height,
                        "steps": generation.steps,
                    },
                }
            )
            progress.step(name)

    plan = {
        "status": "planned",
        "kind": "lora-candidate-plan",
        "brief_id": brief.id,
        "candidate_brief": {
            "path": brief_path.as_posix(),
            "sha256": sha256_file(brief_path),
        },
        "character": character,
        "canon": {
            "directory": canon_dir.as_posix(),
            "manifest": (canon_dir / CANON_MANIFEST).as_posix(),
        },
        "quality_contract": lora_quality_contract(),
        "counts": {
            "candidate_templates": len(brief.candidates),
            "candidates": len(candidates),
            "seeds_per_candidate": generation.seeds_per_candidate,
        },
        "generation": {
            "width": generation.width,
            "height": generation.height,
            "steps": generation.steps,
            "seed_start": generation.seed_start,
        },
        "candidate_templates": [template.model_dump(mode="json") for template in brief.candidates],
        "candidates": candidates,
        "output": {
            "directory": output_dir.as_posix(),
            "images": (output_dir / "images").as_posix(),
            "generation_prompts": (output_dir / "generation_prompts").as_posix(),
            "manifest": (output_dir / CANDIDATES_MANIFEST).as_posix(),
        },
    }
    write_json(output_dir / CANDIDATES_MANIFEST, plan)
    return {
        "status": plan["status"],
        "kind": plan["kind"],
        "brief_id": plan["brief_id"],
        "candidate_brief": plan["candidate_brief"],
        "character": plan["character"],
        "canon": plan["canon"],
        "quality_contract": plan["quality_contract"],
        "counts": plan["counts"],
        "generation": plan["generation"],
        "candidate_templates": plan["candidate_templates"],
        "output": plan["output"],
    }


def run_lora_candidate_plan(
    candidate_dir: Path,
    *,
    profile: LoraCandidateProfile,
    guidance_scale: float,
    max_sequence_length: int,
    overwrite: bool,
    progress: StatusReporter,
) -> dict[str, Any]:
    candidate_dir = candidate_dir.resolve()
    plan = read_json(candidate_dir / CANDIDATES_MANIFEST, label="LoRA candidate manifest")
    if plan.get("kind") != "lora-candidate-plan":
        raise LoraCandidateError(f"Not a LoRA candidate manifest: {(candidate_dir / CANDIDATES_MANIFEST).as_posix()}")
    existing = [candidate["image"]["path"] for candidate in plan["candidates"] if Path(candidate["image"]["path"]).exists()]
    if existing and not overwrite:
        raise LoraCandidateError(f"Candidate output exists and overwrite=false: {existing[0]}")

    preflight = nvidia_smi_preflight_limit(NVIDIA_SMI_PREFLIGHT_LIMIT_MB)
    memory_sampler = NvidiaSmiMemorySampler(preflight)
    outputs = []
    output_dir = candidate_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_sampler.start()
    memory: dict[str, Any] | None = None
    try:
        progress.phase("encode unique candidate prompts")
        prompt_embeddings, prompt_encode_ms = encode_flux_prompts(
            profile.model,
            prompts=_unique_generation_prompts(plan),
            dtype=profile.dtype,
            max_sequence_length=max_sequence_length,
        )
        progress.phase("load identity generation models")
        session = CharacterKontextIdentitySession(
            profile.model,
            dtype=profile.dtype,
            nunchaku_transformer_model=profile.nunchaku_transformer_model,
            attention_impl=profile.attention_impl,
            vae_tiling=profile.vae_tiling,
        )
        try:
            torch = session.torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats("cuda")
            total_start = synchronized_time(torch)
            progress.begin(len(plan["candidates"]), "generate candidates")
            for candidate in plan["candidates"]:
                image, timings = session.generate(
                    reference_image=Path(candidate["identity_primer"]["path"]),
                    prompt_embedding=prompt_embeddings[candidate["generation_prompt"]],
                    width=candidate["generation"]["width"],
                    height=candidate["generation"]["height"],
                    steps=candidate["generation"]["steps"],
                    guidance_scale=guidance_scale,
                    seed=candidate["seed"],
                    max_sequence_length=max_sequence_length,
                )
                output_path = Path(candidate["image"]["path"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path)
                outputs.append(
                    {
                        "name": candidate["name"],
                        "seed": candidate["seed"],
                        "candidate": candidate["candidate"],
                        "identity_primer": candidate["identity_primer"],
                        "generation_prompt": candidate["generation_prompt"],
                        "training_caption": candidate["training_caption"],
                        "image": image_asset_json(output_path),
                        "timings_ms": timings,
                    }
                )
                progress.step(candidate["name"])
            save_contact_sheet(
                [{"name": output["name"], "path": output["image"]["path"]} for output in outputs],
                candidate_dir / "contact_sheet.png",
                thumb_width=192,
                max_label_chars=24,
            )
            memory = memory_sampler.stop()
            result = {
                "status": "completed",
                "kind": "lora-candidate-generation-result",
                "candidate_manifest": (candidate_dir / CANDIDATES_MANIFEST).as_posix(),
                "profile": {
                    "name": profile.name,
                    "dtype": profile.dtype,
                    "attention_impl": profile.attention_impl,
                    "vae_tiling": profile.vae_tiling,
                    "prompt_encoding": "precomputed_prompt_embeds",
                    "models": {
                        **profile.model_revisions,
                        "kontext": {
                            **profile.model_revisions["kontext"],
                            "path": profile.model,
                        },
                        "nunchaku_transformer": {
                            **profile.model_revisions["nunchaku_transformer"],
                            "path": profile.nunchaku_transformer_model.as_posix(),
                        },
                    },
                },
                "generation": {
                    "guidance_scale": guidance_scale,
                    "max_sequence_length": max_sequence_length,
                },
                "outputs": outputs,
                "timings_ms": {
                    "prompt_encode_ms": prompt_encode_ms,
                    "model_load_ms": session.model_load_ms,
                    "total_ms": (synchronized_time(torch) - total_start) * 1000,
                },
                "memory": cuda_memory_stats(torch, "cuda") | memory,
                "environment": session.environment(),
                "output": {
                    "directory": candidate_dir.as_posix(),
                    "images": output_dir.as_posix(),
                    "contact_sheet": (candidate_dir / "contact_sheet.png").as_posix(),
                    "result": (candidate_dir / "generation_result.json").as_posix(),
                },
            }
            write_json(candidate_dir / "generation_result.json", result)
            return result
        finally:
            session.close()
    finally:
        if memory is None:
            memory_sampler.stop()


def build_lora_candidate_evidence(
    candidate_dir: Path,
    *,
    overwrite: bool,
    progress: StatusReporter,
) -> dict[str, Any]:
    candidate_dir = candidate_dir.resolve()
    plan = read_json(candidate_dir / CANDIDATES_MANIFEST, label="LoRA candidate manifest")
    if plan.get("kind") != "lora-candidate-plan":
        raise LoraCandidateError(f"Not a LoRA candidate manifest: {(candidate_dir / CANDIDATES_MANIFEST).as_posix()}")
    evidence_dir = (candidate_dir / CANDIDATE_EVIDENCE_DIR).resolve()
    if evidence_dir.exists():
        if not overwrite:
            raise LoraCandidateError(f"Output exists and overwrite=false: {evidence_dir.as_posix()}")
        shutil.rmtree(evidence_dir)
    evidence_dir.mkdir(parents=True)

    review_items: list[dict[str, Any]] = []
    rejected_images: list[dict[str, Any]] = []
    existing_images: list[dict[str, str]] = []
    progress.begin(len(plan["candidates"]), "prepare candidate review evidence")
    for candidate in plan["candidates"]:
        path = Path(candidate["image"]["path"])
        name = candidate["name"]
        if not path.exists():
            rejected_images.append(
                _candidate_evidence_item(
                    candidate,
                    status="missing_candidate_image",
                    hard_rejects={"missing_candidate_image": True},
                    evidence={},
                )
            )
            progress.step(name)
            continue
        evidence = _image_evidence(path)
        hard_rejects = {}
        if evidence["image"]["width"] != candidate["image"]["width"] or evidence["image"]["height"] != candidate["image"]["height"]:
            hard_rejects["wrong_dimensions"] = True
        if hard_rejects:
            rejected_images.append(
                _candidate_evidence_item(
                    candidate,
                    status="invalid_candidate_image",
                    hard_rejects=hard_rejects,
                    evidence=evidence,
                )
            )
        else:
            review_items.append(
                _candidate_evidence_item(
                    candidate,
                    status="ready_for_model_judgment",
                    hard_rejects={},
                    evidence=evidence,
                )
            )
            existing_images.append({"name": name, "path": path.as_posix()})
        progress.step(name)

    if existing_images:
        save_contact_sheet(existing_images, evidence_dir / "contact_sheet.png", thumb_width=192, max_label_chars=24)

    report = {
        "status": "completed",
        "kind": "lora-candidate-evidence",
        "candidate_manifest": (candidate_dir / CANDIDATES_MANIFEST).as_posix(),
        "quality_contract": lora_quality_contract(),
        "counts": {
            "candidates": len(plan["candidates"]),
            "review_items": len(review_items),
            "rejected_images": len(rejected_images),
        },
        "review_items": review_items,
        "rejected_images": rejected_images,
        "output": {
            "directory": evidence_dir.as_posix(),
            "review_items": (evidence_dir / "review_items.json").as_posix(),
            "rejected_images": (evidence_dir / "rejected_images.json").as_posix(),
            "contact_sheet": (evidence_dir / "contact_sheet.png").as_posix() if existing_images else None,
            "report": (evidence_dir / "evidence_report.json").as_posix(),
        },
    }
    write_json(evidence_dir / "review_items.json", {"items": review_items})
    write_json(evidence_dir / "rejected_images.json", {"items": rejected_images})
    write_json(evidence_dir / "evidence_report.json", report)
    return report


def review_lora_candidates(
    candidate_dir: Path,
    *,
    accepted_names: list[str],
    approved_by: str,
    overwrite: bool,
    progress: StatusReporter,
) -> dict[str, Any]:
    if not accepted_names:
        raise LoraCandidateError("Candidate review requires at least one --accept NAME")
    candidate_dir = candidate_dir.resolve()
    passed_items_path = candidate_dir / CANDIDATE_EVIDENCE_DIR / "passed.json"
    passed_items = read_json(passed_items_path, label="model-passed LoRA candidate items")["items"]
    if not passed_items:
        raise LoraCandidateError(f"No model-passed LoRA candidates are available: {passed_items_path.as_posix()}")
    by_name = {item["name"]: item for item in passed_items}
    missing = [name for name in accepted_names if name not in by_name]
    if missing:
        raise LoraCandidateError(f"Model-passed evidence has no candidate: {', '.join(missing)}")

    review_dir = (candidate_dir / REVIEW_DIR).resolve()
    if review_dir.exists():
        if not overwrite:
            raise LoraCandidateError(f"Output exists and overwrite=false: {review_dir.as_posix()}")
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True)

    accepted = []
    rejected_human = []
    progress.begin(len(passed_items), "write human review")
    accepted_set = set(accepted_names)
    for item in passed_items:
        if item["name"] in accepted_set:
            accepted_item = dict(item)
            accepted_item["approval"] = {
                "mode": "human_approved_lora_candidate",
                "approved_by": approved_by,
            }
            accepted.append(accepted_item)
        else:
            rejected = dict(item)
            rejected["approval"] = {"mode": "human_rejected_lora_candidate"}
            rejected_human.append(rejected)
        progress.step(item["name"])

    if accepted:
        save_contact_sheet(
            [{"name": item["name"], "path": item["image"]["path"]} for item in accepted],
            review_dir / "accepted_contact_sheet.png",
            thumb_width=192,
            max_label_chars=24,
        )

    quota = _quota_report(accepted)
    result = {
        "status": "completed",
        "kind": "lora-candidate-review",
        "candidate_manifest": (candidate_dir / CANDIDATES_MANIFEST).as_posix(),
        "candidate_evidence": (candidate_dir / CANDIDATE_EVIDENCE_DIR).as_posix(),
        "counts": {
            "model_passed": len(passed_items),
            "accepted": len(accepted),
            "rejected_human": len(rejected_human),
        },
        "accepted": accepted,
        "rejected_human": rejected_human,
        "quota_report": quota,
        "output": {
            "directory": review_dir.as_posix(),
            "accepted": (review_dir / "accepted.json").as_posix(),
            "rejected_human": (review_dir / "rejected_human.json").as_posix(),
            "quota_report": (review_dir / "quota_report.json").as_posix(),
            "accepted_contact_sheet": (review_dir / "accepted_contact_sheet.png").as_posix() if accepted else None,
            "report": (review_dir / "review_report.json").as_posix(),
        },
    }
    write_json(review_dir / "accepted.json", {"kind": "lora-candidate-accepted", "status": "completed", "items": accepted})
    write_json(review_dir / "rejected_human.json", {"items": rejected_human})
    write_json(review_dir / "quota_report.json", quota)
    write_json(review_dir / "review_report.json", result)
    return result


def _identity_primer_for_template(
    primer_name: str,
    canon_images: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if primer_name not in canon_images:
        raise LoraCandidateError(f"Candidate needs missing canon identity primer: {primer_name}")
    return canon_images[primer_name]


def _validate_candidate_prompts_for_generation(candidates: list[Any], trigger_token: str) -> None:
    for candidate in candidates:
        if caption_contains_token(candidate.prompt.positive, trigger_token):
            raise LoraCandidateError(
                f"Candidate {candidate.name} includes LoRA trigger token {trigger_token} in its generation prompt"
            )


def _unique_generation_prompts(plan: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(candidate["generation_prompt"] for candidate in plan["candidates"]))


def _generation_prompt(identity_prompt: str, template: Any) -> str:
    return join_prompt_parts(identity_prompt, template.prompt.positive)


def _training_caption(trigger_token: str, identity_prompt: str, template: Any) -> str:
    return join_prompt_parts(trigger_token, identity_prompt, template.prompt.positive)


def _image_evidence(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    return {
        "image": {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "width": width,
            "height": height,
            "mode": mode,
        },
    }


def _candidate_evidence_item(
    candidate: dict[str, Any],
    *,
    status: str,
    hard_rejects: dict[str, bool],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": candidate["name"],
        "status": status,
        "candidate": candidate["candidate"],
        "seed": candidate["seed"],
        "generation_prompt": candidate["generation_prompt"],
        "training_caption": candidate["training_caption"],
        "identity_primer": candidate["identity_primer"],
        "image": candidate["image"],
        "hard_rejects": hard_rejects,
        "evidence": evidence,
    }


def _quota_report(accepted: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate: dict[str, int] = {}
    by_view: dict[str, int] = {}
    by_pose: dict[str, int] = {}
    by_framing: dict[str, int] = {}
    for item in accepted:
        candidate = item["candidate"]
        by_candidate[candidate["name"]] = by_candidate.get(candidate["name"], 0) + 1
        by_view[candidate["view"]] = by_view.get(candidate["view"], 0) + 1
        by_pose[candidate["pose"]] = by_pose.get(candidate["pose"], 0) + 1
        by_framing[candidate["framing"]] = by_framing.get(candidate["framing"], 0) + 1
    total = len(accepted)
    dominant = [
        {"axis": "candidate", "name": name, "count": count}
        for name, count in by_candidate.items()
        if total and count / total > 0.4
    ]
    dominant.extend(
        {"axis": "view", "name": name, "count": count}
        for name, count in by_view.items()
        if total and count / total > 0.5
    )
    dominant.extend(
        {"axis": "framing", "name": name, "count": count}
        for name, count in by_framing.items()
        if total and count / total > 0.6
    )
    return {
        "accepted": total,
        "by_candidate": by_candidate,
        "by_view": by_view,
        "by_pose": by_pose,
        "by_framing": by_framing,
        "warnings": [
            f"{item['axis']} {item['name']} dominates accepted candidates ({item['count']}/{total})"
            for item in dominant
        ],
    }
