from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from aigen.keyframe_image_ops import save_contact_sheet
from aigen.keyframe_job_models import (
    ControlConditionSpec,
    OutputSpec,
    PromptSpec,
    VariantSpec,
    KeyframeJobSpec,
    load_keyframe_job,
)
from aigen.keyframe_score import DEFAULT_SCORER_ID, KeyframeScoreConfig, score_keyframe_run
from aigen.keyframes import resolve_keyframe_spec
from aigen.manifest_io import read_json, write_json


CONTROL_AUDIT_SCHEMA_VERSION = 1
CONTROL_AUDIT_SCORER_ID = "control-audit"
CONTROL_AUDIT_MINIMAL_PROMPT = (
    "Same character, full body, clean neutral background, preserve outfit and side-view character design."
)
MATERIAL_SCORE_DELTA = 0.05
CONTROL_AUDIT_VARIANT_TIMEOUT_SECONDS = 300


class KeyframeControlAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class ControlAuditVariant:
    name: str
    conditions: list[ControlConditionSpec]


def run_keyframe_control_audit(
    run_dir: Path,
    *,
    project_root: Path,
    seed: int | None = None,
    scorer_id: str = CONTROL_AUDIT_SCORER_ID,
    score_runner: Any | None = None,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.resolve()
    source_result = read_json(resolved_run_dir / "result.json", label="source keyframe result")
    source_config = source_result["effective_config"]
    source_job_path = Path(source_config["job_path"])
    source_spec = load_keyframe_job(source_job_path)
    source_resolved = resolve_keyframe_spec(
        source_spec,
        source_job_path,
        _profile_from_resolved(source_config),
        project_root=project_root,
        check_outputs=False,
    )
    audit_seed = seed if seed is not None else int(source_result["outputs"][0]["seed"])
    variants = _audit_variants(source_spec)
    audit_dir = resolved_run_dir / "control_audit"
    _prepare_audit_dir(audit_dir, variants)
    _save_audit_conditions(source_resolved, audit_dir)

    total_start = perf_counter()
    outputs = []
    variant_results = []
    for variant in variants:
        _release_cuda_cache()
        variant_result = _run_audit_variant(
            source_spec,
            source_resolved,
            variant,
            audit_seed,
            audit_dir,
            project_root,
        )
        variant_results.append(variant_result)
        _release_cuda_cache()
        if variant_result["status"] == "completed":
            variant_output = variant_result["outputs"][0]
            output_path = audit_dir / f"{variant.name}.png"
            shutil.copy2(variant_output["path"], output_path)
            outputs.append({**variant_output, "name": variant.name, "path": output_path.as_posix()})

    completed_results = [result for result in variant_results if result["status"] == "completed"]
    if not completed_results:
        raise KeyframeControlAuditError("Control audit produced no completed variants")
    save_contact_sheet(outputs, audit_dir / "contact_sheet.png", thumb_width=256, label_x=8)
    run_result = {
        "status": "completed" if len(completed_results) == len(variant_results) else "partial",
        "job_id": f"{source_result['job_id']}.control-audit",
        "source_run_dir": resolved_run_dir.as_posix(),
        "source_job_id": source_result["job_id"],
        "git_commit": source_resolved["git_commit"],
        "models": source_resolved["profile"]["models"],
        "assets": source_resolved["assets"] | {"identity_primer": source_resolved["character"]["identity_primer"]},
        "outputs": outputs,
        "effective_config": {
            **source_resolved,
            "prompt": {
                "clip": CONTROL_AUDIT_MINIMAL_PROMPT,
                "t5": CONTROL_AUDIT_MINIMAL_PROMPT,
                "true_cfg_scale": 1.0,
            },
            "audit_variants": [_variant_json(variant, source_spec.sampling.steps) for variant in variants],
        },
        "token_metadata": completed_results[0]["token_metadata"],
        "timings_ms": {
            "total_ms": (perf_counter() - total_start) * 1000,
            "variant_total_ms": {
                variant.name: result.get("timings_ms", {}).get("total_ms")
                for variant, result in zip(variants, variant_results, strict=True)
            },
        },
        "memory": _aggregate_variant_memory(variants, variant_results),
        "variant_status": {
            variant.name: {
                "status": result["status"],
                "reason": result.get("reason"),
                "log": result.get("log"),
            }
            for variant, result in zip(variants, variant_results, strict=True)
        },
        "variant_results": {
            variant.name: result.get("result_path")
            for variant, result in zip(variants, variant_results, strict=True)
        },
    }
    write_json(audit_dir / "result.json", run_result)

    if score_runner is None:
        score_runner = score_keyframe_run
    score_result = score_runner(
        audit_dir,
        KeyframeScoreConfig(scorer_id=scorer_id),
        project_root=project_root,
    )
    audit_payload = _audit_payload(
        source_result,
        audit_dir,
        audit_seed,
        variants,
        run_result,
        score_result,
        scorer_id,
    )
    write_json(audit_dir / "audit.json", audit_payload)
    return audit_payload


def _profile_from_resolved(resolved: dict[str, Any]) -> Any:
    from aigen.keyframe_profiles import KeyframeProfile

    profile = resolved["profile"]
    return KeyframeProfile(
        name=profile["name"],
        model=profile["models"]["kontext"]["path"],
        controlnet_model=profile["models"]["controlnet"]["path"],
        nunchaku_transformer_model=Path(profile["models"]["nunchaku_transformer"]["path"]),
        dtype=profile["dtype"],
        attention_impl=profile["attention_impl"],
        pipeline_cpu_offload=profile["pipeline_cpu_offload"],
        nunchaku_layer_offload=profile["nunchaku_layer_offload"],
        vae_tiling=profile["vae_tiling"],
        model_revisions={
            "kontext": {
                key: value for key, value in profile["models"]["kontext"].items() if key != "path"
            },
            "controlnet": {
                key: value for key, value in profile["models"]["controlnet"].items() if key != "path"
            },
            "nunchaku_transformer": {
                key: value
                for key, value in profile["models"]["nunchaku_transformer"].items()
                if key != "path"
            },
        },
    )


def _run_audit_variant(
    source_spec: KeyframeJobSpec,
    source_resolved: dict[str, Any],
    variant: ControlAuditVariant,
    seed: int,
    audit_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    variant_dir = audit_dir / "variant_runs" / variant.name
    variant_spec = source_spec.model_copy(
        deep=True,
        update={
            "id": f"{source_spec.id}.control-audit.{variant.name}",
            "prompt": PromptSpec(
                clip=CONTROL_AUDIT_MINIMAL_PROMPT,
                t5=CONTROL_AUDIT_MINIMAL_PROMPT,
                negative=None,
                true_cfg_scale=1.0,
            ),
            "conditions": variant.conditions,
            "variants": [VariantSpec(name=variant.name, seed=seed)],
            "output": OutputSpec(
                directory=variant_dir.as_posix(),
                filename="{variant}.png",
                overwrite=True,
                save_conditions=False,
                save_contact_sheet=False,
            ),
        },
    )
    variant_job_path = variant_dir / "job.json"
    write_json(variant_job_path, _audit_variant_job_payload(variant_spec, source_resolved))
    result_path = variant_dir / "result.json"
    if result_path.exists():
        result = read_json(result_path, label=f"control audit result for {variant.name}")
        if _has_completed_variant_output(result, variant_spec, seed):
            return _variant_result_with_runner_metadata(result, result_path)
    process_result = _run_variant_job_process(variant_job_path, project_root)
    if process_result["status"] != "completed":
        failed_path = variant_dir / "failed.json"
        write_json(failed_path, process_result)
        return process_result | {"result_path": failed_path.as_posix()}
    return _variant_result_with_runner_metadata(
        read_json(result_path, label=f"control audit result for {variant.name}"),
        result_path,
    )


def _audit_variant_job_payload(
    variant_spec: KeyframeJobSpec,
    source_resolved: dict[str, Any],
) -> dict[str, Any]:
    payload = variant_spec.model_dump(mode="json", by_alias=True, exclude_none=True)
    payload["character"]["identity_primer"]["path"] = source_resolved["character"]["identity_primer"]["path"]
    for name, asset in source_resolved["assets"].items():
        if name in payload["assets"]:
            payload["assets"][name]["path"] = asset["path"]
    return payload


def _variant_result_with_runner_metadata(
    result: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    log_path = result_path.parent / "process.log"
    metadata = {"result_path": result_path.as_posix()}
    if log_path.exists():
        metadata["log"] = log_path.as_posix()
    return result | metadata


def _run_variant_job_process(job_path: Path, project_root: Path) -> dict[str, Any]:
    log_path = job_path.parent / "process.log"
    command = [
        sys.executable,
        "-m",
        "aigen.cli",
        "keyframes",
        "run-audit-variant",
        job_path.as_posix(),
        "--compact",
    ]
    start = perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            subprocess.run(
                command,
                cwd=project_root,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=CONTROL_AUDIT_VARIANT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            log_file.write(
                f"\nControl audit variant timed out after "
                f"{CONTROL_AUDIT_VARIANT_TIMEOUT_SECONDS} seconds.\n"
            )
            return {
                "status": "failed",
                "reason": "timeout",
                "log": log_path.as_posix(),
                "timings_ms": {"total_ms": (perf_counter() - start) * 1000},
            }
        except subprocess.CalledProcessError as exc:
            log_file.write(f"\nControl audit variant failed with exit code {exc.returncode}.\n")
            return {
                "status": "failed",
                "reason": f"exit_{exc.returncode}",
                "log": log_path.as_posix(),
                "timings_ms": {"total_ms": (perf_counter() - start) * 1000},
            }
    return {
        "status": "completed",
        "log": log_path.as_posix(),
        "timings_ms": {"total_ms": (perf_counter() - start) * 1000},
    }


def _has_completed_variant_output(
    result: dict[str, Any],
    variant_spec: KeyframeJobSpec,
    seed: int,
) -> bool:
    if result.get("status") != "completed":
        return False
    if result.get("job_id") != variant_spec.id:
        return False
    outputs = result.get("outputs", [])
    if len(outputs) != 1:
        return False
    if outputs[0].get("seed") != seed:
        return False
    return Path(outputs[0]["path"]).exists()


def _prepare_audit_dir(audit_dir: Path, variants: list[ControlAuditVariant]) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        audit_dir / "audit.json",
        audit_dir / "result.json",
        audit_dir / "contact_sheet.png",
    ):
        path.unlink(missing_ok=True)
    shutil.rmtree(audit_dir / "score", ignore_errors=True)
    for variant in variants:
        (audit_dir / f"{variant.name}.png").unlink(missing_ok=True)


def _aggregate_variant_memory(
    variants: list[ControlAuditVariant],
    variant_results: list[dict[str, Any]],
) -> dict[str, Any]:
    peaks = [
        result["memory"].get("nvidia_smi_peak_used_mb", 0)
        for result in variant_results
        if result["status"] == "completed"
    ]
    return {
        "nvidia_smi_peak_used_mb": max(peaks) if peaks else 0,
        "variants": {
            variant.name: result["memory"]
            for variant, result in zip(variants, variant_results, strict=True)
            if result["status"] == "completed"
        },
    }


def _release_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _audit_variants(spec: KeyframeJobSpec) -> list[ControlAuditVariant]:
    return [
        ControlAuditVariant(
            "control_off",
            [
                _condition("pose", "pose", "pose", 0.0, 0.0, 0.65),
            ],
        ),
        ControlAuditVariant("current_contour_baseline", [condition.model_copy() for condition in spec.conditions]),
        ControlAuditVariant("pose_only_strong", [_condition("pose", "pose", "pose", 0.90, 0.0, 0.65)]),
        ControlAuditVariant(
            "canny_lineart_unmasked",
            [_condition("canny_lineart", "canny", "canny_lineart", 0.70, 0.0, 0.80)],
        ),
        ControlAuditVariant(
            "softedge_unmasked",
            [_condition("softedge", "softedge", "softedge", 0.70, 0.0, 0.80)],
        ),
        ControlAuditVariant("gray_unmasked", [_condition("gray", "gray", "gray", 0.90, 0.0, 0.80)]),
        ControlAuditVariant(
            "pose_plus_softedge",
            [
                _condition("pose", "pose", "pose", 0.80, 0.0, 0.65),
                _condition("softedge", "softedge", "softedge", 0.55, 0.0, 0.80),
            ],
        ),
        ControlAuditVariant(
            "pose_plus_gray",
            [
                _condition("pose", "pose", "pose", 0.80, 0.0, 0.65),
                _condition("gray", "gray", "gray", 0.65, 0.0, 0.80),
            ],
        ),
        ControlAuditVariant(
            "pose_plus_canny_lineart",
            [
                _condition("pose", "pose", "pose", 0.80, 0.0, 0.65),
                _condition("canny_lineart", "canny", "canny_lineart", 0.55, 0.0, 0.80),
            ],
        ),
    ]


def _condition(
    name: str,
    condition_type: str,
    image: str,
    scale: float,
    start: float,
    end: float,
) -> ControlConditionSpec:
    return ControlConditionSpec(name=name, type=condition_type, image=image, scale=scale, start=start, end=end)


def _variant_json(variant: ControlAuditVariant, steps: int) -> dict[str, Any]:
    return {
        "name": variant.name,
        "conditions": [
            condition.model_dump(mode="json", exclude_none=True) | {"active_steps": _active_steps(condition, steps)}
            for condition in variant.conditions
        ],
    }


def _active_steps(condition: ControlConditionSpec, steps: int) -> int:
    return sum(
        condition.scale > 0.0
        and condition.start <= index / steps
        and (index + 1) / steps <= condition.end
        for index in range(steps)
    )


def _save_audit_conditions(resolved: dict[str, Any], audit_dir: Path) -> None:
    condition_dir = audit_dir / "conditions"
    condition_dir.mkdir(parents=True, exist_ok=True)
    for name, asset in resolved["assets"].items():
        shutil.copy2(asset["path"], condition_dir / f"{name}{Path(asset['path']).suffix}")


def _audit_payload(
    source_result: dict[str, Any],
    audit_dir: Path,
    seed: int,
    variants: list[ControlAuditVariant],
    run_result: dict[str, Any],
    score_result: dict[str, Any],
    scorer_id: str,
) -> dict[str, Any]:
    score_by_name = {candidate["candidate"]: candidate for candidate in score_result["candidates"]}
    score_deltas = _score_deltas(score_by_name)
    pass_status = _audit_pass_status(run_result, score_by_name, score_deltas)
    return {
        "schema_version": CONTROL_AUDIT_SCHEMA_VERSION,
        "status": "passed" if pass_status["passed"] else "failed",
        "passed": pass_status["passed"],
        "blockers": pass_status["blockers"],
        "source_run_dir": source_result["effective_config"]["output"]["directory"],
        "source_job_id": source_result["job_id"],
        "seed": seed,
        "minimal_prompt": {
            "clip": CONTROL_AUDIT_MINIMAL_PROMPT,
            "t5": CONTROL_AUDIT_MINIMAL_PROMPT,
            "true_cfg_scale": 1.0,
        },
        "variants": [_variant_json(variant, source_result["effective_config"]["sampling"]["steps"]) for variant in variants],
        "score_deltas_vs_control_off": score_deltas,
        "plain_flux_controlnet_strong": {
            "status": "not_run",
            "reason": "No plain FLUX ControlNet generation owner exists in this codebase yet.",
        },
        "outputs": {
            "audit": (audit_dir / "audit.json").as_posix(),
            "result": (audit_dir / "result.json").as_posix(),
            "contact_sheet": (audit_dir / "contact_sheet.png").as_posix(),
            "scores": score_result["outputs"]["scores"],
            "ranked_contact_sheet": score_result["outputs"]["ranked_contact_sheet"],
            "condition_evidence_ranked": score_result["outputs"]["condition_evidence_ranked"],
            "pose_evidence_ranked": score_result["outputs"]["pose_evidence_ranked"],
        },
        "generation_outputs": run_result["outputs"],
        "memory": run_result["memory"],
        "timings_ms": run_result["timings_ms"],
    }


def _score_deltas(score_by_name: dict[str, Any]) -> dict[str, dict[str, float]]:
    control_off = score_by_name["control_off"]["scores"]
    deltas = {}
    for name, candidate in score_by_name.items():
        deltas[name] = {
            score_name: float(score - control_off[score_name])
            for score_name, score in candidate["scores"].items()
        }
    return deltas


def _audit_pass_status(
    run_result: dict[str, Any],
    score_by_name: dict[str, Any],
    score_deltas: dict[str, dict[str, float]],
) -> dict[str, Any]:
    strong_names = (
        "pose_only_strong",
        "canny_lineart_unmasked",
        "softedge_unmasked",
        "gray_unmasked",
        "pose_plus_softedge",
        "pose_plus_gray",
        "pose_plus_canny_lineart",
    )
    completed_strong_names = [name for name in strong_names if name in score_deltas]
    best_condition_delta = (
        max(score_deltas[name]["condition"] for name in completed_strong_names)
        if completed_strong_names
        else 0.0
    )
    best_pose_delta = (
        max(score_deltas[name]["pose"] for name in completed_strong_names)
        if completed_strong_names
        else 0.0
    )
    blockers = []
    failed_variants = [
        name
        for name, status in run_result.get("variant_status", {}).items()
        if status["status"] != "completed"
    ]
    if failed_variants:
        blockers.extend(f"audit_variant_failed:{name}" for name in failed_variants)
    if max(best_condition_delta, best_pose_delta) <= MATERIAL_SCORE_DELTA:
        blockers.append("strong_control_not_materially_closer_to_conditions")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "best_condition_delta": best_condition_delta,
        "best_pose_delta": best_pose_delta,
        "control_off_scores": score_by_name["control_off"]["scores"],
    }
