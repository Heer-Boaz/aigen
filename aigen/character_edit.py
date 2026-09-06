from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

from aigen.character_edit_audit import (
    CharacterAuditGroup, CharacterAuditResult, audit_character_candidates,
    default_character_audit_config,
)
from aigen.generation.image_batch_postprocess import postprocess_image_batch
from aigen.generation.image_edit_batch import (
    ImageEditBatchOutput, ImageEditBatchRequest, run_image_edit_batch,
)
from aigen.generation.vosr_backend import VOSR_POSTPROCESS_NAME
from aigen.manifest_io import atomic_write_json, file_manifest
from aigen.progress import StatusReporter
from aigen.vlm_qwen import QwenVlm, QwenVlmConfig, qwen_vlm_config_json, validate_local_qwen_model


CHARACTER_EDIT_REVISION = "1"
CHARACTER_EDIT_BACKENDS = ("flux2-klein", "qwen-image-edit-2511-lightning")
DEFAULT_CHARACTER_AUDIT_ITERATIONS = 2


class CharacterEditError(RuntimeError):
    pass


@dataclass(frozen=True)
class CharacterEditSelection:
    group_id: str
    candidate_identity: str
    raw: ImageEditBatchOutput
    image: Path
    audit_report: Path

    def to_json(self) -> dict[str, object]:
        return {
            "group_id": self.group_id, "candidate_identity": self.candidate_identity,
            "raw": self.raw.model_dump(mode="json"), "image": file_manifest(self.image),
            "audit_report": str(self.audit_report),
        }


@dataclass(frozen=True)
class CharacterEditResult:
    backend: str
    selections: tuple[CharacterEditSelection, ...]
    report: Path
    iterations: int

    def to_json(self) -> dict[str, object]:
        return {
            "status": "completed", "kind": "audited-character-edit-result",
            "backend": self.backend, "implementation_revision": CHARACTER_EDIT_REVISION,
            "iterations": self.iterations, "audit_report": str(self.report),
            "outputs": [selection.to_json() for selection in self.selections],
        }


def run_character_edit(
    request: ImageEditBatchRequest,
    *,
    audit_groups: Sequence[CharacterAuditGroup],
    output_dir: Path,
    progress: StatusReporter,
    audit_config: QwenVlmConfig | None = None,
    max_iterations: int = DEFAULT_CHARACTER_AUDIT_ITERATIONS,
    upscale_long_side: int | None = None,
) -> CharacterEditResult:
    if request.backend not in CHARACTER_EDIT_BACKENDS:
        raise CharacterEditError(f"The active character route requires an explicit backend from {CHARACTER_EDIT_BACKENDS}")
    if max_iterations < 1:
        raise CharacterEditError("character audit iterations must be at least one")
    if upscale_long_side is not None and upscale_long_side < 1:
        raise CharacterEditError("character upscale long side must be positive")
    if upscale_long_side is not None and any(case.mask is not None for case in request.cases):
        raise CharacterEditError("whole-image upscale would invalidate exact preservation outside the edit mask")
    groups = tuple(audit_groups)
    candidate_ids = [candidate_id for group in groups for candidate_id in group.candidate_ids]
    if len(candidate_ids) != len(set(candidate_ids)) or set(candidate_ids) != {case.id for case in request.cases}:
        raise CharacterEditError("audit groups must assign every generation case exactly once")
    if len({group.id for group in groups}) != len(groups):
        raise CharacterEditError("character audit group ids must be unique")
    if any(not group.references or not group.candidate_ids for group in groups):
        raise CharacterEditError("every character audit group needs reference images and candidates")
    audit_config = default_character_audit_config() if audit_config is None else audit_config
    validate_local_qwen_model(audit_config)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True)
    atomic_write_json(output_dir / "request.json", {
        "implementation_revision": CHARACTER_EDIT_REVISION,
        "generation": request.model_dump(mode="json"),
        "audit": qwen_vlm_config_json(audit_config),
        "max_iterations": max_iterations,
        "upscale_long_side": upscale_long_side,
        "groups": [{
            **asdict(group), "references": [file_manifest(path) for path in group.references],
            "context_images": [{"role": image.role, **file_manifest(image.path)} for image in group.context_images],
        } for group in groups],
    })
    report_path = output_dir / "audit.json"
    rounds: list[dict[str, object]] = []
    accepted: dict[str, CharacterAuditResult] = {}
    active_groups = groups
    active_cases = request.cases
    seed_by_id = {case.id: case.seed for case in active_cases}
    next_seeds = {group.id: max(seed_by_id[case_id] for case_id in group.candidate_ids) + 1 for group in groups}
    group_by_candidate = {case_id: group.id for group in groups for case_id in group.candidate_ids}
    stage = "generation"

    try:
        for iteration in range(1, max_iterations + 1):
            stage = "generation"
            directory = output_dir / f"round-{iteration}"
            cases = tuple(case.model_copy(update={
                "output_path": directory / "raw" / f"{case.id}.png",
            }) for case in active_cases)
            decisions = []
            rounds.append({
                "iteration": iteration,
                "seeds": {case.id: case.seed for case in cases},
                "decisions": decisions,
            })
            progress.phase(f"character raw generation round {iteration}/{max_iterations}")
            generated = run_image_edit_batch(
                request.model_copy(update={"cases": cases}), progress=progress,
                record_dir=directory / "generation",
            )
            by_id = {output.case_id: output for output in generated.outputs}
            if set(by_id) != {case.id for case in cases} or len(by_id) != len(generated.outputs):
                raise CharacterEditError("character generation returned an incomplete or duplicate candidate batch")
            # The generation owner has returned and released its weights here.
            stage = "audit"
            progress.phase(f"character raw audit round {iteration}/{max_iterations}")
            rejected = []
            with closing(QwenVlm(audit_config)) as judge:
                for index, group in enumerate(active_groups):
                    audit = audit_character_candidates(
                        judge, group, tuple(by_id[case_id] for case_id in group.candidate_ids),
                        directory / "audit" / f"group-{index}",
                    )
                    if audit.verdict.passed:
                        accepted[group.id] = audit
                        response = "select"
                    else:
                        rejected.append(group)
                        response = "regenerate_with_new_seeds" if iteration < max_iterations else "fail"
                    decisions.append({
                        "group_id": group.id, **audit.verdict.model_dump(mode="json"),
                        "candidate_identity": audit.candidate_identity,
                        "report": str(audit.report), "response": response,
                    })
            atomic_write_json(report_path, {
                "status": "passed" if not rejected else "rejected",
                "rounds": rounds, "max_iterations": max_iterations,
            })
            if not rejected:
                break
            if iteration == max_iterations:
                break
            active_groups = tuple(rejected)
            rejected_ids = {case_id for group in active_groups for case_id in group.candidate_ids}
            retry_cases = tuple(case for case in active_cases if case.id in rejected_ids)
            retried = []
            for case in retry_cases:
                group_id = group_by_candidate[case.id]
                retried.append(case.model_copy(update={"seed": next_seeds[group_id]}))
                next_seeds[group_id] += 1
            active_cases = tuple(retried)
        if len(accepted) != len(groups):
            raise CharacterEditError(f"Character audit still rejects {len(groups) - len(accepted)} edit group(s) after {max_iterations} rounds; see {report_path}")

        selected = tuple(accepted[group.id] for group in groups)
        paths = tuple(audit.selected.path for audit in selected)
        if upscale_long_side is not None:
            stage = "postprocess"
            progress.phase("upscale accepted character images")
            postprocess = postprocess_image_batch(
                paths, output_dir / "upscaled", model=VOSR_POSTPROCESS_NAME,
                output_names=tuple(f"image-{index}.png" for index in range(len(paths))),
                long_side=upscale_long_side, progress=progress,
            )
            atomic_write_json(output_dir / "postprocess.json", postprocess.to_json())
            paths = postprocess.outputs
        result = CharacterEditResult(
            backend=request.backend,
            selections=tuple(CharacterEditSelection(group.id, audit.candidate_identity, audit.selected, path, audit.report)
                             for group, audit, path in zip(groups, selected, paths, strict=True)),
            report=report_path, iterations=len(rounds),
        )
        atomic_write_json(output_dir / "result.json", result.to_json())
        return result
    except BaseException as error:
        atomic_write_json(report_path, {
            "status": "failed", "rounds": rounds, "max_iterations": max_iterations,
            "stage": stage, "error": type(error).__name__, "message": str(error),
        })
        raise
