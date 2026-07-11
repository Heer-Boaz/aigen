from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aigen.character_task_route_models import CHARACTER_TASK_ROUTE_KINDS


CHARACTER_VERIFICATION_MATRIX_KIND = "character-verification-matrix"
_SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class CharacterVerificationError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CharacterVerificationPoseSpec(StrictModel):
    source: str
    mode: Literal["native", "keypoint"]


class CharacterVerificationCaseSpec(StrictModel):
    name: str
    route_kind: str
    instruction: str
    seeds: list[int] = Field(min_length=1)
    pose: CharacterVerificationPoseSpec | None = None


class CharacterVerificationCanvasSpec(StrictModel):
    max_side: int = Field(ge=16)
    output_format: str
    max_sequence_length: int = Field(ge=1, le=1024)


class CharacterVerificationMatrixSpec(StrictModel):
    kind: Literal["character-verification-matrix"]
    id: str
    reference_pack: str
    reference_indices: list[int]
    pose_sources: dict[str, str]
    cases: list[CharacterVerificationCaseSpec]
    canvas: CharacterVerificationCanvasSpec


def load_character_verification_matrix(path: Path) -> CharacterVerificationMatrixSpec:
    try:
        spec = CharacterVerificationMatrixSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise CharacterVerificationError(f"Invalid character verification matrix {path}: {error}") from error
    _validate_matrix(spec, path)
    return spec


def _validate_matrix(spec: CharacterVerificationMatrixSpec, path: Path) -> None:
    _validate_name(spec.id, label="matrix id", path=path)
    if not spec.cases:
        raise CharacterVerificationError(f"Invalid character verification matrix {path}: cases is empty")
    names = [case.name for case in spec.cases]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise CharacterVerificationError(
            f"Invalid character verification matrix {path}: duplicate case(s) {', '.join(duplicates)}"
        )
    for source_name, source_path in spec.pose_sources.items():
        _validate_name(source_name, label="pose source", path=path)
        if not source_path.strip():
            raise CharacterVerificationError(
                f"Invalid character verification matrix {path}: pose source {source_name} path is empty"
            )
    expected_seeds = spec.cases[0].seeds
    for case in spec.cases:
        _validate_name(case.name, label="case", path=path)
        if case.route_kind not in CHARACTER_TASK_ROUTE_KINDS:
            raise CharacterVerificationError(
                f"Invalid character verification matrix {path}: case {case.name} has unknown route "
                f"{case.route_kind}"
            )
        if not case.instruction.strip():
            raise CharacterVerificationError(
                f"Invalid character verification matrix {path}: case {case.name} instruction is empty"
            )
        if case.seeds != expected_seeds:
            raise CharacterVerificationError(
                f"Invalid character verification matrix {path}: case {case.name} must use the shared fixed seeds "
                f"{expected_seeds}"
            )
        if any(seed < 0 for seed in case.seeds) or len(set(case.seeds)) != len(case.seeds):
            raise CharacterVerificationError(
                f"Invalid character verification matrix {path}: case {case.name} seeds must be unique non-negative integers"
            )
        if case.route_kind == "pose_transfer":
            if case.pose is None:
                raise CharacterVerificationError(
                    f"Invalid character verification matrix {path}: pose case {case.name} has no pose source"
                )
            if case.pose.source not in spec.pose_sources:
                raise CharacterVerificationError(
                    f"Invalid character verification matrix {path}: case {case.name} uses unknown pose source "
                    f"{case.pose.source}"
                )
        elif case.pose is not None:
            raise CharacterVerificationError(
                f"Invalid character verification matrix {path}: non-pose case {case.name} has pose settings"
            )


def _validate_name(value: str, *, label: str, path: Path) -> None:
    if _SAFE_NAME_PATTERN.fullmatch(value) is None:
        raise CharacterVerificationError(
            f"Invalid character verification matrix {path}: {label} must be file-safe ASCII: {value}"
        )
