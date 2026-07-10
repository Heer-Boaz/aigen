from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError


CHARACTER_EDIT_PLAN_KIND = "qwen-character-edit-plan"


class CharacterEditPlanError(RuntimeError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CharacterEditPlanCaseSpec(StrictModel):
    name: str
    references: list[str]
    prompt: str
    prompt_source: Literal["user_instruction", "generic_case_prompt"]
    portrait_canvas: bool
    reference_selector: str
    route_kind: str


class CharacterEditPlanSpec(StrictModel):
    """Reusable pre-generation plan: decisions only, never character descriptions.

    The plan freezes which reference images each case feeds the editor and the
    (user-authored or generic) prompt. Reference names are pointers to images;
    no field may ever hold a visual fact extracted from those images.
    """

    kind: Literal["qwen-character-edit-plan"]
    character_id: str
    reference_pack: str
    reference_sha256: dict[str, str]
    source_instruction: str | None
    cases: list[CharacterEditPlanCaseSpec]


def character_edit_plan_schema() -> dict[str, Any]:
    schema = CharacterEditPlanSpec.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def load_character_edit_plan(data: dict[str, Any], *, path_label: str) -> CharacterEditPlanSpec:
    try:
        plan = CharacterEditPlanSpec.model_validate(_without_completed_status(data))
    except ValidationError as error:
        raise CharacterEditPlanError(f"Invalid character edit plan {path_label}: {error}") from error
    if not plan.cases:
        raise CharacterEditPlanError(f"Invalid character edit plan {path_label}: cases is empty")
    if not plan.reference_sha256:
        raise CharacterEditPlanError(f"Invalid character edit plan {path_label}: reference_sha256 is empty")
    names = [case.name for case in plan.cases]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise CharacterEditPlanError(
            f"Invalid character edit plan {path_label}: duplicate case(s) {', '.join(duplicates)}"
        )
    for case in plan.cases:
        if not case.prompt.strip():
            raise CharacterEditPlanError(f"Invalid character edit plan {path_label}: case {case.name} prompt is empty")
        if not 1 <= len(case.references) <= 3:
            raise CharacterEditPlanError(
                f"Invalid character edit plan {path_label}: case {case.name} must use 1 to 3 references"
            )
        unknown = sorted(ref for ref in case.references if ref not in plan.reference_sha256)
        if unknown:
            raise CharacterEditPlanError(
                f"Invalid character edit plan {path_label}: case {case.name} references unknown reference(s) "
                f"{', '.join(unknown)}"
            )
    return plan


def _without_completed_status(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("status") != "completed":
        return data
    payload = dict(data)
    del payload["status"]
    return payload
