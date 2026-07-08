from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aigen.character_conditioning_models import (
    CHARACTER_CONDITIONING_PLAN_KIND,
    CharacterConditioningPlanSpec,
)
from aigen.character_instruction_models import CharacterInstructionPlanSpec
from aigen.character_task_route_models import CharacterTaskRoutePlanSpec


class CharacterConditioningPlanner:
    def plan(
        self,
        *,
        instruction_plan: CharacterInstructionPlanSpec,
        task_route_plan: CharacterTaskRoutePlanSpec,
        visual_analysis: Mapping[str, Any],
    ) -> CharacterConditioningPlanSpec:
        route_kind = task_route_plan.route_kind
        if route_kind == "local_repair_or_inpaint":
            return _local_repair_plan(instruction_plan, task_route_plan, visual_analysis)
        if route_kind == "pose_transfer":
            return _pose_transfer_plan(task_route_plan, visual_analysis)
        if route_kind == "layout_or_sheet":
            return _layout_plan(task_route_plan, visual_analysis)
        if route_kind == "text_or_label_heavy":
            return _text_plan(task_route_plan, visual_analysis, supports_current_command=False)
        if _has_text_risk(task_route_plan):
            return _text_plan(task_route_plan, visual_analysis, supports_current_command=True)
        return _no_extra_conditioning_plan(task_route_plan, visual_analysis)


def _local_repair_plan(
    instruction_plan: CharacterInstructionPlanSpec,
    task_route_plan: CharacterTaskRoutePlanSpec,
    visual_analysis: Mapping[str, Any],
) -> CharacterConditioningPlanSpec:
    missing_inputs: list[str] = []
    if not instruction_plan.envelope.source_image_present:
        missing_inputs.append("source_image")
    if not (instruction_plan.envelope.mask_present or instruction_plan.envelope.region_plan_present):
        missing_inputs.extend(["mask", "region_plan"])
    ready = not missing_inputs
    planned_tools = []
    if "region_plan" in missing_inputs or "region_grounding" in task_route_plan.conditioning_needs:
        planned_tools.append("florence2_region_grounding")
    if "mask" in missing_inputs or "mask_generation" in task_route_plan.conditioning_needs:
        planned_tools.append("sam2_mask_generation")
    return _conditioning_plan(
        task_route_plan,
        visual_analysis,
        status="ready" if ready else "required_but_missing_inputs",
        conditioning_modes=["region_mask"],
        required_inputs=_dedupe(missing_inputs),
        planned_tools=_dedupe(planned_tools),
        deferred_to="qwen-edit-refine" if ready else "region-plan",
        supports_current_command=ready,
        audit_notes=[
            "Local repair uses source-image region or mask conditioning outside the normal qwen-edit generation path.",
        ],
    )


def _pose_transfer_plan(
    task_route_plan: CharacterTaskRoutePlanSpec,
    visual_analysis: Mapping[str, Any],
) -> CharacterConditioningPlanSpec:
    return _conditioning_plan(
        task_route_plan,
        visual_analysis,
        status="deferred",
        conditioning_modes=["pose_keypoint"],
        required_inputs=["pose_source"],
        planned_tools=["dwpose_keypoint_map"],
        deferred_to="qwen-edit-pose",
        supports_current_command=False,
        audit_notes=[
            "Pose transfer needs a later keypoint/control asset path; qwen-edit-run does not create it.",
        ],
    )


def _layout_plan(
    task_route_plan: CharacterTaskRoutePlanSpec,
    visual_analysis: Mapping[str, Any],
) -> CharacterConditioningPlanSpec:
    return _conditioning_plan(
        task_route_plan,
        visual_analysis,
        status="deferred",
        conditioning_modes=["layout_planning"],
        required_inputs=[],
        planned_tools=[],
        deferred_to="none",
        supports_current_command=False,
        audit_notes=["Layout-heavy outputs are route-only for now; no hidden conditioning assets are generated."],
    )


def _text_plan(
    task_route_plan: CharacterTaskRoutePlanSpec,
    visual_analysis: Mapping[str, Any],
    *,
    supports_current_command: bool,
) -> CharacterConditioningPlanSpec:
    return _conditioning_plan(
        task_route_plan,
        visual_analysis,
        status="deferred",
        conditioning_modes=["text_layout_risk"],
        required_inputs=[],
        planned_tools=[],
        deferred_to="none",
        supports_current_command=supports_current_command,
        audit_notes=["Text or logo rendering risk is tracked for audit; no text specialist is run in qwen-edit-run."],
    )


def _no_extra_conditioning_plan(
    task_route_plan: CharacterTaskRoutePlanSpec,
    visual_analysis: Mapping[str, Any],
) -> CharacterConditioningPlanSpec:
    return _conditioning_plan(
        task_route_plan,
        visual_analysis,
        status="no_extra_conditioning",
        conditioning_modes=[],
        required_inputs=[],
        planned_tools=[],
        deferred_to="none",
        supports_current_command=True,
        audit_notes=[],
    )


def _conditioning_plan(
    task_route_plan: CharacterTaskRoutePlanSpec,
    visual_analysis: Mapping[str, Any],
    *,
    status: str,
    conditioning_modes: list[str],
    required_inputs: list[str],
    planned_tools: list[str],
    deferred_to: str,
    supports_current_command: bool,
    audit_notes: list[str],
) -> CharacterConditioningPlanSpec:
    return CharacterConditioningPlanSpec(
        kind=CHARACTER_CONDITIONING_PLAN_KIND,
        route_kind=task_route_plan.route_kind,
        status=status,
        conditioning_modes=_dedupe(conditioning_modes),
        required_inputs=_dedupe(required_inputs),
        planned_tools=_dedupe(planned_tools),
        deferred_to=deferred_to,
        supports_current_command=supports_current_command,
        audit_notes=audit_notes + _visual_analysis_notes(visual_analysis),
    )


def _has_text_risk(task_route_plan: CharacterTaskRoutePlanSpec) -> bool:
    return (
        "text_rendering_risk" in task_route_plan.conditioning_needs
        or task_route_plan.final_editor_constraints.text_rendering
    )


def _visual_analysis_notes(visual_analysis: Mapping[str, Any]) -> list[str]:
    deferred = visual_analysis.get("deferred_conditioning")
    if not deferred:
        return []
    return ["Step 3 visual analysis reported deferred conditioning details."]


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
