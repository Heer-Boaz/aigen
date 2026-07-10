from __future__ import annotations

from aigen.character_instruction_models import CharacterInstructionPlanSpec, InstructionTargetConstraintsSpec
from aigen.character_task_route_models import (
    CHARACTER_TASK_ROUTE_PLAN_KIND,
    CharacterTaskRoutePlanSpec,
    FinalEditorConstraintsSpec,
    ModelCapabilityRegistrySpec,
    ReferenceBudgetSpec,
)


class CharacterTaskRouter:
    def __init__(self, capability_registry: ModelCapabilityRegistrySpec | None = None) -> None:
        self.capability_registry = capability_registry or ModelCapabilityRegistrySpec()

    def route(
        self,
        instruction_plan: CharacterInstructionPlanSpec,
        *,
        pose_source_present: bool = False,
    ) -> CharacterTaskRoutePlanSpec:
        if _is_local_repair(instruction_plan):
            return _local_repair_route(instruction_plan, self.capability_registry)
        if pose_source_present or _is_pose_transfer(instruction_plan):
            return _pose_transfer_route(instruction_plan, self.capability_registry)
        if _is_layout_or_sheet(instruction_plan):
            return _layout_or_sheet_route(instruction_plan, self.capability_registry)
        if _is_scene_insertion(instruction_plan):
            return _scene_insertion_route(instruction_plan, self.capability_registry)
        if _is_portrait(instruction_plan):
            return _portrait_route(instruction_plan, self.capability_registry)
        if _is_full_body(instruction_plan):
            return _full_body_route(instruction_plan, self.capability_registry)
        if _is_outfit_swap(instruction_plan):
            return _outfit_swap_route(instruction_plan, self.capability_registry)
        if _is_view_change(instruction_plan):
            return _view_change_route(instruction_plan, self.capability_registry)
        if _is_style_transfer(instruction_plan):
            return _style_transfer_route(instruction_plan, self.capability_registry)
        if _is_text_primary(instruction_plan):
            return _text_primary_route(instruction_plan, self.capability_registry)
        return _unknown_reference_edit_route(instruction_plan, self.capability_registry)


def _is_local_repair(plan: CharacterInstructionPlanSpec) -> bool:
    return (
        plan.task_family == "local_repair"
        or plan.edit_scope == "local"
        or plan.envelope.mask_present
        or plan.envelope.region_plan_present
    )


def _is_pose_transfer(plan: CharacterInstructionPlanSpec) -> bool:
    return plan.task_family == "pose_transfer"


def _is_layout_or_sheet(plan: CharacterInstructionPlanSpec) -> bool:
    constraints = plan.target_constraints
    return plan.task_family == "layout_or_sheet" or _contains_any(
        _constraint_text(constraints),
        ("reference sheet", "concept sheet", "grid", "comic panel", "id card", "sticker sheet", "layout"),
    )


def _is_scene_insertion(plan: CharacterInstructionPlanSpec) -> bool:
    constraints = plan.target_constraints
    return plan.task_family == "scene_insertion" or bool(constraints.scene) or _contains_any(
        _constraint_text(constraints),
        ("cafe", "street", "room", "city", "paris", "alley", "desk", "background scene"),
    )


def _is_portrait(plan: CharacterInstructionPlanSpec) -> bool:
    constraints = plan.target_constraints
    return plan.task_family == "reference_character_portrait" or _contains_any(
        _constraint_text(constraints),
        ("portrait", "close-up", "close up", "face", "headshot"),
    )


def _is_full_body(plan: CharacterInstructionPlanSpec) -> bool:
    constraints = plan.target_constraints
    return plan.task_family == "reference_character_full_body" or _contains_any(
        _constraint_text(constraints),
        ("full body", "full-body", "standing character", "entire character"),
    )


def _is_view_change(plan: CharacterInstructionPlanSpec) -> bool:
    return plan.task_family == "view_change"


def _is_outfit_swap(plan: CharacterInstructionPlanSpec) -> bool:
    return plan.task_family == "outfit_swap"


def _is_style_transfer(plan: CharacterInstructionPlanSpec) -> bool:
    return plan.task_family == "style_transfer"


def _is_text_primary(plan: CharacterInstructionPlanSpec) -> bool:
    if plan.task_family != "text_or_label_heavy":
        return False
    text = _constraint_text(plan.target_constraints)
    return _contains_any(
        text,
        ("poster", "logo", "newspaper", "id card", "label sheet", "typography", "nameplate"),
    ) and not (plan.target_constraints.scene or plan.target_constraints.action)


def _portrait_route(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
) -> CharacterTaskRoutePlanSpec:
    return _route_plan(
        plan,
        registry,
        route_kind="portrait_identity_generation",
        editor_route="qwen_image_edit_multi_reference",
        output_mode="single_image_portrait",
        constraints=_constraints(registry, mask_required=False, pose_conditioning=False, text_rendering=False, layout="simple"),
    )


def _full_body_route(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
) -> CharacterTaskRoutePlanSpec:
    return _route_plan(
        plan,
        registry,
        route_kind="full_body_identity_generation",
        editor_route="qwen_image_edit_multi_reference",
        output_mode="single_image_full_body",
        constraints=_constraints(registry, mask_required=False, pose_conditioning=False, text_rendering=False, layout="simple"),
    )


def _view_change_route(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
) -> CharacterTaskRoutePlanSpec:
    return _route_plan(
        plan,
        registry,
        route_kind="view_change",
        editor_route="qwen_image_edit_multi_reference",
        output_mode="single_image_view",
        constraints=_constraints(registry, mask_required=False, pose_conditioning=False, text_rendering=False, layout="simple"),
    )


def _scene_insertion_route(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
) -> CharacterTaskRoutePlanSpec:
    return _route_plan(
        plan,
        registry,
        route_kind="scene_insertion",
        editor_route="qwen_image_edit_multi_reference",
        output_mode="single_image_scene",
        constraints=_constraints(
            registry,
            mask_required=False,
            pose_conditioning=False,
            text_rendering=_text_risk(plan),
            layout="simple",
        ),
    )


def _pose_transfer_route(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
) -> CharacterTaskRoutePlanSpec:
    return _route_plan(
        plan,
        registry,
        route_kind="pose_transfer",
        editor_route="qwen_image_edit_with_control_condition",
        output_mode="single_image_pose",
        constraints=_constraints(
            registry,
            mask_required=False,
            pose_conditioning=True,
            text_rendering=_text_risk(plan),
            layout="simple",
            reserved_control_images=1,
        ),
    )


def _local_repair_route(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
) -> CharacterTaskRoutePlanSpec:
    return _route_plan(
        plan,
        registry,
        route_kind="local_repair_or_inpaint",
        editor_route="qwen_image_edit_inpaint_or_masked_edit",
        output_mode="masked_refine_candidates",
        constraints=_constraints(registry, mask_required=True, pose_conditioning=False, text_rendering=_text_risk(plan), layout="simple"),
    )


def _outfit_swap_route(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
) -> CharacterTaskRoutePlanSpec:
    return _route_plan(
        plan,
        registry,
        route_kind="outfit_swap",
        editor_route="qwen_image_edit_multi_reference",
        output_mode="single_image_reference_edit",
        constraints=_constraints(registry, mask_required=False, pose_conditioning=False, text_rendering=_text_risk(plan), layout="simple"),
    )


def _style_transfer_route(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
) -> CharacterTaskRoutePlanSpec:
    return _route_plan(
        plan,
        registry,
        route_kind="style_transfer",
        editor_route="qwen_image_edit_multi_reference",
        output_mode="single_image_reference_edit",
        constraints=_constraints(registry, mask_required=False, pose_conditioning=False, text_rendering=_text_risk(plan), layout="simple"),
    )


def _layout_or_sheet_route(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
) -> CharacterTaskRoutePlanSpec:
    return _route_plan(
        plan,
        registry,
        route_kind="layout_or_sheet",
        editor_route="layout_heavy_reference_generation",
        output_mode="layout_or_sheet",
        constraints=_constraints(registry, mask_required=False, pose_conditioning=False, text_rendering=_text_risk(plan), layout="layout_heavy"),
    )


def _text_primary_route(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
) -> CharacterTaskRoutePlanSpec:
    return _route_plan(
        plan,
        registry,
        route_kind="text_or_label_heavy",
        editor_route="qwen_image_edit_text_heavy",
        output_mode="text_or_label_image",
        constraints=_constraints(registry, mask_required=False, pose_conditioning=False, text_rendering=True, layout="moderate"),
    )


def _unknown_reference_edit_route(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
) -> CharacterTaskRoutePlanSpec:
    return _route_plan(
        plan,
        registry,
        route_kind="unknown_reference_edit",
        editor_route="qwen_image_edit_unknown_reference",
        output_mode="single_image_reference_edit",
        constraints=_constraints(registry, mask_required=False, pose_conditioning=False, text_rendering=_text_risk(plan), layout="simple"),
    )


def _route_plan(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
    *,
    route_kind: str,
    editor_route: str,
    output_mode: str,
    constraints: FinalEditorConstraintsSpec,
) -> CharacterTaskRoutePlanSpec:
    return CharacterTaskRoutePlanSpec(
        kind=CHARACTER_TASK_ROUTE_PLAN_KIND,
        route_kind=route_kind,
        source_instruction=plan.normalized_instruction_text,
        editor_route=editor_route,
        output_mode=output_mode,
        final_editor_constraints=constraints,
        capability_registry=registry,
    )


def _constraints(
    registry: ModelCapabilityRegistrySpec,
    *,
    mask_required: bool,
    pose_conditioning: bool,
    text_rendering: bool,
    layout: str,
    reserved_control_images: int = 0,
) -> FinalEditorConstraintsSpec:
    return FinalEditorConstraintsSpec(
        reference_budget=ReferenceBudgetSpec(
            min=1,
            max=registry.max_qwen_edit_images - reserved_control_images,
        ),
        mask_required=mask_required,
        pose_conditioning=pose_conditioning,
        text_rendering=text_rendering,
        layout_complexity=layout,
    )


def _text_risk(plan: CharacterInstructionPlanSpec) -> bool:
    return (
        bool(plan.target_constraints.text_or_logo)
        or _contains_any(
            _constraint_text(plan.target_constraints),
            ("poster", "logo", "newspaper", "label", "speech bubble", "typography", "nameplate", "id card"),
        )
    )


def _constraint_text(constraints: InstructionTargetConstraintsSpec) -> str:
    values = [
        value
        for values in constraints.model_dump(mode="json").values()
        for value in values
    ]
    return " ".join(values).lower()


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)
