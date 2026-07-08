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

    def route(self, instruction_plan: CharacterInstructionPlanSpec) -> CharacterTaskRoutePlanSpec:
        if _is_local_repair(instruction_plan):
            return _local_repair_route(instruction_plan, self.capability_registry)
        if _is_pose_transfer(instruction_plan):
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


def compact_vlm_planner_context(
    instruction_plan: CharacterInstructionPlanSpec,
    task_route_plan: CharacterTaskRoutePlanSpec,
) -> dict[str, object]:
    return {
        "instruction_context": {
            "task_family": instruction_plan.task_family,
            "edit_scope": instruction_plan.edit_scope,
            "subject_binding": instruction_plan.subject_binding.model_dump(mode="json"),
            "target_constraints": instruction_plan.target_constraints.model_dump(mode="json"),
            "named_external_concepts": list(instruction_plan.named_external_concepts),
            "downstream_requirements": list(instruction_plan.downstream_requirements),
        },
        "task_route": {
            "route_kind": task_route_plan.route_kind,
            "editor_route": task_route_plan.editor_route,
            "visual_analysis_focus": list(task_route_plan.visual_analysis_focus),
            "reference_selection_intent": list(task_route_plan.reference_selection_intent),
            "conditioning_needs": list(task_route_plan.conditioning_needs),
            "final_editor_constraints": task_route_plan.final_editor_constraints.model_dump(mode="json"),
        },
    }


def _is_local_repair(plan: CharacterInstructionPlanSpec) -> bool:
    return (
        plan.task_family == "local_repair"
        or plan.edit_scope == "local"
        or "region_grounding" in plan.downstream_requirements
        or "mask_generation" in plan.downstream_requirements
        or plan.envelope.mask_present
        or plan.envelope.region_plan_present
    )


def _is_pose_transfer(plan: CharacterInstructionPlanSpec) -> bool:
    return plan.task_family == "pose_transfer" or "pose_conditioning" in plan.downstream_requirements


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
        visual_analysis_focus=[
            "primary subject identity",
            "face",
            "eyes",
            "hair",
            "expression",
            "gaze or head angle",
            "visible neck, collar, shoulders, or upper outfit",
            "art style",
        ],
        reference_selection_intent=[
            "prefer portrait or headshot reference if present",
            "include front or three-quarter reference for face identity",
            "include side reference when gaze or profile direction needs support",
        ],
        conditioning_needs=_base_conditioning_needs(plan),
        constraints=_constraints(registry, mask_required=False, pose_conditioning=False, text_rendering=False, layout="simple"),
        audit_notes=[
            "Treat full-body/back-view details as low-priority evidence for close-up portrait unless later VLM analysis needs them.",
        ],
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
        visual_analysis_focus=[
            "primary subject identity",
            "full outfit",
            "body proportion",
            "silhouette",
            "footwear",
            "front and side consistency",
            "art style",
        ],
        reference_selection_intent=[
            "prefer full-body/front reference when available",
            "include portrait reference for face identity",
            "include side or back reference when body silhouette or rear details matter",
        ],
        conditioning_needs=_base_conditioning_needs(plan),
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
        visual_analysis_focus=[
            "primary subject identity",
            "requested camera or view direction",
            "face and outfit consistency",
            "body orientation",
            "art style",
        ],
        reference_selection_intent=[
            "select references that best support the requested view",
            "include portrait or front reference for identity",
            "include side/back reference when the requested view needs it",
        ],
        conditioning_needs=_base_conditioning_needs(plan),
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
        visual_analysis_focus=[
            "primary subject identity",
            "visible outfit",
            "art style",
            "requested scene",
            "action or interaction",
            "pose compatibility",
            "scene integration",
        ],
        reference_selection_intent=[
            "select strongest identity reference",
            "include outfit or full-body reference when visible body context matters",
            "include scene/style reference only if supplied and relevant",
        ],
        conditioning_needs=_base_conditioning_needs(plan) + _scene_conditioning_needs(plan),
        constraints=_constraints(
            registry,
            mask_required=False,
            pose_conditioning="pose_conditioning" in plan.downstream_requirements,
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
        visual_analysis_focus=[
            "primary subject identity",
            "pose source",
            "body orientation",
            "limb arrangement",
            "visible or occluded components",
            "art style",
        ],
        reference_selection_intent=[
            "separate identity evidence from pose evidence",
            "select identity references that preserve face and outfit",
            "mark the pose source for downstream keypoint or edge conditioning",
        ],
        conditioning_needs=_base_conditioning_needs(plan) + ["pose_conditioning", "possible_keypoint_map"],
        constraints=_constraints(registry, mask_required=False, pose_conditioning=True, text_rendering=_text_risk(plan), layout="simple"),
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
        visual_analysis_focus=[
            "selected source image",
            "target component or region",
            "matching evidence in references",
            "unmasked source preservation",
            "local material and shape consistency",
        ],
        reference_selection_intent=[
            "use the current source image as the edit base",
            "select references that show the target component",
            "prefer references that show the local detail clearly",
        ],
        conditioning_needs=_base_conditioning_needs(plan) + _local_repair_conditioning_needs(plan),
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
        visual_analysis_focus=[
            "primary subject identity",
            "source outfit",
            "requested outfit change",
            "components to preserve",
            "body proportion",
        ],
        reference_selection_intent=[
            "select references that best preserve identity",
            "select references that show outfit evidence",
        ],
        conditioning_needs=_base_conditioning_needs(plan),
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
        visual_analysis_focus=[
            "primary subject identity",
            "requested style",
            "style reference if supplied",
            "identity/style balance",
        ],
        reference_selection_intent=[
            "select identity references first",
            "include style evidence only when it is supplied or explicitly requested",
        ],
        conditioning_needs=_base_conditioning_needs(plan),
        constraints=_constraints(registry, mask_required=False, pose_conditioning=False, text_rendering=_text_risk(plan), layout="simple"),
    )


def _layout_or_sheet_route(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
) -> CharacterTaskRoutePlanSpec:
    deferred = [] if registry.supports_layout_sheet == "supported" else ["layout_sheet_route_is_experimental"]
    return _route_plan(
        plan,
        registry,
        route_kind="layout_or_sheet",
        editor_route="layout_heavy_reference_generation",
        output_mode="layout_or_sheet",
        visual_analysis_focus=[
            "primary subject identity",
            "requested views or panels",
            "outfit and component consistency",
            "layout semantics",
            "text or annotation placement when requested",
        ],
        reference_selection_intent=[
            "select references that cover the requested views",
            "include portrait reference for identity",
            "include full-body or component references when sheet content needs them",
        ],
        conditioning_needs=_base_conditioning_needs(plan) + ["layout_planning"] + _text_conditioning_needs(plan),
        constraints=_constraints(registry, mask_required=False, pose_conditioning=False, text_rendering=_text_risk(plan), layout="layout_heavy"),
        unsupported_or_deferred=deferred,
    )


def _text_primary_route(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
) -> CharacterTaskRoutePlanSpec:
    deferred = [] if registry.supports_text_heavy_generation == "supported" else ["text_heavy_generation_is_experimental"]
    return _route_plan(
        plan,
        registry,
        route_kind="text_or_label_heavy",
        editor_route="qwen_image_edit_text_heavy",
        output_mode="text_or_label_image",
        visual_analysis_focus=[
            "primary subject identity when present",
            "requested text or logo",
            "text placement",
            "composition",
            "style",
        ],
        reference_selection_intent=[
            "select identity references when the text output includes the character",
            "select references that support the requested layout or label target",
        ],
        conditioning_needs=_base_conditioning_needs(plan) + _text_conditioning_needs(plan),
        constraints=_constraints(registry, mask_required=False, pose_conditioning=False, text_rendering=True, layout="moderate"),
        unsupported_or_deferred=deferred,
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
        visual_analysis_focus=[
            "primary subject identity",
            "user-written request",
            "visible references relevant to the request",
        ],
        reference_selection_intent=[
            "select references that best support the user-written request",
            "prefer identity-preserving references when a referenced character is involved",
        ],
        conditioning_needs=_base_conditioning_needs(plan),
        constraints=_constraints(registry, mask_required=False, pose_conditioning=False, text_rendering=_text_risk(plan), layout="simple"),
        audit_notes=["Router could not classify the request more specifically."],
    )


def _route_plan(
    plan: CharacterInstructionPlanSpec,
    registry: ModelCapabilityRegistrySpec,
    *,
    route_kind: str,
    editor_route: str,
    output_mode: str,
    visual_analysis_focus: list[str],
    reference_selection_intent: list[str],
    conditioning_needs: list[str],
    constraints: FinalEditorConstraintsSpec,
    unsupported_or_deferred: list[str] | None = None,
    audit_notes: list[str] | None = None,
) -> CharacterTaskRoutePlanSpec:
    return CharacterTaskRoutePlanSpec(
        kind=CHARACTER_TASK_ROUTE_PLAN_KIND,
        route_kind=route_kind,
        source_instruction=plan.normalized_instruction_text,
        editor_route=editor_route,
        output_mode=output_mode,
        visual_analysis_focus=visual_analysis_focus,
        reference_selection_intent=reference_selection_intent,
        conditioning_needs=_dedupe(conditioning_needs),
        final_editor_constraints=constraints,
        unsupported_or_deferred=unsupported_or_deferred or [],
        audit_notes=audit_notes or [],
        capability_registry=registry,
    )


def _constraints(
    registry: ModelCapabilityRegistrySpec,
    *,
    mask_required: bool,
    pose_conditioning: bool,
    text_rendering: bool,
    layout: str,
) -> FinalEditorConstraintsSpec:
    return FinalEditorConstraintsSpec(
        reference_budget=ReferenceBudgetSpec(min=1, max=registry.max_qwen_edit_refs),
        mask_required=mask_required,
        pose_conditioning=pose_conditioning,
        text_rendering=text_rendering,
        layout_complexity=layout,
        supports_route_now=_supports_route_now(registry, mask_required=mask_required, pose_conditioning=pose_conditioning),
    )


def _supports_route_now(
    registry: ModelCapabilityRegistrySpec,
    *,
    mask_required: bool,
    pose_conditioning: bool,
) -> bool:
    if mask_required and not registry.supports_masked_refine:
        return False
    if pose_conditioning and not registry.supports_keypoint_condition:
        return False
    return registry.supports_multi_image_edit


def _base_conditioning_needs(plan: CharacterInstructionPlanSpec) -> list[str]:
    needs = []
    if "visual_identity_analysis" in plan.downstream_requirements:
        needs.append("multi_reference_visual_identity_analysis")
    if "multi_reference_alignment" in plan.downstream_requirements:
        needs.append("multi_reference_alignment")
    if "visual_disambiguation" in plan.downstream_requirements:
        needs.append("visual_disambiguation")
    return needs


def _scene_conditioning_needs(plan: CharacterInstructionPlanSpec) -> list[str]:
    needs = []
    if plan.named_external_concepts or "external_concept_resolution" in plan.downstream_requirements:
        needs.append("external_concept_resolution")
    needs.extend(_text_conditioning_needs(plan))
    return needs


def _local_repair_conditioning_needs(plan: CharacterInstructionPlanSpec) -> list[str]:
    needs = []
    if "region_grounding" in plan.downstream_requirements:
        needs.append("region_grounding")
    if "mask_generation" in plan.downstream_requirements or not plan.envelope.mask_present:
        needs.append("mask_generation")
    return needs


def _text_conditioning_needs(plan: CharacterInstructionPlanSpec) -> list[str]:
    return ["text_rendering_risk"] if _text_risk(plan) else []


def _text_risk(plan: CharacterInstructionPlanSpec) -> bool:
    return (
        "text_rendering_risk" in plan.downstream_requirements
        or bool(plan.target_constraints.text_or_logo)
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


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
