from __future__ import annotations

import unittest

from aigen.character_conditioning_planner import CharacterConditioningPlanner
from aigen.character_instruction_models import (
    CharacterInstructionPlanSpec,
    InstructionEnvelopeSpec,
    InstructionSubjectBindingSpec,
    InstructionTargetConstraintsSpec,
)
from aigen.character_task_router import CharacterTaskRouter


def envelope(
    *,
    source_image_present: bool = False,
    mask_present: bool = False,
    region_plan_present: bool = False,
) -> InstructionEnvelopeSpec:
    return InstructionEnvelopeSpec(
        raw_instruction="Character as shown in referenced images.",
        ui_mode="reference_conditioned_generation",
        reference_count=3,
        source_image_present=source_image_present,
        mask_present=mask_present,
        region_plan_present=region_plan_present,
        generation_panel_settings={"case": "portrait"},
        requested_model_family="qwen-image-edit",
    )


def constraints(**overrides: list[str]) -> InstructionTargetConstraintsSpec:
    values = {
        "framing": [],
        "camera_view": [],
        "pose": [],
        "gaze": [],
        "expression": [],
        "lighting": [],
        "background": [],
        "explicit_style_or_role": [],
        "scene": [],
        "action": [],
        "mood_or_personality": [],
        "composition": [],
        "text_or_logo": [],
    }
    values.update(overrides)
    return InstructionTargetConstraintsSpec(**values)


def instruction_plan(
    *,
    task_family: str,
    target_constraints: InstructionTargetConstraintsSpec,
    downstream_requirements: list[str] | None = None,
    named_external_concepts: list[str] | None = None,
    instruction_envelope: InstructionEnvelopeSpec | None = None,
) -> CharacterInstructionPlanSpec:
    return CharacterInstructionPlanSpec(
        kind="character-instruction-plan",
        raw_instruction="Character as shown in referenced images.",
        normalized_instruction_text="Character as shown in referenced images.",
        envelope=instruction_envelope or envelope(),
        language="en",
        task_family=task_family,
        edit_scope="global",
        subject_binding=InstructionSubjectBindingSpec(
            kind="referenced_character",
            reference_mentions=["referenced images"],
            note="use referenced character",
        ),
        target_constraints=target_constraints,
        named_external_concepts=named_external_concepts or [],
        downstream_requirements=downstream_requirements or ["visual_identity_analysis", "multi_reference_alignment"],
        ambiguities=[],
        conflicts=[],
        parser={"id": "fake"},
        raw_model_response={},
    )


def conditioning_plan_for(plan: CharacterInstructionPlanSpec):
    route = CharacterTaskRouter().route(plan)
    return CharacterConditioningPlanner().plan(
        instruction_plan=plan,
        task_route_plan=route,
        visual_analysis={},
    )


class CharacterConditioningPlannerTests(unittest.TestCase):
    def test_portrait_route_needs_no_extra_conditioning(self) -> None:
        plan = instruction_plan(
            task_family="reference_character_portrait",
            target_constraints=constraints(framing=["close-up face"]),
        )
        conditioning = conditioning_plan_for(plan)

        self.assertEqual(conditioning.status, "no_extra_conditioning")
        self.assertEqual(conditioning.conditioning_modes, [])
        self.assertEqual(conditioning.deferred_to, "none")
        self.assertTrue(conditioning.supports_current_command)

    def test_local_repair_defers_region_mask_conditioning(self) -> None:
        plan = instruction_plan(
            task_family="local_repair",
            target_constraints=constraints(action=["fix the local detail"]),
            downstream_requirements=["visual_identity_analysis", "region_grounding", "mask_generation"],
        )
        conditioning = conditioning_plan_for(plan)

        self.assertEqual(conditioning.status, "required_but_missing_inputs")
        self.assertEqual(conditioning.conditioning_modes, ["region_mask"])
        self.assertEqual(conditioning.required_inputs, ["source_image", "mask", "region_plan"])
        self.assertEqual(conditioning.planned_tools, ["florence2_region_grounding", "sam2_mask_generation"])
        self.assertEqual(conditioning.deferred_to, "region-plan")
        self.assertFalse(conditioning.supports_current_command)

    def test_pose_transfer_defers_keypoint_conditioning(self) -> None:
        plan = instruction_plan(
            task_family="pose_transfer",
            target_constraints=constraints(pose=["use pose from Image 2"]),
            downstream_requirements=["visual_identity_analysis", "pose_conditioning", "visual_disambiguation"],
        )
        conditioning = conditioning_plan_for(plan)

        self.assertEqual(conditioning.status, "deferred")
        self.assertEqual(conditioning.conditioning_modes, ["pose_keypoint"])
        self.assertEqual(conditioning.required_inputs, ["pose_source"])
        self.assertEqual(conditioning.planned_tools, ["dwpose_keypoint_map"])
        self.assertEqual(conditioning.deferred_to, "qwen-edit-pose")

    def test_scene_text_risk_is_audit_only_without_region_or_pose_tools(self) -> None:
        plan = instruction_plan(
            task_family="scene_insertion",
            target_constraints=constraints(scene=["cafe in Paris"], text_or_logo=["Snatcher poster"]),
            downstream_requirements=[
                "visual_identity_analysis",
                "multi_reference_alignment",
                "external_concept_resolution",
                "text_rendering_risk",
            ],
            named_external_concepts=["Konami's Snatcher"],
        )
        conditioning = conditioning_plan_for(plan)

        self.assertEqual(conditioning.status, "deferred")
        self.assertEqual(conditioning.conditioning_modes, ["text_layout_risk"])
        self.assertEqual(conditioning.planned_tools, [])
        self.assertEqual(conditioning.deferred_to, "none")
        self.assertTrue(conditioning.supports_current_command)

    def test_layout_route_defers_layout_planning(self) -> None:
        plan = instruction_plan(
            task_family="layout_or_sheet",
            target_constraints=constraints(composition=["3x3 reference sheet grid"]),
        )
        conditioning = conditioning_plan_for(plan)

        self.assertEqual(conditioning.status, "deferred")
        self.assertEqual(conditioning.conditioning_modes, ["layout_planning"])
        self.assertEqual(conditioning.deferred_to, "none")
        self.assertFalse(conditioning.supports_current_command)


if __name__ == "__main__":
    unittest.main()
