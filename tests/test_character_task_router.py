from __future__ import annotations

import unittest

from aigen.character_instruction_models import (
    CharacterInstructionPlanSpec,
    InstructionEnvelopeSpec,
    InstructionSubjectBindingSpec,
    InstructionTargetConstraintsSpec,
)
from aigen.character_task_router import CharacterTaskRouter, compact_vlm_planner_context


def envelope() -> InstructionEnvelopeSpec:
    return InstructionEnvelopeSpec(
        raw_instruction="Character as shown in referenced images.",
        ui_mode="reference_conditioned_generation",
        reference_count=3,
        source_image_present=False,
        mask_present=False,
        region_plan_present=False,
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
) -> CharacterInstructionPlanSpec:
    return CharacterInstructionPlanSpec(
        kind="character-instruction-plan",
        raw_instruction="Character as shown in referenced images.",
        normalized_instruction_text="Character as shown in referenced images.",
        envelope=envelope(),
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


class CharacterTaskRouterTests(unittest.TestCase):
    def test_close_up_prompt_routes_to_portrait_identity_generation(self) -> None:
        plan = instruction_plan(
            task_family="reference_character_portrait",
            target_constraints=constraints(
                framing=["close-up face"],
                gaze=["looking left"],
                expression=["neutral expression"],
                lighting=["neutral lighting"],
                background=["white background"],
            ),
        )
        route = CharacterTaskRouter().route(plan)

        self.assertEqual(route.route_kind, "portrait_identity_generation")
        self.assertEqual(route.editor_route, "qwen_image_edit_multi_reference")
        self.assertIn("multi_reference_visual_identity_analysis", route.conditioning_needs)
        self.assertFalse(route.final_editor_constraints.mask_required)
        self.assertFalse(route.final_editor_constraints.pose_conditioning)

    def test_scene_with_external_text_risk_stays_scene_route(self) -> None:
        plan = instruction_plan(
            task_family="scene_insertion",
            target_constraints=constraints(
                explicit_style_or_role=["noir detective"],
                scene=["cafe in Paris"],
                action=["drinking coffee"],
                mood_or_personality=["heroic", "warm", "protective"],
                text_or_logo=["Snatcher poster"],
            ),
            downstream_requirements=[
                "visual_identity_analysis",
                "multi_reference_alignment",
                "external_concept_resolution",
                "text_rendering_risk",
            ],
            named_external_concepts=["Konami's Snatcher"],
        )
        route = CharacterTaskRouter().route(plan)

        self.assertEqual(route.route_kind, "scene_insertion")
        self.assertIn("external_concept_resolution", route.conditioning_needs)
        self.assertIn("text_rendering_risk", route.conditioning_needs)
        self.assertTrue(route.final_editor_constraints.text_rendering)

    def test_text_primary_route_requires_text_layout_as_primary_goal(self) -> None:
        plan = instruction_plan(
            task_family="text_or_label_heavy",
            target_constraints=constraints(text_or_logo=["newspaper front page typography"]),
            downstream_requirements=["visual_identity_analysis", "text_rendering_risk"],
        )
        route = CharacterTaskRouter().route(plan)

        self.assertEqual(route.route_kind, "text_or_label_heavy")
        self.assertIn("text_rendering_risk", route.conditioning_needs)

    def test_layout_prompt_routes_to_layout_sheet(self) -> None:
        plan = instruction_plan(
            task_family="layout_or_sheet",
            target_constraints=constraints(composition=["3x3 reference sheet grid"]),
        )
        route = CharacterTaskRouter().route(plan)

        self.assertEqual(route.route_kind, "layout_or_sheet")
        self.assertIn("layout_planning", route.conditioning_needs)
        self.assertEqual(route.final_editor_constraints.layout_complexity, "layout_heavy")

    def test_local_repair_routes_to_masked_edit(self) -> None:
        plan = instruction_plan(
            task_family="local_repair",
            target_constraints=constraints(action=["fix the hand"]),
            downstream_requirements=["visual_identity_analysis", "region_grounding", "mask_generation"],
        )
        route = CharacterTaskRouter().route(plan)

        self.assertEqual(route.route_kind, "local_repair_or_inpaint")
        self.assertEqual(route.editor_route, "qwen_image_edit_inpaint_or_masked_edit")
        self.assertIn("region_grounding", route.conditioning_needs)
        self.assertTrue(route.final_editor_constraints.mask_required)

    def test_pose_prompt_routes_to_control_condition(self) -> None:
        plan = instruction_plan(
            task_family="pose_transfer",
            target_constraints=constraints(pose=["use pose from Image 2"]),
            downstream_requirements=["visual_identity_analysis", "pose_conditioning", "visual_disambiguation"],
        )
        route = CharacterTaskRouter().route(plan)

        self.assertEqual(route.route_kind, "pose_transfer")
        self.assertEqual(route.editor_route, "qwen_image_edit_with_control_condition")
        self.assertIn("possible_keypoint_map", route.conditioning_needs)
        self.assertTrue(route.final_editor_constraints.pose_conditioning)

    def test_vlm_planner_context_excludes_parser_audit_fields(self) -> None:
        plan = instruction_plan(
            task_family="reference_character_portrait",
            target_constraints=constraints(framing=["close-up face"]),
        )
        route = CharacterTaskRouter().route(plan)
        context = compact_vlm_planner_context(plan, route)

        self.assertIn("instruction_context", context)
        self.assertIn("task_route", context)
        self.assertNotIn("parser", context["instruction_context"])
        self.assertNotIn("raw_model_response", context["instruction_context"])
        self.assertNotIn("capability_registry", context["task_route"])


if __name__ == "__main__":
    unittest.main()
