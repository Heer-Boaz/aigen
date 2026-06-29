from __future__ import annotations

import unittest

from pathlib import Path

from aigen.character_training_data import (
    TRAINING_VALIDATION_MAX_NEW_TOKENS,
    TRAINING_VALIDATION_MAX_PIXELS,
    TRAINING_VALIDATION_MIN_PIXELS,
    TRAINING_PLANNING_MAX_NEW_TOKENS,
    TRAINING_PLANNING_MAX_PIXELS,
    TRAINING_PLANNING_MIN_PIXELS,
    PlannedTrainingView,
    TrainingIdentityPlan,
    TrainingViewPlan,
    _training_planning_config,
    _compile_training_view,
    _training_validation_config,
    _validate_training_plan,
)
from aigen.character_view_models import (
    CharacterViewBankSpec,
    CharacterViewError,
    ImageAssetSpec,
    ViewBankCharacterSpec,
    ViewBankEntrySpec,
    ViewBankViewSpec,
)
from aigen.vlm_qwen import QwenVlmConfig


def image_asset(path: str) -> ImageAssetSpec:
    return ImageAssetSpec(path=path, sha256="0" * 64, mode="RGB", width=64, height=96)


def view_bank() -> CharacterViewBankSpec:
    return CharacterViewBankSpec(
        kind="character-view-bank",
        character=ViewBankCharacterSpec(
            id="ai51",
            source_reference=image_asset("front.png"),
            identity_notes=[
                "AI51 wears a brown leather skirt as the waist garment.",
                "AI51 wears a separate belt at the waist with a silver buckle.",
            ],
        ),
        views={
            "left_profile": ViewBankEntrySpec(
                view=ViewBankViewSpec(name="left_profile", camera="orthographic-side", pose="neutral-standing"),
                image=image_asset("left-profile.png"),
                accepted_candidate="seed_001",
            )
        },
    )


def planned_view() -> PlannedTrainingView:
    identity = (
        "right profile neutral full-body view, brown leather skirt as waist garment, "
        "separate belt at the waist with silver buckle"
    )
    return PlannedTrainingView(
        name="right_profile_neutral_full_body_view",
        intent="right profile neutral full-body view",
        identity_primer_view="left_profile",
        camera=f"right profile orthographic side camera, {identity}",
        pose=f"neutral standing right profile full body pose, {identity}",
        clip=f"AI51 anime girl, {identity}, plain light neutral background",
        t5=f"AI51 anime girl, {identity}, plain light neutral background",
        acceptance_checks=[
            "right profile view",
            "brown leather skirt as waist garment",
            "separate belt at the waist with silver buckle",
        ],
    )


def good_identity_details() -> dict[str, str]:
    return {
        "subject": "anime girl",
        "proportions": "slim full-body character proportions",
        "hair": "short pink bob",
        "face": "blue eyes",
        "upper_clothing": "brown leather jacket over white shirt",
        "waist_garment": "brown leather skirt",
        "legwear": "blue thigh-high socks",
        "footwear": "brown boots",
        "accessories": "separate belt at the waist with silver buckle",
        "materials": "glossy brown leather",
        "palette": "pink hair, brown leather, white shirt, blue tie and socks",
        "style": "clean anime character art",
    }


class CharacterTrainingDataTests(unittest.TestCase):
    def test_training_plan_preserves_approved_skirt_and_belt_notes(self) -> None:
        plan = TrainingViewPlan(
            identity_caption=(
                "AI51 anime girl with brown leather skirt as the waist garment and a separate belt at "
                "the waist with a silver buckle"
            ),
            identity_details=good_identity_details(),
            views=[planned_view()],
        )

        _validate_training_plan(view_bank(), plan, ["right profile neutral full-body view"])

    def test_training_plan_rejects_missing_accessories_slot(self) -> None:
        details = good_identity_details()
        del details["accessories"]
        plan = TrainingViewPlan(
            identity_caption=(
                "AI51 anime girl with brown leather skirt as the waist garment and a separate belt at "
                "the waist with a silver buckle"
            ),
            identity_details=details,
            views=[planned_view()],
        )

        with self.assertRaisesRegex(CharacterViewError, "omits identity slots: accessories"):
            _validate_training_plan(view_bank(), plan, ["right profile neutral full-body view"])

    def test_training_plan_rejects_belt_without_skirt_identity_note(self) -> None:
        plan = TrainingViewPlan(
            identity_caption="AI51 anime girl with a separate belt at the waist with a silver buckle",
            identity_details=good_identity_details() | {"waist_garment": "brown belt"},
            views=[planned_view()],
        )

        with self.assertRaisesRegex(CharacterViewError, "brown leather skirt"):
            _validate_training_plan(view_bank(), plan, ["right profile neutral full-body view"])

    def test_training_validation_uses_bounded_qwen_budget(self) -> None:
        config = QwenVlmConfig(
            judge_id="qwen",
            model=Path("models/qwen"),
            repo_id="repo",
            revision="rev",
            dtype="bfloat16",
            attention_impl="sdpa",
            quantization="bitsandbytes-8bit",
            min_pixels=512 * 28 * 28,
            max_pixels=512 * 28 * 28,
            max_new_tokens=4096,
            temperature=0.0,
        )

        bounded = _training_validation_config(config)

        self.assertEqual(bounded.min_pixels, TRAINING_VALIDATION_MIN_PIXELS)
        self.assertEqual(bounded.max_pixels, TRAINING_VALIDATION_MAX_PIXELS)
        self.assertEqual(bounded.max_new_tokens, TRAINING_VALIDATION_MAX_NEW_TOKENS)

    def test_training_planning_uses_bounded_qwen_budget(self) -> None:
        config = QwenVlmConfig(
            judge_id="qwen",
            model=Path("models/qwen"),
            repo_id="repo",
            revision="rev",
            dtype="bfloat16",
            attention_impl="sdpa",
            quantization="bitsandbytes-8bit",
            min_pixels=512 * 28 * 28,
            max_pixels=512 * 28 * 28,
            max_new_tokens=4096,
            temperature=0.0,
        )

        bounded = _training_planning_config(config)

        self.assertEqual(bounded.min_pixels, TRAINING_PLANNING_MIN_PIXELS)
        self.assertEqual(bounded.max_pixels, TRAINING_PLANNING_MAX_PIXELS)
        self.assertEqual(bounded.max_new_tokens, TRAINING_PLANNING_MAX_NEW_TOKENS)

    def test_compile_training_view_uses_model_extracted_identity_details(self) -> None:
        identity = TrainingIdentityPlan(
            identity_caption="ai51char anime girl with brown leather skirt and separate belt",
            identity_details=good_identity_details(),
        )

        view = _compile_training_view(
            view_bank(),
            identity,
            "back neutral full-body view",
        )

        self.assertEqual(view.intent, "back neutral full-body view")
        self.assertIn("brown leather skirt", view.t5)
        self.assertIn("separate belt at the waist with silver buckle", view.t5)
        self.assertIn("back neutral full-body view", view.clip)
        self.assertLess(len(view.t5.split()), 90)

    def test_compile_training_view_uses_available_profile_primer_for_opposite_profile(self) -> None:
        identity = TrainingIdentityPlan(
            identity_caption="ai51char anime girl with brown leather skirt and separate belt",
            identity_details=good_identity_details(),
        )

        view = _compile_training_view(
            view_bank(),
            identity,
            "right profile neutral full-body view",
        )

        self.assertEqual(view.identity_primer_view, "left_profile")


if __name__ == "__main__":
    unittest.main()
