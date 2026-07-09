from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from aigen.character_reference_models import load_completed_character_reference_pack
from aigen.character_reference_observation_models import CharacterReferenceObservationError
from aigen.character_reference_observer import observe_character_references
from aigen.character_reference_pack import build_character_reference_pack, character_planner_reference_map
from aigen.manifest_io import read_json


def write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


class FakeObservationRunner:
    def __init__(self, *, wrong_reference_id: bool = False) -> None:
        self.wrong_reference_id = wrong_reference_id
        self.prompts: list[str] = []
        self.image_paths: list[list[Path]] = []

    def describe_image(self, prompt: str, image_paths: list[Path]) -> str:
        self.prompts.append(prompt)
        self.image_paths.append(image_paths)
        reference_id = prompt.split("Reference id: ", 1)[1].splitlines()[0]
        if self.wrong_reference_id:
            reference_id = "reference999"
        return json.dumps(
            {
                "reference_id": reference_id,
                "visual_summary": f"observed {reference_id}",
                "visible_subjects": [],
                "view_or_framing": "single fake view",
                "visible_components": [],
                "occlusion_or_quality_notes": [],
                "text_or_symbol_notes": [],
                "uncertainties": [],
            }
        )


class CharacterReferenceObserverTests(unittest.TestCase):
    def test_observer_calls_vlm_once_per_neutral_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pack, reference_paths = build_pack(root)
            runner = FakeObservationRunner()

            observations = observe_character_references(
                runner=runner,
                pack=pack,
                reference_paths=reference_paths,
                planner_reference_map=character_planner_reference_map(pack),
                planner_context={"task_route": {"route_kind": "portrait_identity_generation"}},
                path_label="test-pack#case",
            )

        self.assertEqual([len(paths) for paths in runner.image_paths], [1, 1])
        self.assertEqual(sorted(observations.observations), ["reference1", "reference2"])
        self.assertEqual(observations.observations["reference1"].reference_id, "reference1")
        self.assertIn("single-reference visual observation", runner.prompts[0])
        self.assertIn("portrait_identity_generation", runner.prompts[0])
        self.assertIn("Determine view_or_framing only from the pixels", runner.prompts[0])
        self.assertIn("Do not use filename, pack reference label, neutral reference id, user request, or planner context as truth", runner.prompts[0])
        self.assertIn("Separate visible reference evidence from user-requested output constraints", runner.prompts[0])
        self.assertIn("art_style or rendering_style", runner.prompts[0])
        self.assertIn("reference background as context", runner.prompts[0])

    def test_observer_rejects_mismatched_reference_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pack, reference_paths = build_pack(root)
            runner = FakeObservationRunner(wrong_reference_id=True)

            with self.assertRaises(CharacterReferenceObservationError):
                observe_character_references(
                    runner=runner,
                    pack=pack,
                    reference_paths=reference_paths,
                    planner_reference_map=character_planner_reference_map(pack),
                    planner_context={},
                    path_label="test-pack#case",
                )


def build_pack(root: Path):
    refs = {
        "asset_a": root / "reference-a.png",
        "asset_b": root / "reference-b.png",
    }
    write_image(refs["asset_a"], (80, 144), (50, 50, 200))
    write_image(refs["asset_b"], (96, 96), (50, 200, 50))
    build_character_reference_pack(
        character_id="subject",
        references=refs,
        output_dir=root / "references",
        overwrite=False,
    )
    pack_path = root / "references" / "reference_pack.json"
    pack = load_completed_character_reference_pack(
        read_json(pack_path, label="character reference pack"),
        path_label=pack_path.as_posix(),
    )
    return pack, refs


if __name__ == "__main__":
    unittest.main()
