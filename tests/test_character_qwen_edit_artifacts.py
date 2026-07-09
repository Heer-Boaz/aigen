from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from aigen.character_qwen_edit import (
    PlannedQwenCharacterEdit,
    run_qwen_character_edit,
)


class CharacterQwenEditArtifactTests(unittest.TestCase):
    def test_run_does_not_write_persistent_edit_plan(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            planned = PlannedQwenCharacterEdit(
                reference_paths={},
                edit_cases=(),
            )
            result = {
                "status": "completed",
                "output": {
                    "directory": output_dir.as_posix(),
                    "result": (output_dir / "result.json").as_posix(),
                },
            }

            with (
                patch("aigen.character_qwen_edit._build_qwen_character_edit_plan", return_value=planned),
                patch("aigen.character_qwen_edit.run_qwen_image_edit_cases", return_value=result) as run_cases,
            ):
                returned = run_qwen_character_edit(
                    pack_path=Path("reference_pack.json"),
                    output_dir=output_dir,
                    profile=object(),
                    instruction_parser_config=object(),
                    vlm_config=object(),
                    cases=(),
                    instruction=None,
                    max_side=640,
                    steps=None,
                    true_cfg_scale=None,
                    guidance_scale=None,
                    seed=0,
                    max_sequence_length=512,
                    candidates_per_case=1,
                    output_format="2:3",
                    resolution="1k",
                    overwrite=True,
                    nunchaku_blocks_on_gpu=None,
                    progress=object(),
                )

            self.assertIs(returned, result)
            self.assertFalse((output_dir / "edit_plan.json").exists())
            self.assertNotIn("edit_plan", returned["output"])
            self.assertIsNone(run_cases.call_args.kwargs["manifest_context"])


if __name__ == "__main__":
    unittest.main()
