"""Raw/audit/publication contracts with explicit neural and upscale doubles."""
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from aigen.character_edit import CharacterEditError, run_character_edit
from aigen.character_edit_audit import AuditContextImage, CharacterAuditError, CharacterAuditGroup
from aigen.generation.image_batch_postprocess import ImageBatchPostprocessResult
from aigen.generation.image_edit_batch import ImageEditBatchCase, ImageEditBatchOutput, ImageEditBatchRequest, ImageEditBatchResult
from aigen.progress import SILENT_STATUS
from aigen.vlm_qwen import QwenVlmError


class CharacterEditAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference.png"
        self.context = self.root / "context.png"
        Image.new("RGB", (32, 48), "blue").save(self.reference)
        Image.new("RGB", (48, 32), "green").save(self.context)
        self.events = []
        self.responses = []
        self.audit_loaded = False
        self.generation_loaded = False
        test = self

        class JudgeDouble:
            def __init__(self, config):
                test.assertFalse(test.generation_loaded)
                test.assertFalse(test.audit_loaded)
                test.audit_loaded = True
                test.events.append("audit load")

            def judge_candidate(self, prompt, paths):
                test.assertTrue(test.audit_loaded)
                test.assertFalse(test.generation_loaded)
                test.assertEqual(paths[:2], [test.reference, test.context])
                test.assertIn("Image 2: scene source.", prompt)
                test.assertIn("requested change of style", prompt)
                return test.responses.pop(0)

            def close(self):
                test.audit_loaded = False
                test.events.append("audit close")

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("aigen.character_edit.validate_local_qwen_model"))
        self.stack.enter_context(patch("aigen.character_edit.QwenVlm", JudgeDouble))
        self.stack.enter_context(patch("aigen.character_edit.run_image_edit_batch", side_effect=self.generate))
        self.upscale = self.stack.enter_context(patch("aigen.character_edit.postprocess_image_batch", side_effect=self.postprocess))

    def request(self, groups=1):
        cases = []
        audit_groups = []
        for group_index in range(groups):
            ids = tuple(f"g{group_index}c{index}" for index in range(2))
            for index, case_id in enumerate(ids):
                cases.append(ImageEditBatchCase(id=case_id, prompt="Retain the input composition.",
                    image_paths=(self.reference, self.context), width=32, height=48,
                    seed=71 + 28 * (group_index * 2 + index), output_path=self.root / f"{case_id}.png"))
            audit_groups.append(CharacterAuditGroup(
                id=f"group{group_index}", instruction="Retain the input composition.", route="scene_insertion",
                references=(self.reference,), context_images=(AuditContextImage("scene source", self.context),), candidate_ids=ids,
            ))
        request = ImageEditBatchRequest(backend="flux2-klein", cases=tuple(cases),
            sampler="flowmatch-euler", scheduler="flowmatch-dynamic-shift")
        return request, tuple(audit_groups)

    def generate(self, request, *, progress, record_dir):
        self.assertFalse(self.audit_loaded)
        self.generation_loaded = True
        self.events.append(tuple((case.id, case.seed, case.prompt) for case in request.cases))
        outputs = []
        for case in request.cases:
            case.output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (case.width, case.height), (case.seed % 255, 70, 40)).save(case.output_path)
            outputs.append(ImageEditBatchOutput(case_id=case.id, path=case.output_path, width=case.width, height=case.height, seed=case.seed))
        self.generation_loaded = False
        self.events.append("generation close")
        return ImageEditBatchResult(backend=request.backend, outputs=tuple(outputs))

    def postprocess(self, paths, output_dir, *, model, output_names, **kwargs):
        self.assertFalse(self.audit_loaded)
        self.assertFalse(self.generation_loaded)
        self.events.append("upscale")
        output_dir.mkdir()
        outputs = tuple(output_dir / name for name in output_names)
        for source, destination in zip(paths, outputs, strict=True):
            destination.write_bytes(source.read_bytes())
        return ImageBatchPostprocessResult(model, output_dir, outputs, (), 0)

    def execute(self, *, groups=1, upscale=None):
        request, audit_groups = self.request(groups)
        return run_character_edit(request, audit_groups=audit_groups, output_dir=self.root / "edit",
            progress=SILENT_STATUS, upscale_long_side=upscale)

    def test_best_of_n_is_selected_on_raw_before_one_upscale_batch(self):
        self.responses = ['{"passed":true,"image_index":4,"regions":[]}'] * 2
        result = self.execute(groups=2, upscale=2048)
        self.assertEqual(result.iterations, 1)
        self.assertEqual([selected.raw.case_id for selected in result.selections], ["g0c1", "g1c1"])
        self.assertEqual(self.events.count("audit load"), 1)
        self.assertEqual(self.events[-2:], ["audit close", "upscale"])
        self.assertEqual(self.upscale.call_count, 1)
        self.assertEqual(len(self.upscale.call_args.args[0]), 2)
        self.assertTrue(all(selected.raw.path.is_file() for selected in result.selections))
        self.assertTrue(all(len(selected.candidate_identity) == 64 for selected in result.selections))

    def test_retry_uses_new_seeds_and_only_rejected_groups(self):
        self.responses = [
            '{"passed":true,"image_index":4,"regions":[]}',
            '{"passed":false,"image_index":3,"regions":["whole image"]}',
            '{"passed":true,"image_index":3,"regions":[]}',
        ]
        result = self.execute(groups=2)
        calls = [event for event in self.events if isinstance(event, tuple)]
        self.assertEqual(len(calls[0]), 4)
        self.assertEqual([item[:2] for item in calls[1]], [("g1c0", 156), ("g1c1", 157)])
        self.assertEqual({item[2] for call in calls for item in call}, {"Retain the input composition."})
        self.assertEqual(self.events.count("audit load"), 2)
        self.assertEqual(result.iterations, 2)
        self.assertEqual([selection.raw.seed for selection in result.selections], [99, 156])
        report = json.loads(result.report.read_text())
        self.assertEqual(report["rounds"][0]["decisions"][1]["response"], "regenerate_with_new_seeds")
        self.assertEqual(len(tuple((self.root / "edit").glob("round-*/raw/*.png"))), 6)
        self.upscale.assert_not_called()

    def test_two_rejected_rounds_fail_with_evidence_and_no_successful_output(self):
        self.responses = ['{"passed":false,"image_index":3,"regions":["whole image"]}'] * 2
        with self.assertRaisesRegex(CharacterEditError, "after 2 rounds"):
            self.execute(upscale=2048)
        self.assertFalse((self.root / "edit/result.json").exists())
        report = json.loads((self.root / "edit/audit.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["rounds"]), 2)
        self.assertEqual(report["rounds"][-1]["decisions"][0]["response"], "fail")
        self.assertFalse(self.audit_loaded)
        self.upscale.assert_not_called()

    def test_retry_seeds_do_not_depend_on_other_groups_in_the_batch(self):
        observed = []
        for count in (1, 2):
            self.responses = ['{"passed":false,"image_index":3,"regions":["whole image"]}']
            if count == 2:
                self.responses.append('{"passed":true,"image_index":3,"regions":[]}')
            self.responses.append('{"passed":true,"image_index":3,"regions":[]}')
            self.events.clear()
            request, groups = self.request(groups=count)
            run_character_edit(request, audit_groups=groups, output_dir=self.root / f"batch-{count}", progress=SILENT_STATUS)
            calls = [event for event in self.events if isinstance(event, tuple)]
            observed.append(calls[-1])
        self.assertEqual(observed[0], observed[1])
        self.assertEqual([item[1] for item in observed[0]], [100, 101])

    def test_invalid_vlm_json_or_index_fails_without_generation_retry(self):
        for raw in ("not JSON", '{"passed":true,"image_index":5,"regions":[]}',
                    '{"passed":"true","image_index":3,"regions":[]}',
                    '{"passed":true,"image_index":3,"regions":["whole image"]}'):
            with self.subTest(raw=raw), TemporaryDirectory(dir=self.root) as temporary:
                self.responses = [raw]
                self.events.clear()
                request, groups = self.request()
                output = Path(temporary) / "edit"
                with self.assertRaises(CharacterAuditError):
                    run_character_edit(request, audit_groups=groups, output_dir=output, progress=SILENT_STATUS)
                self.assertEqual(self.events.count("generation close"), 1)
                self.assertFalse(self.audit_loaded)
                response = json.loads((output / "round-1/audit/group-0/response.json").read_text())
                self.assertEqual(response["raw_response"], raw)
                self.assertFalse((output / "result.json").exists())

    def test_later_audit_error_retains_earlier_group_decision_in_main_report(self):
        self.responses = ['{"passed":true,"image_index":4,"regions":[]}', 'not JSON']
        with self.assertRaises(CharacterAuditError):
            self.execute(groups=2)
        report = json.loads((self.root / "edit/audit.json").read_text())
        self.assertEqual(report["rounds"][0]["seeds"], {"g0c0": 71, "g0c1": 99, "g1c0": 127, "g1c1": 155})
        self.assertEqual(report["rounds"][0]["decisions"][0]["group_id"], "group0")
        self.assertTrue(report["rounds"][0]["decisions"][0]["passed"])

    def test_missing_audit_model_fails_before_generation(self):
        with patch("aigen.character_edit.validate_local_qwen_model", side_effect=QwenVlmError("missing audit shard")):
            with self.assertRaisesRegex(QwenVlmError, "missing audit shard"):
                self.execute()
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
