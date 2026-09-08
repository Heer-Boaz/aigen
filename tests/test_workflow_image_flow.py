"""Real document/compiler/executor/cache with an explicit CPU generation double."""
import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from aigen.generation.image_edit_batch import ImageEditBatchOutput, ImageEditBatchResult
from aigen.progress import SILENT_STATUS
from aigen.workflow_cache import NodeExecutionProvenance, RevisionedComponent
from aigen.workflow_compilation import compile_workflow, compile_workflow_run
from aigen.workflow_document_io import load_workflow_document, save_workflow_document
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_execution import execute_workflow, WorkflowExecutionError
from aigen.workflow_graph import (
    ImageEditConfig, ImageEditNode, ImageSourceConfig, ImageSourceNode,
    NodePortRef, WorkflowConnection, WorkflowGraph,
)
from aigen.workflow_results import load_node_result, node_result_history, resolve_image_result
from aigen.workflow_run_state import WorkflowRunState


class ImageFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.png"
        Image.new("RGB", (32, 48), (17, 60, 150)).save(self.source)
        graph = WorkflowGraph(name="Image flow", nodes=(
            ImageSourceNode(id="source", title="Source", config=ImageSourceConfig(path=str(self.source))),
            ImageEditNode(id="edit", title="Edit", config=ImageEditConfig(
                backend="flux2-klein", prompt="Retain the input composition.", width=32, height=48, seed=71)),
            # Deliberately unfinished; must not block running up to the collection.
            ImageEditNode(id="next", title="Continue", config=ImageEditConfig(backend="flux2-klein")),
        ), connections=(
            WorkflowConnection(id="input", source=NodePortRef(node_id="source", port="image"), target=NodePortRef(node_id="edit", port="references")),
            WorkflowConnection(id="continue", source=NodePortRef(node_id="edit", port="image"), target=NodePortRef(node_id="next", port="references")),
        ))
        self.buffer = WorkflowEditBuffer(graph)
        self.collection, self.selection = self.buffer.create_image_variants("edit", (71, 72))
        self.calls = []
        self.fail_after_first = False
        provenance = NodeExecutionProvenance(executor=RevisionedComponent(name="cpu-double", revision="1"), backend=RevisionedComponent(name="cpu-double", revision="1"))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("aigen.workflow_execution.workflow_node_provenance", return_value=provenance))
        self.stack.enter_context(patch("aigen.workflow_execution.run_image_edit_batch", side_effect=self.generate))
        self.stack.enter_context(patch("aigen.workflow_execution._node_progress", return_value=SILENT_STATUS))

    def generate(self, request, *, progress, record_dir, on_output):
        outputs = []
        for case in request.cases:
            self.calls.append(case.seed)
            case.output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (case.width, case.height), (case.seed % 255, 40, 90)).save(case.output_path)
            output = ImageEditBatchOutput(case_id=case.id, path=case.output_path, width=case.width, height=case.height, seed=case.seed)
            outputs.append(output)
            on_output(output)
            if self.fail_after_first:
                raise RuntimeError("CPU double: interrupted after a complete image")
        return ImageEditBatchResult(backend=request.backend, outputs=tuple(outputs))

    def run_collection(self):
        return execute_workflow(compile_workflow_run(self.buffer.document, target_node_ids=(self.collection,)), runs_root=self.root, progress=SILENT_STATUS)

    def test_seed_granularity_history_and_original_metadata(self):
        first = self.run_collection()
        candidates = first.terminal_outputs[self.collection]["collection"].candidates
        second = self.run_collection()
        self.assertEqual(self.calls, [71, 72])
        original = load_node_result(first.node_manifests["edit"])
        reused = load_node_result(second.node_manifests["edit"])
        self.assertEqual(reused.status, "reused")
        self.assertEqual(original.details, reused.details)
        self.assertEqual(len(node_result_history(self.root, self.buffer.document.workflow_id, "edit")), 2)
        self.assertEqual(resolve_image_result(candidates[0].reference), candidates[0].image)
        self.buffer.update_node_config("edit", "seed", "73")
        self.run_collection()
        self.assertEqual(self.calls, [71, 72, 73])

    def test_edit_reads_captured_bytes_even_when_source_changes_after_intake(self):
        original = self.source.read_bytes()

        def generate(request, *, progress, record_dir, on_output):
            outputs = []
            for case in request.cases:
                self.calls.append(case.seed)
                case.output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(case.image_paths[0], case.output_path)
                output = ImageEditBatchOutput(case_id=case.id, path=case.output_path,
                                              width=32, height=48, seed=case.seed)
                outputs.append(output)
                on_output(output)
            return ImageEditBatchResult(backend=request.backend, outputs=tuple(outputs))

        def replace_source(event):
            if event.get("status") == "running":
                Image.new("RGB", (32, 48), (0, 255, 0)).save(self.source)

        with patch("aigen.workflow_execution.run_image_edit_batch", side_effect=generate):
            first = execute_workflow(compile_workflow_run(self.buffer.document, target_node_ids=(self.collection,)),
                                     runs_root=self.root, progress=SILENT_STATUS, event_sink=replace_source)
        image = load_node_result(first.node_manifests["edit"]).outputs["image"]
        self.assertEqual(Path(image.path).read_bytes(), original)
        self.source.write_bytes(original)
        resumed = self.run_collection()
        self.assertEqual(self.calls, [71, 72])
        self.assertEqual(load_node_result(resumed.node_manifests["edit"]).status, "reused")
        self.source.unlink()
        saved = load_node_result(first.node_manifests["edit"]).details.inputs["references"][0]
        self.assertEqual(Path(saved.path).read_bytes(), original)

    def test_random_selection_survives_reload_and_cuts_generator_dependencies(self):
        self.buffer.update_node_config("edit", "seed_mode", "random")
        first = self.run_collection()
        candidate = first.terminal_outputs[self.collection]["collection"].candidates[0]
        self.buffer.select_image(self.selection, candidate.reference)
        self.buffer.update_node_config("next", "prompt", "Retain the input composition.")
        self.buffer.update_node_config("next", "seed", "81")
        path = self.root / "flow.json"
        save_workflow_document(self.buffer.document, path)
        loaded = load_workflow_document(path)
        # The old generator can become invalid without changing the saved choice.
        self.buffer = WorkflowEditBuffer(loaded)
        self.buffer.update_node_config("source", "path", "absent.png")
        compiled = compile_workflow_run(self.buffer.document)
        self.assertEqual(set(compiled.execution_order), {self.selection, "next"})
        execute_workflow(compiled, runs_root=self.root, progress=SILENT_STATUS)
        self.assertEqual(self.calls[-1], 81)
        self.assertEqual(len(self.calls), 3)
        state = WorkflowRunState()
        state.start(self.buffer.document)
        state.update("next", "completed")
        self.buffer.update_node_config("edit", "seed", "91")
        self.assertEqual(state.project(self.buffer.document)["next"], "completed")

    def test_batch_failure_keeps_completed_candidate_and_resumes_other_seed(self):
        self.fail_after_first = True
        with self.assertRaisesRegex(WorkflowExecutionError, "interrupted after"):
            self.run_collection()
        history = node_result_history(self.root, self.buffer.document.workflow_id, "edit")
        self.assertEqual(len(history), 1)
        self.assertEqual(load_node_result(history[0]).details.effective_config["seed"], 71)
        self.fail_after_first = False
        self.run_collection()
        self.assertEqual(self.calls, [71, 72])

    def test_unselected_continuation_fails_before_generation(self):
        with self.assertRaisesRegex(ValueError, "choose and save an image"):
            compile_workflow(self.buffer.document)
        self.assertEqual(self.calls, [])

    def test_old_result_without_content_hash_still_resolves_against_immutable_cache(self):
        result = self.run_collection()
        path = result.node_manifests["edit"]
        payload = json.loads(path.read_text())
        image = payload["outputs"]["image"]
        del image["content_sha256"]
        path.write_text(json.dumps(payload))
        candidate = load_node_result(path).candidate(path, "image")
        resolved = resolve_image_result(candidate.reference)
        self.assertEqual(resolved.identity, candidate.image.identity)
        self.assertIsNotNone(resolved.content_sha256)

    def test_legacy_identity_roundtrip_undo_and_rename(self):
        path = self.root / "legacy.json"
        legacy = self.buffer.document.model_dump(mode="json")
        legacy["version"] = 2
        del legacy["workflow_id"]
        path.write_text(json.dumps(legacy))
        loaded = load_workflow_document(path)
        self.assertEqual(loaded.workflow_id, load_workflow_document(path).workflow_id)
        buffer = WorkflowEditBuffer(loaded)
        buffer.rename_graph("Renamed")
        buffer.update_node_title("edit", "New title")
        buffer.undo()
        self.assertEqual(buffer.document.workflow_id, loaded.workflow_id)
        save_workflow_document(buffer.document, self.root / "copy.json")
        self.assertEqual(load_workflow_document(self.root / "copy.json").workflow_id, loaded.workflow_id)
        original_id = self.buffer.document.workflow_id
        self.buffer.undo()
        self.assertEqual(len(self.buffer.document.nodes), 3)
        self.buffer.redo()
        self.assertEqual(self.buffer.document.workflow_id, original_id)


if __name__ == "__main__":
    unittest.main()
