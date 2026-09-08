"""Real character workflow boundaries with explicit neural doubles."""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from aigen.dwpose_control import DWPoseControl
from aigen.character_edit_audit import default_character_audit_config
from aigen.generation.image_edit_batch import ImageEditBatchOutput, ImageEditBatchResult
from aigen.progress import SILENT_STATUS
from aigen.workflow_cache import NodeExecutionProvenance, RevisionedComponent
from aigen.workflow_compilation import compile_workflow_run
from aigen.workflow_document_io import load_workflow_document, save_workflow_document
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_execution import WorkflowExecutionError, execute_workflow
from aigen.workflow_graph import CharacterEditConfig, CharacterEditNode, ImageSourceConfig, ImageSourceNode, NodeKind, NodePortRef, WorkflowConnection, WorkflowGraph
from aigen.workflow_results import load_node_result
from aigen.workflow_templates import create_workflow_node


class CharacterWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "reference.png"
        Image.new("RGB", (640, 960), "blue").save(self.reference)
        self.graph = WorkflowGraph(name="Character flow", nodes=(
            ImageSourceNode(id="source", title="Reference", config=ImageSourceConfig(path=str(self.reference))),
            CharacterEditNode(id="edit", title="Character edit", config=CharacterEditConfig(
                prompt="Retain the input composition.", width=320, height=480, seed=71, upscale_long_side=None)),
        ), connections=(WorkflowConnection(id="ref", source=NodePortRef(node_id="source", port="image"),
                                           target=NodePortRef(node_id="edit", port="references")),))
        self.calls = []
        self.audit_inputs = []
        self.responses = []
        test = self

        class JudgeDouble:
            def __init__(self, config):
                pass

            def judge_candidate(self, prompt, paths):
                test.audit_inputs.append(paths)
                return test.responses.pop(0)

            def close(self):
                pass

        provenance = NodeExecutionProvenance(
            executor=RevisionedComponent(name="cpu-double", revision="1"),
            backend=RevisionedComponent(name="cpu-double", revision="1"))
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("aigen.workflow_execution.workflow_node_provenance", return_value=provenance))
        stack.enter_context(patch("aigen.workflow_execution._node_progress", return_value=SILENT_STATUS))
        stack.enter_context(patch("aigen.character_edit.validate_local_qwen_model"))
        stack.enter_context(patch("aigen.character_edit.QwenVlm", JudgeDouble))
        stack.enter_context(patch("aigen.character_edit.run_image_edit_batch", side_effect=self.generate))
        self.ordinary = stack.enter_context(patch("aigen.workflow_execution.run_image_edit_batch",
                                                side_effect=AssertionError("character edits bypassed audit")))

    def generate(self, request, *, progress, record_dir):
        self.calls.append(request)
        outputs = []
        for case in request.cases:
            case.output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (case.width, case.height), (case.seed % 255, 40, 90)).save(case.output_path)
            outputs.append(ImageEditBatchOutput(case_id=case.id, path=case.output_path, width=case.width, height=case.height, seed=case.seed))
        return ImageEditBatchResult(backend=request.backend, outputs=tuple(outputs))

    def run_graph(self):
        return execute_workflow(compile_workflow_run(self.graph), runs_root=self.root, progress=SILENT_STATUS)

    def test_selection_metadata_cache_and_audit_configuration(self):
        self.responses = ['{"passed":true,"image_index":3,"regions":[]}'] * 2
        first = self.run_graph()
        manifest = load_node_result(first.node_manifests["edit"])
        self.assertEqual(manifest.candidate(first.node_manifests["edit"], "image").seed, 72)
        self.assertEqual(manifest.details.effective_config["seed"], 71)
        self.assertTrue(Path(manifest.details.case_record).is_file())
        self.assertEqual(self.audit_inputs[0][0].read_bytes(), self.reference.read_bytes())
        self.assertNotEqual(self.audit_inputs[0][0], self.reference)
        cached = self.run_graph()
        self.assertEqual(load_node_result(cached.node_manifests["edit"]).status, "reused")
        self.assertEqual(len(self.calls), 1)
        changed_audit = replace(default_character_audit_config(), max_pixels=768 * 28 * 28)
        with patch("aigen.workflow_compilation.default_character_audit_config", return_value=changed_audit):
            self.run_graph()
        self.assertEqual(len(self.calls), 2)
        self.ordinary.assert_not_called()

    def test_rejected_character_never_publishes_successful_cache(self):
        self.responses = ['{"passed":false,"image_index":2,"regions":["whole image"]}'] * 2
        with self.assertRaises(WorkflowExecutionError):
            self.run_graph()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual([case.seed for case in self.calls[1].cases], [73, 74])
        audit_files = tuple(self.root.rglob("audit.json"))
        self.assertEqual(len(audit_files), 1)
        self.assertEqual(json.loads(audit_files[0].read_text())["status"], "failed")
        self.responses = ['{"passed":true,"image_index":2,"regions":[]}']
        self.run_graph()
        self.assertEqual(len(self.calls), 3)

    def test_qwen_uses_original_audit_refs_and_prepared_native_generation_refs(self):
        buffer = WorkflowEditBuffer(self.graph)
        buffer.update_node_config("edit", "backend", "qwen-image-edit-2511-lightning")
        buffer.update_node_config("edit", "max_sequence_length", "768")
        buffer.update_node_config("edit", "guidance_scale", "1.0")
        self.graph = buffer.document
        self.responses = ['{"passed":true,"image_index":2,"regions":[]}']
        self.run_graph()
        request = self.calls[0]
        self.assertEqual(request.max_sequence_length, 768)
        self.assertEqual(request.guidance_scale, 1.0)
        prepared_path = request.cases[0].image_paths[0]
        self.assertNotEqual(prepared_path, self.reference)
        with Image.open(prepared_path) as prepared, Image.open(self.reference) as original:
            self.assertEqual(prepared.size, (640, 960))
            self.assertEqual(prepared.tobytes(), original.tobytes())
        self.assertEqual(self.audit_inputs[0][0].read_bytes(), self.reference.read_bytes())
        self.assertNotEqual(self.audit_inputs[0][0], self.reference)
        self.ordinary.assert_not_called()

    def test_node_factory_serialization_and_explicit_backend_limits(self):
        node = create_workflow_node(NodeKind.CHARACTER_EDIT)
        self.assertEqual(node.config.backend, "flux2-klein")
        self.assertEqual(node.config.upscale_long_side, 2048)
        path = self.root / "workflow.json"
        save_workflow_document(self.graph, path)
        self.assertEqual(load_workflow_document(path), self.graph)
        scene = WorkflowConnection(id="scene", source=NodePortRef(node_id="source", port="image"),
                                   target=NodePortRef(node_id="edit", port="scene"))
        with self.assertRaisesRegex(ValueError, "explicit Qwen"):
            compile_workflow_run(self.graph.model_copy(update={"connections": (*self.graph.connections, scene)}))

    def test_scene_and_pose_share_framing_and_audit_image_numbering(self):
        from aigen.canny_control import CannyControl

        scene_path, pose_path = self.root / "scene.png", self.root / "pose.png"
        Image.new("RGB", (128, 64), "white").save(scene_path)
        Image.new("RGB", (64, 128), "green").save(pose_path)
        original = self.graph
        for mode in ("native", "keypoint"):
            with self.subTest(mode=mode):
                config = CharacterEditConfig(backend="qwen-image-edit-2511-lightning", prompt="Retain the input composition.",
                    pose_mode=mode, structure_control="edge", seed=71, upscale_long_side=None)
                graph = original.model_copy(update={"nodes": (
                    original.node("source"), original.node("edit").model_copy(update={"config":config}),
                    ImageSourceNode(id="scene", title="Scene", config=ImageSourceConfig(path=str(scene_path))),
                    ImageSourceNode(id="pose", title="Pose", config=ImageSourceConfig(path=str(pose_path))),
                ), "connections": (*original.connections, *(
                    WorkflowConnection(id=port, source=NodePortRef(node_id=port, port="image"), target=NodePortRef(node_id="edit", port=port))
                    for port in ("scene", "pose")))})
                self.graph = graph
                # One reference and two context images; candidate images are 4 and 5.
                self.responses = ['{"passed":true,"image_index":4,"regions":[]}']
                control = DWPoseControl(Image.new("RGB", (64, 128), "green"), (0, 32, 64, 96), {}, "cpu", self.root / "det", self.root / "pose")
                with patch("aigen.character_qwen_edit._render_pose_conditioning", return_value=(pose_path, control)), patch(
                    "aigen.character_qwen_edit._render_structure_conditioning", return_value=(scene_path, CannyControl(Image.new("RGB", (128,64), "white"), {}))):
                    self.run_graph()
                case = self.calls[-1].cases[0]
                self.assertEqual((case.width, case.height), (1792, 896))
                self.assertEqual(len(case.image_paths), 3)
                scene_index = 2 if mode == "native" else 1
                with Image.open(case.image_paths[scene_index]) as scene:
                    self.assertEqual(scene.size, (1792, 896))
                    self.assertEqual(scene.getextrema(), ((255,255),) * 3)
                expected = [self.reference, pose_path, scene_path] if mode == "native" else [self.reference, scene_path, pose_path]
                self.assertEqual([path.read_bytes() for path in self.audit_inputs[-1][:3]],
                                 [path.read_bytes() for path in expected])


if __name__ == "__main__":
    unittest.main()
