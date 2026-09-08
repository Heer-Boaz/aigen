"""Actual SAM/graph/cache/refine owners; only neural inference is replaced."""
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from aigen.manifest_io import sha256_file
from aigen.progress import SILENT_STATUS
from aigen.sam_tui_model import SamEditForm
from aigen.workflow_artifacts import ImageArtifact, MaskArtifact
from aigen.workflow_cache import NodeExecutionProvenance, RevisionedComponent
from aigen.workflow_compilation import compile_workflow_run
from aigen.workflow_document_io import load_workflow_document, save_workflow_document
from aigen.workflow_execution import WorkflowExecutionError, execute_workflow
from aigen.workflow_graph import (
    CharacterRefineConfig, CharacterRefineNode, ImageSourceConfig, ImageSourceNode,
    NodePortRef, WorkflowConnection, WorkflowGraph,
)
from aigen.workflow_mask_execution import bind_mask
from aigen.workflow_results import load_node_result
from aigen.workflow_sam_import import sam_form_workflow


class WorkflowMaskFlowTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.png"
        Image.new("RGBA", (67, 101), (20, 40, 80, 180)).save(self.source)
        self.form = SamEditForm()
        self.form.field("input").value = str(self.source)
        graph = sam_form_workflow(self.form)
        refine = CharacterRefineNode(id="refine", title="Regional edit", config=CharacterRefineConfig(
            prompt="Change the selected area to red.", seed=31, candidates=2))
        wires = tuple(WorkflowConnection(id=f"refine-{port}", source=NodePortRef(node_id=node, port=output),
                                        target=NodePortRef(node_id="refine", port=port))
                      for node, output, port in (("source", "image", "source"), ("segment", "mask", "mask")))
        self.graph = graph.model_copy(update={"nodes": (*graph.nodes, refine), "connections": (*graph.connections, *wires)})
        self.batches = []
        self.audit_images = []
        test = self

        class Judge:
            def __init__(self, config):
                pass

            def judge_candidate(self, prompt, paths):
                test.audit_images.append(paths)
                return '{"passed":true,"image_index":3,"regions":[]}'

            def close(self):
                pass

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("aigen.character_edit.QwenVlm", Judge))
        stack.enter_context(patch("aigen.character_edit.validate_local_qwen_model"))
        stack.enter_context(patch("aigen.workflow_execution._node_progress", return_value=SILENT_STATUS))
        provenance = NodeExecutionProvenance(executor=RevisionedComponent(name="test", revision="1"),
                                            backend=RevisionedComponent(name="test", revision="1"))
        stack.enter_context(patch("aigen.workflow_execution.workflow_node_provenance", return_value=provenance))
        stack.enter_context(patch("aigen.sam_commands._build_mask", side_effect=self.segment))
        stack.enter_context(patch("aigen.generation.qwen_image_edit_masked.run_lightx2v_qwen_image_edit", side_effect=self.generate))

    def segment(self, input_path, **kwargs):
        with Image.open(input_path) as source:
            image = source.convert("RGB")
        mask = np.zeros((image.height, image.width), dtype=np.float32)
        mask[20:50, 10:30] = 1
        mask[19, 10:30] = 0.5
        return image, mask

    def generate(self, *, cases, on_output, **kwargs):
        self.batches.append(cases)
        for case in cases:
            self.assertIsNotNone(case.mask)
            for output in case.outputs:
                output.path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (case.width, case.height), "red").save(output.path)
                on_output({"case": case.name, "name": output.name, "path": str(output.path),
                           "width": case.width, "height": case.height, "seed": output.seed})

    def run_graph(self):
        return execute_workflow(compile_workflow_run(self.graph), runs_root=self.root, progress=SILENT_STATUS)

    def test_mask_cache_refine_raw_audit_and_original_rgba_preservation(self):
        path = self.root / "workflow.json"
        save_workflow_document(self.graph, path)
        self.graph = load_workflow_document(path)
        result = self.run_graph()
        mask = load_node_result(result.node_manifests["segment"]).outputs["mask"]
        self.assertIsInstance(mask, MaskArtifact)
        self.assertEqual(mask.source_sha256, sha256_file(self.source))
        self.assertEqual(mask.source_size, (67, 101))
        refine = load_node_result(result.node_manifests["refine"])
        self.assertEqual(refine.candidate(result.node_manifests["refine"], "image").seed, 31)
        self.assertEqual(self.audit_images[0][0].read_bytes(), self.source.read_bytes())
        self.assertNotEqual(self.audit_images[0][0], self.source)
        self.assertEqual(self.audit_images[0][1], Path(mask.path))
        with Image.open(self.source) as original, Image.open(mask.path) as repaint, Image.open(refine.outputs["image"].path) as edited:
            source_pixels, result_pixels = np.asarray(original), np.asarray(edited)
            mask_pixels = np.asarray(repaint)
            self.assertIn(128, np.unique(mask_pixels))
            np.testing.assert_array_equal(source_pixels[mask_pixels == 0], result_pixels[mask_pixels == 0])
            np.testing.assert_array_equal(source_pixels[..., 3], result_pixels[..., 3])
        cached = self.run_graph()
        for node in ("segment", "refine"):
            self.assertEqual(load_node_result(cached.node_manifests[node]).status, "reused")
        self.assertEqual(len(self.batches), 1)
        self.assertEqual(load_node_result(cached.node_manifests["segment"]).outputs["mask"], mask)
        case_records = list(self.root.rglob("cases/candidate-0.json"))
        self.assertEqual(len(case_records), 1)
        record = json.loads(case_records[0].read_text())
        self.assertTrue(record["preservation"]["passed"])
        self.assertTrue(Path(record["decoded"]["path"]).is_file())

    def test_different_source_is_rejected_before_qwen_generation(self):
        other = self.root / "other.png"
        Image.new("RGB", (67, 101), "green").save(other)
        nodes = (*self.graph.nodes, ImageSourceNode(id="other", title="Other", config=ImageSourceConfig(path=str(other))))
        wires = tuple(wire.model_copy(update={"source": NodePortRef(node_id="other", port="image")})
                      if wire.id == "refine-source" else wire for wire in self.graph.connections)
        self.graph = self.graph.model_copy(update={"nodes": nodes, "connections": wires})
        with self.assertRaisesRegex(WorkflowExecutionError, "different source"):
            self.run_graph()
        self.assertEqual(self.batches, [])

    def test_same_source_bytes_reimport_with_different_artifact_identity(self):
        mask = self.root / "mask.png"
        Image.new("L", (67, 101), 127).save(mask)
        checksum = sha256_file(self.source)
        first = bind_mask(ImageArtifact(path=str(self.source), identity="generated-cache-identity", content_sha256=checksum), mask)
        second = bind_mask(ImageArtifact(path=str(self.source), identity=checksum, content_sha256=checksum), mask)
        self.assertEqual(first, second)

    def test_form_import_without_extra_references_keeps_profile_defaults(self):
        mask = self.root / "mask.png"
        Image.new("L", (67, 101), 127).save(mask)
        for name, value in (("operation", "qwen-edit"), ("mask_source", "mask"),
                            ("mask", str(mask)), ("instruction", "Change the selected area."),
                            ("reference_pack", ""), ("true_cfg_scale", ""), ("guidance_scale", "")):
            self.form.set_value(self.form.field(name), value)
        command, _ = self.form.generation_command()
        self.assertNotIn("--pack", command)
        self.assertNotIn("--true-cfg-scale", command)
        self.assertNotIn("--guidance-scale", command)
        self.graph = sam_form_workflow(self.form)
        self.assertNotIn("references", {node.id for node in self.graph.nodes})
        settings = self.graph.node("refine").config
        self.assertEqual((settings.guidance, settings.guidance_scale), (1.0, 1.0))
        result = self.run_graph()
        self.assertEqual(load_node_result(result.node_manifests["refine"]).status, "completed")
        self.assertEqual([path.read_bytes() for path in self.audit_images[0][:2]],
                         [self.source.read_bytes(), mask.read_bytes()])

    def test_region_plan_import_pins_original_source_and_mask_after_reload(self):
        mask = self.root / "mask.png"
        Image.new("L", (67, 101), 127).save(mask)
        plan = self.root / "region.json"
        plan.write_text(json.dumps({"kind": "character-region-plan", "status": "completed",
            "image": {"sha256": sha256_file(self.source)}, "conditioning": {"planned_tools": []},
            "regions": [{"name": "region", "segmentation": {"mask": {"path": str(mask), "sha256": sha256_file(mask)}}}]}))
        for name, value in (("operation", "qwen-edit"), ("mask_source", "region-plan"),
                            ("region_plan", str(plan)), ("region", "region"), ("instruction", "Change the selected area.")):
            self.form.set_value(self.form.field(name), value)
        graph = sam_form_workflow(self.form)
        path = self.root / "import.json"
        save_workflow_document(graph, path)
        binding = load_workflow_document(path).node("mask").config
        source = ImageArtifact(path=str(self.source), identity="source", content_sha256=sha256_file(self.source))
        bind_mask(source, mask, binding)
        Image.new("L", (67, 101), 64).save(mask)
        with self.assertRaisesRegex(ValueError, "mask contents changed"):
            bind_mask(source, mask, binding)
        with self.assertRaisesRegex(ValueError, "different source"):
            bind_mask(source.model_copy(update={"content_sha256": "a" * 64}), mask, binding)


if __name__ == "__main__":
    unittest.main()
