from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from aigen.cli import build_parser
from aigen.image_tui_model import ImageEditForm
from aigen.manifest_io import sha256_file
from aigen.workflow_artifacts import ImageArtifact, ReferencePackArtifact
from aigen.workflow_compilation import compile_workflow
from aigen.workflow_form_import import image_form_workflow
from aigen.workflow_graph import ImageEditNode, LoraSourceNode
from aigen.workflow_results_tui import WorkflowResults


class WorkflowResultContractTests(unittest.TestCase):
    def test_form_import_preserves_cli_whitespace_default_seed_and_input_order(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            Image.new("RGB", (32, 48)).save(source)
            form = ImageEditForm()
            for name, value in {"model": "flux2-klein", "prompt": " Keep the composition. ",
                                "sampler": " flowmatch-euler ", "scheduler": " flowmatch-dynamic-shift ",
                                "steps": " 4 ", "guidance": "", "strength": "",
                                "width": " 32 ", "height": " 48 "}.items():
                form.field(name).value = value
            form.field("image").value = f" {source} "
            form.remove_slot(form.field("seed").slot_id)
            command, _ = form.generation_command()
            request = build_parser().parse_args(command[3:])
            graph = image_form_workflow(form)
            edit = graph.node("edit")
            compiled = compile_workflow(graph, target_node_ids=("edit",))
            self.assertEqual(compiled.execution_order, ("reference1", "edit"))
            self.assertEqual(edit.config.seed, 0)
            self.assertEqual(edit.config.prompt, request.prompt)
            self.assertEqual(edit.config.sampler, request.sampler)
            self.assertEqual(graph.node("reference1").config.path, str(request.image[0]))

            form.add_slot("seed")
            form.field("seed").value = " 19 "
            other_seed = form.add_slot("seed")
            next(field for field in form.fields if field.slot_id == other_seed).value = " 7 "
            form.add_slot("reference_pack")
            form.field("reference_pack").value = " assets/reference-packs/example.json "
            form.add_slot("image")
            images = [field for field in form.fields if field.slot_kind == "image"]
            images[-1].value = " second.png "
            form.add_slot("lora")
            form.field("lora").value = " weights.safetensors "
            form.field("lora_weight").value = " 0.75 "
            graph = image_form_workflow(form)
            self.assertEqual([node.config.seed for node in graph.nodes if isinstance(node, ImageEditNode)], [19, 7])
            wires = graph.incoming_connections()["edit"]["references"]
            self.assertEqual([graph.node(wire.source.node_id).config.path for wire in wires],
                             [str(source), "second.png", "assets/reference-packs/example.json"])
            lora = next(node for node in graph.nodes if isinstance(node, LoraSourceNode))
            self.assertEqual((lora.config.path, lora.config.weight), ("weights.safetensors", 0.75))

    def test_replaced_or_unverifiable_historical_inputs_never_show_current_pixels(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            Image.new("RGB", (16, 16), "red").save(path)
            original_hash = sha256_file(path)
            image = ImageArtifact(path=str(path), identity=original_hash, content_sha256=original_hash)
            pack = ReferencePackArtifact(path="pack.json", references=(str(path),), identity="pack",
                                         reference_sha256s=(original_hash,))
            inputs = {"references": (image, pack)}
            self.assertTrue(all(preview is not None for _, preview in WorkflowResults._input_previews(inputs)))
            Image.new("RGB", (16, 16), "blue").save(path)
            changed = WorkflowResults._input_previews(inputs)
            self.assertTrue(all(preview is None and "changed" in label for label, preview in changed))
            legacy = image.model_copy(update={"content_sha256": None})
            label, preview = WorkflowResults._input_previews({"references": (legacy,)})[0]
            self.assertIsNone(preview)
            self.assertIn("legacy", label)


if __name__ == "__main__":
    unittest.main()
