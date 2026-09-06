from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from aigen.cli import build_parser
from aigen.generation.ltx23_wangp_worker import _build_wangp_settings
from aigen.image_io import fit_image_canvas
from aigen.model_artifacts import local_model_files
from aigen.video_tui_model import VideoForm, ANIMEGEN_BACKEND, HUNYUANVIDEO15_BACKEND
from aigen.workflow_compilation import compile_workflow_run, execution_config_payload
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_form_import import video_form_workflow
from aigen.workflow_graph import HunyuanI2VNode, Ltx23Node
from aigen.workflow_document_io import save_workflow_document, load_workflow_document


class WorkflowVideoContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "image.png"
        Image.new("RGB", (32, 48), "blue").save(self.source)

    def populate(self, form):
        form.field("prompt").value = " The visible subject blinks. "
        for field in form.fields:
            if field.name in ("keyframe", "image"):
                form.set_value(field, f" {self.source} ")

    def test_form_cli_graph_and_adapter_keep_ltx_positions_seeds_and_effective_canvas(self):
        form = VideoForm()
        self.populate(form)
        form.field("resolution").value = " 513x641 "
        form.field("frames").value = "33"
        form.field("steps").value = "8"
        form.field("solver").value = "distilled_8_steps"
        form.field("keyframe_fit").value = "pad"
        fields = [f for f in form.fields if f.name == "frame"]
        form.set_value(fields[1], "32")
        slot = form.add_slot("keyframe")
        form.set_value(next(f for f in form.fields if f.name == "keyframe" and f.slot_id == slot), str(self.source))
        form.set_value(next(f for f in form.fields if f.name == "frame" and f.slot_id == slot), "16")
        slot = form.add_slot("seed")
        form.set_value(next(f for f in form.fields if f.slot_id == slot), "19")
        command, _, _ = form.generation_command()
        cli = build_parser().parse_args(command[3:])
        graph = video_form_workflow(form)
        document = self.root / "video.json"
        save_workflow_document(graph, document)
        graph = load_workflow_document(document)
        compiled = compile_workflow_run(graph)
        nodes = [n for n in graph.nodes if isinstance(n, Ltx23Node)]
        self.assertEqual([n.config.seed for n in nodes], cli.seed)
        self.assertEqual([graph.node(w.source.node_id).config.frame for w in compiled.node("video1").incoming["keyframes"]], [0, 32, 16])
        config = compiled.node("video1").config
        self.assertEqual((config.settings.width, config.settings.height), (576, 704))
        self.assertEqual(config.settings.keyframe_fit, cli.keyframe_fit)
        request = {**execution_config_payload(config), "resolution": config.settings.resolution,
                   "keyframes": [{"image": f"frame{position}.png", "frame": position} for position in (0, 32, 16)]}
        native = _build_wangp_settings(request, {}, "ltx2_22B_nvfp4")
        self.assertEqual(native["image_start"], "frame0.png")
        self.assertEqual(native["image_end"], "frame32.png")
        self.assertEqual(native["image_refs"], ["frame16.png"])
        self.assertEqual(native["frames_positions"], "17")
        buffer = WorkflowEditBuffer(graph)
        buffer.update_node_config("keyframe3", "frame", "32")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compile_workflow_run(buffer.document)

    def test_anime_form_fit_and_hunyuan_missing_runtime_preflight(self):
        form = VideoForm()
        form.set_value(form.field("backend"), ANIMEGEN_BACKEND)
        self.populate(form)
        form.field("keyframe_fit").value = "pad"
        form.field("frames").value = "33"
        form.add_slot("image")
        self.populate(form)
        command, _, _ = form.generation_command()
        cli = build_parser().parse_args(command[3:])
        compiled = compile_workflow_run(video_form_workflow(form))
        node = compiled.node("video1")
        self.assertEqual(node.config.settings.keyframe_fit, cli.keyframe_fit)
        self.assertEqual(set(node.incoming), {"start", "end"})
        form.set_value(form.field("backend"), HUNYUANVIDEO15_BACKEND)
        self.populate(form)
        graph = video_form_workflow(form)
        self.assertIsInstance(graph.node("video1"), HunyuanI2VNode)
        with patch("aigen.generation.hunyuanvideo15._runtime_root", return_value=self.root / "missing"):
            with self.assertRaisesRegex(ValueError, "Hunyuan node video1.*runtime"):
                compile_workflow_run(graph)

    def test_fit_policy_changes_only_the_prepared_canvas(self):
        image = Image.new("RGB", (40, 20), "blue")
        for mode in ("pad", "crop", "stretch"):
            with self.subTest(mode=mode):
                fitted = fit_image_canvas(image, (64, 64), mode)
                self.assertEqual(fitted.size, (64, 64))
                self.assertEqual(fitted.getpixel((32, 32)), (0, 0, 255))
                self.assertEqual(fitted.getpixel((0, 0)), (255, 255, 255) if mode == "pad" else (0, 0, 255))
        self.assertEqual(image.size, (40, 20))

    def test_missing_checkpoint_shard_is_rejected_before_loading(self):
        index = self.root / "model.safetensors.index.json"
        index.write_text('{"weight_map":{"a":"part.safetensors","b":"part.safetensors"}}')
        with self.assertRaisesRegex(FileNotFoundError, "part.safetensors"):
            local_model_files((index,))
        shard = self.root / "part.safetensors"
        shard.write_bytes(b"CPU fixture")
        self.assertEqual(local_model_files((index,)), (index, shard))


if __name__ == "__main__":
    unittest.main()
