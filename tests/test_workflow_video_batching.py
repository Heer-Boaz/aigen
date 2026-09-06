"""Real graph, video probing and cache; only neural generation is doubled."""
from contextlib import ExitStack
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from aigen.generation.animegen_i2v import AnimeGenI2VResult
from aigen.generation.ltx23_keyframes import Ltx23KeyframesResult
from aigen.progress import SILENT_STATUS
from aigen.video_tui_model import ANIMEGEN_BACKEND, VideoForm
from aigen.workflow_cache import NodeExecutionProvenance, RevisionedComponent
from aigen.workflow_compilation import compile_workflow_run
from aigen.workflow_edit_buffer import WorkflowEditBuffer
from aigen.workflow_execution import WorkflowExecutionError, execute_workflow
from aigen.workflow_form_import import video_form_workflow
from aigen.workflow_results import load_node_result, node_result_history


class WorkflowVideoBatchingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "source.png"
        Image.new("RGB", (64, 64), "blue").save(self.image)
        self.calls = []
        self.fail_after_first = False
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        component = RevisionedComponent(name="CPU neural double", revision="1")
        provenance = NodeExecutionProvenance(executor=component, backend=component)
        self.stack.enter_context(patch("aigen.workflow_execution.workflow_node_provenance", return_value=provenance))
        self.stack.enter_context(patch("aigen.workflow_execution._node_progress", return_value=SILENT_STATUS))
        for backend in ("animegen", "ltx23"):
            self.stack.enter_context(patch(f"aigen.workflow_compilation.resolve_{backend}_installation", return_value=None))
        self.stack.enter_context(patch("aigen.workflow_video_execution.generate_animegen_i2v_seed_sweep", side_effect=self.generate))
        self.stack.enter_context(patch("aigen.workflow_video_execution.generate_ltx23_keyframes_seed_sweep", side_effect=self.generate))

    def graph(self, backend):
        form = VideoForm()
        if backend == "anime":
            form.set_value(form.field("backend"), ANIMEGEN_BACKEND)
        form.field("prompt").value = "Retain the input composition."
        form.field("frames").value = "17"
        form.field("keyframe_fit").value = "pad"
        if backend == "ltx":
            form.field("resolution").value = "64x64"
        for field in form.fields:
            if field.name in ("image", "keyframe"):
                form.set_value(field, str(self.image))
            elif field.name == "frame" and field.value != "0":
                form.set_value(field, "16")
            elif field.name == "seed":
                form.set_value(field, "17")
        slot = form.add_slot("seed")
        form.set_value(next(field for field in form.fields if field.slot_id == slot), "19")
        return WorkflowEditBuffer(video_form_workflow(form))

    def generate(self, *, output, seeds, on_output, **request):
        self.calls.append(tuple(seeds))
        output.parent.mkdir(parents=True, exist_ok=True)
        results = []
        for seed in seeds:
            path = output.with_name(f"{output.stem}-{seed}.mp4")
            config = path.with_suffix(".json")
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                f"color=c=blue:s=64x64:r={request['fps']}", "-frames:v", str(request["frames"]),
                "-c:v", "libx264", "-bf", "0", str(path),
            ], check=True, capture_output=True)
            config.write_text('{"neural_double":true}')
            common = dict(output=path, config=config, frames=request["frames"], fps=request["fps"],
                          seed=seed, steps=request["steps"], elapsed_seconds=0, keyframe_fit=request["keyframe_fit"])
            if "sampling" in request:
                result = AnimeGenI2VResult(**common, image=request["image"], last_image=request["last_image"],
                    width=64, height=64, sampling=request["sampling"], precision=request["precision"],
                    guidance=1, flow_shift=3, scheduler="CPU double", cuda_memory={})
            else:
                result = Ltx23KeyframesResult(**common, log=config, keyframes=request["keyframes"],
                    resolution=request["resolution"], phases=request["phases"], solver=request["solver"],
                    negative_prompt=request["negative_prompt"], conditioning_strength=request["conditioning_strength"],
                    model=request["model"], model_type="CPU double", phase_metrics=(), environment={})
            on_output(result)
            results.append(result)
            if self.fail_after_first:
                raise RuntimeError("CPU double: interrupted after a complete video")
        return tuple(results)

    def execute(self, buffer):
        return execute_workflow(compile_workflow_run(buffer.document), runs_root=self.root / "workflows", progress=SILENT_STATUS)

    def test_compatible_seeds_share_one_sweep_and_cache_invalidation_is_per_seed(self):
        for backend in ("anime", "ltx"):
            with self.subTest(backend=backend):
                self.calls.clear()
                buffer = self.graph(backend)
                first = self.execute(buffer)
                self.assertEqual(self.calls, [(17, 19)])
                a, b = (load_node_result(first.node_manifests[node]) for node in ("video1", "video2"))
                self.assertEqual(a.details.record_dir, b.details.record_dir)
                self.assertNotEqual(a.details.case_record, b.details.case_record)
                self.assertTrue(Path(a.details.case_record).is_file())
                self.execute(buffer)
                self.assertEqual(self.calls, [(17, 19)])
                buffer.update_node_config("video2", "seed", "23")
                changed = self.execute(buffer)
                self.assertEqual(self.calls, [(17, 19), (23,)])
                self.assertEqual(load_node_result(changed.node_manifests["video1"]).status, "reused")

    def test_completed_video_survives_later_seed_failure_and_resumes(self):
        for backend in ("anime", "ltx"):
            with self.subTest(backend=backend):
                self.calls.clear()
                buffer = self.graph(backend)
                self.fail_after_first = True
                with self.assertRaisesRegex(WorkflowExecutionError, "interrupted after"):
                    self.execute(buffer)
                history = node_result_history(self.root / "workflows", buffer.document.workflow_id, "video1")
                self.assertEqual(len(history), 1)
                self.assertEqual(load_node_result(history[0]).status, "completed")
                self.fail_after_first = False
                result = self.execute(buffer)
                self.assertEqual(self.calls, [(17, 19), (19,)])
                self.assertEqual(load_node_result(result.node_manifests["video1"]).status, "reused")

    def test_different_fit_policies_do_not_share_a_sweep(self):
        buffer = self.graph("anime")
        buffer.update_node_config("video2", "keyframe_fit", "crop")
        self.execute(buffer)
        self.assertEqual(self.calls, [(17,), (19,)])

    def test_same_seed_nodes_generate_once_and_both_publish(self):
        buffer = self.graph("anime")
        buffer.update_node_config("video2", "seed", "17")
        result = self.execute(buffer)
        self.assertEqual(self.calls, [(17,)])
        self.assertEqual(result.terminal_outputs["video1"], result.terminal_outputs["video2"])


if __name__ == "__main__":
    unittest.main()
