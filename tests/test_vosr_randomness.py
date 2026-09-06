"""Pinned upstream tiled inference plus real batch/cache code; neural layers run as CPU doubles."""
import ast
from contextlib import ExitStack, nullcontext
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
import torch.nn.functional as functional
from torchvision import transforms

from aigen.generation import vosr_backend
from aigen.generation.vosr_runtime import VosrRuntime
from aigen.manifest_io import sha256_file
from aigen.progress import SILENT_STATUS
from aigen.workflow_compilation import compile_workflow_run
from aigen.workflow_execution import execute_workflow
from aigen.workflow_graph import (
    ImageSourceNode, ImageSourceConfig, ImagePostprocessNode, NodePortRef,
    WorkflowConnection, WorkflowGraph,
)
from aigen.workflow_templates import default_postprocess_config


class VosrRandomnessTests(unittest.TestCase):
    def test_output_is_independent_of_batch_order_and_partial_cache(self):
        upstream = vosr_backend._vosr_source_root() / "inference_vosr.py"
        if not upstream.is_file():
            self.skipTest("Install the pinned VOSR source to run its inference contract test")
        names = {"_make_tile_grid", "_gaussian_weights", "tiled_latent_inference"}
        functions = [node for node in ast.parse(upstream.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in functions}, names)
        namespace = {
            "torch": torch, "math": math,
            "_encode_latent": lambda vae, x, args, device: (torch.randn(1, 3, x.shape[-2] // 8, x.shape[-1] // 8), None, None),
            "_decode_latent": lambda vae, x, *args: functional.interpolate(x, scale_factor=8).clamp(-1, 1),
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(upstream), "exec"), namespace)
        fork_rng = torch.random.fork_rng
        cpu_torch = SimpleNamespace(
            default_generator=torch.default_generator,
            random=SimpleNamespace(fork_rng=lambda **_: fork_rng(devices=[])),
            cuda=SimpleNamespace(device=lambda _: nullcontext(), manual_seed=lambda _: None),
        )
        loads = []

        class CpuVosr(VosrRuntime):
            def __init__(self, *, seed, tile_size, infer_steps, cfg_scale, align_method, **kwargs):
                loads.append(seed)
                self.torch, self.functional, self.transforms = cpu_torch, functional, transforms
                self.device = torch.device("cpu")
                self.align_method = align_method
                self.args = SimpleNamespace(seed=seed, tile_size=tile_size, tile_overlap=4, patch_size=2,
                                            infer_steps=infer_steps, weak_cond_strength_aelq_list=[0.2, 0.2])
                self.official = SimpleNamespace(tiled_latent_inference=namespace["tiled_latent_inference"])
                self.vae, self.venc = None, None
                self.model = lambda x, *args, **kwargs: torch.zeros_like(x[:, :3])
                self.vosr = SimpleNamespace(interpolate=lambda lq, *args: lq, interp_type="linear", cfg_scale=cfg_scale, t_start=0, t_end=1)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with TemporaryDirectory() as directory, ExitStack() as patches:
            directory = Path(directory)
            paths = {name: directory / f"{name}.png" for name in ("a", "b")}
            for name, color in (("a", "red"), ("b", "blue")):
                Image.new("RGB", (128, 128), color).save(paths[name])
            config = default_postprocess_config().model_copy(update={
                "long_side": 128, "tile_size": 64, "infer_steps": 1, "align_method": "nofix", "seed": 42,
            })

            def run(names, root):
                nodes, wires = [], []
                for name in names:
                    nodes.extend((ImageSourceNode(id=f"src-{name}", title=name, config=ImageSourceConfig(path=str(paths[name]))),
                                  ImagePostprocessNode(id=name, title=name, config=config)))
                    wires.append(WorkflowConnection(id=f"wire-{name}", source=NodePortRef(node_id=f"src-{name}", port="image"),
                                                    target=NodePortRef(node_id=name, port="image")))
                graph = WorkflowGraph(name="Randomness", nodes=nodes, connections=wires)
                result = execute_workflow(compile_workflow_run(graph), runs_root=root, progress=SILENT_STATUS)
                return {name: (sha256_file(Path(result.terminal_outputs[name]["image"].path)),
                               json.loads(result.node_manifests[name].read_text())["signature"]) for name in names}

            patches.enter_context(patch.object(vosr_backend, "VosrRuntime", CpuVosr))
            patches.enter_context(patch.object(vosr_backend, "_require_cuda", return_value="CPU neural doubles"))
            patches.enter_context(patch.object(vosr_backend, "_require_vosr_installation"))
            rng_before = torch.get_rng_state().clone()
            together = run(("a", "b"), directory / "together")
            self.assertEqual(len(loads), 1)
            alone = run(("b",), directory / "alone")
            reverse = run(("b", "a"), directory / "reverse")
            run(("a",), directory / "partial")
            partial = run(("a", "b"), directory / "partial")
            self.assertEqual(len(loads), 5)
            self.assertEqual(together["b"], alone["b"])
            self.assertEqual(together, reverse)
            self.assertEqual(together, partial)
            torch.testing.assert_close(rng_before, torch.get_rng_state(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
