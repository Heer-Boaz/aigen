"""CPU dataflow proofs for VOSR batch randomness and FLUX dev duplicate seeds.

VOSR uses its actual pinned tiled inference function and aigen's real upscale,
batch, scheduler and cache code. Neural encode/decode/DiT are CPU doubles.
"""
import ast
from contextlib import ExitStack
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from PIL import Image
import torch
import torch.nn.functional as functional
from torchvision import transforms
from aigen.generation import flux2_dev_wangp as fluxdev
from aigen.generation import vosr_backend
from aigen.generation.vosr_runtime import VosrRuntime
from aigen.generation.image_edit import ImageEditRequest, run_image_edit
from aigen.manifest_io import sha256_file
from aigen.progress import SILENT_STATUS
from aigen.workflow_compilation import compile_workflow_run
from aigen.workflow_execution import execute_workflow
from aigen.workflow_graph import (
    ImageSourceNode, ImageSourceConfig, ImagePostprocessNode, NodePortRef,
    WorkflowConnection, WorkflowGraph,
)
from aigen.workflow_templates import default_postprocess_config

EVIDENCE = Path(__file__).resolve().parent
UPSTREAM_VOSR = Path.home() / ".cache/aigen-vosr/VOSR/inference_vosr.py"


def cpu_vosr_type():
    source = UPSTREAM_VOSR.read_text()
    tree = ast.parse(source)
    names = {"_make_tile_grid", "_gaussian_weights", "tiled_latent_inference"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    namespace = {"torch": torch, "math": math}
    namespace["_encode_latent"] = lambda vae, tensor, args, device: (torch.zeros(1, 3, tensor.shape[-2] // 8, tensor.shape[-1] // 8), None, None)
    namespace["_decode_latent"] = lambda vae, tensor, *args: functional.interpolate(tensor, scale_factor=8).clamp(-1, 1)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(UPSTREAM_VOSR), "exec"), namespace)

    class CPUVosr(VosrRuntime):
        def __init__(self, *, seed, tile_size, infer_steps, cfg_scale, align_method, **kwargs):
            # Same once-per-runtime RNG policy as the actual constructor, no CUDA.
            torch.manual_seed(seed)
            self.torch, self.functional, self.transforms = torch, functional, transforms
            self.device = torch.device("cpu")
            self.align_method = align_method
            self.args = SimpleNamespace(tile_size=tile_size, tile_overlap=4, patch_size=2, infer_steps=infer_steps, weak_cond_strength_aelq_list=[0.2, 0.2])
            self.official = SimpleNamespace(tiled_latent_inference=namespace["tiled_latent_inference"])
            self.vae, self.venc = None, None
            self.model = lambda inp, *args, **kwargs: torch.zeros_like(inp[:, :3])
            self.vosr = SimpleNamespace(interpolate=lambda lq, *args: lq, interp_type="linear", cfg_scale=cfg_scale, t_start=0, t_end=1)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    return CPUVosr


def main():
    workspace = EVIDENCE / f"backend-data-{uuid4().hex[:8]}"
    workspace.mkdir()
    paths = {}
    for name, color in (("a", "red"), ("b", "blue")):
        paths[name] = workspace / f"{name}.png"
        Image.new("RGB", (128, 128), color).save(paths[name])
    config = default_postprocess_config().model_copy(update={"long_side": 128, "tile_size": 64, "infer_steps": 1, "align_method": "nofix", "seed": 42})

    def graph(names):
        nodes, wires = [], []
        for name in names:
            nodes.extend((ImageSourceNode(id=f"src-{name}", title=name, config=ImageSourceConfig(path=str(paths[name]))), ImagePostprocessNode(id=name, title=name, config=config)))
            wires.append(WorkflowConnection(id=f"wire-{name}", source=NodePortRef(node_id=f"src-{name}", port="image"), target=NodePortRef(node_id=name, port="image")))
        return WorkflowGraph(name="VOSR CPU batch audit", nodes=nodes, connections=wires)

    def run(names, root):
        result = execute_workflow(compile_workflow_run(graph(names)), runs_root=root, progress=SILENT_STATUS)
        return {name: {"sha256": sha256_file(Path(result.terminal_outputs[name]["image"].path)), "signature": json.loads(result.node_manifests[name].read_text())["signature"], "path": result.terminal_outputs[name]["image"].path} for name in names}

    with ExitStack() as patches:
        patches.enter_context(patch.object(vosr_backend, "VosrRuntime", cpu_vosr_type()))
        patches.enter_context(patch.object(vosr_backend, "_require_cuda", return_value="CPU neural doubles"))
        patches.enter_context(patch.object(vosr_backend, "_require_vosr_installation", return_value=None))
        together = run(("a", "b"), workspace / "together")
        alone = run(("b",), workspace / "alone")
        run(("a",), workspace / "partial-cache")
        partial = run(("a", "b"), workspace / "partial-cache")
    assert together["b"]["signature"] == alone["b"]["signature"] == partial["b"]["signature"]
    assert together["b"]["sha256"] != alone["b"]["sha256"] == partial["b"]["sha256"]

    requests_seen = []
    def fake_worker(**kwargs):
        responses = []
        for index, request in enumerate(kwargs["requests"]):
            requests_seen.append(request)
            Image.new("RGB", (512, 512), (index * 100, 0, 0)).save(request["output"])
            responses.append({"elapsed_seconds": 0.1, "environment": {"audit": "CPU double"}})
        return responses
    with patch.object(fluxdev, "_validate_runtime", return_value=None), patch.object(fluxdev, "_run_worker", fake_worker):
        flux = run_image_edit(ImageEditRequest(backend="flux2-dev-nvfp4", prompt="Keep the image unchanged.", output_dir=workspace / "flux-duplicate", images=(paths["a"],), width=512, height=512, seeds=(7, 7)), progress=SILENT_STATUS)
    assert len(requests_seen) == len(flux.outputs) == 2
    assert requests_seen[0]["output"] == requests_seen[1]["output"]
    assert len({x.path for x in flux.outputs}) == 1

    palette = Image.new("P", (16, 16), 0)
    palette.putpalette([255, 0, 0, 0, 0, 255] + [0] * (768 - 6))
    palette.info["transparency"] = 0
    palette_path = workspace / "palette-alpha.png"
    palette.save(palette_path)
    prepared = vosr_backend._prepare_vosr_file(palette_path, workspace / "unused.png", 2, None)
    assert prepared.alpha is None
    prepared.image.close()

    results = {
        "scope": "CPU dataflow only. Real pinned VOSR noise generation with neural doubles; real workflow/cache/batching; FLUX dev worker doubled. No neural quality or VRAM claim.",
        "vosr": {"together": together, "alone": alone, "one_cached": partial, "upstream_sha256": sha256(UPSTREAM_VOSR.read_bytes()).hexdigest()},
        "flux_duplicate_seeds": {"request_count": len(requests_seen), "returned_count": len(flux.outputs), "unique_output_count": len({x.path for x in flux.outputs}), "requested_paths": [r["output"] for r in requests_seen]},
        "palette_alpha": {"input_mode": "P", "input_transparency": 0, "prepared_alpha": None},
    }
    (EVIDENCE / "backend-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"status": "passed", "workspace": str(workspace), "vosr_same_signature_different_output": True, "flux_two_results_one_file": True}))


if __name__ == "__main__":
    main()
