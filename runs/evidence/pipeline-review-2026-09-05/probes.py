"""CPU review probes; no model weights, CUDA work, or generated character images.

Run from the checkout with .venv/bin/python runs/evidence/pipeline-review-2026-09-05/probes.py.
The upscaler and VAE are doubles: these probes establish dataflow and dimensions,
not the visual quality of a neural upscaler or a diffusion model.
"""

from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from PIL import Image
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers import Flux2KleinPipeline
import torch

from aigen.generation import flux2_klein as klein
from aigen.generation.image_generation_requests import (
    ImageGenerationCaseRequest,
    ImageGenerationOutputRequest,
)
from aigen.image_edit_defaults import FLUX2_KLEIN_SCHEDULER
from aigen.workflow_cache import NodeInputIdentity, build_node_signature
from aigen.workflow_compilation import _compile_node_config, execution_config_payload
from aigen.workflow_graph import ArtifactType, ImageEditConfig, ImageEditNode
from aigen.workflow_provenance import workflow_node_provenance
from aigen.workflow_execution import _image_edit_batch_key


class TraceTensor:
    def __init__(self, shape, evidence=None):
        self.shape = tuple(shape)
        self.evidence = evidence

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self


class TraceProcessor:
    def __init__(self):
        self.real = Flux2ImageProcessor(vae_scale_factor=16)

    def _resize_to_target_area(self, image, area):
        return self.real._resize_to_target_area(image, area)

    def preprocess(self, image, **kwargs):
        tensor = self.real.preprocess(image, **kwargs)
        return TraceTensor(tensor.shape)


class TracePipeline:
    def __init__(self):
        self.image_processor = TraceProcessor()
        self.vae_scale_factor = 8
        self.default_sample_size = 128
        self.vae = SimpleNamespace(
            dtype="bf16", requires_grad_=lambda _: None, to=lambda _: None
        )
        self.encoded = []

    def encode_prompt(self, **kwargs):
        return None, None

    def prepare_image_latents(self, *, images, **kwargs):
        sizes = [list(image.shape[-2:]) for image in images]
        self.encoded.append({"mode": "reference", "sizes_hw": sizes})
        return TraceTensor((1,), sizes), TraceTensor((1,))

    def _encode_vae_image(self, image, generator):
        self.encoded.append({"mode": "init", "sizes_hw": [list(image.shape[-2:])]})
        return TraceTensor((1, 128, image.shape[-2] // 16, image.shape[-1] // 16))


class TraceUpscaler:
    calls = []

    def __init__(self, **kwargs):
        pass

    def upscale(self, image, *, target_size, **kwargs):
        self.calls.append({"source_wh": list(image.size), "target_wh": list(target_size)})
        return SimpleNamespace(image=image.resize(target_size))


class CpuControl:
    device = staticmethod(lambda _: "cpu")
    no_grad = staticmethod(nullcontext)
    Generator = staticmethod(lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: None))


progress = SimpleNamespace(phase=lambda _: None)
out = Path(__file__).parent
source = out / "synthetic-portrait.png"
second = out / "synthetic-landscape.png"
Image.new("RGB", (128, 256), "gray").save(source)
Image.new("RGB", (256, 128), "white").save(second)


def case(name, size, paths=(source,)):
    return ImageGenerationCaseRequest(
        name=name, prompt="CPU probe; never sent to a model", image_paths=paths,
        width=size[0], height=size[1],
        outputs=(ImageGenerationOutputRequest(name=name, seed=42, path=out / (name + ".png")),),
    )


def prepare(cases, strength=None):
    pipeline = TracePipeline()
    TraceUpscaler.calls = []
    with patch("aigen.generation.image_upscale.IllustrationUpscaler", TraceUpscaler), patch.object(klein, "_release_cuda"):
        prepared = klein._prepare_flux2_klein_cases(
            pipeline, cases=cases,
            prompt_embeddings={item.prompt: SimpleNamespace(prompt_embeds=None) for item in cases},
            torch=CpuControl, strength=strength, progress=progress,
        )
    return pipeline, prepared, list(TraceUpscaler.calls)


small = case("small", (512, 512))
large = case("large", (1024, 1024))
ab_pipeline, ab, calls = prepare((small, large))
ba_pipeline, ba, _ = prepare((large, small))
single_pipeline, single, _ = prepare((large,))
assert ab[1].image_latents.evidence == [[512, 512]]
assert single[0].image_latents.evidence == [[1024, 1024]]
assert ba[0].image_latents.evidence == [[1024, 1024]]
batch_options = dict(
    backend="flux2-klein", loras=(), steps=4, guidance=None,
    strength=None, sampler=klein.FLUX2_KLEIN_DEFAULT_SAMPLER, scheduler=FLUX2_KLEIN_SCHEDULER,
)
small_batch_key = _image_edit_batch_key(SimpleNamespace(**batch_options, width=512, height=512))
large_batch_key = _image_edit_batch_key(SimpleNamespace(**batch_options, width=1024, height=1024))
assert small_batch_key == large_batch_key

init_pipeline, init, init_calls = prepare((case("init", (512, 512), (source, second)),), strength=0.5)
assert len(init_pipeline.encoded) == 1
assert init_pipeline.encoded[0]["mode"] == "init"
assert init[0].image_latents is None

big_pipeline, big, big_calls = prepare((case("big_init", (1536, 1536)),), strength=0.5)
source_shape = big[0].init_source_latent.shape
target_shape = (1, 128, 1536 // 16, 1536 // 16)
packed = Flux2KleinPipeline._pack_latents(torch.zeros(source_shape))
ids = Flux2KleinPipeline._prepare_latent_ids(torch.zeros(target_shape))
assert packed.shape[1] != ids.shape[1]
try:
    Flux2KleinPipeline._unpack_latents_with_ids(packed, ids, 96, 96)
except RuntimeError as error:
    unpack_error = str(error)
else:
    raise AssertionError("expected mismatched packed latent/position shapes to fail")

node = ImageEditNode(id="edit", title="CPU provenance probe", config=ImageEditConfig(
    backend="qwen-image-edit-2511-base", prompt="CPU probe; never sent to a model",
    seed=42, width=1104, height=1472,
))
head_module = ModuleType("head_workflow_provenance")
head_source = subprocess.check_output(["git", "show", "HEAD:aigen/workflow_provenance.py"], cwd=ROOT, text=True)
exec(compile(head_source, "HEAD:aigen/workflow_provenance.py", "exec"), head_module.__dict__)
head_provenance = head_module.workflow_node_provenance(node)
current_provenance = workflow_node_provenance(node)
payload = execution_config_payload(_compile_node_config(node))
signature_args = dict(
    node_kind=node.kind, execution_config=payload,
    inputs={"references": (NodeInputIdentity(artifact_type=ArtifactType.IMAGE, identity=sha256(source.read_bytes()).hexdigest()),)},
    source_outputs=None,
)
head_key = build_node_signature(**signature_args, provenance=head_provenance)
current_key = build_node_signature(**signature_args, provenance=current_provenance)
assert head_key == current_key
changed_owners = {}
for relative in (
    "aigen/generation/qwen_image_edit_conditioner.py",
    "aigen/generation/qwen_image_edit_identity.py",
    "aigen/generation/qwen_image_edit_lightx2v_worker.py",
):
    before = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT)
    after = (ROOT / relative).read_bytes()
    assert before != after
    changed_owners[relative] = {"head_sha256": sha256(before).hexdigest(), "working_sha256": sha256(after).hexdigest()}

postprocess_records = []
for run_name in (
    "sprite-front-qwen2511-base-40step-v1",
    "sprite-front-qwen2511-native-refs-40step-v2",
):
    payload = json.loads((ROOT / "runs" / run_name / "result.json").read_text())
    first = payload["outputs"][0]
    postprocess_records.append({
        "run": run_name,
        "canvas_postprocess": payload["generation"]["output_canvas"]["postprocess"],
        "environment_postprocess": payload["environment"]["postprocess"],
        "actual_output_postprocess": first["postprocess"],
        "raw_equals_final_sha256": first["raw_image"]["sha256"] == first["image"]["sha256"],
    })
    assert first["postprocess"] == {"mode": "none"}
    assert first["raw_image"]["sha256"] == first["image"]["sha256"]

report = {
    "kind": "pipeline-review-cpu-probes", "gpu_models_run": False,
    "limits": "Neural upscaler and VAE are doubles; actual owner control flow, Diffusers image processor, packing and cache signature code execute.",
    "reference_upscale_calls": calls,
    "batch_order": {
        "workflow_batch_keys_equal": small_batch_key == large_batch_key,
        "small_then_large_encodings": ab_pipeline.encoded,
        "large_then_small_encodings": ba_pipeline.encoded,
        "large_alone_encodings": single_pipeline.encoded,
        "large_after_small_reference_sizes_hw": ab[1].image_latents.evidence,
        "large_alone_reference_sizes_hw": single[0].image_latents.evidence,
    },
    "strength_multiple_references": {
        "supplied": 2, "upscaler_calls": init_calls,
        "encoded": init_pipeline.encoded, "reference_context_present": init[0].image_latents is not None,
    },
    "strength_large_canvas": {
        "requested_wh": [1536, 1536], "source_latent_shape": list(source_shape),
        "packed_tokens": packed.shape[1], "position_tokens": ids.shape[1], "unpack_error": unpack_error,
    },
    "workflow_provenance": {
        "changed_owners": changed_owners,
        "head_key": head_key, "working_key": current_key,
        "provenance": current_provenance.model_dump(mode="json"),
        "limit": "Compares HEAD provenance code with working-tree code; imported profile/schema constants are unchanged. This proves key reuse, not that a particular production cache entry exists.",
    },
    "existing_run_postprocess_metadata": postprocess_records,
    "upscaler_accumulator_calculation": {
        "source_wh": [3840, 3890], "scale": 4, "dtype_bytes": 4, "channels": 3,
        "natural_pixels": 3840 * 3890 * 16,
        "two_accumulators_GiB": 3840 * 3890 * 16 * 3 * 4 * 2 / 1024**3,
        "limit": "Calculated tensor sizes in the live upscaler implementation; no GPU allocation or model execution performed.",
    },
}
(out / "probe-results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
