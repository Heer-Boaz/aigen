"""Run the actual USO worker with per-process CUDA and denoiser-step telemetry."""
import json
import os
from pathlib import Path
import sys
from threading import Event, Thread
import time

root = Path(__file__).parent.resolve()
config = json.loads((root / "manifest.json").read_text())
job = config["jobs"]["uso"]
directory = root / sys.argv[1]
directory.mkdir()
models = Path("/home/boaz/aigen/aigen/models/uso")
os.environ.update(
    FLUX_DEV_FP8=str(models / "black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"),
    AE=str(models / "black-forest-labs/FLUX.1-dev/ae.safetensors"),
    T5=str(models / "xlabs-ai/xflux_text_encoders"),
    CLIP=str(models / "openai/clip-vit-large-patch14"),
    LORA=str(models / "bytedance-research/USO/uso_flux_v1.0/dit_lora.safetensors"),
    PROJECTION_MODEL=str(models / "bytedance-research/USO/uso_flux_v1.0/projector.safetensors"),
    SIGLIP_PATH=str(models / "google/siglip-so400m-patch14-384"),
    TOKENIZERS_PARALLELISM="false",
)
sys.path.insert(0, "/home/boaz/.local/share/aigen/runtimes/uso/USO")
import torch
from aigen.gpu_status import nvidia_smi_memory_snapshot
from aigen.generation.image_edit import ImageEditRequest, resolve_image_edit_request
from aigen.generation.uso_flux1_worker import _run_requests
from uso.flux.pipeline import USOPipeline

preflight = nvidia_smi_memory_snapshot()
assert preflight["nvidia_smi_device_total_mb"] - preflight["nvidia_smi_used_mb"] >= 12000
resolved = resolve_image_edit_request(ImageEditRequest(
    backend=job["model"], prompt=job["prompt"], images=tuple(Path(p) for p in config["klein"]["references"]),
    output_dir=directory / "output", width=job["width"], height=job["height"], seeds=(job["seed"],),
))
request = dict(prompt=resolved.prompt, references=[str(p) for p in resolved.images],
    output=str(directory / "image.png"), width=resolved.width, height=resolved.height,
    seed=resolved.seeds[0], steps=resolved.steps, guidance=resolved.guidance)
(directory / "request.json").write_text(json.dumps(request, indent=2) + "\n")

started = time.monotonic()
def snapshot(phase):
    return dict(phase=phase, elapsed_seconds=time.monotonic() - started,
        torch_allocated_mib=torch.cuda.memory_allocated() / 1024**2,
        torch_reserved_mib=torch.cuda.memory_reserved() / 1024**2,
        torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2)

step_records = []
original_init = USOPipeline.__init__
def measured_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.model.register_forward_pre_hook(lambda module, args: step_records.append(snapshot("step_begin")))
    self.model.register_forward_hook(lambda module, args, result: step_records.append(snapshot("step_end")), always_call=True)
USOPipeline.__init__ = measured_init

stop = Event()
def monitor():
    with (directory / "telemetry.jsonl").open("w", buffering=1) as log:
        while not stop.is_set():
            log.write(json.dumps(snapshot("sample") | nvidia_smi_memory_snapshot()) + "\n")
            stop.wait(1)
thread = Thread(target=monitor, daemon=True)
thread.start()
try:
    code = _run_requests([request], sys.stdout)
finally:
    stop.set()
    thread.join()
    result = snapshot("finished") | dict(allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), preflight=preflight, steps=step_records)
    (directory / "memory.json").write_text(json.dumps(result, indent=2) + "\n")
raise SystemExit(code)
