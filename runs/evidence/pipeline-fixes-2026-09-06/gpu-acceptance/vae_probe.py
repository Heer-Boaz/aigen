"""Isolate native Qwen VAE memory using the actual installed Diffusers encoder."""
import argparse
import json
from pathlib import Path
import time

import torch
from diffusers import AutoencoderKLQwenImage
from diffusers.image_processor import VaeImageProcessor

from aigen.generation.qwen_image_edit_lightx2v import QWEN_IMAGE_EDIT_2511_LOCAL_MODEL
from aigen.gpu_status import nvidia_smi_memory_snapshot
from aigen.image_io import open_image

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("reference", type=int)
parser.add_argument("--tile", type=int)
args = parser.parse_args()
config = json.loads(args.manifest.read_text())
snapshot = nvidia_smi_memory_snapshot()
print(json.dumps({"gpu_preflight": snapshot}), flush=True)
assert snapshot["nvidia_smi_device_total_mb"] - snapshot["nvidia_smi_used_mb"] >= 12000
source = Path(config["klein"]["references"][args.reference])
vae = AutoencoderKLQwenImage.from_pretrained(
    Path(QWEN_IMAGE_EDIT_2511_LOCAL_MODEL) / "vae", torch_dtype=torch.bfloat16, local_files_only=True,
).to("cuda")
if args.tile:
    vae.enable_tiling(tile_sample_min_height=args.tile, tile_sample_min_width=args.tile,
                      tile_sample_stride_height=args.tile * 3 // 4, tile_sample_stride_width=args.tile * 3 // 4)
with open_image(source) as image:
    pixels = VaeImageProcessor(vae_scale_factor=16).preprocess(image).unsqueeze(2).to("cuda", torch.bfloat16)
torch.cuda.reset_peak_memory_stats()
print(json.dumps({"source": str(source), "input_shape": list(pixels.shape), "tile": args.tile}), flush=True)
start = time.perf_counter()
try:
    with torch.inference_mode():
        latents = vae.encode(pixels).latent_dist.mode()
        torch.cuda.synchronize()
    print(json.dumps({"status": "completed", "latent_shape": list(latents.shape), "finite": bool(latents.isfinite().all())}), flush=True)
finally:
    print(json.dumps({"elapsed_seconds": time.perf_counter() - start,
                      "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2}), flush=True)
