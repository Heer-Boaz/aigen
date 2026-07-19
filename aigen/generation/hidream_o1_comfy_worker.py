from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO


def main() -> int:
    requests = [json.loads(line) for line in sys.stdin if line.strip()]
    if len(requests) != 1:
        raise SystemExit("hidream_o1_comfy_worker requires exactly one JSON request on stdin")
    with _open_progress_stream() as progress_stream:
        try:
            return _run_request(requests[0], progress_stream)
        except Exception:
            traceback.print_exc()
            return 1


def _run_request(request: dict[str, Any], progress_stream: TextIO) -> int:
    models_root = Path(os.environ["AIGEN_COMFY_MODELS_ROOT"]).resolve()
    _send(progress_stream, "phase", text="loading ComfyUI and HiDream-O1 Full FP8")

    import cuda_malloc
    import numpy as np
    import torch
    import comfy.model_management as model_management
    import comfy.utils
    import folder_paths
    from comfy_extras.nodes_custom_sampler import BasicScheduler, KSamplerSelect, SamplerCustom
    from comfy_extras.nodes_hidream_o1 import (
        EmptyHiDreamO1LatentImage,
        HiDreamO1PatchSeamSmoothing,
        HiDreamO1ReferenceImages,
    )
    from comfy_extras.nodes_model_advanced import ModelNoiseScale
    from nodes import CheckpointLoaderSimple, CLIPTextEncode, VAEDecode
    from PIL import Image

    folder_paths.add_model_folder_path(
        "checkpoints",
        (models_root / "checkpoints").as_posix(),
        is_default=True,
    )

    session_started = time.monotonic()
    checkpoint_loader = CheckpointLoaderSimple()
    model, clip, vae = checkpoint_loader.load_checkpoint("hidream_o1_image_fp8_scaled.safetensors")
    model = ModelNoiseScale().patch(model, request["noise_scale"])[0]
    model = HiDreamO1PatchSeamSmoothing.execute(
        model=model,
        start_percent=0.8,
        end_percent=1.0,
        pattern="single_shift",
        passes="ramp_2_4",
        blend="median",
        strength=1.0,
    )[0]

    encoder = CLIPTextEncode()
    positive = encoder.encode(clip, request["prompt"])[0]
    negative = encoder.encode(clip, "")[0]
    reference_tensors = {
        f"image_{index}": _load_image_tensor(Path(path), Image, np, torch)
        for index, path in enumerate(request["references"], start=1)
    }
    positive, negative = HiDreamO1ReferenceImages.execute(
        positive=positive,
        negative=negative,
        images=reference_tensors,
    ).result
    latent = EmptyHiDreamO1LatentImage.execute(
        width=request["width"],
        height=request["height"],
        batch_size=1,
    )[0]
    sigmas = BasicScheduler.execute(
        model=model,
        scheduler=request["scheduler"],
        steps=request["steps"],
        denoise=1.0,
    )[0]
    sampler = KSamplerSelect.execute(request["sampler"])[0]
    diffusion_model = model.get_model_object("diffusion_model")
    setup_seconds = time.monotonic() - session_started

    denoise_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    _send(progress_stream, "phase", text=f"HiDream-O1 seed {request['seed']}")
    _install_progress_hook(comfy.utils, progress_stream, request["steps"], request["seed"])
    try:
        samples = SamplerCustom.execute(
            model=model,
            add_noise=True,
            noise_seed=request["seed"],
            cfg=request["guidance"],
            positive=positive,
            negative=negative,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent,
        )[0]
    finally:
        comfy.utils.set_progress_bar_global_hook(None)

    samples = samples.copy()
    samples["samples"] = samples["samples"].to("cpu")
    denoise_seconds = time.monotonic() - denoise_started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    diffusion_model.clear_kv_cache()

    model_management.unload_all_models()
    model_management.soft_empty_cache()

    decoder = VAEDecode()
    _send(progress_stream, "phase", text=f"decoding HiDream-O1 seed {request['seed']}")
    decode_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    image_tensor = decoder.decode(vae=vae, samples=samples)[0][0]
    decode_seconds = time.monotonic() - decode_started
    _save_image(image_tensor, Path(request["output"]), Image, np)
    environment = {
        "engine": "ComfyUI native HiDream-O1",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(str(value) for value in torch.cuda.get_device_capability(0)),
        "peak_allocated_mib": round(
            max(peak_allocated, torch.cuda.max_memory_allocated()) / 1024**2,
            1,
        ),
        "peak_reserved_mib": round(
            max(peak_reserved, torch.cuda.max_memory_reserved()) / 1024**2,
            1,
        ),
        "setup_seconds": round(setup_seconds, 3),
        "denoise_seconds": round(denoise_seconds, 3),
        "decode_seconds": round(decode_seconds, 3),
        "elapsed_seconds": round(denoise_seconds + decode_seconds, 3),
    }
    _send(
        progress_stream,
        "result",
        response={
            "status": "completed",
            "output": request["output"],
            "environment": environment,
        },
    )

    del model, clip, vae
    model_management.unload_all_models()
    model_management.soft_empty_cache()
    return 0


def _load_image_tensor(path: Path, Image: Any, np: Any, torch: Any) -> Any:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _save_image(tensor: Any, output: Path, Image: Any, np: Any) -> None:
    array = tensor.detach().cpu().clamp(0, 1).numpy()
    image = Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")
    image.save(output)


def _install_progress_hook(comfy_utils: Any, stream: TextIO, steps: int, seed: int) -> None:
    sent = 0
    _send(stream, "begin", total=steps, text=f"denoising HiDream-O1 seed {seed}")

    def hook(value: int, total: int, _preview: Any, node_id: Any = None) -> None:
        nonlocal sent
        target = min(int(value), steps)
        while sent < target:
            sent += 1
            _send(stream, "step", text=f"HiDream-O1 seed {seed} step {sent}/{steps}")

    comfy_utils.set_progress_bar_global_hook(hook)


def _open_progress_stream() -> TextIO:
    sys.stdout.flush()
    progress_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return os.fdopen(progress_fd, "w", encoding="utf-8", buffering=1)


def _send(stream: TextIO, kind: str, **payload: Any) -> None:
    stream.write(json.dumps({"kind": kind, **payload}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
