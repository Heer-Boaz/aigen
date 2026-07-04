from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from aigen.generation.qwen_image_edit_identity import (
    NUNCHAKU_QWEN_EDIT_2509_DIR,
    QWEN_EDIT_2509_LOCAL_MODEL,
    QWEN_IMAGE_EDIT_2509_LIGHTNING_SCHEDULER_CONFIG,
)
from aigen.gpu_status import GpuStatusError, nvidia_smi_memory_snapshot
from aigen.image_assets import image_asset_json
from aigen.keyframe_memory import NvidiaSmiMemorySampler
from aigen.manifest_io import write_json


DEFAULT_CHECKPOINT = (
    NUNCHAKU_QWEN_EDIT_2509_DIR
    / "lightning-251115/svdq-fp4_r32-qwen-image-edit-2509-lightning-4steps-251115.safetensors"
)
DEFAULT_PROMPT = (
    "Use the input image as the reference for the same character. Generate a clean character reference image, "
    "neutral standing pose, plain light studio background. Preserve the face, hair, outfit, body proportions, "
    "colors, and art style. Do not add props, text, extra characters, or alternate outfits."
)
LOW_VRAM_THRESHOLD_MB = 18 * 1024


class Qwen2509SmokeError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-image Nunchaku Qwen-Image-Edit-2509 fit smoke")
    parser.add_argument("--input-image", type=Path, required=True, help="One local reference image")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output.png and smoke_result.json")
    parser.add_argument("--base-model", default=QWEN_EDIT_2509_LOCAL_MODEL, help="Local Qwen/Qwen-Image-Edit-2509 dir")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="One local Nunchaku Qwen-Image-Edit-2509 safetensors checkpoint",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Edit prompt")
    parser.add_argument("--max-side", type=int, default=512, help="Longest input/output side")
    parser.add_argument("--steps", type=int, help="Inference steps; inferred from checkpoint name when omitted")
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        help="True CFG scale; defaults to 1.0 for lightning and 4.0 for full checkpoints",
    )
    parser.add_argument("--seed", type=int, default=0, help="Generation seed")
    parser.add_argument(
        "--offload-mode",
        choices=("auto", "always", "never"),
        default="auto",
        help="Low-VRAM offload policy; auto enables it on GPUs with 18GB VRAM or less",
    )
    parser.add_argument(
        "--blocks-on-gpu",
        type=int,
        default=1,
        help="Nunchaku transformer blocks retained on GPU when offload is enabled",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    parser.add_argument("--compact", action="store_true", help="Write compact JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_smoke(args)
    except (Qwen2509SmokeError, GpuStatusError) as error:
        payload = {"status": "error", "error": error.__class__.__name__, "message": str(error)}
        json.dump(payload, sys.stderr, ensure_ascii=False, indent=None if args.compact else 2, sort_keys=True)
        sys.stderr.write("\n")
        return 1
    json.dump(result, sys.stdout, ensure_ascii=False, indent=None if args.compact else 2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise Qwen2509SmokeError(f"Output exists and overwrite=false: {output_dir.as_posix()}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_image_path = args.input_image.resolve()
    checkpoint = args.checkpoint.resolve()
    base_model = Path(args.base_model).resolve()
    _require_file(input_image_path, "input image")
    _require_file(checkpoint, "Nunchaku checkpoint")
    _require_dir(base_model, "Qwen-Image-Edit-2509 base model")
    _validate_args(args)

    preflight_snapshot = nvidia_smi_memory_snapshot()
    preflight = {
        "nvidia_smi_preflight_used_mb": preflight_snapshot["nvidia_smi_used_mb"],
        "nvidia_smi_device_total_mb": preflight_snapshot["nvidia_smi_device_total_mb"],
        "nvidia_smi_preflight_utilization_gpu": preflight_snapshot["nvidia_smi_utilization_gpu"],
    }
    offload_enabled = _offload_enabled(args.offload_mode, preflight_snapshot)
    image = _load_image(input_image_path, max_side=args.max_side)
    width, height = _aligned_size(image.size)
    steps = args.steps if args.steps is not None else _default_steps_for_checkpoint(checkpoint)
    true_cfg_scale = (
        args.true_cfg_scale if args.true_cfg_scale is not None else _default_true_cfg_scale_for_checkpoint(checkpoint)
    )

    memory_sampler = NvidiaSmiMemorySampler(preflight)
    memory_sampler.start()
    memory: dict[str, Any] | None = None
    try:
        result = _run_pipeline(
            base_model=base_model,
            checkpoint=checkpoint,
            image=image,
            prompt=args.prompt,
            output_dir=output_dir,
            width=width,
            height=height,
            steps=steps,
            true_cfg_scale=true_cfg_scale,
            seed=args.seed,
            offload_enabled=offload_enabled,
            blocks_on_gpu=args.blocks_on_gpu,
        )
        memory = memory_sampler.stop()
    finally:
        if memory is None:
            memory_sampler.stop()

    payload = {
        "status": "completed",
        "kind": "qwen-image-edit-2509-nunchaku-fit-smoke",
        "base_model": base_model.as_posix(),
        "checkpoint": checkpoint.as_posix(),
        "input_image": image_asset_json(input_image_path),
        "output_image": image_asset_json(Path(result["output_image"])),
        "prompt": args.prompt,
        "generation": {
            "width": width,
            "height": height,
            "max_side": args.max_side,
            "steps": steps,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": " " if true_cfg_scale > 1.0 else None,
            "seed": args.seed,
        },
        "offload": {
            "mode": args.offload_mode,
            "enabled": offload_enabled,
            "blocks_on_gpu": args.blocks_on_gpu if offload_enabled else None,
        },
        "memory": result["cuda_memory"] | memory,
        "timings_ms": result["timings_ms"],
        "output": {
            "directory": output_dir.as_posix(),
            "image": result["output_image"],
            "result": (output_dir / "smoke_result.json").as_posix(),
        },
    }
    write_json(output_dir / "smoke_result.json", payload)
    return payload


def _run_pipeline(
    *,
    base_model: Path,
    checkpoint: Path,
    image: Image.Image,
    prompt: str,
    output_dir: Path,
    width: int,
    height: int,
    steps: int,
    true_cfg_scale: float,
    seed: int,
    offload_enabled: bool,
    blocks_on_gpu: int,
) -> dict[str, Any]:
    try:
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
        from nunchaku import NunchakuQwenImageTransformer2DModel
    except ImportError as error:
        raise Qwen2509SmokeError("Qwen 2509 smoke requires `pip install -e .[generation]` and nunchaku") from error

    if not torch.cuda.is_available():
        raise Qwen2509SmokeError("Qwen 2509 smoke requires CUDA")

    torch.cuda.reset_peak_memory_stats("cuda")
    load_start = _sync_time(torch)
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
        checkpoint.as_posix(),
        torch_dtype=torch.bfloat16,
        offload=False,
    )
    pipeline_kwargs: dict[str, Any] = {
        "transformer": transformer,
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if _is_lightning_checkpoint(checkpoint):
        pipeline_kwargs["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(
            QWEN_IMAGE_EDIT_2509_LIGHTNING_SCHEDULER_CONFIG
        )
    pipeline = QwenImageEditPlusPipeline.from_pretrained(base_model.as_posix(), **pipeline_kwargs)
    pipeline.set_progress_bar_config(disable=True)
    if offload_enabled:
        transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=blocks_on_gpu)
        pipeline._exclude_from_cpu_offload.append("transformer")
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.to("cuda")
    model_load_ms = _elapsed_ms(load_start, _sync_time(torch))

    generation_start = _sync_time(torch)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    negative_prompt = " " if true_cfg_scale > 1.0 else None
    with torch.inference_mode():
        output = pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )
    pipeline_ms = _elapsed_ms(generation_start, _sync_time(torch))
    output_image = output_dir / "output.png"
    output.images[0].save(output_image)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    cuda_memory = {
        "max_allocated_mb": round(torch.cuda.max_memory_allocated("cuda") / 2**20),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved("cuda") / 2**20),
        "free_after_run_mb": round(free_bytes / 2**20),
        "device_total_mb": round(total_bytes / 2**20),
    }
    del pipeline
    torch.cuda.empty_cache()
    return {
        "output_image": output_image.as_posix(),
        "cuda_memory": cuda_memory,
        "timings_ms": {
            "model_load_ms": model_load_ms,
            "pipeline_ms": pipeline_ms,
        },
    }


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise Qwen2509SmokeError(f"Missing {label}: {path.as_posix()}")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise Qwen2509SmokeError(f"Missing {label}: {path.as_posix()}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_side < 16:
        raise Qwen2509SmokeError("max_side must be at least 16")
    if args.steps is not None and args.steps < 1:
        raise Qwen2509SmokeError("steps must be at least 1")
    if args.true_cfg_scale is not None and args.true_cfg_scale <= 0:
        raise Qwen2509SmokeError("true_cfg_scale must be positive")
    if args.blocks_on_gpu < 1:
        raise Qwen2509SmokeError("blocks_on_gpu must be at least 1")


def _offload_enabled(offload_mode: str, snapshot: dict[str, int]) -> bool:
    if offload_mode == "always":
        return True
    if offload_mode == "never":
        return False
    return snapshot["nvidia_smi_device_total_mb"] <= LOW_VRAM_THRESHOLD_MB


def _load_image(path: Path, *, max_side: int) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    longest = max(rgb.size)
    if longest <= max_side:
        return rgb
    scale = max_side / longest
    width = max(16, _align_down(round(rgb.width * scale), 16))
    height = max(16, _align_down(round(rgb.height * scale), 16))
    return rgb.resize((width, height), Image.Resampling.LANCZOS)


def _aligned_size(size: tuple[int, int]) -> tuple[int, int]:
    return _align_down(size[0], 16), _align_down(size[1], 16)


def _align_down(value: int, multiple: int) -> int:
    return max(multiple, value // multiple * multiple)


def _default_steps_for_checkpoint(checkpoint: Path) -> int:
    name = checkpoint.name
    if "4steps" in name:
        return 4
    if "8steps" in name:
        return 8
    return 40


def _default_true_cfg_scale_for_checkpoint(checkpoint: Path) -> float:
    return 1.0 if _is_lightning_checkpoint(checkpoint) else 4.0


def _is_lightning_checkpoint(checkpoint: Path) -> bool:
    return "lightning" in checkpoint.name


def _sync_time(torch_module: Any) -> float:
    torch_module.cuda.synchronize()
    return perf_counter()


def _elapsed_ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 3)


if __name__ == "__main__":
    raise SystemExit(main())
