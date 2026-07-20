from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO


os.environ["TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"] = "1"
os.environ["device"] = "cuda:0"


def main() -> int:
    requests = [json.loads(line) for line in sys.stdin if line.strip()]
    if len(requests) != 1:
        raise SystemExit("boogu_image_edit_worker requires exactly one JSON request on stdin")
    with _open_progress_stream() as progress_stream:
        try:
            return _run_request(requests[0], progress_stream)
        except Exception:
            traceback.print_exc()
            return 1


def _run_request(request: dict[str, Any], progress_stream: TextIO) -> int:
    _send(progress_stream, "phase", text="loading official Boogu-Image Edit Turbo FP8")

    import torch
    from PIL import Image
    from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
    from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline

    model_root = Path(os.environ["AIGEN_BOOGU_MODEL_ROOT"]).resolve()
    with Image.open(request["reference"]) as source:
        reference = source.convert("RGB")
    input_images = [[reference]]
    session_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    transformer = BooguImageTransformer2DModel.from_pretrained(
        model_root / "transformer",
        torch_dtype=torch.bfloat16,
        use_safetensors=False,
        local_files_only=True,
    )
    pipeline = BooguImageTurboPipeline.from_pretrained(
        model_root,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        transformer=transformer,
        local_files_only=True,
    )
    pipeline.enable_model_cpu_offload_flag = True
    pipeline.enable_model_cpu_offload(device="cuda:0")
    load_seconds = time.monotonic() - session_started

    _send(progress_stream, "phase", text="encoding shared Boogu instruction and reference")
    encode_started = time.monotonic()
    encoded = pipeline.encode_instruction(
        instruction=[request["prompt"]],
        do_classifier_free_guidance=False,
        input_images=input_images,
        max_vlm_input_pil_pixels=147456,
        max_vlm_input_pil_side_length=768,
        num_images_per_instruction=1,
        device=torch.device("cuda:0"),
        max_sequence_length=1024,
        truncate_instruction_sequence=False,
        use_rewrite_text_instruction=False,
        system_prompt_follows_task_type=True,
        task_type="ti2i",
    )
    instruction_embeds, instruction_attention_mask = encoded[:2]
    encode_seconds = time.monotonic() - encode_started

    output_timings = []
    for generation in request["outputs"]:
        seed = int(generation["seed"])
        _send(progress_stream, "phase", text=f"Boogu-Image seed {seed}")
        _send(
            progress_stream,
            "begin",
            total=request["steps"],
            text=f"denoising Boogu-Image seed {seed}",
        )
        seed_started = time.monotonic()

        def report_step(index: int, total: int) -> None:
            _send(
                progress_stream,
                "step",
                text=f"Boogu-Image seed {seed} step {index + 1}/{total}",
            )

        result = pipeline(
            instruction=[request["prompt"]],
            instruction_embeds=instruction_embeds,
            instruction_attention_mask=instruction_attention_mask,
            input_images=input_images,
            input_image_paths=[[request["reference"]]],
            width=request["width"],
            height=request["height"],
            align_res=False,
            max_input_image_pixels=request["width"] * request["height"],
            max_input_image_side_length=max(request["width"], request["height"]) * 2,
            num_inference_steps=request["steps"],
            max_vlm_input_pil_pixels=147456,
            max_vlm_input_pil_side_length=768,
            max_sequence_length=1024,
            truncate_instruction_sequence=False,
            text_guidance_scale=1.0,
            image_guidance_scale=1.0,
            empty_instruction_guidance_scale=0.0,
            negative_instruction="",
            num_images_per_instruction=1,
            generator=torch.Generator(device="cuda:0").manual_seed(seed),
            output_type="pil",
            use_rewrite_text_instruction=False,
            system_prompt_follows_task_type=True,
            use_dmd_student_inference=True,
            dmd_conditioning_sigma=0.0,
            step_func=report_step,
            device="cuda:0",
        )
        image = result.images[0]
        image.save(generation["path"])
        output_timings.append(
            {"seed": seed, "elapsed_seconds": round(time.monotonic() - seed_started, 3)}
        )

    environment = {
        "engine": "official Boogu-Image Turbo FP8",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(
            str(value) for value in torch.cuda.get_device_capability(0)
        ),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        "load_seconds": round(load_seconds, 3),
        "encode_seconds": round(encode_seconds, 3),
        "elapsed_seconds": round(time.monotonic() - session_started, 3),
        "instruction_cache": "one multimodal encode shared by every seed",
        "offload": "model CPU offload",
    }
    _send(
        progress_stream,
        "result",
        response={
            "status": "completed",
            "outputs": output_timings,
            "environment": environment,
        },
    )
    return 0


def _open_progress_stream() -> TextIO:
    sys.stdout.flush()
    progress_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return os.fdopen(progress_fd, "w", encoding="utf-8", buffering=1)


def _send(stream: TextIO, kind: str, **payload: Any) -> None:
    stream.write(json.dumps({"kind": kind, **payload}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
