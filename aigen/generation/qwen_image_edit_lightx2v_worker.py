from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO


# cudaHostAlloc on this WSL2 host starts refusing past ~14 GiB even with 15 GiB of RAM
# still free -- a driver ceiling, not a shortage. Stay well under it: the CUDA context and
# the denoise activations still need host memory, and page-locked pages cannot be reclaimed.
PINNED_HOST_BUDGET_BYTES = 10 * 1024**3

os.environ.setdefault("PROFILING_DEBUG_LEVEL", "0")
os.environ.setdefault("DTYPE", "BF16")
os.environ.setdefault("SENSITIVE_LAYER_DTYPE", "None")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: qwen_image_edit_lightx2v_worker REQUEST RESPONSE")
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    with _open_progress_stream() as progress_stream:
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            response = _run(request, progress_stream)
        except Exception as error:
            traceback.print_exc()
            response = {
                "status": "error",
                "error": error.__class__.__name__,
                "message": str(error),
            }
            response_path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
            return 1
        response_path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
    return 0


def _run(request: dict[str, Any], progress_stream: TextIO) -> dict[str, Any]:
    _send_progress(progress_stream, "phase", text="loading Qwen runtime")
    from loguru import logger
    import torch
    from PIL import Image

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    from lightx2v import LightX2VPipeline
    from lightx2v.common.ops import utils as ops_utils
    from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
    from lightx2v.utils.utils import seed_all

    from aigen.generation.qwen_image_edit_conditioner import QwenImageEditFp8Conditioner

    profile = request["profile"]
    cases = request["cases"]
    total_start = time.perf_counter()
    host_before = _host_snapshot()
    phase_peaks: dict[str, float] = {}
    timings: dict[str, float] = {}

    _send_progress(progress_stream, "phase", text="loading Qwen pipeline")
    pipe = LightX2VPipeline(
        model_path=profile["base_model"],
        model_cls="qwen-image-edit-2511",
        task="i2i",
    )
    pipe.unload_modules = True
    pipe.enable_offload(
        cpu_offload=True,
        offload_granularity="block",
        text_encoder_offload=False,
        vae_offload=False,
    )
    pipe.enable_quantize(
        dit_quantized=True,
        dit_quantized_ckpt=profile["transformer_model"],
        quant_scheme="fp8-triton",
    )
    loras = profile.get("loras", [])
    if loras:
        pipe.enable_lora(loras, lora_dynamic_apply=True)
    pipe.tokenizer_max_length = profile["max_sequence_length"]
    pipe.create_generator(
        attn_mode="flash_attn2",
        infer_steps=profile["steps"],
        guidance_scale=profile["true_cfg_scale"],
        rope_type="torch",
        resize_mode="adaptive",
    )
    runner = pipe.runner
    with runner.config.temporarily_unlocked():
        runner.config["max_custom_size"] = max(
            max(case["width"], case["height"]) for case in cases
        )
        runner.config["min_custom_size"] = min(
            min(case["width"], case["height"]) for case in cases
        )

    _reset_cuda_peak(torch)
    conditioner_load_start = time.perf_counter()
    _send_progress(progress_stream, "phase", text="loading image conditioner")
    with runner.config.temporarily_unlocked():
        runner.config["cpu_offload"] = False
        runner.config["qwen25vl_cpu_offload"] = False
    runner.text_encoders = [
        QwenImageEditFp8Conditioner(
            runner.config,
            Path(profile["conditioner_model"]),
        )
    ]
    torch.cuda.synchronize()
    timings["conditioner_load_ms"] = _elapsed_ms(conditioner_load_start)
    encode_start = time.perf_counter()
    _send_progress(progress_stream, "phase", text="encoding reference images")
    conditioned: dict[str, dict[str, Any]] = {}
    vae_source_groups: dict[tuple[str, ...], dict[str, Any]] = {}
    for case in cases:
        input_info = _input_info(pipe, case, case["outputs"][0]["seed"], init_empty_input_info, update_input_info_from_dict)
        runner.input_info = input_info
        images = []
        try:
            for image_path in case["image_paths"]:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
            input_info.original_size = [image.size for image in images]
            text_output = runner.run_text_encoder(case["prompt"], images, neg_prompt="")
            image_info = text_output["image_info"]
            group_key = tuple(case["image_paths"])
            if group_key not in vae_source_groups:
                vae_source_groups[group_key] = {
                    "images": [tensor.detach().to("cpu") for tensor in image_info["vae_image_list"]],
                    "info": image_info["vae_image_info_list"],
                }
            condition = {
                "prompt_embeds": text_output["prompt_embeds"].detach().to("cpu"),
                "vae_group": group_key,
                "vae_image_info_list": image_info["vae_image_info_list"],
                "txt_seq_lens": list(input_info.txt_seq_lens),
            }
            if profile["true_cfg_scale"] != 1.0:
                condition["negative_prompt_embeds"] = text_output["negative_prompt_embeds"].detach().to("cpu")
            conditioned[case["name"]] = condition
            del text_output
        finally:
            for image in images:
                image.close()
    torch.cuda.synchronize()
    timings["condition_encode_ms"] = _elapsed_ms(encode_start)
    phase_peaks["conditioning"] = _cuda_peak_mib(torch)
    del runner.text_encoders
    runner.input_info = None
    _release_cuda(torch)
    with runner.config.temporarily_unlocked():
        runner.config["cpu_offload"] = True

    _reset_cuda_peak(torch)
    vae_encode_start = time.perf_counter()
    _send_progress(progress_stream, "phase", text="encoding reference latents")
    runner.vae = runner.load_vae()
    encoded_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for group_key, source_group in vae_source_groups.items():
        encoded_groups[group_key] = []
        for image in source_group["images"]:
            image_latents = runner.vae.encode_vae_image(image.to(device="cuda", dtype=torch.bfloat16))
            encoded_groups[group_key].append({"image_latents": image_latents.detach().to("cpu")})
            del image_latents
    torch.cuda.synchronize()
    timings["vae_encode_ms"] = _elapsed_ms(vae_encode_start)
    phase_peaks["vae_encode"] = _cuda_peak_mib(torch)
    del runner.vae, vae_source_groups
    _release_cuda(torch)

    ops_utils.create_pin_tensor = _file_backed_cpu_tensor_factory()
    _reset_cuda_peak(torch)
    transformer_load_start = time.perf_counter()
    _send_progress(progress_stream, "phase", text="loading denoiser")
    runner.model = runner.load_transformer()
    runner.model.set_scheduler(runner.scheduler)
    torch.cuda.synchronize()
    timings["transformer_load_ms"] = _elapsed_ms(transformer_load_start)
    resident_upload_start = time.perf_counter()
    _send_progress(progress_stream, "phase", text="placing denoiser blocks on GPU")
    resident_buffers = _enable_resident_blocks(torch, runner, cases)
    timings["resident_upload_ms"] = _elapsed_ms(resident_upload_start)
    host_pin_start = time.perf_counter()
    _send_progress(progress_stream, "phase", text="pinning streamed weights")
    host_buffers = _pin_streamed_host_weights(torch, runner, len(resident_buffers))
    timings["host_pin_ms"] = _elapsed_ms(host_pin_start)

    latent_outputs: list[dict[str, Any]] = []
    denoise_total_start = time.perf_counter()
    output_count = sum(len(case["outputs"]) for case in cases)
    _send_progress(
        progress_stream,
        "begin",
        total=output_count * profile["steps"],
        text=f"denoising 0/{profile['steps']}",
    )
    output_index = 0
    for case in cases:
        condition = conditioned[case["name"]]
        for output in case["outputs"]:
            output_index += 1
            input_info = _input_info(pipe, case, output["seed"], init_empty_input_info, update_input_info_from_dict)
            input_info.txt_seq_lens = condition["txt_seq_lens"]
            runner.input_info = input_info
            text_encoder_output = {
                "prompt_embeds": condition["prompt_embeds"].to("cuda"),
                "image_info": {
                    "vae_image_info_list": condition["vae_image_info_list"],
                },
            }
            if profile["true_cfg_scale"] != 1.0:
                text_encoder_output["negative_prompt_embeds"] = condition["negative_prompt_embeds"].to("cuda")
            runner.inputs = {
                "text_encoder_output": text_encoder_output,
                "image_encoder_output": [
                    {"image_latents": item["image_latents"].to("cuda")}
                    for item in encoded_groups[condition["vae_group"]]
                ],
            }
            del text_encoder_output
            input_info.image_encoder_output = runner.inputs["image_encoder_output"]
            runner.set_target_shape()
            runner.set_img_shapes()
            seed_all(output["seed"])
            runner.model.scheduler.generator = torch.Generator(device="cuda").manual_seed(output["seed"])
            runner.model.scheduler.prepare(input_info)
            output_step = 0

            def report_denoise_step(_percent: float, _total: float) -> None:
                nonlocal output_step
                output_step += 1
                text = f"denoising {output_step}/{profile['steps']}"
                if output_count > 1:
                    text += f" (image {output_index}/{output_count})"
                _send_progress(progress_stream, "step", text=text)

            runner.progress_callback = report_denoise_step
            denoise_start = time.perf_counter()
            latents, generator = runner.run()
            torch.cuda.synchronize()
            latent_outputs.append(
                {
                    "name": output["name"],
                    "case": case["name"],
                    "seed": output["seed"],
                    "path": output["path"],
                    "width": case["width"],
                    "height": case["height"],
                    "latents": latents.detach().to("cpu"),
                    "denoise_ms": _elapsed_ms(denoise_start),
                }
            )
            del latents, generator, runner.inputs
            runner.model.scheduler.latents = None
            runner.model.scheduler.noise_pred = None
            runner.model.scheduler.input_info = None
            runner.input_info = None
            del input_info
            _release_cuda(torch)
    timings["denoise_ms"] = _elapsed_ms(denoise_total_start)
    phase_peaks["denoise"] = _cuda_peak_mib(torch)
    _send_progress(progress_stream, "phase", text="releasing denoiser")
    resident_block_count = len(resident_buffers)
    _release_resident_blocks(torch, runner, resident_buffers)
    runner.end_run()
    _release_cuda(torch)

    _reset_cuda_peak(torch)
    decode_start = time.perf_counter()
    _send_progress(progress_stream, "phase", text="decoding images")
    runner.vae = runner.load_vae()
    outputs = []
    for latent_output in latent_outputs:
        output_decode_start = time.perf_counter()
        input_info = init_empty_input_info(pipe.task, pipe.support_tasks)
        input_info.auto_width = latent_output["width"]
        input_info.auto_height = latent_output["height"]
        images = runner.vae.decode(latent_output["latents"].to("cuda"), input_info)
        image = images[0]
        expected_size = (latent_output["width"], latent_output["height"])
        if image.size != expected_size:
            raise RuntimeError(
                f"LightX2V decoded {latent_output['name']} at {image.size[0]}x{image.size[1]}, "
                f"expected {expected_size[0]}x{expected_size[1]}"
            )
        output_path = Path(latent_output["path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        outputs.append(
            {
                "name": latent_output["name"],
                "case": latent_output["case"],
                "seed": latent_output["seed"],
                "path": output_path.as_posix(),
                "width": latent_output["width"],
                "height": latent_output["height"],
                "denoise_ms": latent_output["denoise_ms"],
                "vae_decode_ms": _elapsed_ms(output_decode_start),
            }
        )
        image.close()
        del images
    torch.cuda.synchronize()
    timings["vae_decode_ms"] = _elapsed_ms(decode_start)
    phase_peaks["decode"] = _cuda_peak_mib(torch)
    del runner.vae, latent_outputs
    _release_cuda(torch)

    timings["total_ms"] = _elapsed_ms(total_start)
    host_after = _host_snapshot()
    swap_growth = (
        host_after["swap_free_kib"] < host_before["swap_free_kib"]
        or host_after["pswpout"] > host_before["pswpout"]
    )
    return {
        "status": "completed",
        "outputs": outputs,
        "timings_ms": timings,
        "memory": {
            "peak_vram_mib": phase_peaks,
            "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "read_gib": (host_after["read_bytes"] - host_before["read_bytes"]) / 1024**3,
            "mem_available_before_mib": host_before["mem_available_kib"] / 1024,
            "mem_available_after_mib": host_after["mem_available_kib"] / 1024,
            "swap_free_before_mib": host_before["swap_free_kib"] / 1024,
            "swap_free_after_mib": host_after["swap_free_kib"] / 1024,
            "pswpin_delta": host_after["pswpin"] - host_before["pswpin"],
            "pswpout_delta": host_after["pswpout"] - host_before["pswpout"],
            "timing_valid_no_swap_growth": not swap_growth,
        },
        "environment": {
            "engine": "LightX2V",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "matrix_multiply": "fp8-triton",
            "conditioner": "qwen25-vl-fp8-w8a8-language-bf16-vision",
            "attention": "flash_attn2",
            "rope": "torch",
            "host_buffers": host_buffers["mode"],
            "pinned_blocks": host_buffers["pinned_blocks"],
            "pinned_gib": round(host_buffers["pinned_gib"], 2),
            "streamed_blocks": host_buffers["streamed_blocks"],
            "resident_blocks": resident_block_count,
            "phase_order": "conditioner->vae_encode->transformer->vae_decode",
        },
    }


def _open_progress_stream() -> TextIO:
    sys.stdout.flush()
    progress_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return os.fdopen(progress_fd, "w", encoding="utf-8", buffering=1)


def _send_progress(stream: TextIO, kind: str, **values: Any) -> None:
    stream.write(json.dumps({"kind": kind, **values}, separators=(",", ":")) + "\n")


def _enable_resident_blocks(torch: Any, runner: Any, cases: list[dict[str, Any]]) -> list[Any]:
    """Keep as many transformer blocks permanently on the GPU as VRAM allows.

    The 60-block fp8 transformer does not fit in VRAM, so LightX2V streams every
    block from CPU each step. Pinned host memory is not usable on this WSL2 host,
    which makes that streaming the step-time bottleneck; parking part of the
    blocks on the GPU cuts the per-step transfer volume proportionally. Uses the
    engine's preallocated-buffer copy path instead of WeightModule.to_cuda():
    per-tensor .to() reallocation runs an order of magnitude slower here.
    """
    import copy
    import types

    infer = runner.model.transformer_infer
    if getattr(infer, "offload_manager", None) is None:
        return []
    blocks = runner.model.transformer_weights.blocks
    # Clone the engine's own stream buffer: it went through load(weight_dict) at
    # checkpoint time, which sets attributes (bias, scale buffers) that a freshly
    # constructed QwenImageTransformerAttentionBlock never receives. The shared
    # LockableDict config refuses deepcopy, so pre-seed the memo to keep every
    # config reference pointing at the original instead of copying it.
    buffer_template = infer.offload_manager.cuda_buffers[0]
    shared = [runner.config, getattr(runner.model, "config", None), buffer_template.config]
    deepcopy_memo = {id(obj): obj for obj in shared if obj is not None}
    max_pixels = max(case["width"] * case["height"] for case in cases)
    free_bytes, _ = torch.cuda.mem_get_info()
    # Activation reserve scales from the measured ~2.4 GiB denoise peak at ~1.11 MP,
    # plus a fixed safety margin for fragmentation and the double stream buffers.
    reserve_bytes = int(2.4 * 1024**3 * max_pixels / 1_110_000) + int(2.2 * 1024**3)
    budget = free_bytes - reserve_bytes
    resident_buffers: list[Any] = []
    block_bytes = 0
    for block_idx in range(len(blocks)):
        if resident_buffers and budget < block_bytes:
            break
        before = torch.cuda.memory_allocated()
        with runner.config.temporarily_unlocked():
            buffer = copy.deepcopy(buffer_template, dict(deepcopy_memo))
        buffer.load_state_dict(blocks[block_idx].state_dict(), block_idx, None)
        block_bytes = torch.cuda.memory_allocated() - before
        budget -= block_bytes
        resident_buffers.append(buffer)
        if budget < 0:
            break
    torch.cuda.synchronize()
    if not resident_buffers:
        return []
    resident = len(resident_buffers)

    def infer_with_resident_blocks(
        self: Any,
        blocks: Any,
        hidden_states: Any,
        encoder_hidden_states: Any,
        temb_img_silu: Any,
        temb_txt_silu: Any,
        image_rotary_emb: Any,
        modulate_index: Any,
    ) -> Any:
        manager = self.offload_manager
        streamed_start = min(resident, self.num_blocks)
        manager.compute_stream.wait_stream(torch.cuda.current_stream())
        if streamed_start < self.num_blocks:
            with torch.cuda.stream(manager.init_stream):
                manager.cuda_buffers[0].load_state_dict(blocks[streamed_start].state_dict(), streamed_start, None)
        for block_idx in range(streamed_start):
            self.block_idx = block_idx
            with torch.cuda.stream(manager.compute_stream):
                encoder_hidden_states, hidden_states = self.infer_block(
                    block=resident_buffers[block_idx],
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_img_silu=temb_img_silu,
                    temb_txt_silu=temb_txt_silu,
                    image_rotary_emb=image_rotary_emb,
                    modulate_index=modulate_index,
                )
        if streamed_start < self.num_blocks:
            manager._sync()
            for block_idx in range(streamed_start, self.num_blocks):
                self.block_idx = block_idx
                next_idx = block_idx + 1 if block_idx + 1 < self.num_blocks else streamed_start
                manager.prefetch_weights(next_idx, blocks)
                with torch.cuda.stream(manager.compute_stream):
                    encoder_hidden_states, hidden_states = self.infer_block(
                        block=manager.cuda_buffers[0],
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb_img_silu=temb_img_silu,
                        temb_txt_silu=temb_txt_silu,
                        image_rotary_emb=image_rotary_emb,
                        modulate_index=modulate_index,
                    )
                manager.swap_blocks()
        manager.compute_stream.synchronize()
        return hidden_states

    infer.infer_func = types.MethodType(infer_with_resident_blocks, infer)
    return resident_buffers


def _pin_streamed_host_weights(torch: Any, runner: Any, streamed_start: int) -> dict[str, Any]:
    """Page-lock the host weights of the blocks that still stream every step.

    The engine's cuda buffers already copy with non_blocking=True, but a pageable
    source silently degrades that to a staged, synchronous transfer: measured 3.2 GB/s
    against 50 GB/s from page-locked memory on this host. Streaming 36 blocks of ~340 MiB
    is what makes the step time, so this is the difference between a transfer-bound and a
    compute-bound denoise.

    Only the streamed blocks are pinned. Pinning at load time (LightX2V's own default)
    would lock all 60 blocks, i.e. the entire 20 GiB checkpoint, in unswappable memory --
    which does not fit next to the activations on a 30 GiB host. The resident blocks
    already live on the GPU and never transfer, so their host copies stay file-backed.
    """
    mode = os.environ.get("AIGEN_LIGHTX2V_HOST_BUFFERS", "pinned")
    blocks = runner.model.transformer_weights.blocks
    result: dict[str, Any] = {
        "mode": mode,
        "pinned_blocks": 0,
        "pinned_gib": 0.0,
        "streamed_blocks": max(0, len(blocks) - streamed_start),
    }
    if mode != "pinned" or streamed_start >= len(blocks):
        result["mode"] = "file_backed" if mode != "pinned" else "pinned"
        return result

    # Two ceilings apply. cudaHostAlloc on this WSL2 host refuses past ~14 GiB regardless
    # of free RAM (measured), and page-locked pages cannot be reclaimed under pressure, so
    # the interpreter, the CUDA context and the page cache backing the read need room left.
    # Whatever does not fit stays file-backed and simply streams at the old speed.
    available_bytes = _host_snapshot()["mem_available_kib"] * 1024
    budget = min(PINNED_HOST_BUDGET_BYTES, available_bytes - 8 * 1024**3)
    pinned_bytes = 0
    for block_idx in range(streamed_start, len(blocks)):
        block_bytes = _pin_block_host_weights(torch, blocks[block_idx], budget - pinned_bytes)
        if block_bytes is None:
            break
        pinned_bytes += block_bytes
        result["pinned_blocks"] += 1
    gc.collect()
    result["pinned_gib"] = pinned_bytes / 1024**3
    if result["pinned_blocks"] < result["streamed_blocks"]:
        result["mode"] = "pinned_partial"
    return result


def _pin_block_host_weights(torch: Any, block: Any, remaining_bytes: int) -> int | None:
    """Replace one block's file-backed host tensors with page-locked copies.

    Returns the number of bytes pinned, or None when the block does not fit the budget.
    A block is pinned all-or-nothing: a half-pinned block would still stall the step on
    its pageable remainder. The copies are contiguous, which also removes the transposed
    views the file-backed path hands to copy_() -- those cannot take the DMA fast path.
    """
    targets = []
    block_bytes = 0
    for holder in _iter_weight_objects(block):
        for attribute, value in vars(holder).items():
            if not attribute.startswith("pin_") or not isinstance(value, torch.Tensor):
                continue
            if value.device.type != "cpu" or value.is_pinned():
                continue
            targets.append((holder, attribute, value))
            block_bytes += value.numel() * value.element_size()
    if block_bytes > remaining_bytes:
        return None
    # All-or-nothing: a half-pinned block still stalls the step on its pageable remainder,
    # so allocate everything first and only publish once the whole block is page-locked.
    pinned_tensors = []
    try:
        for _, _, value in targets:
            pinned = torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
            pinned.copy_(value)
            pinned_tensors.append(pinned)
    except RuntimeError:
        return None
    for (holder, attribute, _), pinned in zip(targets, pinned_tensors):
        setattr(holder, attribute, pinned)
    return block_bytes


def _iter_weight_objects(module: Any) -> Any:
    """Walk every object in a block's weight tree, leaves included.

    The engine's own named_parameters() cannot be used here: it recurses into _modules as
    if every entry were a WeightModule, but the weight leaves are registered there too and
    carry no such method.
    """
    stack = [module]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        for children in (getattr(node, "_parameters", {}), getattr(node, "_modules", {})):
            stack.extend(child for child in children.values() if child is not None)


def _release_resident_blocks(torch: Any, runner: Any, resident_buffers: list[Any]) -> None:
    if not resident_buffers:
        return
    infer = runner.model.transformer_infer
    infer.infer_func = infer.infer_with_blocks_offload
    resident_buffers.clear()
    _release_cuda(torch)


def _input_info(pipe: Any, case: dict[str, Any], seed: int, init_empty_input_info: Any, update_input_info_from_dict: Any) -> Any:
    pipe.seed = seed
    pipe.prompt = case["prompt"]
    pipe.negative_prompt = ""
    pipe.image_path = ",".join(case["image_paths"])
    pipe.save_result_path = None
    pipe.return_result_tensor = False
    pipe.target_shape = [case["height"], case["width"]]
    input_info = init_empty_input_info(pipe.task, pipe.support_tasks)
    update_input_info_from_dict(input_info, pipe)
    return input_info


def _file_backed_cpu_tensor_factory() -> Any:
    def preserve_file_backed_tensor(tensor: Any, transpose: bool = False, dtype: Any = None) -> Any:
        target = tensor if dtype is None or tensor.dtype == dtype else tensor.to(dtype)
        if transpose:
            target = target.t()
        return target

    return preserve_file_backed_tensor


def _host_snapshot() -> dict[str, int]:
    meminfo = _key_value_file(Path("/proc/meminfo"))
    vmstat = _key_value_file(Path("/proc/vmstat"))
    process_io = _key_value_file(Path("/proc/self/io"))
    return {
        "mem_available_kib": meminfo["MemAvailable"],
        "swap_free_kib": meminfo["SwapFree"],
        "pswpin": vmstat["pswpin"],
        "pswpout": vmstat["pswpout"],
        "read_bytes": process_io["read_bytes"],
    }


def _key_value_file(path: Path) -> dict[str, int]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator:
            key, _, raw_value = line.partition(" ")
        values[key] = int(raw_value.strip().split()[0])
    return values


def _reset_cuda_peak(torch: Any) -> None:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _cuda_peak_mib(torch: Any) -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


def _release_cuda(torch: Any) -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


if __name__ == "__main__":
    raise SystemExit(main())
