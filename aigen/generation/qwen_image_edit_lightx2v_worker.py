from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any


os.environ.setdefault("PROFILING_DEBUG_LEVEL", "1")
os.environ.setdefault("DTYPE", "BF16")
os.environ.setdefault("SENSITIVE_LAYER_DTYPE", "None")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: qwen_image_edit_lightx2v_worker REQUEST RESPONSE")
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        response = _run(request)
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


def _run(request: dict[str, Any]) -> dict[str, Any]:
    import torch
    from PIL import Image
    from lightx2v import LightX2VPipeline
    from lightx2v.common.ops import utils as ops_utils
    from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
    from lightx2v.utils.utils import seed_all

    profile = request["profile"]
    cases = request["cases"]
    total_start = time.perf_counter()
    host_before = _host_snapshot()
    phase_peaks: dict[str, float] = {}
    timings: dict[str, float] = {}

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
        text_encoder_quantized=True,
        text_encoder_quantized_ckpt=profile["conditioner_model"],
        text_encoder_quant_scheme="int4",
    )
    pipe.tokenizer_max_length = profile["max_sequence_length"]
    pipe.create_generator(
        attn_mode="flash_attn2",
        infer_steps=profile["steps"],
        guidance_scale=profile["guidance_scale"],
        rope_type="torch",
        resize_mode="adaptive",
    )
    runner = pipe.runner

    _reset_cuda_peak(torch)
    conditioner_load_start = time.perf_counter()
    with runner.config.temporarily_unlocked():
        runner.config["cpu_offload"] = False
        runner.config["qwen25vl_cpu_offload"] = False
    runner.text_encoders = runner.load_text_encoder()
    torch.cuda.synchronize()
    timings["conditioner_load_ms"] = _elapsed_ms(conditioner_load_start)
    encode_start = time.perf_counter()
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
            conditioned[case["name"]] = {
                "prompt_embeds": text_output["prompt_embeds"].detach().to("cpu"),
                "vae_group": group_key,
                "vae_image_info_list": image_info["vae_image_info_list"],
                "txt_seq_lens": list(input_info.txt_seq_lens),
            }
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

    ops_utils.create_pin_tensor = _regular_cpu_tensor_factory(torch)
    _reset_cuda_peak(torch)
    transformer_load_start = time.perf_counter()
    runner.model = runner.load_transformer()
    runner.model.set_scheduler(runner.scheduler)
    torch.cuda.synchronize()
    timings["transformer_load_ms"] = _elapsed_ms(transformer_load_start)

    latent_outputs: list[dict[str, Any]] = []
    denoise_total_start = time.perf_counter()
    for case in cases:
        condition = conditioned[case["name"]]
        for output in case["outputs"]:
            input_info = _input_info(pipe, case, output["seed"], init_empty_input_info, update_input_info_from_dict)
            input_info.txt_seq_lens = condition["txt_seq_lens"]
            runner.input_info = input_info
            runner.inputs = {
                "text_encoder_output": {
                    "prompt_embeds": condition["prompt_embeds"].to("cuda"),
                    "image_info": {
                        "vae_image_info_list": condition["vae_image_info_list"],
                    },
                },
                "image_encoder_output": [
                    {"image_latents": item["image_latents"].to("cuda")}
                    for item in encoded_groups[condition["vae_group"]]
                ],
            }
            input_info.image_encoder_output = runner.inputs["image_encoder_output"]
            runner.set_target_shape()
            runner.set_img_shapes()
            seed_all(output["seed"])
            runner.model.scheduler.generator = None
            runner.model.scheduler.prepare(input_info)
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
    runner.end_run()
    _release_cuda(torch)

    _reset_cuda_peak(torch)
    decode_start = time.perf_counter()
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
            "attention": "flash_attn2",
            "rope": "torch",
            "host_buffers": "regular_cpu",
            "phase_order": "conditioner->vae_encode->transformer->vae_decode",
        },
    }


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


def _regular_cpu_tensor_factory(torch: Any) -> Any:
    def create_regular_cpu_tensor(tensor: Any, transpose: bool = False, dtype: Any = None) -> Any:
        target = torch.empty(tensor.shape, dtype=dtype or tensor.dtype)
        target.copy_(tensor)
        if transpose:
            target = target.t()
        del tensor
        return target

    return create_regular_cpu_tensor


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
