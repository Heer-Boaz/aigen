"""Compare the pinned LightX2V FP32 intermediate with a direct FP8 store."""
import argparse
import gc
import json
from pathlib import Path

import torch
from triton import next_power_of_2

from lightx2v.common.ops.mm.triton_kernels import fp8_quantize_kernel, fp8_quantize_triton


def quantize(x, output_dtype):
    shape = x.shape
    flat = x.view(-1, shape[-1])
    out = torch.empty(shape, dtype=output_dtype, device=x.device)
    scales = torch.empty(flat.shape[0], dtype=torch.bfloat16, device=x.device)
    fp8_quantize_kernel[(flat.shape[0],)](
        flat, out, scales, shape[-1], next_power_of_2(shape[-1]),
        FP8_MAX_VAL=448.0, num_warps=8,
    )
    return out.to(torch.float8_e4m3fn), scales.view(shape[:-1])


def fixtures(width, dtype):
    generator = torch.Generator().manual_seed(821)
    random = torch.randn((7, width), generator=generator)
    random *= torch.tensor([0., 1e-35, 1e-10, 1., 1e10, 1e30, 1e35])[:, None]
    yield "scaled_random", random.to(device="cuda", dtype=dtype)

    positive = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    middle = (positive[:-1] + positive[1:]) * 0.5
    boundaries = torch.cat((positive, middle,
        torch.nextafter(middle, torch.full_like(middle, float("inf"))),
        torch.nextafter(middle, torch.full_like(middle, -float("inf")))))
    values = torch.cat((boundaries, -boundaries, torch.tensor([0., -0., 448., -448.])))
    row = values.repeat((width + values.numel() - 1) // values.numel())[:width]
    row[-1] = 448.  # fixes the row scale at one, including exact conversion ties
    yield "rounding_subnormal_signed_zero", row.repeat(3, 1).to(device="cuda", dtype=dtype)
    yield "all_zero", torch.zeros((3, width), device="cuda", dtype=dtype)
    yield "negative_zero", torch.full((3, width), -0., device="cuda", dtype=dtype)


def run(mode):
    candidate = fp8_quantize_triton if mode == "installed" else lambda x: quantize(x, torch.float8_e4m3fn)
    records = []
    for width in (3072, 12288):
        for dtype in (torch.bfloat16, torch.float32):
            for name, x in fixtures(width, dtype):
                expected = quantize(x, torch.float32)
                actual = candidate(x)
                identical = [torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                             for a, b in zip(expected, actual)]
                record = {"width": width, "dtype": str(dtype), "case": name,
                          "quantized_bytes_equal": identical[0], "scale_bytes_equal": identical[1]}
                records.append(record)
                assert all(identical), record
                del expected, actual, x
    peaks = {}
    for label, operation in (("fp32_intermediate", lambda x: quantize(x, torch.float32)),
                             ("direct_fp8", candidate)):
        gc.collect()
        torch.cuda.empty_cache()
        x = torch.randn((2048, 12288), device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = operation(x)
        torch.cuda.synchronize()
        peaks[label] = (torch.cuda.max_memory_allocated() - baseline) / 1024**2
        del x, result
    return {"mode": mode, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
            "status": "passed", "cases": records, "peak_additional_mib": peaks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("candidate", "installed"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.mode)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
