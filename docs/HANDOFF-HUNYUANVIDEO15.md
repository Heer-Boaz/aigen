# HunyuanVideo-1.5 16 GB I2V review handoff

## Review contract

This is a read-only review assignment for Claude. Do not run inference, install
packages, download models, edit files, or commit changes. Inspect the official
source, this integration and the existing logs, then report likely root causes
and the smallest evidence-backed next experiment.

Use only Tencent's official repository and official Hugging Face checkpoint as
model/runtime authorities:

- <https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5>
- <https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main/transformer/480p_i2v_step_distilled>

Community comments inside Tencent's issue tracker may be treated as leads, not
as authoritative fixes. Do not propose an older generic HunyuanVideo pipeline,
an unofficial four-step model, 720p, super-resolution, or a different backbone.

## Objective

Run Tencent's official 480p image-to-video step-distilled checkpoint on an RTX
5070 Ti with 16 GB VRAM:

- image-to-video
- `480p_i2v_step_distilled`
- 8 inference steps
- CFG 1
- flow shift 7
- short 49-frame clip
- super-resolution disabled
- prompt rewriting disabled
- CPU model offloading enabled
- transformer group offloading enabled

The initial visual test uses:

- image: `assets/characters/jillian/a-pose-threequarter.jpg`
- seed: 42
- motion prompt: `Subtle natural breathing and one blink. Slight secondary motion in the hair and clothing. The camera remains completely still.`

The prompt was separately reviewed before execution and deliberately contains
only motion and camera instructions. Prompt wording is not implicated in the
runtime failure.

## Pinned provenance

- Official source revision and current upstream `main`:
  `60783e704160023913bee78f0b47036d393d4dfa`
- Official Hugging Face revision:
  `9b49404b3f5df2a8f0b31df27a0c7ab872e7b038`
- Exact transformer:
  `transformer/480p_i2v_step_distilled/diffusion_pytorch_model.safetensors`
- Transformer size: 33,325,523,336 bytes
- VAE size: 5,042,560,980 bytes
- Selected local model set: approximately 40 GB

The local official checkout is:

```text
/home/boaz/.cache/aigen-hunyuanvideo15/HunyuanVideo-1.5
```

It is exactly the pinned commit plus the single tracked patch described below.
The runtime virtual environment is:

```text
/home/boaz/.cache/aigen-hunyuanvideo15/venv
```

Observed environment:

```text
GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB
driver: 610.74
compute capability: 12.0
system RAM visible to WSL: 30 GiB
swap: 40 GiB
Python: 3.12
torch: 2.8.0+cu128
CUDA runtime: 12.8
diffusers: 0.35.0
transformers: 4.57.1
flash-attn: 2.8.3
```

During the cold run, host swap usage rose substantially while the 33.3 GB BF16
transformer and other components were staged. Current low swap usage is after
the failed process exited and memory was reclaimed.

## Repository changes under review

- `aigen/generation/hunyuanvideo15.py`: validates the pinned runtime/model and
  invokes official `generate.py` with an explicit fixed profile.
- `aigen/hunyuanvideo15_commands.py`: direct CLI surface.
- `aigen/cli.py`: command registration.
- `scripts/install_hunyuanvideo15.sh`: isolated official runtime installer.
- `scripts/download_hunyuanvideo15.sh`: selective official model download and
  reuse of the existing exact Qwen2.5-VL model through a symlink.
- `model_sources/hunyuanvideo15_480p_i2v_step_distilled.json`: pinned selective
  Hugging Face downloads.
- `patches/hunyuanvideo15/0001-release-cuda-cache-after-component-offload.patch`:
  adds one `torch.cuda.empty_cache()` call after a component is returned to the
  CPU.

The runner intentionally forces:

```text
--resolution 480p
--num_inference_steps 8
--video_length 49
--dtype bf16
--enable_step_distill true
--cfg_distilled false
--offloading true
--group_offloading true
--overlap_group_offloading false
--sr false
--rewrite false
--enable_cache false
--sparse_attn false
--use_sageattn false
--enable_torch_compile false
--use_fp8_gemm false
```

It also sets Tencent's recommended allocator configuration:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

The generated configuration banner confirmed that Tencent selected:

```text
task: I2V
resolution: 512 x 768
frames: 49
guidance scale: 1.0
flow shift: 7.0
sampling steps: 8
Meanflow: true
attention: flash
transformer dtype: bfloat16
```

The source image is 1684 x 2528. Tencent selected its 512 x 768 vertical bucket;
there was no forced square resize or aspect-ratio stretching.

## Validation history

No attempt produced an MP4 or generation-config JSON. Only logs exist.

### Attempt 1: official source without local patch

Log:

```text
runs/hunyuanvideo15/jillian-threequarter-breathe-seed42.mp4.log
```

The Qwen text encoder approached the VRAM limit. At the next component boundary,
official `auto_offload_model` attempted to move the VAE to CUDA and failed:

```text
with auto_offload_model(self.vae, self.execution_device, enabled=self.enable_offloading)
model.to(device)
RuntimeError: CUDA driver error: device not ready
```

The run began with roughly 2.8 GB of unrelated VRAM usage, so a clean retry was
performed.

### Attempt 2: clean GPU baseline, official source without local patch

Log:

```text
runs/hunyuanvideo15/jillian-threequarter-breathe-seed42-v2.mp4.log
```

The GPU began near 455 MB. The same Qwen-to-VAE transition failed at the same
`model.to(device)` call. This rejected stale baseline VRAM as the primary cause.

Inspection showed that official `auto_offload_model` moved the completed model
back to CPU but did not release PyTorch's CUDA allocator cache. The tracked
one-line patch was added at that component lifecycle boundary.

### Attempt 3: official source plus CUDA cache-release patch

Log:

```text
runs/hunyuanvideo15/jillian-threequarter-breathe-seed42-v3.mp4.log
```

The patch changed observed behavior in the intended place:

1. Qwen peaked around 15.0 GB.
2. After Qwen returned to CPU, VRAM fell to about 1.1 GB.
3. The VAE and subsequent component transitions completed instead of crashing.
4. The transformer then rose to approximately 15.93 of 16.30 GB.
5. Before completing denoise step 1 of 8, the first double block failed in
   `img_attn_v` after about 2 minutes inside the step:

```text
The module 'HunyuanVideo_1_5_DiffusionTransformer' is group offloaded and moving it using `.to()` is not supported.
...
img_v = self.img_attn_v(img_modulated)
...
RuntimeError: CUDA driver error: device not ready
```

The entire command had run for about 24 minutes. Most of that time was cold
component construction and staging; no denoise step completed.

## Primary code-level suspicion

Review this before proposing quantization or compilation.

At the pinned official revision,
`hyvideo/commons/__init__.py::auto_offload_model` imports
`_is_group_offload_enabled` but never calls it:

```python
@contextmanager
def auto_offload_model(models, device, enabled=True):
    from diffusers.hooks.group_offloading import _is_group_offload_enabled
    if enabled:
        if isinstance(models, nn.Module):
            models = [models]
        for model in models:
            if model is not None:
                model.to(device)
    yield
    if enabled:
        for model in models:
            if model is not None:
                model.to(torch.device('cpu'))
```

The denoise loop in
`hyvideo/pipelines/hunyuan_video_pipeline.py` wraps the transformer in that
context even after `enable_group_offload(...)` has installed Diffusers hooks:

```python
with self.progress_bar(total=num_inference_steps) as progress_bar, \
        auto_offload_model(
            self.transformer,
            self.execution_device,
            enabled=self.enable_offloading,
        ):
    ...
```

This produces the explicit warning that a group-offloaded transformer is being
moved with `.to()`. Determine from Diffusers 0.35.0 semantics whether this:

- defeats or partially defeats group offloading;
- leaves incompatible device state on hooked linear layers;
- explains the 15.93 GB peak and first-block failure; and
- was intended to be guarded with `_is_group_offload_enabled(model)`.

Do not assume the existing `empty_cache()` patch is the final fix. It is proven
only to release the Qwen cache and allow the next component to start.

## Other official options to evaluate on paper

### Non-overlapped versus overlapped group offloading

The first run deliberately used `--overlap_group_offloading false` because
Tencent says overlap significantly increases CPU memory use and this WSL guest
has only 30 GiB RAM. However, Tencent's implementation uses:

```python
'num_blocks_per_group': 1 if overlap_group_offloading else 4
```

Therefore the memory trade-off may be important: disabling overlap saves host
RAM but groups four transformer blocks together on the GPU. Analyze whether that
alone makes the BF16 path exceed 16 GB, and whether overlap could fit only by
using more swap or would be operationally untenable.

### Official FP8 GEMM

Tencent exposes:

```text
--use_fp8_gemm true
--quant_type fp8-per-token-sgl
--include_patterns double_blocks
```

Their README requires `sgl-kernel==0.3.18`. It has not been installed or tested.
Assess whether this official dynamic transformer quantization is supported on
SM 12.0 with the pinned Torch/CUDA versions, whether it works together with
group offloading, and whether a weight-only quantization mode would be the more
appropriate first memory validation. Do not recommend an unofficial checkpoint.

### Compilation and feature cache

`--enable_torch_compile true` compiles each transformer double block. It does
not reduce the 33.3 GB cold load and adds first-call compilation overhead, so it
does not appear to address the current pre-step failure. It may matter only
after a stable warm pipeline exists.

Official `generate.py` explicitly rejects `enable_step_distill && enable_cache`
because Tencent says the combination degrades performance. Feature cache is not
a candidate for this exact checkpoint.

## GitHub issue audit

Open and closed issues plus issue comments in the official repository were
searched for:

```text
device not ready
CUDA driver
auto_offload_model
empty_cache
VAE offloading
offloading VAE
16 GB
14 GB
OOM
```

There is no issue addressing this Qwen-to-VAE or group-offloaded-transformer
failure. The exact search currently returns no results:

<https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/issues?q=is%3Aissue+%22device+not+ready%22>

Issue 12 discusses a separate VAE decode OOM and smaller VAE tiles, not this
startup/first-transformer-block failure:

<https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/issues/12>

Issue 43 concerns VAE memory during training, not inference:

<https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/issues/43>

## Requested review output

Return a concise, line-level analysis covering:

1. Whether the unused `_is_group_offload_enabled` import and unconditional
   transformer `.to()` calls are a real official lifecycle bug.
2. Whether the local cache-release patch is correct, incomplete, misplaced, or
   masking a more precise owner fix.
3. The expected GPU- and host-memory consequences of one-block overlapped versus
   four-block non-overlapped group offloading on this machine.
4. Whether Tencent's official FP8-GEMM path is the correct next fallback for a
   16 GB Blackwell card, including the exact supported quantization mode and
   dependencies.
5. Any version mismatch or official flag interaction we missed.
6. One smallest next experiment, but do not execute it.

Separate confirmed facts from inferences. Do not suggest prompt changes: the
failure occurs before the model completes its first denoise step.

## Useful read-only commands

```bash
git show --stat --oneline HEAD
git diff HEAD^ -- aigen scripts model_sources patches docs README.md

source_root=/home/boaz/.cache/aigen-hunyuanvideo15/HunyuanVideo-1.5
git -C "$source_root" rev-parse HEAD
git -C "$source_root" diff --no-ext-diff --binary
sed -n '228,246p' "$source_root/hyvideo/commons/__init__.py"
sed -n '1190,1245p' "$source_root/hyvideo/pipelines/hunyuan_video_pipeline.py"
sed -n '1398,1418p' "$source_root/hyvideo/pipelines/hunyuan_video_pipeline.py"

tail -n 100 runs/hunyuanvideo15/jillian-threequarter-breathe-seed42.mp4.log
tail -n 100 runs/hunyuanvideo15/jillian-threequarter-breathe-seed42-v2.mp4.log
tail -n 120 runs/hunyuanvideo15/jillian-threequarter-breathe-seed42-v3.mp4.log
```

The three logs are local ignored run evidence and are intentionally not part of
the Git commit.
