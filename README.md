# aigen

Private AI character concept pipeline. It is intentionally small: download the
chosen model, run a direct character generator, inspect the output, iterate.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
```

Models are declared in source manifests and downloaded into `aigen/models`. Hub
repo IDs and revisions are pinned for reproducible runs.

```bash
.venv/bin/python -m aigen.cli models download \
  --manifest model_sources/character_pipeline.json \
  --models-root aigen/models \
  --dry-run
```

Remove `--dry-run` after accepting the model license on Hugging Face.

The included starter manifest targets an RTX 5070 Ti with 16 GB VRAM. It uses a
4-bit Diffusers FLUX Kontext conversion for local iteration. The full BF16 BFL
model remains the quality reference and is exposed as an explicit production
profile.

```bash
.venv/bin/python -m aigen.cli generate character-concept \
  --profile local \
  --reference-image ../ai-art/references/characters/ai51.png \
  --prompt "Create production-quality full-body anime character concept art for a game. Preserve the same character identity, face, outfit language, blue eyes, short reddish-brown hair, leather jacket, skirt, long boots, and confident readable silhouette. Improve polish, anatomy, line clarity, material rendering, and game character appeal." \
  --output runs/characters/ai51_concept.png \
  --seed 1
```

## Production BF16

The production profile uses the downloaded BFL BF16 transformer single-file
weights through Diffusers CPU offload. On the 16 GB RTX 5070 Ti this is a
quality path, not an interactive path.

```bash
.venv/bin/python -m aigen.cli generate character-concept \
  --profile production \
  --reference-image ../ai-art/references/characters/ai51.png \
  --prompt "Create production-quality full-body anime character concept art for a game. Preserve the same character identity, face, outfit language, blue eyes, short reddish-brown hair, leather jacket, skirt, long boots, and confident readable silhouette. Improve polish, anatomy, line clarity, material rendering, and game character appeal." \
  --output runs/characters/ai51_production_bf16.png \
  --seed 1
```

For a larger final frame, override only the dimensions and step count:

```bash
.venv/bin/python -m aigen.cli generate character-concept \
  --profile production \
  --reference-image ../ai-art/references/characters/ai51.png \
  --prompt "Create production-quality full-body anime character concept art for a game. Preserve the same character identity, face, outfit language, blue eyes, short reddish-brown hair, leather jacket, skirt, long boots, and confident readable silhouette. Improve polish, anatomy, line clarity, material rendering, and game character appeal." \
  --output runs/characters/ai51_production_bf16_large.png \
  --width 1024 \
  --height 1536 \
  --steps 32 \
  --seed 1
```

There is also a separate manifest for the official full BF16 Diffusers shard
layout. It intentionally downloads only the Diffusers subfolders and skips the
duplicate root safetensors files.

```bash
.venv/bin/python -m aigen.cli models download \
  --manifest model_sources/character_pipeline_full_bf16_offload.json \
  --models-root aigen/models \
  --dry-run
```

## Pose Control

For stricter pose direction, use the FLUX ControlNet pose pipeline. It expects a
DWPose/OpenPose-style pose control image; the pose image, not the text prompt,
owns the body layout.

```bash
.venv/bin/python -m aigen.cli models download \
  --manifest model_sources/pose_control_pipeline.json \
  --models-root aigen/models \
  --dry-run
```

For local pose iteration on the RTX 5070 Ti, use the 4-bit profile. It keeps the
quantized FLUX base on CUDA and combines it with the Shakker pose ControlNet:

```bash
.venv/bin/python -m aigen.cli models download \
  --manifest model_sources/pose_control_pipeline_4bit.json \
  --models-root aigen/models \
  --dry-run
```

After downloading the 4-bit FLUX base and the pose ControlNet:

```bash
.venv/bin/python -m aigen.cli generate character-pose \
  --profile flux-pose-4bit \
  --pose-image ../ai-art/references/poses/running_openpose.png \
  --prompt "Anime game character concept art of the same girl: blue eyes, short reddish-brown bob haircut, white shirt, blue tie, burgundy leather jacket, skirt, gloves, long blue socks, burgundy boots." \
  --output runs/characters/pose_control/ai51_pose_controlled.png \
  --seed 1
```

The `flux-pose-4bit` profile uses Shakker-Labs FLUX.1-dev ControlNet Union Pro
2.0 defaults for pose: `controlnet_conditioning_scale=0.9`,
`control_guidance_end=0.65`, `guidance_scale=3.5`, and 20 steps. Use
`--controlnet-conditioning-scale` when you want to trade pose strictness against
image polish.

For same-character pose work, use the Kontext pose route. Kontext owns character
identity from the reference image; ControlNet owns only the generated-token pose
prefix. With `--controlnet-conditioning-scale 0`, this route becomes the plain
Kontext baseline for the same seed. The local profile is tuned for iteration on
the RTX 5070 Ti: it generates `384x576`, caps the reference canvas to
`384x768`, uses `max_sequence_length=128`, disables VAE tiling, and stops pose
ControlNet at half the denoising schedule.

```bash
.venv/bin/python -m aigen.cli models download \
  --manifest model_sources/kontext_pose_control_pipeline_4bit.json \
  --models-root aigen/models \
  --dry-run
```

```bash
.venv/bin/python -m aigen.cli generate character-kontext-pose \
  --profile local \
  --reference-image ../ai-art/references/characters/ai51.png \
  --pose-image ../ai-art/references/poses/running_openpose.png \
  --prompt "Anime game character concept art of the same girl: blue eyes, short reddish-brown bob haircut, white shirt, blue tie, burgundy leather jacket, skirt, gloves, long blue socks, burgundy boots." \
  --output runs/characters/pose_control/ai51_kontext_pose_controlled.png \
  --seed 1
```

Initial pose sweep with fixed noise:

```bash
for scale in 0.40 0.50 0.60; do
  for end in 0.40 0.50 0.60; do
  .venv/bin/python -m aigen.cli generate character-kontext-pose \
    --profile local \
    --reference-image ../ai-art/references/characters/ai51.png \
    --pose-image ../ai-art/references/poses/running_openpose.png \
    --prompt "Anime game character concept art of the same girl: blue eyes, short reddish-brown bob haircut, white shirt, blue tie, burgundy leather jacket, skirt, gloves, long blue socks, burgundy boots." \
    --controlnet-conditioning-scale "$scale" \
    --control-guidance-end "$end" \
    --output "runs/characters/pose_control/ai51_kontext_pose_scale_${scale}_end_${end}.png" \
    --seed 1
  done
done
```

The `production` Kontext pose profile is the BF16/offload comparison path. It
uses a larger `512x1024` reference budget and VAE tiling. It is much slower on
16 GB VRAM and should not be used for normal iteration.

## Nunchaku Kontext Benchmark

The bitsandbytes FP4 Kontext transformer is too slow for interactive sweeps on
this RTX 5070 Ti setup, so the next backend benchmark uses Nunchaku's Blackwell
FP4 FLUX Kontext transformer. Install the wheel that matches this Python,
PyTorch, and CUDA build:

```bash
.venv/bin/python -m pip install --no-deps \
  https://github.com/nunchaku-tech/nunchaku/releases/download/v1.3.0dev20260306/nunchaku-1.3.0.dev20260306%2Bcu12.8torch2.12-cp312-cp312-linux_x86_64.whl
```

Download the Nunchaku transformer into `aigen/models`:

```bash
.venv/bin/python -m aigen.cli models download \
  --manifest model_sources/nunchaku_kontext_pipeline_fp4.json \
  --models-root aigen/models
```

Run a plain Kontext three-step timing pass at the same preview token budget used
for the current pose experiments:

```bash
.venv/bin/python -m aigen.cli generate character-nunchaku-kontext \
  --profile local \
  --reference-image ../ai-art/references/characters/ai51.png \
  --prompt "Anime game character concept art of the same girl: blue eyes, short reddish-brown bob haircut, white shirt, blue tie, burgundy leather jacket, skirt, gloves, long blue socks, burgundy boots." \
  --output runs/characters/benchmarks/ai51_nunchaku_kontext_3step.png \
  --seed 1 \
  --compact
```

For pose-controlled same-character work, use the fused Nunchaku Kontext pose
route. It keeps the existing prefix-only ControlNet residual logic and swaps only
the Kontext transformer backend. The local profile uses Diffusers pipeline
offload with `controlnet` excluded from the offload sequence, keeps Nunchaku
layer offload disabled, and enables `nunchaku-fp16` attention. Fully resident
ControlNet plus Nunchaku fits in 16 GB, but it pushes the Nunchaku transformer
onto a much slower path on this RTX 5070 Ti.

The Nunchaku pose profiles are split by intent:

- `benchmark`: three-step scale-zero timing.
- `local`: 20-step pose iteration at `384x576`, scale `0.50`, end `0.50`.
- `quality`: 28-step higher-token pass at `512x768`.

```bash
.venv/bin/python -m aigen.cli generate character-nunchaku-kontext-pose \
  --profile local \
  --reference-image ../ai-art/references/characters/ai51.png \
  --pose-image runs/characters/pose_control/ai51_running_openpose_control_half.png \
  --prompt "Same anime girl running. Blue eyes, short reddish-brown bob, white shirt, blue tie, burgundy leather jacket, skirt, gloves, blue socks, burgundy boots." \
  --output runs/characters/pose_control/ai51_nunchaku_kontext_pose.png \
  --steps 20 \
  --controlnet-conditioning-scale 0.50 \
  --control-guidance-end 0.50 \
  --cpu-offload \
  --no-nunchaku-layer-offload \
  --seed 1
```

For controlled background-leakage checks, run the phase-batched sweep. It loads
the pipeline once, prepares the current and nearest-resized pose conditions,
denoises all variants first, and decodes the images in small VAE chunks:

```bash
.venv/bin/python -m aigen.cli generate character-nunchaku-kontext-pose-sweep \
  --profile local \
  --reference-image ../ai-art/references/characters/ai51.png \
  --pose-image runs/characters/pose_control/ai51_running_openpose_control_half.png \
  --prompt "Same anime girl running. Blue eyes, short reddish-brown bob, white shirt, blue tie, burgundy leather jacket, skirt, gloves, blue socks, burgundy boots. clean plain neutral studio background, uniform soft gray backdrop, no graphic lines or colored streaks." \
  --output-dir runs/characters/pose_control/ai51_background_ablation \
  --seed 1 \
  --compact
```

For the next quality pass, keep the normal control-map path and compare pose
strength against guidance duration at the quality profile size:

```bash
.venv/bin/python -m aigen.cli generate character-nunchaku-kontext-pose-sweep \
  --profile quality \
  --sweep-variant-set quality-strength \
  --reference-image ../ai-art/references/characters/ai51.png \
  --pose-image runs/characters/pose_control/ai51_running_openpose_control_half.png \
  --prompt "Same anime girl running. Blue eyes, short reddish-brown bob, white shirt, blue tie, burgundy leather jacket, skirt, gloves, blue socks, burgundy boots. clean plain neutral studio background, uniform soft gray backdrop, no graphic lines or colored streaks." \
  --output-dir runs/characters/pose_control/ai51_quality_strength \
  --seed 1 \
  --compact
```

For generation dependencies:

```bash
.venv/bin/python -m pip install -e ".[generation]"
```

```bash
.venv/bin/python -m unittest discover -s tests
```
