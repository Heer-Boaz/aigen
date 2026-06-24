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
Kontext baseline for the same seed.

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

Initial pose-strength sweep:

```bash
for scale in 0.0 0.45 0.65 0.85; do
  .venv/bin/python -m aigen.cli generate character-kontext-pose \
    --profile local \
    --reference-image ../ai-art/references/characters/ai51.png \
    --pose-image ../ai-art/references/poses/running_openpose.png \
    --prompt "Anime game character concept art of the same girl: blue eyes, short reddish-brown bob haircut, white shirt, blue tie, burgundy leather jacket, skirt, gloves, long blue socks, burgundy boots." \
    --controlnet-conditioning-scale "$scale" \
    --output "runs/characters/pose_control/ai51_kontext_pose_scale_${scale}.png" \
    --seed 1
done
```

The `flux-pose` profile is the BF16/offload comparison path. It is much slower
on 16 GB VRAM and should not be used for normal iteration.

For generation dependencies:

```bash
.venv/bin/python -m pip install -e ".[generation]"
```

```bash
.venv/bin/python -m unittest discover -s tests
```
