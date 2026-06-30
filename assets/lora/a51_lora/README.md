# a51_lora

Portable AI51 LoRA package for the repo.

Contents:

- `images/`: selected training and validation images with per-image captions.
- `weights/final.safetensors`: final trained LoRA weights.
- `weights/checkpoint_600.safetensors`: useful intermediate checkpoint kept as an alternate.
- `settings.json`: compact package index with training, source, image-review and quality-check settings.
- `settings/`: full manifests and reports copied from the original runs.
- `quality/`: small quality-check samples for the packaged weights.

Use trigger token `ai51char` in prompts. The useful tested ranges were final weights at LoRA strength `0.4` and checkpoint 600 at strength `0.5`; full strength `1.0` is kept only as evidence because it overfits.
