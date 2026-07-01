# a51_lora

Rejected AI51 LoRA package kept for forensics.

This is not a valid training dataset and not a production LoRA. Human review
found the selected image set inconsistent enough to invalidate the LoRA test as
evidence against LoRA itself.

Contents:

- `images/`: selected training and validation images with per-image captions.
- `weights/final.safetensors`: final trained LoRA weights.
- `weights/checkpoint_600.safetensors`: useful intermediate checkpoint kept as an alternate.
- `settings.json`: compact package index with training, source, image-review and quality-check settings.
- `settings/`: full manifests and reports copied from the original runs.
- `quality/`: small quality-check samples for the packaged weights.

Use trigger token `ai51char` in prompts. The useful tested ranges were final weights at LoRA strength `0.4` and checkpoint 600 at strength `0.5`; full strength `1.0` is kept only as evidence because it overfits.

Future LoRA work should start from `aigen lora canon-init` and
`aigen lora dataset-audit`, not from this image pool.
