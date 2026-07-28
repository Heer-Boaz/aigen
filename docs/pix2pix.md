# Paired pix2pix pixel-art generator

This is a separate local training route for a narrow, supervised image-to-image
mapping. It does not replace or call the FLUX.2 Klein or Qwen Image Edit
character routes.

The implemented reference family is:

```text
aligned 128x128 or 256x256 RGB source
             |
             v
       U-Net-128 or U-Net-256
             |
             v
generated RGB target on the same canvas
             |
             +---- source + generated ----+
             |                             v
             +---- source + real ------> PatchGAN-16 or PatchGAN-70
```

U-Net-128 has 41,828,995 trainable parameters; U-Net-256 has 54,413,955.
PatchGAN-16 has 139,585 parameters and PatchGAN-70 has 2,768,705. Inference
exports and loads only the generator, while its metadata retains the complete
training model configuration.

## What is known about PixelMe

Google Cloud's 2020 interview with the PixelMe author explicitly says that the
service used pix2pix technology, trained with pixelated images, used TensorFlow
and Colab for training, and deployed a container on Cloud Run behind Firebase:

- <https://cloud.google.com/blog/products/ai-machine-learning/using-google-cloud-platform-free-tier-to-scale-out-an-ai-service>

That is evidence for pix2pix, not evidence that PixelMe copied every layer,
training step, crop rule, or hyperparameter from a tutorial. Its original
training pairs, exact detector, checkpoint, dataset size, and complete
configuration are not public. There is no corresponding evidence that its
conversion model was CycleGAN.

This implementation therefore records its own provenance and never labels its
configuration as PixelMe's private configuration. The two code references are
pinned in every `run.json`:

- TensorFlow tutorial at revision
  `cf2a57e77485c371f04cc486d9d1e632ef552739`
- `junyanz/pytorch-CycleGAN-and-pix2pix` at revision
  `2a7afba2895d52556dd5dfe07e8555ef657ced6f`

PyTorch is used because the repository already pins a Blackwell-compatible
PyTorch/CUDA runtime. Adding TensorFlow would duplicate the GPU runtime without
changing the pix2pix objective.

## Dataset contract

Print the machine-readable contract:

```bash
.venv/bin/aigen pix2pix contract
```

A dataset directory contains one `dataset.json`, one JSONL pair manifest, and
the referenced images:

```text
my-dataset/
  dataset.json
  pairs.jsonl
  source/
    frame-0001.png
  target/
    frame-0001.png
```

`dataset.json`:

```json
{
  "format": "aigen.pix2pix.paired.v2",
  "name": "my-paired-pixel-art-data",
  "image_size": 128,
  "pairs": "pairs.jsonl"
}
```

Each `pairs.jsonl` line is one complete record:

```json
{"id":"frame-0001","group":"subject-0001","split":"train","source":"source/frame-0001.png","target":"target/frame-0001.png"}
{"id":"frame-0101","group":"subject-0101","split":"validation","source":"source/frame-0101.png","target":"target/frame-0101.png"}
```

The contract is intentionally strict:

- Every source and target is decoded completely during audit.
- Every image is RGB and exactly the declared 128×128 or 256×256 canvas.
- Pair IDs are unique and safe to use as prediction filenames.
- Every pair names its original identity/source-sequence group.
- A group occurs in exactly one split; the audit rejects cross-split leakage.
- Paths are relative and cannot escape the dataset directory.
- Both `train` and `validation` contain at least one pair. `test` is optional.
- Source and target already have the intended matching composition.
- The audit hashes every referenced image and fingerprints group and split
  ownership together with the pair contents.

There is no detector, cropper, alpha compositor, automatic alignment, or
fallback resize hidden in training. Those operations would define the task and
must therefore be decided while producing the paired data.

Use the smallest supported training canvas that represents the intended
logical sprite grid. A logical 128×128 target belongs on the native 128×128
canvas; do not nearest-neighbour-scale it to 256×256 and ask the network to
reproduce every logical pixel four times. A genuinely lower-resolution target
such as 32×32 or 64×64 still needs one explicit, consistent nearest-neighbour
mapping to the 128×128 training canvas. Display scaling happens after inference,
not inside the learned task.

When constructing data:

- Put the complete subject, pose, held objects, loose accessories, and intended
  background treatment in both halves of each pair.
- Split by original subject or source sequence, not by random near-duplicate
  frames. Otherwise validation measures memorization leakage.
- Preserve a single pixel-grid convention within a dataset.
- Treat target design quality as the main supervision signal. The network
  learns mistakes and inconsistencies just as directly as good sprite design.

Audit before training:

```bash
.venv/bin/aigen pix2pix audit path/to/my-dataset
```

Training runs the same complete audit again and refuses invalid or incomplete
data.

## Reproducible native-128 iRO gate corpus

[`configs/pix2pix-iro-gate512.json`](../configs/pix2pix-iro-gate512.json)
defines the first scaled gate without deriving its size or contents from an
arbitrary image count. Its checked-in catalog mirrors the 79 normal jobs exposed
by the iRO Character Sprite Simulator, including 146 valid job/gender bodies.
Seasonal bodies, mounts, jRO outfits, headgear, garments, and body-palette
variants are excluded from this first coherent target domain.

The exact accepted-pair quotas are:

| split | pairs | lineages |
|---|---:|---|
| train | 384 | Novice, Swordman, Magician, Archer, Acolyte, Merchant, Thief |
| validation | 64 | Taekwon |
| test | 64 | Ninja, Gunslinger, Summoner |

Within every split the manifest fixes a 50/50 gender balance, equal counts for
all eight body directions and all eight hair palettes, and a largest-remainder
balance over all eleven simulator action bases. Human hair-style IDs are also
disjoint between the three splits. Both the broad lineage group and the
species/gender/head identity are assigned before any network or renderer work;
all downstream artifacts inherit that assignment.

The target acquisition contract is:

```text
immutable quota plan
  -> POST exact simulator payload
  -> cache original PNG/APNG bytes
  -> Pillow seek/load composited animation frames
  -> domain-separated RGBA pixel hash
  -> reject top/left/right canvas contact
  -> permit only the shared bottom ground baseline
  -> select one stable-ranked globally unique frame per request
  -> composite native 128x128 target on white
```

Pillow's APNG loader already returns full logical-screen composites after blend
and disposal. When an APNG has a separate default image, selection begins at
frame 1 because that default is not part of the animation. The corpus code does
not reconstruct APNG disposal itself.

Plan and acquire targets:

```bash
.venv/bin/aigen pix2pix iro-plan \
  --config configs/pix2pix-iro-gate512.json \
  --output-dir runs/pix2pix/iro-gate512-v1

.venv/bin/aigen pix2pix iro-render runs/pix2pix/iro-gate512-v1
.venv/bin/aigen pix2pix iro-select runs/pix2pix/iro-gate512-v1
```

The exact exploratory FLUX.2 Klein prompt and 768×1024 canvas are part of the
immutable config fingerprint. That fingerprint preserves experimental
provenance; it does not make the prompt a recommended template. Every new
source-corpus prompt must follow
[`docs/image-edit-prompting.md`](image-edit-prompting.md), receive a review from
an agent that read that guide, and use a new config and output root when its
wording changes. The selected native 128×128 target is passed to FLUX unchanged.
A same-seed proof on representative front and back poses compared native input
with nearest-neighbour 512×512 and 768×768 references: the larger references
added invented detail without resolving ambiguous body direction, while the
native input retained the closest overall pose and silhouette. Reference
pre-upscaling is therefore explicitly absent from the source contract.

Before any GPU work, `flux-v3/source-plan.json` binds the target selection,
prompt, raster and sampler contract, corpus-generator revision, FLUX backend
revision, SHA-256 of every transformer/VAE/scheduler/conditioner artifact, and
the installed Python, torch, diffusers, transformers, tokenizers, Accelerate,
FLUX, Einops, Comfy Kitchen, SafeTensors, NumPy, and Pillow runtime revisions.
Reverse sources are generated
by one serial GPU owner in batches of 16. Every batch is written to a unique
sibling `.incomplete` directory, decoded and checksummed, given a completion
manifest, and only then atomically renamed into `flux-v3/shards`. An existing
shard is reused only after its source-plan binding, inventory, seeds, raster
contract, sizes, and SHA-256 hashes all verify. Once
`flux-v3/result.json` exists, the completed source corpus is immutable; a
missing or changed shard fails instead of being regenerated:

```bash
.venv/bin/aigen pix2pix iro-generate-sources \
  runs/pix2pix/iro-gate512-v1
```

Assembly performs the only learned-input resampling: each smooth 768×1024 FLUX
source is Lanczos-resized to 96×128 and placed at x=16 on a white native-128
canvas. The pixel-art target is never resized, cropped, or recentered:

```bash
.venv/bin/aigen pix2pix iro-prepare runs/pix2pix/iro-gate512-v1
.venv/bin/aigen pix2pix audit runs/pix2pix/iro-gate512-v1/dataset
```

Training reads only the resulting audited local dataset. It never calls the
live renderer or FLUX.

### Qwen reverse-source control

[`configs/pix2pix-iro-qwen2511-lightning-source.json`](../configs/pix2pix-iro-qwen2511-lightning-source.json)
defines a separate conservative source-generation control over the same frozen
iRO target selection. It uses the explicit Qwen-Image-Edit-2511 FP8 Lightning
8-step route and its native 1328×1328 canvas. It does not replace or silently
fall back from the FLUX source route.

As with the FLUX config, its checked-in prompt records an exploratory run and
is not a reusable prompt template. Prompt revisions require a new config and
source output root under the rules in
[`docs/image-edit-prompting.md`](image-edit-prompting.md).

```bash
.venv/bin/aigen pix2pix iro-generate-qwen-sources \
  runs/pix2pix/iro-gate512-v1 \
  --config configs/pix2pix-iro-qwen2511-lightning-source.json

.venv/bin/aigen pix2pix iro-prepare-qwen \
  runs/pix2pix/iro-gate512-v1
```

The Qwen source plan independently fingerprints its prompt, sampler contract,
LightX2V and model revisions, exact model files, and the LightX2V Python
runtime. Shards publish atomically and the assembled dataset is audited through
the same paired-data contract as the FLUX variant.

## Training

The 256 control is
[`configs/pix2pix-baseline.json`](../configs/pix2pix-baseline.json). The native
128 experiments are:

- [`configs/pix2pix-native128-patch70.json`](../configs/pix2pix-native128-patch70.json)
- [`configs/pix2pix-native128-patch16.json`](../configs/pix2pix-native128-patch16.json)

Four 5,000-step diagnostic controls preserve the exact short-run settings used
to compare discriminator field size and L1 dominance. They are not substitutes
for the 40,000-step starting profiles:

- [`configs/pix2pix-native128-patch70-control5k.json`](../configs/pix2pix-native128-patch70-control5k.json)
- [`configs/pix2pix-native128-patch16-control5k.json`](../configs/pix2pix-native128-patch16-control5k.json)
- [`configs/pix2pix-native128-lambda1000-control5k.json`](../configs/pix2pix-native128-lambda1000-control5k.json)
- [`configs/pix2pix-native128-l1-control5k.json`](../configs/pix2pix-native128-l1-control5k.json)

The baseline and two 40,000-step profiles retain the canonical pix2pix
optimizer and loss settings:

- U-Net with batch normalization and decoder dropout
- vanilla adversarial loss plus `100 × L1`
- Adam at `0.0002`, `beta1=0.5`, `beta2=0.999`
- batch size 1, 40,000 optimizer steps, FP32
- deterministic epoch shuffling and paired horizontal flips
- metrics every 100 steps and checkpoints every 5,000 steps

PatchGAN-70 is the canonical three-layer discriminator. PatchGAN-16 is the
explicit one-layer local discriminator ablation for a native 128 sprite grid;
it is not selected automatically.

The 40,000-step limit is this repository's reproducible starting profile, not a
claim about PixelMe's private training duration. At FP32 width, the exported
U-Net-256 generator is about 218 MB. U-Net-128 is about 167 MB.

The old tutorial's 286→256 resize/crop jitter is deliberately not part of this
pixel-art baseline: resampling a target can shift or unevenly scale its learned
pixel grid. The shared horizontal flip preserves exact pair alignment.

Start a new run in an empty destination:

```bash
.venv/bin/aigen pix2pix train path/to/my-dataset \
  --config configs/pix2pix-native128-patch70.json \
  --output-dir runs/pix2pix/my-run
```

Resume the exact dataset and exact configuration from one of that run's
checkpoints:

```bash
.venv/bin/aigen pix2pix train path/to/my-dataset \
  --config configs/pix2pix-native128-patch70.json \
  --output-dir runs/pix2pix/my-run \
  --resume runs/pix2pix/my-run/checkpoints/step-00005000
```

The resume contract verifies dataset, configuration, model files, hashes,
optimizer states, step position, and random-number state. Checkpoints are never
rotated or deleted automatically.

A completed run contains:

```text
run.json
metrics.jsonl
checkpoints/step-N/
  checkpoint.json
  generator.safetensors
  discriminator.safetensors
  training-state.pt
previews/step-N.png
final/
  model.json
  generator.safetensors
```

Each preview row is `source | target | generated`. Validation records mean L1,
MSE, and PSNR over the complete held-out split. Those metrics measure paired
pixel reconstruction; the preview remains the evidence for pose, accessory,
silhouette, and pixel-design quality.

The baseline keeps FP32 tensors and uses the GPU's recorded TF32 acceleration
for FP32 convolution and matrix multiplication. `bf16` autocast is supported as
an explicit configuration change for CUDA training, but it is not silently
enabled.

## Inference and evaluation

Inference requires the same already-prepared RGB canvas size recorded by the
model:

```bash
.venv/bin/aigen pix2pix infer \
  --model-dir runs/pix2pix/my-run/final \
  --input path/to/prepared-source.png \
  --output runs/pix2pix/result.png
```

Export a native 64×64 result with nearest-neighbour sampling:

```bash
.venv/bin/aigen pix2pix infer \
  --model-dir runs/pix2pix/my-run/final \
  --input path/to/prepared-source.png \
  --output runs/pix2pix/result-64.png \
  --output-size 64
```

Write every prediction plus aggregate metrics and a comparison sheet:

```bash
.venv/bin/aigen pix2pix evaluate \
  --model-dir runs/pix2pix/my-run/final \
  --dataset path/to/my-dataset \
  --split validation \
  --output-dir runs/pix2pix/my-run-evaluation
```

The model bundle and every checkpoint use SafeTensors for network weights.
Metadata includes SHA-256 hashes and loading is strict; incompatible or damaged
artifacts fail instead of falling back to another model.
