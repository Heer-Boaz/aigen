# Paired pix2pix pixel-art generator

This is a separate local training route for a narrow, supervised image-to-image
mapping. It does not replace or call the FLUX.2 Klein or Qwen Image Edit
character routes.

The implemented reference family has two raster contracts:

```text
native contract:
  aligned 128x128 or 256x256 source -> same-size U-Net -> RGB target

lossless high-resolution source contract:
  1024x1024 RGB source
    -> PixelUnshuffle(8), preserving every source sample as 192x128x128
    -> U-Net-128
    -> native 3x128x128 RGB prediction
    -> deterministic nearest-neighbour 1024x1024 artifact

conditional discriminator:
  source + real/generated target
    -> native pair, or the same lossless PixelUnshuffle(8)
    -> PatchGAN-16, PatchGAN-70, global PatchGAN-142,
       or two-scale PatchGAN-70
```

U-Net-128 has 41,828,995 trainable parameters; U-Net-256 has 54,413,955.
PatchGAN-16 has 139,585 parameters, PatchGAN-70 has 2,768,705, and the native
128 global PatchGAN-142 has 6,964,033. The latter uses the same N-layer
architecture with one additional stride-2 layer: its 6×6 logits each have a
142×142 theoretical receptive field, larger than the complete 128×128 canvas.
Inference exports and loads only the generator, while its metadata retains the
complete training model configuration.

The width-92 lossless high-resolution generator has 86,703,195 parameters.
Its adapted PatchGAN-70 has 3,155,777, for 89,858,972 trainable parameters
together. `PixelUnshuffle(8)` is a reversible tensor rearrangement, not a
resize: all 1024×1024 source values survive. The generator cannot emit
independent subpixels inside one logical target pixel because it predicts the
128×128 raster before deterministic nearest-neighbour expansion.

The two-scale discriminator follows pix2pixHD: one PatchGAN evaluates the
native 128×128 pair and a second evaluates a 64×64 average-pooled pair. The
coarse discriminator's 70-pixel field is therefore 140 native pixels, while
the native discriminator retains exact pixel-grid detail. Its generator loss
also matches the intermediate discriminator features of real and generated
pairs with weight 10. Real and generated candidates share one discriminator
forward, matching SPADE's BatchNorm contract instead of comparing features
normalized by separate batch statistics.

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
- Every image is RGB and exactly the declared 128×128, 256×256, or 1024×1024
  canvas.
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

Use the smallest learned output raster that represents the intended logical
sprite grid. The ordinary U-Net contract therefore keeps a logical 128×128
target native. The lossless high-resolution contract is different: it retains
the smooth source at 1024×1024, rearranges it losslessly to 128×128 feature
space, predicts exactly one RGB value per logical target pixel, and expands
that prediction only at the model boundary. A genuinely lower-resolution
target such as 32×32 or 64×64 still needs one explicit, consistent
nearest-neighbour mapping to the 128×128 logical target raster.

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
[`docs/prompting.md`](prompting.md), receive a review from
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

The historical native-128 assembly resizes each smooth 768×1024 FLUX source
with Lanczos to 96×128 and places it at x=16 on a white 128×128 canvas. The
native pixel-art target is not changed:

```bash
.venv/bin/aigen pix2pix iro-prepare runs/pix2pix/iro-gate512-v1
.venv/bin/aigen pix2pix audit runs/pix2pix/iro-gate512-v1/dataset
```

Training reads only the resulting audited local dataset. It never calls the
live renderer or FLUX.

That native source mapping discards almost all smooth-source samples. The
lossless high-resolution ablation instead places the original 768×1024 source
at x=128 on a white 1024×1024 canvas without resampling. It expands the native
128×128 target by exactly 8× with nearest-neighbour sampling for the audited
artifact contract; the lossless model still predicts the logical 128×128
target before that deterministic expansion.

### Coverage wave v2

[`configs/pix2pix-iro-coverage-wave512-v2.json`](../configs/pix2pix-iro-coverage-wave512-v2.json)
defines the next target-acquisition wave. It is aligned to the actual
`reviewed-v3-unanimous-job-holdout-v1` training parent, not to the obsolete
lineage splits in the original acquisition plan. All 77 shared job groups
retain their parent train/validation owner; the eight newly supported groups
are train-only. The plan contains 403 train and 109 validation requests across
85 renderer jobs and 158 job/gender body variants. Every configured body
receives three or four requests.

The v2 planner solves one binary mixed-integer program per split. It enforces
the body multiplicities, action quotas, direction quotas, at most one copy of
an action and direction per body, and bounded joint action/direction counts in
one optimization problem. All 80 positive action/direction cells occur in
both train and validation; the simulator's unusable `dead` action has quota
zero. The persisted loader independently checks the same constraints instead
of trusting the solver output.

The previous 512-request plan contributes 502 distinct body-pose cells to an
exclusion manifest. Of those, 457 exist in the current job/action namespace,
and none may recur in the v2 requests. Exclusion provenance records the source
config fingerprint, renderer namespace, fixed renderer defaults, action
catalog, and renderer-job name/species semantics. A colliding numeric job ID
with different catalog semantics is an error rather than an exclusion.

SciPy/HiGHS returns the same request manifest in repeated runs in the pinned
local runtime. The plan also records the NumPy and SciPy versions and hashes
the immutable requests. This is artifact-level reproducibility; it is not a
promise that a future HiGHS version will choose the same optimum from the
config alone.

After rendering, v2 target selection constructs a sparse bipartite graph from
requests to the pixel hashes of the final white-composited RGB targets. One
global minimum-weight full matching chooses all frames together, so an early
request cannot greedily consume the only valid target of a later request.
The selection loader requires exact plan order and proves that all final RGB
target pixels are unique. It also binds each target to the checksums of the
concrete render result and chosen RGBA frame, and records the target matcher's
NumPy/SciPy runtime.

```bash
.venv/bin/aigen pix2pix iro-plan \
  --config configs/pix2pix-iro-coverage-wave512-v2.json \
  --output-dir runs/pix2pix/iro-coverage-wave512-v2 \
  --exclude-coverage-from runs/pix2pix/iro-gate512-v1

.venv/bin/aigen pix2pix iro-render \
  runs/pix2pix/iro-coverage-wave512-v2

.venv/bin/aigen pix2pix iro-select \
  runs/pix2pix/iro-coverage-wave512-v2
```

The materialized 2026-07-30 artifacts passed their own reload boundary. The
request manifest is
`8d34197b9e2319729faecce05093bc093ab2d02fb8d7077da0262f1a56c63a41`;
the selected-record manifest is
`6c45c974a53000482ffc92570f44810731fdbdb956e1d78aedfa5b09f80d670d`.
Selection found a complete assignment with 512 unique final RGB targets and
275 realized rig/frame poses. Neither a body-pose cell nor a final target PNG
is shared with `iro-gate512-v1`.

The exact quotas describe the planned and selected native-target corpus.
Per-image FLUX review may reject reverse sources. The v2 source-set provenance
therefore reports planned, accepted, and deficit counts for every split,
group, body variant, rig family, requested rig pose, action, and direction.
It does not pretend that a filtered training dataset still has exact quotas,
and it does not silently replenish rejected pairs. Any replenishment is a new,
separately planned coverage wave.

The checked-in generic FLUX prompt remains provenance from the old exploratory
route and is not authorized for this wave. Before generation, every source
instruction must follow
[`docs/prompting.md`](prompting.md) and pass an
independent prompt review.

### Reviewed per-image FLUX source sets

The exploratory corpus above has one shared prompt and is retained as an
immutable control. Production paired-data experiments use a separately named
source set with one short, image-specific instruction and one chosen seed per
target. A source set contains:

```text
reviewed-v1/
  source-set.json
  records.jsonl
  prompt-reviews.jsonl
```

`records.jsonl` is in the exact frozen selection order. Every record binds the
selected target checksum to its prompt, author, and seed.
`prompt-reviews.jsonl` binds that prompt and target checksum to a passing
review. Generation rejects a record whose author and reviewer are the same.
The manifest checksums both files, records the target-selection fingerprint,
and records the SHA-256 of
[`docs/prompting.md`](prompting.md). Generation refuses a
missing review, stale prompt hash, changed target, reordered record, or changed
seed.

Generate a new immutable source corpus:

```bash
.venv/bin/aigen pix2pix iro-generate-flux-source-set \
  runs/pix2pix/iro-gate512-v1 \
  --source-set runs/pix2pix/iro-gate512-v1/prompt-sets/reviewed-v1/source-set.json
```

The source plan freezes the complete normalized source set, source-set
manifest hash, model and runtime artifacts, 4-step scheduler/sampler contract,
canvas, reference raster, and shard size. All missing prompts are encoded in
one conditioner session; all missing shards share one FLUX model session.
Each shard is generated and validated in a sibling temporary directory before
an atomic rename. Completed shards and completed corpora are immutable.

Generated images are still candidate supervision, not accepted supervision.
Before assembly, the reviewer writes
`flux-source-set-reviewed-v1/output-audit.json` plus its checksummed JSONL
records. There is exactly one audit record for every frozen target. It binds
the generated image checksum, target checksum, prompt checksum, chosen seed,
all seeds actually tested, verdict, reviewer, and review notes. Pose, body
direction, silhouette, dominant design and color blocking, held objects, and
subject count are assessed visually; a plausible illustration with the wrong
front/back view or a materially different outfit is rejected.

Only records with a `pass` verdict enter the paired dataset:

```bash
.venv/bin/aigen pix2pix iro-prepare-flux-source-set \
  runs/pix2pix/iro-gate512-v1 \
  --name reviewed-v1

.venv/bin/aigen pix2pix audit \
  runs/pix2pix/iro-gate512-v1/dataset-flux-source-set-reviewed-v1
```

The default assembly performs the historical Lanczos mapping. A curated
audited subset can instead be derived into the lossless 1024 contract while
binding both the source-set audit and the exact pair-filter fingerprint:

```bash
.venv/bin/aigen pix2pix iro-prepare-flux-source-set \
  runs/pix2pix/iro-gate512-v1 \
  --name reviewed-v1 \
  --pair-filter \
    runs/pix2pix/iro-gate512-v1/dataset-flux-source-set-reviewed-v3-unanimous \
  --training-raster lossless1024
```

This produces
`dataset-flux-source-set-reviewed-v3-unanimous-lossless1024-v1`. Its rebuild
was byte-identical for every source PNG, target PNG, and pair record to the
first diagnostic 1024 dataset. Rejected candidates remain in the immutable
source corpus for provenance but cannot silently enter training. Corrections
require a new named source set or pair filter.

### Qwen reverse-source control

[`configs/pix2pix-iro-qwen2511-lightning-source.json`](../configs/pix2pix-iro-qwen2511-lightning-source.json)
defines a separate conservative source-generation control over the same frozen
iRO target selection. It uses the explicit Qwen-Image-Edit-2511 FP8 Lightning
8-step route and its native 1328×1328 canvas. It does not replace or silently
fall back from the FLUX source route.

As with the FLUX config, its checked-in prompt records an exploratory run and
is not a reusable prompt template. Prompt revisions require a new config and
source output root under the rules in
[`docs/prompting.md`](prompting.md).

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

Despite its historical filename, `pix2pix-native128-l1-control5k.json` still
constructs and trains PatchGAN; it is an adversarial run with an extreme
`lambda_l1=10000`. The matched 10,000-step objective controls are:

- [`configs/pix2pix-native128-adversarial-l1-control10k.json`](../configs/pix2pix-native128-adversarial-l1-control10k.json)
- [`configs/pix2pix-native128-l1-only-control10k.json`](../configs/pix2pix-native128-l1-only-control10k.json)

Both initialize and update the complete 41,828,995-parameter generator from
scratch with the same data, initialization, sampler, augmentation, optimizer,
dropout RNG, and `100 × L1` term. `adversarial_l1` additionally constructs and
trains PatchGAN. `l1_only` never constructs a discriminator and its checkpoint
contains neither discriminator weights nor discriminator optimizer state.

The explicit full-parameter scale controls are the 250-step VRAM/checkpoint
smoke
[`configs/pix2pix-native128-2b-patch70-control250.json`](../configs/pix2pix-native128-2b-patch70-control250.json)
and the bounded 2,000-step capacity run
[`configs/pix2pix-native128-2b-patch70-control2k.json`](../configs/pix2pix-native128-2b-patch70-control2k.json).
It widens only the U-Net generator to 448 base channels: 2,048,905,603
generator parameters plus the unchanged 2,768,705-parameter PatchGAN. Every
generator parameter is initialized from scratch and updated. This is not LoRA,
transfer learning, or a quantized checkpoint. BF16 parameter storage and the
paged Adam optimizer are explicit memory formats that make full-parameter
training possible on the 16 GB target. Adam's large state tensors use
block-wise 8-bit storage; tensors smaller than 4,096 elements remain FP32.
Gradients still update the complete network.

The full 200-epoch reviewed-data experiment is
[`configs/pix2pix-native128-2b-patch70-reviewed-v3-unanimous-200e.json`](../configs/pix2pix-native128-2b-patch70-reviewed-v3-unanimous-200e.json).
Its audited training split contains 190 pairs, so 38,000 updates are exactly
200 epochs. The initial learning rate is used while `completed_steps <= 19000`;
later updates linearly decay against the remaining 19,000-step span.
Recoverable checkpoints are requested after epochs 2, 5, and 10, followed by a
regular 20-epoch interval. Those dataset-sized values are an explicit
experiment contract, not a generic default for another corpus.

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
it is not selected automatically. PatchGAN-142 is the four-layer
global-structure discriminator for the native 128 canvas. It preserves the
same conditional N-layer construction and gives interior logits a theoretical
field wider than the canvas. Padding and shifted edge logits mean this is not a
guarantee that every decision observes the complete subject.

For a white-canvas corpus, training v5 can balance reconstruction by computing
the foreground and exact-white background means separately and averaging the
two. This preserves the penalty for false background ink while preventing a
small subject from contributing only its raw pixel-area fraction to L1. The
training boundary rejects a configured target that lacks either region.

The reviewed-data multiscale experiment is
[`configs/pix2pix-native128-89m-multiscale-balanced-fm10-reviewed-v3-unanimous-200e.json`](../configs/pix2pix-native128-89m-multiscale-balanced-fm10-reviewed-v3-unanimous-200e.json).
It changes the discriminator/objective only: the generator, data, seed,
optimizer, schedule, epoch count, and checkpoint cadence remain identical to
the 89M PatchGAN-70 control. That v5 run is retained as an ablation: its
feature-matching passes used separate BatchNorm statistics and its adversarial
maps were still uniformly averaged.

The corrected v6 experiment is
[`configs/pix2pix-native128-89m-multiscale-balanced-all-fm10-translate-reviewed-v3-unanimous-200e.json`](../configs/pix2pix-native128-89m-multiscale-balanced-all-fm10-translate-reviewed-v3-unanimous-200e.json).
It area-pools the exact-non-white target mask to every discriminator logit and
intermediate feature map. Foreground and background means each contribute 50%
to discriminator real/fake loss, generator adversarial loss, and feature
matching. It also applies one shared integer translation in the range
`[-16, 16]` to source, real target, and generated candidate before the
discriminator. Translation uses exact pixels and white padding; it never
touches generator input, reconstruction loss, validation, or inference.

The completed v6 run still produced severe foreground speckling. A held-target
conditionality audit then compared `D(source, target)` with
`D(shuffled_source, target)` in one discriminator forward. Both training and
validation data showed anti-conditioning: the native-scale balanced logit
preferred the wrong source by about `1.5–1.7`, and the coarse scale preferred
it by about `9.4–9.9`. Only `10–16%` of source swaps preferred the aligned
pair. The discriminator had therefore learned that a mismatched real pair was
more real, not merely failed to observe the source.

The v7 [a-contrario cGAN](https://arxiv.org/abs/2106.15011) experiment is
[`configs/pix2pix-native128-89m-multiscale-a-contrario-all-fm10-translate-reviewed-v3-unanimous-200e.json`](../configs/pix2pix-native128-89m-multiscale-a-contrario-all-fm10-translate-reviewed-v3-unanimous-200e.json).
It leaves the generator and generator objective unchanged. The discriminator
receives four equally weighted modalities in one BatchNorm context:

- aligned source plus real target, labelled real;
- aligned source plus generated target, labelled fake;
- mismatched source plus real target, labelled fake;
- mismatched source plus generated target, labelled fake.

Foreground and background still contribute equally inside every modality.
Each epoch constructs a group-disjoint derangement of the training split:
every source is used exactly once as a mismatched condition, never for itself
or another pair from the same sprite group. The derangement is derived from
the epoch sampler seed, so checkpoint resume preserves the exact stream.

The completed v7 run reached 38,000 steps in 845 seconds with 5,025 MB peak
VRAM. Its final region-balanced validation L1 was `0.10836`, only slightly
below v6's `0.10886`, and the foreground still contained severe speckling.
The conditionality failure itself was fixed. On the held-out, single-group
`taekwon` split, the native and coarse discriminators preferred the aligned
source in `99.6%` and `94.5%` of within-group source swaps, with mean balanced
logit margins of `+20.5` and `+38.5`. V6 had preferred the aligned source in
only about `17%` and `11%` of the same audit. The remaining output failure is
therefore downstream of source conditioning.

The isolated v8 target-palette proximity experiment is
[`configs/pix2pix-native128-89m-multiscale-a-contrario-palette100-all-fm10-translate-reviewed-v3-unanimous-200e.json`](../configs/pix2pix-native128-89m-multiscale-a-contrario-palette100-all-fm10-translate-reviewed-v3-unanimous-200e.json).
It preserves the complete v7 generator, discriminator, data order, optimizer,
schedule, and losses. For each paired target, the loader extracts and caches
its exact RGB palette once. A generator pixel receives its mean squared RGB
distance to the nearest color in that target palette; target foreground and
background means each contribute 50%, and the result enters the generator
objective at weight `100`. The implementation adapts the nearest-palette loss
from the pinned
[`multi-domain` reference](https://github.com/fegemo/multi-domain/blob/01639a2795467b65b7f77d98e441fd54fe58880d/utility/palette_utils.py#L108-L135).
V8 requires FP32 parameters and compute. Distance evaluation uses a direct
FP32 difference outside autocast and avoids matrix multiplication, so autocast
and TF32 cannot erase small color differences inside the objective.

V8 deliberately adds no palette coverage term, quantized forward pass,
palette-index classifier, discriminator change, or inference-time palette.
It tests only whether target-palette membership pressure reduces continuous
RGB noise. It does not mathematically guarantee a discrete number of output
colors.

The completed v8 run rejected that isolated hypothesis. Across all 34
validation pairs, foreground target-palette proximity improved by only `1.19%`
relative to v7, while mean RGB8 colors per output increased from `2512.15` to
`2514.44`. Exact foreground target-palette membership moved only from
`5.1439%` to `5.2298%`; overall membership and foreground L1 became slightly
worse. Perfect post-hoc snapping to each validation target's exact palette
also left the spatial fragmentation visible. Palette drift is measurable, but
it is not the cause of the misplaced local structure.

The canonical 89M L1-only control is
[`configs/pix2pix-native128-89m-l1-only-reviewed-v3-unanimous-200e.json`](../configs/pix2pix-native128-89m-l1-only-reviewed-v3-unanimous-200e.json).
It differs from the completed 89M PatchGAN-70 config in exactly one field:
`objective` changes from `adversarial_l1` to `l1_only`. The generator, initial
weights, data order, optimizer, learning-rate schedule, horizontal flips,
precision, and 38,000 updates are identical.

That control isolated the dominant failure. The final generator reconstructs
the 190 training pairs cleanly at global L1 `0.00718` and foreground L1
`0.04938`, but reaches only `0.03828` and `0.21117` on the group-disjoint
validation split. Validation output is fragmented even without a
discriminator, feature matching, region balancing, palette loss, or
discriminator augmentation. Running BatchNorm from per-image statistics or
enabling decoder dropout at inference does not remove the artifact, and a
full-validation Fourier audit found no dominant stride-2 or stride-4
checkerboard signature.

The checkpoint curve shows ordinary severe overfitting rather than an
incapable decoder. Region-balanced validation L1 is best at step 3,800
(`0.10380`, 20 epochs) and worsens to `0.11055` by step 38,000, while training
error continues falling to `0.02481`. The current split holds out whole job
families: seven families and 190 pairs train the model, all 34 validation
pairs are `taekwon`, and the test split contains only `gunslinger`, `ninja`,
and `summoner`. The clean training reconstructions prove that the U-Net has
enough output capacity; this 256-pair corpus does not teach the complex
foreground transformation well enough to generalize to unseen families.
Simple, repeated structures such as the non-white ground shadow do generalize.

The follow-up job-holdout diagnosis removes the single-lineage validation
confound. Its split joins all 256 reviewed pairs to the frozen structured iRO
requests, groups both genders and all variants by `(lineage, job_id)`, and
uses a family-stratified SHA-256 rank to hold out complete jobs. It contains
202 training pairs from 60 jobs and 54 validation pairs from 17 other jobs.
All eleven lineages occur on both sides and no job crosses the split. The
matched config is
[`configs/pix2pix-native128-89m-l1-only-job-holdout-v1-200e.json`](../configs/pix2pix-native128-89m-l1-only-job-holdout-v1-200e.json).

That run reproduces the same failure. At epoch 200, training reaches global L1
`0.00705`, foreground L1 `0.04896`, and visually clean near-copies. Held-job
validation remains fragmented at global L1 `0.03933` and foreground L1
`0.21709`. Validation again peaks at epoch 20: global L1 `0.03577` and
region-balanced L1 `0.10575`; epoch 200 is respectively `10.0%` and `6.6%`
worse. Whole-lineage OOD evaluation made the original test stricter, but did
not cause the failure. The gate needs substantially more independent
body/outfit/object groups and clean aligned pairs; additional epochs or a
wider generator would only increase memorization on the present corpus.

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
  discriminator.safetensors  # adversarial_l1 only
  training-state.pt
previews/step-N.png
final/
  model.json
  generator.safetensors
```

Each preview row is `source | target | generated`. Validation records global
L1, foreground L1, exact-white-background L1, their equal-region mean, MSE, and
PSNR over the complete held-out split. Those metrics measure paired pixel
reconstruction; the preview remains the evidence for pose, accessory,
silhouette, and pixel-design quality.

The baseline keeps FP32 parameters and computation and uses the GPU's recorded
TF32 acceleration for FP32 convolution and matrix multiplication.
Training v3 records `adversarial_l1` or `l1_only` as an explicit objective.
Training v4 additionally records an exact constant or linear-decay
learning-rate schedule and any extra checkpoint steps. The learning rate for
each update is derived from its completed-step position and applied to every
optimizer, so resume does not depend on the learning rate serialized inside an
optimizer state. V2 and v3 configurations retain their historical constant
learning rate. Training v5 additionally records discriminator scale count,
feature-matching weight, and the reconstruction-region balance. Training v6
additionally records the adversarial-region balance and discriminator
augmentation policy. Training v7 additionally records the conditional-negative
policy and requires the complete a-contrario discriminator objective. Training
v8 additionally records a positive target-palette proximity weight.
Checkpoint v3 has an objective-specific exact file/state contract; v2
checkpoints are explicitly interpreted as the historical adversarial-L1
contract and remain resumable only by an adversarial-L1 configuration.
`parameter_precision`, compute `precision`, and `optimizer` are independent,
explicit training contracts. BF16 parameter storage requires BF16 compute.
`paged_adam8bit` requires CUDA and quantizes only Adam's large optimizer-state
tensors; it does not freeze or quantize the model parameters. Checkpoint
restore recreates paged buffers before copying optimizer state, so resumed
training preserves the same memory contract.

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
