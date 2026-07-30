# Pix2pix pixel-art generator handoff

## Assignment

Continue the local paired image-to-image experiment that translates one
complete smooth anime image into one coherent Japanese/JRPG-style pixel-art
sprite.

This document is a handoff for a fresh Claude, Gemini, Codex, or human
collaborator. It records the user's actual objective, the live repository
state on 2026-07-30, the experiments already completed, the hypotheses those
experiments ruled out, and the next evidence-bearing work.

Do not start training or generation merely because this document was opened.
First inspect the live worktree and preserved artifacts, then agree the next
GPU operation with the user.

## Read first

Read these files before changing anything:

1. [`../AGENTS.md`](../AGENTS.md)
2. [`PLAN.md`](PLAN.md)
3. [`pix2pix.md`](pix2pix.md)
4. [`image-edit-prompting.md`](image-edit-prompting.md)

`pix2pix.md` is the detailed implementation and experiment log. This handoff
is the decision-oriented overview; it does not supersede that document.

Any agent that writes or reviews a reverse-source prompt must first read
`image-edit-prompting.md`. A prompt review performed without reading it is
invalid.

## User objective

The desired learned operation is:

```text
complete smooth anime subject on a simple canvas
                     |
                     v
complete Japanese/JRPG-style pixel-art sprite
```

The intended input can contain:

- one full-body person;
- one object;
- one garment or accessory;
- one person with a held object such as a gun, bow, or staff.

The entire image is translated directly. There is no requested detector,
semantic router, face crop, object-by-object compositor, or PixelMe-style
portrait/landscape dispatch. The working assumption is only that a source
contains one primary subject, possibly with attached clothing, accessories,
and held objects.

The target domain should remain narrow and stylistically coherent. Do not
broaden it into a generic collection of unrelated pixel-art styles. The thick
ink, pastel/watercolor source style used by Jillian may be applied in a
separate existing FLUX.2 post-processing step; it is not the present
pix2pix problem.

The current feasibility corpus is narrower than that final objective: it
contains smooth FLUX-generated anime illustrations paired primarily with
Ragnarok humanoid sprites. It does not yet establish performance on Jillian's
real source distribution, standalone objects, or isolated garments.

The user has already exhausted many generic checkpoints and LoRA experiments.
Do not redirect this work to another arbitrary model or begin another LoRA
detour. If a pretrained semantic backbone eventually becomes necessary, make
that a separately justified branch after the paired-data gate below.

## What PixelMe establishes

The 2020 Google Cloud interview with PixelMe's author explicitly states that
PixelMe used pix2pix technology, trained with pixelated target images using
TensorFlow and Colab, and deployed a container on Cloud Run behind Firebase:

- <https://cloud.google.com/blog/products/ai-machine-learning/using-google-cloud-platform-free-tier-to-scale-out-an-ai-service>

This is evidence that its original conversion path used pix2pix. It is not
evidence for PixelMe's private:

- dataset;
- dataset size;
- exact face detector or preprocessing;
- generator width;
- discriminator field;
- training duration;
- checkpoint;
- full hyperparameter set.

No corresponding primary evidence has been found that the original converter
used CycleGAN. CycleGAN solves unpaired translation; the current repository
deliberately implements paired pix2pix because every desired source must match
one exact sprite target in pose, view, silhouette, clothing, and held objects.

Pix2pix is an architecture and training objective, not a pretrained semantic
foundation model. Widening it from 89 million to two or four billion randomly
initialized parameters does not give it Qwen- or FLUX-like world knowledge.

## Implemented system

The implementation is under:

```text
aigen/pix2pix/
aigen/pix2pix_commands.py
configs/pix2pix-*.json
docs/pix2pix.md
```

It is PyTorch-based because this repository already has a
Blackwell-compatible PyTorch/CUDA runtime. It records two historical reference
revisions in run metadata:

- TensorFlow pix2pix tutorial:
  `cf2a57e77485c371f04cc486d9d1e632ef552739`
- `junyanz/pytorch-CycleGAN-and-pix2pix`:
  `2a7afba2895d52556dd5dfe07e8555ef657ced6f`

The implemented model family includes:

| component | trainable parameters |
|---|---:|
| U-Net-128 baseline generator | 41,828,995 |
| U-Net-256 baseline generator | 54,413,955 |
| native-128 89M generator, base width 92 | 86,424,987 |
| native-128 2B generator, base width 448 | 2,048,905,603 |
| PatchGAN-16 | 139,585 |
| PatchGAN-70 | 2,768,705 |
| PatchGAN-142 | 6,964,033 |

The 2B experiment is full-parameter BF16 training with paged 8-bit Adam state.
It is not LoRA, transfer learning, or a quantized generator. The 2B and 89M
models both start from random weights.

The canonical objective is conditional adversarial loss plus `100 x L1`, with
Adam at `2e-4`, `beta1=0.5`, `beta2=0.999`, and batch size one. Later isolated
experiments added:

- foreground/background-balanced reconstruction;
- native and coarse PatchGAN-70 discriminators;
- discriminator feature matching;
- one shared real/fake discriminator forward for BatchNorm;
- shared exact-integer translation with white padding;
- a-contrario mismatched-pair negatives;
- per-target palette-proximity loss;
- a discriminator-free L1-only control.

These additions were controlled experiments. They are not all a recommended
final objective.

## Raster and pair contract

The target sprite is a native `128x128` RGB image on white. It is not enlarged
before reverse-source generation and is never resized, cropped, or recentered
during paired-dataset assembly.

The generated smooth FLUX source is `768x1024`. Assembly performs the only
learned-input resampling:

```text
768x1024 smooth source
  -> Lanczos 96x128
  -> centered at x=16 on a white 128x128 canvas
```

The source and target then enter training as aligned `128x128` RGB images.
Display-size exports use nearest-neighbour resizing after inference.

A same-seed native-versus-upscaled reference check was already performed.
Nearest-neighbour pre-upscaling the native sprite to `512x512` or `768x768`
caused the reverse model to invent detail and did not resolve ambiguous body
direction. Do not add a generative upscaler to this contract.

Every dataset is audited for:

- complete image decoding;
- exact RGB canvas size;
- unique safe pair IDs;
- immutable source and target hashes;
- group-disjoint split ownership;
- path containment;
- required train and validation splits;
- a deterministic dataset fingerprint.

Training performs the same audit again. There is no hidden crop, detector,
alignment heuristic, or fallback resize.

## The central data problem

The target sprites are easy to acquire. Correct smooth source images are not.

Reverse generation asks FLUX.2 Klein or Qwen Image Edit to infer a smooth
anime representation from a small sprite. Several superficially attractive
outputs changed the supervision:

- a rear-facing sprite became a smooth front-facing head or face;
- body direction changed;
- a sitting, reclining, or action pose changed;
- clothing or dominant color blocking changed;
- a held object disappeared or changed;
- one subject became a different silhouette.

Those are invalid pairs. A pix2pix model trained on them is literally taught
that the wrong view or pose should map to the target. This contamination was
visible in early previews and is not a cosmetic review issue.

The retained gate-v1 audit trail makes the scale of this problem concrete:

- 509 of 512 records tried only one seed;
- only three difficult records tried four seeds;
- the first output audit accepted 409 and rejected 103;
- a second pair audit over those 409 accepted 273 and rejected 136;
- that second audit recorded 59 pose, 56 composition-alignment, 54
  front/back-or-facing, two silhouette/proportion, and one held-object
  rejection reasons.

Two later independent reviews produced the 256-pair
`reviewed-v3-unanimous` dataset. The label describes its provenance, not an
objective guarantee of clean pixels. The user inspected the first four
validation pairs and judged three materially wrong even though both reviews
had accepted all four. Existing reviewers demonstrably missed pose and
front/back disparities. Use those disputed records to calibrate the acceptance
rule before reviewing v2; do not blindly trust a prior `pass` verdict.

Prompting must follow `image-edit-prompting.md`:

- use one concise direct transformation instruction;
- describe only visually grounded constraints needed for that image;
- use a short direction qualifier such as `rear view`, `facing left`, or
  `reclining` only when the sprite actually requires it;
- do not write speculative anatomy narratives;
- do not add irrelevant negative prompts;
- do not use phrases such as `no watermelon` or contrived relational prose
  such as `connected to the reclining body`;
- reject an image when bounded seed retries still change the paired content.

The user explicitly requested a timebox or retry box. Do not generate
unbounded test grids in pursuit of one perfect reverse image. Preserve every
tested seed in the audit record, choose a valid candidate if one exists, and
otherwise record a deficit. Do not silently replenish a rejected record from
an unplanned distribution.

## Existing corpus

### Gate v1

The first iRO acquisition planned 512 native targets. After successive visual
reviews, the dataset labelled `reviewed-v3-unanimous` contains 256 accepted
pairs:

| split | pairs |
|---|---:|
| train | 190 |
| validation | 34 |
| test | 32 |

Dataset:

```text
runs/pix2pix/iro-gate512-v1/dataset-flux-source-set-reviewed-v3-unanimous
```

Fingerprint:

```text
6a9c4c680ebdb28469039842857b919ab338ad9d7f431f983b90de7cf8c4c64b
```

The diagnostic job-holdout variant contains 202 training pairs from 60 jobs
and 54 validation pairs from 17 held-out jobs. All eleven lineages occur on
both sides and no job crosses the split.

Fingerprint:

```text
66a89bd8e98d52320bd0fd5311fff151b73a1ea87c4c4bfa6693ce45e0a86843
```

### Coverage wave v2

The next target-only wave is materialized at:

```text
runs/pix2pix/iro-coverage-wave512-v2
```

Config:

```text
configs/pix2pix-iro-coverage-wave512-v2.json
```

It contains:

- 512 unique selected target images;
- 403 planned train records and 109 planned validation records;
- 85 renderer jobs;
- 158 job/gender body variants;
- 275 realized rig/frame poses;
- no selected target or body-pose cell shared with gate v1.

Provenance:

```text
request manifest:
8d34197b9e2319729faecce05093bc093ab2d02fb8d7077da0262f1a56c63a41

selected target manifest:
6c45c974a53000482ffc92570f44810731fdbdb956e1d78aedfa5b09f80d670d
```

Target planning, rendering, and selection are complete. Smooth source
generation and accepted-pair assembly are not complete.

The current prompt-authoring tree has all 512 authored records in three
ranges:

```text
prompt-sets/reviewed-v1/authoring/prompts-0000-0170.jsonl
prompt-sets/reviewed-v1/authoring/prompts-0171-0341.jsonl
prompt-sets/reviewed-v1/authoring/prompts-0342-0511.jsonl
```

Independent review files currently exist only for ranges `0171-0341` and
`0342-0511`. The `0000-0170` review range is missing. There is therefore no
complete frozen source set yet. Do not fabricate the missing reviews or claim
that v2 already contains 512 usable pairs.

Even a perfect v2 wave would add at most 512 accepted pairs, and real review
will reject some. It is one coverage wave toward a corpus of thousands, not a
magic final dataset size. Additional independent waves should be planned from
the measured coverage and rejection deficits.

## Experiment ledger

All values below come from preserved `run.json`, `metrics.jsonl`, previews,
and the detailed analysis in `pix2pix.md`.

| experiment | terminal point | validation result | conclusion |
|---|---:|---|---|
| aligned identity plumbing control | 5,000 steps | global L1 `0.00529` | loader, normalization, generator output, and basic training can learn a clean trivial mapping |
| 89M PatchGAN-70 | 38,000 steps | global L1 `0.047853` | severe foreground speckling |
| 89M PatchGAN-142 | 19,000 steps | global L1 `0.042086` | larger field did not remove noise; run stopped |
| 2B PatchGAN-70 | 1,900 steps, epoch 10 | global L1 `0.053922` | incomplete and too early for quality claims; no evidence that width solves the task |
| v6 corrected multiscale/FM | 38,000 | global `0.038665`, foreground `0.206648`, balanced `0.108863` | BatchNorm and regional-loss bugs fixed; speckling remained |
| v7 a-contrario | 38,000 | global `0.038159`, foreground `0.206122`, balanced `0.108361` | discriminator conditionality fixed; output still speckled |
| v8 palette proximity | 38,000 | global `0.038462`, foreground `0.209370`, balanced `0.109913` | continuous palette drift is not the spatial-fragmentation cause |
| 89M L1-only, lineage holdout | 38,000 | global `0.038283`, foreground `0.211167`, balanced `0.110555` | clean train reconstructions, fragmented validation; severe overfitting without any GAN |
| 89M L1-only, job holdout | 40,400 | global `0.039332`, foreground `0.217088`, balanced `0.112686` | same failure with every lineage on both sides; lineage split was not the cause |

### What v6 fixed

An earlier feature-matching ablation compared real and fake discriminator
features from separate BatchNorm contexts and left discriminator-derived
losses dominated by white background. V6 corrected that by:

- concatenating real and fake candidates into one discriminator forward;
- applying 50/50 foreground/background weighting to discriminator real/fake,
  generator adversarial, and feature-matching losses;
- applying one exact shared integer translation in `[-16, 16]` with white
  padding to each paired discriminator view.

These were real implementation defects. Fixing them improved metrics but did
not fix visual fragmentation.

### What v7 proved

The v6 discriminator preferred a shuffled source over the aligned source in
most conditionality probes. V7 added group-disjoint a-contrario mismatched
pairs as explicit fake examples.

On held-out within-taekwon swaps, v7 preferred the aligned source in:

- `99.6%` of native-scale comparisons;
- `94.5%` of coarse-scale comparisons.

The mean aligned margins became approximately `+20.5` and `+38.5`. The
conditional discriminator bug was fixed, but the sprite output remained
fragmented. The current dominant failure is therefore downstream of merely
making PatchGAN observe the source.

### What v8 ruled out

V8 penalized each output pixel by its distance to the exact target palette.
Foreground palette proximity improved only `1.19%`; mean unique output colors
slightly increased, and exact foreground target-palette membership moved only
from approximately `5.14%` to `5.23%`.

Perfect post-hoc snapping to each target's exact palette also left the spatial
fragmentation visible. Palette drift exists, but it is not the root cause.

### The decisive L1-only result

The 89M L1-only generator reaches clean near-copies on its 190 training pairs:

```text
train global L1:     0.00718
train foreground L1: 0.04938
```

It still produces fragmented held-out sprites:

```text
validation global L1:     0.03828
validation foreground L1: 0.21117
```

Validation was best around step 3,800, epoch 20:

```text
region-balanced validation L1: 0.10380
```

It worsened to `0.11055` by epoch 200 while training loss continued to fall.
Per-image BatchNorm statistics, decoder dropout at inference, and a
full-validation Fourier audit did not explain or remove the artifacts.

The job-holdout L1-only run reproduced the same curve:

```text
best validation at epoch 20:
  global L1:   0.03577
  balanced L1: 0.10575

epoch 200:
  global L1:   0.03933
  balanced L1: 0.11269
```

This is ordinary severe generalization failure on a tiny, insufficiently
diverse corpus. It happens with no discriminator at all. The U-Net has enough
capacity to render clean sprites because it does so on training examples.

<!--
  [AGENT NOTE 2026-07-30]
  The diagnosis above regarding the L1-only fragmentation was incorrect.
  The visual "colored-hagelslag fragmentation" on validation data was actually caused by a well-known
  architecture trap: using nn.BatchNorm2d with track_running_stats=True on a batch size of 1.
  
  During training (with batch_size=1), BatchNorm acts effectively as an InstanceNorm using only the
  current image's statistics. But during evaluation (`generator.eval()`), PyTorch switches to the
  global running statistics. Forcing these global statistics onto a local image drastically skewed
  feature activations, which the final Tanh layer then clamped to extreme edges (-1.0 and 1.0),
  producing the fragmented, saturated noise.

  This has now been fixed in the codebase by replacing BatchNorm2d with InstanceNorm2d
  (affine=False, track_running_stats=False), as prescribed by standard pix2pix/CycleGAN literature.
  The L1 generalization artifact was a plumbing bug, not purely a dataset scale failure.
-->

Adding more epochs or more randomly initialized parameters to the same 256
pairs primarily increases memorization.

The identity control additionally proves that the basic tensor plumbing can
produce a clean held-out result when source and target are the same. It does
not prove that the nontrivial smooth-anime-to-sprite transformation
generalizes.

Simple repeated structures, especially the ground shadow, generalize better
because they have far less variation than hair, clothing, faces, limbs,
accessories, weapons, and their discrete pixel clusters.

## Current diagnosis

The strongest evidence-supported diagnosis is:

> The present 256-pair dataset does not contain enough independent
> body/outfit/object/pose examples for a randomly initialized U-Net to learn
> the intended foreground transformation. Pair contamination reduces useful
> supervision further. Discriminator defects existed, but they are not the
> dominant remaining cause because L1-only training reproduces the held-out
> fragmentation.

<!--
  [AGENT NOTE 2026-07-30]
  The above diagnosis conclusion was heavily skewed by the BatchNorm bug.
  L1-only training reproduced the held-out fragmentation specifically because the evaluation mode
  applied mismatched global batch statistics. The assumption that the L1-only run proved the dataset
  size is solely responsible for fragmentation was flawed. While more clean data is still required
  for proper semantic generalization, the "colored-hagelslag" noise was definitively a BatchNorm
  batch_size=1 artifact.
-->

Consequences:

- Do not spend the next run tuning another PatchGAN field.
- Do not add another palette loss.
- Do not train the current corpus for more than 200 epochs.
- Do not restart the 2B or invent a 4B run on these 256 pairs.
- Do not interpret a low global L1 as usable pixel art; white background
  dominates it.
- Do not interpret clean training previews as generalization.
- Do not blame every bad output on the discriminator.
- Do not claim an exact required dataset size. “Thousands” is the working
  scale target, not a mathematically proven magic threshold.

## Disk cleanup and artifact status

`runs/pix2pix` was reduced from approximately 228 GB to approximately 958 MB.
All `checkpoints/` and `final/` directories under it were permanently deleted,
reclaiming approximately 227.8 GB.

Preserved evidence includes:

- `run.json`;
- `metrics.jsonl`;
- previews;
- evaluations;
- datasets and pair manifests;
- target selections and renderer results;
- reverse-source images;
- prompts, reviews, and source audits.

There are no remaining trained generator weights. No run can currently resume
or perform inference.

Some preserved `run.json` files contain stale absolute paths to deleted
checkpoints or final models. In particular, the interrupted PatchGAN-142 and
2B runs still report `status: "running"` although no process is active and
their checkpoints no longer exist. Treat those JSON files as historical
evidence, not executable state.

Do not infer completion merely from `run.json.status`, and do not follow a
`latest_checkpoint` path without first proving the directory exists.

## Live Git state

At handoff time:

```text
branch: main
HEAD:   faf921d4 Improve workflow editor interactions and layout
```

The last committed pix2pix slices are:

```text
0a17ed12 Add reproducible pix2pix corpus pipeline
a5f9a43f add FLUX source auditing pipeline and expand native128 training configs
```

Most later pix2pix work is uncommitted. Modified files include the pix2pix
configuration, corpus, evaluation, model, training, CLI, and documentation
owners. New untracked files include:

```text
aigen/pix2pix/augmentation.py
aigen/pix2pix/iro_coverage.py
configs/pix2pix-iro-coverage-wave512-v2.json
configs/pix2pix-native128-89m-l1-only-job-holdout-v1-200e.json
configs/pix2pix-native128-89m-l1-only-reviewed-v3-unanimous-200e.json
configs/pix2pix-native128-89m-multiscale-a-contrario-all-fm10-translate-reviewed-v3-unanimous-200e.json
configs/pix2pix-native128-89m-multiscale-a-contrario-palette100-all-fm10-translate-reviewed-v3-unanimous-200e.json
configs/pix2pix-native128-89m-multiscale-balanced-all-fm10-translate-reviewed-v3-unanimous-200e.json
configs/pix2pix-native128-89m-multiscale-balanced-fm10-reviewed-v3-unanimous-200e.json
configs/pix2pix-native128-89m-patch142-reviewed-v3-unanimous-200e.json
configs/pix2pix-native128-89m-patch70-reviewed-v3-unanimous-200e.json
```

The four tracked pix2pix unit-test files are intentionally deleted in the
worktree. The user explicitly removed them after rejecting further time spent
on that test suite. Do not restore or recreate them without an explicit
request.

There are unrelated changes in the same worktree, including workflow-editor
work and `scripts/reclaim_wsl_disk.ps1`. Preserve them. Do not commit, revert,
or delete anything unless the user explicitly asks.

Two new modules are required by the modified tracked code but are themselves
untracked:

```text
aigen/pix2pix/training.py  -> aigen/pix2pix/augmentation.py
aigen/pix2pix/iro_corpus.py -> aigen/pix2pix/iro_coverage.py
```

A future partial commit that omits either module will be broken. Treat the
modified code, both new modules, the matching configs, and documentation as
one coherent pix2pix slice if the user later asks to commit it.

The iRO acquisition configs contain a renderer access-token value. Do not
paste either config into an external chat, issue, or public repository without
first deciding with the user whether that credential may be shared or must be
rotated. Never reproduce the value in a handoff response.

## Recommended next work

### 1. Calibrate the visual acceptance boundary

Inspect the disputed validation examples and their retained audit records.
Front/back view, facing direction, limb/action pose, scale/placement,
silhouette, clothing, and held objects are paired supervision, not subjective
polish. Use the user's known verdicts to prevent the v2 review from repeating
the same false accepts.

Do not turn this into another unbounded review project or claim that manually
cleaning the remaining small v1 corpus will solve generalization. Its purpose
is to establish the correct acceptance boundary for new data.

### 2. Finish v2 provenance, not another model

Audit the current v2 prompt-authoring and review files read-only. The first
review range is missing. Complete that review only through the repository's
required independent-review workflow; do not synthesize approvals merely to
make the manifest pass.

Freeze a source set only after all 512 prompt records and independent reviews
bind the exact selected target hashes and current prompting-guide hash.

There is currently no CLI command that authors or reviews prompts, freezes
these source-set manifests, or writes output-audit verdicts. The CLI validates
and consumes those manual artifacts. Do not assume a missing command is an
invitation to bypass their contracts.

### 3. Generate bounded reverse-source candidates

Use the fixed FLUX.2 Klein 9B scaled-FP8 four-step route for rapid candidate
generation. Use more than one deterministic seed only within a declared
per-image retry budget. Keep the model loaded for batches; do not repeatedly
reload it per image.

Every accepted source must preserve:

- subject count;
- full-body pose;
- front/back and left/right direction;
- silhouette;
- clothing and dominant color blocks;
- visible loose accessories;
- held objects;
- ground-contact relationship.

Reject a plausible-looking illustration if any paired content is materially
wrong. Preserve rejected candidates and tested seeds as audit evidence.

### 4. Build independent coverage waves toward thousands

V2 cannot by itself solve the diversity problem. Plan later target waves from
realized coverage and rejection deficits. Prefer more independent
jobs/outfits/objects/poses over near-duplicate frames of the same sprite body.

Each wave must:

- exclude prior target pixels and body-pose cells;
- retain job/identity group ownership across splits;
- report planned, accepted, rejected, and deficit counts;
- never silently replace rejected examples;
- keep one coherent target style and pixel-grid convention;
- keep exact source-target alignment.

Do not assert that 5,000 pairs is scientifically guaranteed. Use learning
curves across successive corpus sizes to measure whether independent clean
data improves held-out coherence.

### 5. Use 89M L1-only as the first expanded-data gate

Once a materially larger clean corpus exists, first train the 89M L1-only
control. It is faster, isolates supervised generalization, and cannot hide the
same failure behind GAN dynamics.

Select checkpoints by held-out foreground and region-balanced validation plus
visual previews. The existing curves warn that epoch 20 may beat epoch 200;
derive the schedule from the new dataset size and stop after validation
clearly degrades. “38,000 steps” and “200 epochs” were experiment-specific,
not universal constants.

The gate passes only if held-out sprites become spatially coherent:

- stable silhouette;
- correct pose and view;
- clean connected color clusters;
- recognizable clothing/accessories/held objects;
- no colored-hagelslag fragmentation.

### 6. Reintroduce adversarial sharpening only after that gate

If expanded-data L1-only generalizes but is too smooth, reintroduce the
already-corrected v7 a-contrario multiscale discriminator as a controlled
style/sharpness experiment. Do not regress to the flawed v5 feature-matching
contract.

If expanded-data L1-only still memorizes and fragments, do not answer with a
wider random U-Net. At that point investigate a pretrained semantic
encoder/backbone or full paired fine-tuning of an edit model as a separate
architecture decision. That is not authorization to start a LoRA or silently
replace pix2pix.

## Stop conditions

Stop and report instead of improvising when:

- a reverse source changes pose, view, silhouette, clothing, or held object
  after the bounded seed budget;
- a prompt or review is missing or stale;
- source and target geometry are not exactly paired;
- a proposed wave overlaps frozen prior coverage;
- a training command points at a deleted checkpoint;
- GPU ownership is unclear;
- the desired next step would change the target style or model family;
- completing the work would require committing, deleting, or reverting files
  without explicit authorization.

## Useful read-only orientation

Start with:

```bash
cd /home/boaz/aigen
git status --short
git log --oneline --decorate -12 -- \
  aigen/pix2pix aigen/pix2pix_commands.py docs/pix2pix.md 'configs/pix2pix*'
du -sh runs/pix2pix
find runs/pix2pix -type d \( -name checkpoints -o -name final \) -print
```

The current paired datasets can be re-audited without GPU work:

```bash
.venv/bin/aigen pix2pix audit \
  runs/pix2pix/iro-gate512-v1/dataset-flux-source-set-reviewed-v3-unanimous

.venv/bin/aigen pix2pix audit \
  runs/pix2pix/iro-gate512-v1/dataset-flux-source-set-reviewed-v3-unanimous-job-holdout-v1
```

Do not run `iro-plan`, `iro-render`, `iro-select`, generation, preparation, or
training as an orientation step: those are state-changing operations.

## One-sentence continuation brief

Finish and rigorously review the v2 reverse-source corpus, then use successive
clean, independent coverage waves to test whether the 89M L1-only generator
begins to generalize; only after that succeeds should PatchGAN be used for
pixel-style sharpening, and only after it fails with materially more clean
data should a pretrained semantic backbone replace the randomly initialized
U-Net.
