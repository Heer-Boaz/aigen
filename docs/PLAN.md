# AI51 Character Identity Pipeline — Plan

Status: authoritative design document for the PixAI-style character pipeline.
If a task or instruction conflicts with this file, stop and ask — do not
silently reinterpret. Hard behavioral rules for agents live in `AGENTS.md`
at the repo root.

## Goal

A local, PixAI-style multi-reference character editing pipeline that runs on a
16GB GPU (RTX 50-series / Blackwell). Scope:

- character identity (same character across generations)
- view/camera (front, back, left/right profile, three-quarter, portrait)
- simple pose; platformer pose later
- local identity repair (fix bow, skirt back, chest size, boots, face)

North star: PixAI Edit Pro can take four reference images plus the single line
*"Character as shown in referenced images. Generate reference card that
illustrates the character's proportions as if it were annotated concept-art."*
and return a near-perfect annotated reference card. All character understanding
— what each image shows, the proportions, the outfit — happens inside the
models. Our pipeline must have that same property: the intelligence lives in
the models; our code only routes images, instructions and conditioning between
them.

## Golden rule: character-agnostic code

Pipeline code contains **zero character-specific content**. Every fact about a
character (hair, outfit, bow, chest size, silhouette, which image is the back
view, …) is extracted at runtime by a model from the reference images, or read
from a model-produced artifact. All JSON examples in this document illustrate
the *shape* of model output — their values must never appear as constants,
defaults, prompt fragments or fixtures in code.

**Litmus test:** swap in a completely different character's reference pack. If
any code change is needed to make the pipeline work, the code is wrong.

## Architecture

```text
VLM/semantic parser
→ normalized instruction JSON
→ task-based reference selection
→ Nunchaku Qwen-Image-Edit-2509 FP4/r32 (multi-image edit)
→ Florence-2 / SAM2 / DWPose hidden region/pose conditioning (when needed)
→ candidate contact sheet → human selection → selected-image refine
```

### 1. VLM / semantic parser

Purpose: understand the reference images and produce a compact, machine-readable
character dossier (`identity_profile.json`) that the rest of the pipeline
consumes.

- Model: the locally available Qwen-VL (Qwen2.5-VL class). It is strong at
  visual recognition, object localization and structured JSON extraction —
  parser work, not generation.
- Input: all images in the reference pack (e.g. front, portrait, side, back,
  optionally body_shape).
- The VLM infers **itself** what each image is (reference roles). Role metadata
  in a manifest is optional evidence, never required input.

Output shape (values below are *illustrations of model output*, never code
constants):

```json
{
  "identity": {
    "hair": "…", "eyes": "…", "neckwear": "…", "top": "…",
    "bottom": "…", "legwear": "…", "boots": "…", "style": "…"
  },
  "body_proportion": {
    "chest_size": "…", "build": "…", "shoulder_width": "…", "waist": "…",
    "hip_skirt_silhouette": "…", "side_body_thickness": "…",
    "leg_proportion": "…", "skirt_back_shape": "…",
    "do_not_change": ["…"],
    "evidence_refs": ["…"]
  },
  "reference_roles": { "…": "…" }
}
```

#### body_proportion is a model-extracted invariant

```text
body_proportion = model-extracted identity invariant
body_proportion ≠ required reference image role
body_proportion ≠ generation case
```

- The VLM infers body proportion from all available refs.
- An explicit `body_shape` reference image is **optional extra evidence**: use
  it when present, never hard-fail when absent.
- Every generation case consumes `identity_profile.body_proportion` as an
  invariant.
- Plan output must show: `refs_used`, `identity_profile_used`,
  `body_proportion_source: "model_extracted_from_reference_pack"`, and
  `optional_missing_refs` (e.g. `["body_shape"]`) when applicable — with no
  hard failure for a missing optional ref.

A simple geometric measurement layer (segmentation → bbox → ratios) may be
added **later, as supporting evidence only, and only when explicitly
requested**. It is never the primary extraction path.

### 2. Prompt / instruction normalization

The user's prompt stays simple (e.g. `same character, right profile, clean
background`). The normalizer turns it into an unambiguous structured
instruction:

```json
{
  "task": "identity_edit",
  "requested_view": "…",
  "requested_pose": "…",
  "background": "…",
  "must_preserve": ["…"],
  "avoid": ["…"]
}
```

`must_preserve` / `avoid` are populated from `identity_profile.json` (model
output), never hardcoded. No long prompt-craft, no creative reinterpretation —
structural instruction only.

### 3. Task-based reference selection

Qwen-Image-Edit-2509 states 1–3 input images is currently optimal, so each case
selects 1–3 refs. The case→role mapping is mechanism-level and may live in
code (it references *roles*, which the VLM inferred, not characters):

```text
front            = front + portrait + side
left_profile     = side + portrait + front
right_profile    = side + portrait + front
back             = back + portrait + front
three_quarter    = front + side + portrait
platformer_pose  = side + portrait + poseplate/keypoint
repair           = relevant ref(s) + current image
```

Every case additionally consumes `identity_profile.body_proportion` (plus
case-specific invariants such as `skirt_back_shape` for `back`).

### 4. Edit diffusion model

First concrete model:

```text
nunchaku-ai/nunchaku-qwen-image-edit-2509
lightning-251115/svdq-fp4_r32-qwen-image-edit-2509-lightning-4steps-251115.safetensors
```

- Pipeline class: `QwenImageEditPlusPipeline` with
  `NunchakuQwenImageTransformer2DModel`; multi-image input via
  `image=[image1, image2, image3]`.
- FP4 for Blackwell/RTX 50-series; per-layer offload and sequential CPU offload
  can bring VRAM down to roughly 3–4GB at a speed cost.
- Escalation ladder (only when quality falls short at the current rung):

```text
4-step r32 lightning → 8-step r32 lightning → full r32 → r128
```

Qwen-Image-Edit-2511 is better on paper (character consistency, geometric
reasoning) but the full 2511 route is too heavy for 16GB — out of scope for now.

### 5. Hidden pose / region / segmentation conditioning

Full capability, but not a generative obligation on every run:

- Identity-only views: **no pose/region conditioning needed**.
- Poseplate/platformer cases: pose/keypoint map + edge/sketch map + mask/region
  map. Qwen-Image-Edit-2509 natively supports ControlNet-like image conditions
  (depth, edge, keypoint, sketch).

Local tools and flow:

```text
VLM/parser names a region ("arm/fist region", "skirt back", "face", "bow")
→ Florence-2 produces box/region (object detection, phrase grounding,
  region proposals, referring-expression segmentation)
→ SAM2 produces the mask (SAM2ImagePredictor / automatic mask generation)
→ DWPose/OpenPose produces keypoint maps for body poses
→ Qwen edit receives image refs + optional keypoint/sketch/mask condition images
```

### 6. Candidate / refinement / filtering

- Per case: 2–4 candidates → contact sheet → **the human picks**. No model as
  final judge.
- Refinement is the same edit mechanism: selected image + original refs +
  mask/region + instruction (e.g. fix bow shape, fix skirt back, reduce chest
  size, fix boots, fix face).
- This refine loop matters more than more seeds.

## Explicitly out of scope

```text
collage/layout engine        magazine/poster generation
text rendering module        product identity module
OCR/text specialist          multilingual typography
scene composition for ads
```

These make the scope muddy; do not build them.

## Relationship to the existing FLUX/Kontext code

FLUX/Kontext stays useful, but not as the PixAI clone. The two stand side by
side:

```text
FLUX/Kontext: existing single-reference identity generator / fallback / polish
Qwen Edit:    multi-reference identity/view/pose editing
```

Do not wedge Qwen into the FLUX prompt-embedding code.

## Implementation phases

### Phase 1 — Reference pack + parser (no generation)
```text
aigen characters reference-pack build
aigen characters reference-pack parse
→ reference_pack.json, identity_profile.json
```

### Phase 2 — Qwen edit runner
```text
aigen characters qwen-edit-run
  --pack assets/characters/<name>/reference_pack.json
  --case front|side|right_profile|back|three_quarter|portrait
  --model nunchaku-qwen-edit-2509-r32-4step
  --output-dir …
→ contact_sheet.png, result.json, case PNGs
```

### Phase 3 — Region/mask helper (no generation)
```text
aigen characters region-plan
  input:  image + request ("blue bow", "skirt back", "face", "boots")
  output: region boxes, SAM2 masks, debug sheet
```

### Phase 4 — Refine runner
```text
aigen characters qwen-edit-refine
  input:  selected image + reference pack + mask/region + instruction
  output: fixed candidates + contact sheet
```

### Phase 5 — Platformer pose (only after identity/view works)
```text
aigen characters qwen-edit-pose
  input:  side ref + portrait ref + poseplate/keypoint/sketch
  output: platformer pose candidates
```

## First smoke test

Not six features at once. First run:

```text
case: right_profile
refs: side + portrait + front
model: 2509 r32 4-step
long side: 640
candidates: 2
```

If that works: back, three_quarter, portrait. If 4-step identity is weak:
8-step r32. If 8-step fits but is insufficient: full r32.

## Sources

- https://huggingface.co/Qwen/Qwen-Image-Edit-2509 — multi-image editing, 1–3
  optimal inputs, person consistency, native depth/edge/keypoint conditions
- https://huggingface.co/Qwen/Qwen-Image-Edit-2511 — better consistency/LoRA,
  too heavy for 16GB
- https://huggingface.co/Qwen/Qwen-Image-Edit — dual path: Qwen2.5-VL for
  semantic control + VAE encoder for appearance control
- https://nunchaku.tech/docs/nunchaku/usage/qwen-image-edit.html — Nunchaku
  low-VRAM route, pipeline classes, lightning variants, offload options
- https://arxiv.org/abs/2502.13923 — Qwen2.5-VL technical report
- https://huggingface.co/docs/transformers/en/model_doc/florence2 — Florence-2
  prompt-based detection/grounding/segmentation tasks
- https://github.com/facebookresearch/sam2 — SAM2 promptable segmentation
