# Local PixAI Edit Pro — Implementation Blueprint

Status: authoritative build blueprint for the local, <=16GB PixAI-Edit-Pro-style
character edit pipeline. This is THE plan: it fully replaces the previous
step 3–7 "external VLM planner / identity dossier" design. `AGENTS.md`
behavioral rules still hold.

Goal restated: a **local** editor that behaves like PixAI Edit Pro. Lower quality
and slower are acceptable. Lower quality **must not** come from a thinner
pipeline — only from smaller/quantized models. The sophistication stays; it just
lives in the correct layer.

---

## 0. The one rule that fixes everything

> **The character never becomes text.**
>
> Reference images enter the generation model **as images** (VAE latents +
> in-model vision tokens). External code is allowed to route, to select *which*
> reference indices to feed, to build *structural* conditions (pose / mask /
> depth as images), and to pass the user's instruction as the prompt. External
> code is **never** allowed to serialize the character (hair, eyes, outfit, body,
> "which view this is") into text/JSON that then conditions generation.

Every failure you have hit — JSON that won't parse on an 8B model, "back-view =
close-up face" corrupting the output, endless prompt-tightening, the drift into
IP-Adapter — is one symptom: **visual reasoning was pushed through a text
bottleneck.** All the "workarounds" (JSON → line-protocol → free-text →
IP-Adapter) tried to repair the bottleneck instead of deleting it.

### Quality invariants

These are visual, system-level quality targets. They are not pixel-exact test
contracts and must never be translated into per-character fixtures or special
cases:

- Identity remains visually consistent across pose, view, and shot changes.
- Body proportions, outfit construction, palette, and ink/line style remain
  visually faithful to the supplied references.
- Large pose changes follow the user's intent without sacrificing identity.
- A local edit preserves the visual content and composition outside the intended
  target region.
- Detail level must match the shot scale and the available raw-generation pixel
  budget. Postprocessing cannot substitute for missing raw detail.
- Pipeline code and tests contain zero character-specific constants, prompts,
  geometry, reference-role assumptions, or fixtures.

---

## 1. What PixAI Edit Pro actually is (de-hallucinated)

`docs/pixai.md` is mostly the PixAI *assistant guessing about its own
architecture*. It cannot introspect its weights; it pattern-matches a
plausible-sounding 2022-era modular pipeline. Treat almost all of it as
speculation, **except one sentence**, which is the true mental model:

> pixai.md line 16: *"It treats your reference images as visual information, not
> text. The model interprets them holistically."*

That line is correct and it is the whole design. The rest of the document —
ArcFace/InsightFace face embeddings, GFPGAN/CodeFormer face restoration, separate
"component encoders", "anti-drift loops", a bolt-on pose ControlNet stack — is the
hallucinated modular story. Modern editors (PixAI Edit Pro, Qwen-Image-Edit,
FLUX Kontext, Gemini/GPT native image) are **end-to-end multimodal diffusion
transformers (MMDiT)**. The "VLM parser", "component understanding" and "identity
preservation" happen *inside one model's forward pass*, in latent space — not as
external stages that emit text.

**You have been reverse-engineering the hallucinated half of a hallucinated
document, with an LLM that hallucinates more modules on top.** That is why it
never converges.

### The single most important architectural fact

`Qwen-Image-Edit` **already contains `Qwen2.5-VL` as its condition encoder.** The
input image is fed through *both* Qwen2.5-VL (semantic understanding) *and* the
VAE (appearance), and those become the conditioning. Multi-image consistency
across the supplied image inputs is resolved **in-model** by joint attention.

So the current pipeline runs **two** Qwen2.5-VLs in series: an external one that
looks at the refs and writes a JSON dossier, whose text becomes the prompt for
the internal one. That is redundant (the editor has its own VLM), lossy (text
bottleneck), and fragile (deep JSON on a quantized 8B model). Confirmed in the
code: [character_qwen_edit.py:383-384](../aigen/character_qwen_edit.py#L383-L384)
feeds `prompt = planned.edit_instruction`, tagged
`edit_instruction_source: qwen_vlm_edit_planner`
([character_qwen_edit.py:450](../aigen/character_qwen_edit.py#L450)). The
character, serialized to text by the external VLM, *is* the generation prompt.

---

## 2. Correct architecture — two lanes

```text
                    ┌──────────────────────── TEXT LANE (thin) ───────────────────────┐
user instruction ──▶│ 1. parse instruction   2. route   (opt) scene-only expansion    │──┐
                    └──────────────────────────────────────────────────────────────────┘  │
                                                                                            ▼
                    ┌──────────────────── IMAGE / LATENT LANE (all the work) ─────────┐   prompt (thin)
reference images ──▶│ 3. feed the complete ordered reference pack                    │──┐  │
                    │ 8. build optional structural conditions (pose/mask/depth) — IMG │  │  │
                    └──────────────────────────────────────────────────────────────────┘  │  │
                                                                                            ▼  ▼
                                          ┌──────────────────────────────────────────────────────┐
                                          │  Qwen-Image-Edit-2511  (the MMDiT: contains Qwen2.5-VL)│
                                          │  does steps 4–7 INTERNALLY:                            │
                                          │  subject-id · component extraction · cross-ref align   │
                                          │  · relevance weighting · identity-consistent redraw    │
                                          └──────────────────────────────────────────────────────┘
                                                                     │
                                                10. audit loop (raw, bounded) → 11. optional postprocess (upscale)
```

- **Steps 4–7 of `PLAN.md` are not deleted — they move into the model.** That is
  where PixAI does them too. You are not doing *less* semantic control; you are
  doing it *model-native* instead of through an 8B model's JSON. This is the
  opposite of a "simpler pipeline": the heavy lifting is a 20B MMDiT plus real
  structural conditioning, not a text blob.
- **The text lane never describes the character.** It carries the user's request
  ("close-up, looking left, white background") — which is *output intent*, not a
  character dossier. Note your own `PLAN.md` example instruction is already exactly
  this shape.
- **All supplied references remain native visual inputs.** External code does
  not discard them to satisfy a guessed image-count budget.

---

## 3. Concrete models (Hugging Face)

Target: single RTX 50-series / Blackwell GPU, ≤16GB VRAM. Stages run
**sequentially** (load → run → unload), never concurrently (see §6).

| Role | Model (HF) | Notes |
|------|------------|-------|
| **Edit backbone (core)** | `Qwen/Qwen-Image-Edit-2511` via **LightX2V** | Official scaled FP8 Lightning 8-step is the active 16GB route. LightX2V owns block offload, Qwen2.5-VL conditioning, VAE encoding/decoding and the ordered multi-image input bundle. 4-step FP8 is speed-only evidence; BF16 + distilled LoRA is a quality-reference experiment if FP8 artifacts become structural. |
| **Backbone, later candidate (evidence-gated)** | FLUX.2 **klein** (NVFP4-class quant) | Multi-reference editor, consumer-GPU positioning. New pipeline class + conditioning API = a real adapter build; anime-character fidelity unproven. Only after direct 2511 output experiments show a persistent gap the Qwen line does not close. |
| **Audit VLM (narrow)** | `Qwen/Qwen2.5-VL-7B-Instruct` (8-bit) | **Only** job: stage 10 — compare the candidate image against all supplied reference images and return a pass/fail verdict plus short region pointers ("skirt", "right hand") for regions that visually deviate. It also receives the user instruction, route, and pose/scene context (output intent — allowed text-lane input) so intent-consistent changes (a flaring skirt mid-jump, open fingertips) are not flagged as identity errors. Region pointers feed Florence-2 grounding only. **Never** a character description, never appearance text toward generation. An integrated pipeline stage, **not** a separate CLI product layer. |
| Pose preprocessor | DWPose (via `IDEA-Research/DWPose`, or `controlnet_aux`) | Produces a keypoint control image for `pose_transfer`; it is appended to the native 2511 visual input bundle. |
| Depth preprocessor | `depth-anything/Depth-Anything-V2-Large` | Produces a depth control image for structured scene routes; it is appended to the native 2511 visual input bundle. |
| Edge preprocessor | Canny (OpenCV) or `controlnet_aux` | Produces an edge control image for structured scene routes; it is appended to the native 2511 visual input bundle. |
| Mask / segmentation | `facebook/sam2` (SAM2) | **Preservation/compositing boundary only** for `local_repair_or_inpaint` / regional outfit swap — the exact-pixel guarantee, never the carrier of character semantics (see §4 defect response ladder). |
| Region grounding (text→box) | `microsoft/Florence-2-large` | Ground a region pointer ("her left glove") to a box in the candidate **and** in the supplied references, to build close-up conditioning crops and to seed SAM2. Image-domain output (regions), not a character description. |
| Upscale (postprocess) | IllustrationJaNai V1 DAT2 | Anime/illustration upscaler. Stage 11 only, after the audit loop passes. |
| Anime polish (optional) | A current SOTA anime **SDXL** checkpoint (Illustrious-XL / NoobAI-XL family) as **low-denoise img2img refine** | Optional stage-11 finish for anime crispness. It is a *refiner*, never the identity carrier. Pick the current best checkpoint; this space moves fast. |

### Explicit anti-recommendations (these are the traps)

- **No GFPGAN / CodeFormer.** They restore *photoreal human* faces and will
  destroy anime faces. The pixai.md "face restoration" line is hallucination.
- **No IP-Adapter as a PixAI replacement.** 2511's multi-image path *is* the
  faithful identity mechanism. IP-Adapter is a generic anime-diffusion reflex.
- **No external character dossier, no deep nested JSON from the VLM, no
  line-protocol/free-text variants.** All were attempts to fix a bottleneck that
  must be removed, not fixed.
- **No hand-written component taxonomy or measurement/geometry extraction.** The
  model owns component understanding (also `AGENTS.md` §2).

### Anime domain (the one legitimate ChatGPT concern, correctly placed)

Qwen-Image is trained with heavy illustration data, so 2511 is a strong anime
editor, but not elite. If anime fidelity falls short, the lever is a **style LoRA
on the backbone** or the **stage-11 anime SDXL refine pass** — *not* a pipeline rewrite
and *not* an external VLM. Keep it as a model-swap slot behind the same boundary.

---

## 4. Route → conditioning map (the real, image-domain sophistication)

Your `CharacterConditioningPlanner`
([character_conditioning_planner.py:14](../aigen/character_conditioning_planner.py#L14))
already has this exact shape (`route_kind → conditioning_modes → planned_tools`).
Keep it. Change only its **input**: it keys off the **route** (+ structural
sources the user supplied), **not** off `visual_analysis` text.

| Route | Extra conditioning | Tool → control image |
|-------|--------------------|----------------------|
| `portrait_identity_generation` | none | refs + instruction only |
| `full_body_identity_generation` | none | refs + instruction only |
| `view_change` | none | refs + instruction only |
| `scene_insertion` | optional depth/edge | Depth-Anything-V2 / Canny |
| `pose_transfer` | pose keypoints | DWPose → 2511 visual control input |
| `local_repair_or_inpaint` | region mask (user-directed); close-up variant is an evidence-gated reserve (see defect ladder below) | Florence-2 (box) → SAM2 (**boundary only**) |
| `object_removal` (future, route-gated) | none (maskless) | full-image "remove X" edit + diff-composite: accept only the changed zone, restore everything else pixel-exact from the source. Until this route exists, removals run as `local_repair_or_inpaint` and residual artifacts are the audit loop's job. |
| `outfit_swap` | optional region mask | SAM2 if regional; else refs + instruction |
| `style_transfer` | style ref as image (+ opt. style LoRA) | — |
| `layout_or_sheet` | feed the sheet whole | — |
| `text_or_label_heavy` | none (2511 renders text well) | instruction carries the literal text |

Identity/portrait/view/normal-scene routes need **no** extra conditioning — the
backbone handles them. That is not "too simple"; it is the model doing its job.
Sophistication appears exactly where a route genuinely needs geometry.

### Defect response ladder (single-pass first)

The proven root cause of nearly every quality defect so far — small figure,
bare finger, unconvincing hands — was **generation-time raw pixel budget**,
fixed by single-pass measures (canvas fill, native resolution), not by
patching. Modern editors resolve semantics in one holistic pass; iterative
surgery is the exception, never the architecture. When the audit (stage 10)
or the user flags a defect, respond in this order:

1. **Select.** Pick the best of the N candidates already generated.
2. **Re-run the single pass, escalated.** Other seeds, 8-step FP8, a larger
   raw canvas where evidence supports it, or the §3 BF16 quality-reference
   route — model-native levers only. Slower is explicitly acceptable.
3. **Targeted close-up repair (evidence-gated reserve).** Only for a defect
   class that demonstrably survives a fully escalated single pass. Flow:
   Florence-2 grounds the region pointer in the candidate **and** in the
   originally supplied references (same refs throughout) → image-only
   close-up crops (the one legitimate motivation: references are far
   higher-res than the ~1MP conditioning canvas, so a ref close-up carries
   detail the full-frame pass physically destroys) → Qwen inpaints the
   candidate crop with the ref close-ups as native visual conditioning, thin
   prompt ("make this region match the references") → SAM2 provides **only**
   the preservation/compositing boundary → paste back; the hard pixel-diff
   assert outside the boundary stands. If the pointer grounds in no
   reference, the repair runs with the full supplied references — an explicit
   route condition, not a fallback. **Do not build or invoke this rung
   pre-emptively.**

`local_repair_or_inpaint` as a *user-directed* route (explicit region/mask
from the user) keeps its existing mechanics: masked inpaint, SAM2 as boundary
only, exact preservation outside. The alternative for *removals* (maskless
full-image edit + diff-composite) remains the `object_removal` route —
deletions, not construction repairs.

### Structural-source composition rules

**Scene-source ownership.** For `scene_insertion` depth/edge conditioning, the
scene source **owns the composition**: never crop it — its framing *is* the
intent. Only pose controls use the content-box crop. A pose sheet is not a
scene source; feeding one as the depth/edge source is a routing error, not a
reason to add composition heuristics.

**Pose-control fit rule.** The DWPose control image must be fitted to the
generation canvas aspect **without black letterbox bars** — crop around the
skeleton content (pure geometry) so the control canvas matches the output
canvas aspect. Never heuristically enlarge the figure; the source framing owns
composition. Black padding bars are spurious control input and shrink the
figure on the canvas. If the target-aspect window around the skeleton content
exceeds the source bounds, extend the canvas with the control's own background
(a DWPose render is skeleton-on-black, so extension is background continuation
at unchanged skeleton scale — not a letterboxed, shrunken figure). Pure
geometry; this case is never a hard failure and never an enlargement
heuristic.

**Pose-proportion rule.** A keypoint control encodes the *source body's
proportions* as well as the pose; a pose source with off-character
head-to-body-to-legs ratios will override the references (keypoint control
wins over refs by design). The fix is model-owned retargeting, in this order:

1. **Native pose transfer** — feed the pose source as a plain input image
   ("match the pose of image N; identity from the references"), no keypoint
   control. The model reconciles pose and proportions from the refs
   in-model.
2. **Self-derived skeleton lock** (when the pose must be exact, e.g.
   keyframes) — extract DWPose from the accepted native-transfer result and
   use that skeleton as the strict keypoint control for the final
   high-quality run. Correct anatomy by construction; no hand-written bone
   math — the model did the retargeting, geometry only locks in what the
   model decided.
3. **Bone-length retargeting** (source joint angles + per-run, ref-derived
   bone ratios; keypoints→keypoints, image-domain, character-agnostic) is a
   last-resort reserve, never the default.

---

## 5. Stage-by-stage data flow (what is text, image, latent)

| # | Stage | Input | Output | Domain |
|---|-------|-------|--------|--------|
| 1 | Parse instruction | raw text + envelope | `instruction_plan` | text |
| 2 | Route | `instruction_plan` | `route_plan` (route_kind + editor constraints) | text |
| 3 | Reference intake | all supplied ref images | ordered `reference1..N` handles | image |
| 8 | Conditioning build (route-gated) | supplied refs + user structural sources | control image(s): pose/mask/depth | image |
| 9 | Generate | all supplied ref/source/control images + **thin prompt** | edited image | latent→image |
| 10 | **Audit loop (bounded, on raw)** | raw candidates + supplied refs + user instruction/route context (+ repair mask if any) | pass (best candidate selected), **or** defect report → §4 defect response ladder | image → verdict + region pointers |
| 11 | Postprocess (optional, after audit passes) | audited raw image | upscaled/anime-refined image | image |

External code imposes no identity-reference count, total-image count or
one-control-per-case limit. The complete ordered visual bundle is passed to the
backend. Actual backend or resource exhaustion fails visibly instead of being
predicted by an arbitrary guard.

The **prompt** at stage 9 is the user's instruction, lightly normalized (and, only
for scene/style, optionally expanded on the *scene* — never the character). It is
**not** VLM-authored from ref analysis.

### Stage 10: the audit loop (M5)

The pipeline must find its own errors instead of relying on a human to point at
the broken skirt or notice the missing glove. It is an **integrated pipeline
stage** inside the edit/refine flow — not a separate CLI product layer. Two
complementary tracks, both character-agnostic:

- **Deterministic pixel-diff track (no model).** For masked repairs, zero
  changed pixels outside the mask is a **hard assert**. Changed-pixel clusters
  outside the intended zone become region candidates themselves. This catches
  preservation violations and boundary/contour artifacts mechanically.
- **Semantic VLM track.** The audit VLM (see §3) receives the selected
  reference images plus the candidate, **and** the user instruction, route,
  and pose/scene context, and is asked, generically, where the candidate
  visually deviates from the references *given the requested change* — or
  "none". No checklist, no character constants. The intent context is what
  keeps a flaring skirt mid-jump, open fingertips, or any other legitimate
  consequence of the instruction from being flagged as an identity error.
  This track catches errors with no diff signature, e.g. a glove missing
  relative to the references.

The audit's primary role is **detector and best-of-N selector** — the
automated pair of eyes that a human operator is today. Combined verdict is
pass only if both tracks pass on the selected candidate. On fail, follow the
**§4 defect response ladder**: select a better candidate, else re-run the
single pass escalated (seeds / steps / rank / canvas); targeted close-up
repair only for defect classes proven to survive full escalation. The loop is
bounded (default 2 iterations); each iteration logs its verdict and chosen
response into the run directory. If the audit still fails after N iterations,
the run **fails visibly with the report** — no pending states, no silent
downgrade to "good enough".

**Ordering: audit before postprocess.** The loop runs on the **raw**
generation at native resolution; Real-ESRGAN runs **once**, after the loop
passes. Repairing raw pixels is what the backbone conditions on; upscaling
first would launder missing raw detail (quality invariant §0) and force every
repair iteration through a re-upscale. The upscaled result gets no semantic
re-audit — the upscaler is deterministic and a second VLM load buys nothing.

**Tiled detail refine (optional stage-11 extension, to validate).** For
2K–4K final outputs the anime polish pass may run **tiled**: ESRGAN upscale,
then overlapping-tile low-denoise img2img with the SDXL anime refiner —
overlap blending, identity anchored by the upscaled base image. This raises
detail crispness only; it cannot correct structure and is no substitute for
generation-time pixel budget. Per-tile prompts stay empty or the global thin
instruction — **never per-tile captions** (that is the text bottleneck
through the back door). Be wary of using the Qwen editor itself as tile
refiner: every tile re-enters its VL encoder, and an ambiguous body-part tile
invites hallucination; the SDXL refiner slot is the ecosystem-proven tool for
this job. Validate visually before adopting.

### Resolution policy

`--max-side` is a **cap** for VRAM escalation, never a target. The refine
canvas is the native source size (×16-aligned) — no ~1MP area bucketing — and
the source is never downscaled below the generation canvas size. Measured on
the pistol repair: a 1008 max-side default caused a double resize
(down to ~672×1008, back up to 832×1248) and 114,603 changed pixels outside the
mask; native 1248 gave exactly 0, at the same denoise canvas and VRAM. Small
defaults (640/1008) are silent quality killers, not optimizations.

---

## 6. 16GB VRAM execution plan

Qwen-Image-Edit-2511 runs through LightX2V's FP8 block-offload route. Run stages
sequentially:

1. Build any control images with lightweight preprocessors (Depth-Anything /
   DWPose / SAM2 / Florence-2 — each load→run→unload).
2. Load the LightX2V 2511 conditioner and FP8 Lightning backbone, feed the full
   ordered visual bundle, generate all requested candidates in one worker, then
   unload.
3. Audit loop (stage 10): pixel-diff track is model-free; then load audit VLM
   (8-bit) → verdict + region pointers → **unload**. On fail, rerun the
   preprocessors + backbone on the region **close-ups** (§4) — a crop canvas,
   cheaper than a full-frame pass (bounded iterations).
4. (Optional) load upscaler/refiner → finish.

Weights stream from
pinned system RAM. Every timing run records seconds/step, peak VRAM, WSL RAM,
and swap usage — the moment swap grows, the measurement is invalid as a
normal offload figure. LightX2V owns its own offload mechanism. VAE decode stays
on the GPU after denoising weights are released. Reuse one worker for multiple
cases and seeds so model loading is paid once.

For the common identity/portrait/view/scene case: **load backbone, feed all refs
and the instruction, generate.** No selector VLM runs before the editor.

---

## 7. Surgery on the existing repo

Keep the routing/conditioning scaffolding; excise the text-serialization organs.

**Keep (reuse as-is or lightly):**
- `character_instruction_*` — step 1.
- `character_task_router.py` / `character_task_route_models.py` — step 2. Its
  `ModelCapabilityRegistrySpec` / `FinalEditorConstraintsSpec` remain routing
  contracts; reference-count budgeting does not.
- `character_conditioning_planner.py` / `_models.py` — step 8. Re-key its input
  from `visual_analysis` to `route_plan` + supplied structural sources.
- `character_reference_pack.py` **pack-building only** (`build_character_reference_pack`).
- `vlm_qwen.py` — keep the runner for the stage-10 audit only.
- `diffusers_kontext_adapter.py`, `keyframe_*`, `lora_*` — separate tracks, leave.

**Gut (these are the disease):**
- `character_reference_observer.py` — the per-ref text observations. **Delete.**
- In `character_reference_pack.py`: the **identity dossier** parser
  (`_identity_parser_prompt`, `parse_character_reference_pack`) and the
  `visual_analysis` / `reference_semantics` / VLM-authored `edit_instruction`
  fields in `parse_character_edit_plan`. No replacement selector is needed.
- In `character_qwen_edit.py`: stop sourcing `prompt` from
  `qwen_vlm_edit_planner`. `prompt` = user instruction (normalized). Feed every
  ordered reference from the pack.

**Add (small):**
- Preprocessor adapters that each emit a control *image*: `dwpose`, `depth_v2`,
  `sam2_mask`, `florence_region`, `canny`. Wire them into the conditioning
  planner's `planned_tools`.
- LightX2V 2511 generation adapter that accepts the complete ordered visual
  bundle and the thin prompt.

---

## 8. Active work order (so Codex stops thrashing)

1. **2511 backbone — established.** LightX2V FP8 Lightning 8-step at the 1.77MP
   raw sweet spot is the only active Qwen edit route. Identity and detail passed
   the accepted comparison runs.
2. **Complete native reference bundle — current.** Feed every ordered pack
   reference plus every route-required source/guide/control image. No selector
   VLM and no guessed count limits.
3. **Structural-conditioning evidence — current.** Judge native pose, DWPose,
   depth and edge by real two-seed outputs. A control path is accepted only when
   it improves requested geometry without unacceptable identity drift.
4. **Audit (stage 10).** Pixel-diff track plus an intent-aware audit VLM as
   detector and best-of-N selector; bounded response per the §4 defect ladder.
5. **Stage-11 postprocess.** Upscale and optional anime refine only after the raw
   candidate has passed visual/audit selection.
6. **Second character pack.** The golden-rule acid test: zero code changes,
   pure assets and direct visually judged outputs.
7. **Evidence-gated reserves only:** close-up repair, object-removal
   diff-composite, bone-length pose retargeting and tiled detail refine.
8. Only then, if anime fidelity is short: a 2511-compatible style adaptation or
   the separate stage-11 anime refiner. FLUX/Kontext remains a separate pipeline.

Do not advance a rung until the previous one renders.

## 9. First smoke that must pass

```text
refs:        assets/characters/jillian/references/*   (all fed directly, in order)
instruction: "Character as shown in referenced images. Close-up face,
              looking to the left with neutral expression. Neutral
              lighting, white background."
route:       portrait_identity_generation
generation:  LightX2V Qwen-Image-Edit-2511(image=[refs], prompt=instruction)
assert:      an image is produced; NO identity_profile.json, NO
             reference observations, NO visual_analysis, NO VLM-authored
             edit_instruction anywhere in the path.
```

## 10. Hard rules (do NOT)

1. **Never serialize the character to text.** No dossier, no `visual_analysis`,
   no per-ref observations, no VLM-authored `edit_instruction`.
2. The audit VLM returns **verdicts and short region pointers**. Nothing else.
   Region pointers feed
   Florence-2 grounding and the thin repair instruction only — never appearance
   descriptions that condition generation. The audit is **intent-aware**: it
   receives the user instruction, route, and pose/scene context (output
   intent — allowed text-lane input, not character text) and judges against
   references *and* the requested change; deviations the instruction asked for
   are not defects.
3. No GFPGAN/CodeFormer. No IP-Adapter as identity mechanism. No deep nested JSON
   from any VLM. No line-protocol/free-text substitutes.
4. Structural conditioning is **image-domain** (pose/mask/depth maps), route-gated,
   optional. Never a text description of geometry.
5. Character-agnostic code stands (`AGENTS.md` golden rule). The reason it now
   *holds naturally*: there is no character text anywhere to hardcode.
6. Lower quality is allowed only from smaller/quantized models — never from
   removing a conditioning path that a route needs.
7. **No artifact-driven constants.** Never raise mask expansion, margins, or
   similar knobs to paper over one failing case (e.g. 4→8 px dilation to hide a
   leftover contour). A small fixed dilation is edge-uncertainty margin only;
   residual artifacts are handled per the §4 defect response ladder
   (select / regenerate escalated), not by wider masks.
8. **No pending states, no silent simplification.** When the audit loop
   exhausts its iterations without a pass, the run fails visibly with the
   audit report. Checklists, warnings, or "known issue" markers are not a
   substitute for the repair actually happening.
9. **No per-tile captions** in any tiled pass — tile prompts stay empty or
   carry the global thin instruction. Describing tile content is the text
   bottleneck through the back door.
10. **Single-pass first.** Iterative surgery (close-up repair, retargeting
    math) is an evidence-gated exception, never the architecture. If the
    pipeline routinely needs repair loops, the single pass is misconfigured
    or the model slot is too small — fix that instead.
