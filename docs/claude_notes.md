# Local PixAI Edit Pro — Implementation Blueprint

Status: authoritative build blueprint for the local, <=16GB PixAI-Edit-Pro-style
character edit pipeline. This supersedes the data-flow of `docs/PLAN.md` steps
3–7. Where this file and `PLAN.md` disagree, this file wins. `AGENTS.md` behavioral
rules still hold.

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
across the 1–3 inputs is resolved **in-model** by joint attention.

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
reference images ──▶│ 3. select ≤3 ref indices (only if >3 supplied)                  │──┐  │
                    │ 8. build optional structural condition (pose/mask/depth) — IMG  │  │  │
                    └──────────────────────────────────────────────────────────────────┘  │  │
                                                                                            ▼  ▼
                                          ┌──────────────────────────────────────────────────────┐
                                          │  Qwen-Image-Edit-2509  (the MMDiT: contains Qwen2.5-VL)│
                                          │  does steps 4–7 INTERNALLY:                            │
                                          │  subject-id · component extraction · cross-ref align   │
                                          │  · relevance weighting · identity-consistent redraw    │
                                          └──────────────────────────────────────────────────────┘
                                                                     │
                                                          10. optional postprocess (upscale / anime refine)
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
- **Reference selection outputs indices, not descriptions.** Its worst-case
  failure is "picked a slightly worse image" (bounded), never "hallucinated the
  character" (catastrophic).

---

## 3. Concrete models (Hugging Face)

Target: single RTX 50-series / Blackwell GPU, ≤16GB VRAM. Stages run
**sequentially** (load → run → unload), never concurrently (see §6).

| Role | Model (HF) | Notes |
|------|------------|-------|
| **Edit backbone (core)** | `Qwen/Qwen-Image-Edit-2509` | Multi-image (1–3 optimal), person/text consistency, **native** depth/edge/keypoint control. This is the whole engine. |
| **Backbone, quantized for 16GB** | `nunchaku-tech/nunchaku-qwen-image-edit-2509` (SVDQuant **FP4**, r32) + a Lightning 4/8-step variant | FP4 for Blackwell. Pipeline: `QwenImageEditPlusPipeline` + `NunchakuQwenImageTransformer2DModel`. Multi-image via `image=[img1,img2,img3]`. Verify exact lightning filename on the org (weights churn). Escalation: 4-step → 8-step → full r32 → r128. |
| **Ref-selector VLM (narrow)** | `Qwen/Qwen2.5-VL-7B-Instruct` (or `-3B-Instruct`) | **Only** job: when >3 refs supplied, return the ≤3 best indices for the route. Optionally a one-word route hint. **Never** a character description. Anime-domain worries mostly evaporate here — "which image is the full-body front view" is robust even on anime. |
| Pose preprocessor | DWPose (via `IDEA-Research/DWPose`, or `controlnet_aux`) | Produces the keypoint control image for `pose_transfer`. Feeds 2509's native keypoint control. |
| Depth preprocessor | `depth-anything/Depth-Anything-V2-Large` | Depth control image for structured scene routes. Feeds 2509's native depth control. |
| Edge preprocessor | Canny (OpenCV) or `controlnet_aux` | Edge control image. Feeds 2509's native edge control. |
| Mask / segmentation | `facebook/sam2` (SAM2) | Region mask for `local_repair_or_inpaint` / regional outfit swap. |
| Region grounding (text→box) | `microsoft/Florence-2-large` | Turn "fix her left glove" into a box/region to seed SAM2. Image-domain output (a region), not a character description. |
| Upscale (postprocess) | Real-ESRGAN anime (`RealESRGAN_x4plus_anime_6B`) | Safe, stable anime upscaler. Step 10 only. |
| Anime polish (optional) | A current SOTA anime **SDXL** checkpoint (Illustrious-XL / NoobAI-XL family) as **low-denoise img2img refine** | Optional §10 finish for anime crispness. It is a *refiner*, never the identity carrier. Pick the current best checkpoint; this space moves fast. |

### Explicit anti-recommendations (these are the traps)

- **No GFPGAN / CodeFormer.** They restore *photoreal human* faces and will
  destroy anime faces. The pixai.md "face restoration" line is hallucination.
- **No IP-Adapter as a PixAI replacement.** 2509's multi-image path *is* the
  faithful identity mechanism. IP-Adapter is a generic anime-diffusion reflex.
- **No external character dossier, no deep nested JSON from the VLM, no
  line-protocol/free-text variants.** All were attempts to fix a bottleneck that
  must be removed, not fixed.
- **No hand-written component taxonomy or measurement/geometry extraction.** The
  model owns component understanding (also `AGENTS.md` §2).

### Anime domain (the one legitimate ChatGPT concern, correctly placed)

Qwen-Image is trained with heavy illustration data, so 2509 is a reasonable anime
editor, but not elite. If anime fidelity falls short, the lever is a **style LoRA
on the backbone** or the **§10 anime SDXL refine pass** — *not* a pipeline rewrite
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
| `pose_transfer` | pose keypoints | DWPose → 2509 keypoint control |
| `local_repair_or_inpaint` | region mask | Florence-2 (box) → SAM2 (mask) |
| `outfit_swap` | optional region mask | SAM2 if regional; else refs + instruction |
| `style_transfer` | style ref as image (+ opt. style LoRA) | — |
| `layout_or_sheet` | feed the sheet whole | — |
| `text_or_label_heavy` | none (2509 renders text well) | instruction carries the literal text |

Identity/portrait/view/normal-scene routes need **no** extra conditioning — the
backbone handles them. That is not "too simple"; it is the model doing its job.
Sophistication appears exactly where a route genuinely needs geometry.

---

## 5. Stage-by-stage data flow (what is text, image, latent)

| # | Stage | Input | Output | Domain |
|---|-------|-------|--------|--------|
| 1 | Parse instruction | raw text + envelope | `instruction_plan` | text |
| 2 | Route | `instruction_plan` | `route_plan` (route_kind + editor constraints) | text |
| 3 | Reference intake | all supplied ref images | ordered `reference1..N` handles | image |
| 3b | Select (only if N>3) | ref images + route | **≤3 indices** | indices |
| 8 | Conditioning build (route-gated) | selected refs + user structural sources | control image(s): pose/mask/depth | image |
| 9 | Generate | ≤3 ref images (+ control imgs) + **thin prompt** | edited image | latent→image |
| 10 | Postprocess (optional) | edited image | upscaled/anime-refined image | image |

The **prompt** at stage 9 is the user's instruction, lightly normalized (and, only
for scene/style, optionally expanded on the *scene* — never the character). It is
**not** VLM-authored from ref analysis.

---

## 6. 16GB VRAM execution plan

Qwen-Image-Edit-2509 is ~20B; Qwen2.5-VL-7B is ~16GB-class on its own. **They do
not coexist on 16GB.** Run stages sequentially:

1. (If N>3) load selector VLM → return indices → **unload**.
2. Build any control images with lightweight preprocessors (Depth-Anything /
   DWPose / SAM2 / Florence-2 — each load→run→unload).
3. Load Nunchaku FP4 backbone (+ Lightning) with CPU offload enabled → generate →
   unload.
4. (Optional) load upscaler/refiner → finish.

For the common case (≤3 refs, identity/portrait/view/scene) stage 1 and 2 are
skipped entirely: **load backbone, feed refs + instruction, generate.** That is
the first thing to make work.

---

## 7. Surgery on the existing repo

Keep the routing/conditioning scaffolding; excise the text-serialization organs.

**Keep (reuse as-is or lightly):**
- `character_instruction_*` — step 1.
- `character_task_router.py` / `character_task_route_models.py` — step 2. Its
  `ModelCapabilityRegistrySpec` / `FinalEditorConstraintsSpec` / reference budget
  are good.
- `character_conditioning_planner.py` / `_models.py` — step 8. Re-key its input
  from `visual_analysis` to `route_plan` + supplied structural sources.
- `character_reference_pack.py` **pack-building only** (`build_character_reference_pack`).
- `vlm_qwen.py` — keep the runner; **narrow** its use to index-selection.
- `diffusers_kontext_adapter.py`, `keyframe_*`, `lora_*` — separate tracks, leave.

**Gut (these are the disease):**
- `character_reference_observer.py` — the per-ref text observations. **Delete.**
- In `character_reference_pack.py`: the **identity dossier** parser
  (`_identity_parser_prompt`, `parse_character_reference_pack`) and the
  `visual_analysis` / `reference_semantics` / VLM-authored `edit_instruction`
  fields in `parse_character_edit_plan`. Replace `parse_character_edit_plan` with a
  thin **selector** that returns `selected_ref_indices` only.
- In `character_qwen_edit.py`: stop sourcing `prompt` from
  `qwen_vlm_edit_planner`. `prompt` = user instruction (normalized). `references`
  = selected indices.

**Add (small):**
- Preprocessor adapters that each emit a control *image*: `dwpose`, `depth_v2`,
  `sam2_mask`, `florence_region`, `canny`. Wire them into the conditioning
  planner's `planned_tools`.
- Nunchaku `QwenImageEditPlusPipeline` generation adapter that accepts
  `image=[...selected refs...]`, optional control image(s), and the thin prompt.

---

## 8. Build order (so Codex stops thrashing)

1. **Backbone smoke, no VLM.** ≤3 refs + literal instruction →
   `QwenImageEditPlusPipeline` (Nunchaku FP4, 4-step) → image. Prove the jillian
   close-up case renders with **zero** external character analysis. This is the
   milestone the whole current apparatus was failing to reach.
2. Wire the router (step 2) so `route_kind` selects the conditioning path.
3. Add the **index selector** VLM path, active only when N>3.
4. Add route-gated structural conditioning: `pose_transfer` (DWPose) first, then
   `local_repair` (Florence-2 → SAM2), then depth/edge for scenes.
5. Add §10 postprocess (upscale, optional anime refine).
6. Only then, if anime fidelity is short: backbone style LoRA.

Do not advance a rung until the previous one renders.

## 9. First smoke that must pass

```text
refs:        assets/characters/jillian/references/*   (≤3, fed directly)
instruction: "Character as shown in referenced images. Close-up face,
              looking to the left with neutral expression. Neutral
              lighting, white background."
route:       portrait_identity_generation
generation:  QwenImageEditPlusPipeline(image=[refs], prompt=instruction)
assert:      an image is produced; NO identity_profile.json, NO
             reference observations, NO visual_analysis, NO VLM-authored
             edit_instruction anywhere in the path.
```

## 10. Hard rules (do NOT)

1. **Never serialize the character to text.** No dossier, no `visual_analysis`,
   no per-ref observations, no VLM-authored `edit_instruction`.
2. The selector VLM returns **indices** (and at most a route hint). Nothing else.
3. No GFPGAN/CodeFormer. No IP-Adapter as identity mechanism. No deep nested JSON
   from any VLM. No line-protocol/free-text substitutes.
4. Structural conditioning is **image-domain** (pose/mask/depth maps), route-gated,
   optional. Never a text description of geometry.
5. Character-agnostic code stands (`AGENTS.md` golden rule). The reason it now
   *holds naturally*: there is no character text anywhere to hardcode.
6. Lower quality is allowed only from smaller/quantized models — never from
   removing a conditioning path that a route needs.
```
