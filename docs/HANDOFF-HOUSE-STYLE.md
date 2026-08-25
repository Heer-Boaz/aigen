# House style handoff

## Assignment

Get every one of Boaz's characters drawn in one consistent house style — the
heavy ink line with pastel, watercolour-like colouring that Jillian is drawn
in. The immediate deliverable is a style LoRA trained on a corpus that does
not yet exist.

This document is a handoff for a fresh Claude, Codex, or human collaborator.
It records the objective, the live state on 2026-08-25, what is settled by
measurement, what is falsified, and what has been declined.

Do not start training or generation merely because this document was opened.
Agree the next GPU operation with Boaz first.

There is no test suite in this repository — commit `35f7dd0f` deleted the four
pix2pix test files before the current `main`. `pytest` collects zero tests. The
honest pre-flight is a smoke check: the eleven `aigen.pix2pix.*` modules import,
`compileall` passes on changed files, and `configs/pix2pix-*.json` parse.

## Read first

1. [`../AGENTS.md`](../AGENTS.md)
2. [`style-transfer.md`](style-transfer.md) — the measurements. **Its "What
   that makes the training target" section is a superseded objective**; it is
   banner-marked, but read the banner before acting on the section.
3. [`../runs/evidence/README.md`](../runs/evidence/README.md) — one line per
   evidence sheet, saying what each one proves.
4. [`../assets/lora/inkstyle/CORPUS_SPEC.md`](../assets/lora/inkstyle/CORPUS_SPEC.md)
   — what Boaz has been asked to produce.

## The two problems, held separate

Boaz split the work on 2026-08-14, and the split is load-bearing. Do not
re-merge them.

**Problem 1 — enforce a consistent drawing style.** Nothing to do with pixel
art. This is what matters for generating characters other than Jillian, and it
is the active work.

**Problem 2 — reduce an image to a pixel raster.** Explicitly *not* a
reimagining: "reduceren van pixels op een manier die doet denken aan de
MSX2-stijl", i.e. a filter. Output must be larger than 128×128, around
320×240. Boaz believes pix2pix can learn this. Not started.

## Standing constraints

These are Boaz's rules, not suggestions. Violating them has cost real time.

- **No GPU run without an explicit go.** Broad autonomy for this line of work
  does not extend to starting a 45-minute job.
- **Never `pkill` on a pattern** — Codex runs in parallel on this machine. Kill
  by PID only, then verify the VRAM came back.
- **Edit instructions state only the change**, never a keep-list.
- **Judgement is by eye.** Palette discipline and a perfect pixel raster are
  explicitly not criteria for output — "dat boeit mij geen reet". They matter
  only for training data.
- **The pipeline runs locally.** A cloud API is never a pipeline step; at most
  a one-off data source.
- **Hardcode explicit data** (captions, wardrobe) as tables. Do not write clever
  rule-based derivation: "je probeert het te slim te doen".
- Results must land somewhere Boaz can see them. `/tmp` is invisible to him;
  `runs/evidence/` is now tracked in git for exactly this reason.

## Settled by measurement

| finding | evidence |
| --- | --- |
| Local edit models treat a style instruction as a surface treatment and never move the body plan. 15/15 outputs held the input's proportions across Qwen 2511, FLUX.2 Klein and FLUX.2 dev, with and without "small". | `runs/sprite-words/`, sheets 02–03 |
| The same cloud model does the same thing on a comparable instruction, so **the variable is the wording, not the backbone**. | sheet 04 |
| `"Maak turn-around sheet"` moved the body plan to ~4 heads with no style word in it — template retrieval, not planning. | sheet 05 |
| No upscaler can do sprite→smooth. Three ESRGAN/DAT models plus VOSR all fail; VOSR has a generative prior and still fails because it is *fidelity-constrained* by design (`weak_cond_strength_aelq=0.2`, wavelet align). | sheets 06–09 |
| De-pixelation is **scale-dependent, not semantic** — the same model smooths pixel steps at 57×70 and preserves them at 181×241. | sheets 06–07 |
| FLUX.2 dev holds identity and viewing direction on information-poor input; Klein drifts. One seed per cell — a pattern, not a rate. | sheets 10–11 |
| The nine PixAI/ChatGPT references are drawn at 82–235 designed px tall, a factor 2.9. Resampling to a common height does **not** normalise proportions. | sheet 12 |
| **The ink style is promptable on the base model.** `Black ink line art with light watercolor-like coloring.` renders it with no LoRA at all. | sheet 15 |
| **JSEED v8 leaks Jillian's identity.** It puts her blue bow, brown skirt and blue knee socks — at seed 43 her leather jacket — onto a black-haired schoolboy. | sheet 15 |
| **USO normalises, it does not replicate.** 3/3 and 4/4 seed-consistent with identity preserved, but the reference's heavy ink did not transfer. Its SigLIP channel carries coarse texture, not the ink-vs-clean-line distinction. Disqualified as the corpus factory. | sheets 13–14 |
| Jillian's master figure measures **6.2 heads** — chin at y≈600 of a 3718 px figure. | measured 2026-08-25 |

### The one number that has actually been measured

`assets/lora/JSEED/masters/front-full.png` and
`assets/characters/jillian/views/front.png` are the **same drawing** on
different canvases — identical 3718 px figure blob, different file hashes. So
they cannot carry different head counts, and the two figures previously
recorded (6.5 in `style-transfer.md`, 5.8 in the evidence README) were the same
figure read twice, both by eye, both wrong. The measured value is **6.2**.

Everything else in the proportions table of `CORPUS_SPEC.md` is still an
eyeball off a 620 px strip, and the one figure since checked was low by 0.4.
Re-measure rather than trusting those numbers.

## Falsified — do not re-derive these

Each of these was believed, acted on, and killed by a measurement. They are
listed so a fresh agent does not spend the same time.

| claim | what killed it |
| --- | --- |
| The reference dimensions imply an SDXL backbone. | All nine are ~1.573 Mpx — PixAI's fixed upscale. Dimensions say nothing about the backbone. |
| The sprites came from image→image, confirming the surface-only law. | Boaz: ChatGPT got the reference and roughly "zet om in kleine pixel-art sprite". The law was broken, not confirmed. |
| The ~4-head prompt was "convert into a small pixel-art sprite". | A paraphrase. Both literal variants give 6.5 locally. **The real wording is still unknown.** |
| Upscalers have no generative prior. | VOSR has one. The real axis is fidelity-constrained vs free. |
| De-pixelation *is* the sprite→smooth operation. | Generalised from one 57×70 fragment; the vs-Qwen comparison disproved it. |
| FLUX.2 is 5× faster. | True of Klein (52 s), but Klein fails on low-res input. dev is ~200 s — 1.4× over Qwen's 274 s. |
| Fidelity matters, quality does not. | Boaz: quality is "bijzonder belangrijk". **Fidelity is a gate; quality is the objective.** |
| Specify a pixel height and proportions follow. | Sheet 12 row 2: `sheet8` still reads ~5 heads at 96 px. |
| A homegrown modal-run-length detector can find the native cell size. | Returns 2 for every image — PixAI's upscale destroys exact colour runs. Use `aigen pixel-art-fixer --mode fast`. |
| Colour counts describe the artist's palette. | 1456–13197 colours at 97–100% speckle. That measures upscale noise. Discarded. |
| The pink dress is off-style for gloss. | Wrong reject: it compared satin to leather. Compare **like material to like material**. Only the scarf man is a genuine technique outlier. |
| Two red refine tests are a known pre-existing breakage. | Both the tests and the signature mismatch are gone since `c48c7673`. The memory saying otherwise was deleted 2026-08-25. |

## Declined directions

- **Prompt sweep over game/scale semantics.** Stopped by Boaz on 2026-08-13 —
  it cannot settle anything while the original instruction is unknown.
- **A LoRA trained on the nine existing sprites.** Boaz: they "verschillen
  subtiel, maar fundamenteel, in stijl", so they cannot define one.
- **Generating several characters in one image to force consistency.** Boaz:
  "Er is geen model dat zo stoer is dat het meerdere personages netjes in 1
  stijl kan genereren op 1 output image."
- **USO as the corpus factory.** Disqualified by sheets 13–14 above.

## Problem 1: the ink-style LoRA

Five steps, agreed with Boaz. Step 2 is where the work is blocked.

1. **Authority image** — done. `assets/lora/JSEED/masters/front-full.png` is
   what "the style" means. `assets/reference-packs/jillian-inkstyle.json`
   points at it. Note reference packs take a *path*, not a name.
2. **Generate the corpus** — in Boaz's hands. He is producing characters in
   PixAI with the master as style reference. Drop location:
   `assets/lora/inkstyle/incoming/`, currently empty.
3. **Cull against the master**, in this order: fabric treatment (matte mottled
   wash with ink hatching; reject hard speculars) → proportions → no borrowed
   identity.
4. **Train the style LoRA** with inverted captions. Needs an explicit go.
5. **Pass/fail**: generate a character not in the corpus and check for Jillian
   leakage — the exact failure sheet 15 caught in JSEED v8.

### The caption inversion is the whole trick

`dataset-v15` (59 images, `metadata.jsonl`) **named the style and varied it**,
so style stayed promptable and identity was absorbed into the trigger:

```
JSEED. Black ink line art with light watercolor-like coloring. Full-body front
view with a neutral expression. Plain white background.
```

The style LoRA must do the exact inverse — **name the character and the
clothing, never the style**:

```
INKSTYLE. A girl of about seven with blond curly hair, in a pink frilled dress
and pink shoes. Full-body front view. Plain white background.
```

The governing law: *whatever is constant across the corpus gets absorbed by the
trigger, weighted by loss cost — wardrobe included, not just faces.* That is
why identity leaked in v8 and why the corpus must vary everything except the
drawing hand.

### Two traps in corpus building

**Do not prompt for random.** "Generate a random character, with random age,
random clothing" returns the model's *mode*: four of Boaz's first eight came
back in a jacket and cargo trousers. `CORPUS_SPEC.md` carries an explicit
sixteen-row wardrobe table for this reason.

**The 22 watercolour masters are only 18 distinct drawings.**
`three-quarter-upperbody-blue`, `warm_smile_portrait-blue`,
`warm_smile_portrait-pink` and `warm_smile_nearly_fullbody-pink.jpg` are
background recolours of poses already in the set — figure-difference of 15–30
out of 765 against their plain counterparts. Inside a cap of ~12 Jillian
images, taking four near-duplicates is a real dent in a corpus whose entire
purpose is variation. Count drawings, not files.

Target composition: 60–70 images over 16+ identities, Jillian at most ~12.
Breadth of identity matters more than images per identity. Views must be
varied **and captioned**, or the LoRA only ever draws full-body front.

## Problem 2: the pixel-reduction filter

Not started. Read [`HANDOFF-PIX2PIX.md`](HANDOFF-PIX2PIX.md) for the
implementation, but **be aware its objective is not this one**:

| | HANDOFF-PIX2PIX.md | Boaz's problem 2 |
| --- | --- | --- |
| operation | smooth anime → JRPG sprite, a reimagining | deterministic pixel reduction, a filter |
| output | native 128×128 | ~320×240, explicitly "groter dan 128x128" |
| ink/pastel style | excluded, "a separate existing FLUX.2 post-processing step" | the source style is the whole input |

Two live obstacles:

- **`aigen/pix2pix/config.py` hard-returns 128** for
  `NATIVE_DOWNSCALE_ARCHITECTURES`, and `MODEL_IMAGE_SIZES` is
  `{128, 256, 1024}`. Symmetric 256→256 works today; 320×240 is not reachable
  without a new architecture entry.
- **Registration.** U-Net skip connections need pixel-aligned pairs, which
  disqualifies original-Jillian↔sprite pairs: the very difference you want to
  learn is what makes the pair unusable. This is why problem 2 is framed as a
  deterministic filter — a reduction *can* be exactly registered where a
  redraw cannot.

The `bilinear_control.py` / asymmetric 1024→128 work merged on 2026-08-25 is
closer to this framing than the old handoff describes.

## State on disk

| path | what it is |
| --- | --- |
| `assets/lora/inkstyle/CORPUS_SPEC.md` | the brief Boaz is working to |
| `assets/lora/inkstyle/incoming/` | drop location, empty |
| `assets/lora/JSEED/masters/` | 59 files: 22 watercolour (18 distinct), 18 cel-shaded, 19 lineart |
| `assets/lora/JSEED/dataset-v15/` | 59 images + `metadata.jsonl`, the style-named captions to invert |
| `runs/flux2_jseed_subject_lora_9b_nf4_v8/` | the leaking identity LoRA, checkpoints 250–3000 |
| `runs/evidence/` | 18 sheets + `figs.py`, tracked in git via a carve-out in `.gitignore` |
| `runs/sprite-words/`, `runs/promptstyle/`, `runs/inkstyle/`, `runs/{klein,dev}-smooth/`, `runs/scaletest/` | the raw outputs the sheets were built from, **not** tracked |
| `loras/` | third-party pixel-art LoRAs, not tracked |

`figs.py` is the measuring tool: it isolates the largest non-background blob
and draws head-count rules. Read a number off a pixel ruler, not off a 620 px
strip — that is what produced the 5.8/6.5 confusion.

## Next actions

1. **Wait for images in `incoming/`.** Nothing downstream can start without
   them. The batch of eight Boaz has already generated exists only in chat
   history and is not on disk — ask for it.
2. When they land: measure each against the three acceptance criteria, and
   re-measure the proportions table properly while you have the files.
3. Build the dataset with an explicit per-image caption table. No rule-based
   derivation.
4. Train. **Explicit go required.**
5. Run the leakage test from step 5 above before declaring anything.

One experiment is proposed but unapproved: whether Qwen with the master as
`Picture 2` closes the family-to-hand gap better than words alone. One
character, four seeds, ~20 minutes. Sheet 15's base outputs are in the
*family* of the master, not its hand — finer line, sketchier, and the two
seeds differ. That gap is what a LoRA is meant to close, and the Qwen test
would say whether a reference image closes it more cheaply.
