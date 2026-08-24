# Sprite style: what local models will and will not change

The goal is a character redrawn in a sprite art style. The criterion is
**body proportions**, measured as head-count: overlay horizontal rules at 1/4,
1/5 and 1/6 of the figure's height and read where the chin lands. Palette,
grain, edge hardness and rasterization are explicitly *not* the criterion —
that is why this is called sprite style and not pixel-art style.

Jillian's own references measure ~4 heads. Her original character sheet
measures ~6.5. Everything below is about closing that gap.

## The finding

**Local edit models treat a style instruction as a surface treatment. They
change the raster and leave the skeleton frozen.**

The cleanest evidence is `runs/sprite-words/`. One input
(`assets/characters/jillian/views/front.png`, ~6.5 heads), one minimal
instruction, no keep-list, no LoRA, three seeds per cell:

| cell | backend | prompt | heads out |
| --- | --- | --- | --- |
| `qwen-small` | qwen-image-edit-2511-base | Convert into a small pixel-art sprite. | ~6.5 |
| `qwen-plain` | qwen-image-edit-2511-base | Convert into a pixel-art sprite. | ~6.5 |
| `klein-small` | flux2-klein | Convert into a small pixel-art sprite. | ~6.5 |
| `klein-plain` | flux2-klein | Convert into a pixel-art sprite. | ~6.5 |
| `dev-small` | flux2-dev-nvfp4 | Convert into a small pixel-art sprite. | ~6.5 |

Fifteen of fifteen produced competent pixel art at the input's original
proportions. Not one chin moved. Two backbone families, two prompt lengths,
and the word "small" present or absent make no difference.

A corroborating accident: on a sprite input, `"light watercolor-like
coloring"` recoloured the figure and left the pixel grid completely intact
(`runs/sprite-to-ink-v2`). Surface-only is the default mode of operation, not
a failure of a particular prompt.

## The cloud model does the same thing on the same instruction

It is tempting to conclude the local backends lack a capability. They do not.

A ChatGPT conversation, supplied by Boaz on 2026-08-13, produced a pixel-art
Jillian from the same original sheet with the instruction **"Zet dit om in
low-resolution pixel art"**. That output measures **~6.5 heads** — identical
to the input and proportionally indistinguishable from the fifteen local
outputs above. Same cloud model that produced the 4-head sprites, different
instruction, same surface-only behaviour as Qwen and FLUX.2.

So the variable that moves proportions is the **instruction**, not the
backbone.

**The instruction that produced the ~4-head sprites is not known.** It was
recalled approximately as "zet om in kleine pixel-art sprite", but that was a
paraphrase; the two literal variants derived from it ("Convert into a small
pixel-art sprite." / "Convert into a pixel-art sprite.") both return ~6.5 heads
locally. Finding the real wording — from the ChatGPT history — collapses this
question faster than any sweep. Failing that, the open experiment is a prompt
sweep over game/scale semantics ("Game Boy sprite", "RPG character sprite", an
explicit small pixel dimension) measured on head-count.

Note also that the same ChatGPT transcript contains a long technical
explanation of why local models supposedly cannot do this — checkpoints,
LoRAs, ControlNet, denoise ranges — which the model then retracted as
invented. Its claims do not match measurement here: the local outputs are
clean pixel art, not a crude filter, and the high-res-then-nearest-neighbour
downscale it prescribes is already covered deterministically by
`pixel-art-wu`. Use the image from that conversation as evidence; do not mine
the explanation.

## What that makes the training target

Not style. The surface transfer already works — locally and in the cloud, at
the same fidelity. The missing piece is one geometric association: that the
instruction licenses redrawing the body plan.

Training is the reliable route, but it is no longer proven to be the *only*
one: since the cloud model's behaviour also hinges on wording, a local prompt
that carries the same semantics may exist. Run the prompt sweep before
committing GPU time to training.

The training pairs already exist and were mistaken for references all along:

- **before**: `assets/characters/jillian/views/front.png` and the other
  original views, ~6.5 heads
- **after**: `assets/characters/jillian/references/*.png`, the ChatGPT sprites,
  ~4 heads, same character and outfit

Two cautions measured on that set:

1. **The references are not internally consistent.** Nine images span ~4 to ~5
   heads. The 4-head cluster (three single renders plus the four-view sheet)
   sits closest to generation-0; the 5-head ones are drift accumulated across
   successive PixAI-edit generations from the original seed. Prefer the 4-head
   cluster as canon.
2. **Synthetic "before" images carry the generator's bias.** `runs/scaletest/`
   showed a sprite can be converted to a smooth illustration to manufacture
   the missing before-side for sprite-only corpora (Princess Crown / Gradriel,
   ~758 frames). It ran 3/3 on Jillian's own high-resolution sprites — but
   those are the character whose real pairs already exist, so they are not
   where the route earns anything. On another character (Gradriel, low-res
   frames) it ran **2 of 4**. That is the ratio to plan against.
   Those inputs are also Qwen-shaped, so training on them mostly teaches
   "Qwen-smooth → sprite". The nine real pairs are the only in-distribution
   ones. Gradriel should stay a minority ingredient, not the bulk.

**Fidelity is a gate; quality is the objective.** A pair has to depict the same
thing, so an output that tilts the viewing direction or replaces a piece of
clothing is rejected however good it looks — a subtle change of that kind makes
the quality worthless, because the pair now teaches the wrong mapping. But
among the outputs that pass the gate you want the *best* one, since that is
what enters the dataset. Quality is not a lesser criterion; it is the one you
optimise once fidelity holds.

On the single-seed comparison (Boaz, 2026-08-17) FLUX.2 dev retained identity
and viewing direction best, and Klein drifted outright on information-poor
input. **No quality ranking between dev and Qwen is established** — one seed per
cell cannot support one, and Qwen's apparent polish came with more invention of
its own on that seed.

That places the three backends on one axis. VOSR is pinned so hard it cannot
redraw at all; Qwen is loose enough to invent; **dev redraws while staying
anchored**, which is the wanted position for pair generation.

| backend | s/image (batched) | holds identity on low-res | note |
| --- | --- | --- | --- |
| flux2-klein | 52 | no — went pink/blonde on a purple sprite | fine on HD input |
| flux2-dev-nvfp4 | ~200 | yes | best identity and viewing-direction retention |
| qwen-image-edit-2511 | 274 | yes | most invention of its own on this seed |

One seed per cell — a pattern, not a rate, and not a quality verdict.

Filtering the synthesized pairs should not need judgement. Candidate rejection
rules: background no longer plain; aspect or composition changed; figure
palette shifted (would catch the frame whose hair went blonde); **viewing
direction flipped or rotated** — measurable from DWPose shoulder/hip keypoint
x-order via `aigen/dwpose_control.py`, comparing source against output; output
too close to the input (teaches nothing). **None of these are implemented.** The
2-of-4 above is an eyeball pass on same-character/same-pose, not a measured
filter yield — do not size a large run on it until the rules exist and have
been run.

Corpus size: the 758 Gradriel frames deduplicate to 447 / 356 / 290 / 239
distinct frames at perceptual thresholds 0.04 / 0.06 / 0.08 / 0.10 — roughly
13–25 h on dev rather than 42. Halved, not transformed, and still one character
in one suit of armour.

## Upscalers cannot do the sprite-to-smooth step

Tested 2026-08-16 on two inputs at their native pixel grid — Jillian's HD
sprite reduced to 184×244, and a Gradriel frame at 68×92 — all at x4:

| model | kind | HD pixel art | low-res sprite |
| --- | --- | --- | --- |
| `animesharp-x4`, `illustrationjanai-esrgan`, `illustrationjanai-dat2` | ESRGAN / DAT regression | near-identical to input | vectorised, face destroyed |
| `VOSR-1.4B-ms` | latent diffusion SR | near-identical to input | vertical streaking, unusable |
| qwen-image-edit-2511 | generative edit | clean anime illustration | clean anime illustration |

The distinction that matters is **not** regression versus generative. VOSR is a
DiT with 25 steps, CFG and a seed — it has a generative prior and still cannot
do it, because it is *fidelity-constrained*: `weak_cond_strength_aelq=0.2` and
`align_method="wavelet"` re-impose the input's low-frequency content on the
output by design. A super-resolution model exists to add plausible
high-frequency detail without touching composition. Redrawing is the thing it
is built not to do.

The ESRGAN family's de-pixelating effect is also **scale-dependent, not
semantic**: at 57×70 it reads the pixel steps as artifacts and smooths them, at
181×241 it reads the same steps as detail and keeps them. It does not know it
is looking at pixel art.

Consequence: the ~5 min/image generative edit is the only route to a smooth
counterpart, so the pix2pix registration problem stays open — what aligns does
not redraw, and what redraws does not align.

Note `aigen/models/upscale_models/amd/realesrgan-x4plus-anime-6b/` holds only a
README and licence; the weights were never fetched. `realesrganX4plusAnime_v1.pt`
sits at the root of that directory and is not registered in `UPSCALE_MODELS`.

## USO: a real style channel, but content-preserving

USO (ByteDance, FLUX.1-dev) is the only local backend with an architectural
style path: `references[0]` is content (VAE, spatial,
`USO_CONTENT_REFERENCE_SIZE = 512`), `references[1:]` go through a separate
SigLIP vision encoder and projector into cross-attention, never touching the
text encoder (`aigen/generation/uso_flux1_worker.py`). `USO_MAX_REFERENCES = 3`.

It is genuinely consistent where prompt-only style transfer is not — 4/4 and
3/3 seeds on-style, including a completely different character through
Jillian's style reference. But it preserves the content image's geometry by
construction, so it returns the source's proportions unchanged. For the
criterion in this document it scores zero. Use it for palette and shading
transfer, not for body plan.

## The pixel budget is deterministic

Rasterization needs no model. `aigen pixel-art-fixer` measures the reference's
native grid; on the Jillian reference it reports `step_x 5.83 / step_y 5.98`
on 1536×1024. `aigen pixel-art-wu --cell-size 9` then enforces it. This step
was never the problem.

## Measuring

`figs.py` (scratch) isolates the largest non-background blob per image and
stacks figures normalised to equal height with the 1/4, 1/5, 1/6 rules drawn
on. Write the head count down per cell *before* forming an opinion about how
the raster looks — every wrong conclusion recorded above came from judging
grain instead of geometry.
