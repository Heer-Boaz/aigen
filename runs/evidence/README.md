# Evidence for the sprite-style findings

Every claim in `docs/style-transfer.md` traces to one of these sheets. Read the
head-count ones the same way each time: figures are normalised to equal height
and ruled at **1/4 (red), 1/5 (blue), 1/6 (green)** of that height. Where the
chin lands is the head count. Nothing here is judged on grain or palette.

`figs.py` is the measuring script — it isolates the largest non-background blob
per image and draws the rules.

| file | what it shows | what to look at |
| --- | --- | --- |
| `01-reference-scatter.png` | the nine PixAI/ChatGPT Jillian references | the singles and `sheet4` sit on red (~4 heads), `sheet8` and the smooth turnaround sit near blue (~5). The set is not internally consistent. `spear` is invalid — the spear counts toward the height. |
| `02-sprite-words-small.png` | 9 local outputs, prompt "Convert into a **small** pixel-art sprite." | every chin on green (~6.5), same as the input. Qwen, FLUX.2 Klein and FLUX.2 dev all rasterise and none reproportion. |
| `03-sprite-words-plain.png` | 6 local outputs, same prompt without "small" | identical result. The word "small" is not the lever. |
| `04-chatgpt-control.png` | original / ChatGPT "low-resolution pixel art" / ChatGPT sprite ref / two local outputs | the ChatGPT pixel-art output sits on green like the local ones. Same cloud model, different instruction, same surface-only behaviour — so the variable is the instruction, not the backbone. |
| `05-turnaround.png` | original, ChatGPT pixel art, and the four views of the ChatGPT turn-around sheet | all four turn-around views sit on red (~4). `"Maak turn-around sheet"` moved the body plan with no style word in it. |
| `06-upscalers-lowres.png` | a 47×89 game sprite through nearest-x4 and the three ESRGAN/DAT upscalers | they de-pixelate into flat vector shapes; the face becomes a smear. |
| `07-upscalers-hd.png` | a 57×70 crop of HD pixel art, same treatment | here the same models smooth it into clean anime lineart. The effect is scale-dependent, not semantic. |
| `08-upscaler-vs-qwen.png` | sprite / Qwen redraw / upscaler, two characters | at native scale the upscaler barely changes the HD sprite and destroys the low-res one. It cannot replace the redraw. |
| `09-vosr-vs-qwen.png` | adds VOSR-1.4B-ms | VOSR has a generative prior and still fails: near-identical on HD, vertical streaking on low-res. It is fidelity-constrained by design. |
| `10-klein-vs-qwen.png` | sprite / Qwen 274 s / Klein 52 s | Klein matches Qwen on the HD sprite and drifts to pink-and-blonde on the low-res one. |
| `11-backends-four-way.png` | adds FLUX.2 dev | dev holds the sprite's purple where Klein did not. Robustness to information-poor input tracks model size. One seed per cell. |
| `12-canonicalised.png` | row 1: each reference at its true native pixel grid. row 2: all forced to 96 designed px tall on one 24-colour palette | **row 1 is the finding** — the references are drawn at 82, 91, 123, 128 and 235 designed pixels tall, a factor 2.9. Different sprite resolutions, not one style. **Row 2 refutes a hypothesis**: normalising the height did *not* normalise the proportions — `sheet8` still reads ~5 heads at 96 px, so proportions cannot be retrofitted by resampling. Row 2 also shows a badly designed palette step: a global median cut over all five muddied the blues and greyed the shirt, so it degrades rather than harmonises. How much a *well-built* shared palette buys is still unmeasured. |

Native cell sizes in sheet 12 come from `aigen pixel-art-fixer --mode fast`. A
homegrown modal-run-length detector returned 2 for every image, because PixAI's
fixed ~1.573 Mpx upscale destroys exact colour runs — use the tool.

## Caveats worth carrying

- Sheets 10 and 11 are **one seed per cell**. They show a pattern, not a rate.
- The head-count here is crown-to-chin. The DWPose figure of 4.52 heads recorded
  elsewhere for the same front view is crown-to-neck and is **not comparable**;
  mixing them makes a 4-head sprite look taller than its 6.5-head original.
- `runs/sprite-words/`, `runs/scaletest/`, `runs/klein-smooth/` and
  `runs/dev-smooth/` hold the originals these sheets were built from.

## Ink-style transfer (2026-08-18)

| file | what it shows | verdict |
| --- | --- | --- |
| `13-inkstyle-uso.png` | USO, style reference `assets/lora/JSEED/masters/front-full.png`, content Sinterklaas, 4 seeds | 4/4 seed-consistent, design fully preserved. **Confounded**: that Sinterklaas was already rendered in a related style, so the jump was small. |
| `14-inkstyle-offstyle.png` | same style reference against two genuinely off-style inputs — `AI46.png` (glossy cel shading on black) and the FLUX.2-dev Gradriel redraw — 3 seeds each | 3/3 seed-consistent per character and identity preserved, **but the reference's style did not transfer**. The reference has heavy black ink and hatched, washed colour; the outputs have thin lines and flat pastel fills. The two characters do not resemble each other stylistically beyond "clean anime". |

**Conclusion: USO normalises, it does not replicate.** Its SigLIP channel carries
coarse texture signatures — pixel art scored 4/4 earlier — but not the fine
distinction between heavy ink and a clean anime line, which is exactly the
difference that matters here. USO is therefore not usable as the corpus factory
for the illustration style: training on its output would teach a generic clean
anime style, not Jillian's.

| `15-promptable-style.png` | FLUX.2 Klein, caption phrase `Black ink line art with light watercolor-like coloring.` on two non-Jillian characters, with and without the JSEED v8 LoRA at w1.0, 2 seeds each | **The style is promptable.** The base column (no LoRA) already renders ink linework with light washed colour on white. **JSEED leaks identity**: `lora-boy` puts Jillian's blue bow, brown skirt and blue knee socks — and at seed 43 her leather jacket — onto a black-haired schoolboy. The wizard resists because a robed old man sits far from Jillian. Caveat: the base outputs are in the *family* of `masters/front-full.png`, not its hand — finer line, sketchier, and the two seeds differ. |

## House style: head count and fabric treatment (2026-08-18)

| file | what it shows |
| --- | --- |
| `16-headcount-ruler.png` | fine ruler (5.0 / 5.5 / 6.0 / 6.5 / 7.0 / 7.5) on the master and four PixAI-generated characters. **The ~5.8 this sheet reports for the master is wrong — the measured value is 6.2** (chin at y≈600 of a 3718 px figure, 2026-08-25). The error is an eyeball against rules on a 620 px strip; every other number on this sheet was read the same way and is probably low by a similar amount, though only the master has been checked. Adults in the new batch sit at ~6.5, the child at ~5.5 — a stylised band roughly one head below realism (7.5–8 for adults), not the 7.5 estimated by eye without a ruler. |
| `17-fabric-zoom.png` | garment patches at matched scale. The master's leather is **matte and mottled** with fine ink hatching. The scarf man's leather is the same material rendered **glossy**, with hard specular highlights and smooth gradients — technique, not subject. The pink satin dress is *within range*: its broad highlights are the material, and the fill still carries wash texture. |

**Cull method: compare like material to like material.** Judging a satin dress
against a leather jacket confuses subject with technique; that mistake produced
a wrong reject on the first pass.

**Proportion policy:** hold the *offset from realism* constant, not one head
count. Height only reads through head count in a full-body figure on white, so
fixing a single value makes every character read as the same height. The style's
signature is drawing about one head shorter than life.
