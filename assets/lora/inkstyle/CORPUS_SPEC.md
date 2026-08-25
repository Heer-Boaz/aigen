# Ink-style corpus spec

Target: a style LoRA that makes Jillian's ink-and-wash illustration style the
model's **default** rather than a request. Style is already promptable on the
base model (`runs/evidence/15-promptable-style.png`); what training buys is a
smaller spread and the last step from *family* to *hand*.

Authority image: `assets/lora/JSEED/masters/front-full.png`.

## The one rule

**Everything except the drawing style must vary.** Whatever is constant across
the corpus gets absorbed by the trigger. That is not limited to faces — if most
characters wear utility jackets, the trigger learns "utility jacket".

Corollary for captions, and it is the inverse of `dataset-v15`: **name the
character and the clothing, never name the style.**

```
INKSTYLE. A girl of about seven with blond curly hair, in a pink frilled dress
and pink shoes. Full-body front view. Plain white background.
```

`dataset-v15` named the style and varied it, so style stayed promptable and the
identity — never named — was absorbed. Here the roles swap.

## Do not prompt for "random"

A model's idea of random is its own mode. The first batch of eight asked for
random clothing and returned **four jacket-plus-cargo-trouser characters**. Use
an explicit list instead, one line per character (see
[[explicit-data-over-clever-derivation]] as applied elsewhere in this repo).

## Acceptance, in order

1. **Fabric treatment** — matte, mottled wash with fine ink hatching. Reject
   hard specular highlights and smooth gradients. Judge like material against
   like material: a satin dress legitimately has broader highlights than
   leather. Reference: `runs/evidence/17-fabric-zoom.png`.
2. **Proportions** — hold the offset from realism, roughly one head below life,
   rather than a single head count.

   **Only one figure here is measured.** `masters/front-full.png` reads
   **6.2 heads**: chin at y≈600 of a 3718 px figure, read off a 10 px ruler.
   Everything else in this table was eyeballed against
   `runs/evidence/16-headcount-ruler.png`, and that sheet under-read the one
   figure since checked by 0.4 (it said 5.8). Treat the corrected column as a
   working estimate, not a measurement.

   | subject | as read on sheet 16 | corrected (+0.4) |
   | --- | --- | --- |
   | young child | 4.5–5 | 4.9–5.4 |
   | older child | 5–5.5 | 5.4–5.9 |
   | young woman (Jillian) | 5.8 | **6.2, measured** |
   | adult man | 6.3–6.5 | 6.7–6.9 |
   | tall man | 6.8–7 | 7.2–7.4 |

   The +0.4 assumes one constant reading bias across the sheet, which is
   plausible but untested. Re-measure each character once the images are on
   disk in `incoming/`; the batch they came from lives only in chat history.

3. **No borrowed identity** — reject anything wearing Jillian's outfit (brown
   leather jacket + blue bow + brown skirt + blue socks) on another face.

## Composition

Target 60–70 images over 16+ identities. Breadth of identity beats images per
identity: 16 characters × 3 beats 6 × 8.

Jillian: at most ~12 of the total, drawn from the 22 watercolour masters in
`dataset-v15`. She is the ground truth for the style and also the largest
leak risk, so name her appearance and outfit in full in every caption.

Views: mostly full-body front, but include some three-quarter, side and
upper-body — and caption each, reusing the `dataset-v15` vocabulary
("Full-body front view", "Left-facing three-quarter view", "Waist-up view").
Vary and name it, or the LoRA will only ever draw full-body front.

## Characters still needed

Six from the first batch are keepers (blond boy in shorts, girl in pink dress,
woman in olive field jacket, young man in denim, man with backpack, boy in
dungarees) — but three of those are utility wear, so the list below deliberately
avoids it.

| # | character | wardrobe |
| --- | --- | --- |
| 1 | elderly woman, grey hair in a bun, stooped | knitted cardigan, long wool skirt, slippers |
| 2 | middle-aged man, balding, heavy build | three-piece suit and tie |
| 3 | teenage girl, black hair in twin tails | sailor-style school uniform |
| 4 | young woman, dark skin, braided hair | light summer dress, bare arms, sandals |
| 5 | muscular man, shaved head | tank top and running shorts |
| 6 | woman in her forties, glasses | medical scrubs and clogs |
| 7 | heavyset man, moustache | chef's whites and apron |
| 8 | old fisherman, weathered face | thick cable-knit sweater, oilskin trousers |
| 9 | toddler, chubby | striped pyjamas, bare feet |
| 10 | teenage boy, lanky, freckles | basketball kit, bare arms |
| 11 | woman, tall and slim | floor-length evening gown |
| 12 | construction worker, stocky | hi-vis vest, helmet, work trousers |
| 13 | ballet dancer, very slim | leotard and tights |
| 14 | businesswoman, east-asian, short hair | tailored blazer and pencil skirt |
| 15 | farmer, sun-tanned, wiry | rolled shirtsleeves, waistcoat, flat cap |
| 16 | girl of about ten, wheelchair user | denim jacket, leggings |

Prompt shape, one per line, explicit rather than random:

```
Generate a different character: an elderly woman with grey hair in a bun, in a
knitted cardigan and long wool skirt, full-body front view, plain white
background, in the same art-style as the style reference.
```

## Handing images over

Drop files in `assets/lora/inkstyle/incoming/`. Any filename. The caption table
gets written per image as an explicit table — no rule-based derivation.
