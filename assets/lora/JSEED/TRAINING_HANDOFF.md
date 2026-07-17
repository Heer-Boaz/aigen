# Handoff 2026-07-17: dataset-v15 + v8-training — sweep met weight-as

> Vervolgkoers ná deze handoff: 2D→3D via Pixal3D, plan in `docs/PLAN-PIXAL3D.md`.
> De v8-LoRA levert daar de identiteits-inputs (werkformule hieronder).

Vervangt de v10-sectie hieronder waar strijdig. Status bij overdracht: v8-training
hervat van checkpoint-2000 richting 3000 (~70 min), NF4 ~4,3 s/step.

## Vaststaand (met bewijs)

- **dataset-v13 + v7-run zijn besmet.** Bouwer verdeelde 59 masters modulo over
  alle 17 buckets: 26/59 verkeerde oriëntatie, subject-dekking tot 3× te klein.
  Bewijs: `assets/lora/JSEED/review/dataset-v13/manifest.json`. Ook
  `dataset-v13-cache.pt` niet hergebruiken.
- **dataset-v15 is de geldige dataset** (`scripts/build_jseed_dataset_v15.py`):
  59 masters 1:1, pad-only via randreplicatie naar de DICHTSTBIJZIJNDE bucket
  (max 0,9% padding, 3 buckets: 832,1248;1184,880;1248,832), niet-circulaire
  framing-validatie, side-by-side-review in `review/dataset-v15/`. v14 =
  identieke pixels met 3 oude captions; mag weg na akkoord Boaz.
- **v8-training** (`runs/flux2_jseed_subject_lora_9b_nf4_v8`): op v15, TE-layers
  9 18 27, checkpoints per 250, log (append) in
  `runs/flux2_jseed_subject_lora_9b_nf4_v8-launch.log`; het volledige
  accelerate-commando staat in de laatste sectie hieronder + `--resume_from_checkpoint latest`.

## Kernbevinding tussenevaluatie (runs/jseed_lora_v8_eval_v1/sweep/)

- v8-2000 ≈ v7-3000 bij weight 1.0: framing/outfit/kleuren goed (datasetfix
  werkt), maar generiek gezicht en ~6-koppenbouw i.p.v. 4,5.
- Vroege checkpoints v7/v8 vrijwel identiek (beide renderen "JSEED." als
  lettertekst t/m stap 750) → bottleneck is NIET de dataset.
- **Weight-sweep checkpoint-2000 (seed 42): w1.0 → 1.25 → 1.5 beweegt monotoon
  richting master** (blauwe strik terug, groter hoofd, gedrongener bouw,
  grotere ogen), zonder artefacten. Identiteit zit dus in de LoRA maar staat
  bij w1.0 te zacht t.o.v. de distilled 4-staps-prior. NF4-precisie en dataset
  als oorzaak ontkracht. (w1.75-render faalde; oorzaak nog niet onderzocht.)

## Stand opvragen

`scripts/jseed_v8_status.sh` — print of het trainingsproces loopt, de huidige
stap/ETA uit de log, de laatste checkpoints, logfouten en GPU-gebruik.
`scripts/jseed_v8_pause.sh` — pauzeert netjes: wacht op de eerstvolgende
checkpoint-wegschrijving en stopt alleen het v8-proces (met `--nu`: direct,
verliest de stappen sinds het laatste checkpoint).
`scripts/jseed_v8_resume.sh` — hervat vanaf het laatste checkpoint.
Alle drie werken vanuit elke shell, onafhankelijk van de sessie die de
training startte.

## Sweep-protocol na stap 3000

1. Checkpoints 2000/2250/2500/2750/3000 × weights 1.0/1.25/1.5 × seeds 42/43,
   canon-prompt, 832×1248 (`.venv/bin/aigen flux2-klein`, ~60 s/render).
   Bestaande renders staan al in `runs/jseed_lora_v8_eval_v1/sweep/`.
2. Beste kandidaat daarna: stijl-switch (cel-shaded/lineart), expressie-switch,
   niet-witte achtergrond ("Plain blue/pink background"), prompt zonder
   stijlfrase, drie aspecten.
3. Proporties meten (DWPose + silhouet) tegen `masters/front-full.png`;
   doel 4,52 koppen.
4. Als w1.5 niet volstaat: hogere lora_alpha (32) of lr in een vervolgrun
   overwegen; TE is frozen — triggertoken-gedrag ("JSEED" als letterwoord)
   meenemen. GEEN datasetwijzigingen zonder nieuw bewijs.

## Werkformule checkpoint-3000 (audit 2026-07-17, renders in runs/jseed_lora_v8_eval_v1/checkpoint-3000-audit/)

`JSEED` + outfit-tags + stijlfrase, weight 1.0–1.25. Voorbeeld dat alles
tegelijk goed doet (outfit, stijl, scène): "JSEED drinking coffee in a cafe in
Paris. Black ink line art with light watercolor-like coloring. short brown
hair, blue eyes, white collared shirt tucked in, blue ribbon bow, open brown
leather jacket, brown leather skirt, brown leather gloves, blue thighhighs,
kneehigh brown boots."

- Kale trigger in vrije scènes → onderlijf defaultt naar jeans (dataset heeft
  alleen studio-poses); outfit-tags lossen dit op.
- Weight ≥1.5 lekt kleuren in scènes (blauwe handschoenen/benen); de
  weight-hendel is alleen voor de canon-pose nuttig.
- Zonder stijlfrase → realistische vrouw (ontvlechting werkt zoals ontworpen).
- "Friends" worden klonen; bijfiguren expliciet anders beschrijven.
- Pixel-art-stijlen dragen de identiteit vlekkeloos; korrelbudget NIET met
  woorden sturen maar met het budget-als-beeld-recept.
- Proporties (4,52 koppen) blijven het zwakke punt in detailstijlen; fix zit
  in een vervolgrun (lora_alpha 32), niet in prompts of weight.

## Pixel-art: NIET via flux2-klein-prompts

Audit 2026-07-17: korrelbudget stuurt niet via woorden ("tiny"/"NES"/"Saturn"
geven dezelfde HD-korrel) en ook niet via het 3-referenties-recept op
flux2-klein met een instructie-prompt — de identiteits-ref-korrel domineert
daar (bewijs: `checkpoint-3000-audit/pixel-budget-recept-{32,40}px.png`).
Het bewezen recept draait op de LightX2V/Qwen-Image-Edit-route (andere venv,
zie `runs/pixel_budget_test/budget_32/result.json` voor de volledige config)
met de prompt "Redraw this as clean pixel art of the same character." en het
spec-beeld als beeldinvoer. Enige wijziging voor het vervolg: de
identiteits-invoer vervangen door een LoRA-render (bijv.
`checkpoint-3000-audit/w1.0-seed5-tags-plus-stijl_pixelart.png` of een
canon-pose-render).

## Discipline

- Nooit `pkill` op patroon — alleen de eigen taak/PID stoppen.
- Niets committen; Boaz commit zelf. Masters zijn heilig.
- Stop/hervat alleen direct ná een checkpoint-wegschrijving
  (`--resume_from_checkpoint latest`).

---

# Handoff: JSEED dataset v10 — klaar voor training

Datum: 2026-07-16. Vervangt de v9-afkeuring (evidence blijft staan, zie onder).

## Wat er is veranderd

De destructieve segmentatieroute is geschrapt en vervangen door
`scripts/build_jseed_dataset_v10.py`:

- **Bronownership (vraag 1): opgelost.** Alle 52 goedgekeurde masters staan in
  één canonieke map `assets/lora/JSEED/masters/` (19 gepromoveerd uit train/,
  33 uit uncleaned/). De bouwer leest uitsluitend die map, via een expliciete
  tabel van 52 rijen met per master de volledige caption als literal.
- **Achtergrondvariatie (vraag 2): geschrapt.** De achtergrond staat in élke
  caption benoemd ("Plain white background") en blijft daardoor promptbaar
  (captioning-wet). Of dat volstaat is ná de training goedkoop te testen met
  één prompt op een andere achtergrond; alleen bij falen is variatie nodig.
- **Niet-destructief by construction (vragen 3-5):** de bouwer padt elke
  master uitsluitend met de eigen randkleur (mediaan van de randpixels) tot de
  dichtstbijzijnde officiële FLUX.2-bucketverhouding en schaalt dan Lanczos
  naar exact die bucket. Geen alpha, geen masker: onderwerppixels kúnnen niet
  beschadigen. Randen waar de figuur is afgesneden (portret-onderkant) krijgen
  geen padding (anker), dus geen zwevende banden. Mocht achtergrondvariatie
  ooit nodig zijn: flood-fill vanaf de canvasrand (alleen met de buitenrand
  verbonden bijna-witte pixels vervangen) behoudt wit binnen gesloten
  inktcontouren exact — maar dat is nu niet nodig.
- **Geen kunstmatige inflatie.** v9 blies 52 masters op tot 416 beelden
  (8 duplicaten per master in andere buckets/kleuren) — nul nieuwe informatie;
  herhaling komt al uit de epochs. v10 = 52 unieke beelden, 1:1 met de masters.

## Dataset

- `assets/lora/JSEED/dataset-v10/` — 52 beelden + metadata.jsonl
- Review: `assets/lora/JSEED/review/dataset-v10-contact-sheet.png` en
  `dataset-v10-manifest.json` (per beeld: bron, maat, padkleur, bucket).
- Steekproef op 100% gecontroleerd (o.a. `portrait-lookleft-lineart.png`):
  haar en lineart-vulling intact.
- Captions: kale `JSEED` + stijlfrase + inhoud/expressie + achtergrond.
  Drie stijlen; expressies (determined/upset/warm smile) benoemd.

## Startcommando

```bash
.venv/bin/accelerate launch tools/diffusers/train_dreambooth_lora_flux2_klein.py \
  --pretrained_model_name_or_path aigen/models/flux2/black-forest-labs/FLUX.2-klein-base-9B-training \
  --pretrained_text_encoder_name_or_path aigen/models/flux2/Qwen/Qwen3-8B-FP8 \
  --dataset_name assets/lora/JSEED/dataset-v10 --caption_column prompt \
  --instance_prompt "JSEED" \
  --output_dir runs/flux2_jseed_subject_lora_9b_nf4_v6 \
  --bnb_quantization_config_path runs/jillian_subject_lora_dataset_clean_v1/nf4-bf16.json \
  --cache_latents \
  --precomputed_cache_path assets/lora/JSEED/dataset-v10-cache.pt \
  --aspect_ratio_buckets "832,1248;1184,880;1248,832" \
  --text_encoder_out_layers 9 18 27 \
  --center_crop --rank 16 --lora_alpha 16 --train_batch_size 1 --gradient_accumulation_steps 1 \
  --gradient_checkpointing --mixed_precision bf16 \
  --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 100 \
  --optimizer adamW --use_8bit_adam --max_sequence_length 512 \
  --max_train_steps 3000 --checkpointing_steps 250 --checkpoints_total_limit 12 \
  --offload --seed 42 --skip_final_inference --report_to tensorboard
```

Nieuw datasetpad, nieuwe cache, nieuwe output — v5-run niet hervatten.
Beelden zijn al exact op bucketmaat; de trainer hoeft niets meer te croppen.

## Evaluatieprotocol (runtime-defaults, ~3 s/render)

- Checkpoint-sweep per 250; optimum lag bij v2 vóór het einde (2500 > 3000).
- Per kandidaat, weight 1.0, seeds 42/43: canon-frase × drie aspecten;
  stijl-switch (cel-shaded / lineart / pixel-art-frase); expressie-switch
  ("with a warm smile" vs "determined"); prompt zonder stijlfrase (default?);
  én één prompt met een niet-witte achtergrond (test van vraag 2).
- Proporties vergelijken met `assets/lora/JSEED/masters/front-full.png`.

## Niet hergebruiken (v9-erfenis, niets verwijderd)

- `assets/lora/JSEED/train/` — legacy (oude v4-crops, stale metadata,
  `rejected-jseed-dataset-v9/`-archief); mag na akkoord van Boaz weg.
- `assets/lora/JSEED/uncleaned/` — leeg na promotie; mag weg.
- `runs/jseed_dataset_v5_preview_v9/...cache.pt`, `runs/flux2_jseed_subject_lora_9b_nf4_v5/`
  (checkpoints 250/500: getraind op kapotte data).
- `scripts/prepare_jillian_subject_lora_dataset_v4.py` — vervangen door
  `scripts/build_jseed_dataset_v10.py`.

## Waarschuwingen

- NF4-route bewezen: ~4,3-4,8 s/step, ~13,8 GiB VRAM → 3000 stappen ≈ 4 uur.
- Geen FP8/TorchAO-training; "GPU 100%" is geen bewijs van voortgang.
- Niets committen; Boaz commit zelf.
