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
