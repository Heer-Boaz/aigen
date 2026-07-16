# Handoff: Jillian subject-LoRA v4 (FLUX.2 Klein 9B, NF4)

## Doel
Een personage-LoRA: `JSEED` ís Jillian (gezicht, haar, outfit, proporties).
De tekenstijl is bewust GEEN onderdeel van de trigger — stijl moet bij
inferentie promptbaar zijn (o.a. pixel art). Daarom staat in elke caption de
stijl expliciet benoemd (captioning-wet: wat je niet benoemt absorbeert de
trigger; wat je wél benoemt blijft een vrije prompt-as).

## Dataset (klaar, niets meer aan doen)
`assets/lora/JSEED/train/` — 34 beelden, metadata.jsonl:
- 14× canon "Black ink lineart with light watercolor shading." (v3-masters + crops)
- 10× "Flat cel-shaded anime coloring with bold clean outlines." (PixAI, door Boaz gekeurd)
- 10× "Black-and-white ink line art, uncolored, with sketch hatching." (PixAI, door Boaz gekeurd)
- Drie aspect-buckets via afgeleide crops: portrait-masters + vierkante
  waist-up/head-crops + landscape head-and-shoulders-crops.
- Trigger: kale `JSEED`, geen klasse-woord (besluit Boaz; hoofdletters — tokeniseert als J+SE+ED zonder "seed"-woordprior).
- Reproduceerbaar met `scripts/prepare_jillian_subject_lora_dataset_v4.py`
  (leest v3-masters + de PixAI-bestanden die al in train/ staan).
- Review sheet: `assets/lora/JSEED/review/train-set.png`.

## Startcommando
```bash
.venv/bin/accelerate launch tools/diffusers/train_dreambooth_lora_flux2_klein.py \
  --pretrained_model_name_or_path aigen/models/flux2/black-forest-labs/FLUX.2-klein-base-9B-training \
  --pretrained_text_encoder_name_or_path aigen/models/flux2/Qwen/Qwen3-8B-FP8 \
  --dataset_name assets/lora/JSEED/train --caption_column prompt \
  --instance_prompt "JSEED" \
  --output_dir runs/flux2_jillian_subject_lora_9b_nf4_v4 \
  --bnb_quantization_config_path runs/jillian_subject_lora_dataset_clean_v1/nf4-bf16.json \
  --cache_latents \
  --precomputed_cache_path assets/lora/JSEED/train_cache.pt \
  --aspect_ratio_buckets "1248,832;1024,1024;832,1248" \
  --text_encoder_out_layers 9 18 27 \
  --center_crop --rank 16 --lora_alpha 16 --train_batch_size 1 --gradient_accumulation_steps 1 \
  --gradient_checkpointing --mixed_precision bf16 \
  --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 100 \
  --optimizer adamW --use_8bit_adam --max_sequence_length 512 \
  --max_train_steps 3000 --checkpointing_steps 250 --checkpoints_total_limit 12 \
  --offload --seed 42 --skip_final_inference --report_to tensorboard
```

## Waarom `--text_encoder_out_layers 9 18 27` (NIET de default [10,20,30])
Vandaag gediagnosticeerd: de "zwakke distilled-overdracht" van v2 was een
conditioning-mismatch. v2 trainde op Qwen3-layers [10,20,30] (trainer-default),
maar de productieruntime (`aigen/generation/flux2_klein.py` →
`Flux2KleinPipeline._get_qwen3_prompt_embeds`) gebruikt de diffusers-default
(9,18,27). Met layers+aspect gecorrigeerd draagt v2-checkpoint-2500 identiteit
prima over op het snelle 4-step FP8-model (bewijs:
`runs/flux2_jillian_lora_te_layer_probe_v1`, `..._checkpoint_probe_v1`,
`..._aspect_probe_v1`). Door nu op (9,18,27) te trainen verdwijnt de mismatch
bij de wortel en werkt de LoRA op de runtime zónder speciale layer-knop.
Evalueer dus ook gewoon met de runtime-defaults.

## Evaluatieprotocol (goedkoop: ~3 s/render op de FP8-runtime)
- Checkpoint-sweep per 250; verwacht het optimum vóór 3000 (bij v2 generaliseerde
  2500 beter dan 3000 — houding/proporties).
- Matrix per kandidaat-checkpoint, weight 1.0, seeds 42/43:
  1. canon-stijlfrase × {832×1248, 1024×1024, 1248×832}
  2. cel-shaded-frase en lineart-frase (stijl-switch moet werken)
  3. pixel-art-frase ("pixel art, 16-colour palette, chunky pixels, hard
     outlines, flat fills") — verwachting: beter ontvlochten dan v2
     (baseline: `runs/flux2_jillian_lora_pixelart_bleed_probe_v1`, waterverf-
     schaduw lekte in de vulling)
  4. prompt zónder stijlfrase (welke stijl is de default?)
- Vergelijk proporties met `runs/jillian_subject_lora_dataset_v3/train/front-full.png`
  (~6 koppen) — niet met 1024²-renders beoordelen, aspect vertekent.

## Waarschuwingen
- NF4-route bewezen: ~5,8 s/step → 3000 stappen ≈ 5 uur. Geen FP8/TorchAO
  (`do_fp8_training`): traint niet werkbaar op deze kaart en bespaart geen
  geheugen; "GPU 100%" is daar geen bewijs van voortgang.
- Er draait nu niets op de GPU; geen andere trainer starten naast deze.
- Niets committen; Boaz commit zelf.
- Knoopjes-detail op het overhemd is een mogelijk v5-master-refresh (Boaz
  overweegt); niet blokkerend voor deze run.
