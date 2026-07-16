# Handoff aan Claude: JSEED-dataset v9 is afgekeurd

Datum: 2026-07-16

## Kort oordeel

De afgekeurde dataset is als bewijsmateriaal gearchiveerd in:

`assets/lora/JSEED/train/rejected-jseed-dataset-v9/`

Deze set is niet bruikbaar voor training. De eerdere goedkeuring was fout. De
compositiestap heeft bij veel beelden echte delen van het haar verwijderd en
tegelijk witte bronachtergrond of witte randen rond het haar behouden. Volgens
de volledige visuele controle is dit geen incident bij een paar bestanden:
vrijwel de hele samengestelde set moet als afgekeurd worden behandeld.

De FLUX.2-LoRA-training is gestopt. Er draait geen trainer meer en de GPU is
vrij. De run bereikte ongeveer stap 520; checkpoint 500 is volledig opgeslagen,
maar is semantisch ongeldig omdat het op de kapotte dataset is getraind.

Niet hervatten en niet hergebruiken:

- `assets/lora/JSEED/train/rejected-jseed-dataset-v9/train/`
- `runs/jseed_dataset_v5_preview_v9/flux2_klein_base_9b_jseed_17bucket_cache.pt`
- `runs/flux2_jseed_subject_lora_9b_nf4_v5/checkpoint-250/`
- `runs/flux2_jseed_subject_lora_9b_nf4_v5/checkpoint-500/`

Er is niets verwijderd.

## Wat daadwerkelijk als bron is gebruikt

De datasetbouwer is:

`scripts/prepare_jillian_subject_lora_dataset_v4.py`

Hij leest twee bronmappen:

- `assets/lora/JSEED/train/`: momenteel 40 bestanden, waarvan de expliciete
  selectie in het script 19 masters gebruikt;
- `assets/lora/JSEED/uncleaned/`: 33 bestanden, alle 33 als master gebruikt.

De nieuwe PixAI-beelden zijn dus technisch wel meegenomen. De training gebruikte
niet alleen oude data. De bronownership was desondanks verkeerd: goedgekeurde
masters bleven in `uncleaned/` staan en de bouwer las rechtstreeks uit zowel de
canonieke als tijdelijke bronmap. Daardoor was niet in één oogopslag vast te
stellen wat de echte goedgekeurde trainingsbron was.

Het manifest bevestigt:

- 52 unieke masters;
- 19 uit `assets/lora/JSEED/train/`;
- 33 uit `assets/lora/JSEED/uncleaned/`;
- 8 varianten per master;
- 416 gegenereerde trainingsbeelden.

Zie
`assets/lora/JSEED/train/rejected-jseed-dataset-v9/manifest.json`.

## Waar de beeldschade ontstaat

De originele bronbeelden die bij de onderstaande voorbeelden horen hebben
intact haar. De schade wordt dus door de datasetvoorbewerking geïntroduceerd.

De actieve route in `scripts/prepare_jillian_subject_lora_dataset_v4.py` is:

1. `AnimeForegroundSegmenter.segment_image(...)`;
2. een harde binaire grens van `>= 0.90`;
3. alpha uitsluitend als volledig transparant of volledig opaak;
4. bijsnijden tot de alpha-bounding box;
5. schalen met Lanczos en compositen op een effen achtergrond.

Die route kan in deze dataset niet betrouwbaar onderscheiden tussen:

- witte achtergrond en witte delen die bij het onderwerp horen;
- dunne bruine/zwarte haarstrengen en achtergrond;
- lichte pixels langs de haarcontour en achtergebleven bronachtergrond;
- witte line-artvulling en de witte achtergrond;
- echte heldere reflecties op het leren jack of de laarzen en te verwijderen
  wit.

Het gevolg is een combinatie van uitgeknipt haar, gaten in de silhouet, witte
plekken tussen haarstrengen en zichtbare witte randen op donkere achtergronden.

De review-contactbladen waren geen geldige kwaliteitscontrole. Op hun
overzichtsschaal waren de pixel- en contourfouten te klein om betrouwbaar te
zien. De dataset had op oorspronkelijke resolutie, bestand voor bestand,
gecontroleerd moeten worden voordat de cache of training werd gestart.

## Concrete kapotte voorbeelden

| Gegenereerd trainingsbeeld | Originele bron | Zichtbaar probleem |
| --- | --- | --- |
| `assets/lora/JSEED/train/rejected-jseed-dataset-v9/train/determined_waistup_wide__b08_1024x1024__cool-light-gray.png` | `assets/lora/JSEED/uncleaned/determined_waistup_wide.png` | Haarcontour en losse strengen zijn weggeknipt. |
| `assets/lora/JSEED/train/rejected-jseed-dataset-v9/train/determined_waistup_wide__b12_800x1328__white.png` | `assets/lora/JSEED/uncleaned/determined_waistup_wide.png` | Dezelfde structurele haarschade, minder opvallend door de witte achtergrond. |
| `assets/lora/JSEED/train/rejected-jseed-dataset-v9/train/determined_fullbody_wide__b01_1504x688__dark-slate.png` | `assets/lora/JSEED/uncleaned/determined_fullbody_wide.png` | Witte plekken en een lichte rand rond het haar. |
| `assets/lora/JSEED/train/rejected-jseed-dataset-v9/train/determined_nearly_fullbody__b14_720x1456__dark-slate.png` | `assets/lora/JSEED/uncleaned/determined_nearly_fullbody.png` | Duidelijke witte halo en restpixels rond het haar. |
| `assets/lora/JSEED/train/rejected-jseed-dataset-v9/train/portrait-lookleft-lineart__b02_1456x720__dark-slate.png` | `assets/lora/JSEED/train/portrait-lookleft-lineart.png` | Buitenste haarvorm is rafelig en deels verdwenen; wit blijft lokaal achter. |

De line-artbron is een bijzonder duidelijk bewijs van het fundamentele
probleem: wit binnen gesloten zwarte haarcontouren is onderwerpinhoud, terwijl
het omringende canvas eveneens wit is. Een generieke binaire
voorgrondsegmentatie kan dat niet zonder inhoudsverlies oplossen.

## Ongeldige afgeleide artefacten

De cache:

`runs/jseed_dataset_v5_preview_v9/flux2_klein_base_9b_jseed_17bucket_cache.pt`

- grootte: 5.676.951.499 bytes;
- inhoudssignatuur:
  `3c2a08f996e3f1f022bbcbdf004237d84def03950b69468f3de5e90bc4668fdf`;
- technisch correct voor exact dataset v9, maar daardoor juist onbruikbaar voor
  een gerepareerde dataset.

De gestopte trainingsrun:

`runs/flux2_jseed_subject_lora_9b_nf4_v5/`

Checkpoint 250 en 500 bevatten ieder LoRA-weights plus optimizer-, scheduler-
en RNG-state. Het log meldt een succesvolle checkpoint-save op stap 500. De
training liep daarna door tot ongeveer stap 520 en werd toen gestopt. Geen van
deze checkpoints mag als kwaliteitskandidaat of hervattingspunt worden gebruikt.

## Wat technisch wel werkte

De trainermechaniek zelf is niet de oorzaak van deze mislukking:

- FLUX.2 Klein 9B training-base;
- bevroren transformer in NF4-opslag met BF16-rekenwerk;
- Qwen3-8B-FP8 als bevroren conditioner;
- conditionerlagen `9 18 27`;
- rank en alpha 16;
- persistente latents en promptembeddings op CPU, alleen de actuele batch naar
  CUDA;
- circa 4,3 tot 4,8 seconden per trainingsstap;
- circa 13,8 GiB VRAM tijdens training;
- geen geconstateerde nieuwe swapgroei.

Voor de trainermechaniek zijn de relevante bestanden:

- `tools/diffusers/train_dreambooth_lora_flux2_klein.py`
- `scripts/patch_flux2_klein_lora_trainer.py`
- `runs/flux2_jseed_subject_lora_9b_nf4_v5/train.log`

Een nieuwe, gewijzigde dataset vereist altijd een nieuw datasetpad, een nieuwe
cache en een nieuwe trainingsoutput. De v5-run mag niet worden hervat.

## Leesvolgorde voor onderzoek

1. Dit document.
2. `scripts/prepare_jillian_subject_lora_dataset_v4.py`.
3. `aigen/keyframe_segmentation.py`.
4. `assets/lora/JSEED/train/rejected-jseed-dataset-v9/manifest.json`.
5. De originele bestanden in `assets/lora/JSEED/train/` en
   `assets/lora/JSEED/uncleaned/`.
6. De vijf concrete bron/output-paren hierboven, op 100% zoom.
7. Pas daarna de contactbladen in
   `assets/lora/JSEED/train/rejected-jseed-dataset-v9/`.
8. Alleen voor trainerperformance:
   `runs/flux2_jseed_subject_lora_9b_nf4_v5/train.log` en
   `scripts/patch_flux2_klein_lora_trainer.py`.

## Vragen voor Claude

1. Moeten alle goedgekeurde masters eerst naar één canonieke map
   `assets/lora/JSEED/train/` worden gepromoveerd en moet de bouwer uitsluitend
   die map lezen?
2. Is achtergrondvariatie voor deze subject-LoRA de risico's van
   voorgrondextractie waard, of is trainen op de schone originele witte
   achtergronden verstandiger?
3. Als achtergrondvariatie wel nodig is: welke route kan de originele
   onderwerp-RGB exact behouden, inclusief haarstrengen, witte line-artvulling
   en leren highlights, en uitsluitend bewezen achtergrondpixels vervangen?
4. Hoe moet de kwaliteitscontrole worden ingericht zodat geen enkel
   trainingsbeeld wordt gebruikt voordat de samengestelde output op volledige
   resolutie naast de bron is bekeken?
5. Is er een betrouwbare niet-destructieve manier om witte, met de buitenrand
   verbonden achtergrond te vervangen zonder wit binnen gesloten
   inktcontouren te verwijderen?

Er is nog geen keuze gemaakt voor een vervangende segmentatie- of
compositieroute. Eerst moet worden vastgesteld welke route de broninhoud
daadwerkelijk pixelgetrouw behoudt.

## Relevante implementatiebestanden

```text
aigen/keyframe_segmentation.py
assets/lora/JSEED/TRAINING_HANDOFF.md
assets/lora/JSEED/uncleaned/
model_sources/anime_foreground_segmentation.json
scripts/download_models.sh
scripts/patch_flux2_klein_lora_trainer.py
scripts/prepare_jillian_subject_lora_dataset_v4.py
tools/clean_backgrounds.py
```
