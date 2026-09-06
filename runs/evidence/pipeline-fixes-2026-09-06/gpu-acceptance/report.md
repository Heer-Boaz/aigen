# GPU-acceptatie van herstelronde 1 — 6 september 2026

De bestaande beeld-, video-, upscale- en segmentatieroutes zijn met de aanwezige modellen uitgevoerd op de RTX 5070 Ti. Deze ronde heeft extra concrete fouten in Qwen-geheugenbeheer, USO-geheugenbeheer en randomness, voortgang en cache-identiteit gerepareerd. Dit is acceptatie van de bestaande routes en de fixes uit [herstelronde 1](../implementation.md). De collectie-/selectieflow en extra videonodes uit stappen 2–5 van het [implementatieplan](../../pipeline-review-full-2026-09-05/implementation-plan.md) zijn hiermee niet gebouwd.

De uitvoering is betrouwbaarder geworden. De gewenste tekenstijl over verschillende seeds is nog niet bewezen. Herhaalbaarheid van dezelfde seed en stijlconsistentie tussen verschillende seeds zijn afzonderlijke eigenschappen.

## Proefopzet en grenzen

- Iedere neurale job begon met een actuele VRAM-check. Eigen GPU-jobs liepen achtereenvolgens. Andere Windows/WSL-jobs konden tussendoor geheugen gebruiken.
- De beeldbewerkers kregen de originele sprite van 1086×1448 en, waar aangegeven, de originele multiview van 3840×3890. Model-native preprocessing bleef per backend bestaan. Alleen de contact sheets gebruiken kleine weergavebeelden.
- De positieve prompts en de noodzakelijke LTX-negatieve prompt zijn vooraf onafhankelijk beoordeeld na lezing van `docs/prompting.md`. Zie [promptreview](prompt-review.json), [LTX-aanvulling](ltx-prompt-review-followup.json) en [Qwen één bron](qwen-single-prompt-review.json).
- Geen ontbrekende modelgewichten gedownload. Klein moest wel online de versietag van zijn reeds aanwezige FP8-kernel kunnen opvragen. De model-loaders bleven `local_files_only=True`; de mislukte strikt offline poging staat apart bewaard.
- VRAM in de tabel is MiB. **Torch allocated** is de door het betreffende proces of de worker gemeten allocatiepiek; **totaal GPU** is de eenmaal per seconde bemonsterde bezetting van de hele kaart. Dat laatste omvat andere processen en is geen geïsoleerde modelmeting. Een nul in de ruwe parent-Torch-metingen van subprocessroutes betekent dat de worker de GPU gebruikte; diens eigen meetwaarden staan hieronder en in de backendmetadata.
- Looptijden met swapgroei zijn bruikbaar als uitvoeringslog, maar niet als normale offloadbenchmark. Alle oorspronkelijke metingen en mislukte pogingen blijven bewaard.

## Beeldbewerking

| Route | Werkelijk uitgevoerde configuratie | Torch allocated | Totaal GPU | Uitkomst en bewijs |
|---|---|---:|---:|---|
| FLUX.2 Klein 9B scaled-FP8 | Twee refs, 768×1024 én 1152×1536, 4 stappen, seed 71 | 11.989 | 15.973 | Beide outputs voltooid; [resultaat](klein-edit/result.json), [echte transformerinputs](klein-edit/transformer-inputs.json) |
| Klein strength 0,5 | Dezelfde twee refs en canvassen, 2 effectieve stappen per output | 11.989 | 15.673 | Alle refgroepen behouden; voortgang 4/4; [bytevergelijking vóór/na voortgangsfix](klein-strength-progress-comparison.json) |
| FLUX.2 Dev NVFP4 | Twee refs, 1152×1536, 30 stappen, CFG 4, seed 71 | 6.247 | 9.223 | Voltooid; [resultaat](flux-dev-v2/result.json) |
| Qwen Lightning, één bron | Sprite op eigen VAE-resolutie, 1152×1536, 8 stappen, seed 71 | 12.617 | 14.888 | Publieke image-edit-API voltooid; 32 resident / 28 gestreamde blokken; [resultaat](qwen-single-native/result.json) |
| Qwen base, één bron | Dezelfde sprite en canvas, 40 stappen, CFG 4, seed 71 | 12.625 | 14.633 | Publieke image-edit-API voltooid; 32 resident / 28 gestreamde blokken; [resultaat](qwen-base-single-native/result.json) |
| Qwen Lightning, twee grote refs | 1152×1536, 8 stappen, seed 71, 71.692 denoisetokens | 9.890 grootste fase | 14.530 | Volledige native streamingproef voltooid met 0 extra residentblokken; [fasemetingen](qwen-native-fp8-direct/acceptance-summary.json), [beeld](qwen-native-fp8-direct/image.png) |
| USO FLUX.1-dev FP8 | Content + stijlref, 768×1024, 25 stappen, CFG 4; seeds 70/71 | 14.357 | 15.730 maximaal | Vijf outputs voltooid; byte-identiek per seed in losse/gewijzigde batchvolgorde; [vergelijking](uso-seed-comparison.json) |

De grote Qwen-proef gebruikte de werkelijke conditioner, VAE, transformer en decode met de native dubbele streambuffers. Alleen de lokale optimalisatie voor extra residente blokken was in die diagnostische harness uitgeschakeld. De definitieve budgetberekening laat deze grote configuratie eveneens op native streaming uitkomen, maar die grote twee-ref-configuratie is niet nogmaals door de volledige publieke API uitgevoerd. De publieke één-ref-run toetst die API en de nieuwe residentieberekening met 32 blokken daadwerkelijk op de GPU.

De oorspronkelijke Qwen één-ref-resultaatheader vermeldt ten onrechte VOSR. De per-outputmetadata zegt correct `none` en raw/output hebben dezelfde hash. De eigenaar van de resultaatmetadata is inmiddels gerepareerd; [publicatieproef zonder modelherhaling](qwen-postprocess-metadata.json). Historische resultaten zijn niet herschreven.

De daaropvolgende base-run bevestigt de metadatafix ook via een volledige modeluitvoering: `none` op beide niveaus, geen VOSR-velden en identieke raw/output-hash. Hij voltooide alle 40 stappen met CFG 4 in 376,8 seconden inclusief laden; er was 0,53 MiB tijdelijke swapgroei, dus ook deze tijd is geen schone offloadbenchmark. Dit was een expliciete test van de algemene base-backend, geen nieuwe baseline of fallback in de karakterpipeline; [promptreview](qwen-base-prompt-review.json).

### Wat de beelden laten zien

De [contact sheet van de beeldbewerkers](image-comparison.png) toont één seed per route op dezelfde schaal en in hetzelfde 3:4-aspect. De [bronlijst](image-comparison.json) verwijst naar de originele bestanden. USO gebruikt zijn eigen vooraf beoordeelde content-/stijlprompt; dit is geen gecontroleerde ranglijst met identieke modelinstellingen.

- Klein produceert een gladdere celtekening, maar vereenvoudigt de arcering en verandert details. Het grotere canvas geeft ook visuele veranderingen. Bij strength 0,5 blijft de output dichter bij de spriteweergave.
- FLUX.2 Dev geeft een herkenbare, gladde illustratie met glanzende schaduwen. Dat bewijst nog geen overeenkomst met de specifieke inkstijl.
- Qwen met de grote multiview maakt de binnenvlakken gladder, maar behoudt deels de blokkerige contour en brengt eigen illustratiekeuzes aan.
- De [Qwen-proef met uitsluitend de sprite](qwen-single-comparison.png) geeft bij zowel Lightning als base een duidelijk gladde tekening. Base gebruikt zachtere schaduwen. Beide veranderen kledingdetails. De prompt en bron verschillen van de twee-ref-proef; dit is geen bewijs dat minder referenties of meer stappen op zichzelf beter zijn.
- USO blijft in deze twee seeds sterk bij de spriteweergave. De overdracht naar de gladde inkstijl is gering. De seedfix maakt dit resultaat reproduceerbaar; hij vergroot de stijloverdracht niet.

Er zijn geen pixelrand-, rasterzuiverheids- of zelfgeschreven identiteitsgeometriescores gebruikt.

## Video en de echte TUI-keten

| Route en bron | Gemeten bestand | Torch allocated / reserved | Totaal GPU | Bewijs |
|---|---|---:|---:|---|
| AnimeGen, sprite | 544×720, 33 frames, 16 fps, 2,0625 s, geen audio | 8.063 / 12.652 | 13.807 | [Media](anime-tui-sprite/media-verification.json), [TUI-uitvoering en cache](anime-tui-sprite/result.json) |
| AnimeGen, glad | 544×720, 33 frames, 16 fps, 2,0625 s, geen audio | 8.063 / 12.652 | 13.874 | [Media](anime-tui-smooth/media-verification.json), [TUI-uitvoering en cache](anime-tui-smooth/result.json) |
| LTX-2.3 NVFP4, sprite | 768×1024, 33 frames, 24 fps, 1,375 s, geen audio | 5.152 / 5.438 | 6.091 | [Media](ltx-sprite-v2/media-verification.json), [resultaat](ltx-sprite-v2/result.json), [frames](ltx-sprite-v2/video-contact.png) |
| LTX-2.3 NVFP4, glad | 768×1024, 33 frames, 24 fps, 1,375 s, geen audio | 5.152 / 5.438 | 13.491 | [Media](ltx-smooth/media-verification.json), [resultaat](ltx-smooth/result.json), [frames](ltx-smooth/video-contact.png) |

AnimeGen is via echte Textual Pilot-acties gestart: opgeslagen graph → Run → CLI-subproces → AnimeGen → FFmpeg-contact sheet en 33 geëxtraheerde frames → opnieuw Run. Alle drie afgeleide nodes werden eerst uitgevoerd en daarna uit de cache hergebruikt. De oorspronkelijke graph bleef gelijk en stdout werd gesloten. De graph, snapshots, node-/runmanifesten, SVG-schermen en bestanden staan onder de twee `anime-tui-*`-directories.

AnimeGen bewaart in deze korte voorbeelden de herkenbare sprite- respectievelijk inkweergave; knipperen en een kleine houdingsverandering zijn zichtbaar. LTX behoudt beide weergaves, maar toont in deze configuratie nauwelijks beweging en geen duidelijke knippering. Beide videobackends kregen hetzelfde start- en eindbeeld; dit zijn korte idle-proeven, geen acceptatie van ingewikkelde bewegingen, verschillende keyframe-aspecten of lange animaties.

De hogere totale GPU-bezetting aan het einde van de gladde LTX-run is niet als extra LTX-allocatie geboekt: de worker meldde dezelfde eigen pieken en na zijn exit bleef circa 11 GiB totaal bezet. De volgende Qwen-run begon pas toen een nieuwe VRAM-check weer voldoende ruimte meldde.

## Upscale en segmentatie

| Route | Controle | Torch allocated | Bewijs |
|---|---|---:|---|
| VOSR | B los, A+B, B+A, A in cache + B vers; 25 stappen, seed 42, 512px tiles, 1024 lange zijde | 3.947–3.959 | Zeven beelden: bytes en cache-identiteiten gelijk per input; één modelload per batch; [vergelijking](vosr-comparison.json) |
| IllustrationJanai DAT2 | RGB en palet-PNG met alfa, doel 1024 lange zijde | 2.553 | CUDA-uitvoering en exacte deterministische alfa-resample; [resultaat](illustrationjanai-dat2/result.json) |
| IllustrationJanai ESRGAN | Dezelfde twee inputs | 2.426 | [Resultaat](illustrationjanai-esrgan/result.json) |
| AnimeSharp x4 | Dezelfde twee inputs | 2.426 | [Resultaat](animesharp-x4/result.json) |
| SAM | Origineel 2848×3798, één visueel geplaatst voorgrondpunt | 2.794 | Eindig masker op originele maat, gekozen punt voorgrond en achtergrondhoek achtergrond; [resultaat](sam/result.json) |
| SAM2 | Dezelfde bron en punt | 458 | [Resultaat](sam2/result.json) |
| Anime segmentation ONNX | Dezelfde bron; CUDA Execution Provider | n.v.t. (ORT) | Eindig masker op originele maat; totale GPU-piek 2.056 MiB; [resultaat](anime-segmentation/result.json) |

Dit bewijst uitvoering en de genoemde data-/alphacontracten. Het is geen algemene kwaliteitsranglijst van upscalers of segmentatiemodellen.

## Extra reparaties uit de GPU-proeven

**Qwen: geheugen bij grote native referenties.** VAE-encoding en decode gebruiken de bestaande overlappende Diffusers-tiles boven 2048 pixels, met stride 1536; kleinere beelden behouden de untiled route. De sequentielengte telt daadwerkelijk gepakte referenties, output en tekst samen. De toelating van extra residente transformerblokken gebeurt vóór iedere allocatie, inclusief het eerste blok, met ruimte voor de later geladen pre/postgewichten, workspace en headroom. Dit volgt de eigendom van [LightX2V offload](https://github.com/ModelTC/LightX2V/blob/b96309e82899145aebd8ecf95c387894aba66b1e/lightx2v/models/networks/qwen_image/model.py) en de [Qwen-VAE](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py). Het workspacebudget is een lokale, met deze kleine en grote configuraties getoetste schatting; geen universele bovengrens voor iedere aanvraag.

**Qwen: directe FP8-output.** De pinned LightX2V Triton-quantizer schreef eerst een volledige FP32-output en converteerde die daarna naar FP8. De patch laat dezelfde berekening rechtstreeks naar FP8 opslaan, zoals gebruikelijk bij [productie-FP8-kernels in vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/fp8_utils.py). In 16 echte GPU-proeven waren quantisatie én schalen byte-identiek, ook bij afrondingsgrenzen, subnormale waarden en signed zero. Voor de gemeten 2048×12288-tensor daalde de tijdelijke piek van 120,004 naar 24,004 MiB. Zie [geïnstalleerde kernelproef](fp8-installed.json). De installer past uitsluitend de bekende patch toe, accepteert reeds toegepast en weigert afwijkende broncode; [installerproef](lightx2v-patch-contract.json). Implementatierevisie Qwen is 4 en de runtimepatch staat in de provenance.

**USO: allocator en randomness.** De oorspronkelijke allocator reserveerde tijdens de eerste stap 15.518 MiB en de tweede stap liep vast. `expandable_segments:True`, vóór Torch geladen, liet dezelfde 25 stappen voltooien met 14.518 MiB reserved. De instelling volgt de [PyTorch 2.12 allocatoroptie](https://docs.pytorch.org/docs/2.12/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf); expliciete gebruikersinstellingen blijven voorgaan. Een aparte fout zat in de [upstream referentie-VAE](https://github.com/bytedance/USO/blob/6587514aa3adf8e8f46e5f7e804239651d30b32d/uso/flux/modules/autoencoder.py): die gebruikt globale RNG terwijl de beginruis al een lokale seed had. Nu omvat een per-output `fork_rng` de hele pipeline-call, met herstel van CPU/CUDA-RNG. De echte vijfbeeldenmatrix en een onafhankelijke hashcontrole bevestigen de reparatie. USO-implementatierevisie 2 maakt oude workflowcache-resultaten ongeldig.

**Cache-identiteit van modelbestanden.** USO-validatie, worker en cache gebruiken dezelfde zeven modelpaden. Gewichten, tokenizers en processorconfiguraties tellen mee in de bestaande stat-inventaris. Die inventaris en de Klein-bestandenlijst werden eerder alleen op pad gecachet en konden wijzigingen binnen hetzelfde proces missen. Ze verversen nu aan de uitvoeringsgrens. Dit sluit aan bij het per-uitvoering bepalen van afhankelijkheden in [ComfyUI caching](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_execution/caching.py). Er worden hiervoor geen modelgewichten ingelezen in rendering of inference. Tests vervangen modelbestanden en voegen een conditionerbestand toe; de nieuwe signature ziet beide veranderingen. [Werkelijke USO-inventaris](uso-model-identity.json).

**Voortgang en foutbewijs.** Klein-strength telt effectieve stappen. De beelden vóór/na deze voortgangsfix zijn byte-identiek. WanGP-beeld/video-adapters beginnen een fase zonder stappentotaal met een lege teller, zodat decode geen oude denoisestappen toont. Qwen-workerfouten bewaren de volledige stack; USO logt de oorspronkelijke stack bij een mislukte seed. Qwen publiceert de werkelijk gekozen nabewerking en vermeldt geen VOSR-instellingen bij `none`.

## Regressies en afbakening

26 appregressies zijn geslaagd: [volledige suite vóór de laatste inventarisaanvulling](app-contracts-final-v3.log) en [gerichte tests inclusief die aanvulling](model-identity-contracts-v2.log). Vier Qwen-reference-regressies slaagden in de aparte LightX2V-omgeving: [log](qwen-contracts.log). De [testinstructies](../../../../tests/README.md) bevatten de actuele commando's. CPU-tests gebruiken waar nodig herkenbare neurale doubles; de bovenstaande GPU-runs gebruiken echte modellen.

Het [machineleesbare overzicht](acceptance-index.json) verwijst naar de oorspronkelijke resultaten, fouten, workeromgeving en metingen. `git diff --check` en de shellsyntaxcontrole van de LightX2V-installer zijn geslaagd.

Boogu, HiDream O1 en Hunyuan zijn in deze ronde niet uitgevoerd: hun standaardruntime/modelinstallatie ontbreekt en downloaden was uitgesloten. Wu, nieuwe LoRA-training, de karakteraudit/retry-route en uitgebreide kandidaatselectie hebben hier geen nieuwe GPU-acceptatie gekregen. De bestaande Fixer-, document-, undo/redo-, cache- en Stop-contracten zijn op CPU/TUI getest.

De oorspronkelijke mislukte runs zijn bewijs van gevonden problemen, geen verborgen modelwissels. Qwen Lightning en Qwen base blijven expliciete routes; Klein, FLUX.2 Dev en USO eveneens. Er zijn geen commits gemaakt en geen projectbestanden verwijderd.
