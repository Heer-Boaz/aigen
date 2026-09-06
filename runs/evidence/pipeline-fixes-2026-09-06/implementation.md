# Herstelronde 1 — 6 september 2026

Dit is het historische verslag van stap 1. De actuele implementatie en
acceptatie van alle vijf stappen staan in [eindverslag](final-report.md).

De bewezen fouten uit stap 1 van het [implementatieplan](../pipeline-review-full-2026-09-05/implementation-plan.md) zijn in de werkboom hersteld. De wijzigingen zijn niet gecommit. Dit is de herstelronde voor bestaande functionaliteit; variantenvergelijking, duurzame kandidaatselectie en uitbreiding van de videonodes uit de volgende stappen zijn hiermee niet gebouwd.

## Gewijzigd gedrag

| Onderdeel | Resultaat | Regressiebewijs |
|---|---|---|
| TUI-eigenschappen | Openen/selecteren verandert geen instellingen. Dezelfde backend/model/profielwaarde is een no-op in het editbuffer. Echte wisselingen blijven defaults toepassen. | Aangepaste Qwen-, AnimeGen- en postprocessconfig; identieke opgeslagen bytes; Save/Undo/Redo; geldige en ongeldige drafts |
| Impliciete samplerdefaults | Een expliciete keuzelijstoptie toont de backenddefault en bewaart `None` in het document. | Compiler-geldige graph opent; expliciet/default wisselen en save/load behouden de waarden |
| Klein-referenties | Eigen aspect en maximaal 1024² pixels volgens de geïnstalleerde Diffusers-route; geen verborgen DAT2-upscale naar het outputcanvas. | Portret, landschap en grote bron; dezelfde context bij andere canvassen, seeds en batchvolgorde |
| Klein-strength | Alle referenties blijven modelcontext. Het eerste beeld wordt apart op het outputcanvas passend geschaald en gecentreerd uitgesneden als init. De initcache gebruikt bronpad én canvas. | Twee ref-ID-groepen bereiken de transformer; 1536² init/noise/tokens/positie-ID’s passen; geen zwarte padding bij vergroten; strength=1 heeft dezelfde noise als gewone editing |
| VOSR-randomness | Elke output krijgt een eigen seedbereik voor upstream VAE-sampling en ruis. CPU- en gebruikte CUDA-RNG worden daarna hersteld. Gewichten blijven per batch geladen. | B alleen, A+B, B+A en A-gecachet+B leveren gelijke bytes en signatures in de CPU-proef; één runtime per uitvoerbatch |
| Seeds/outputbestemming | Dubbele seeds worden bij de algemene aanvraag en de direct aanroepbare FLUX.2-dev-sweep afgewezen vóór outputcreatie. | Alle aangeboden image-edit-backends; geen workerstart of outputdirectory bij dubbele seeds |
| Oriëntatie/transparantie | Gedeelde image-I/O normaliseert EXIF; SAM-canvas, handmatige invoer en automatische loader gebruiken dezelfde pixels. Actieve Klein/Qwen-referenties en automatische canvasmaten volgen displayoriëntatie. VOSR en illustration-upscalers bewaren alfa inclusief palet- en RGB-kleurtransparantie. | Pixelvergelijkingen, puntcoördinaten, canvasafmetingen, verwijderde oriëntatietag, alfa vóór/na; JPEG weigert VOSR-alfa |
| Runstatus | Eén app-owned runstate bewaart de aangevraagde graph en oorspronkelijke statussen. Inhoudelijke wijzigingen markeren de geraakte tak en afstammelingen als `outdated`. Undo herstelt de toepasselijke status; sluiten/heropenen bewaart dit. | Takken/join, reconnect/reorder, verwijderen/undo, naam/titel/layout, nieuwe run/document, echte TUI→CLI→cache-proef |
| Subprocessen | De reader sluit de outputpipe na lezen. | Echte uitvoering, cachehit en SIGTERM via Stop; de pipes zijn daarna gesloten |

Runstatus vergelijkt de documentinhoud met de laatste aanvraag. Het controleert geen extern gewijzigde bronbytes tijdens tekenen; de uitvoeringscache controleert de inputidentiteit bij uitvoering. Graphvergelijking gebeurt bij documentwijzigingen; voortgangsevents gebruiken de berekende geldigheid en bestaande gerichte canvasverversing.

## Implementatiekeuzes en productievoorbeelden

- [Textual Select](https://github.com/Textualize/textual/blob/main/src/textual/widgets/_select.py) meldt ook de gemounte beginwaarde als `Changed`. De UI negeert die ongewijzigde projectie vóór draftverwerking. Het editbuffer bezit daarnaast de semantische no-op, zodat ook andere callers geen defaults resetten.
- [Diffusers Klein](https://github.com/huggingface/diffusers/blob/d4b2bfc04615787846bac271e74a1c033e714e67/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py) bepaalt referentieafmetingen per afbeelding en gebruikt deterministische VAE-encoding. De referentiecache kan daarom onafhankelijk blijven van outputcanvas en seed. De [Flux2-imageprocessor](https://github.com/huggingface/diffusers/blob/d4b2bfc04615787846bac271e74a1c033e714e67/src/diffusers/pipelines/flux2/image_processor.py) doet bij `crop` uitsluitend een uitsnede. Voor het initbeeld is daarom expliciete `ImageOps.fit` nodig; alleen grotere cropmaten zouden zwarte padding toevoegen. Dit punt is door de afzonderlijke planreview gevonden en in een inhoudelijke regressie vastgelegd.
- [Diffusers FLUX img2img](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_img2img.py) koppelt init, noise, timestep en latentcanvas. Hier wordt de reeds gegenereerde passende noise hergebruikt; een tweede willekeurige tensor en de oude vormmismatch vervallen.
- [Upstream VOSR](https://github.com/CSWRY/VOSR/blob/25fbf8e6cb9656b8991c24474f408bdce6fcb1b1/inference_vosr.py) gebruikt globale Torch-RNG bij VAE-sampling en tiled inference. De adapter gebruikt [PyTorch fork_rng](https://github.com/pytorch/pytorch/blob/main/torch/random.py) rond één afbeelding, met alleen het gebruikte CUDA-device in de scope. Geen modelreload of RNG-wijziging per tile.
- [ComfyUI LoadImage](https://github.com/Comfy-Org/ComfyUI/blob/master/nodes.py) normaliseert oriëntatie vóór tensorconversie. [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py) verwerkt kleur en alfa afzonderlijk; hier wordt alfa deterministisch geresampled naar de doelmaat, zonder tweede neurale upscale.
- De scheiding tussen uitvoeringsinhoud en weergave sluit aan bij [ComfyUI cache-identiteit](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_execution/caching.py). De bestaande gesorteerde incoming-connections-helper is naar `WorkflowGraph` verplaatst en wordt gedeeld door compiler en runstate.

## Cache en provenance

De workflowcache bindt nu de lokale Klein-, Qwen-, VOSR- en illustration-upscale-implementatie aan de backendidentiteit, naast de bestaande modelidentiteiten. Resultaatmetadata vermeldt de implementatierevisie; SAM-resultaten krijgen eveneens een revisie. De huidige revisies zijn Klein 2, Qwen 2, VOSR 3, illustration-upscale 2, segmentation 2 en algemene image-edit-API 2. De Qwen-revisie dekt ook de al aanwezige native-VAE-wijzigingen.

Een regressie vervangt per gewijzigde backend de implementatierevisie en controleert dat de werkelijke cache-signature verandert. Dit is een gerichte reparatie van lokale uitvoeringsidentiteit, geen voltooiing van de bredere runtime-/resultatenprovenance uit stap 2 van het plan.

## Uitgevoerde verificatie

- 18 appregressies geslaagd in de appomgeving, inclusief echte Textual Pilot-acties, een echte CLI-subprocessrun, Pixel Art Fixer, cachehergebruik en Stop.
- 4 Qwen-regressies geslaagd in de LightX2V-omgeving.
- 8 bestaande workflowproeven opnieuw geslaagd: eerste uitvoering, cachehit, één gewijzigde seed, gewijzigde LoRA, gewijzigde bron, failure, interruption en hervatting. Compiler, scheduler, cache, FFmpeg en Pixel Art Fixer zijn echt; neurale generators en backendprovenance zijn in deze acht proeven doubles.
- `git diff --check` geslaagd. Geen commits of verwijderingen van projectbestanden.

De [testinstructies](../../../tests/README.md) bevatten de reproduceerbare opdrachten. Logs van deze herstelronde staan naast dit verslag. De extra workflowproef schreef zijn artifacts naar [workflow-data-4f1bb051](../pipeline-review-full-2026-09-05/workflow-data-4f1bb051).

Er is in deze herstelronde geen nieuwe neurale GPU-acceptatie uitgevoerd. De RTX 5070 Ti was tijdens controle bezet met ongeveer 13,5 GB VRAM en 60% GPU-belasting. Daarom zijn nieuwe stijlkwaliteit, VOSR-bytegelijkheid op CUDA en piek-VRAM van Klein met alle referenties plus strength nog niet gemeten. De CPU-bewijzen worden niet als beeldkwaliteitsbewijs gebruikt.

## Vervolg op 6 september: echte GPU-acceptatie

Na vrijgave van de GPU zijn Klein, Qwen Lightning en base, FLUX.2 Dev, USO, VOSR, DAT2/ESRGAN/AnimeSharp, SAM/SAM2/ONNX en AnimeGen/LTX uitgevoerd. AnimeGen liep met een sprite- en gladde bron via de echte TUI → CLI → video/frames → cache-keten. VOSR en USO zijn byte-identiek per seed/input ongeacht de geteste batchvolgorde.

Deze proeven leverden aanvullende reparaties op voor Qwen-VAE/FP8/residentiegeheugen, USO-allocator en VAE-randomness, voortgang, Qwen-resultaatmetadata en verversing van modelinventarissen. De huidige revisies zijn Qwen 4 en USO 2; de overige hierboven vermelde revisies blijven gelden. Er zijn nu 26 appregressies en 4 aparte Qwen-regressies geslaagd.

Het [GPU-verslag](gpu-acceptance/report.md) bevat de concrete configuraties, VRAM- en swapmetingen, beelden, video's en bewijsgrenzen. Het vervangt de bovenstaande historische uitspraak over ontbrekende GPU-acceptatie. De volledige kandidaat-/selectieflow uit volgende implementatiestappen is hiermee niet gebouwd; gewenste stijlconsistentie over seeds is evenmin bewezen.
