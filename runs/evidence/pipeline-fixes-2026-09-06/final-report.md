# Pipeline- en TUI-herstel: eindverslag

De vijf implementatiestappen zijn in de werkboom uitgevoerd. De TUI ondersteunt
referenties, seedvarianten, vergelijken, opgeslagen selectie, vervolgbewerking,
regionale edits en video-export. CLI en TUI gebruiken dezelfde compiler,
uitvoeringscode, resultaatrecords en cache.

De opdracht is [het implementatieplan](../pipeline-review-full-2026-09-05/implementation-plan.md),
binnen [PLAN.md](../../../docs/PLAN.md). De [workflowhandleiding](../../../docs/workflows.md)
beschrijft het concrete gebruik.

| Stap | Gebouwd of hersteld | Bewijs |
| --- | --- | --- |
| 1. Backends | Inspectorinstellingen, native referenties, initcanvas, RNG per seed, alfa/oriëntatie, Qwen-geheugenbeheer, voortgang en workeropruiming | [CPU-herstel](implementation.md), [brede GPU-acceptatie](gpu-acceptance/report.md) |
| 2. Capabilities en resultaten | Backendcontracten, gemeten uitvoer, duurzame geschiedenis, model-/runtime-identiteit en afzonderlijke LoRA-capabilities | [Handleiding](../../../docs/workflows.md), [LoRA-check](lora-capabilities-check.json) |
| 3. Beeldflow | Varianten, vergelijken, keuze opslaan, herstarten, gericht doorwerken en originelen exporteren | [Klein en Qwen via de echte TUI](image-flow-acceptance/report.md) |
| 4. Videoflow | AnimeGen/LTX/Hunyuan-nodes, keyframeposities, seedbatches, extractie, assemblage, rationele timing en audiobeleid | [AnimeGen/LTX via de echte TUI](video-flow-acceptance/report.md) |
| 5. Karakter en maskers | Raw-audit, begrensde retries, geselecteerde seed, optionele VOSR; SAM/MASK-bronbinding en native Qwen-regio-edit | [Karakterflow](character-flow-acceptance/report.md), [regionale flow](character-mask-acceptance/report.md) |

## Concrete acceptatie

Klein en Qwen hebben ieder variantenselectie en vervolgbewerking na herstart
doorlopen. Eén gewijzigde seed maakte alleen die variant opnieuw. Opgeslagen
keuzes bleven dezelfde oorspronkelijke beelden gebruiken. VOSR en USO waren
in de geteste echte GPU-batchvergelijkingen byte-identiek per seed/input,
ook na gewijzigde volgorde en cachehergebruik.

AnimeGen en LTX maakten twee-seedbatches via de echte TUI, gevolgd door
frame-extractie, verwerking, assemblage en export. Frameaantal, timestamps,
duur en audio worden vóór publicatie gecontroleerd. De CPU-mediatests
gebruiken echte bestanden en audiotracks.

De laatste regionale proef maakte met SAM2 en Qwen twee kandidaten op
768×1024. Met strength 1.0 kregen beide de gevraagde lichtblauwe achtergrond;
alle 135969 beschermde pixels bleven gelijk. De complete flow met audit duurde
81,86 seconden, de cacheherhaling 1,30 seconde. De totale GPU-piek was
15244 MiB; de native denoise-allocatiepiek 13270,54 MiB. Totale GPU-bezetting
omvat andere processen en is geen geïsoleerde modelmeting.

Nieuwe regionale Lightning-edits starten daarom op strength 1.0 en gebruiken
de volledige achtstappenschedule. Expliciet opgeslagen lagere waarden blijven
behouden. De gecontroleerde vergelijking veranderde alleen de generationele
strength; dat verandert zowel beginruis als het aantal stappen.

## Kwaliteitsgrenzen

De automatische karakteraudit garandeert niet dat een edit klopt. Bij
strength 0.6 veranderde Qwen vooral de schaduw en bleef de achtergrond wit.
De VLM keurde dat ten onrechte goed. Een onafhankelijk beoordeelde,
explicietere auditvoorwaarde verhielp dit niet in de herhaalde proef. De fout,
prompt, modelrespons en beelden staan in het [regionale verslag](character-mask-acceptance/report.md).

Eerdere karakterproeven tonen bruikbare selectie, waaronder behoud van
handschoenen bij de gekozen Klein-kandidaat, en gemiste expressiewijzigingen
bij Qwen. AnimeGen introduceerde ongewenste bewegingsstrepen; LTX bewoog onder
identieke keyframeankers nauwelijks. Consistente gewenste tekenstijl over
verschillende seeds blijft onbewezen.

Boogu, HiDream O1 en Hunyuan hebben geen nieuwe GPU-acceptatie: hun lokale
runtime/modelinstallatie ontbreekt en downloaden was uitgesloten. De lokale
LoRA-trainer ondersteunt FLUX.1; datasetvoorbereiding en adapterimport zijn
afzonderlijke capabilities. Er is geen nieuwe LoRA-training uitgevoerd.

## Verificatie en review

73 applicatietests slaagden in 35,11 seconden, gevolgd door 18 gerichte checks
na de laatste defaultwijziging. De aparte LightX2V-omgeving heeft vier
referentie- en vijf native-samplingtests doorlopen. De
[testinstructies](../../../tests/README.md) vermelden opdrachten en waar
neurale doubles worden gebruikt. De mislukte semantische auditproef is
afzonderlijk gerapporteerd en telt niet als geslaagde regressie.

Onafhankelijke reviews controleerden prompts, sampling, modellevensduur,
TUI-selectie, cache-identiteit, bronbinding, formulierimport en video-uitvoering.
Laatste concrete reparaties: optionele reference packs, lege guidancevelden,
CLI-validatiefouten en engine-specifieke SAM-runtimeversies. De normale
pipeline kreeg geen pixel-art-randclassifier of handgeschreven karaktergeometrie.

De wijzigingen zijn niet gecommit. Ontbrekende modellen zijn niet gedownload.
Projectmateriaal en mislukte proefresultaten zijn behouden.
De afsluitende `compileall` en `git diff --check` zijn geslaagd; er bleven geen
eigen workflow- of Qwen-workerprocessen actief.
