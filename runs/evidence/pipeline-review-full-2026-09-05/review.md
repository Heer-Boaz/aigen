# Pipeline- en TUI-review, 5–6 september 2026

De applicatie heeft een bruikbare uitvoeringskern, maar is nog geen betrouwbare volledige productieflow. Er zijn aantoonbare fouten die instellingen veranderen, resultaten afhankelijk maken van batchindeling en de interpretatie van experimenten verstoren. Het oordeel **“Klein slecht, de rest goed” volgt niet uit het bewijs**.

Dit onderzoek voegt geen productcodewijzigingen toe. Het levert reproduceerbare proeven en een [concreet implementatieplan](implementation-plan.md). De eerdere [Klein/Qwen-review](../pipeline-review-2026-09-05/review.md) blijft onderdeel van het oordeel; haar bevindingen worden hieronder samengebracht met het aanvullende onderzoek.

## Bereik en bewijs

Onderzocht: de zeven backends van `image-edit`, drie videoformulieren/backends, zes nabewerkingsroutes, SAM-invoer, de workfloweditor, graph/configmodel, compiler, scheduler, caching, opslag, procesbesturing en resultaten. Daarnaast zijn alle CLI-commandofamilies geïnventariseerd en de oudere Kontext-, karakter-, caption- en LoRA-routes op eigenaarschap en aansluiting bekeken. De afgewezen pix2pix/iRO-route is geïnventariseerd, niet opnieuw als oplossing onderzocht. Er is geen nieuwe training of neurale generatierun gestart.

De historische Kontext/ControlNet-, beoordelings- en trainingssubsystemen hebben in deze uitbreiding geen volledige numerieke/GPU-regressietest gehad. Hun aanwezigheid in de CLI is dus geen goedkeuring van al hun interne gedrag. Dit onderscheid staat ook in de dekkingsmatrix.

De checkout stond op `9a78fb5f`, met bestaande lokale Qwen-referentiewijzigingen en andere ongewijzigd gelaten gebruikersbestanden. `docs/PLAN.md` en `docs/prompting.md` zijn gelezen. Het plan en de bewijsinterpretatie zijn door een tweede agent beoordeeld.

| Bewijs | Daadwerkelijk uitgevoerd |
|---|---|
| [Workflowproeven](workflow-results.json) | Acht scenario's met echte compiler, scheduler, diskcache, FFmpeg en Pixel Art Fixer; beeld- en videogenerator vervangen door CPU-doubles |
| [TUI-proeven](tui-results.json) | Echte Textual Pilot, Save/Undo/Run, echt CLI-kindproces, CPU-nabewerking, cachehergebruik, ongeldige invoer, resize en stoppen van een eigen procesgroep |
| [Optionele instellingen](tui-optional-results.json) | Een compiler-geldige graph veroorzaakt een echte inspectorcrash |
| [Backendproeven](backend-results.json) | Echte VOSR-ruisfunctie, batching en cache met neurale CPU-doubles; publieke FLUX-dev-aanroep met worker-double |
| [Mediaproeven](media-results.json) | Echte alfa-/oriëntatieverwerking; SAM-neurale stap vervangen; gepinde Hunyuan-latentberekening |
| [Historische video's](existing-video-results.json) | Alle 136 MP4's onder de drie video-runmappen met FFprobe onderzocht |
| [Inventaris](inventory.json) | CLI, backends, node-/artifacttypen en aanwezige runtimebronnen |

Een geslaagde CPU-proef bewijst geen tekenstijl, modelkwaliteit of VRAM-geschiktheid. Een leesbare historische video bewijst dat die run een video opleverde; de huidige runtime en andere instellingen kunnen verschillen.

## Wat kan daadwerkelijk door de TUI-flow?

“Workflow” betekent hieronder een verbinding via een getypeerde node, compiler en executor, niet alleen een los formulier.

| Route/functie | CLI | Losse TUI | Workflow | Oordeel |
|---|---|---|---|---|
| Klein 9B | Ja | Ja | `image-edit` | Vier eerdere integratiefouten; eerst herstellen |
| Qwen 2511 Lightning en base | Ja | Ja | `image-edit` | Werkende expliciete routes; provenance en bewijs rond native VAE onvoldoende |
| FLUX.2 dev NVFP4 | Ja | Ja | `image-edit` | Workercontract bekeken; dubbele seeds botsen op hetzelfde bestand |
| HiDream-O1 Full FP8 | Ja | Ja | `image-edit` | Comfy-owner vergeleken; standaardruntime hier ontbreekt |
| Boogu Edit Turbo FP8 | Ja | Ja | `image-edit` | Officiële DMD-route vergeleken; standaardruntime hier ontbreekt |
| USO FLUX.1 FP8 | Ja | Ja | `image-edit` | Expliciete content-/stijlrollen; geen nieuwe neurale kwaliteitsvalidatie |
| AnimeGen-I2V | Ja | Ja | Eigen node | Historisch uitvoeringsbewijs; besturing en frameverwerking getest |
| LTX-2.3 | Ja | Ja | **Nee** | Losse keyframebackend; niet vanuit de graph te verbinden |
| HunyuanVideo-1.5 | Ja | Ja | **Nee** | Eén geslaagde historische run; standaardruntime hier ontbreekt |
| VOSR | Ja | Ja | Beeld en frames | Batchruis schendt het contract van onafhankelijke nodes |
| DAT2 / ESRGAN / AnimeSharp | Ja | Ja | Beeld en frames | Tiled verwerking; RGB-conversie verwijdert alfa |
| Wu / Pixel Art Fixer | Ja | Ja | Beeld en frames | Wu produceert RGB; Fixer daadwerkelijk op CPU uitgevoerd |
| SAM1/SAM2/anime-segmentatie | Ja | Ja | **Nee** | Geen mask-artifact/node; EXIF-coördinaten komen niet overeen |
| Video importeren | Via losse postprocess-opdracht | Via bestandsveld | **Geen videobronnode** | Bestaande video's niet als graphbron te gebruiken |
| Frames → video/export | Geen assemblageopdracht in deze CLI | Nee | **Nee** | Nabewerkte frames blijven losse bestanden |
| Seedvarianten vergelijken/selecteren | Meerdere seeds mogelijk | Geen geïntegreerde selectie | **Geen collectie-/selectienode** | Centrale schakel voor stijlontwikkeling ontbreekt |
| Karakteraudit, view acceptance, captioning, LoRA-dataset/training | Oudere aparte commando's | Geen volledige flow | Geen nodes | Andere owners; niet gelijk aan de moderne image-edit-flow |

De bestaande template verbindt referentiepack → twee beelden → nabewerking → AnimeGen → contact sheet/frame-extractie → framenabewerking. Ze eindigt met een oorspronkelijke video en losse bewerkte frames. Een video van die laatste frames bestaat niet automatisch. Zie [nodecontracten](/home/boaz/aigen/aigen/workflow_graph.py:31), [template](/home/boaz/aigen/aigen/workflow_templates.py:157) en [artifacttypen](/home/boaz/aigen/aigen/workflow_artifacts.py:38).

## Aantoonbare fouten, op prioriteit

### P1 — De inspector wijzigt instellingen bij het tonen

Textual stuurt ook tijdens initialisatie `Select.Changed`. De editor behandelt dit als een gebruikerswijziging, waarna het editbuffer bij `backend`, `sampling` of postprocess-`model` defaults toepast. Eenzelfde geselecteerde modelnaam kan dus andere instellingen wissen. Owners: [eventverwerking](/home/boaz/aigen/aigen/workflow_editor.py:438), [modelvervanging](/home/boaz/aigen/aigen/workflow_edit_buffer.py:591), [backenddefaults](/home/boaz/aigen/aigen/workflow_edit_buffer.py:729). Dit gedrag is vergeleken met de [Textual Select-implementatie](https://raw.githubusercontent.com/Textualize/textual/main/src/textual/widgets/_select.py).

Echt gereproduceerd, zonder een model te draaien:

- Pixel Art Fixer `fast`, `force_step=4` → alleen selecteren → `full`, `force_step=None`.
- Save met `force_step=2` schrijft **2** naar het bestand, waarna de actieve graph weer **None** bevat en opnieuw dirty is.
- Undo veroorzaakt opnieuw een defaultwijziging; Redo is daarna niet meer beschikbaar.
- Qwen base `steps=37`, `guidance=5.25` → inspector openen → **40 en 4.0**.
- AnimeGen `steps=9` → inspector openen → **8**.

Dit is direct relevant voor modelvergelijkingen: de ingestelde behandeling kan vóór Run veranderd zijn. Herstel hoort bij de scheiding tussen propertyprojectie en een daadwerkelijke configuratiewijziging, plus tests van openen/selecteren/save/undo. Geen nieuwe UI-laag nodig.

### P1 — Een geldige graph kan de inspector laten crashen

`ImageEditConfig` staat `sampler=None` en `scheduler=None` toe; de compiler resolveert deze naar backenddefaults. De inspector geeft `None` echter door aan een niet-lege `Select`, waarvoor dat geen toegestane optie is. De echte fout is `Illegal select value None`.

De aparte [proef](tui_optional_settings_probe.py) compileert eerst een volledig verbonden graph en opent daarna zijn inspector. Dit is een contractverschil tussen opgeslagen configuratie, effectieve instellingen en UI-projectie. Toon een expliciete keuze voor de backenddefault, of projecteer de effectieve waarde zonder het document te wijzigen. Zie [configschema](/home/boaz/aigen/aigen/workflow_graph.py:62) en [PropertySelect](/home/boaz/aigen/aigen/workflow_inspector.py:74).

### P1 — VOSR-output hangt af van andere nodes in de batch

`VosrRuntime` seedt eenmaal in de constructor. Iedere volgende afbeelding verbruikt verder uit dezelfde RNG. De scheduler voegt onafhankelijke postprocessnodes met dezelfde instellingen samen, terwijl hun cachekeys elk alleen de eigen inputs en instellingen vertegenwoordigen. Owners: [seed](/home/boaz/aigen/aigen/generation/vosr_runtime.py:59), [groepering](/home/boaz/aigen/aigen/workflow_execution.py:477), [batchexecutie](/home/boaz/aigen/aigen/workflow_execution.py:929). De daadwerkelijke ruis komt uit de [gepinde VOSR-inferentiecode](https://github.com/CSWRY/VOSR/blob/25fbf8e6cb9656b8991c24474f408bdce6fcb1b1/inference_vosr.py).

De CPU-proef doorloopt de echte workflow, cache, batchadapter en upstream tiled-ruisfunctie, met encode/decode/DiT-doubles. Node B krijgt **dezelfde signature maar andere outputbytes** wanneer A eerst draait. Wanneer A al gecachet is, krijgt B de output van “B alleen”. Daarmee is de afhankelijkheid van batchsamenstelling bewezen; de omvang van een zichtbaar CUDA-beeldverschil is niet gemeten.

De seed moet per zelfstandig outputartifact gelden. Modelgewichten blijven herbruikbaar binnen de batch. De wijziging vereist tegelijk een nieuwe implementatie-/cacherevisie.

### P1 — Klein heeft nog de vier eerder bewezen eigenaarsfouten

De [eerste review](../pipeline-review-2026-09-05/review.md) bevat de precieze probes en bronregels:

1. Referenties krijgen impliciet een neurale upscale en worden naar het outputaspect getrokken.
2. De referentiecache gebruikt alleen paden, terwijl preprocessing van het outputcanvas afhangt. Resultaten worden daarmee afhankelijk van casevolgorde.
3. Met `strength` wordt alleen de eerste referentie als init-image gebruikt en verdwijnt de overige referentiecontext.
4. Boven het 1MP-initcanvas kunnen het aantal latenttokens en de outputpositie-ID's uiteenlopen.

Dit wijkt af van de [gepinde Diffusers Klein-owner](https://github.com/huggingface/diffusers/blob/d4b2bfc04615787846bac271e74a1c033e714e67/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py). De remedie is correcte scheiding van referentieconditioning en output/init-geometrie, met cache-identiteit die daarbij past. Deze fouten bewijzen niet dat het Klein-model zelf de gewenste tekenstijl niet kan leren of volgen.

### P1 — Cacheprovenance omvat niet alle uitvoeringssemantiek

De eerdere Qwen-proef vergelijkt de HEAD- en huidige implementatie van drie gewijzigde preprocessing-/workerbestanden: gewijzigde broncode, maar dezelfde workflowsignature. De huidige [provenance-owner](/home/boaz/aigen/aigen/workflow_provenance.py:28) gebruikt veelal constante API-/modelrevisies. Lokale preprocesswijzigingen, relevante runtimepatches en uitvoeringswijzigingen moeten daar expliciet in doorwerken.

Bron- en LoRA-inhoudsinvalidatie werkt in de nieuwe workflowproeven. Dit is dus geen algemene afkeuring van de cache. Iedere outputveranderende reparatie moet haar eigen semantiekrevisie meenemen; anders kunnen oude resultaten de reparatie verbergen. Bestandshashes horen bij opname van artifacts/provenance, niet bij iedere denoisestap.

### P2 — De TUI toont oude successen na relevante wijzigingen

Na een echte run en cachehit blijft `fix: reused` staan wanneer het bronpad naar een ander beeld wordt gewijzigd. De graph en de getoonde runstatus hebben geen zichtbare revisiekoppeling. Dit is een **presentatiefout**: opnieuw uitvoeren invalideert de gewijzigde bron correct.

Koppel resultaatstatus aan de uitgevoerde graph/configrevisie en markeer relevante opvolgers als gewijzigd. Bewaar het oude resultaat als geschiedenis. Alleen nodes verplaatsen of titels wijzigen hoort geen inhoudelijke invalidatie te veroorzaken. Owners: [runtime-statussen](/home/boaz/aigen/aigen/image_tui.py:1707), [documentprojectie](/home/boaz/aigen/aigen/workflow_canvas.py:148).

De outputstatus zelf is wél zichtbaar in de editor; `_set_status` stuurt hem door. De eerdere verdenking dat voltooiing alleen “Ready” liet zien is door de echte proef verworpen. Wat ontbreekt is artifactinspectie, vergelijken, openen en een opgeslagen selectie.

### P2 — FLUX dev retourneert twee resultaten voor één bestemming

De publieke `image-edit`-route accepteert twee gelijke seeds. De FLUX-dev-adapter maakt voor beide dezelfde bestandsnaam en de worker gebruikt `replace`. In de CPU-workerproef geven seeds `(7, 7)` twee outputrecords met één pad. Dit bewijst een padbotsing en overschrijfbare bestemming, geen twee verschillende neurale uitkomsten. Owners: [seedverwerking](/home/boaz/aigen/aigen/generation/flux2_dev_wangp.py:77), [publicatie](/home/boaz/aigen/aigen/generation/flux2_dev_wangp_worker.py:98).

Kies één algemeen contract: dubbele seeds afwijzen, zoals andere adapters al doen, of een expliciete output-ID gebruiken wanneer herhalingen bewust gewenst zijn. Vermijd validatie op meerdere plaatsen voor hetzelfde contract.

### P2 — Alfa en oriëntatie worden niet overal hetzelfde behandeld

- DAT2, ESRGAN en AnimeSharp delen een RGB-tensorpad. RGBA wordt RGB en alfa komt niet terug; de echte conversie is op CPU bewezen. [Owner](/home/boaz/aigen/aigen/generation/image_upscale.py:351).
- VOSR bewaart een expliciet A-kanaal, maar verliest transparantie van palet-PNG's met `tRNS`. [Owner](/home/boaz/aigen/aigen/generation/vosr_backend.py:150).
- De SAM-canvas past EXIF-rotatie toe; de uitvoeringsinvoer doet dat niet. De proef toont een canvas van 32×64 tegenover een modelinvoer van 64×32, met dezelfde doorgegeven puntcoördinaten. [Canvas](/home/boaz/aigen/aigen/sam_prompt_canvas.py:248), [uitvoering](/home/boaz/aigen/aigen/sam_commands.py:224).

Normaliseer het beeld bij de invoereigenaar, zodat presentatie en uitvoeringscoördinaten hetzelfde beeld gebruiken. Leg alfa-/kleurbeleid vast per bewerking. Dit is beeld-I/O, geen handmatige karaktergeometrie of nieuwe extractieroute.

## Beoordeling van de overige backends

**AnimeGen.** De 50 aanwezige MP4's zijn leesbaar; 45 hebben een aangrenzend configbestand. De wrapper valideert `4n+1` frames, maakt seeds per output en heeft expliciete Lightning/full-profielen. Start en eindbeeld worden beide naar het startcanvas geresized: bij verschillende bronaspecten vervormt het eindbeeld. Maak het gekozen fit/crop/pad-beleid zichtbaar en test het; eenzelfde outputcanvas voor videokeyframes blijft noodzakelijk. Offload en modelkeuze zijn bekeken tegenover de [modeldocumentatie](https://huggingface.co/aidealab/AnimeGen-I2V). Pixelart-stijlbehoud in video is hiermee niet getest.

**LTX.** Alle 85 MP4's zijn leesbaar; 71 hebben een naastgelegen config. De huidige adapter gebruikt start/eind/interne keyframes, valideert de frameposities, schakelt promptenhancement expliciet uit en remuxt zonder audio. Zes historische video's bevatten nog audio: oude outputs vertegenwoordigen dus niet automatisch de huidige route. De twee lokale runtimepatches zijn zichtbaar en hebben tracked patchbestanden. De start/eind-guiding-latentwijziging is vergeleken met de [gepinde WanGP-owner](https://github.com/deepbeepmeep/Wan2GP/blob/5582327dc25e45fec6cda0f27144d4dcf7ed104b/models/ltx2/ltx2.py). Geen nieuwe numerieke solver- of beeldkwaliteitsvergelijking uitgevoerd.

**Hunyuan.** De handoff eindigt bij drie mislukte pogingen, maar de vierde run van 18 juli 2026 is wel geslaagd: 512×768, 49 frames, 24 fps, `group_offloading=true`, `overlap_group_offloading=false`. Config, complete log en FFprobe bevestigen dit. De standaardruntime is momenteel niet aanwezig. Verder accepteert de adapter ieder positief frameaantal, terwijl de [upstream pipeline](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/60783e704160023913bee78f0b47036d393d4dfa/hyvideo/pipelines/hunyuan_video_pipeline.py) 49–52 gevraagde frames naar dezelfde temporele latentlengte projecteert. De wrapper verifieert de daadwerkelijke videolengte niet. Dit is een open outputcontractcontrole; een nieuwe afwijkende-frame-GPU-run is niet uitgevoerd.

**HiDream.** De adapter gebruikt native Comfy-nodes, 1–10 refs en de juiste familie van grote canvassen. Deze backend is een concreet tegenvoorbeeld voor een universeel 1MP-budget: de [upstream latentnode](https://raw.githubusercontent.com/Comfy-Org/ComfyUI/26515acd23fa291a8f5ab53c5997258598de0701/comfy_extras/nodes_hidream_o1.py) beschrijft circa 4MP als trainingsgebied. De adapter zet experimentele seam-smoothing vast aan en begint voor iedere seed een nieuw proces. Dat moet in de effectieve instellingen en het kostenbeeld staan; het is geen bewezen stijloplossing. De standaard Comfy-runtime ontbreekt hier, dus geen huidige uitvoering bewezen.

**Boogu.** De adapter volgt de officiële Turbo/DMD-route: één referentie, geen CFG, eigen korte schedule, geen promptrewriter. De [officiële beeldprocessor](https://raw.githubusercontent.com/Boogu-Project/Boogu-Image/29c040ff975d19231911753a0dbf976ae98621b1/boogu/pipelines/image_processor.py) behoudt het aspect bij verkleinen en schaalt referenties niet op. Dit verschilt van de Klein-bewerking in aigen. De standaardruntime ontbreekt; integratiecode is geen 16GB- of kwaliteitsbewijs.

**USO.** De volgorde content → stijlreferenties en lege prompt voor behoud van de compositie passen bij de [officiële implementatie-instructies](https://github.com/bytedance/USO/blob/6587514aa3adf8e8f46e5f7e804239651d30b32d/README.md). De wrapper werkt met expliciete offload en zijn eigen contentpreprocessing. Het TUI-formulier kent slotlabels; de graph presenteert generieke geordende referenties. Stijllekkage/identiteitsvermenging uit oudere experimenten is geen nieuwe reden om USO als stijlfabriek aan te bevelen.

**Qwen.** De huidige native-VAE-keuze heeft één beperkte gecontroleerde ablation als bewijs. De semantische encoder behoudt zijn oorspronkelijke budget. Er is geen aangetoonde winst in seedvaste tekenstijl of maximale veilige multireferentiebelasting. De eerder gevonden onjuiste postprocessmetadata en verdwenen Qwen-batchdetails blijven van toepassing. Grote referenties zijn niet per definitie waardeloos of beter; preprocessingbeleid en bewijs horen per backend te bestaan.

**LoRA en oudere karakterroutes.** De bestaande `lora train-run` gebruikt `train_dreambooth_lora_flux.py` en een FLUX.1-basismodel, zoals zichtbaar in [de trainer-owner](/home/boaz/aigen/aigen/lora_training.py:21) en het [Diffusers-voorbeeld](https://github.com/huggingface/diffusers/blob/v0.38.0/examples/dreambooth/train_dreambooth_lora_flux.py). Dat is geen bestaande Klein-/Qwen-trainingsroute. Oudere briefs/views/keyframes lopen via Kontext/ControlNet, terwijl moderne image-edit en workflow daarvan losstaan. Import van een LoRA en productie van een compatibele LoRA zijn verschillende capabilities. De TUI mag de huidige aanwezigheid van het ene niet als het andere presenteren.

## Wat werkt en moet behouden blijven

De uitvoeringskern gebruikt getypeerde artifacts, topologische afhankelijkheden, expliciete backends, snapshots, contentidentiteiten en atomische cachepublicatie. De acht workflowproeven bevestigen cachehits, gerichte invalidatie bij een seedwijziging, wijzigingen aan bron-/LoRA-bytes, foutpropagatie, onderbrekingsregistratie en hervatten met hergebruik. Geen algemene runnerherschrijving nodig.

De echte TUI kan een opgeslagen CPU-flow uitvoeren en de tweede run uit de cache halen. Ongeldige property-invoer wordt atomisch geweigerd. Stop stuurt SIGTERM naar de eigen procesgroep en de app komt terug uit de actieve toestand. Op 80×24 blijven de actieknoppen binnen het scherm; grote graphs vereisen pannen. [SVG van de gecontroleerde editor](tui-data-204e3141/editor-80x24.svg).

De voornaamste uitbreiding is daarom een betrouwbare authoring-/resultatenlaag op de bestaande uitvoeringskern. Voor stijlontwikkeling is het eerste volledige pad: **referenties → expliciete backend en instellingen → onafhankelijke seedvarianten → vergelijking en opgeslagen keuze → volgende bewerking of export**. Video, maskers en verplichte audit voor expliciete karakterroutes sluiten daarop aan volgens het [implementatieplan](implementation-plan.md).

## Reproduceren

Alle onderstaande commando's draaien zonder neurale modelgeneratie. Ze schrijven nieuwe synthetische evidence onder deze map; de bestaande runs worden alleen door de inventaris gelezen.

```bash
.venv/bin/python runs/evidence/pipeline-review-full-2026-09-05/workflow_probes.py
.venv/bin/python runs/evidence/pipeline-review-full-2026-09-05/tui_probes.py
.venv/bin/python runs/evidence/pipeline-review-full-2026-09-05/tui_optional_settings_probe.py
.venv/bin/python runs/evidence/pipeline-review-full-2026-09-05/backend_probes.py
.venv/bin/python runs/evidence/pipeline-review-full-2026-09-05/media_probes.py
.venv/bin/python runs/evidence/pipeline-review-full-2026-09-05/inventory.py
```

De bugproeven slagen wanneer ze het **huidige foutieve gedrag reproduceren**. Ze zijn gedateerde auditevidence, geen groene regressiesuite voor gerepareerd gedrag. `backend_probes.py` gebruikt de geïnstalleerde gepinde VOSR-bron; `media_probes.py` gebruikt de bijgesloten gepinde upstreambron. De neurale doubles, verworpen verdenkingen en beperkingen zijn expliciet vermeld om codebewijs niet met modelkwaliteit te verwarren.
