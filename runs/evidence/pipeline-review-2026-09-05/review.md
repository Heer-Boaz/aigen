De huidige beeldpipeline bevat concrete fouten die modelvergelijkingen en stijlproeven kunnen vertekenen. Vooral FLUX.2 Kleins referentieverwerking en de workflowcache moeten worden hersteld voordat verschillen tussen instellingen betrouwbaar kunnen worden geïnterpreteerd.

Review van de werkboom op 5 september 2026, gebaseerd op HEAD `9a78fb5f` plus de bestaande, ongecommitte wijzigingen. De scope is de actieve beeldpipeline: CLI/TUI, workflowcompilatie en uitvoering, referenties, prompts, FLUX.2 Klein, Qwen-2511, resultaatregistratie en de aansluiting op `docs/PLAN.md`. Andere beeldbackends zijn alleen op dispatchniveau bekeken; hun modelimplementaties en de video- en trainingspipelines zijn niet opnieuw gevalideerd. Deze review wijzigt geen applicatiecode. De eerdere Qwen-referentiewijzigingen zijn expliciet mee beoordeeld.

De [CPU-proeven](probes.py) hebben de bevindingen over gegevensstromen, caching en dimensies gereproduceerd; [de resultaten](probe-results.json) staan ernaast. VAE en neurale upscaler zijn testdubbels. De echte aigen-functies voor voorbereiding, de geïnstalleerde Diffusers-imageprocessor, latentpacking en workflowcache-sleutelberekening worden wel uitgevoerd. Dit bewijst uitvoeringsfouten, geen visuele kwaliteitswinst. Tijdens deze review zijn geen modellen gedraaid.

| Prioriteit | Bevinding | Direct gevolg |
| --- | --- | --- |
| P1 | 1. Klein bewerkt en vervormt referenties vóór generatie | Referentieverhoudingen en mogelijk de tekenstijl veranderen; onnodig neuraal werk |
| P1 | 2. Kleins referentiecache negeert relevante canvasinstellingen | De conditionering van een taak hangt af van andere taken in dezelfde batch |
| P1 | 3. Klein laat extra referenties weg zodra `strength` is ingesteld | De opgegeven referentiebundel wordt niet volledig gebruikt |
| P1 | 4. Kleins init-latents en positie-ID's kunnen verschillende groottes krijgen | Geaccepteerde grotere canvassen hebben bij `strength` een fout in tensordimensies |
| P1 | 5. Gewijzigde Qwen-code verandert de workflowcache-sleutel niet | Een workflow kan oude resultaten teruggeven voor de nieuwe referentieverwerking |
| P2 | 6. Runregistratie is onvolledig en deels onjuist | Achteraf is niet altijd vast te stellen wat werkelijk is uitgevoerd |
| P2 | 7. De voorgeschreven audit ontbreekt in de character-edit-route | `completed` betekent daar geen visuele acceptatie volgens het plan |
| P2 | 8. Native VAE-resolutie is al gedeelde Qwen-werking, maar slechts beperkt beproefd | Een experiment bepaalt ook het gedrag voor grote en meervoudige referenties |

P1 betekent hier: herstellen voor betrouwbare vergelijkingen op de getroffen route. Het zijn geen claims dat iedere bestaande run fout is. P2 zijn aantoonbare hiaten in evaluatie, registratie of onderbouwing.

**1. Klein verandert referenties met een impliciete anime-upscaler en trekt ze naar het uitvoerformaat.**

In [flux2_klein.py:842](/home/boaz/aigen/aigen/generation/flux2_klein.py:842) wordt voor iedere referentie waarvan de afmetingen afwijken van het doelcanvas een nieuwe `IllustrationUpscaler` geladen. Dit gebeurt ook als de referentie groter is dan het uitvoerbeeld. De upscaler werkt op de oorspronkelijke invoer, produceert een natuurlijke 4× vergroting en [resizet die vervolgens exact naar `target_size`](/home/boaz/aigen/aigen/generation/image_upscale.py:269). Verschillende beeldverhoudingen worden dus uitgerekt. Daarna wordt het resultaat alsnog tot maximaal ongeveer 1 MP teruggebracht.

Reproductie: een synthetische referentie van 128×256 wordt voor een canvas van 512×512 eerst naar 512×512 getrokken. Dit is een verandering van 1:2 naar 1:1, geen afrondingsverschil voor latentpacking. Alle referenties krijgen de uitvoerverhouding, ook wanneer ze oorspronkelijk verschillende verhoudingen hebben. De publieke `image-edit`-route resolveert altijd een doelcanvas en bereikt deze code ook zonder expliciete `--width`/`--height`.

De [Diffusers-implementatie op de lokaal geïnstalleerde commit](https://github.com/huggingface/diffusers/blob/d4b2bfc04615787846bac271e74a1c033e714e67/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L760) verwerkt iedere referentie vanuit haar eigen afmetingen en begrenst alleen beelden boven 1024² pixels voordat latent-alignment wordt toegepast. Zij voert hier geen neurale upscaler uit en trekt referenties niet naar het uitvoercanvas. De lokale vervorming komt uit aigen.

Ook de kosten zijn substantieel. Bij een bron van 3840×3890 maakt de impliciete 4× upscaler eerst een beeld van 15360×15560: 239.001.600 pixels. De twee verplichte float32 RGB-accumulatoren in [image_upscale.py:383](/home/boaz/aigen/aigen/generation/image_upscale.py:383) vragen samen **5,34 GiB**. Dat is alleen deze twee buffers, zonder model, activaties of tijdelijke normalisatietensors. Dit is een berekening uit de allocaties, geen gemeten GPU-piek. Bovendien staat Kleins VAE dan al op CUDA en wordt de upscaler per afwijkende referentie opnieuw geconstrueerd.

Herstelrichting: normale edit-referenties verwerken op hun eigen verhouding met de modelgebonden pixelbegroting. Een neurale upscale hoort als expliciete beeldbewerking in de workflow. Alleen een init-image voor img2img moet bewust bij de geometrie van de uitvoerlatents worden gebracht. Het exacte effect van de huidige upscaler op de gewenste tekenstijl vergt een visuele vergelijking; de ongewenste geometrische wijziging en het extra werk zijn al vastgesteld.

**2. Kleins batchcache maakt conditionering afhankelijk van de eerste taak.**

De [sleutel van de referentiecache](/home/boaz/aigen/aigen/generation/flux2_klein.py:505) bevat alleen de geordende bronpaden. De functie die de gecachte waarde maakt ontvangt wel de canvasbreedte en -hoogte en gebruikt die bij bovenstaande preprocessing. De workflow [groepeert Klein-taken zonder canvas in de batchsleutel](/home/boaz/aigen/aigen/workflow_execution.py:554), dus de fout is via de gewone workflow bereikbaar.

De CPU-proef gebruikt dezelfde bron in twee taken. Taak B vraagt 1024×1024:

| Uitvoering | Referentie die taak B krijgt |
| --- | --- |
| A van 512×512, daarna B in dezelfde batch | 512×512 |
| B afzonderlijk | 1024×1024 |
| B eerst in de batch | 1024×1024 |

De echte workflow-batchsleutel is voor beide canvassen gelijk. De proef toont andere conditionering; er is geen GPU-claim over de resulterende pixels. Dit schendt wel de aanname van de globale cache dat een node-uitkomst alleen van haar eigen invoer en instellingen afhangt.

Herstelrichting: maak normale referentievoorbereiding canvas-onafhankelijk; neem bij voorbereiding die werkelijk van het canvas afhangt, zoals init-latents, die afhankelijkheid in de betreffende cachesleutel op. Het uitschakelen van alle batching is niet nodig.

**3. `strength` schakelt de overige referenties uit.**

Bij [flux2_klein.py:527](/home/boaz/aigen/aigen/generation/flux2_klein.py:527) wordt alleen `condition_images[0]` tot een init-latent geëncodeerd. De normale `prepare_image_latents`-aanroep staat in een `elif` en wordt overgeslagen. Daardoor blijven `image_latents` en `image_latent_ids` leeg voor de hele bundel. De extra referenties zijn daarvoor wel geladen en eventueel door de neurale upscaler gehaald.

CPU-reproductie: twee opgegeven beelden, twee upscale-aanroepen, één VAE-init-encoding en geen referentiecontext. De TUI/CLI beschrijft `strength` als denoise strength en accepteert nog steeds meerdere referenties; er wordt niet gemeld dat die andere invoer wegvalt. Ook `reference_count` telt de opgegeven paden en vertegenwoordigt hier dus niet het aantal daadwerkelijk gebruikte referenties.

Herstelrichting: onderscheid de eerste bron als init-image van de overige visuele referenties en voer die overige referenties door naar de editor. Dit betreft Kleins aangepaste img2img-pad; de normale route zonder `strength` heeft dit specifieke probleem niet.

**4. Bij `strength` kan de init-image een ander tokenraster hebben dan het uitvoercanvas.**

Kleins referentievoorbereiding begrenst ook de init-image tot 1024² pixels. [De denoiser maakt eerst positie-ID's voor het gevraagde canvas](/home/boaz/aigen/aigen/generation/flux2_klein.py:644), vervangt vervolgens de latents door [de geëncodeerde init-image](/home/boaz/aigen/aigen/generation/flux2_klein.py:669), maar vervangt die positie-ID's niet.

Een expliciet canvas van 1536×1536 is toegestaan door de invoervalidatie. Met `strength=0.5` bevat de naar 1024×1024 begrensde init-image **4096 latente tokens**, terwijl de uitvoer-ID's **9216 posities** bevatten. De echte Diffusers-unpackfunctie faalt op die combinatie in de CPU-proef. Op de GPU kan de mismatch al eerder bij positional encoding/attention optreden; een volledige GPU-run is hiervoor niet uitgevoerd. De cachefout uit bevinding 2 kan hetzelfde soort inconsistentie ook bij kleinere canvassen veroorzaken.

Herstelrichting: de init-image en haar ruislatents moeten dezelfde geometrie als het uitvoercanvas krijgen. De gewone referentie-pixelbegroting hoort niet ook de uitvoer-init te begrenzen. Alleen positie-ID's aanpassen zou het aangevraagde canvas stilzwijgend veranderen en lost het contract niet op.

**5. De workflowcache ziet onze gewijzigde Qwen-referentieverwerking niet.**

De [Qwen-provenance](/home/boaz/aigen/aigen/workflow_provenance.py:83) bevat executorrevisie `3`, batchversie `1` en gepinde externe model/runtime-revisies. De lokale conditioner, staging en worker hebben geen eigen revisie in deze sleutel. De nieuwe preprocessing-policy wordt alleen achteraf in het workerresultaat geschreven.

De CPU-proef vergelijkt de provenance-code van HEAD met de werkboom, registreert dat alle drie gewijzigde Qwen-ownerbestanden andere hashes hebben en berekent met dezelfde bron/instructie/seed/instellingen exact dezelfde node-sleutel:

`356f518b125c4bb03c26f92c5029fbbbbe37433f61a094077bcdd9c1964c8fdb`

Dit bewijst dat een bestaande matchende workflowcache-entry als hit kan terugkomen zonder de nieuwe code uit te voeren. Het bewijst niet dat er voor een specifieke echte workflow al zo'n entry aanwezig is. De eerder uitgevoerde losse sprite-experimenten gebruikten deze workflowcache niet; hun resultaten worden hierdoor niet ongeldig.

Herstelrichting: geef de lokale backendsemantiek/preprocessing een expliciete revisie in de bestaande provenance. Laat die veranderen bij wijzigingen die de output kunnen veranderen. Een aparte implementatierevisie is hier geschikter dan de versie van het batch-JSON-schema. Neem ook werkelijk gebruikte extra modellen en relevante runtime-revisies op; Kleins impliciete upscaler staat bijvoorbeeld niet in zijn modelinventaris.

**6. Runmetadata is niet betrouwbaar genoeg voor experimenten via de TUI en workflows.**

Drie concrete paden:

- Qwen schrijft [globaal altijd VOSR als postprocess](/home/boaz/aigen/aigen/generation/qwen_image_edit_identity.py:782), evenals in [de canvasmetadata](/home/boaz/aigen/aigen/generation/qwen_image_edit_identity.py:2050). In twee bestaande sprite-runs zegt de output zelf `mode: none` en zijn raw/final SHA-256 gelijk. De beelden zijn dus onbewerkt, maar de globale beschrijving is onjuist.
- [Qwens batchadapter](/home/boaz/aigen/aigen/generation/image_edit_batch.py:329) maakt het uitgebreide resultaat in een tijdelijke directory, kopieert alleen de beelden en retourneert pad, afmetingen en seed. Timings, geheugenmeting, workerlog en effectieve referentiedimensies verdwijnen met die directory. De workflow bewaart wel de graaf, inputs, seeds, outputs en cache-provenance; de uitvoeringsdetails verdwijnen.
- Kleins gewone `image-edit`-route [slaat de PNG op](/home/boaz/aigen/aigen/generation/flux2_klein.py:805) en [schrijft het API-resultaat naar stdout](/home/boaz/aigen/aigen/image_edit_commands.py:129). Zij schrijft geen runmanifest met de instructie en geordende bronidentiteiten. De [TUI leest stdout in een tijdelijke deque](/home/boaz/aigen/aigen/image_tui.py:1647) en bewaart die niet als runrecord. Het bewaren van de laatst gebruikte formulierstand is geen historie per gegenereerd beeld.

Herstelrichting: sla bij iedere run de genormaliseerde aanvraag en het effectieve backendresultaat naast de output op. Behoud in batches de relevante uitvoeringsmetadata in de bestaande resultaatstructuur. Leid postprocessvelden af van de uitgevoerde postprocess-stap. De code heeft hiervoor al model-/runtime-provenancehelpers; er is geen extra los registratiesysteem nodig.

**7. De audit uit het character-pipelineplan is niet geïntegreerd.**

[PLAN.md:296](/home/boaz/aigen/docs/PLAN.md:296) vereist een beperkte audit op raw candidates, met selectie, eventueel een begrensde nieuwe poging en expliciet falen als geen kandidaat slaagt. De [actieve character-edit-route](/home/boaz/aigen/aigen/character_qwen_edit.py:221) gaat rechtstreeks naar generatie en postprocess. In de [LightX2V-owner](/home/boaz/aigen/aigen/generation/qwen_image_edit_identity.py:718) volgt postprocess rechtstreeks op raw generatie. Er is op dit pad geen audit-VLM, visuele verdictregistratie of best-of-N-selectie.

Dit is een implementatiegat ten opzichte van het plan. Het is geen bewijs dat de gegenereerde beelden visueel slecht zijn en geen eis dat de losse experimentele `image-edit`-API zelf automatisch een character-audit moet uitvoeren. Momenteel is de gebruiker de visuele beoordelaar.

Een daarmee samenhangende documentatiefout: [PLAN.md:81](/home/boaz/aigen/docs/PLAN.md:81) beschrijft een externe VLM die de huidige generatieprompt uit een dossier zou maken. De [huidige code](/home/boaz/aigen/aigen/character_qwen_edit.py:285) geeft juist de gebruikersinstructie rechtstreeks door. Het oude probleem is daar al verwijderd; het document moet dat als historische diagnose beschrijven.

**8. Native VAE-resolutie is een experiment met een te brede standaardwerking.**

Onze huidige [conditioner](/home/boaz/aigen/aigen/generation/qwen_image_edit_conditioner.py:29) houdt de upstream semantische beeldbegroting aan, maar verwerkt de VAE-referentie op bronresolutie met alignment. Dat geldt gedeeld voor de Qwen-2511-routes. De [gepinde LightX2V-encoder](https://github.com/ModelTC/LightX2V/blob/b96309e82899145aebd8ecf95c387894aba66b1e/lightx2v/models/input_encoders/hf/qwen25/qwen25_vlforconditionalgeneration.py) gebruikt normaal een afzonderlijk VAE-doel van 1024² pixels. Dat is een preprocessingkeuze, geen bewijs dat het model uitsluitend op precies 1 MP is getraind.

De [bestaande gecontroleerde vergelijking](/home/boaz/aigen/runs/sprite-front-qwen2511-native-vae-40step-v1/comparison.md) ondersteunt het terugbrengen van het semantische kanaal naar upstream: frontaanzicht en witte achtergrond kwamen bij dezelfde seed terug toen alleen die uitbreiding werd teruggedraaid. Zij toont geen consistente stijl over meerdere seeds aan en rechtvaardigt nog geen onbeperkte native VAE-resolutie voor bijvoorbeeld een 15 MP multiview-sheet.

Daar komt bij dat [de VRAM-reservering voor residente Qwen-blokken](/home/boaz/aigen/aigen/generation/qwen_image_edit_lightx2v_worker.py:443) uitsluitend van het uitvoercanvas afhangt. Aantal en afmetingen van VAE-referenties komen niet in die berekening voor. Dat is een onbeproefde resource-aanname voor grotere bundels; in de uitgevoerde single-reference ablation is geen OOM vastgesteld.

Herstelrichting: maak de overeengekomen referentiebegroting expliciet, met bronresolutie als afzonderlijke experimentele keuze en dezelfde keuze in request, provenance en resultaat. Het oorspronkelijke grotere bronbestand blijft nuttig als master voor uitsneden of goede verkleining. Een brede standaardwijziging verdient herhaalde visuele evaluatie en geheugenmetingen op de ondersteunde invoervarianten.

Er zijn ook belangrijke onderdelen die de review bevestigt. Qwen ontvangt op het actieve pad beelden en de gebruikersinstructie, geen gegenereerde tekstbeschrijving van het personage. Beide modellen worden expliciet gekozen zonder stille onderlinge fallback. De Qwen-worker scheidt conditioner, VAE-encoding, denoising en VAE-decoding; seeds worden per output gezet. De globale workflowcache valideert bestanden en publiceert resultaten atomair. De TUI start een eigen procesgroep en stopt die gericht; de workflow heeft een interruptpad dat een onderbroken run vastlegt. Dit zijn bevindingen uit code-inspectie, geen volledige stresstest van iedere backend of iedere annuleringssituatie.

De herstelvolgorde is: eerst Kleins preprocessing en de daarvan afhankelijke caches/init-latents; vervolgens implementatieprovenance en runregistratie; daarna de overeengekomen Qwen-referentiepolicy en de ontbrekende auditintegratie. Die eerste punten bepalen of een experiment doet wat zijn instellingen beloven.

Voor de oorspronkelijke stijlkwestie blijft de conclusie begrensd: de afzonderlijke Qwen-spriteproeven tonen nog geen stabiele specifieke tekenstijl over seeds. De Klein-bugs verklaren die Qwen-uitkomsten niet. Ze geven wel concrete redenen waarom bredere vergelijkingen tussen backends en workflows momenteel extra variabelen bevatten die niet door de gebruiker zijn gekozen. Het gladde tussenbeeld blijft een toetsbare hypothese; een LoRA is op basis van deze review nog niet als noodzakelijke oplossing aangetoond.

Validatie: alle assertions in de CPU-proeven zijn geslaagd. Reproduceren met `.venv/bin/python runs/evidence/pipeline-review-2026-09-05/probes.py`. De proeven vereisen de huidige werkboom, de geïnstalleerde generation-dependencies en de twee genoemde lokale Qwen-resultaten; het zijn bewijsproeven voor deze review, geen algemene regressiesuite voor een toekomstige implementatie.
