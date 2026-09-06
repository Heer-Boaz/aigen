# Beeldflow: uitvoering, resultaten en duurzame selectie

Opdracht: implementeer stap 2 (beeldcapabilities en resultaten) en stap 3 van
`../pipeline-review-full-2026-09-05/implementation-plan.md` op de bestaande
graph/compiler/executor/cache. De afgeronde GPU-acceptatie staat afzonderlijk
in `gpu-acceptance/report.md`.

Aannames en keuzes, vooraf door review_native_refs beoordeeld:

- Een variant is een bestaande ImageEditNode met een eigen seed. De editbuffer
  maakt fanout, collectie en selectie in één undo-transactie. Geen tweede
  scheduler of private seedcache.
- Een collectie draagt kandidaten zonder temporele betekenis. Selectie
  verwijst naar een duurzaam node-resultaat, producer-signature, outputpoort
  en artifactidentiteit. Verdergaan resolveert deze vaste keuze en stopt de
  afhankelijkheidstraversal op die grens. Nieuwe varianten uitvoeren heeft
  juist de collectie als target. Random voorouders worden dus niet opnieuw
  geloot door voortzetten.
- Document v4 heeft een workflow_id, buiten de nodecachesleutels. Legacy-ID
  komt van het oorspronkelijke documentpad. Save As behoudt een bestaande ID.
- Targetscope wordt vóór bestands-/backendvalidatie en seedresolutie bepaald.
  De volledige authored graph blijft in het snapshot; effectieve instellingen
  staan per uitgevoerd resultaat.
- De cache bewaart oorspronkelijke uitvoeringsdetails bij de image-identiteit;
  een runmanifest registreert nieuw gebruik of hergebruik. Batchrecords/logs
  worden eenmaal duurzaam geschreven. Een backend meldt ieder volledig
  opgeslagen beeld zodat publicatie niet op de rest van de batch hoeft te
  wachten. Denoised latents zijn nog geen voltooid beeld; de bestaande
  sequentiële transformer/VAE-lifecycle blijft behouden.
- Resultaatweergave leest manifests buiten het renderpad. Thumbnails gebruiken
  gelijke weergavegrenzen en veranderen geen modelreferenties. Selecteren,
  openen en exporteren gebruiken de originele output.

Productievoorbeelden vooraf bestudeerd:

- [ComfyUI caching](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_execution/caching.py):
  afzonderlijke node-identiteit en inputafhankelijkheden.
- [ComfyUI graph naar uitvoering](https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/src/utils/executionUtil.ts):
  authored graph en uitvoeringsaanvraag hebben verschillende verantwoordelijkheden.
- [InvokeAI batch/session queue](https://github.com/invoke-ai/InvokeAI/blob/main/invokeai/app/services/session_queue/session_queue_common.py):
  concrete variantwaarden met afzonderlijke resultaten en gedeelde uitvoering.

Acceptatie: legacy roundtrip; target zonder onvolledige downstreamvalidatie;
ongewijzigde seeds uit cache; keuze na herladen/herschikken; random → selectie
→ vervolg zonder hergeneratie; voltooide beelden behouden bij batchfout;
resultaten/logs na herstart; echte Textual → CLI → vergelijking → selectie →
vervolg → export. Daarna één beoordeelde neurale matrix, met VRAM-preflight.
