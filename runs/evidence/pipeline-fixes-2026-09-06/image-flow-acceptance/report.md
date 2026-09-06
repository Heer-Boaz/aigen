# Beeldflow: echte TUI-acceptatie

Klein en Qwen hebben ieder de volledige TUI → CLI → native model → cache →
vergelijking → opgeslagen keuze → herstart → vervolgbewerking doorlopen.
Dit is acceptatie van de applicatieflow, geen bewijs van stijlconsistentie
over willekeurige seeds of nieuwe referentiepakken.

## Uitgevoerde matrix

De exacte prompts en bronpaden staan in `manifest.json`; de onafhankelijke
promptreview staat in `prompt-review.md`. De auteur heeft beide daadwerkelijk
gekozen seed-72-afbeeldingen bekeken voordat seed 73 met die afbeelding als
enige input werd uitgevoerd. De opgeslagen manifests leggen inputidentiteit,
backend, sampler, canvas, stappen en seed vast vóór generatie.

| Route | Canvas / stappen / refs | Twee varianten | Beide uit cache | Alleen seed 71 → 74 | Vervolg seed 73 |
|---|---|---:|---:|---:|---:|
| Klein scaled FP8 | 768×1024 / 4 / 2 | 44,37 s | 1,17 s | 28,34 s | 31,35 s |
| Qwen 2511 Lightning | 1152×1536 / 8 / 1 | 113,55 s | 1,17 s | 62,43 s | 67,44 s |

De vervolgstap gebruikte één geselecteerde input. De tijden omvatten TUI/CLI
en modeluitvoering, maar niet de afzonderlijke VRAM-wachttijd. Het zijn
metingen van deze runs, geen geïsoleerde modelbenchmarks.

- Klein: native gemelde VRAM-piek 11257 MiB voor de varianten en 10310 MiB
  voor het vervolg. De bemonsterde totale GPU-piek was 13028 respectievelijk
  11809 MiB.
- Qwen: eigen denoise-allocatiepiek 12617,27 MiB voor de varianten en 12700,90
  MiB voor het vervolg. De bemonsterde totale GPU-piek was 14627 respectievelijk
  14767 MiB. De worker gebruikte 32 residente en 28 gestreamde blokken.
- Geen gemeten swap-out tijdens deze generaties. De host registreerde per run
  hoogstens 0,0078125 MiB swap-in. VRAM bevat ook eventuele andere GPU-belasting.
- Meerdere eerste pogingen zijn vóór modelstart geweigerd wegens bezette
  VRAM. Het acceptatiescript wacht nu op acht seconden voldoende vrije ruimte
  en controleert opnieuw vóór iedere uitvoering. Die geweigerde pogingen zijn
  bewaard; er zijn geen andere processen beëindigd.

## Vastgesteld gedrag

De proef bediende echte Textual-buttons en inspectorvelden. Eerst opgeslagen
en herladen, daarna uitgevoerd. Ongewijzigde varianten hergebruikten dezelfde
cache-artifacts. Bij een seedwijziging werd alleen die variant opnieuw gemaakt;
seed 72 bleef hergebruikt. De tweede kandidaat werd bewust geselecteerd om de
opslag van een concrete keuze te testen. Na herstart werden uitsluitend de
selectienode en vervolgnode uitgevoerd: geen hergeneratie van de voorouders.

De geëxporteerde originelen zijn inhoudelijk dezelfde bestanden als de gekozen
cache-artifacts. De vervolgbewerking maakte de achtergrond lichtgrijs; beide
gekozen beelden en hun vervolg zijn visueel geïnspecteerd. Verschillen tussen
Klein en Qwen hier zijn geen gecontroleerde backendkwaliteitsvergelijking:
referentieaantal, canvas en prompt verschillen.

De resultatenweergave en records/logs blijven na herstart toegankelijk.
`comparison.svg` legt de echte TUI vast. Voor Klein is ook 80×24-terminalformaat
getest (`comparison-80x24.svg`). De OS-openactie is met dispatch-mocking getest;
er is geen extern Windows-venster geopend voor deze acceptatie.

## Regressies en review

`../image-flow-regression.log`: 36 applicatietests geslaagd in 20,615 s.
`../image-flow-ui-final.log`: drie TUI-tests geslaagd in 9,204 s, inclusief de
nadien toegevoegde voortgangseigenaarschapstest. De gezamenlijke set bevat 37
verschillende tests. De aparte LightX2V-tests en bredere GPU-backendacceptatie
van stap 1 staan in `../gpu-acceptance/report.md`.

De onafhankelijke code-/lifecycle-review vond en hertestte:

- geschiedeniswissel met een al gequeueëde Inspect-actie op een oude kandidaat;
- mislukte geschiedenisload die oude resultaten actionable kon laten;
- gewijzigde historische bronbestanden die als oorspronkelijke input verschenen;
- witruimte en ontbrekende seedslots bij formulierimport.

Deze fouten zijn hersteld bij de view-, artifact- en formulier-eigenaren en
vastgelegd in regressies. Voltooide batchbeelden blijven na een latere fout
gepubliceerd; een CPU-generatiedouble test gericht dat alleen het ontbrekende
beeld wordt hervat. De gezonde native callbackketen is met de bovenstaande
echte Klein- en Qwen-runs uitgevoerd. Er is geen opzettelijke GPU-OOM opgewekt.

## Bewijsbestanden

- `klein-v2/acceptance-summary.json`, `qwen-v3/acceptance-summary.json`:
  statussen, tijden, totale GPU-samples en swapdelta's.
- `klein-v2/workflow.json`, `qwen-v3/workflow.json`: herlaadbare gekozen flows.
- Beide directories: `selected.png`, `continued.png`, `selected.json`,
  preflights, telemetrie, schermcapture, native recorddirs en cachemanifests.
- CLI-/TUI-uitleg: `docs/workflows.md` in de repository.

Geen commits, modeldownloads of verwijdering van projectmateriaal uitgevoerd.
