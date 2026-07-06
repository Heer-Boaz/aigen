# Brief voor Codex — hoe de refine/edit-route PixAI-achtig wordt

> Standalone brief. Zelfstandig leesbaar; vervangt `docs/PLAN.md` en `AGENTS.md`
> niet, maar wijst naar de juiste lezing ervan.

**Lees eerst zelf:** `docs/PLAN.md` §3 (task-based reference selection), §6
(candidate/refinement), de north star (PLAN regels 18–25), en `AGENTS.md`
(workflow + golden rule).

## De enige fout die twee keer gemaakt werd

De **conceptlaag** werd samengeklapt in de **mechanismelaag**. "Kies één
reference" (`--reference-role portrait` als kern-contract) werd leidend. Daardoor
leest de pipeline als: *reduceer naar één ref en hoop dat het goed komt*. Dat is
precies verkeerd.

Houd deze twee lagen strikt gescheiden:

1. **Conceptlaag (staande context):** elke operatie — edit én refine — draait
   áltijd op het volledige `reference_pack` + `identity_profile`. De
   karakterkennis is holistisch. Niets reduceert ooit tot één beeld.
2. **Mechanismelaag (transport):** Qwen-Image-Edit-2509 neemt een beperkte set
   beelden per call (PLAN §3). Voor deze refine-route is de projectkeuze
   **1–4 pack-beelden per call**. Dus je **routeert** per call 1–4 pack-beelden naar binnen.
   Routing ≠ reductie. Het pack blijft de bron; je kiest per taak welke subset
   erin gaat en waaróm.

## Het doelgedrag (observeerbaar, niet "PixAI doet intern X")

Uit de north star (PLAN 18–25): *N references + één instructie → één coherent
resultaat dat ál die references eert.* De intelligentie zit in de modellen, niet
in door-de-mens-gekozen refs. Beweer niets over PixAI's binnenkant — dat is niet
te verifiëren. Richt je op dit gedrag.

## Correctie — en let op de spiegel-fout

- **Onder-correctie (de gemaakte fout):** exact 1 ref als kern-contract. → fout.
- **Over-correctie (de valkuil nú):** "het model moet refs dynamisch selecteren,
  sloop de case→role-tabel." → óók fout, en een stille planwijziging. PLAN §3
  staat een **statische case→role-mapping in code expliciet toe**
  ("mechanism-level and may live in code"), want die verwijst naar *rollen die de
  VLM afleidde*, niet naar karakters.
- **Juist:** niet één, maar **1–4**, uitgedrukt als **routing met het volledige
  pack als staande context**. VLM-gedreven routing is het *latere* doel ("liefst
  via VLM zodra dat kan"), **geen eis nu**.

## Specifiek voor refine (Phase 4)

PLAN §6 zegt letterlijk: refine = **selected image + original refs [meervoud] +
mask/region + instruction**.

- Het **geselecteerde beeld** is het inpaint-canvas dat je repareert.
- De **refs** (task-geroute subset uit het volledige pack) + `identity_profile`
  zijn conditionering die identiteit bewaart.
- Het gemaskeerde gebied wordt herschilderd; het gezicht wordt **niet**
  "vervangen door de portrait-ref". Er is geen verplichte `--reference-role`-keuze
  als kern-contract.

## Het manifest-contract (wat de output altijd moet bewijzen)

Elke plan/run-output moet aantonen — nooit "gereduceerd tot één reference":

- volledige `reference_pack` als bron gerefereerd;
- `identity_profile_used: true` + `body_proportion_source`;
- de **geroute subset** (`refs_used`) mét per ref een reden/purpose en
  input-volgorde;
- `optional_missing_refs` waar van toepassing (geen hard-fail op ontbrekende
  optionele ref).

## Twee losse punten

- **`jillian` in de README:** verzin geen karakter-specifieke voorbeeldnamen in
  docs. Houd voorbeelden generiek (`<character-id>` / `<run-dir>`). Golden rule:
  character-agnostische code én docs.
- **De echte meta-fix (dit brandde de gebruiker twee keer):** volg de
  `AGENTS.md`-workflow. Vóór je code aanraakt: park het concept — (a) taak in
  eigen woorden, (b) exacte lijst files die je maakt/wijzigt, (c) 2–3 regels per
  file, (d) je aannames. Wacht op een expliciete **GO**. Geen code vóór GO. Een
  stille versimpeling of eigenhandige richting telt als een gefaalde taak.
