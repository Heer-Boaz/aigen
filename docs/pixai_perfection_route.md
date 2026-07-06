# Perfectieroute — van "hopen" naar "sturen"

> **Status: voorstel. Nog geen code.** Gegrond in `docs/PLAN.md` §5 (hidden
> pose/region/segmentation conditioning) en Fase 5, en in wat feitelijk in de
> repo staat (onderaan geverifieerd). Vereist expliciete **GO** per stap. Rond
> niets af op een exit-code; de gate is een mens die de contact sheet beoordeelt.

## De eerlijke kern: het zijn twee losse problemen

De uitkomst faalt op (mogelijk) twee dingen die je niet door elkaar mag halen:

1. **Verkeerd aanzicht/oriëntatie** — je vroeg rechter-profiel en kreeg een
   vooraanzicht. Dat is een *correctheids*-probleem.
2. **Framing/marge** — de figuur staat te groot/ingezoomd (of afgesneden) in
   beeld, terwijl de referentie klein-met-marge is. Dat is een *compositie*-probleem.

Deze hebben **verschillende oorzaken en verschillende fixes**. De valkuil is er
één fix op plakken en de andere ongemerkt laten staan.

## Stap 0 — randvoorwaarden (vóór welke run dan ook)

- **Wie draait het:** de GPU-runs draaien bij jou/Codex, niet bij mij (mijn
  shell mag de GPU-pipeline niet starten). Ik lever alleen de opdracht + de
  beoordeling.
- **Gewichten:** de hogere ladder-treden zijn los van de installer (zie PLAN).
  Controleer/haal eerst: `nunchaku-qwen-edit-2509-r32-8step`, `...-r32` (vol),
  `...-r128`. Anders stalt het experiment op ontbrekende modellen.

## Stap 1 — het onderscheidende experiment (géén nieuwe code)

Draai op de **huidige** pipeline dezelfde refs, maar klim de kwaliteitsladder.
Nu draai je 4-step — de snelste, laagste-kwaliteitstrede.

- Cases: `front`, `right_profile`, **en `three_quarter`** (zie waarom hieronder).
- Modellen: eerst `-r32-8step`, daarna `-r32` (vol). Zelfde refs/seed.
- Beoordeel per probleem apart: (a) klopt de oriëntatie nu? (b) staat de figuur
  nu volledig en met marge?

**Wat dit wél en niet kan** (geen loze belofte):
- Meer stappen / hogere trede verbetert plausibel de **oriëntatie-trouw** en de
  detailkwaliteit.
- Er is **geen sterke reden** dat het vanzelf **marge/compositie** toevoegt. Als
  de oriëntatie goed komt maar de framing te strak blijft, is dat de winst van
  dit experiment: het vertelt je *welk* probleem je overhoudt.

## Stap 1 is al grotendeels beantwoord door bestaande runs

Er stonden al runs op elke trede (`4step`, `8step`, `full_r32`, `r128`).
**Gemeten** framing (afstand content → canvasrand; 0px = afgesneden), niet op het
oog, voor het `right_profile`-geval:

| Trede | Aanzicht | Ondermarge | Framing |
|---|---|---|---|
| 4-step (`framing_ratio_fix_...`) | **fout** — vooraanzicht | 0px (+boven 0px) | **afgesneden boven+onder** |
| vol r32 (`right_profile_smoke_full_r32`) | **fout** — vooraan/driekwart | 0px | **afgesneden onder** (voeten) |
| r128 (`right_profile_smoke_r128`) | **fout** — vooraanzicht | 0px | **afgesneden onder** (voeten) |

**Conclusie:** de ladder lost **noch het aanzicht noch de framing** op. Op elke
trede — tot en met r128 — komt het geval er vooraan uit i.p.v. rechter-profiel,
én lopen de voeten systematisch van het canvas. Tekst + refs sturen de camera
niet, en het model plaatst de figuur consequent te laag/te groot. Beide gebreken
zijn dus **trede-onafhankelijk**.

### Hoe de framing is getoetst (meetmethode, reproduceerbaar)

Niet op het oog — dat gaf twee keer een verkeerde lezing. De framing is
**gemeten** met een pixel-check op de output-PNG:

1. **Achtergrondkleur bepalen** uit de vier hoeken (8×8px): mediaan-RGB. Dit is
   nodig omdat de achtergrond niet puur wit is (r128/r32 ≈ grijs `[232,231,232]`,
   4-step ≈ wit `[254,253,254]`) — een vaste wit-drempel telt de grijze
   achtergrond ten onrechte als figuur.
2. **Content-masker:** pixel hoort bij de figuur als de euclidische RGB-afstand
   tot de achtergrondkleur `> 30`.
3. **Bounding box** van dat masker → afstand van de figuur tot elke canvasrand
   (boven/onder/links/rechts, in px).
4. **Regel:** marge `≤ 2px` op een rand = de figuur is dáár **afgesneden**.

Gemeten resultaat: r128 en vol r32 → ondermarge `0px` (voeten afgesneden);
4-step → boven- én ondermarge `0px`. Zijmarges waren telkens ruim (±115–133px).

Het script staat niet in de repo (read-only toets); de methode hierboven is
compleet genoeg om 'm te herbouwen. Voor de definitieve acceptatie kan dezelfde
bbox eventueel uit een SAM2-masker komen i.p.v. de hoek-achtergrond-heuristiek —
strenger, maar dezelfde marge-regel.

## Beslis-gate (na Stap 1)

- ~~Oriëntatie + framing beide goed op vol r32~~ → **weerlegd door de runs.**
- **Oriëntatie nog fout op vol r32/r128** → bevestigd. Het model volgt het
  gevraagde aanzicht niet op refs alleen. → Stap 2 is verdiend.
- PLAN §5 ("identity views hebben geen conditionering nodig") is hiermee
  **empirisch weerlegd** voor niet-frontale aanzichten. Conditionering naar voren
  halen vereist jouw expliciete GO (afwijking van de PLAN-fasevolgorde).

## Spanning met het PLAN (bewust niet stil opgelost)

PLAN §5 stelt: *"Identity-only views: no pose/region conditioning needed."* Stap
1 is precies de **toets van die aanname**:
- Komt vol r32 goed → aanname klopt, niets aan de hand.
- Blijft het falen → de aanname is **empirisch weerlegd**, en conditionering
  naar voren halen in de identity-view-fase is *verdiend* — maar dat is een
  **afwijking van het PLAN** en vereist jouw expliciete akkoord. Het experiment
  verdient de afwijking; ik herinterpreteer het plan niet op eigen houtje.

## Stap 2 — sturing die de modellen zélf leveren (alleen ná GO)

**Kernregel (gouden regel / AGENTS.md regel 2): wij bouwen geen mal.** Geen
pose-skelet, rand-, silhouet- of dieptemap die onze code uit de refs rendert —
dat is hand-geschreven geometrie en precies verboden. Elke structurele stap wordt
door een **model** geproduceerd; onze code routeert alleen beelden en instructies.

Volgorde, van zuiver/goedkoop naar zwaar:

**2a — eerst: is een gids überhaupt nodig?** Het huidige falen bewijst alleen dat
*dit recept* (tekst + 3 refs, vaste hand-prompt) de camera niet stuurt — niet dat
Qwen het niet **kan**. Dus eerst de modellen beter benutten, zónder enige gids:
- VLM-genormaliseerde instructie i.p.v. de vaste hand-geschreven case-prompt.
- Reference-routing die het gevraagde aanzicht laat domineren (voor een profiel de
  side-ref leidend, niet front+portrait die naar frontaal trekken).
Dit is de zuiverste toets en voegt geen enkele geometrie toe. Mogelijk is dit al
genoeg — dan is er geen Stap 2b.

**2b — alleen als 2a faalt: een model-geproduceerde tussenstap.** Laat een model
zelf de structuur/het doelaanzicht produceren, en voer díe modeloutput terug in de
edit. Onze code kiest en routeert; het visuele en structurele denken blijft in het
model. **Eerlijk open punt:** of Qwen zichzelf zo betrouwbaar kan sturen, is nog
niet aangetoond — te testen, niet aan te nemen.

**Als 2b een controlebeeld voert:** de ongebruikte `controlnet_block_samples`-hook
+ een echte Qwen-ControlNet betekent extra gewichten/VRAM bovenop een piek van
~15.7/16.3 GB — reëel risico op 16GB. De lichtere weg is de model-output als extra
beeld in `image=[...]`. 16GB blijft het plafond; PixAI-op-de-pixel is een apart
hardware-gesprek.

## Meetbare acceptatie (per trede, nooit "exit-code = klaar")

1. **Oriëntatie** klopt met het gevraagde aanzicht (visueel op de sheet).
2. Figuur **volledig zichtbaar** — alle vier de marges `> 2px` volgens de
   meetmethode hierboven ("Hoe de framing is getoetst"). Niets afgesneden.
3. Figuur vult een **doelfractie met marge** — zelfde bbox, marges binnen een
   afgesproken band (niet te strak, niet te klein).
4. **Identiteit** behouden (kapsel, jasje, strik, rok, kousen, laarzen).
5. Eindoordeel: **mens** op de contact sheet, met `three_quarter` erbij.

## Waarom `three_quarter` in de testset

`right_profile` is de PLAN-smoke-test, maar het is het énige aanzicht dat te
"faken" is door de bestaande side-ref te spiegelen — jij noemde de output zelf
"een kopie van reference05". Perfect op `right_profile` **bewijst niet** dat de
pipeline generaliseert. `three_quarter` vereist echt refs combineren; dáár blijkt
of het werkt. (Voor `right_profile` zelf is "lijkt op de side-ref" prima — dat is
wat je wilt.)

---

## Geverifieerd in de repo (bron voor dit plan)

- Edit-call voert alleen `image=reference_images` in; geen controlebeeld
  (`qwen_image_edit_identity.py:961`).
- Ladder-modellen bestaan: `-r32-4step` (default), `-r32-8step`, `-r32`, `-r128`
  (`qwen_image_edit_identity.py:223`).
- `controlnet_block_samples` wordt door de forward gethread maar met None gevoed
  (`qwen_image_edit_identity.py:1532`).
- Canvas leidt nu af uit de anchor-ref-verhouding (Codex' recente wijziging,
  `_case_canvas(..., anchor_image=...)`, `:762`) — verklaart 480×640 i.p.v.
  416×640, maar stuurt de figuur-grootte binnen dat canvas níet.
- Pose-tooling extraheert keypoints maar rendert (nog) geen skelet-controlebeeld
  (`keyframe_pose.py`).
