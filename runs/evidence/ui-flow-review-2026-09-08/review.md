**Review van de volledige TUI-flow — 8 september 2026**

De canvasbediening heeft een bruikbare basis. De belangrijkste resterende problemen zitten in de samenhang tussen invoeren, uitvoeren, beoordelen en verderwerken, en in de prioriteit die deze taken op het scherm krijgen. Een aantal compacte layouts maakt hoofdhandelingen moeilijker te herkennen. De vaste inspectorbreedte en editorhoogte zijn ontwerpkeuzes die nog geen onderbouwing uit gebruikersonderzoek hebben.

Dit is een review van de huidige lokale checkout, inclusief de bestaande wijzigingen bovenop `4ef89bdfd47e8ff7e3f32d4d6a81328b491d4887`. Voor deze review is geen applicatiecode gewijzigd of gecommit.

**Aannames en werkwijze.** Het belangrijkste gebruik is iteratief: een flow opbouwen, instellingen wijzigen, varianten maken, vergelijken, kiezen en verder bewerken. 80×24 is meegenomen als klein scherm; 120×40 en 160×50 controleren of meer ruimte ook goed wordt benut. Dit zijn reviewformaten, geen afgeleide eis dat alles tegelijk op 80×24 zichtbaar moet zijn.

Alle hoofdtabbladen zijn geopend en vastgelegd: Images, Videos, SAM Edit, Post-processing en Workflows. Daarnaast zijn de workflow-editor, inspector, nodekiezer, contextmenu's, verbindingsdialog, bestandskiezer, SAM-canvas, foutmelding, naamdialog en Results onderzocht. Een synthetische CPU-flow is daadwerkelijk uitgevoerd tot een collectie; daarna zijn Results geopend en is een kandidaat gekozen. De downstream-uitvoering na die keuze is in deze review niet opnieuw getest. Er is geen GPU/model gebruikt. Screenshots en invoerproeven gebruiken de echte Textual-app met Pilot; fysieke toetscodering in Windows Terminal is hiermee niet bewezen. Dit is een taakgerichte expert-review, geen gebruikersonderzoek.

**1. P1 — Results kan een onzichtbare kandidaat laten kiezen.**

Op 80×24 opent Results met uitsluitend een afgekapte invoerafbeelding in beeld. De eerste kandidaat staat onder de viewport, maar is intern al gemarkeerd en `Select image` is actief. De gebruiker kan daardoor denken de zichtbare afbeelding te kiezen, terwijl de actie op kandidaat 0 werkt. Op een kandidaatbeeld klikken verandert de markering bovendien niet; daarvoor moet de aparte knop `Inspect` worden gevonden.

De oorzaak is structureel: invoer en kandidaten staan onder elkaar, gevolgd door een permanent JSON-paneel. Meer terminalruimte vergroot dat JSON-paneel mee. Ook op 160×50 zijn de outputbeelden bij openen niet volledig zichtbaar.

| Terminal | Hoogte beeldviewport | Hoogte permanent JSON-paneel | Kandidaten bij openen |
| --- | ---: | ---: | --- |
| 80×24 | 13 regels | 7 regels | Geen zichtbaar |
| 120×40 | 24 regels | 12 regels | Alleen bovenkant zichtbaar |
| 160×50 | 30 regels | 16 regels | Gedeeltelijk zichtbaar |

De invoertegel neemt 21 regels in; de kandidaatrij begint op schermregel 24, gerekend vanaf nul. Op 80×24 eindigt de beeldviewport op regel 15.

Advies: geef de actuele kandidaat, haar identiteit en de keuzehandeling voorrang. Maak de hele tegel bedienbaar en houd de actieve kandidaat zichtbaar. Bied bij voldoende breedte een vergelijking naast elkaar; gebruik op smalle schermen één grote beeldweergave met een duidelijke wissel tussen invoer en uitvoer. Toon seed, backend en opgeslagen keuze compact; open JSON en uitgebreide records op verzoek. Behoud het verschil tussen bekijken en definitief kiezen, en behoud toegang tot het origineel voor beoordeling op volledige resolutie.

Broncode: [tegelinteractie](/home/boaz/aigen/aigen/workflow_results_tui.py:49), [layout en acties](/home/boaz/aigen/aigen/workflow_results_tui.py:73), [automatische markering](/home/boaz/aigen/aigen/workflow_results_tui.py:181). Bewijs: [80×24](results-80x24.png), [160×50](results-160x50.png), [interactiemetingen](results-metrics.json).

**2. P1 — De hoofdformulieren verliezen geplakte promptregels.**

In zowel Images als Videos resulteert plakken van twee regels in alleen de eerste regel, zowel in het widget als in het formuliermodel. De promptvelden zijn daar nog single-line `Input`-widgets. De workflow-node gebruikt inmiddels een `TextArea` en bewaart beide regels. Een latere omzetting via `Open as workflow` kan tekst die eerder verloren ging niet herstellen.

Advies: geef vrije tekst in alle invoerroutes dezelfde multiline-bewerking en dezelfde bewaarsemantiek. Dit is een correctheidsprobleem dat vóór verdere layoutverfijning moet worden opgelost.

Broncode: [formuliervelden](/home/boaz/aigen/aigen/image_tui.py:179). Reproductie: plak `First line`, een newline en `Second line` in het promptveld. Bewijs: [editing-probes.json](editing-probes.json).

**3. P2 — De primaire uitvoeractie sluit niet aan op een menselijke keuzestap.**

In een flow met een nog niet ingevulde afbeeldingskeuze levert de prominente `Run` meteen een fout op: eerst een afbeelding kiezen en de collectie uitvoeren. De compiler bewaakt die afhankelijkheid terecht. De UI biedt de benodigde voorafgaande stap echter via het contextmenu van de collectie.

De huidige route is: collectie selecteren → contextmenu → `Run to here` → contextmenu → `Results` → `Inspect` → `Select image` → terug naar de editor → `Run`. Niet elke flow bevat deze keuzestap; dit is dus geen fout bij iedere eerste Run.

Advies: maak het uitvoerdoel en de relevante vervolgstap zichtbaar vanuit de echte workflowstatus. Bijvoorbeeld varianten uitvoeren tot de keuzestap, beschikbare resultaten beoordelen, of met de opgeslagen keuze doorgaan. Gebruik bestaande rungegevens, actuele outputs en opgeslagen keuzes; kies geen kandidaat automatisch. Contextmenu's blijven geschikt voor aanvullende nodehandelingen. Dat `Results` voor een nodetype mogelijk is, zegt nog niet dat er al bruikbare resultaten beschikbaar zijn.

Broncode: [actiebeschikbaarheid](/home/boaz/aigen/aigen/workflow_editor.py:508), [contextmenu](/home/boaz/aigen/aigen/workflow_editor.py:549), [compilerregel](/home/boaz/aigen/aigen/workflow_compilation.py:494). Bewijs: [Run vóór keuze](full-run-before-choice-80x24.png), [metingen](results-metrics.json).

**4. P2 — Knoppen passen geometrisch, maar verliezen hun betekenis.**

Op 80×24 verschijnt `Open as workflow` als `Open as`, `Save Config` als `Save` en `New character flow` als `New`. De Images-footer bevat dertien acties inclusief Quit en gebruikt vier schermregels. De grid schaalt de beschikbare breedte, maar bewaakt de leesbaarheid van de handeling onvoldoende.

Advies: kies zichtbare hoofdacties op basis van de taak en plaats aanvullende acties in herkenbare groepen. Labels moeten hun betekenis behouden. Een layoutelement dat binnen de schermrand valt is daarmee nog niet bruikbaar.

Broncode: [Images-footer](/home/boaz/aigen/aigen/image_tui_footer.py:62), [Workflows-footer](/home/boaz/aigen/aigen/image_tui_footer.py:155). Bewijs: [Images](images-80x24.png), [Workflows](workflows-80x24.png).

**5. P2 — Inspectorruimte en veldvolgorde volgen onvoldoende de taak.**

De inspector is vast 42 kolommen breed: 35% van een terminal van 120 kolommen. Ook zonder geselecteerde node blijft dit paneel daar open. Sluiten is alleen in de smalle drawerlayout bereikbaar. Bij een node staan eerst de globale workflownaam en verschillende titels, gevolgd door velden in schemavolgorde. De prompteditor krijgt steeds zes regels.

De waarden 42 en 6 zijn praktische implementatiekeuzes; geen geraadpleegde UX-bron schrijft ze voor. De grens tussen splitlayout en drawer gebruikt bovendien de minimuminhoudsbreedte van de inspector, terwijl het daadwerkelijke paneel breder wordt gezet.

Advies: toon documenteigenschappen bij documentselectie en node-eigenschappen bij nodeselectie. Groepeer kerninstellingen en geavanceerde instellingen volgens de taak, met behoud van inzicht in effectieve waarden. Maak het paneel op alle schermformaten te sluiten en de actieve tekst- of beeldbewerking te vergroten. Behoud drafts, selectie en focus bij een layoutwissel.

Broncode: [breedte en omslagpunt](/home/boaz/aigen/aigen/workflow_editor.py:61), [inspectorinhoud](/home/boaz/aigen/aigen/workflow_inspector.py:108), [vaste teksthoogte](/home/boaz/aigen/aigen/workflow_property_widgets.py:116). Bewijs: [canvas 120×40](canvas-default-120x40.png), [inspector 80×24](inspector-80x24.svg).

**6. P2 — Teruggaan is niet consistent.**

Escape sluit de nieuwe keuzemenu's en verbindingsdialog, maar niet Results, de foutmelding, de naamdialog, de bestandskiezer of het SAM-canvas. Hun expliciete sluitknoppen werken wel. Bij het wisselen tussen schermen moet de gebruiker daardoor steeds een andere uitweg zoeken.

Advies: definieer een consistente betekenis voor terug, sluiten en annuleren, inclusief focusherstel en de afhandeling van wijzigingen. De huidige selectie-/opslagsemantiek mag daarbij niet ongemerkt veranderen. Een samengestelde save-dialog kan daarnaast locatie en bestandsnaam samen behandelen; de huidige workflow-save vraagt die na elkaar.

Broncode: [algemene dialogs](/home/boaz/aigen/aigen/tui_dialogs.py:10), [bestandskiezer](/home/boaz/aigen/aigen/tui_file_browser.py:292), [SAM-dialog](/home/boaz/aigen/aigen/sam_prompt_dialog.py:11). Bewijs: [dialogmetingen](dialog-metrics.json), [Results- en naamdialogproeven](results-metrics.json).

**7. P3 — Een actieve tekstwijziging verschijnt nog niet als gewijzigd document.**

Een gewijzigde nodeprompt staat correct in een draft, terwijl de documenttitel nog geen ster toont en `buffer_dirty` nog false is. Dit is een verschil tussen zichtbare tekst en statusfeedback. Het is geen aangetoond gegevensverlies: de bestaande commitroutes verwerken de draft later.

Advies: laat de gewijzigde-status ook actieve drafts weerspiegelen, zonder bij iedere toetsaanslag de hele graph opnieuw te valideren of historie aan te maken.

Broncode: [documenttitel](/home/boaz/aigen/aigen/workflow_editor.py:768). Bewijs: [editing-probes.json](editing-probes.json).

**Keuzes om te behouden.** Direct nodes en poorten manipuleren past bij het ruimtelijke object dat wordt bewerkt. De compacte workflow-toolbar, doorzoekbare nodekiezer en contextmenu's voor aanvullende acties geven een bruikbare basis. Native tekstbewerking en het behouden van drafts over layoutwissels zijn eveneens goede keuzes. De SAM-dialog geeft de eigenlijke selectietaak op 80×24 al 70×16 cellen; dat is een beter voorbeeld van inhoud voorrang geven. Expliciete kandidaatkeuze, resultaatgeschiedenis en toegang tot originele bestanden blijven waardevol.

**Onderbouwing en ontwerprichting.** Progressive disclosure geeft veelgebruikte taken voorrang en maakt aanvullende opties later bereikbaar. Welke taken daarbij horen moet uit het gebruik volgen; alles in een contextmenu plaatsen volgt niet uit dat principe. NN/g waarschuwt ook dat een dwingende stapsgewijze flow slecht kan passen bij taken met veel heen-en-weerbeweging. Daarom past hier een blijvende werkruimte waarin opbouwen, instellingen en resultaten eenvoudig afwisselen. [NN/g: Progressive Disclosure](https://www.nngroup.com/articles/progressive-disclosure/)

Herkenbare labels, zichtbare status en consistente uitwegen zijn brede heuristieken. Ze bewijzen geen specifieke kolombreedte of optimale knopverdeling. [NN/g: 10 Usability Heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/)

Productievoorbeelden ondersteunen controle over de beschikbare werkruimte: VS Code laat panelen verplaatsen, verbergen en maximaliseren. Lazygit biedt normale, halve en volledige schermmodi, contextgebonden toetsbediening en een algemene Escape-actie. De toepasselijke les is dat de actieve taak meer ruimte kan krijgen met behoud van oriëntatie en toegang tot andere taken. Dat is een ontwerpafleiding voor deze app, geen bewijs dat een kopie van die layouts hier optimaal is. [VS Code: Custom Layout](https://code.visualstudio.com/docs/configure/custom-layout), [Lazygit: Keybindings](https://github.com/jesseduffield/lazygit/blob/master/docs/keybindings/Keybindings_en.md)

Mijn voorstel voor vervolgwerk is drie samenhangende slices:

1. **Betrouwbare invoer en bediening.** Multiline-tekst in alle routes behouden, betekenisvolle labels op kleine schermen, consistente terug-/sluitbediening en eerlijke wijzigingsstatus. Toets met echte paste-, save-, resize- en annuleringsevents.
2. **Uitvoeren → beoordelen → verdergaan.** Herontwerp Results rond kandidaten en expliciete keuze; maak uitvoerdoel en vervolgstap zichtbaar. Acceptatie: vanaf een flow met keuzestap varianten uitvoeren, invoer en uitvoer vergelijken, een zichtbare kandidaat kiezen en verder uitvoeren op zowel 80×24 als een ruim scherm.
3. **Samenhangende werkruimte en eigenschappen.** Maak canvas, eigenschappen en resultaten passend bij de actieve taak bereikbaar. Geef de inspector instelbare ruimte en inhoudelijke veldgroepen. Laat Images/Videos waar nuttig als snelle invoerroutes bestaan met dezelfde bewerkingssemantiek. Toets deze indeling vervolgens met de handelingen die de gebruiker werkelijk het vaakst doet.

De prioritering hierboven is een review-oordeel. De afgekapt weergegeven labels, verdwenen promptregels, viewportmetingen, selecties en Escape-uitkomsten zijn rechtstreeks gereproduceerd. Alle screenshots en JSON-metingen staan naast dit rapport. Bij viewportmetingen zijn echte rechthoekintersecties gebruikt: Textuals `is_on_screen` alleen bewijst niet dat een widget binnen een scrollviewport zichtbaar is.
