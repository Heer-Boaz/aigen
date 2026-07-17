# Pixal3D-route — plan 2D → textured GLB → pixel-art-shading

Status: lokaal werkende route per 2026-07-17. Aanvulling op `docs/PLAN.md` (de
karakter-wordt-nooit-tekst-regel geldt onverkort: de 3D-route krijgt beelden
als invoer, nooit een tekstbeschrijving van Jillian). De HF-Space heeft
bewezen dat Pixal3D voor Jillian een acceptabele reconstructie maakt; alleen
de export van de Space is kapot. De lokale route heeft inmiddels een werkende
textured-GLB-export opgeleverd. Dit document bewaart naast het vervolgplan de
installatiekennis die een volgende agent op een verse computer nodig heeft.

Qwen-2511-LoRA is geschrapt (referentiebeelden zijn al met Qwen gemaakt). De
v8-FLUX-LoRA (`assets/lora/JSEED/TRAINING_HANDOFF.md`, werkformule) is in dit
plan de generator van schone identiteits-inputs.

---

## Fase 0 — installatie-handoff voor een verse computer

Dit is bewust geen generieke installer. Pixal3D combineert een specifieke
PyTorch/CUDA-stack met zes lokaal gecompileerde extensies. Een installer die
alle Linux-distributies, drivers en compilers probeert te verbergen is zelf een
nieuw onderhoudsproject. Een volgende agent moet de officiële installatie
volgen, onderstaande bewezen baseline respecteren en machine-afwijkingen
expliciet oplossen.

### Bewezen baseline — niet gedachteloos "moderniseren"

| onderdeel | werkende waarde op 2026-07-17 |
|---|---|
| host | WSL2, Ubuntu 26.04, kernel `6.18.33.2-microsoft-standard-WSL2` |
| GPU | RTX 5070 Ti 16 GB, compute capability `12.0`, driver `610.74` |
| hostgeheugen | 30 GiB RAM + 40 GiB swap; succesvolle run piekte op circa 26 GB resident RAM |
| Python | `3.12.13` in de aparte venv `.venv-pixal3d` |
| PyTorch | `torch==2.9.1+cu130`, `torchvision==0.24.1+cu130` |
| CUDA-compiler | CUDA Toolkit `13.3` (`nvcc 13.3.73`), lokaal uitgepakt; host-GCC/G++ `15.2.0` |
| Pixal3D | commit `cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af` van de actuele hoofdbranch, niet `paper` |
| TRELLIS.2 | commit `75fbf0183001ed9876c8dbb35de6b68552ee08bd` inclusief `o-voxel` |
| Pixal-deps | `natten==0.21.0`, `utils3d==0.0.2`, `einops==0.8.0`, MoGe commit `07444410f1e33f402353b99d6ccd26bd31e469e8` |
| native extensies | nvdiffrast `253ac4fcea7de5f396371124af597e6cc957bfae`; nvdiffrec/renderutils `b296927cc7fd01c2ac1087c8065c4d7248f72da4`; CuMesh `12289e1062f0603f2f0d0771b02e1395d247f26f`; FlexGEMM `6dd94a859c26ee8246888502eada3dd8ad85532e` |

De omgeving is circa 12 GB. De al gedownloade modelcache is circa 25,5 GB
(Pixal3D 23 GB, MoGe 1,3 GB, DINOv3 1,2 GB). Houd op een verse computer
minstens 50 GB vrije schijfruimte aan voor omgeving, modellen, buildproducten
en één of meer GLB-runs.

### Installatievolgorde en beslispunten voor de volgende agent

1. Lees eerst de officiële installatie van
   [Pixal3D](https://github.com/TencentARC/Pixal3D) en
   [TRELLIS.2](https://github.com/microsoft/TRELLIS.2). Pin daarna de hierboven
   genoemde commits; bouw niet toevallig de nieuwste HEAD.
2. Zorg dat `git`, `uv`, GCC/G++, CMake en Ninja aanwezig zijn. Controleer
   vervolgens met `nvidia-smi` en PyTorch dat de NVIDIA-driver, GPU en compute
   capability zichtbaar zijn. Op de vaste RTX 50-series-doelkaart moeten
   `torch.cuda.get_device_capability()` en de build-arch beide `12.0` zijn.
3. Gebruik een eigen Python-3.12-venv. Meng deze omgeving niet met `.venv`,
   LightX2V of een LoRA-trainer: hun Torch-pins zijn niet hetzelfde.
4. Installeer eerst PyTorch 2.9.1/torchvision 0.24.1 vanaf de officiële
   `cu130`-index. Installeer daarna de gewone Pixal-requirements, nooit
   `requirements-hfdemo.txt`: die laatste is voor de H-series-HF-demo.
5. `requirements.txt` laat MoGe en Gradio ongepind. Voor een reproduceerbare
   CLI-omgeving moet MoGe op de commit uit de tabel worden vastgezet en moet de
   uiteindelijke package-resolutie worden vastgelegd. Gradio is voor
   `inference.py` niet bepalend. Vergeet `einops==0.8.0` niet; dit ontbreekt in
   de Pixal-requirements maar is nodig voor NAF.
6. Er is een echte CUDA Toolkit met `nvcc` nodig om de extensies te bouwen; de
   CUDA-runtime uit een Torch-wheel is daarvoor niet genoeg. Deze machine
   gebruikte CUDA 13.3 lokaal onder `.venv-pixal3d/cuda-root/`, zonder een
   systeembrede CUDA-installatie. De installatiemethode mag op een andere
   computer verschillen, maar `CUDA_HOME`, `PATH` en de library-paths moeten
   allemaal naar dezelfde toolkit wijzen.
7. Bouw voor deze kaart met `TORCH_CUDA_ARCH_LIST=12.0` en voor NATTEN tevens
   `NATTEN_CUDA_ARCH=12.0`. Gebruik maximaal acht parallelle build-workers.
   Bouw in deze volgorde: NATTEN, nvdiffrast, nvdiffrec/renderutils, CuMesh,
   FlexGEMM en ten slotte TRELLIS' `o-voxel`. Gebruik voor de native extensies
   `--no-build-isolation`; anders ziet de build niet gegarandeerd dezelfde
   Torch/CUDA-stack.
8. CUDA 13.3 wees tijdens de NATTEN-build één expressie in PyTorch 2.9.1 af.
   Pas uitsluitend wanneer die oude expressie aanwezig is de upstream
   [PyTorch-fix](https://github.com/pytorch/pytorch/commit/c2671b853e1553b258a30c15d66f9647faae7aee)
   toe in `torch/include/ATen/core/List_inl.h`:
   ```text
   typename decltype(impl_->list)::difference_type
   -> typename c10::detail::ListImpl::list_type::difference_type
   ```
   Als een toekomstige Torch-versie de fix al bevat: niets patchen. Als de
   omgeving een derde variant bevat: stoppen en onderzoeken, niet globaal tekst
   vervangen.
9. Installeer de door Pixal voorgeschreven
   [`utils3d-0.0.2`-wheel](https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl),
   niet willekeurig de huidige utils3d-repository.
10. Gebruik tijdens inference `ATTN_BACKEND=sdpa`; FlashAttention is voor deze
    route niet nodig en was geen onderdeel van de bewezen installatie.

Native builds zijn langzaam. Bewaar hun bron- en buildmappen en hervat na een
fout vanaf de falende extensie; de volledige omgeving verwijderen en opnieuw
beginnen verspilt de al geslaagde compilaties.

### Verplichte 16GB-wijziging

`--low_vram --resolution 1024` was op zichzelf onvoldoende. De upstream
texture-config gebruikt NAF 1024 en liep vóór texture sampling vast op 15.772
MiB GPU-geheugen. Alleen `tex_1024["naf_target_size"]` is daarom lokaal naar
512 gezet in `model_sources/Pixal3D/inference.py`, met deze comment erbij:

```python
# Required on 16 GB GPUs: 1024 NAF peaked at 15,772 MiB and failed before texture sampling.
"naf_target_size": 512,
```

Dit is een vaste eigenschap van onze 16GB-installatie, geen gebruikersoptie.
Niet opnieuw 1024 proberen, geen resolutie boven 1024 kiezen en geen CLI-vlag
toevoegen om deze grens alsnog makkelijk te omzeilen.

### Installatie valideren vóór een dure run

Een import van alleen `torch` bewijst niets. Controleer in één proces de native
modules `natten._libnatten`, `nvdiffrast.torch`, `_nvdiffrast_c`,
`nvdiffrec_render.renderutils._C`, `cumesh._C`, `cumesh._cubvh`,
`cumesh._cumesh_xatlas`, `flex_gemm.kernels.cuda` en `o_voxel._C`. Controleer
daarna vanuit de gepinde Pixal3D-checkout:

```bash
ATTN_BACKEND=sdpa /pad/naar/.venv-pixal3d/bin/python inference.py --help
```

Pas wanneer beide checks slagen en circa 50 GB schijfruimte beschikbaar is,
mag een agent modelgewichten laten downloaden en een generatie starten.

## Fase 1 — lokale reproductietest (uitgevoerd)

- Input: `assets/characters/jillian/a-pose.png`; de run-kopie heeft exact
  dezelfde SHA-256 (`733e0ef0b4dd85b95e2a178396ef4f10c5dbe51d3332c526f9681e1c302d8ce0`).
- Instellingen: seed 42, `ATTN_BACKEND=sdpa`, `--low_vram`, resolution 1024,
  texture-NAF 512 zoals hierboven vastgelegd.
- Artefacten: `runs/pixal3d/a_pose_naf512_v1/` bevat input, log, timing,
  VRAM-log, textured GLB en vaste front/side/back-renders.
- Resultaat: `character.glb` is 36.064.496 bytes en opent correct. De run duurde
  7:17 en de gemeten totale GPU-bezetting piekte op 9.548 MiB. Boaz beoordeelde
  het resultaat als eng, maar globaal wel het karakter en overeenkomstig met
  de webuitvoer.

Deze run bewijst de lokale installatie en export. Hij is geen definitieve
kwaliteits-go voor geometrie, kleur of proporties; die beoordeling hoort bij
de volgende fasen.

## Fase 2 — Input-hygiëne (één wijziging per iteratie)

Bekende defecten uit de Space-test en hun vermoedelijke oorzaak:

| defect | hypothese | test |
|---|---|---|
| zwevende handschoenen naast het lichaam | input-pose/uitsnede (handen los van romp-silhouet) | schone A-pose-input, armen los van de romp maar aan het lichaam, marge rondom |
| haar bijna zwart i.p.v. warmbruin | inputkleur of texture-bake | haarkleur van de input meten vs master; zo nodig andere input |

- De bewezen baseline-input is `assets/characters/jillian/a-pose.png`. De
  eerstvolgende gecontroleerde vergelijking is
  `assets/characters/jillian/a-pose-threequarter.jpg`. Pixal3D accepteert exact
  **één** afbeelding — geen multi-view-sheets in één canvas.
- Per iteratie precies één inputwijziging; output naar
  `runs/pixal3d/input_vXX/` met dezelfde artefacten als fase 1.

**Klaar-criterium fase 2:** GLB zonder zwevende geometrie en met haarkleur
binnen redelijke afstand van de masters, beoordeeld op de vaste
viewer-hoeken naast de masters (master-naast-output-review, zoals bij de
datasets).

## Fase 3 — Validatie van het kandidaat-model

- 360°-turnaround-renders op vaste hoeken; naast de masters leggen
  (front/side/back bestaan als masters — directe vergelijking).
- Proporties meten op de front-render: doel 4,52 koppen (DWPose + silhouet,
  zelfde meetprotocol als de LoRA-sweep).
- Texture-kleuren steekproefsgewijs vergelijken (jack, strik, kousen, haar).

**Klaar-criterium fase 3:** een GLB die op alle drie de assen (geometrie,
kleur, proporties) expliciet is beoordeeld, met een genoteerd go/no-go van
Boaz voor de shading-stap.

## Fase 4 — Later, expliciet buiten scope nu

- Rigging/animatie.
- Pixel-art-shading van de 3D-renders (koppeling met het
  budget-als-beeld-recept op de edit-route; de korrel komt van het spec-beeld,
  nooit van promptwoorden).
- Hogere resoluties, alternatieve forks, ComfyUI-integratie.

## Discipline (geldt voor elke fase)

- Eén variabele per experiment; elk run-resultaat in een eigen
  `runs/pixal3d/…`-map met input, output, screenshots en meetwaarden.
- Geen commits; Boaz commit zelf. Masters zijn heilig.
- GPU-runs alleen na go van Boaz per fase.
- Bij een defect: eerst bewijs verzamelen (welke input, welk artefact),
  dan pas een fix — geen fixes op vermoedens.

## Open vragen

1. Levert de three-quarter-input aantoonbaar betere geometrie dan de huidige
   A-pose-baseline?
2. Hoe stabiel is de export bij herhaalde runs met dezelfde input
   (determinisme/seed-gedrag van Pixal3D)?
