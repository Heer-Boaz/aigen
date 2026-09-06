# Video aansluiten op de workflow

Opdracht: stap 4 van het beoordeelde implementatieplan, op dezelfde graph,
compiler, executor, artifactcache en resultatenweergave. Stap 3 blijft een
zelfstandig bruikbare beeldflow. Geen ontbrekende modellen downloaden.

Vooraf besproken aannames en reviewcorrecties:

- AnimeGen, LTX en Hunyuan houden aparte configuraties en expliciete routes.
  LTX ontvangt beelden met 0-based frameposities; de bestaande native adapter
  bezit de eenmalige +1 voor interne WanGP-keyframes. Het effectieve LTX-canvas
  heeft veelvouden van 64, vastgesteld in de gepinde native modelcode.
- Fit/crop/pad gebeurt op het effectieve canvas. De bestaande AnimeGen-stretch
  blijft bij oude documenten behouden; nieuwe keuzes worden opgeslagen.
- Geordende framepixels en hun tijdlijn zijn onderscheiden artifactgegevens.
  Assemblage hangt af van beide. Pixelnabewerking hangt alleen af van pixels
  en bewerkingsinstellingen; actuele timing/audio wordt ook na cachehergebruik
  doorgedragen. Geen gecachte oude audio/timing uit een pixelbewerking.
- Tijdlijn bevat rationele tijdsbasis, PTS en duur per frame. VFR wordt niet
  ongemerkt CFR. Video bepaalt de eindduur, ook bij korte vervangende audio.
  Het einde is de laatste decoded PTS plus de decoded laatste frameduur;
  stream.duration kan door FFmpeg uit langere audio worden overgenomen en is
  daarom geen betrouwbare eindgrens. Ontbrekende laatste frameduur is een
  concrete fout; nominale fps is geen vervanging voor ontbrekende timing.
- Behouden audio is de expliciet vastgelegde eerste audiostream, met absolute
  containerstreamindex, bestandshash en tijdsoorsprong ten opzichte van video.
  Een audiobron kan een andere stream kiezen. Verwijderen en vervangen zijn
  expliciete assemblagekeuzes. LTX blijft een route zonder gegenereerde audio.
- Meet video-eigenschappen vóór publicatie. De cache houdt de bestaande
  image-identiteit; videoveranderingen krijgen specifieke implementatierevisies
  inclusief PyAV/FFmpeg/libavcodec. Oud cachemateriaal wordt niet verwijderd.
- Hunyuan krijgt CPU-contractvalidatie en vroege foutmelding voor ontbrekende
  modellen. Geen claim van neurale GPU-acceptatie zonder beschikbare gewichten.
- Compatibele AnimeGen/LTX-seeds delen hun bestaande native seed-sweep-owner.
  Voltooide video's worden afzonderlijk gemeten en gepubliceerd, ook wanneer
  een volgende seed faalt. Geen modelreload per kandidaat vanuit een formulier.

Productiecode vooraf bestudeerd:

- [FFmpeg concat demuxer](https://github.com/FFmpeg/FFmpeg/blob/master/libavformat/concatdec.c)
  en [demux timestamp handling](https://github.com/FFmpeg/FFmpeg/blob/master/fftools/ffmpeg_demux.c).
- [PyAV encode tests](https://github.com/PyAV-Org/PyAV/blob/main/tests/test_encode.py)
  en de geïnstalleerde PyAV 18 mux-owner: de muxer schaalt packet-tijdsbasis.
- [LTX inference](https://github.com/Lightricks/LTX-Video/blob/main/ltx_video/inference.py),
  gepinde WanGP `models/ltx2/ltx2.py` en `wgp.py` voor echte canvas/indexering.
- [Hunyuan inference](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/generate.py).
- [FFmpeg stream timing](https://github.com/FFmpeg/FFmpeg/blob/n8.0/libavformat/demux.c),
  [mpv presentation endpoint](https://github.com/mpv-player/mpv/blob/v0.40.0/player/video.c)
  en [decoded frame duration](https://github.com/mpv-player/mpv/blob/v0.40.0/video/decode/vd_lavc.c).
- [Diffusers local checkpoint shard resolution](https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/hub_utils.py)
  en [Wan pipeline lifecycle](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_i2v.py).

CPU-ontwerpproef: PTS 0,40,100,120 ms, duren 40,60,20,80 ms. PyAV H264 met
expliciete packetduur en `max_b_frames=0` behoudt exact alle tijden en 200 ms
eindduur. Alleen frame.duration of dezelfde packetduur met standaard B-frames
deed dat niet. De drie bestanden staan in `video-flow-design/` en zijn ook door
`review_native_refs` onafhankelijk teruggelezen.

Acceptatie: import/extractie/bewerking/assemblage op CPU, CFR en VFR, audiobehoud,
verwijderen/vervangen, verschillende audioduur/offset/streamindex, metadata bij
cachehergebruik, LTX-positie/canvascontract, ontbrekend Hunyuan-model vóór een
lange uitvoering. Daarna beschikbare neurale routes afzonderlijk via TUI/CLI.
