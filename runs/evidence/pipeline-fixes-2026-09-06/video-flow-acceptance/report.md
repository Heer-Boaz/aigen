# Video workflow acceptance, 2026-09-06

Stage 4 is implemented and the installed AnimeGen and LTX routes passed the
real saved Textual → CLI → generation → extraction → CPU frame processing →
assembly → Results/export flow. Hunyuan has a distinct graph/configuration and
early runtime validation; its absent models were not downloaded or run.

## Evidence

- [Exact initial prompt, input identity and settings](manifest.json),
  [prompt review](prompt-review.md). Both batch extensions use seeds 83 and 84
  with that unchanged prompt and input; each run saves its request and graph.
- [AnimeGen single-seed acceptance](anime-v1/acceptance.json),
  [AnimeGen batch acceptance](anime-v2/acceptance.json),
  [LTX batch acceptance](ltx-v1/acceptance.json).
- [Machine-readable measurements](measurements.json),
  [TUI harness](tui_video_probe.py).
- [52 passing CPU regressions](../video-flow-regression-v2.log),
  [four specific video batching regressions](../video-batching-cpu-v2.log).
  Neural inference is explicitly doubled in CPU batching tests; media files,
  compiler, executor, cache and frame timestamps are real.

| Route | Raw output per seed | Processed export | Observed total VRAM peak | Native allocated / reserved peak |
| --- | --- | --- | --- | --- |
| AnimeGen, seeds 83/84 | 544×720, 33 frames, 16 fps, 2.0625 s | 272×360, same frames/timing | 13,894 MiB | 8,063 / 12,672 MiB |
| LTX NVFP4, seeds 83/84 | 576×704, 33 frames, 24 fps, 1.375 s | 288×352, same frames/timing | 5,664 MiB | 5,118.1 / 5,232 MiB |

Each route generated silent video as declared. The CPU acceptance separately
tests audio preservation, removal and replacement with actual audio waveforms,
offsets, absolute stream selection and shorter/longer tracks. VFR timestamps,
frame order, fractional fps, source relocation, current audio/timing through
pixel-cache reuse and exported frame archive reassembly all passed.

LTX's requested 513×641 canvas became 576×704 before white padding. The three
input positions are 0, 16 and 32; native records contain start, end and one
internal reference with `frames_positions="17"`. The same approved image was
used at each position; this verifies index transport and fitting, not an
arbitrary motion transition between different poses.

Both seeds share the same durable execution directory and model lifecycle.
Each complete video was published separately before the sweep ended. LTX's
second native record contains no model-loading phase. Its measured native
first/second durations were 124.988 / 32.053 s. AnimeGen loaded one pipeline;
its measured generation calls were 49.149 / 51.173 s after initialization.
Full workflow durations including initialization and assembly were 180.237 s
for LTX and 397.736 s for AnimeGen. The following cached assembly requests took
2.209 / 2.294 s and reused generation, extraction, processing and assembly.

AnimeGen seed 83 produced byte-identical decoded RGB video frames when run
alone and in the two-seed sweep. Both decoded frame-stream SHA256 values are
`b4586546eb0185a204a6715583bcf4b78505d0a32878df1a82cb1d982f92b828`.

Global system swap-out increased by about 17.7 MiB during the AnimeGen batch
and 113.2 MiB during the LTX batch (the earlier AnimeGen single run: 35.0 MiB).
These are functional timings, not clean offload benchmarks. Global counters
do not identify which process's pages were swapped. GPU availability was
checked before every neural run; no unrelated jobs were stopped.

## Visual inspection and review findings

The raw and processed contact sheets were inspected. AnimeGen blinks and moves
the character, but introduces unrequested grey/blue motion marks in both tested
seeds. LTX preserves the appearance under the three identical anchors and
stays almost still; this is not full acceptance of the requested movement.
Neither observation proves sprite-style consistency across seeds. Frame
processing is only a downstream-flow test, not a pixel-edge quality criterion.

Read-only code review found and resolved three concrete issues: WanGP's native
root-FFmpeg quarantine policy, Gemma's first-folder tokenizer lookup versus
separate model-config lookup, and missing graph seed-sweep reuse. Additional
reviewer CPU probes exercised actual AnimeGen wrapper cleanup and an actual
small LTX worker subprocess interrupted after its first result; cleanup and
termination propagated. The final review found no further concrete batching
or media-result issue.

An earlier CPU run exposed FFmpeg copying longer audio/container duration into
the video stream. Timeline extraction now uses decoded presentation PTS and
the final decoded frame's duration. The failing evidence was retained; the
corrected timing/audio cases pass in the final regression bundle.

The [workflow guide](../../../../docs/workflows.md) documents all routes,
keyframe fitting/indexing, audio policies, previews, export and portable frame
archives. The final stage-4 `git diff --check` passed. No commit or model
download was performed.
