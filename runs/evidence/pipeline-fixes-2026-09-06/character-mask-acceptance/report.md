# Native masked Qwen and SAM workflow acceptance

The saved TUI → CLI → SAM2 → native masked Qwen-2511 Lightning → raw
audit → cache → original export flow completed. Two controlled strength
settings used identical source/mask bytes, instruction and seeds 101/102 at
768×1024. Strength 1.0 produced the requested pale blue background in both
candidates. Strength 0.6 did not, despite an automatic audit pass.

Exact inputs and settings are in [manifest.json](manifest.json) and
[strength-1.json](strength-1.json). Prompts were independently reviewed after
the reviewer read PLAN and prompting and inspected the images:
[initial review](prompt-review.md), [follow-up review](audit-v3-prompt-review.md).

## Actual execution

| Configuration | Denoising per candidate | Whole TUI run | Cache run | Sampled total GPU peak | Visual result |
| --- | ---: | ---: | ---: | ---: | --- |
| Strength 0.6, audit v2 | 5 steps | 85.75 s | 1.28 s | 14925 MiB | Mostly white background; blue mainly around the shadow |
| Strength 1.0, audit v3 | 8 steps | 81.86 s | 1.30 s | 15244 MiB | Both backgrounds pale blue |

The first edit reused SAM's mask. The second generated SAM2 again under the
final engine-specific runtime provenance and produced the identical mask SHA.
Both cache repetitions reused SAM and Qwen. Original export SHA matched the
published artifact. Each execution checked VRAM; the TUI harness required
eight seconds with at least 13000 MiB free. Total GPU usage includes other
Windows/WSL activity.

Native allocated peaks were identical in both runs: conditioning 9889.64 MiB,
VAE encoding 1813.03 MiB, denoise 13270.54 MiB and decode 3434.14 MiB.
Both used 36 resident and 24 streamed blocks. Extra mask/source/scratch state
was budgeted at 1.5 MiB. Packed source and mask shapes were `[1, 3072, 64]`;
the residency calculation counted 6345 output, reference and text tokens.

Neither run recorded swap-out. Global swap-in was 32 KiB and 156 KiB,
respectively. The higher-strength run's lower total time is not a speed
improvement: native weight reads changed from 35.48 GiB to zero with a warm
filesystem cache, and external GPU activity differed. Denoise increased from
12.29/7.95 seconds for seeds 101/102 to 16.93/12.74 seconds.

The initial standalone SAM2 run took 8.41 seconds. Its first export probe
selected the cutout instead of the mask and failed the harness SHA assertion.
Reopening Results and explicitly exporting the MASK passed. That first export
and log remain as evidence; the reviewed mask is `tui-v1/mask-original.png`.

## Sampling, preservation and defaults

LightX2V owns the strength suffix. The sampler initializes the whole canvas
from source/noise at the starting sigma, then restores protected latents at
each next sigma using the same seed noise. The mask uses the VAE grid and
native 2×2 packing, retaining its four distinct positions. Source latents
remain independent of the complete ordered visual reference bundle.

Every native decode is retained before final compositing at the original
source dimensions. Across all four real candidates, all 135969 protected RGB
pixels remained exactly equal. This tests the local preservation contract,
not pixel-art quality. CPU tests also cover alpha, grey mask edges and
non-aligned/capped source canvases.

The implementation follows the denoising boundary in
[Diffusers Qwen inpaint](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py)
and [ComfyUI KSamplerX0Inpaint](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/samplers.py),
adapted to the pinned LightX2V scheduler and sequential model lifecycle.

New regional Lightning edits default to strength 1.0 in the graph, SAM form,
CLI and shared execution owner. Independent review approved this bounded
choice. Strength changes both starting noise and the step suffix, so the
visual improvement cannot be attributed solely to the extra three steps.
Explicit lower strengths remain supported and saved values remain unchanged.
This is a local eight-step default; Diffusers' separate fifty-step Qwen
inpaint call signature defaults to 0.6.

Source/mask checksums survive graph save/load and cache publication. Identical
source bytes imported through a new artifact remain valid; a different source
fails before generation. Region plans pin both source and mask checksums.
Extra reference packs are optional and blank guidance retains profile defaults.

## Audit limitation

Audit v2 selected seed 101 and passed the incomplete strength-0.6 edit.
Independent visual review confirmed that the pair should fail on background
completion. Revision 3 makes that generic pass condition explicit, keeping
the flat `passed`, `image_index`, `regions` protocol.

A real audit-only comparison used the retained failed pair and the earlier
Klein front-view pair. The VLM again wrongly passed the background edit and
correctly selected the positive control's seed 91. The calibration assertion
therefore failed. Its [results](audit-v3-probe/result.json) remain intact.
Both judgments took 27.19 seconds with one model load, peaking at 9676.50 MiB
allocated and leaving 32 MiB allocated / 40 MiB reserved after close.

The wording clarification does not establish improved accuracy. The current
7B audit can miss an incomplete edit. The successful strength-1.0 results were
visually inspected as well. Fine white edging remains along parts of the
source/mask boundary; perfect segmentation or style consistency is not proven.

## Evidence and verification

- [Measurements and original record locations](measurements.json).
- [Successful image](tui-strength-1-v3/accepted.png) and
  [saved workflow](tui-strength-1-v3/workflow.json).
- [Incomplete image](tui-v1/accepted.png), retained with its original verdict
  and export filename.
- [TUI harness](tui_mask_probe.py) and [audit harness](audit_completion_probe.py).
- [73 passing app tests](../final-app-regression-73.log),
  [18 targeted checks after the default change](../character-mask-default-final.log),
  [five native scheduler tests](../character-masked-sampling-final.log) and
  [four Qwen reference tests](../character-qwen-reference-final.log).

The failed real-model calibration is not counted as a passing test. CPU audit
tests use explicit neural doubles for bounded retries, malformed-output
failure, lifetimes and publication. No missing models were downloaded,
unrelated GPU jobs stopped or changes committed.
