# Unmasked character flow acceptance

The real TUI → CLI → native generator → raw VLM audit → optional VOSR →
cache/export chain completed on both explicit installed character backbones.
Source identity, literal prompts, seed matrix and reviewed protocol are in
`manifest.json` and `prompt-review.md`. No missing models were downloaded.

| Route | Raw candidates | Selected | Published | Whole run | Cached run | Sampled total GPU peak |
|---|---|---|---|---:|---:|---:|
| Klein, 4 steps | seeds 91/92, 768×1024 | 91, first round | VOSR 1536×2048 | 102.82 s | 1.25 s | 11808 MiB |
| Qwen2511 Lightning, 8 steps, CFG1 | seeds 91/92, 1152×1536 | 91, first round | accepted raw | 195.77 s | 1.35 s | 14710 MiB |

Every run checked for at least 13000 MiB free VRAM, stable for eight seconds.
Measured outputs, backend phase timings, system swap deltas and original
artifact/report paths are retained in `measurements.json`. The GPU values are
sampled total usage, including other users of the device. Qwen reported its
own denoising allocation peak as 12985.03 MiB. Qwen also observed system swap
growth; these timings establish the functioning flow, not a clean offload
benchmark.

## Real protocol failure and correction

The first Klein run (`klein-v1`) retained both raw candidates but failed the
audit protocol: the VLM returned candidate index 2 for a two-candidate batch
whose zero-based candidate indices were 0 and 1. The prompt simultaneously
used one-based image numbers. No candidate was published and VOSR did not run.

Revision 2 uses a single image-numbering scheme. The selected number must
identify one of the candidate images; code then maps it immediately to the
durable candidate identity. It does not guess or repair an invalid result.
After separate prompt review and inspection of all three images, a real
audit-only repetition on the retained raws selected image 2 / seed 91.
That candidate retains the gloves; seed 92 has bare hands. The generation
prompt did not acquire a garment checklist.

The audit-only repetition took 17.07 s, peaked at 9561.87 MiB allocated, and
left 32 MiB allocated / 88 MiB reserved after the VLM closed. Both complete
TUI runs then used the reviewed protocol. Klein reproduced the selected raw
candidate's stable identity exactly. No additional neural generation was
needed for either cache run. Original export hashes matched the cache images.

## Visual inspection and limits

All raw candidates and the Klein VOSR output were inspected. The requested
front-facing full-body edit is present in both selected images. Klein's
selection avoids the visible glove loss in the other candidate; the VOSR
output retains the selected composition and outfit. Both Qwen candidates
retain gloves and the main outfit construction. Qwen changes the small mouth
expression from the source; the VLM did not flag that difference.

These runs prove an integrated detector/selector with useful behavior on this
case. They do not establish perfect semantic rejection, style consistency
over arbitrary seeds, or accuracy on unrelated reference packs. No pixel-art
crispness classifier or hand-written character geometry was introduced.

## CPU contracts

`character-app-regression.log`: 64 application tests pass, including twelve
character core/workflow tests. `character-qwen-preprocessing.log`: four tests
pass in the separate LightX2V environment. A first broad discovery attempt
used the application interpreter for the LightX2V-specific module and failed
its import; rerunning the documented environment-specific suites passed.

Character coverage includes bounded failure, strict malformed-response
failure, partial audit reports, retry seeds independent of unrelated groups,
generation/audit/upscale lifetime ordering, audit-aware caching, selected-seed
metadata and no publication of rejected characters. Native Qwen preparation
retains reference pixels and explicit sequence/guidance settings.

Joint scene/pose tests use the real control-fitting and ordering owners with
neural preprocessor doubles. Scene maps now resize uniformly after
aspect-preserving padding even when the source is smaller than the target;
the prior small-map branch incorrectly left a miniature in a large black
canvas. The audit's original pose/scene sources follow the generator's image
order. This is a CPU composition contract, not a new GPU quality claim for
combined DWPose/depth conditioning.

The masked character route is a separate implementation and acceptance scope.
