# Character route implementation disclosure

Scope: stage 5 of the approved implementation plan. Implement the raw audit,
bounded response and explicit mask route required by `docs/PLAN.md`, integrated
with the existing workflow engine and character commands. Ordinary image edit
retains its separate unaudited contract.

## 5a: unmasked character edits

Assumptions presented before implementation and reviewed read-only:

- Klein and Qwen-2511 Lightning remain explicit choices. The default is Klein;
  a named Qwen character command never changes to Klein. Legacy 2509 and Qwen
  base comparison profiles are not admitted to this active character route.
- Every supplied reference remains a native visual input. No character
  dossier, hand-written identity measurements or edge-crispness checks.
- Default two rounds means one initial raw batch and at most one new-seed
  batch, on the same selected backend. New seeds are explicitly allowed by
  the plan's single-pass escalation ladder. No pre-emptive close-up repair.
- `character_edit.py` owns raw generation rounds, candidate identities,
  audit decisions and final publication. `character_edit_audit.py` owns the
  narrow VLM protocol, numbered images and parsing. CLI and graph must call
  these same owners, not create another execution engine.
- The VLM returns a flat verdict for its selected candidate: passed,
  image_index and short defect-region pointers. Audit revision 2 uses one
  global, one-based image numbering scheme for references, context and candidates.
  Map the selected image number immediately
  to a durable candidate identity. Malformed output fails with its saved raw
  response; it never authorizes a candidate or triggers a generation retry.
- Audit receives the user's instruction, route, all original reference images
  and original pose/scene sources with explicit roles. Requested visual/style
  changes are part of intent. Region strings never condition generation in 5a.
- The complete raw generation batch returns and releases model weights before
  VLM loading. One VLM load judges all target groups in a round. Retain accepted
  groups; retry only rejected ones. Optional VOSR runs once on accepted raw
  selections after the audit model closes.
- Preserve Qwen's explicit canvas/aspect selection and structural-control fitting.
  Combined scene/pose inputs retain matching generation/audit ordering. Small
  scene controls scale uniformly to fill the canvas instead of remaining a
  small inset in padding.
  Reuse its preparation owner and materialize the same images. Preserve
  explicit CLI max_sequence_length and guidance_scale through the batch API.
- Cache identity includes audit model/processor settings, implementation and
  prompt revision, and the retry policy. Failed rounds retain raw artifacts
  and reports but publish no successful character selection.

Production sources studied before implementation:

- [TRL 0.16.1 judges](https://github.com/huggingface/trl/blob/v0.16.1/trl/trainer/judges.py):
  candidate-index ownership, explicit invalid outputs and combined judgments.
  Its fallback policies are not adopted; this application fails visibly.
- [Transformers Qwen2.5-VL processor](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py):
  aligned multi-image placeholders and visual inputs.
- Existing local native Qwen preprocessing and sequential Klein/LightX2V
  batch owners, already compared against their production implementations in
  stages 1–3.

Review approved 5a and specifically required preserving control preparation,
CLI settings, raw-before-postprocess ordering and audit-aware provenance.

## 5b: explicit mask edits

The old masked character function only supports 2509. It cannot become an
implicit fallback for the active 2511 route. Before enabling the graph route,
implement and verify actual masked latent denoising on the explicit backbone,
plus original-pixel preservation outside the selected mask.

The production references investigated are
[Diffusers Qwen edit inpaint](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_inpaint.py)
and [ComfyUI KSamplerX0Inpaint](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/samplers.py).
Both impose a mask during denoising, rather than treating a final composite as
proof of native masked editing. SAM remains the explicit selection boundary.
The concrete implementation will be reviewed before enabling a model run.

No missing models will be downloaded. The local audit VLM's five checkpoint
shards are present. Model runs require prompt/input review and a live VRAM
check. Acceptance covers raw pass, retry pass, bounded failure, malformed
audit output, lifecycle ordering, actual visual selection and mask preservation.


### Implemented 5b ownership and reviewed assumptions

- `qwen_image_edit_sampling.py` extends the pinned native LightX2V scheduler.
  The public sampler names stay `flowmatch-euler` and `euler-ancestral`.
  Source latents are separate from the full conditioning bundle. The native
  `round(steps * strength)` suffix and `begin_index` govern initialization of
  the whole canvas; after each step the protected area follows the source at
  the actual next sigma, using the original seed noise.
- The mask is resized to the VAE grid, repeated over 16 latent channels and
  packed with the same 2x2 layout as the image latents. All four spatial
  positions remain distinct. Four additional latent buffers are included in
  the GPU residency budget and released per output.
- `qwen_image_edit_masked.py` owns source/mask canvas fitting, native decode
  records and lossless PNG publication. Native source resolution is the
  default, with only 16-pixel padding. An explicit max-side caps generation;
  the original source and mask still own final compositing and RGBA checking.
  No latent resizing, reference dropping, automatic crop repair or VOSR.
  Identical masked requests share conditioning and one worker across seeds.
- `run_character_masked_edit` is shared by CLI and graph and invokes the
  existing raw audit owner. Generation and audit use source first, then every
  ordered reference; the audit also receives the mask as a visual context.
- `SAM segmentation` publishes a typed MASK with source file SHA and displayed
  dimensions. `Bind mask to source` handles imported masks. Cached masks retain
  that binding; importing an identical exported source remains valid even
  though its workflow artifact ID differs. A region-plan import retains both
  original source and mask checksums through save/load.
- `Qwen regional edit` requires source + MASK, allows optional references and
  compatible LoRAs, and publishes only an audit-approved raw composite.
  Results can inspect/export masks. SAM form import preserves segmentation
  and Qwen masked-edit settings, including existing region-plan selection.
  Generating new Florence text-region plans stays on its existing explicit
  command; no automatic defect-repair machinery is introduced.
- Read-only implementation review caught the public Euler token, lossy JPEG
  publication and unresolved mask paths; all were corrected before GPU use.
  Independent graph review required content-based source binding, aligned
  image numbering and persistent region-plan hashes; these are implemented.

### Final review and audit evidence

The final integration review found empty optional reference-pack nodes,
blank guidance fields losing profile defaults and unrelated SAM runtime
dependencies in cache provenance. These are fixed and covered through the
real CPU compiler/executor/form owners. Runtime identity now follows the
selected segmentation engine, including SAM1's package and anime's OpenCV.

Actual masked GPU execution exposed a semantic false positive: a background
edit mainly changed the shadow, but audit revision 2 passed it. Revision 3
explicitly treats incomplete requested changes as defects, with unchanged
image numbering and flat output schema. Independent review approved the
wording and the retained negative/positive examples. The real VLM still
passed the incomplete background edit while correctly selecting the prior
Klein front-view candidate. This wording change is not a proven semantic
accuracy fix. No character-specific test or prompt checklist is added.

The controlled follow-up retains source/mask hashes, literal instruction and
seeds while changing strength to 1.0. Both candidates now show the requested
background. New regional Lightning edits consequently default to strength
1.0 in the graph, SAM form, CLI and shared masked-edit owner. Independent
review approved this bounded choice: strength changes both starting noise and
the timestep suffix, so the result cannot be attributed solely to more steps.
Existing explicit lower strengths retain their values and cache identities.
This local eight-step default is not claimed to be the default of Diffusers'
separate fifty-step Qwen inpaint pipeline, whose call signature uses 0.6.
