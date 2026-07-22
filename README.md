# aigen

Private AI character keyframe pipeline for game character art. The supported
workflow is brief-first: the user supplies an approved character view bank, a
source sprite or frame, and a short action request. Local vision-language and
vision models plan the identity caption, pose caption, prompt text, controls,
scoring checks and polish targets from those images.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[generation]"
```

Models live under `aigen/models`. Hub repo IDs and revisions are pinned in model
source manifests and recorded in run metadata.

## Install

For a fresh workstation, run the single installer:

```bash
scripts/install.sh
```

The installer is modular internally and always installs the current production
stack: FLUX Kontext, Shakker Union Pro ControlNet, Nunchaku, Qwen3 instruction
parser, Qwen2.5-VL judge/planner, DWPose pose scoring models, SAM foreground
segmentation, GroundingDINO polish grounding, Florence-2 polish grounding,
SAM2 character region masks, Depth Anything V2 scene controls, Real-ESRGAN
anime postprocessing and validation checks.

```bash
scripts/check_system.sh
scripts/setup_venv.sh
scripts/install_nunchaku.sh
scripts/download_models.sh
scripts/check_install.sh
```

The model manifests used by the installer are:

- `model_sources/keyframe_generation_kontext_controlnet.json`
- `model_sources/keyframe_generation_nunchaku_transformer.json`
- `model_sources/qwen3_8b_instruction_parser.json`
- `model_sources/keyframe_segmentation_sam_vit_b.json`
- `model_sources/character_region_sam2_tiny.json`
- `model_sources/keyframe_grounding_dino.json`
- `model_sources/keyframe_grounding_florence2.json`
- `model_sources/keyframe_pose_dwpose_onnx.json`
- `model_sources/character_scene_depth_v2_large.json`
- `model_sources/character_postprocess_illustrationjanai_v1.json`
- `model_sources/keyframe_judge_qwen2_5_vl_7b.json`

Legacy Qwen-Image-Edit-2509 baseline manifests remain separate from the 2511
LightX2V runtime:

- `model_sources/qwen_identity_2509_fp4_r32_lightning_4step.json`
- `model_sources/qwen_identity_2509_fp4_r32_lightning_8step.json`
- `model_sources/qwen_identity_2509_fp4_r32_full.json`
- `model_sources/qwen_identity_2509_fp4_r128_full.json`

To inspect a model download manifest manually:

```bash
.venv/bin/python -m aigen.cli models download \
  --manifest model_sources/keyframe_generation_nunchaku_transformer.json \
  --models-root aigen/models \
  --dry-run
```

Remove `--dry-run` after accepting required model licenses.

## Experimental HunyuanVideo-1.5 I2V

HunyuanVideo-1.5 is an optional direct image-to-video route and is not part of
the default character-keyframe installation. It uses Tencent's official source
at a pinned revision and the official `480p_i2v_step_distilled` transformer.
The transformer alone is 33.3 GB, so the runtime keeps model components on the
CPU and uses transformer group offloading instead of loading the complete model
into VRAM.

Install the isolated runtime and download only the required model components:

```bash
scripts/install_hunyuanvideo15.sh
scripts/download_hunyuanvideo15.sh
```

The installer applies the tracked patch under `patches/hunyuanvideo15/`. It
releases CUDA allocator cache after an offloaded component has moved back to
the CPU. This was required to pass the text-encoder-to-VAE transition on the
16 GB RTX 5070 Ti, but the BF16 transformer still fails at the first denoise
step. The route is not end-to-end validated; see
`docs/HANDOFF-HUNYUANVIDEO15.md` for the exact evidence and review questions.

The command below targets the intended short 480p I2V profile:

```bash
.venv/bin/aigen hunyuanvideo15-i2v \
  --image path/to/input.png \
  --prompt "<motion and camera instruction>" \
  --output runs/hunyuanvideo15/output.mp4
```

The defaults are 49 frames, seed 42 and 8 step-distilled inference steps. The
only other supported step count is 12. CFG 1 and flow shift 7 come from
Tencent's official checkpoint profile. Super-resolution, prompt rewriting,
feature caching, sparse attention and compilation remain disabled; CPU model
offloading and transformer group offloading remain enabled. The process writes
Tencent's generation config and a log beside the MP4.

## Command Surface

The character-keyframe workflow has four public owners:

- `models`: download pinned model manifests into `aigen/models`.
- `characters`: validate, run and accept canonical character view-bank entries.
- `briefs`: plan and materialize keyframe jobs from identity images and example sprites.
- `keyframes`: run, score, judge, refine and polish materialized jobs.

Raw one-shot generation commands are separate experimentation utilities, not
part of the character-keyframe workflow. The optional HunyuanVideo command
above is likewise a separate I2V utility.

## Direct Image Editing

`image-edit` provides one command contract for the local image-edit backends.
Repeat `--image` for multiple ordered inputs, use `--reference-pack` for a named
ordered image bundle, and repeat `--seed` for a seed sweep. The command writes
valid PNG names inside `--output-dir`; `--overwrite` replaces a non-empty output
directory.

```bash
.venv/bin/aigen image-edit \
  --backend flux2-klein \
  --image path/to/source.png \
  --image path/to/style-reference.png \
  --prompt "Draw as pixel-art. Limited color palette, retro, 8-bit, no background." \
  --output-dir runs/image-edit/example \
  --aspect-ratio 1:1 \
  --sampler euler-ancestral \
  --seed 0 \
  --overwrite
```

Available backends are `flux2-klein`, `qwen-image-edit-2511-lightning`,
`qwen-image-edit-2511-base`, `hidream-o1-full-fp8` and
`boogu-image-edit-turbo-fp8`. `--aspect-ratio` selects that backend's
recommended canvas: trained presets for Qwen and HiDream, a 1024px long side
for FLUX.2 Klein, and an approximately one-megapixel 1K canvas for Boogu.
`--width` and `--height` remain an exact expert override and cannot be combined
with `--aspect-ratio`. When all three are omitted, the first resolved input
image supplies only the aspect ratio while the backend supplies its recommended pixel
dimensions. `--steps` and `--guidance` are also optional and remain
backend-native when omitted. `--sampler` and `--scheduler` are likewise
backend-specific. FLUX.2 Klein supports `flowmatch-euler` (default) and
`euler-ancestral` on its native dynamic-shift schedule. Both Qwen-2511 routes
support those samplers with either their native dynamic-shift schedule or
ComfyUI-compatible `simple` scheduling. HiDream exposes the samplers and sigma
schedulers provided by its pinned ComfyUI runtime. Boogu Turbo remains fixed to
its checkpoint-native DMD path.
FLUX.2 Klein and both Qwen-2511 backends accept
repeated `--lora path/to/weights.safetensors` arguments. Repeat `--lora-weight`
in the same order to set individual strengths, or omit all weights to use 1.0.
The command inspects every SafeTensors model keyspace before loading the backend
and rejects a Qwen/FLUX mismatch. HiDream and Boogu do not expose LoRA loading
through this command. Reference packs are expanded once before backend dispatch,
preserving their declared image order. Boogu's native one-image limit also
applies to packs; FLUX.2 Klein, Qwen and HiDream accept multi-image packs.

### Terminal UI

```bash
.venv/bin/aigen-tui
```

The Images tab has a free-text prompt, model, LoRA and character-reference-pack
dropdowns, and free-text source-image slots. Multiple LoRAs can be selected with
independent weights, and multiple Seed slots produce a seed sweep. The aspect
ratio dropdown uses backend-compatible presets; Width and Height provide a paired
exact override. Empty Steps and Guidance fields retain backend-native defaults.
The form uses Textual's responsive grid: value fields consume the available
width, slot controls remain visible in fixed trailing columns, and action buttons
reflow as the terminal resizes.
New Seed slots start at the lowest unused non-negative seed value.
Selecting a model replaces Steps and Guidance with that backend's standard
sweet-spot values. Startup reapplies those values for the restored model;
backends without CFG leave Guidance empty.
LoRA files are discovered from `loras/` and filtered for the selected model;
reference packs are discovered from `assets/reference-packs/*.json`.
Its visible buttons add and remove slots and start or stop generation. Each
movable field has compact `↑` and `↓` buttons; unavailable directions are dimmed.
Tab and Shift-Tab focus those buttons; Enter activates the focused button. The
The Videos tab exposes the configured video backends. The SAM Edit tab exposes standalone
SAM mask/cutout/preview generation, Florence-2/SAM2 region plans, and a direct Qwen masked
edit form that consumes either an existing white-on-black mask or a selected region from a
region-plan result through the character refine owner.
For SAM box/point prompting, select `Box`, `Points`, or `Box + points` and open `Edit prompts`
to edit the input image in a large overlay: box mode uses two left clicks for opposite
corners, left click adds a positive point, and right click (or Shift+left click) adds a
negative point. The overlay can clear, save, and load prompt selections as JSON files.
SAM runs write a `result.json` manifest alongside their selected artefacts; rerunning an
existing output directory requires the explicit overwrite action.
TUI generations replace the contents
of the selected output run directory so the same destination can be reused.
The Images form is restored after a normal quit from
`$XDG_CONFIG_HOME/aigen/image-tui.json` (or `~/.config/aigen/image-tui.json`).
`Browse` opens a Textual directory tree for the selected Image slot or Output
directory field. Files are selected directly; folders can be traversed with the
mouse or keyboard.
`Use Result` opens the current output directory and adds the chosen generated
image to the form without replacing a populated Image slot.
`Save Pack` writes the ordered, populated Image slots through the existing
reference-pack builder and selects the new pack without overwriting an existing
pack.
`Save Config` stores the complete Images form as JSON in a folder selected with
the terminal file browser. `Load Config` selects such a JSON file and replaces
the complete form, including seeds, images, packs, LoRAs and their weights.
Stop and Quit terminate the complete generation process group, including native
backend workers, so a cancelled run cannot leave a worker holding GPU memory.
While generation runs, the status line shows the backend's complete progress
snapshot: progress bar, percentage, completed steps, phase, ETA, elapsed time,
CPU usage, GPU usage and VRAM.

## Character Views

Canonical character views are stored in a view bank. A view-bank entry is image
metadata and approval provenance: no prompts are stored there and no generated
keyframe is allowed to infer a random reference image.

```bash
.venv/bin/python -m aigen.cli characters view-schema > schemas/character-view-job.schema.json
.venv/bin/python -m aigen.cli characters view-bank-schema > schemas/character-view-bank.schema.json
.venv/bin/python -m aigen.cli characters view-validate path/to/character_view_job.json
.venv/bin/python -m aigen.cli characters view-run path/to/character_view_job.json
.venv/bin/python -m aigen.cli characters view-accept path/to/character_view_job.json \
  --run-dir runs/characters/<character>/views/<view_run> \
  --candidate seed_003
```

Accepted views are written to `assets/characters/<id>/views/` and registered in
`assets/characters/<id>/view_bank.json` with hashes, view metadata and source
run evidence.

## Reference Packs

PixAI-style character editing starts with a named reference pack. The pack stores
an ordered mapping of stable pack-local handles to image paths; it never serializes
the character into an identity dossier. Qwen-Image-Edit receives all references as
images and resolves identity and appearance inside the edit model.

```bash
.venv/bin/python -m aigen.cli characters reference-pack build \
  --character-id <character-id> \
  --reference reference1=path/to/reference1.png \
  --reference reference2=path/to/reference2.png \
  --reference reference3=path/to/reference3.png \
  --reference reference4=path/to/reference4.png \
  --output assets/reference-packs/<pack-name>.json
```

When filenames already provide suitable handles, omit the manual `name=path`
mapping:

```bash
.venv/bin/python -m aigen.cli characters reference-pack build \
  --character-id <character-id> \
  --file path/to/fullbody-multiview.png \
  --file path/to/front-upperbody.png \
  --file path/to/clothes.png \
  --output assets/reference-packs/<pack-name>.json
```

Each `--file` uses its filename stem as the stable pack-local handle, and
repeated `--file` options establish model-input order. The selected native
backend receives that complete ordered pack plus the user's instruction and any
structural inputs. No image descriptions, inferred reference roles or
count-based selector enter the generation path.

```bash
.venv/bin/aigen characters qwen-edit-run \
  --pack assets/reference-packs/<pack-name>.json \
  --instruction "Full-body three-quarter view. Keep the entire character and footwear visible." \
  --output-dir runs/characters/<character-id>/qwen_edit/three_quarter
```

## Qwen Image Edit 2511

`qwen-edit-run` runs exactly one free-instruction request by default. Its default
model is `lightx2v-qwen-edit-2511-fp8-lightning-8step`; no case name or reusable
JSON plan is involved. The 2511 backend translates the selected aspect ratio to
its proven 1.77-megapixel target raw canvas on Qwen's 16-pixel latent grid. A control, source or guide
owns the aspect ratio in that order; without one the default is 3:4.
Character-reference images never choose the canvas. Override the shape with
`--aspect-ratio W:H`. Raw images remain in `raw/`; final images are upscaled to
a 2048-pixel long side by default. Override that with
`--upscale-long-side PIXELS`. Use `--candidates N` only when multiple outputs
are intentional.

To edit an existing image, pass it as Image 1. The source image owns the output
aspect unless `--aspect-ratio` overrides it. Repeat `--image` only when
the instruction deliberately refers to additional pictures:

```bash
.venv/bin/aigen characters qwen-edit-run \
  --image path/to/source.png \
  --instruction "Change only the expression to a subtle smile. Keep everything else unchanged." \
  --output-dir runs/characters/<character-id>/qwen_edit/smile
```

For non-interactive shells such as Codex tool runs, progress falls back to
stderr and includes elapsed time, ETA, GPU utilization and VRAM. Use
`AIGEN_PROGRESS=0` to disable it, or `AIGEN_PROGRESS_INTERVAL_SECONDS=1` for a
faster update cadence during long Qwen runs.

## Briefs

Briefs are the authoring surface. They point at the approved view bank and the
example sprite; the VLM inspects both and writes the generated plan.

```bash
.venv/bin/python -m aigen.cli briefs schema > schemas/keyframe-brief.schema.json
.venv/bin/python -m aigen.cli briefs plan briefs/<character>/<action>.json
.venv/bin/python -m aigen.cli briefs materialize briefs/<character>/<action>.json
.venv/bin/python -m aigen.cli briefs run briefs/<character>/<action>.json
```

The generated brief plan records what the model saw:

- `identity_description`: subject type, hair, clothing, colors and style from
  the identity images.
- `pose_description`: body pose, hands, arms, legs, feet, silhouette and action
  phase from the example sprite.
- `platformer_camera_description`: the camera/readability interpretation,
  including platformer side-view cheats when useful.
- `prompt`: separate CLIP and T5 text built from the supplied images.
- `controls`: model-selected pose/contour/depth/soft-edge controls and scales.
- `scoring` and `polish`: model-planned checks and local repair budget.

## Keyframes

Keyframe jobs are materialized execution plans. They own the approved identity
primer, extracted pose and contour assets, generated CLIP/T5 prompts, fixed seed
variants, output paths and acceptance notes.

```bash
.venv/bin/python -m aigen.cli keyframes validate runs/briefs/<character>/<action>/job.json
.venv/bin/python -m aigen.cli keyframes plan runs/briefs/<character>/<action>/job.json
.venv/bin/python -m aigen.cli keyframes run runs/briefs/<character>/<action>/job.json
```

The current keyframe generation profile uses:

- FLUX Kontext 4-bit Diffusers components.
- Shakker-Labs FLUX.1-dev ControlNet Union Pro 2.0.
- Nunchaku FP4 Kontext transformer.
- `nunchaku-fp16` attention.
- Diffusers pipeline CPU offload with Nunchaku layer offload disabled.
- Explicit `nvidia-smi` preflight, peak VRAM sampling and token-based
  `vram_max_output_canvas` advice for the current framebuffer headroom.

Runs write:

- `resolved.json` before denoising, with absolute paths, asset hashes, model
  revisions, token counts, active ControlNet steps and output paths.
- generated PNGs for each fixed-seed variant.
- `result.json` after denoising, with outputs, timings, tokens, VRAM,
  environment, ControlNet metadata and the measured framebuffer peak.
- condition copies and contact sheets when requested by the job.

## Example Extraction

Use source sprites or reference frames to extract reusable action conditions
when you need explicit assets. Brief materialization performs this extraction
for its example sprite.

```bash
.venv/bin/python -m aigen.cli keyframes extract-example \
  --source references/platformer/punch.png \
  --output-dir assets/examples/ai51_punch \
  --name ai51_punch_platformer \
  --width 576 \
  --height 864
```

The extracted pose, contour and boundary assets are explicit job inputs. They are
not silently regenerated during keyframe runs.

## Scoring And Selection

The primary scorer is condition-first. It uses SAM for foreground masks, DWPose
for body-keypoint evidence, and the resolved job assets as the target pose,
contour and identity-primer evidence. The VLM judge is a semantic QA gate, not
the final selector for subtle geometry.

```bash
.venv/bin/python -m aigen.cli keyframes score runs/keyframes/ai51/punch_platformer/structure
.venv/bin/python -m aigen.cli keyframes score-select runs/keyframes/ai51/punch_platformer/structure
```

Human review can still accept a structure winner explicitly by writing selection
metadata for later scorer fixtures.

## LoRA Canon

LoRA training starts from canon-worthy identity images, not from broad generated
keyframe pools. Every canon image must be human-approved as the same character
with the correct face, hair, outfit, proportions, style and background quality.

```bash
.venv/bin/python -m aigen.cli lora canon-init \
  --character-id ai51 \
  --trigger-token ai51char \
  --identity-prompt "1girl, blue eyes, gloves, blue thigh-highs, full body, white blouse, button-up shirt, short hair, brown hair, leather skirt, belt, brown long boots, collared shirt, looking at viewer, brown leather jacket, sleeved jacket, smile, light blush, blue necktie, standing, flat-chested, small breasts" \
  --anchor root=assets/characters/ai51/source.png

.venv/bin/python -m aigen.cli lora dataset-audit assets/characters/ai51/canon
```

Loose image folders audited with `dataset-audit` are marked `needs_human_review`;
only `lora-canon` manifests created from explicit anchors are accepted as canon.

## LoRA Candidate Factory

The production LoRA path plans many candidates, filters hard and trains only on
human-approved canon-worthy images. The candidate brief owns candidate names,
views, poses, prompts, identity primers, seed budget and output directory. The
pipeline owns the neutral coverage contract; the local visual planner writes the
actual generation prompt for each requested view/pose from the approved canon
images. `candidate-plan` materializes the brief into exact generation prompts,
training captions, seeds and output paths.
Rear/back candidates are only planned when an approved rear/back canon anchor is
present; otherwise the planner stays with grounded front/profile/three-quarter
and mild-pose identity coverage.

```bash
.venv/bin/python -m aigen.cli lora candidate-brief-plan assets/characters/ai51/canon \
  --output jobs/ai51/lora_candidates.json \
  --candidate-output-dir runs/lora_candidates/ai51_identity
```

The generated candidate brief is model-written. Do not hand-author character
prompts in Python or documentation examples: the planner must infer identity,
clothing, view, pose, background and visual medium from the approved canon
images and the full user identity prompt.

```bash
.venv/bin/python -m aigen.cli lora candidate-plan jobs/ai51/lora_candidates.json

.venv/bin/python -m aigen.cli lora candidate-run runs/lora_candidates/ai51_identity

.venv/bin/python -m aigen.cli lora candidate-evidence runs/lora_candidates/ai51_identity

.venv/bin/python -m aigen.cli lora candidate-judge runs/lora_candidates/ai51_identity

.venv/bin/python -m aigen.cli lora candidate-review runs/lora_candidates/ai51_identity \
  --accept front_neutral_seed_0044 \
  --accept left_profile_neutral_seed_0102 \
  --approved-by boaz
```

`candidate-plan` writes generation prompts without the LoRA trigger token,
training captions with the trigger token, seeds and exact output paths.
`candidate-run` is the only GPU image executor for those planned candidates and
uses only `generation_prompt`. Dataset build uses only `training_caption` after
human approval.
`candidate-evidence` then writes full sheets and crop evidence for face, torso,
waist/lower body, legs/feet and silhouette; it does not claim model approval.
`candidate-judge` compares every candidate against the approved primer and crop
evidence, writes `evidence/passed.json` and blocks non-canon training images.
`candidate-review` only accepts candidates from that model-passed set and writes
`review/accepted.json`, `review/rejected_human.json`, `review/quota_report.json`
and an accepted contact sheet. LoRA dataset specs use canon manifests and
human-approved candidate review manifests:

```json
{
  "$schema": "schemas/lora-dataset.schema.json",
  "kind": "lora-dataset",
  "id": "ai51_identity_from_accepted_candidates",
  "character": {
    "id": "ai51",
    "trigger_token": "ai51char"
  },
  "sources": [
    {
      "type": "candidate_review",
      "path": "runs/lora_candidates/ai51_identity/review/accepted.json"
    }
  ],
  "output": {
    "directory": "runs/lora/ai51_identity",
    "overwrite": true,
    "validation_ratio": 0.1,
    "save_contact_sheet": true
  }
}
```

## Local Polish

Polish is a separate local inpaint phase. The static plan resolves paths without
loading models; diagnosis is model-backed and writes model-discovered regions;
run executes crop/mask inpainting; select picks local variants.

```bash
.venv/bin/python -m aigen.cli keyframes polish-plan runs/briefs/<character>/<action>/polish.json
.venv/bin/python -m aigen.cli keyframes polish-diagnose runs/briefs/<character>/<action>/polish.json
.venv/bin/python -m aigen.cli keyframes polish-run runs/briefs/<character>/<action>/polish.json
.venv/bin/python -m aigen.cli keyframes polish-select runs/briefs/<character>/<action>/polish.json
```

Polish must keep pose and silhouette frozen. Variants that change pixels outside
the feathered mask are rejected.

## Validation

```bash
.venv/bin/python -m pytest
git diff --check
```
