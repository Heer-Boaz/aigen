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
SAM2 character region masks and validation checks.

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
- `model_sources/keyframe_judge_qwen2_5_vl_7b.json`

Optional Qwen identity smoke assets are intentionally separate from the default
installer:

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

## Command Surface

The public CLI has four owners:

- `models`: download pinned model manifests into `aigen/models`.
- `characters`: validate, run and accept canonical character view-bank entries.
- `briefs`: plan and materialize keyframe jobs from identity images and example sprites.
- `keyframes`: run, score, judge, refine and polish materialized jobs.

Raw one-shot generation commands are not a supported public workflow. Model
pipelines are implementation modules behind character-view, keyframe and polish
jobs.

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
image metadata and stable pack-local handles; it never serializes the character
into an identity dossier. Qwen-Image-Edit receives the selected references as
images and resolves identity and appearance inside the edit model.

```bash
.venv/bin/python -m aigen.cli characters reference-pack build \
  --character-id <character-id> \
  --reference reference1=path/to/reference1.png \
  --reference reference2=path/to/reference2.png \
  --reference reference3=path/to/reference3.png \
  --reference reference4=path/to/reference4.png \
  --output-dir assets/characters/<character-id>/references
```

This writes `reference_pack.json`. Packs of one to three images bypass the
external VLM entirely. For larger packs, the narrow Qwen2.5-VL selector returns
only up to three 1-based image indices, unloads, and then Qwen-Image-Edit receives
those images plus the user's instruction. No image descriptions or inferred
reference roles enter the generation prompt.

```bash
.venv/bin/python -m aigen.cli characters qwen-edit-run \
  --pack assets/characters/<character-id>/references/reference_pack.json \
  --case right_profile \
  --model nunchaku-qwen-edit-2509-r32-4step \
  --output-dir runs/characters/<character-id>/qwen_edit/right_profile_smoke
```

## Qwen Identity Smoke

The local Qwen route starts with Nunchaku-quantized Qwen-Image-Edit-2509. Do
not download or run full Qwen-Image-Edit-2511 for the native aigen smoke path.
Each Qwen 2509 manifest allowlists exactly one Nunchaku checkpoint.

```bash
.venv/bin/python -m aigen.cli models download \
  --manifest model_sources/qwen_identity_2509_fp4_r32_lightning_4step.json \
  --models-root aigen/models

.venv/bin/python tools/smoke_qwen_image_edit_2509_nunchaku.py \
  --input-image path/to/front.png \
  --output-dir runs/qwen_identity/fit_4step
```

Only after the single-image fit smoke succeeds, run the multi-reference identity
smoke. It does not train LoRAs, use ControlNet, score outputs or judge with a
VLM.

```bash
.venv/bin/python -m aigen.cli characters qwen-identity-run \
  --reference front=path/to/front.png \
  --reference portrait=path/to/portrait.png \
  --reference side=path/to/side.png \
  --reference back=path/to/back.png \
  --output-dir runs/qwen_identity/smoke
```

The default model is `nunchaku-qwen-edit-2509-r32-4step`.
`--nunchaku-blocks-on-gpu` explicitly enables the slower layer-offload path for
the multi-reference runner. The standalone fit script uses `--offload-mode auto`
by default on 18GB-or-smaller GPUs.

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
