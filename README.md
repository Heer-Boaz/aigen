# aigen

Private AI character keyframe pipeline for game character art. The supported
workflow is JSON-first: jobs define identity primers, conditions, prompts,
variants, outputs and acceptance notes; runs write resolved manifests and result
manifests so useful images remain reproducible.

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
stack: FLUX Kontext, Shakker Union Pro ControlNet, Nunchaku, Qwen judge, DWPose
pose scoring models, SAM foreground segmentation, GroundingDINO polish
grounding, Florence-2 polish grounding and validation checks.

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
- `model_sources/keyframe_segmentation_sam_vit_b.json`
- `model_sources/keyframe_grounding_dino.json`
- `model_sources/keyframe_grounding_florence2.json`
- `model_sources/keyframe_pose_dwpose_onnx.json`
- `model_sources/keyframe_judge_qwen2_5_vl_7b.json`

To inspect a model download manifest manually:

```bash
.venv/bin/python -m aigen.cli models download \
  --manifest model_sources/keyframe_generation_nunchaku_transformer.json \
  --models-root aigen/models \
  --dry-run
```

Remove `--dry-run` after accepting required model licenses.

## Command Surface

The public CLI has three owners:

- `models`: download pinned model manifests into `aigen/models`.
- `characters`: create and accept canonical character view-bank entries.
- `keyframes`: validate, plan, run, score, judge, refine and polish JSON jobs.

Raw one-shot generation commands are not a supported public workflow. Model
pipelines are implementation modules behind character-view, keyframe and polish
jobs.

## Character Views

Canonical character views are stored in a view bank. A keyframe job must name the
approved identity primer it uses; it does not infer a reference image.

```bash
.venv/bin/python -m aigen.cli characters view-init --template ai46-left-profile > jobs/ai46/left_profile_view.json
.venv/bin/python -m aigen.cli characters view-validate jobs/ai46/left_profile_view.json
.venv/bin/python -m aigen.cli characters view-plan jobs/ai46/left_profile_view.json
.venv/bin/python -m aigen.cli characters view-run jobs/ai46/left_profile_view.json
.venv/bin/python -m aigen.cli characters view-accept jobs/ai46/left_profile_view.json \
  --run-dir runs/characters/ai46/views/left_profile_neutral \
  --candidate seed_003
```

Accepted views are written to `assets/characters/<id>/views/` and registered in
`assets/characters/<id>/view_bank.json` with hashes and source run evidence.

## Keyframes

Keyframe jobs own the approved identity primer, source-derived pose and contour
assets, CLIP/T5 prompts, fixed seed variants, output paths and manual acceptance
notes.

```bash
.venv/bin/python -m aigen.cli keyframes init --template c2-profile > jobs/ai46/walk_contact.json
.venv/bin/python -m aigen.cli keyframes validate jobs/ai46/walk_contact.json
.venv/bin/python -m aigen.cli keyframes plan jobs/ai46/walk_contact.json
.venv/bin/python -m aigen.cli keyframes run jobs/ai46/walk_contact.json
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

Use source sprites or reference frames to extract reusable action conditions:

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

## Local Polish

Polish is a separate local inpaint phase. The static plan resolves paths without
loading models; diagnosis is model-backed and writes the region plan; run
executes crop/mask inpainting; select picks local variants.

```bash
.venv/bin/python -m aigen.cli keyframes polish-plan jobs/ai51/punch_platformer_polish.json
.venv/bin/python -m aigen.cli keyframes polish-diagnose jobs/ai51/punch_platformer_polish.json
.venv/bin/python -m aigen.cli keyframes polish-run jobs/ai51/punch_platformer_polish.json
.venv/bin/python -m aigen.cli keyframes polish-select jobs/ai51/punch_platformer_polish.json
```

Polish must keep pose and silhouette frozen. Variants that change pixels outside
the feathered mask are rejected.

## Validation

```bash
.venv/bin/python -m pytest
git diff --check
```
