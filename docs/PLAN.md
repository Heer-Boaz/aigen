# AI51 PixAI-Style Character Edit Pipeline Plan

Status: authoritative design document for the PixAI-style character pipeline.
If a task or instruction conflicts with this file, stop and ask. Do not
silently reinterpret. Hard behavioral rules for agents live in `AGENTS.md` at
the repo root.

## Goal

A local, PixAI-style multi-reference character editing pipeline that runs on a
16GB GPU (RTX 50-series / Blackwell). Scope:

- character identity across generations
- view/camera changes such as front, back, profile, three-quarter, and portrait
- simple pose and later platformer/poseplate outputs
- local identity repair for task-requested regions or components

North star: a PixAI/Edit Pro-like editor can take several reference images plus
a natural-language request such as "Character as shown in referenced images.
Close-up face, looking left, neutral expression, white background" and produce
a result that preserves the referenced character. All character understanding
comes from models inspecting the supplied images. Code routes images,
instructions, model choices, and conditioning only.

## Single Leading Map

This plan is only for the PixAI core reverse-engineering track.

```text
PixAI core reverse-engineering pipeline
Internal/model pipeline for one heavily conditioned generation.
```

Do not add workflow features to this plan. They are not part of the PixAI core
track.

## Golden Rule: Character-Agnostic Code

Pipeline code contains zero character-specific content. Every fact about a
character (hair, outfit, body shape, silhouette, which image shows which view,
and similar facts) is extracted at runtime by a model from the reference images
or read from a model-produced artifact. JSON examples in this document
illustrate shape only. Their values must never appear as constants, defaults,
prompt fragments, or fixtures in code.

Litmus test: swap in a completely different character reference pack. If any
code change is needed, the code is wrong.

## PixAI Core Reverse-Engineering Pipeline

The PixAI core is the model/pipeline-internal path for a single requested
output.

```text
1. Read user instruction.
2. Determine target task.
3. Encode all reference images.
4. Identify the relevant subject(s).
5. Extract task-relevant components from references.
6. Align same components across references.
7. Choose which components matter for this specific output.
8. Build a conditioning bundle.
9. Run one heavily conditioned diffusion/edit generation.
10. Optionally postprocess, upscale, or filter.
```

### 1. Read User Instruction

Purpose: parse the user instruction plus request/UI context. This is text-only.
It does not inspect reference images and does not add reference-derived visual
facts.

Input:

```text
raw user instruction
UI/request context
generation panel settings
available input counts
```

Output:

```text
instruction_plan
```

The plan keeps user-written style, role, scene, action, mood, text, and external
concept requests. Reference phrases such as "as shown in referenced images" are
bindings to visual inputs, not character descriptions.

### 2. Determine Target Task

Purpose: route the request before image analysis.

Output:

```text
task_route_plan
```

Known route kinds:

```text
portrait_identity_generation
full_body_identity_generation
view_change
pose_transfer
scene_insertion
local_repair_or_inpaint
outfit_swap
style_transfer
layout_or_sheet
text_or_label_heavy
unknown_reference_edit
```

Text-heavy is a primary route only when text/layout is the main output. Scene
requests with posters, signs, logos, or newspapers remain scene routes with a
text-rendering risk marker.

### 3. Encode All Reference Images

Purpose: make every supplied reference image available to the visual planner or
multimodal model. This step is about feeding the refs, not choosing a dossier
or forcing static roles.

Rules:

- pass all references through neutral handles such as `reference1`,
  `reference2`, and so on
- preserve the image order used by the planner
- do not rely on file names or reference labels as truth
- do not preselect 1-3 refs before the VLM has looked at the pack

### 4. Identify Relevant Subject(s)

Purpose: the VLM identifies which visible subject(s) matter for the request.

Examples:

```text
same referenced character
subject from Image 2
primary character plus requested object
```

This is visual analysis, not deterministic Python identity extraction.

### 5. Extract Task-Relevant Components

Purpose: the model extracts the visual evidence needed for this task. The
component set depends on the route.

Examples:

```text
portrait -> face, eyes, hair, expression, visible upper outfit, style
full body -> face, outfit, silhouette, body proportions, footwear, style
scene -> identity, visible outfit, action compatibility, scene integration
local repair -> target component or region plus matching reference evidence
pose transfer -> identity evidence plus pose source/body orientation
```

Do not build a persistent hand-written component taxonomy unless explicitly
asked. The model owns component understanding.

### 6. Align Same Components Across References

Purpose: compare the same visual components across multiple refs.

Examples:

```text
face identity across portrait/front/side refs
outfit component consistency across front/back refs
pose source separated from identity refs
target repair region matched to the best supporting ref
```

Alignment is a model-produced visual planning result. It is not a geometry
measurement path and not a static role table.

### 7. Choose Relevant Components For This Output

Purpose: decide which aligned evidence matters for the requested output.

This is where a close-up portrait can down-rank lower-body evidence, while a
full-body or repair request can prioritize it. This choice is case-specific.
The same reference pack can produce a different component plan for a different
instruction.

Output may include:

```text
selected_refs
reference_semantics
visual_analysis
edit_instruction
```

Qwen-Image-Edit-2509 currently performs best with 1-3 selected input images, so
this is the point where the planner chooses the final reference subset.

### 8. Build Conditioning Bundle

Purpose: assemble the actual model inputs for generation.

Inputs can include:

```text
selected reference images
VLM-authored edit instruction
optional mask/region condition
optional keypoint/edge/depth/sketch condition
runtime generation parameters
```

Identity-only portrait, view, full-body, and normal scene routes usually need
no extra hidden conditioning. Local repair and pose transfer can require
region/mask or pose/keypoint conditions. The planner may produce a
conditioning plan first; heavy tools run only for routes that need them.

### 9. Run Diffusion/Edit Generation

Purpose: Qwen Image Edit consumes the step-8 bundle. It does not re-plan.

Generation receives:

```text
selected refs in planner order
VLM-authored edit_instruction
model/runtime parameters
seed/candidate settings
```

Generation must not load a separate character dossier, reselect references,
reinterpret the task route, or enrich prompts with hardcoded character facts.

### 10. Optional Postprocess/Upscale/Filter

Purpose: optional local finishing after the core generation has produced an
image. This may include technical postprocess, upscaling, filtering, or output
packaging.

## Current Implementation Alignment

Current core-oriented pieces:

```text
Step 1:
  CharacterInstructionParser
  raw instruction + envelope -> instruction_plan

Step 2:
  CharacterTaskRouter
  instruction_plan -> task_route_plan

Steps 3-7, partially combined:
  Qwen2.5-VL edit planner
  all refs + planner_context -> selected_refs, reference_semantics,
  visual_analysis, edit_instruction

Step 8, early route-gated planning:
  CharacterConditioningPlanner
  task_route_plan + visual_analysis -> conditioning_plan

Step 9:
  qwen-edit generation handoff
  selected refs + edit_instruction -> Qwen Image Edit
```

Important correction: the current VLM edit planner is a useful bridge, but it
compresses PixAI core steps 3-7 into one prompt. Future work may split that
into subject identification, component extraction, component alignment, and
component relevance selection. Do not pretend that split already exists in
code.

Important correction: `CharacterConditioningPlanner` is conceptually step 8. It
is not PixAI core step 5.

## Concrete Model Route

First concrete editor model:

```text
nunchaku-ai/nunchaku-qwen-image-edit-2509
lightning-251115/svdq-fp4_r32-qwen-image-edit-2509-lightning-4steps-251115.safetensors
```

- Pipeline class: `QwenImageEditPlusPipeline` with
  `NunchakuQwenImageTransformer2DModel`.
- Multi-image input uses `image=[image1, image2, image3]`.
- Qwen-Image-Edit-2509 target budget: 1-3 selected reference images.
- FP4 for Blackwell/RTX 50-series; offload can reduce VRAM use at a speed cost.

Escalation ladder, only when quality falls short at the current rung:

```text
4-step r32 lightning -> 8-step r32 lightning -> full r32 -> r128
```

Qwen-Image-Edit-2511 is stronger on paper but too heavy for this 16GB target
route for now.

## FLUX/Kontext Boundary

FLUX/Kontext stays a separate pipeline next to Qwen edit.

```text
FLUX/Kontext: existing single-reference identity generator / polish path
Qwen Edit:    multi-reference PixAI-style character edit core
```

Do not merge Qwen into FLUX prompt-embedding code.

## First Core Smoke Test

First core smoke remains small:

```text
instruction:
  Character as shown in referenced images.
  Close-up face, looking to the left with neutral expression.
  Neutral lighting, white background.

expected route:
  portrait_identity_generation

core behavior:
  all refs are visible to the VLM planner
  the planner selects 1-3 refs
  Qwen receives the selected refs in planner order
  Qwen receives the VLM-authored edit_instruction
  no separate character dossier is loaded
```

This smoke does not assert exact real-model JSON.

## Sources

- https://huggingface.co/Qwen/Qwen-Image-Edit-2509 - multi-image editing,
  1-3 optimal inputs, person consistency, native depth/edge/keypoint conditions
- https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/qwenimage.md -
  QwenImageEditPlusPipeline multi-image reference usage
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py -
  Qwen Image Edit pipeline implementation
- https://github.com/QwenLM/Qwen-Image - Qwen image/edit model family
- https://nunchaku.tech/docs/nunchaku/usage/qwen-image-edit.html - Nunchaku
  low-VRAM route, pipeline classes, lightning variants, offload options
- https://arxiv.org/abs/2502.13923 - Qwen2.5-VL technical report
- https://huggingface.co/docs/transformers/en/model_doc/florence2 - Florence-2
  prompt-based detection/grounding/segmentation tasks
- https://github.com/facebookresearch/sam2 - SAM2 promptable segmentation
- https://github.com/IDEA-Research/DWPose - pose/keypoint helper
