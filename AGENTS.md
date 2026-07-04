# Agent rules for this repo

Read `docs/PLAN.md` before touching anything under `aigen/` related to the
character pipeline. That document is the assignment. Do not simplify it, and do
not extend it on your own initiative.

## Golden rule: character-agnostic code

Pipeline code contains zero character-specific content. Every fact about a
character (hair, outfit, chest size, silhouette, which image is the back view,
…) is extracted at runtime by a model (VLM) from the reference images, or read
from a model-produced artifact. JSON examples in the plan show the *shape* of
model output — their values must never appear as constants, defaults, prompt
fragments or fixtures in code.

Litmus test: swap in a completely different character's reference pack. If any
code change is needed, the code is wrong.

## Hard prohibitions

1. No hardcoded character content (see golden rule).
2. No hand-written measurement/geometry code as an extraction path (bbox row
   widths, silhouette ratios, boot-height heuristics, head/body ratio math).
   The VLM extracts. A geometric evidence layer may only be added when the user
   explicitly asks for it, as supporting evidence — never as the primary path.
3. No `REQUIRED_*` checklists, no "pending"/"later" states, no warnings that
   stand in for unfinished work. Build it fully, or fail hard with a clear
   error message.
4. No fallbacks that mask failure.
5. `body_proportion` is a model-extracted identity invariant — not a required
   reference role and not a generation case. A missing `body_shape` reference
   is never a hard failure.
6. Never silently simplify or reinterpret the plan. If something seems too
   complex or contradictory: stop and ask. A silent simplification counts as a
   failed task.
7. Create no files beyond the list approved in the plan step below.
8. Do not commit, revert or delete anything without explicit instruction.

## Workflow for every change under `aigen/`

1. Touch no code yet. First present: (a) the task in your own words, (b) the
   exact list of files you will create/modify (and anything you propose to
   delete or revert), (c) 2–3 lines per file describing its contents, (d) every
   assumption you had to make.
2. Wait for an explicit "GO". No code before GO.
3. After GO, implement fully. Anything that does not fit the plan: stop and
   ask — do not pick a direction yourself.

## Fixed facts

- Target GPU: 16GB VRAM (RTX 50-series/Blackwell). Edit-model route: Nunchaku
  Qwen-Image-Edit-2509 FP4/r32; escalation ladder in `docs/PLAN.md`.
- FLUX/Kontext remains a separate pipeline next to Qwen edit; do not merge the
  two.
- The human judges candidate quality via contact sheets; no model is the final
  filter.
