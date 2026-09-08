# Image, character and video workflows

The Workflows tab and `aigen workflow` commands use the same document,
compiler, executor, and cache. Klein and Qwen are separate, explicit backend
choices. A missing runtime or model is an error for the selected route.

## Work on the canvas

Click a node to select it and drag its body to move it. Drag a port to a
compatible port to connect nodes. Select a connection before dragging its
endpoint to reconnect it. A completed drag is one undo step; Escape cancels
the preview. The background and middle mouse button pan the canvas. The wheel
scrolls vertically; Shift+wheel scrolls horizontally.

Right-click or Shift+F10 opens actions for the current node, connection, or
empty canvas. Node actions include **Run to here**, **Results** for viewable
outputs, and **Seed variants** for image edits. The inspector's **⋯** menu
exposes the same actions. **+ Node** or Insert opens a searchable node picker.
Adding from the canvas context menu places the new node near the clicked
location.

The single top bar keeps document commands in **Menu** (F10). Ctrl+S saves,
Ctrl+O opens a workflow, and Ctrl+Z/Ctrl+Y undo and redo graph edits. Text fields
retain their editing keys. Arrow keys select nodes; Shift+arrow moves them.
**Run** (F5) changes to **Stop** while executing. Shift+F5 reaches the running
process even when a menu or results dialog is open.

On a wide terminal the inspector sits beside the canvas. On a narrow terminal,
**Inspect** or Enter opens it as a side drawer; Escape or **×** closes it.
Resizing and closing the drawer preserve unfinished input. Before changing the
document or selection, the editor commits valid input or focuses the invalid
field. A property edit and the following node drag have separate undo steps.

Prompts and negative prompts use a multiline editor. Enter inserts a line;
Tab and Shift+Tab move between fields. Ctrl+Enter applies the current property
edits; Ctrl+S applies them and saves the workflow. Pasting retains all lines,
including blank lines. The active text editor keeps its selection and local
undo history across saves, backend changes, and drawer resizes. Ctrl+Z/Ctrl+Y
operate on that text while it has focus; focus the canvas to undo graph edits.
F7 selects all text. Backend details follow the editable properties.

For ports that accept multiple inputs, select a connection and choose its
**Input order** position in the inspector. The ordered source list shows the
result. Moving from position 1 to 3 inserts the connection after the other two
in one undo step. This orders connections: a reference-pack connection may
contain several images.

## Generate, compare, choose, continue

1. Choose **New image flow**, or **Open as workflow** on the Images form.
   Form import retains populated image/reference-pack order, LoRAs and weights,
   sampling settings, and seeds. The workflow owns its output locations; export
   an original result when a particular destination is needed.
2. Set the reference paths and the first edit's backend, prompt, and settings.
   Incoming reference connections have an explicit order. The inspector shows
   the backend's reference roles, canvas alignment, LoRA compatibility, and
   strength meaning.
3. Select the edit and choose **Seed variants** from its context menu or
   **⋯** menu in the inspector. Enter distinct integer seeds.
   This creates one edit per seed, a collection, and a selection node. Existing
   consumers of the edit are reconnected to the selection. This is one undo
   operation. Importing the Images form already creates this structure.
4. Save the workflow. Select the collection and choose **Run to here**. Only
   its prerequisites run; an unfinished later edit does not block this target.
5. Open **Results**. Inspect a candidate to see its original input previews,
   seed, effective settings, measured dimensions, model/runtime provenance, and
   record/log files. Thumbnails use the same display bounds and do not resize
   model references. The history selector opens earlier runs of this node.
6. Choose **Select image**. The editor saves the producer signature, output
   port, artifact identity, and result-manifest location in the selection node.
   Selection requires the original artifact to pass its integrity check.
   Opening Results on that selection starts at **Saved selection**, even when
   its collection has newer runs. Those runs remain available in the history.
7. Configure the following edit and use **Run**, or target a later node with
   **Run to here**. The saved choice is an input boundary: its generating nodes
   are not rerun. Changing an earlier seed or prompt does not change the choice.
   To choose again, run the collection and select a different saved result.

**Open original** launches the operating system's file viewer. **Export** copies
the original file bytes to a new path with the same extension; it does not
overwrite an existing destination. Export verifies the bytes during the same
read that copies them, including each frame and audio file in a frame archive.
It continues when you inspect another result or close Results, and reports
completion or failure with an application notification. Quitting waits for
active exports to finish. A later edit can also connect directly to an
image-producing node when no human selection is desired.

Running a continuation without a saved choice fails during compilation with a
concrete selection instruction. Merely highlighting a candidate does not save
it. A missing or modified historical input is displayed as unavailable, while
the already-generated output remains inspectable. Older records without input
file checksums cannot supply verified input previews.

New workflow runs capture source images, reference packs, LoRAs, video and audio
in `cache/sources`. Generators and result comparisons read these captured bytes,
so editing or removing an original file later does not change a saved result's
inputs. Capture preserves image resolution and reference order. The original
source location remains in the source node's effective settings. Unchanged
files reuse their stored copy; modified files receive a new content identity.
This storage consumes disk space alongside the generated-output cache.

**Stop** and quitting terminate the active command's process group without
blocking the interface. A process that has not stopped after five seconds is
killed, including descendants. Contact-sheet generation uses this same lifecycle.
Malformed progress events fail visibly and release the controls. Ctrl-C and
SIGTERM record an interrupted workflow and retain already completed outputs.
The TUI assigns the run ID before starting the command. After the process group
has exited, it finalizes that run's interrupted state if needed and restores
completed node statuses from the published results. This also preserves results
whose completion event was lost during a forced kill.

## Documents and results

Each document has a persistent `workflow_id`. Rename, Save As, undo/redo, and
editor restarts retain it. Old documents obtain an ID from their original
absolute path when loaded, then save it in document version 4. Copying the JSON
with its ID preserves the same history; creating a new workflow creates a new
ID. Pre-version-4 execution directories remain on disk but do not appear in
the new ID-based history automatically.

By default, run records live under
`runs/workflows/runs/<workflow_id>/<attempt>/`. They contain the authored
workflow, resolved execution scope and settings, per-node result manifests,
and permanent backend record directories. Effective random seeds are recorded
before generation starts. Cached outputs live under `runs/workflows/cache`;
their records preserve the settings, inputs, measurements, and logs of the
original generation when reused in a later run.

Cache identity depends on effective configuration, ordered artifact inputs,
models, and implementation revisions. Titles and layout do not change it.
Changing one fixed seed regenerates that edit and its dependent outputs;
unchanged variants remain cached. Every fully saved Klein/Qwen batch output is
published separately. If the remainder fails or is stopped, a retry reuses
completed images. An unfinished denoise/decode is not an image result.

Compatible AnimeGen/LTX nodes with the same ordered inputs and effective
settings share one seed sweep. Each completed video is measured and published
before the next result arrives; stopping or failing later in the sweep retains
the completed videos. Different prompts, fit policies, frame positions or
sampling settings use separate sweeps. Cached seeds do not reload the model.

Original backend outputs and cached files are hard-linked within the workflow
run root. Treat these as immutable artifacts and export before editing an
image in another application. Editing the original in place invalidates it.

## Video from images or an existing file

Choose **Open as workflow** on the Video form to retain the selected backend,
keyframes, fit policy, settings and seeds. Each seed becomes a video node.
Alternatively, add nodes to an existing image workflow and connect a saved
image selection to the desired video input.

| Route | Image inputs | Timing and canvas | Generated audio |
| --- | --- | --- | --- |
| AnimeGen | Start and optional end image | Configurable fps; at least 5 frames, `4n+1`; aspect-based native canvas | None |
| LTX-2.3 | Positioned keyframes: start, end and internal frames | Configurable fps; at least 17 frames, `8n+1`; requested dimensions round up to multiples of 64 | Removed by this adapter |
| HunyuanVideo-1.5 | Start image | Fixed 24 fps, `4n+1` frame count; native 480p canvas | None |

LTX frame numbers start at zero. Connect each image through a **Positioned
keyframe** node and enter its frame number. Duplicate positions and positions
outside the requested video fail during compilation. A single keyframe must
be at frame zero. Model availability is checked before execution; Hunyuan's
runtime/model is not installed in the tested local setup.

AnimeGen and LTX expose **crop**, **pad** (white background), and **stretch**.
The chosen policy is applied on the effective generation canvas and recorded
with the result. Source files remain the original references. Older AnimeGen
documents retain their existing stretch policy.

For an existing recording, use a **Video source** node and choose the file with
the inspector's **Browse** button. **Audio source** also supports browsing,
including video containers that hold the desired audio track. Connect a generated
or imported video to **Extract video frames**, optionally through frame
postprocessing, and then to **Assemble video**. The frame sequence carries
ordered file identities, rational timestamps and per-frame durations. Variable
frame timing survives extraction and assembly. The final decoded frame's
duration determines the presentation endpoint; a longer audio track cannot
extend the last frame. Files without a recoverable final-frame duration fail
with a timing error.

Assembly offers three audio policies:

- **Preserve** keeps the source's recorded first audio track and its timing
  relative to the video. Audio is encoded as AAC in the assembled MP4.
- **Remove** creates a silent video.
- **Replace** uses a connected **Audio source**. Its optional stream number
  is the absolute container stream index. The selected track starts at the
  video origin. Short audio does not shorten the video; excess audio is trimmed.

Pixel-only frame processing is cached independently of timing and audio.
Reusing processed pixels still carries the current sequence's timestamps and
audio location into assembly. Transparent frames use assembly's explicit
background color when encoded into H.264.

**Results** shows measured dimensions, frame count, timing and audio alongside
the original settings and logs. Video previews decode one poster frame;
**Open original** opens the complete video. Exporting a video copies its
original bytes. Exporting a frame sequence creates a ZIP with PNG frames,
`frames.json` and its source audio container when present. After unpacking, the
relative manifest paths can be passed directly to `video-postprocess assemble`.

## Audited character editing

Choose **New character flow** and supply a reference pack and a direct edit
instruction. The **Character edit** node defaults to Klein and two raw
candidates. It compares the candidates visually against the complete reference
pack, selects a passing candidate, and runs VOSR to a 2048px long side. Leave
the upscale size blank to retain the accepted raw image.

An unsuccessful audit gets at most one additional generation round by default,
using new seeds on the chosen backend. The retry seeds belong to the individual
edit; other jobs and cache hits do not change them. The run fails with its saved
audit report when the configured rounds are exhausted or the audit response is
invalid. Raw candidates and individual audit requests/responses remain in
**Records / log**. The displayed result seed identifies the selected image;
the effective settings also retain the initial seed.

The audit checks completion of the requested edit as well as unintended visual
changes. Its verdict is a model judgment; inspect the image and saved report
when choosing a result. Acceptance testing has exposed an incomplete edit
that the model initially passed.

The explicit Qwen-2511 Lightning route supports a native pose source or a
DWPose keypoint control, together with optional scene depth/edges. Scene
framing determines the canvas unless an explicit canvas/aspect is supplied.
Scene controls keep their complete framing; pose keypoints use the existing
content-box fitting. Native reference resolution is retained.

For numbered prompts, image order is references, native pose image if used,
then scene control, then pose keypoints if used. The audit receives original
pose/scene sources in the corresponding order. It never rewrites the edit
instruction into a character description.

`characters qwen-edit-run` uses the same audit owner. It accepts the active
2511 Lightning profile and defaults to two candidates and VOSR after approval.
Use `--postprocess none` for raw output or `--max-iterations` to set the bound.
The ordinary **Image edit** node retains its unaudited contract.

## Regional edits with SAM or an existing mask

On **SAM Edit**, set the image and selection settings, then choose **Open as
workflow**. The **SAM segmentation** node exposes a repaint mask, cutout and
preview. Add **Qwen regional edit** and connect the same source image and the
mask. References and compatible Qwen LoRAs are optional extra inputs. Inspect
or export the mask in **Results** before continuing when needed.

A supplied mask goes through **Bind mask to source**: connect the source and
an Image node containing the mask. White means editable; black means preserve;
grey edges retain their feathering. The mask must have the source's displayed
dimensions. Its saved identity includes the original source content, so a mask
cannot silently be reused with a different source. An identical exported and
re-imported source remains compatible.

The SAM form's **Qwen edit** operation also opens as this workflow, including
selection from an existing region plan. Region-plan imports retain the source
and mask checksums saved by that plan. Creating a new text-region plan remains
available through the form's explicit **Text regions + SAM2 masks** operation.

The regional editor uses Qwen-2511 Lightning with native masking during each
denoising step. New edits default to strength 1.0 and use the complete
eight-step schedule. `strength` sets the starting noise level and selects
`round(steps × strength)` steps; 0.6 executes five steps. Existing saved
strength values are retained.
The source determines the canvas, padded to multiples of 16. Leave **max side**
blank for native resolution; an explicit value caps the generation canvas.

Final compositing uses the original source dimensions and mask. PNG output
preserves original RGB and alpha outside the mask. The raw composite passes
through the same bounded audit as whole-character edits; the decode before
compositing and the exact preservation report remain in the run records.
A full-image upscale is excluded from this route because it would change
protected pixels. Image 1 is the source, followed by references in connection
order. The prompt stays the direct user instruction.

`characters qwen-edit-refine` calls this same owner. Its explicit backend is
2511 Lightning; `--pack` is optional and `--max-iterations` controls the audit bound. Old Nunchaku
profile or crop-padding requests fail instead of switching model or strategy.

## LoRA capabilities

Dataset preparation produces image/caption pairs with train/validation splits,
TXT captions and JSONL metadata. The local trainer is the
[FLUX.1 DreamBooth-LoRA trainer](https://github.com/huggingface/diffusers/blob/v0.38.0/examples/dreambooth/train_dreambooth_lora_flux.py).
Training plans and preflight reports identify that architecture and reject a
different transformer class.

LoRA import is a separate capability: the image backend validates the adapter's
architecture before generation. Klein, FLUX.2 dev and Qwen adapters can be
imported for their matching backends. `aigen lora capabilities` lists dataset,
local training and image-edit import support separately. This capability list
does not claim that the corresponding model weights are installed.

## CLI

```bash
.venv/bin/aigen workflow new --output workflow.json
.venv/bin/aigen workflow validate --input workflow.json --target collection-id
.venv/bin/aigen workflow run --input workflow.json --target collection-id
.venv/bin/aigen workflow run --input workflow.json
.venv/bin/aigen workflow new --template video --output video-workflow.json
.venv/bin/aigen workflow new --template character --output character-workflow.json
.venv/bin/aigen lora capabilities
.venv/bin/aigen video-postprocess assemble --frames-manifest unpacked/frames.json --output assembled.mp4 --audio-policy preserve
```

`--target` can be repeated. Without a target the executor runs the authored
terminal nodes and the prerequisites they need, stopping at saved selections.
The TUI's default image template is intentionally unfilled; provide references
and an edit instruction before running it.
