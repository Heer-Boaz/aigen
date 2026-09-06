# Image-flow acceptance prompt review

Reviewer: `/root/review_gpu_prompts`.

I read the current `docs/prompting.md` and `docs/PLAN.md` completely for this
review, read `manifest.json`, and inspected both original input images with
`view_image`:

- `assets/characters/jillian/references/582142a0-f1bd-4a7d-be0a-cf78a3172586.png`
- `assets/characters/jillian/fullbody-multiview.png`

The assignment is to test the real TUI/CLI generation, cache, viewer, selection
save/reload, and continuation flow. Candidate 2 is selected deliberately to
exercise selection state. This selection is not a judgment that it has the
best illustration style. The continuation uses that selected image as its
only reference and keeps the selected backend.

## Initial generation

Both prompts are approved unchanged.

`flux2-klein`: the ordered references above, 768x1024, four steps, seeds 71 and
72. Image 1 owns the sprite, pose, proportions, and background; image 2 supplies
linework and shading style.

> Redraw the sprite in image 1 as a smooth anime illustration in the linework and shading style of image 2, keeping the pose, proportions, and white background of image 1.

`qwen-image-edit-2511-lightning`: the original sprite alone, 1152x1536, eight
steps, seeds 71 and 72, empty negative prompt.

> Redraw this sprite as a smooth anime illustration. Keep the standing pose, proportions, outfit and white background.

The transformation, image roles, and preservation clauses are direct and
grounded in the inspected inputs. Neither prompt invents character details or
uses pseudo-analysis or a generic negative list.

## Continuation

Approved unchanged for each backend, seed 73, using the selected second
candidate as the sole input:

> Change the background to light gray, keeping the subject and illustration style unchanged.

The requested change appears first. Light gray is the intended output color;
subject and illustration style are relevant preservation targets. No character
description or extra reference role is introduced.

The generated continuation inputs do not yet exist at this review point.
Before each continuation run, the prompt author must inspect the actual
selected image, as required by `docs/prompting.md`: "The author inspects every
input image the prompt refers to." Record that image's identity and the actual
backend/sampler settings with the seed-73 request. This review approves the
wording; it does not claim inspection of a future generated image or successful
preservation in the output.

No model was run and no application code was changed for this review.
