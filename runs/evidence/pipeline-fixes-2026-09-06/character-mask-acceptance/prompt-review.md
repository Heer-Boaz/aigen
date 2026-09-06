# Masked edit prompt and input review

The author inspected the original 768x1024 source and the real SAM2 repaint
mask, then wrote the direct instruction in manifest.json. The source and mask
SHA-256 identities are recorded there before Qwen generation.

Independent reviewer `review_gpu_prompts` reread docs/prompting.md and
PLAN.md completely, inspected both images and verified both full hashes.
It approved the generation prompt and the integrated audit protocol without
changes: image 1 source/reference, image 2 repaint mask, images 3/4 candidates.

Approval covers native Qwen-2511 Lightning, 768x1024, eight-step schedule,
strength 0.6 (five suffix steps), seeds 101/102, optional retry seeds 103/104,
two rounds maximum, empty negative prompt, no VOSR. Generated candidate images
may be audited automatically under this approved protocol. This approval does
not predetermine the output quality verdict.
