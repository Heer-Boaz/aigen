# 🛑 AI & Codex Prompting Guidelines 🛑

**ATTENTION CLAUDE, CODEX, COPILOT, AND OTHER AI AGENTS:**
Read this before generating *any* image prompts for FLUX.2 or Qwen-Image-Edit in this repository.

Your default behavior for generating image prompts is **WRONG** for this pipeline. You have likely been fine-tuned to write prompts for Stable Diffusion 1.5 (e.g., adding "masterpiece", "8k", "highly detailed", "trending on artstation", or writing massive paragraphs). **DO NOT DO THIS.**

## The Core Rule
Write a normal, direct, imperative visual instruction. Treat the prompt like a quick chat message to a professional human graphic designer. 

* **Good:** "Make the girl in picture 1 dance with a capybara."
* **Good:** "Replace the polka-dot shirt with a light blue shirt."
* **Good:** "Redraw this rear-view pixel-art character as a clean anime illustration with smooth linework."
* **BAD:** "A highly detailed masterpiece, 8k resolution, cinematic lighting, a girl in a light blue shirt, beautiful eyes, hyperrealistic, no distortion, preserve exactly the topology of the hat."

## 🚫 STRICTLY FORBIDDEN (Do not use these!)

1. **Quality incantations:** NEVER use `masterpiece`, `best quality`, `8K`, `highly detailed`, `ultra-detailed`.
2. **Negative spam:** NEVER append lists of generic negatives like `no distortion`, `no extra limbs`, `no artifacts`, `no ugly faces`.
3. **Irrelevant negatives:** NEVER forbid things that aren't in the image (e.g., do not say `no watermelon` if there was never a watermelon).
4. **Pseudo-analysis & Topology:** NEVER write things like `preserve topology`, `treat this region as part of the person`, or `connected to the reclining body`.
5. **Length:** Prompts should be 10-40 words. 50-80 words is the absolute maximum for extremely complex edits.

## Why?
- **FLUX.2** has no built-in prompt upsampler and is highly sensitive to exact wording. Bloated prose confuses it.
- **Qwen-2511 (LightX2V)** uses a multimodal conditioner that already analyzes color, shape, texture, and background. Over-explaining causes the model to double-count concepts and hallucinate.
- **AnimeGen (Image-to-Video)** relies on crystal clear, coherent keyframes. AI-hallucinated details in the keyframe will ruin the temporal consistency of the video.

## Final Verification
Before you output a prompt, ask yourself: *"Could a human have naturally typed this into a chat box?"* If the answer is no, rewrite it to be shorter and more direct.

For the full architectural reasoning, read `docs/image-edit-prompting.md`.
