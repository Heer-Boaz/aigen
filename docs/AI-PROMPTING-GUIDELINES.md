# 🛑 AI & Codex Prompting Guidelines 🛑

**ATTENTION CLAUDE, CODEX, COPILOT, AND OTHER AI AGENTS:**
Read this before generating *any* prompts for image generation, image editing, video generation, or paired-dataset creation (pix2pix) in this repository.

Your default behavior for generating prompts is likely **WRONG** for this pipeline. You have been fine-tuned to write prompts for Stable Diffusion 1.5 (e.g., adding "masterpiece", "8k", "highly detailed", "trending on artstation", or writing massive paragraphs). **DO NOT DO THIS.**

## 1. The Core Rule (General)
Write a normal, direct, imperative visual instruction. Treat the prompt like a quick chat message to a professional human graphic designer or animator. 

* **Good:** "Make the girl in picture 1 dance with a capybara."
* **Good:** "Replace the polka-dot shirt with a light blue shirt."
* **BAD:** "A highly detailed masterpiece, 8k resolution, cinematic lighting, a girl in a light blue shirt, beautiful eyes, hyperrealistic, no distortion, preserve exactly the topology of the hat."

## 2. Pix2Pix Reverse-Source Prompts
The goal here is single-reference whole-image style conversion (pixel-art sprite -> smooth anime illustration).

* **Baseline shape:** "Redraw this pixel-art character as a clean anime illustration with smooth linework and shading, keeping the same character design and pose on a white background."
* **Disambiguation (only when necessary):** If the 128x128 sprite is visually ambiguous, add the *smallest sufficient set* of grounded modifiers to the baseline.
  * *Example:* "Redraw this **rear-view** pixel-art character..."
  * *Example:* "Redraw this **left-facing** pixel-art character..."
  * *Example:* "Redraw this **reclining** pixel-art character..."
* **Forbidden in pix2pix:** Do not explain how limbs connect, do not invent spatial relationships, do not add speculative anatomy narratives.

## 3. Video Generation (AnimeGen / I2V)
When prompting Image-to-Video models using a Qwen-edited keyframe:
* **Focus on Motion:** State the primary continuous action clearly and concisely. E.g., "The character is walking forward, hair blowing gently in the wind."
* **Avoid Contradictions:** Do not specify multiple sequential actions in one prompt (e.g., "she sits down, then stands up, then runs"). Models struggle with complex timelines. Stick to one temporal concept.
* **Trust the Keyframe:** Do not re-describe the character's clothing or face if it is already perfectly established in the input keyframe. Over-describing static visual elements dilutes the model's focus on the *motion* tokens.
* **No Camera Spam:** Avoid spamming "panning, zooming, 4k cinematic camera, drone shot" unless a specific camera movement is explicitly required by the user.

## 4. LoRA Prompting
When utilizing a LoRA (Low-Rank Adaptation) on top of a base model:
* **Trigger Words First:** Always place the exact LoRA trigger word(s) at the very beginning of the prompt.
* **Understand the LoRA's Captioning Strategy:** How you prompt a character LoRA depends entirely on how its training data was captioned:
  * **Sparse Captions (Entangled):** If the LoRA was trained mostly on the trigger word alone, the trigger word contains the entire character (face, hair, default outfit). *Do not* over-describe the character, as it dilutes the weights. Keep it minimal: "[TRIGGER_WORD], sitting on a bench."
  * **Dense Captions (Disentangled):** If the training images were heavily captioned (describing the exact clothes, hair, and eyes in every image), the LoRA learned to separate the character's core identity from her outfit. In this case, you *must* explicitly prompt for the outfit (e.g., "[TRIGGER_WORD] wearing her blue jacket and black boots") to reconstruct the canonical look.

## 🚫 STRICTLY FORBIDDEN (Do not use these in ANY prompt!)

1. **Quality incantations:** NEVER use `masterpiece`, `best quality`, `8K`, `highly detailed`, `ultra-detailed`.
2. **Negative spam:** NEVER append lists of generic negatives like `no distortion`, `no extra limbs`, `no artifacts`, `no ugly faces`.
3. **Irrelevant negatives:** NEVER forbid things that aren't in the image (e.g., do not say `no watermelon` if there was never a watermelon).
4. **Pseudo-analysis & Topology:** NEVER write things like `preserve topology`, `treat this region as part of the person`, or `connected to the reclining body`.
5. **Length:** Prompts should ideally be 10-40 words. 50-80 words is the absolute maximum for extremely complex edits.

## Final Verification
Before you output a prompt, ask yourself: *"Could a human have naturally typed this into a chat box?"* If the answer is no, rewrite it to be shorter and more direct.
