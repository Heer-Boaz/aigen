#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

dry_run="${AIGEN_MODEL_DOWNLOAD_DRY_RUN:-0}"
models_root="${AIGEN_MODELS_ROOT:-$repo_root/aigen/models}"
runtime_root="${AIGEN_HUNYUANVIDEO15_ROOT:-$HOME/.cache/aigen-hunyuanvideo15}"
runtime_modelscope="$runtime_root/venv/bin/modelscope"
hunyuan_manifest="$repo_root/model_sources/hunyuanvideo15_480p_i2v_step_distilled.json"
qwen_manifest="$repo_root/model_sources/keyframe_judge_qwen2_5_vl_7b.json"
hunyuan_model="$models_root/hunyuanvideo15/tencent/HunyuanVideo-1.5"
shared_qwen="$models_root/vlm/Qwen/Qwen2.5-VL-7B-Instruct"
hunyuan_qwen="$hunyuan_model/text_encoder/llm"
glyph_model="$hunyuan_model/text_encoder/Glyph-SDXL-v2"

require_file "$hunyuan_manifest"
require_file "$qwen_manifest"
[[ -x "$venv_python" ]] || die "venv is missing; run scripts/setup_venv.sh first"
[[ -x "$runtime_modelscope" ]] || die \
  "HunyuanVideo-1.5 runtime is missing; run scripts/install_hunyuanvideo15.sh first"

download_args=()
if [[ "$dry_run" == "1" ]]; then
  download_args+=(--dry-run)
fi

run "$venv_python" -m aigen.cli models download \
  --manifest "$hunyuan_manifest" \
  --models-root "$models_root" \
  "${download_args[@]}"
run "$venv_python" -m aigen.cli models download \
  --manifest "$qwen_manifest" \
  --models-root "$models_root" \
  "${download_args[@]}"

if [[ "$dry_run" == "1" ]]; then
  log "planned ModelScope dependency: AI-ModelScope/Glyph-SDXL-v2 -> $glyph_model"
  exit 0
fi

require_file "$shared_qwen/model.safetensors.index.json"
run mkdir -p "$(dirname "$hunyuan_qwen")"
if [[ ! -e "$hunyuan_qwen" ]]; then
  run ln -s "$shared_qwen" "$hunyuan_qwen"
fi
require_file "$hunyuan_qwen/model.safetensors.index.json"

run "$runtime_modelscope" download \
  --model AI-ModelScope/Glyph-SDXL-v2 \
  --local_dir "$glyph_model" \
  --include \
  assets/color_idx.json \
  assets/multilingual_10-lang_idx.json \
  checkpoints/byt5_model.pt

require_file "$hunyuan_model/transformer/480p_i2v_step_distilled/diffusion_pytorch_model.safetensors"
require_file "$hunyuan_model/vae/diffusion_pytorch_model.safetensors"
require_file "$hunyuan_model/scheduler/scheduler_config.json"
require_file "$hunyuan_model/text_encoder/byt5-small/pytorch_model.bin"
require_file "$glyph_model/assets/color_idx.json"
require_file "$glyph_model/assets/multilingual_10-lang_idx.json"
require_file "$glyph_model/checkpoints/byt5_model.pt"
require_file "$hunyuan_model/vision_encoder/siglip/image_encoder/model.safetensors"
require_file "$hunyuan_model/vision_encoder/siglip/feature_extractor/preprocessor_config.json"

log "official HunyuanVideo-1.5 model set ready: $hunyuan_model"
