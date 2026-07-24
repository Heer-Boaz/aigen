#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

dry_run="${AIGEN_MODEL_DOWNLOAD_DRY_RUN:-0}"
models_root="${AIGEN_MODELS_ROOT:-$repo_root/aigen/models}"
manifest="$repo_root/model_sources/animegen_i2v.json"
animegen_model="$models_root/animegen/aidealab/AnimeGen-I2V"
base_model="$models_root/animegen/Wan-AI/Wan2.2-I2V-A14B-Diffusers"
lightning_model="$models_root/animegen/lightx2v/Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"

require_file "$manifest"
[[ -x "$venv_python" ]] || die "venv is missing; run scripts/setup_venv.sh first"

download_args=()
if [[ "$dry_run" == "1" ]]; then
  download_args+=(--dry-run)
fi

run "$venv_python" -m aigen.cli models download \
  --manifest "$manifest" \
  --models-root "$models_root" \
  "${download_args[@]}"

if [[ "$dry_run" == "1" ]]; then
  exit 0
fi

require_file "$animegen_model/transformer/diffusion_pytorch_model.safetensors.index.json"
require_file "$animegen_model/transformer_2/diffusion_pytorch_model.safetensors.index.json"
require_file "$base_model/model_index.json"
require_file "$base_model/text_encoder/model.safetensors.index.json"
require_file "$base_model/tokenizer/tokenizer.json"
require_file "$base_model/vae/diffusion_pytorch_model.safetensors"
require_file "$lightning_model/high_noise_model.safetensors"
require_file "$lightning_model/low_noise_model.safetensors"

log "official AnimeGen-I2V model set ready: $animegen_model"
