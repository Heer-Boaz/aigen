#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

dry_run="${AIGEN_MODEL_DOWNLOAD_DRY_RUN:-0}"
models_root="${AIGEN_MODELS_ROOT:-$repo_root/aigen/models}"
manifest="$repo_root/model_sources/boogu_image_edit_turbo_fp8.json"
model_root="$models_root/boogu/Boogu-Image-0.1-Edit-Turbo-fp8"

require_file "$manifest"
[[ -x "$venv_python" ]] || die "venv is missing; run scripts/setup_venv.sh first"

command=(
  "$venv_python" -m aigen.cli models download
  --manifest "$manifest"
  --models-root "$models_root"
)
if [[ "$dry_run" == "1" ]]; then
  command+=(--dry-run)
fi
run "${command[@]}"

if [[ "$dry_run" != "1" ]]; then
  require_file "$model_root/model_index.json"
  require_file "$model_root/mllm/model-00001-of-00002.safetensors"
  require_file "$model_root/transformer/diffusion_pytorch_model-00001-of-00002.bin"
  require_file "$model_root/vae/diffusion_pytorch_model.safetensors"
  log "Boogu-Image Edit Turbo FP8 ready: $model_root"
else
  log "Boogu-Image Edit Turbo FP8 download plan validated"
fi
