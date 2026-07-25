#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

dry_run="${AIGEN_MODEL_DOWNLOAD_DRY_RUN:-0}"
models_root="${AIGEN_MODELS_ROOT:-$repo_root/aigen/models}"
manifest="$repo_root/model_sources/uso_flux1_native.json"
model_root="$models_root/uso"

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

require_file "$model_root/bytedance-research/USO/uso_flux_v1.0/dit_lora.safetensors"
require_file "$model_root/bytedance-research/USO/uso_flux_v1.0/projector.safetensors"
require_file "$model_root/xlabs-ai/xflux_text_encoders/model.safetensors.index.json"
require_file "$model_root/openai/clip-vit-large-patch14/model.safetensors"
require_file "$model_root/google/siglip-so400m-patch14-384/model.safetensors"
require_file "$model_root/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"
require_file "$model_root/black-forest-labs/FLUX.1-dev/ae.safetensors"

log "official native USO model set ready: $model_root"
