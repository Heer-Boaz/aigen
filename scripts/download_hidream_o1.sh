#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

dry_run="${AIGEN_MODEL_DOWNLOAD_DRY_RUN:-0}"
models_root="${AIGEN_MODELS_ROOT:-$repo_root/aigen/models}"
manifest="$repo_root/model_sources/hidream_o1_full_fp8_comfy.json"
checkpoint="$models_root/comfy/checkpoints/hidream_o1_image_fp8_scaled.safetensors"

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
  require_file "$checkpoint"
fi
log "HiDream-O1 Full FP8 model ready: $checkpoint"
