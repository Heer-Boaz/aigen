#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

vosr_revision="25fbf8e6cb9656b8991c24474f408bdce6fcb1b1"
runtime_root="${AIGEN_VOSR_ROOT:-$HOME/.cache/aigen-vosr}"
source_root="$runtime_root/VOSR"
models_root="${AIGEN_MODELS_ROOT:-$repo_root/aigen/models}"
checkpoint_root="$models_root/vosr/CSWRY/VOSR/preset/ckpts"

if [[ ! -d "$source_root/.git" ]]; then
  run mkdir -p "$runtime_root"
  run git clone https://github.com/cswry/VOSR.git "$source_root"
fi

run git -C "$source_root" fetch origin "$vosr_revision"
run git -C "$source_root" checkout --detach "$vosr_revision"
run mkdir -p "$source_root/preset"
run ln -sfn "$checkpoint_root" "$source_root/preset/ckpts"

log "official VOSR source ready: $source_root@$vosr_revision"
