#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

models_root="${AIGEN_MODELS_ROOT:-$repo_root/aigen/models}"
model_dir="$models_root/wu_pixelization"

download_weight() {
  local name="$1"
  local file_id="$2"
  local expected_sha256="$3"
  local output="$model_dir/$name"
  local partial="$output.part"

  if [[ -f "$output" ]] && [[ "$(sha256sum "$output" | cut -d' ' -f1)" == "$expected_sha256" ]]; then
    log "$name is already present"
    return
  fi
  run curl -fL --retry 3 \
    "https://drive.usercontent.google.com/download?id=$file_id&export=download&confirm=t" \
    -o "$partial"
  [[ "$(sha256sum "$partial" | cut -d' ' -f1)" == "$expected_sha256" ]] || die "checksum mismatch: $name"
  run mv "$partial" "$output"
}

run mkdir -p "$model_dir"
download_weight \
  alias_net.pth \
  17f2rKnZOpnO9ATwRXgqLz5u5AZsyDvq_ \
  3d273c7cc02ce2e14f8840ad3375942bffe942ac7a92f68dddabe523fbd7ee81
download_weight \
  160_net_G_A.pth \
  1i_8xL3stbLWNF4kdQJ50ZhnRFhSDh3Az \
  0856b831c7eca0e594ca4c037e01497d453b795dfdab040e2a9ed2bcea1168d3
