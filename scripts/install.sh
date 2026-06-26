#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

usage() {
  cat <<'EOF'
Usage: scripts/install.sh

One-shot installer for aigen. It keeps the steps modular by calling the scripts
next to this file. It always installs the Nunchaku backend, downloads the
keyframe generation models, downloads the Qwen judge model, and runs the final
install check.

Environment:
  PYTHON=/path/to/python3.12         Python used to create .venv
  AIGEN_MODELS_ROOT=/path/to/models  Defaults to ./aigen/models
  NUNCHAKU_WHEEL_URL=https://...     Override the pinned Nunchaku wheel
EOF
}

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

run "$script_dir/check_system.sh"
run "$script_dir/setup_venv.sh"
run "$script_dir/install_nunchaku.sh"
run "$script_dir/download_models.sh"
run "$script_dir/check_install.sh"

log "install complete"
