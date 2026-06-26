#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

python_bootstrap="${PYTHON:-python3.12}"
command -v "$python_bootstrap" >/dev/null 2>&1 || die "Python 3.12 is required; set PYTHON=/path/to/python3.12 if needed"
command -v git >/dev/null 2>&1 || die "git is required"

"$python_bootstrap" - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required for the pinned Nunchaku wheel, got {sys.version.split()[0]}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
  log "nvidia-smi not found; GPU runtime checks will be limited"
fi

log "system check passed"
