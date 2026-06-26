#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

[[ -x "$venv_python" ]] || die "venv is missing; run scripts/setup_venv.sh first"

"$venv_python" - <<'PY'
import importlib.util

required = [
    "PIL",
    "numpy",
    "scipy",
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "controlnet_aux",
    "segment_anything",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing Python packages: {', '.join(missing)}")
PY

if [[ "${AIGEN_CHECK_NUNCHAKU:-1}" == "1" ]]; then
  "$venv_python" - <<'PY'
import importlib.util

if importlib.util.find_spec("nunchaku") is None:
    raise SystemExit("missing Python package: nunchaku")
PY
fi

run "$venv_python" -m aigen.cli --help
run "$venv_python" -m aigen.cli keyframes schema --compact

if [[ "${AIGEN_SKIP_MODEL_CHECK:-0}" == "1" ]]; then
  log "model-backed keyframe validation skipped"
else
  run "$venv_python" -m aigen.cli keyframes validate "$repo_root/jobs/ai46/walk_contact_640x960_ref384_seed_sweep.json" --compact
fi

log "install check passed"
