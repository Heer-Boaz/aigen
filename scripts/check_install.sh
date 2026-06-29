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

"$venv_python" - <<'PY'
import onnxruntime as ort

providers = ort.get_available_providers()
if "CUDAExecutionProvider" not in providers:
    raise SystemExit(f"onnxruntime CUDAExecutionProvider is unavailable: {providers}")
PY

run "$venv_python" -m aigen.cli --help
run "$venv_python" -m aigen.cli briefs schema --compact
run "$venv_python" -m aigen.cli briefs plan-schema --compact
run "$venv_python" -m aigen.cli characters view-schema --compact
run "$venv_python" -m aigen.cli characters view-bank-schema --compact
run "$venv_python" -m aigen.cli keyframes schema --compact
run "$venv_python" -m aigen.cli keyframes refine-schema --compact
run "$venv_python" -m aigen.cli keyframes polish-schema --compact
run "$venv_python" -m aigen.cli keyframes polish-plan-schema --compact

log "install check passed"
