#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

[[ -x "$venv_python" ]] || die "venv is missing; run scripts/setup_venv.sh first"

nunchaku_wheel_url="${NUNCHAKU_WHEEL_URL:-https://github.com/nunchaku-tech/nunchaku/releases/download/v1.3.0dev20260306/nunchaku-1.3.0.dev20260306%2Bcu12.8torch2.12-cp312-cp312-linux_x86_64.whl}"

run "$venv_python" -m pip install "peft>=0.17" protobuf sentencepiece
run "$venv_python" -m pip install --no-deps "$nunchaku_wheel_url"

"$venv_python" - <<'PY'
import importlib.metadata

print(importlib.metadata.version("nunchaku"))
PY

log "nunchaku installed"
