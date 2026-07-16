#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

python_bootstrap="${PYTHON:-python3.12}"

if [[ ! -x "$venv_python" ]]; then
  run "$python_bootstrap" -m venv "$venv_dir"
fi

run "$venv_python" -m pip install --upgrade pip "setuptools<82" wheel
run "$venv_python" -m pip install -e "${repo_root}[generation]"
run "$venv_python" -m pip install --no-deps \
  "git+https://github.com/black-forest-labs/flux2.git@50fe5162777813d869182b139e83b10743caef15"
run "$venv_python" -m pip install --force-reinstall onnxruntime-gpu==1.26.0

log "venv ready: $venv_dir"
