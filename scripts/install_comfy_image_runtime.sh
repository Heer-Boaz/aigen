#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

comfy_revision="26515acd23fa291a8f5ab53c5997258598de0701"
runtime_root="${AIGEN_COMFY_IMAGE_ROOT:-$HOME/.cache/aigen-comfy-image}"
source_root="$runtime_root/ComfyUI"
runtime_venv="$runtime_root/venv"
runtime_python="$runtime_venv/bin/python"
python_version="3.12.13"

if [[ -n "${AIGEN_COMFY_IMAGE_PYTHON:-}" ]]; then
  python_bootstrap="$AIGEN_COMFY_IMAGE_PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
  python_bootstrap="$(command -v python3.12)"
elif command -v uv >/dev/null 2>&1; then
  run uv python install "$python_version"
  python_bootstrap="$(uv python find "$python_version")"
else
  die "Python 3.12 is required; install uv or set AIGEN_COMFY_IMAGE_PYTHON=/path/to/python3.12"
fi

command -v "$python_bootstrap" >/dev/null 2>&1 || die \
  "Python bootstrap is not executable: $python_bootstrap"

if [[ ! -d "$source_root/.git" ]]; then
  run mkdir -p "$runtime_root"
  run git clone --filter=blob:none --no-checkout \
    https://github.com/Comfy-Org/ComfyUI.git "$source_root"
elif ! git -C "$source_root" diff --quiet || ! git -C "$source_root" diff --cached --quiet; then
  die "ComfyUI checkout has local changes: $source_root"
fi

run git -C "$source_root" fetch --depth=1 origin "$comfy_revision"
run git -C "$source_root" checkout --detach FETCH_HEAD

if [[ ! -x "$runtime_python" ]]; then
  run "$python_bootstrap" -m venv "$runtime_venv"
fi

run "$runtime_python" -m pip install --upgrade pip "setuptools<82" wheel packaging
run "$runtime_python" -m pip install \
  torch==2.11.0 \
  torchvision==0.26.0 \
  torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
grep -Ev '^(torch|torchvision|torchaudio)$' "$source_root/requirements.txt" | \
  "$runtime_python" -m pip install -r /dev/stdin
run "$runtime_python" -m pip install transformers==5.3.0

run env PYTHONPATH="$source_root" AIGEN_COMFY_SOURCE_ROOT="$source_root" \
  "$runtime_python" - <<'PY'
import os
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--models-directory", str(Path(os.environ["AIGEN_COMFY_SOURCE_ROOT"]) / "models")]

import torch
from comfy_extras.nodes_hidream_o1 import HiDreamO1ReferenceImages
from nodes import CheckpointLoaderSimple

if not torch.cuda.is_available():
    raise RuntimeError("ComfyUI image runtime requires CUDA")
if torch.cuda.get_device_capability() != (12, 0):
    raise RuntimeError(
        f"Expected RTX 50-series compute capability 12.0, got {torch.cuda.get_device_capability()}"
    )
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
print(CheckpointLoaderSimple.__name__, HiDreamO1ReferenceImages.__name__)
PY

log "ComfyUI image runtime ready: $source_root@$comfy_revision"
