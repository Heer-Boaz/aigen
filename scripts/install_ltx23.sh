#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

wangp_revision="5582327dc25e45fec6cda0f27144d4dcf7ed104b"
runtime_root="${AIGEN_LTX23_ROOT:-$HOME/.cache/aigen-wangp}"
source_root="$runtime_root/Wan2GP"
runtime_venv="$runtime_root/venv"
runtime_python="$runtime_venv/bin/python"
python_version="3.11.14"
lightx2v_kernel_url="https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-linux_x86_64.whl"

if [[ -n "${AIGEN_LTX23_PYTHON:-}" ]]; then
  python_bootstrap="$AIGEN_LTX23_PYTHON"
elif command -v python3.11 >/dev/null 2>&1; then
  python_bootstrap="$(command -v python3.11)"
elif command -v uv >/dev/null 2>&1; then
  run uv python install "$python_version"
  python_bootstrap="$(uv python find "$python_version")"
else
  die "Python 3.11 is required; install uv or set AIGEN_LTX23_PYTHON=/path/to/python3.11"
fi

command -v "$python_bootstrap" >/dev/null 2>&1 || die \
  "Python bootstrap is not executable: $python_bootstrap"

if [[ ! -d "$source_root/.git" ]]; then
  run mkdir -p "$runtime_root"
  run git clone --filter=blob:none --no-checkout \
    https://github.com/deepbeepmeep/Wan2GP.git "$source_root"
elif ! git -C "$source_root" diff --quiet || ! git -C "$source_root" diff --cached --quiet; then
  die "WanGP checkout has tracked local changes: $source_root"
fi

run git -C "$source_root" fetch --depth=1 origin "$wangp_revision"
run git -C "$source_root" checkout --detach FETCH_HEAD

if [[ ! -x "$runtime_python" ]]; then
  run "$python_bootstrap" -m venv "$runtime_venv"
fi

"$runtime_python" - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise RuntimeError(f"Expected Python 3.11, got {sys.version}")
PY

run "$runtime_python" -m pip install --upgrade pip "setuptools<82" wheel packaging
run "$runtime_python" -m pip install \
  torch==2.10.0 \
  torchvision==0.25.0 \
  torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130
run "$runtime_python" -m pip install -r "$source_root/requirements.txt"
run "$runtime_python" -m pip install "$lightx2v_kernel_url"

run env PYTHONPATH="$source_root" "$runtime_python" - <<'PY'
import decord
import torch
import lightx2v_kernel
from shared import api

if not torch.cuda.is_available():
    raise RuntimeError("LTX-2.3 requires CUDA")
if torch.cuda.get_device_capability() != (12, 0):
    raise RuntimeError(
        f"Expected RTX 50-series compute capability 12.0, got {torch.cuda.get_device_capability()}"
    )
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
print(decord.__version__, decord.__file__)
print(lightx2v_kernel.__file__)
print(api.__file__)
PY

log "LTX-2.3 WanGP runtime ready: $source_root@$wangp_revision"
