#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

lightx2v_revision="b96309e82899145aebd8ecf95c387894aba66b1e"
runtime_root="${AIGEN_LIGHTX2V_ROOT:-$HOME/.cache/aigen-lightx2v}"
source_root="$runtime_root/LightX2V"
runtime_venv="$runtime_root/venv"
runtime_python="$runtime_venv/bin/python"
python_bootstrap="${AIGEN_LIGHTX2V_PYTHON:-${PYTHON:-python3.12}}"

command -v "$python_bootstrap" >/dev/null 2>&1 || die "Python 3.12 is required; set AIGEN_LIGHTX2V_PYTHON=/path/to/python3.12"

if [[ ! -d "$source_root/.git" ]]; then
  run mkdir -p "$runtime_root"
  run git clone --filter=blob:none --no-checkout https://github.com/ModelTC/LightX2V.git "$source_root"
fi

run git -C "$source_root" fetch --depth=1 origin "$lightx2v_revision"
run git -C "$source_root" checkout --detach FETCH_HEAD

if [[ ! -x "$runtime_python" ]]; then
  run "$python_bootstrap" -m venv "$runtime_venv"
fi

run "$runtime_python" -m pip install --upgrade pip "setuptools<82" wheel packaging ninja
run "$runtime_python" -m pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
run "$runtime_python" -m pip install \
  transformers==4.57.3 \
  diffusers==0.39.0 \
  accelerate==1.14.0 \
  kernels==0.11.3 \
  safetensors==0.8.0
run "$runtime_python" -m pip install -e "$source_root"
run env MAX_JOBS="${MAX_JOBS:-4}" "$runtime_python" -m pip install flash-attn==2.8.3 --no-build-isolation

log "LightX2V runtime ready: $source_root@$lightx2v_revision"
