#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

hunyuan_revision="60783e704160023913bee78f0b47036d393d4dfa"
runtime_root="${AIGEN_HUNYUANVIDEO15_ROOT:-$HOME/.cache/aigen-hunyuanvideo15}"
source_root="$runtime_root/HunyuanVideo-1.5"
runtime_venv="$runtime_root/venv"
runtime_python="$runtime_venv/bin/python"
python_bootstrap="${AIGEN_HUNYUANVIDEO15_PYTHON:-${PYTHON:-python3.12}}"
runtime_patch="$repo_root/patches/hunyuanvideo15/0001-release-cuda-cache-after-component-offload.patch"

command -v "$python_bootstrap" >/dev/null 2>&1 || die \
  "Python 3.12 is required; set AIGEN_HUNYUANVIDEO15_PYTHON=/path/to/python3.12"
require_file "$runtime_patch"

if [[ ! -d "$source_root/.git" ]]; then
  run mkdir -p "$runtime_root"
  run git clone --filter=blob:none --no-checkout \
    https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git "$source_root"
else
  if ! git -C "$source_root" diff --quiet || ! git -C "$source_root" diff --cached --quiet; then
    git -C "$source_root" diff --cached --quiet || die \
      "official HunyuanVideo-1.5 checkout has staged local changes: $source_root"
    cmp -s \
      <(git -C "$source_root" diff --no-ext-diff --binary --unified=0) \
      "$runtime_patch" || die \
      "official HunyuanVideo-1.5 checkout has unexpected local changes: $source_root"
    run git -C "$source_root" apply --unidiff-zero --reverse "$runtime_patch"
  fi
fi

run git -C "$source_root" fetch --depth=1 origin "$hunyuan_revision"
run git -C "$source_root" checkout --detach FETCH_HEAD
run git -C "$source_root" apply --unidiff-zero "$runtime_patch"

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
  tqdm==4.67.1 \
  peft==0.17.0 \
  openai==2.8.0 \
  einops==0.8.0 \
  loguru==0.7.3 \
  numpy==1.26.4 \
  pillow==11.3.0 \
  imageio==2.37.0 \
  imageio-ffmpeg==0.6.0 \
  angelslim==0.2.2 \
  omegaconf==2.3.0 \
  diffusers==0.35.0 \
  safetensors==0.6.2 \
  qwen-vl-utils==0.0.8 \
  huggingface-hub==0.34.0 \
  "transformers[accelerate,tiktoken]==4.57.1" \
  psutil==7.1.3 \
  modelscope==1.38.1
run env MAX_JOBS="${MAX_JOBS:-4}" \
  "$runtime_python" -m pip install flash-attn==2.8.3 --no-build-isolation

run env PYTHONPATH="$source_root" "$runtime_python" - <<'PY'
import torch
from hyvideo.commons import PIPELINE_CONFIGS

if not torch.cuda.is_available():
    raise RuntimeError("HunyuanVideo-1.5 requires CUDA")
if torch.cuda.get_device_capability() != (12, 0):
    raise RuntimeError(
        f"Expected RTX 50-series compute capability 12.0, got {torch.cuda.get_device_capability()}"
    )
profile = PIPELINE_CONFIGS["480p_i2v_step_distilled"]
expected = {
    "guidance_scale": 1.0,
    "embedded_guidance_scale": None,
    "flow_shift": 7.0,
    "num_inference_steps": 12,
}
if profile != expected:
    raise RuntimeError(f"Unexpected Tencent step-distilled profile: {profile}")
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
print(profile)
PY
run "$runtime_python" "$source_root/generate.py" --help

log "official HunyuanVideo-1.5 runtime ready: $source_root@$hunyuan_revision"
