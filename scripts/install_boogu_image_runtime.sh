#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

boogu_revision="29c040ff975d19231911753a0dbf976ae98621b1"
runtime_root="${AIGEN_BOOGU_ROOT:-$HOME/.cache/aigen-boogu}"
source_root="$runtime_root/Boogu-Image"
runtime_venv="$runtime_root/venv"
runtime_python="$runtime_venv/bin/python"
python_bootstrap="${AIGEN_BOOGU_PYTHON:-}"

command -v uv >/dev/null 2>&1 || die "uv is required to install the pinned Boogu runtime"
if [[ -z "$python_bootstrap" ]]; then
  command -v python3.12 >/dev/null 2>&1 || die \
    "Python 3.12 is required; set AIGEN_BOOGU_PYTHON=/path/to/python3.12"
  python_bootstrap="$(command -v python3.12)"
fi
command -v "$python_bootstrap" >/dev/null 2>&1 || die \
  "Python bootstrap is not executable: $python_bootstrap"

if [[ ! -d "$source_root/.git" ]]; then
  run mkdir -p "$runtime_root"
  run git clone --filter=blob:none --no-checkout \
    https://github.com/Boogu-Project/Boogu-Image.git "$source_root"
elif ! git -C "$source_root" diff --quiet || ! git -C "$source_root" diff --cached --quiet; then
  die "Boogu-Image checkout has local changes: $source_root"
fi

run git -C "$source_root" fetch --depth=1 origin "$boogu_revision"
run git -C "$source_root" checkout --detach FETCH_HEAD

if [[ ! -x "$runtime_python" ]]; then
  run uv venv --python "$python_bootstrap" "$runtime_venv"
fi

lock_file="$source_root/requirements/lock-torch2.11-cu128-linux-py312.txt"
require_file "$lock_file"
run uv pip install --python "$runtime_python" -r "$lock_file"
run uv pip install --python "$runtime_python" --no-deps --editable "$source_root"

run env PYTHONPATH="$source_root" device="cuda:0" "$runtime_python" - <<'PY'
import torch
from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline

if not torch.cuda.is_available():
    raise RuntimeError("Boogu-Image runtime requires CUDA")
if torch.cuda.get_device_capability() != (12, 0):
    raise RuntimeError(
        f"Expected RTX 50-series compute capability 12.0, got {torch.cuda.get_device_capability()}"
    )
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
print(BooguImageTransformer2DModel.__name__, BooguImageTurboPipeline.__name__)
PY

log "Boogu-Image runtime ready: $source_root@$boogu_revision"
