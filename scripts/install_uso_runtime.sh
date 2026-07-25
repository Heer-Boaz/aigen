#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

uso_revision="6587514aa3adf8e8f46e5f7e804239651d30b32d"
data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
runtime_root="${AIGEN_USO_ROOT:-$data_home/aigen/runtimes/uso}"
source_root="$runtime_root/USO"

command -v git >/dev/null 2>&1 || die "git is required to install the USO runtime"
[[ -x "$venv_python" ]] || die "venv is missing; run scripts/setup_venv.sh first"

if [[ ! -d "$source_root/.git" ]]; then
  run mkdir -p "$runtime_root"
  run git clone --filter=blob:none --no-checkout \
    https://github.com/bytedance/USO.git "$source_root"
elif ! git -C "$source_root" diff --quiet || ! git -C "$source_root" diff --cached --quiet; then
  die "USO checkout has local changes: $source_root"
fi

run git -C "$source_root" fetch --depth=1 origin "$uso_revision"
run git -C "$source_root" checkout --detach FETCH_HEAD

run env PYTHONPATH="$source_root" "$venv_python" - <<'PY'
from uso.flux.pipeline import USOPipeline

print(USOPipeline.__name__)
PY

log "official USO runtime ready: $source_root@$uso_revision"
