#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

models_root="${AIGEN_MODELS_ROOT:-$repo_root/aigen/models}"
manifest="$repo_root/model_sources/vosr_1_4b_ms.json"

require_file "$manifest"
[[ -x "$venv_python" ]] || die "venv is missing; run scripts/setup_venv.sh first"

run "$venv_python" -m aigen.cli models download \
  --manifest "$manifest" \
  --models-root "$models_root"
