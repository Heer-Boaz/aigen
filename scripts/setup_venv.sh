#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

python_bootstrap="${PYTHON:-python3.12}"

if [[ ! -x "$venv_python" ]]; then
  run "$python_bootstrap" -m venv "$venv_dir"
fi

run "$venv_python" -m pip install --upgrade pip "setuptools<82" wheel
# Pip does not replace an installed release with a VCS fix carrying the same version.
textual_requirement="$("$venv_python" - "$repo_root/pyproject.toml" <<'PY'
from importlib.metadata import PackageNotFoundError, distribution
import json
import sys
import tomllib

with open(sys.argv[1], "rb") as source:
    dependencies = tomllib.load(source)["project"]["dependencies"]
requirement = next(req for req in dependencies if req.startswith("textual @ "))
revision = requirement.rsplit("@", 1)[1]
try:
    source_identity = distribution("textual").read_text("direct_url.json")
except PackageNotFoundError:
    source_identity = None
if source_identity is None or json.loads(source_identity).get("vcs_info", {}).get("commit_id") != revision:
    print(requirement)
PY
)"
if [[ -n "$textual_requirement" ]]; then
  run "$venv_python" -m pip install --force-reinstall --no-deps "$textual_requirement"
fi
run "$venv_python" -m pip install -e "${repo_root}[generation]"
run "$venv_python" -m pip install --no-deps \
  "git+https://github.com/black-forest-labs/flux2.git@50fe5162777813d869182b139e83b10743caef15"
run "$venv_python" -m pip install --force-reinstall onnxruntime-gpu==1.26.0

log "venv ready: $venv_dir"
