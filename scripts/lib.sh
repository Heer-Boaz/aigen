#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
venv_dir="$repo_root/.venv"
venv_python="$venv_dir/bin/python"

log() {
  printf '[aigen] %s\n' "$*"
}

die() {
  printf '[aigen] error: %s\n' "$*" >&2
  exit 1
}

run() {
  log "$*"
  "$@"
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}
