#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

[[ "$(</proc/sys/kernel/osrelease)" == *microsoft*WSL2* ]] || die "memory reclaim is only supported under WSL2"
[[ -n "${WSL_DISTRO_NAME:-}" ]] || die "WSL_DISTRO_NAME is not set"

log "release WSL filesystem caches"
run wsl.exe --distribution "$WSL_DISTRO_NAME" --user root --exec \
  sh -c 'sync; printf "3\n" > /proc/sys/vm/drop_caches'
log "WSL filesystem caches released"
