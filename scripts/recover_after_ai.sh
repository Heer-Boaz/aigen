#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

[[ "$(</proc/sys/kernel/osrelease)" == *microsoft*WSL2* ]] || die "memory reclaim is only supported under WSL2"
[[ -n "${WSL_DISTRO_NAME:-}" ]] || die "WSL_DISTRO_NAME is not set"

if (( EUID != 0 )); then
  script_path="$(readlink -f -- "${BASH_SOURCE[0]}")"
  exec wsl.exe --distribution "$WSL_DISTRO_NAME" --user root --exec "$script_path"
fi

reclaim_control="/sys/fs/cgroup/memory.reclaim"
cache_floor_kib=$((1024 * 1024))
reclaim_chunk_kib=$((1024 * 1024))

[[ -w "$reclaim_control" ]] || die "cgroup v2 memory.reclaim is unavailable"

read_memory_state() {
  awk '
    $1 == "MemAvailable:" { available = $2 }
    $1 == "Buffers:" { buffers = $2 }
    $1 == "Cached:" { cached = $2 }
    $1 == "SReclaimable:" { reclaimable = $2 }
    $1 == "Shmem:" { shmem = $2 }
    END {
      print available, buffers + cached + reclaimable - shmem
    }
  ' /proc/meminfo
}

read -r available_before_kib cache_before_kib < <(read_memory_state)
reclaim_kib=$((cache_before_kib - cache_floor_kib))

if (( reclaim_kib <= 0 )); then
  log "WSL cache is already below 1024 MiB; nothing to reclaim"
else
  log "request $((reclaim_kib / 1024)) MiB of WSL file-cache reclaim"
  remaining_kib=$reclaim_kib
  while (( remaining_kib > 0 )); do
    chunk_kib=$reclaim_chunk_kib
    (( chunk_kib > remaining_kib )) && chunk_kib=$remaining_kib

    if ! printf '%s swappiness=0\n' "$((chunk_kib * 1024))" > "$reclaim_control"; then
      log "kernel reached the currently reclaimable cache limit"
      break
    fi

    remaining_kib=$((remaining_kib - chunk_kib))
  done
fi

read -r available_after_kib cache_after_kib < <(read_memory_state)
log "WSL file cache: $((cache_before_kib / 1024)) MiB -> $((cache_after_kib / 1024)) MiB"
log "WSL available memory: $((available_before_kib / 1024)) MiB -> $((available_after_kib / 1024)) MiB"

if fuser /dev/dxg >/dev/null 2>&1; then
  die "active WSL GPU processes must exit before resetting the NVIDIA driver"
fi

log "reset Windows NVIDIA driver"
powershell.exe -NoProfile -NonInteractive -Command '
  $reset = Start-Process `
    -FilePath "$env:WINDIR\System32\nvidia-smi.exe" `
    -ArgumentList "--gpu-reset" `
    -Verb RunAs `
    -Wait `
    -PassThru

  if ($reset.ExitCode -ne 0) {
    exit $reset.ExitCode
  }

  for ($attempt = 0; $attempt -lt 10; $attempt++) {
    Start-Sleep -Milliseconds 500

    if (-not (Get-Process explorer -ErrorAction SilentlyContinue)) {
      Start-Process "$env:WINDIR\explorer.exe"
    }
  }
'

nvidia-smi.exe \
  --query-gpu=name,driver_model.current,memory.used,utilization.gpu,pstate \
  --format=csv,noheader
