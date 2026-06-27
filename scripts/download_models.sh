#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

dry_run="${AIGEN_MODEL_DOWNLOAD_DRY_RUN:-0}"
models_root="${AIGEN_MODELS_ROOT:-$repo_root/aigen/models}"

usage() {
  cat <<'EOF'
Usage: scripts/download_models.sh

Downloads the fixed production model set:
- FLUX Kontext 4-bit model and Union ControlNet
- Nunchaku FLUX Kontext FP4 transformer
- DWPose ONNX annotator models for keyframe pose scoring
- Qwen2.5-VL-7B keyframe judge
EOF
}

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -x "$venv_python" ]] || die "venv is missing; run scripts/setup_venv.sh first"

manifest_args=(
  "model_sources/kontext_pose_control_pipeline_4bit.json"
  "model_sources/nunchaku_kontext_pipeline_fp4.json"
  "model_sources/keyframe_pose_dwpose_onnx.json"
  "model_sources/keyframe_judge_qwen2_5_vl_7b.json"
)

for manifest in "${manifest_args[@]}"; do
  require_file "$repo_root/$manifest"
  command=(
    "$venv_python" -m aigen.cli models download
    --manifest "$repo_root/$manifest"
    --models-root "$models_root"
  )
  if [[ "$dry_run" == "1" ]]; then
    command+=(--dry-run)
  fi
  run "${command[@]}"
done

log "model step complete"
