#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

[[ -x "$venv_python" ]] || die "venv is missing; run scripts/setup_venv.sh first"

lightx2v_root="${AIGEN_LIGHTX2V_ROOT:-$HOME/.cache/aigen-lightx2v}"
lightx2v_python="$lightx2v_root/venv/bin/python"
models_root="${AIGEN_MODELS_ROOT:-$repo_root/aigen/models}"

"$venv_python" - <<'PY'
import importlib.util

required = [
    "PIL",
    "numpy",
    "scipy",
    "torch",
    "torchvision",
    "diffusers",
    "datasets",
    "ftfy",
    "jinja2",
    "kernels",
    "tensorboard",
    "transformers",
    "controlnet_aux",
    "comfy_kitchen",
    "flux2",
    "segment_anything",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing Python packages: {', '.join(missing)}")
PY

if [[ "${AIGEN_CHECK_NUNCHAKU:-1}" == "1" ]]; then
  "$venv_python" - <<'PY'
import importlib.util

if importlib.util.find_spec("nunchaku") is None:
    raise SystemExit("missing Python package: nunchaku")
PY
fi

[[ -x "$lightx2v_python" ]] || die "LightX2V venv is missing; run scripts/install_lightx2v.sh first"
require_file "$lightx2v_root/LightX2V/lightx2v/__init__.py"
require_file "$models_root/lightx2v/Qwen/Qwen-Image-Edit-2511/text_encoder/config.json"
require_file "$models_root/lightx2v/Qwen/Qwen-Image-Edit-2511/vae/diffusion_pytorch_model.safetensors"
require_file "$models_root/lightx2v/Qwen/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors"
require_file "$models_root/lightx2v/Qwen/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors"
require_file "$models_root/flux2/black-forest-labs/FLUX.2-klein-9b-fp8/flux-2-klein-9b-fp8.safetensors"
require_file "$models_root/flux2/black-forest-labs/FLUX.2-klein-9B/vae/diffusion_pytorch_model.safetensors"
require_file "$models_root/flux2/Qwen/Qwen3-8B-FP8/model.safetensors.index.json"

"$lightx2v_python" - <<'PY'
import importlib.metadata

expected = {
    "torch": "2.8.0",
    "transformers": "4.57.3",
    "diffusers": "0.39.0",
    "flash_attn": "2.8.3",
    "lightx2v": "0.1.0",
}
for distribution, version in expected.items():
    installed = importlib.metadata.version(distribution)
    if installed.split("+")[0] != version:
        raise SystemExit(f"{distribution} {version} is required, got {installed}")
PY

"$venv_python" - <<'PY'
import onnxruntime as ort

providers = ort.get_available_providers()
if "CUDAExecutionProvider" not in providers:
    raise SystemExit(f"onnxruntime CUDAExecutionProvider is missing: {providers}")
PY

run "$venv_python" -m aigen.cli --help
run "$venv_python" -m aigen.cli briefs schema --compact
run "$venv_python" -m aigen.cli briefs plan-schema --compact
run "$venv_python" -m aigen.cli characters view-schema --compact
run "$venv_python" -m aigen.cli characters view-bank-schema --compact
run "$venv_python" -m aigen.cli keyframes schema --compact
run "$venv_python" -m aigen.cli keyframes refine-schema --compact
run "$venv_python" -m aigen.cli keyframes polish-schema --compact
run "$venv_python" -m aigen.cli keyframes polish-plan-schema --compact
[[ -f "$repo_root/tools/diffusers/train_dreambooth_lora_flux.py" ]] || {
  echo "missing LoRA trainer; run scripts/download_lora_trainer.sh" >&2
  exit 1
}
grep -Fq 'check_min_version("0.38.0")' "$repo_root/tools/diffusers/train_dreambooth_lora_flux.py"
grep -Fq "move_training_module" "$repo_root/tools/diffusers/train_dreambooth_lora_flux.py"
grep -Fq "Skipping save-time transformer dtype cast for quantized local LoRA training." "$repo_root/tools/diffusers/train_dreambooth_lora_flux.py"
[[ -f "$repo_root/tools/diffusers/train_dreambooth_lora_flux2_klein.py" ]] || {
  echo "missing FLUX.2 Klein LoRA trainer; run scripts/download_lora_trainer.sh" >&2
  exit 1
}
grep -Fq 'check_min_version("0.40.0.dev0")' "$repo_root/tools/diffusers/train_dreambooth_lora_flux2_klein.py"
grep -Fq -- "--precomputed_cache_path" "$repo_root/tools/diffusers/train_dreambooth_lora_flux2_klein.py"
grep -Fq "Loaded precomputed training cache" "$repo_root/tools/diffusers/train_dreambooth_lora_flux2_klein.py"

log "install check passed"
