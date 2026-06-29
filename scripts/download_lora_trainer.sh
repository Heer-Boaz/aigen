#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

trainer_url="https://raw.githubusercontent.com/huggingface/diffusers/v0.38.0/examples/dreambooth/train_dreambooth_lora_flux.py"
trainer_path="$repo_root/tools/diffusers/train_dreambooth_lora_flux.py"
expected_check='check_min_version("0.38.0")'

usage() {
  cat <<'EOF'
Usage: scripts/download_lora_trainer.sh

Downloads the pinned official Diffusers 0.38 FLUX DreamBooth-LoRA trainer used
by the local 16GB LoRA training profile.
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

mkdir -p "$(dirname -- "$trainer_path")"
run curl -L --fail "$trainer_url" -o "$trainer_path"
grep -Fq "$expected_check" "$trainer_path" || die "downloaded trainer is not the pinned Diffusers 0.38 trainer"
"$venv_python" - "$trainer_path" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
needle = (
    "    vae.to(accelerator.device, dtype=weight_dtype)\n"
    "    transformer.to(accelerator.device, dtype=weight_dtype)\n"
    "    text_encoder_one.to(accelerator.device, dtype=weight_dtype)\n"
    "    text_encoder_two.to(accelerator.device, dtype=weight_dtype)\n"
)
replacement = (
    "    def move_training_module(module, name):\n"
    '        if getattr(module, "is_quantized", False):\n'
    '            logger.info(f"Skipping {name} dtype cast for quantized local LoRA training.")\n'
    "            return\n"
    "        module.to(accelerator.device, dtype=weight_dtype)\n"
    "\n"
    '    move_training_module(vae, "vae")\n'
    '    move_training_module(transformer, "transformer")\n'
    '    move_training_module(text_encoder_one, "text_encoder_one")\n'
    '    move_training_module(text_encoder_two, "text_encoder_two")\n'
)
if replacement not in source:
    if needle not in source:
        raise SystemExit("trainer patch target not found")
    source = source.replace(needle, replacement, 1)
path.write_text(source, encoding="utf-8")
PY
"$venv_python" - "$trainer_path" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
needle = (
    "        if args.upcast_before_saving:\n"
    "            transformer.to(torch.float32)\n"
    "        else:\n"
    "            transformer = transformer.to(weight_dtype)\n"
)
replacement = (
    "        if getattr(transformer, \"is_quantized\", False):\n"
    "            logger.info(\"Skipping save-time transformer dtype cast for quantized local LoRA training.\")\n"
    "        elif args.upcast_before_saving:\n"
    "            transformer.to(torch.float32)\n"
    "        else:\n"
    "            transformer = transformer.to(weight_dtype)\n"
)
if replacement not in source:
    if needle not in source:
        raise SystemExit("trainer save patch target not found")
    source = source.replace(needle, replacement, 1)
path.write_text(source, encoding="utf-8")
PY
grep -Fq "move_training_module" "$trainer_path"
grep -Fq "Skipping save-time transformer dtype cast for quantized local LoRA training." "$trainer_path"
chmod +x "$trainer_path"
log "LoRA trainer downloaded: $trainer_path"
