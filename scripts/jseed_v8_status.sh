#!/usr/bin/env bash
# Stand van zaken van de JSEED v8-LoRA-training, opvraagbaar buiten de trainingssessie.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=runs/flux2_jseed_subject_lora_9b_nf4_v8-launch.log
RUN=runs/flux2_jseed_subject_lora_9b_nf4_v8

if pgrep -f "train_dreambooth_lora_flux2_klein.py" > /dev/null; then
  echo "status : TRAINING LOOPT"
else
  echo "status : geen trainingsproces actief"
fi

echo "stap   : $(tr '\r' '\n' < "$LOG" | grep -oE '\| [0-9]+/3000 \[[^]]*\]' | tail -1)"
echo "checkpoints:"
ls "$RUN" | grep checkpoint | sort -t- -k2 -n | tail -4 | sed 's/^/  /'

err=$(grep -m1 -E "Traceback|CUDA out of memory|RuntimeError" "$LOG" || true)
if [ -n "$err" ]; then
  echo "FOUT in log: $err"
else
  echo "fouten : geen in de log"
fi

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/gpu    : /'
