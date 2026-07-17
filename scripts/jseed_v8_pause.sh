#!/usr/bin/env bash
# Pauzeer de v8-training netjes. Standaard: wachten tot de eerstvolgende
# checkpoint-wegschrijving (max ~18 min) en dan uitsluitend dít trainings-
# proces stoppen — er gaat dan geen enkele voltooide stap verloren.
# Met --nu: direct stoppen (verliest de stappen sinds het laatste checkpoint).
# Hervatten: scripts/jseed_v8_resume.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=runs/flux2_jseed_subject_lora_9b_nf4_v8
PATTERN="train_dreambooth_lora_flux2_klein.py.*flux2_jseed_subject_lora_9b_nf4_v8"

pids=$(pgrep -f "$PATTERN" || true)
if [ -z "$pids" ]; then
  echo "geen v8-trainingsproces actief"
  exit 0
fi

laatste() { ls "$RUN" 2>/dev/null | grep -oE 'checkpoint-[0-9]+$' | sort -t- -k2 -n | tail -1; }

if [ "${1:-}" != "--nu" ]; then
  start=$(laatste)
  echo "laatste checkpoint: ${start:-geen}; wacht op de volgende wegschrijving..."
  while [ "$(laatste)" = "$start" ]; do
    sleep 15
    if ! pgrep -f "$PATTERN" > /dev/null; then
      echo "trainingsproces is intussen zelf geëindigd"
      exit 0
    fi
  done
  sleep 20  # wegschrijving volledig laten afronden
  echo "checkpoint $(laatste) staat op schijf"
fi

pids=$(pgrep -f "$PATTERN" || true)
[ -z "$pids" ] && { echo "trainingsproces al weg"; exit 0; }
echo "stop PIDs: $pids"
kill $pids
echo "gepauzeerd. Hervatten: scripts/jseed_v8_resume.sh"
