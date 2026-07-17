#!/usr/bin/env bash
# Hervat de v8-training vanaf het laatste checkpoint. Veilig om te draaien:
# weigert als er al een v8-proces loopt; bij bereikte 3000 stappen laadt en
# eindigt de trainer vanzelf.
set -euo pipefail
cd "$(dirname "$0")/.."

PATTERN="train_dreambooth_lora_flux2_klein.py.*flux2_jseed_subject_lora_9b_nf4_v8"
if pgrep -f "$PATTERN" > /dev/null; then
  echo "v8-training loopt al; niets gedaan"
  exit 1
fi

nohup .venv/bin/accelerate launch tools/diffusers/train_dreambooth_lora_flux2_klein.py \
  --pretrained_model_name_or_path aigen/models/flux2/black-forest-labs/FLUX.2-klein-base-9B-training \
  --pretrained_text_encoder_name_or_path aigen/models/flux2/Qwen/Qwen3-8B-FP8 \
  --dataset_name assets/lora/JSEED/dataset-v15 --caption_column prompt \
  --instance_prompt "JSEED" \
  --output_dir runs/flux2_jseed_subject_lora_9b_nf4_v8 \
  --bnb_quantization_config_path runs/jillian_subject_lora_dataset_clean_v1/nf4-bf16.json \
  --cache_latents \
  --precomputed_cache_path assets/lora/JSEED/dataset-v15-cache.pt \
  --aspect_ratio_buckets "832,1248;1184,880;1248,832" \
  --text_encoder_out_layers 9 18 27 \
  --center_crop --rank 16 --lora_alpha 16 --train_batch_size 1 --gradient_accumulation_steps 1 \
  --gradient_checkpointing --mixed_precision bf16 \
  --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 100 \
  --optimizer adamW --use_8bit_adam --max_sequence_length 512 \
  --max_train_steps 3000 --checkpointing_steps 250 --checkpoints_total_limit 12 \
  --offload --seed 42 --skip_final_inference --report_to tensorboard \
  --resume_from_checkpoint latest \
  >> runs/flux2_jseed_subject_lora_9b_nf4_v8-launch.log 2>&1 &

echo "hervat (PID $!); volg met scripts/jseed_v8_status.sh"
