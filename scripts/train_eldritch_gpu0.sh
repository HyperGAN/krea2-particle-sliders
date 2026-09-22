#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
exec /ml2/ntc-image-studio/.venv-anima/bin/python -u scripts/train_krea2.py \
  --live --device 0 \
  --name eldritch-krea2-bbox \
  --model_id jimmycarter/krea2-turbo-bbox \
  --transformer_subfolder epoch-14-step-73184/transformer \
  --revision ec7aa643a4da7e56a08c1778e03e837a2e57a94b \
  --skeleton_model krea/Krea-2-Raw \
  --skeleton_revision 6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603 \
  --prompts_file configs/krea2/prompts-eldritch.yaml \
  --config_file configs/krea2/config-eldritch.yaml \
  --save_dir outputs/eldritch-krea2-bbox \
  --rank 16 --resolution 512 --steps 400 --lr 0.00005 \
  --hold_weight 0.1 --save_every 50 --cache_seeds 2 \
  --sample_steps 8 --sample_guidance 0 --mu 1.15 \
  --sample_resolution 768 --final_resolution 1536 --seed 7 "$@"
