#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export HF_HOME=/root/autodl-tmp/cache
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

SEED=${1:-42}
BASE=${BASE:-EleutherAI/pythia-410m}
ROOT=outputs/quickstart_ddorm_firstreal/seed_${SEED}
mkdir -p "$ROOT"

# 这是第一条真正有论文价值的最小对比线：
# SFT warm start -> reward model -> DPO baseline -> DDO-RM -> full held-out eval.
python -m benchmark.train_sft \
  --base_model "$BASE" \
  --output_dir "$ROOT/sft" \
  --split 'train_sft' \
  --seed "$SEED" \
  --use_lora \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1.0

python -m benchmark.train_reward \
  --base_model "$BASE" \
  --output_dir "$ROOT/rm" \
  --split 'train_prefs' \
  --seed "$SEED" \
  --use_lora \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1.0

python -m benchmark.train_dpo \
  --model_name_or_path "$ROOT/sft" \
  --output_dir "$ROOT/dpo" \
  --split 'train_prefs' \
  --seed "$SEED" \
  --use_lora \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1.0

python -m benchmark.train_ddorm \
  --model_name_or_path "$ROOT/sft" \
  --reward_model_name_or_path "$ROOT/rm" \
  --output_dir "$ROOT/ddorm" \
  --split 'train_prefs' \
  --seed "$SEED" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1.0

mkdir -p "$ROOT/metrics"
python -m benchmark.eval_pairwise \
  --model_name_or_path "$ROOT/rm" \
  --model_type reward \
  --split 'test_prefs' \
  --output_path "$ROOT/metrics/rm_test_prefs.json"

python -m benchmark.eval_pairwise \
  --model_name_or_path "$ROOT/dpo" \
  --model_type policy \
  --split 'test_prefs' \
  --output_path "$ROOT/metrics/dpo_test_prefs.json"

python -m benchmark.eval_pairwise \
  --model_name_or_path "$ROOT/ddorm" \
  --model_type policy \
  --split 'test_prefs' \
  --output_path "$ROOT/metrics/ddorm_test_prefs.json"

echo "Done. Metrics are under $ROOT/metrics"
