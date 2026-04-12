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
ROOT=outputs/quickstart_dpo_smoke/seed_${SEED}
mkdir -p "$ROOT"

# 先用很小的 slice 验证：环境、下载、tokenizer、trainer、评测 全部走通。
python -m benchmark.train_sft \
  --base_model "$BASE" \
  --output_dir "$ROOT/sft" \
  --split 'train_sft[:2048]' \
  --seed "$SEED" \
  --use_lora \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1.0

python -m benchmark.train_dpo \
  --model_name_or_path "$ROOT/sft" \
  --output_dir "$ROOT/dpo" \
  --split 'train_prefs[:4096]' \
  --seed "$SEED" \
  --use_lora \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1.0

mkdir -p "$ROOT/metrics"
python -m benchmark.eval_pairwise \
  --model_name_or_path "$ROOT/dpo" \
  --model_type policy \
  --split 'test_prefs[:256]' \
  --output_path "$ROOT/metrics/dpo_test_prefs_256.json"

echo "Done. Metrics saved to $ROOT/metrics/dpo_test_prefs_256.json"
