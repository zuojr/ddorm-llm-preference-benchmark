#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export HF_HOME=/root/autodl-tmp/cache
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
SEED=${1:-42}
BASE=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
ROOT=outputs/stage1_tinyllama1b/seed_${SEED}
mkdir -p "$ROOT"

# AutoDL 官方建议：如果单机多卡足够，优先单机多卡，不要多机。
# 这里默认 2x4090 或 1xA100-40G 以上配置。

accelerate launch --num_processes 2 -m benchmark.train_sft \
  --base_model "$BASE" \
  --output_dir "$ROOT/sft" \
  --seed "$SEED" \
  --use_lora \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1.0

accelerate launch --num_processes 2 -m benchmark.train_reward \
  --base_model "$BASE" \
  --output_dir "$ROOT/rm" \
  --seed "$SEED" \
  --use_lora \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1.0

accelerate launch --num_processes 2 -m benchmark.train_dpo \
  --model_name_or_path "$ROOT/sft" \
  --output_dir "$ROOT/dpo" \
  --seed "$SEED" \
  --use_lora

accelerate launch --num_processes 2 -m benchmark.train_orpo \
  --model_name_or_path "$ROOT/sft" \
  --output_dir "$ROOT/orpo" \
  --seed "$SEED" \
  --use_lora

accelerate launch --num_processes 2 -m benchmark.train_kto \
  --model_name_or_path "$ROOT/sft" \
  --output_dir "$ROOT/kto" \
  --seed "$SEED" \
  --use_lora

accelerate launch --num_processes 2 -m benchmark.train_ddorm \
  --model_name_or_path "$ROOT/sft" \
  --reward_model_name_or_path "$ROOT/rm" \
  --output_dir "$ROOT/ddorm" \
  --seed "$SEED"

mkdir -p "$ROOT/metrics"
python -m benchmark.eval_pairwise --model_name_or_path "$ROOT/dpo"   --model_type policy --output_path "$ROOT/metrics/dpo_test_prefs.json"
python -m benchmark.eval_pairwise --model_name_or_path "$ROOT/orpo"  --model_type policy --output_path "$ROOT/metrics/orpo_test_prefs.json"
python -m benchmark.eval_pairwise --model_name_or_path "$ROOT/kto"   --model_type policy --output_path "$ROOT/metrics/kto_test_prefs.json"
python -m benchmark.eval_pairwise --model_name_or_path "$ROOT/ddorm" --model_type policy --output_path "$ROOT/metrics/ddorm_test_prefs.json"
