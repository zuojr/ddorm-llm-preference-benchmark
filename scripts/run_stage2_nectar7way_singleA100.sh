#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export HF_HOME=/root/autodl-tmp/cache
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
SEED=${1:-42}
BASE=${2:-EleutherAI/pythia-410m}
ROOT=outputs/stage2_nectar7way/seed_${SEED}
mkdir -p "$ROOT/data" "$ROOT/metrics"

python -m benchmark.data \
  --dataset nectar \
  --split train \
  --shuffle_seed "$SEED" \
  --start 0 \
  --max_examples 30000 \
  --output_dir "$ROOT/data/train_nectar7"

python -m benchmark.data \
  --dataset nectar \
  --split train \
  --shuffle_seed "$SEED" \
  --start 30000 \
  --max_examples 3000 \
  --output_dir "$ROOT/data/eval_nectar7"

python -m benchmark.train_sft \
  --base_model "$BASE" \
  --output_dir "$ROOT/sft" \
  --seed "$SEED" \
  --use_lora \
  --num_train_epochs 1.0

python -m benchmark.train_ddorm \
  --model_name_or_path "$ROOT/sft" \
  --dataset_path "$ROOT/data/train_nectar7" \
  --use_gold_rewards \
  --output_dir "$ROOT/ddorm_oracle" \
  --seed "$SEED"

python -m benchmark.eval_listwise \
  --model_name_or_path "$ROOT/ddorm_oracle" \
  --dataset_path "$ROOT/data/eval_nectar7" \
  --output_path "$ROOT/metrics/ddorm_oracle_listwise.json"
