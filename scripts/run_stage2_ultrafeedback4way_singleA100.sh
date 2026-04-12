#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export HF_HOME=/root/autodl-tmp/cache
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
SEED=${1:-42}
BASE=${2:-EleutherAI/pythia-410m}
ROOT=outputs/stage2_uf4way/seed_${SEED}
mkdir -p "$ROOT/data" "$ROOT/metrics"

python -m benchmark.data \
  --dataset uf_listwise \
  --split train \
  --score_mode mean_rating \
  --shuffle_seed "$SEED" \
  --start 0 \
  --max_examples 20000 \
  --output_dir "$ROOT/data/train_uf4_mean"

python -m benchmark.data \
  --dataset uf_listwise \
  --split train \
  --score_mode mean_rating \
  --shuffle_seed "$SEED" \
  --start 20000 \
  --max_examples 2000 \
  --output_dir "$ROOT/data/eval_uf4_mean"

python -m benchmark.train_sft \
  --base_model "$BASE" \
  --output_dir "$ROOT/sft" \
  --seed "$SEED" \
  --use_lora \
  --num_train_epochs 1.0

# Oracle / score-supervised DDO：直接用原始四候选打分做目标分布。
python -m benchmark.train_ddorm \
  --model_name_or_path "$ROOT/sft" \
  --dataset_path "$ROOT/data/train_uf4_mean" \
  --use_gold_rewards \
  --output_dir "$ROOT/ddorm_oracle" \
  --seed "$SEED"

python -m benchmark.eval_listwise \
  --model_name_or_path "$ROOT/ddorm_oracle" \
  --dataset_path "$ROOT/data/eval_uf4_mean" \
  --output_path "$ROOT/metrics/ddorm_oracle_listwise.json"
