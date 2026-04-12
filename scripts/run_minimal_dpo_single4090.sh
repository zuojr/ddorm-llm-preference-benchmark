#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export HF_HOME=/root/autodl-tmp/cache
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets

SEED=${1:-42}
BASE=EleutherAI/pythia-410m
ROOT=outputs/minimal_dpo_pythia410m/seed_${SEED}
mkdir -p "$ROOT" logs

echo "[1/4] Environment check"
python -m benchmark.check_env

echo "[2/4] SFT warm start"
python -m benchmark.train_sft \
  --base_model "$BASE" \
  --output_dir "$ROOT/sft" \
  --seed "$SEED" \
  --use_lora \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1.0

echo "[3/4] DPO baseline"
python -m benchmark.train_dpo \
  --model_name_or_path "$ROOT/sft" \
  --output_dir "$ROOT/dpo" \
  --seed "$SEED" \
  --use_lora

echo "[4/4] Evaluate DPO on test_prefs"
mkdir -p "$ROOT/metrics"
python -m benchmark.eval_pairwise \
  --model_name_or_path "$ROOT/dpo" \
  --model_type policy \
  --output_path "$ROOT/metrics/dpo_test_prefs.json"

echo "Done. Metrics at: $ROOT/metrics/dpo_test_prefs.json"
