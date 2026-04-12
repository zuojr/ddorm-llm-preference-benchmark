#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export HF_HOME=/root/autodl-tmp/cache
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets

SEED=${1:-42}
BASE=EleutherAI/pythia-410m
ROOT=outputs/minimal_ddorm_pythia410m/seed_${SEED}
mkdir -p "$ROOT" logs

echo "[1/5] Environment check"
python -m benchmark.check_env

echo "[2/5] SFT warm start"
python -m benchmark.train_sft \
  --base_model "$BASE" \
  --output_dir "$ROOT/sft" \
  --seed "$SEED" \
  --use_lora \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1.0

echo "[3/5] Reward model"
python -m benchmark.train_reward \
  --base_model "$BASE" \
  --output_dir "$ROOT/rm" \
  --seed "$SEED" \
  --use_lora \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1.0

echo "[4/5] DDO-RM"
python -m benchmark.train_ddorm \
  --model_name_or_path "$ROOT/sft" \
  --reward_model_name_or_path "$ROOT/rm" \
  --output_dir "$ROOT/ddorm" \
  --seed "$SEED"

echo "[5/5] Evaluate reward model + DDO-RM on test_prefs"
mkdir -p "$ROOT/metrics"
python -m benchmark.eval_pairwise \
  --model_name_or_path "$ROOT/rm" \
  --model_type reward \
  --output_path "$ROOT/metrics/rm_test_prefs.json"
python -m benchmark.eval_pairwise \
  --model_name_or_path "$ROOT/ddorm" \
  --model_type policy \
  --output_path "$ROOT/metrics/ddorm_test_prefs.json"

echo "Done. Metrics at: $ROOT/metrics/"
