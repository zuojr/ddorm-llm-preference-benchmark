#!/usr/bin/env bash
set -euo pipefail

# 最小闭环：
# 1) SFT warm start
# 2) Reward model
# 3) DPO baseline
# 4) DDO-RM
# 5) Held-out eval on test_prefs
#
# 用法：
#   bash scripts/run_minimal_pythia410m_single4090.sh 42

cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export HF_HOME=/root/autodl-tmp/cache
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets

SEED=${1:-42}
BASE=EleutherAI/pythia-410m
ROOT=outputs/minimal_pythia410m/seed_${SEED}
mkdir -p "$ROOT" "$ROOT/metrics"

echo "[1/5] SFT warm start -> $ROOT/sft"
python -m benchmark.train_sft \
  --base_model "$BASE" \
  --output_dir "$ROOT/sft" \
  --seed "$SEED" \
  --use_lora \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1.0

echo "[2/5] Reward model -> $ROOT/rm"
python -m benchmark.train_reward \
  --base_model "$BASE" \
  --output_dir "$ROOT/rm" \
  --seed "$SEED" \
  --use_lora \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1.0

echo "[3/5] DPO baseline -> $ROOT/dpo"
python -m benchmark.train_dpo \
  --model_name_or_path "$ROOT/sft" \
  --output_dir "$ROOT/dpo" \
  --seed "$SEED" \
  --use_lora \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1.0

echo "[4/5] DDO-RM -> $ROOT/ddorm"
python -m benchmark.train_ddorm \
  --model_name_or_path "$ROOT/sft" \
  --reward_model_name_or_path "$ROOT/rm" \
  --output_dir "$ROOT/ddorm" \
  --seed "$SEED" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1.0

echo "[5/5] Evaluate on test_prefs"
python -m benchmark.eval_pairwise \
  --model_name_or_path "$ROOT/rm" \
  --model_type reward \
  --output_path "$ROOT/metrics/rm_test_prefs.json"

python -m benchmark.eval_pairwise \
  --model_name_or_path "$ROOT/dpo" \
  --model_type policy \
  --output_path "$ROOT/metrics/dpo_test_prefs.json"

python -m benchmark.eval_pairwise \
  --model_name_or_path "$ROOT/ddorm" \
  --model_type policy \
  --output_path "$ROOT/metrics/ddorm_test_prefs.json"

echo "Done. Metrics saved under: $ROOT/metrics"
ls -lah "$ROOT/metrics"
