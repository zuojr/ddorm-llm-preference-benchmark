#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

for seed in 42 13 3407; do
  bash scripts/run_stage1_pythia410m_single4090.sh "$seed" > "logs/stage1_pythia410m_seed${seed}.log" 2>&1
  tail -n 20 "logs/stage1_pythia410m_seed${seed}.log"
done
