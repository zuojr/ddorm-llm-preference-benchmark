#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME=/root/autodl-tmp/cache
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 可选：显式打开 TF32，4090 / A100 上通常更稳更快。
python - <<'PY'
import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print('TF32 enabled')
else:
    print('CUDA not available yet')
PY
