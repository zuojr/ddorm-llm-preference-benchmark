# Minimal first run on AutoDL

目标：先做一条最小、最稳、最容易排错的链路。

- 单卡：1 x 4090 24GB
- 单 seed：42
- 模型：EleutherAI/pythia-410m
- 数据：HuggingFaceH4/ultrafeedback_binarized
- 方法：SFT -> Reward Model -> DPO -> DDO-RM -> pairwise eval

## Files you actually need first

- `scripts/install_autodl.sh`
- `scripts/run_minimal_pythia410m_single4090.sh`
- `src/benchmark/train_sft.py`
- `src/benchmark/train_reward.py`
- `src/benchmark/train_dpo.py`
- `src/benchmark/train_ddorm.py`
- `src/benchmark/eval_pairwise.py`
- `src/benchmark/data.py`
- `src/benchmark/scoring.py`
- `src/benchmark/trainer_ddorm.py`
- `src/benchmark/ddorm.py`
- `src/benchmark/utils.py`

## Steps

### 1. Upload and unzip

```bash
cd /root/autodl-tmp
unzip autodl_riplm_llm_benchmark.zip
cd autodl_riplm_llm_benchmark
```

### 2. Install

```bash
bash scripts/install_autodl.sh
```

### 3. Start a persistent shell

```bash
screen -U
```

### 4. Run the minimal pipeline

```bash
bash scripts/run_minimal_pythia410m_single4090.sh 42 > logs_minimal_seed42.log 2>&1
```

### 5. Watch logs

```bash
tail -f logs_minimal_seed42.log
```

### 6. Read metrics

```bash
cat outputs/minimal_pythia410m/seed_42/metrics/rm_test_prefs.json
cat outputs/minimal_pythia410m/seed_42/metrics/dpo_test_prefs.json
cat outputs/minimal_pythia410m/seed_42/metrics/ddorm_test_prefs.json
```

## What to check first

- The scripts finish without CUDA OOM.
- `rm_test_prefs.json` has sane `pair_accuracy` and `auc`.
- `dpo_test_prefs.json` is not pathological.
- `ddorm_test_prefs.json` can at least beat or match the simplest baseline on a first pass.

## If you hit OOM

Lower one of these first:

- `--per_device_train_batch_size`
- `--max_length`

Increase this to keep the global batch roughly stable:

- `--gradient_accumulation_steps`
