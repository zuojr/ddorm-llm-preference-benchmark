# First minimal AutoDL run

目标：先把最短路径跑通，不一次性上 DPO / ORPO / KTO。

## 机器建议

- 1 x RTX 4090 24GB
- 8 vCPU+
- 32GB RAM+
- CUDA 12.x / Python 3.10

## 目录

```bash
cd /root/autodl-tmp
unzip autodl_riplm_llm_benchmark_minimal.zip
cd autodl_riplm_llm_benchmark
```

## 安装

```bash
bash scripts/install_autodl.sh
```

## 启动实验

```bash
screen -U
bash scripts/run_first_minimal_single4090.sh 42 > logs_first_seed42.log 2>&1
```

## 看日志

```bash
tail -f logs_first_seed42.log
```

## 看结果

```bash
cat outputs/first_minimal_pythia410m/seed_42/metrics/rm_test_prefs.json
cat outputs/first_minimal_pythia410m/seed_42/metrics/ddorm_test_prefs.json
```

## 结果文件位置

- SFT: `outputs/first_minimal_pythia410m/seed_42/sft`
- Reward model: `outputs/first_minimal_pythia410m/seed_42/rm`
- DDO-RM: `outputs/first_minimal_pythia410m/seed_42/ddorm`
- Metrics: `outputs/first_minimal_pythia410m/seed_42/metrics`

## 下一步

等这条线跑通，再跑：

1. `scripts/run_stage1_pythia410m_single4090.sh`（补 DPO/ORPO/KTO）
2. `scripts/run_stage1_pythia410m_3seeds.sh`（补 3 seeds）

## 常见问题

### 1. 显存不够

把下面三个训练脚本里的长度再降一点：

- `src/benchmark/train_sft.py`
- `src/benchmark/train_reward.py`
- `src/benchmark/train_ddorm.py`

优先把 `max_seq_length` / `max_length` 从 `768` 改到 `512`。

### 2. bf16 报错

如果你租的不是 4090 / A100，而是较老的卡，把训练脚本里的 `bf16=True` 改成 `fp16=True`。

### 3. 下载慢

HF 模型和数据都会缓存到：

- `/root/autodl-tmp/cache/transformers`
- `/root/autodl-tmp/cache/datasets`

不要放系统盘。
