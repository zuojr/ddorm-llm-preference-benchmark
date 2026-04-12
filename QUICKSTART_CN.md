# 最简单起步：AutoDL 上先跑什么

你现在最适合先跑两步，不要一上来就跑全方法全 seeds。

## 路线 A：只验证环境，最简单

目标：确认下面这些都没有问题：

- CUDA / PyTorch
- HuggingFace 下载权限
- TRL trainer
- UltraFeedback Binarized 数据切片
- pairwise 评测脚本

执行：

```bash
cd /root/autodl-tmp
unzip autodl_riplm_llm_benchmark.zip
cd autodl_riplm_llm_benchmark
bash scripts/install_autodl.sh
bash scripts/run_quickstart_dpo_smoke_single4090.sh 42 > logs_quickstart_dpo_smoke.log 2>&1
```

看日志：

```bash
tail -f logs_quickstart_dpo_smoke.log
```

输出文件：

```text
outputs/quickstart_dpo_smoke/seed_42/
├── sft/
├── dpo/
└── metrics/
    └── dpo_test_prefs_256.json
```

这一步只用了很小的数据切片：

- `train_sft[:2048]`
- `train_prefs[:4096]`
- `test_prefs[:256]`

所以它的作用是 **验证流程**，不是给论文出结果。

## 路线 B：第一条真实结果线

目标：最小但有意义的 critic-facing 对比。

- base model: `EleutherAI/pythia-410m`
- dataset: `HuggingFaceH4/ultrafeedback_binarized`
- methods: `RM + DPO + DDO-RM`
- eval: `test_prefs` 全量
- seed: 先只跑 `42`

执行：

```bash
bash scripts/run_quickstart_ddorm_firstreal_single4090.sh 42 > logs_quickstart_ddorm_firstreal.log 2>&1
```

看日志：

```bash
tail -f logs_quickstart_ddorm_firstreal.log
```

输出文件：

```text
outputs/quickstart_ddorm_firstreal/seed_42/
├── sft/
├── rm/
├── dpo/
├── ddorm/
└── metrics/
    ├── rm_test_prefs.json
    ├── dpo_test_prefs.json
    └── ddorm_test_prefs.json
```

## 我建议你实际按这个顺序做

1. 先跑 `run_quickstart_dpo_smoke_single4090.sh`
2. 确认日志里没有 tokenizer / dtype / OOM / dataset schema 错误
3. 再跑 `run_quickstart_ddorm_firstreal_single4090.sh`
4. 真实结果跑通以后，再扩到 3 seeds 和 ORPO/KTO

## 常见问题

### 1. 为什么不是一上来跑全家桶？

因为你现在最需要的是先把 **公开模型 + 公开数据 + official trainer + held-out eval** 这条链路跑通。先确保结果可信，再扩方法。

### 2. 为什么 base 用 Pythia-410M？

因为它足够小，适合 4090 单卡先起步；而且比 `TinyLlama-1.1B-Chat-v1.0` 更干净，不容易引入 UltraFeedback 对齐污染。

### 3. 为什么先只跑 DPO 和 DDO-RM？

因为这是最短的 critic-facing 对比：

- DPO：标准 pairwise baseline
- DDO-RM：你的方法

ORPO/KTO 可以等这条线稳定后再加。
