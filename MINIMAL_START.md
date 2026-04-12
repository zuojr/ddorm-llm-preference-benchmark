# 最小启动方案

如果你现在只想先确认 **AutoDL 环境 + 数据下载 + 训练脚本** 是通的，先跑下面这两个中的第一个：

## 方案 A：最小闭环（推荐先跑）

只跑：**SFT -> DPO -> eval**

```bash
bash scripts/install_autodl.sh
bash scripts/run_minimal_dpo_single4090.sh 42 | tee logs/minimal_dpo_seed42.log
```

输出重点看：

```bash
tail -f logs/minimal_dpo_seed42.log
cat outputs/minimal_dpo_pythia410m/seed_42/metrics/dpo_test_prefs.json
```

## 方案 B：最小方法闭环

只跑：**SFT -> Reward Model -> DDO-RM -> eval**

```bash
bash scripts/install_autodl.sh
bash scripts/run_minimal_ddorm_single4090.sh 42 | tee logs/minimal_ddorm_seed42.log
```

输出重点看：

```bash
tail -f logs/minimal_ddorm_seed42.log
cat outputs/minimal_ddorm_pythia410m/seed_42/metrics/rm_test_prefs.json
cat outputs/minimal_ddorm_pythia410m/seed_42/metrics/ddorm_test_prefs.json
```

## 成功标准

- `benchmark.check_env` 能看到 GPU 信息
- `outputs/.../sft/` 存在
- `outputs/.../dpo/` 或 `outputs/.../ddorm/` 存在
- `metrics/*.json` 里能看到：
  - `pair_accuracy`
  - `auc`
  - `mean_margin`

## 通过后再做什么

如果方案 A 跑通，就接着跑方案 B；
如果方案 B 也跑通，再上 `scripts/run_stage1_pythia410m_single4090.sh` 跑全套：RM + DPO + ORPO + KTO + DDO-RM。
