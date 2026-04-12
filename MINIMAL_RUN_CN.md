# 最小实验：单卡 4090 / 单 seed / 全量 held-out

目标：先跑一个 **最简单但仍然 reviewer-safe** 的闭环。

只跑四个环节：

1. `SFT`：给 base model 一个统一 warm start
2. `RM`：训练 reward model
3. `DPO`：作为最标准的 pairwise baseline
4. `DDO-RM`：你的方法
5. `Eval`：在 `test_prefs` 全量上统一评测

## 为什么这个版本是最合适的第一步

你的批评文本里说得很清楚：
- 4 个 held-out pairs + 3 seeds + binary accuracy 不是可发表证据
- 下一步应该上 **public checkpoint + public preference dataset + official trainer + full held-out split** 的实验fileciteturn2file0

而你现在论文正文也已经把 claim 收窄到了：
- 重点是 formulation / algorithmic structure / theory
- 不宣称在没有完整实验的情况下经验上压过 SPO+ 或一般 KKT layerfileciteturn2file1

所以最小版实验的目的不是“证明统治所有 baseline”，而是先把 critic 最容易攻击的点堵上。

## 推荐机型

- **1 × RTX 4090 24G**
- CPU 至少 8 vCPU
- 内存至少 32GB
- 磁盘建议 100GB 以上

## 目录里你实际会用到的文件

### 环境安装
- `scripts/install_autodl.sh`

### 训练与评测主脚本
- `scripts/run_minimal_pythia410m_single4090.sh`

### Python 源码
- `src/benchmark/data.py`
- `src/benchmark/train_sft.py`
- `src/benchmark/train_reward.py`
- `src/benchmark/train_dpo.py`
- `src/benchmark/train_ddorm.py`
- `src/benchmark/eval_pairwise.py`
- `src/benchmark/trainer_ddorm.py`
- `src/benchmark/ddorm.py`
- `src/benchmark/scoring.py`
- `src/benchmark/utils.py`

## 上机步骤

### 1. 上传 ZIP 到 AutoDL 并解压

假设你把项目上传到了：

```bash
/root/autodl-tmp/autodl_riplm_llm_benchmark.zip
```

解压：

```bash
cd /root/autodl-tmp
unzip autodl_riplm_llm_benchmark.zip
cd autodl_riplm_llm_benchmark
```

### 2. 安装依赖

```bash
bash scripts/install_autodl.sh
```

### 3. 建议先开一个 screen

```bash
screen -U
```

### 4. 直接跑最小实验

```bash
bash scripts/run_minimal_pythia410m_single4090.sh 42 > logs_minimal_seed42.log 2>&1
```

### 5. 查看日志

```bash
tail -f logs_minimal_seed42.log
```

### 6. 看结果文件

```bash
find outputs/minimal_pythia410m/seed_42/metrics -maxdepth 1 -type f | sort
cat outputs/minimal_pythia410m/seed_42/metrics/dpo_test_prefs.json
cat outputs/minimal_pythia410m/seed_42/metrics/ddorm_test_prefs.json
cat outputs/minimal_pythia410m/seed_42/metrics/rm_test_prefs.json
```

## 预期输出

你最终至少会得到三个 json：

- `rm_test_prefs.json`
- `dpo_test_prefs.json`
- `ddorm_test_prefs.json`

每个文件里都有：

- `pair_accuracy`
- `auc`
- `mean_margin`
- `std_margin`
- `num_examples`

## 这一步跑通以后再做什么

下一步不要立刻上很多方法。
更稳的顺序是：

1. 先把这个单 seed 跑通
2. 再补 `seed=13,42,3407`
3. 再加 `ORPO`
4. 再加 `KTO`
5. 最后才做 4-way / Nectar 这种 listwise 实验

## 常见报错与处理

### 1. 显存不够

优先把下面两个值调小：

- `--per_device_train_batch_size`
- `--max_length`

对应修改文件：
- `scripts/run_minimal_pythia410m_single4090.sh`

建议先尝试：

- SFT / RM：batch size 从 `4` 改成 `2`
- DPO / DDO-RM：保持 `2`
- 如果还爆显存：把 `max_length` 从 `1024` 改成 `768`

### 2. 下载慢

多半是 HF 下载问题。确认缓存路径已经设到数据盘：

```bash
echo $HF_HOME
echo $TRANSFORMERS_CACHE
echo $HF_DATASETS_CACHE
```

### 3. 输出目录里只有 adapter 没有 tokenizer

正常情况下脚本已经会保存 tokenizer；如果中途打断，重新从该步骤再跑一次即可。

## 你拿到结果以后怎么判断值不值得继续

先不要盯着“谁赢了谁”。
先看三件事：

1. `num_examples` 是否等于 `test_prefs` 全量
2. `DPO` 是否回到了正常可解释的 accuracy 区间，而不是接近 0
3. `DDO-RM` 的 margin 和 accuracy 是否稳定，不是 toy 那种噪声值

只要这三点成立，这一轮就是成功的，因为它已经比 toy smoke test 更接近真正 critic-facing 的证据链。fileciteturn2file0
