# KuaiRec 48GB 全量优化实验计划书

## 1. 实验目标

本计划面向一台约 10 小时租用窗口的 RTX 4090-48G 机器，目标是在当前 `full_24gb` baseline 基础上，尽量提升近全曝光评测指标，同时保持实验协议干净。

当前冻结 `full_test` baseline 为：

```text
rankmix_lambdarank_din_mmr
```

当前指标：

| 指标 | 当前 full_test |
|---|---:|
| Recall@10 | 0.009218 |
| NDCG@10 | 0.876574 |
| Utility@10 | 1.053530 |
| Recall@200 | 0.139656 |
| NDCG@200 | 0.706562 |
| Coverage@200 | 0.524497 |
| Cold Recall@200 | 0.237664 |

本轮优化目标：

| 指标 | 目标区间 |
|---|---:|
| Recall@200 | 0.142+ |
| NDCG@200 | 0.710+ |
| NDCG@10 | 0.880 左右 |
| Coverage@200 | 0.530+ |

核心约束：

- 只在 `full_val` 上做参数选择。
- `full_test` 只用于最后一次冻结验收。
- 如果 10 小时内时间不够，优先保证候选池扩容、LambdaRank/DIN 主排序、RankMix 权重搜索。
- 不追求所有模型都跑完，优先跑最可能提升最终指标的实验。

## 2. 推荐机器

推荐使用：

```text
RTX 4090-48G / 48GB
14 核 CPU
90GB 内存
```

不优先选择 RTX 5090 / 32GB 的原因：

- 本项目的瓶颈更偏向显存、内存、候选缓存和批量训练稳定性，而不是单纯 FP 算力。
- 48GB 显存更适合候选池扩容、DIN 长序列和大 batch 精排训练。
- 90GB 内存比 60GB 更适合候选特征构造、LightGBM 和 parquet 缓存。
- 5090 新卡环境可能存在 CUDA / PyTorch 兼容性风险，10 小时租用窗口里不值得冒这个风险。

## 3. 新 profile 配置

本项目已新增：

```text
full_48gb_optimized
```

配置文件：

```text
configs/profiles.yaml
```

关键参数：

| 参数 | `full_24gb` | `full_48gb_optimized` | 目的 |
|---|---:|---:|---|
| `recall_k` | 200 | 500 | 扩大候选层评测和推荐截断 |
| `itemcf_history_length` | 100 | 100 | 保持 ItemCF baseline 不变 |
| `itemcf_neighbors` | 400 | 400 | 保持 ItemCF baseline 不变 |
| `two_tower_epochs` | 10 | 10 | 保持已验证的 R3.4 双塔配置 |
| `two_tower_batch_size` | 16384 | 16384 | 避免重复烧时间做低收益双塔增强 |
| `two_tower_embedding_dim` | 128 | 128 | 保持已验证的向量维度 |
| `lightgbm_estimators` | 1200 | 1600 | 增强 LambdaRank 排序 |
| `deep_ranking_epochs` | 5 | 8 | 增强 DIN 训练 |
| `deep_ranking_batch_size` | 16384 | 32768 | 提高 GPU 吞吐 |
| `deep_ranking_embedding_dim` | 32 | 64 | 增强 DIN 表达能力 |
| `din_history_length` | 100 | 150 | 捕获更长用户兴趣历史 |

注意：

`top_per_channel` 和 `max_candidates_per_group` 已写入 `full_48gb_optimized`，`run_rank_mix.py` 与 `run_candidate_ranking.py` 会默认读取 profile。命令行参数只在临时覆盖时使用。

## 4. 10 小时排期

| 时间 | 阶段 | 目标 | 产物 |
|---|---|---|---|
| 0 - 0.5h | 环境检查 | 解压代码、数据、安装依赖、跑测试 | 确认 GPU / CUDA / 测试通过 |
| 0.5 - 1.5h | 准备数据 | 生成 `full_48gb_optimized` processed 数据 | `data/processed/full_48gb_optimized` |
| 1.5 - 2.5h | 召回复用与扩候选检查 | 跑三路召回 full_val，保留 R3.4 双塔为互补候选源 | recall 结果 |
| 2.5 - 4.5h | 主排序实验 | 跑扩候选 RankMix，训练 LambdaRank + DIN | rank_mix full_val |
| 4.5 - 6.5h | RankMix 搜索 | 搜 DIN/LambdaRank 权重和 RRF k | full_val 对比表 |
| 6.5 - 7.5h | MMR 搜索 | 只搜少量 MMR lambda | full_val 对比表 |
| 7.5 - 8.5h | 选型与显著性 | 选择 full_val 最优方案，做 paired bootstrap | selection note |
| 8.5 - 10h | 冻结验收 | 只跑最终方案 full_test | frozen full_test 结果 |

如果中途训练慢于预期，优先保留候选扩容、基础 DIN/LambdaRank 主排序和 RankMix 搜索。不要再跑 R3.5/R3.6 双塔增强、多行为 DIN、OOVAwareDIN、BPR/ListNet 等已验证收益不稳的实验。

## 4.1 已删除的低收益实验

这些实验已经在此前报告中验证过，不再进入本轮 10 小时排期：

| 方向 | 已有证据 | 本轮决策 |
|---|---|---|
| 双塔 hard negative `R3.5/R3.5R` | 单路 Recall/NDCG 有小幅变化，但 Cold Recall 下降；增量 Oracle Recall 不优于组合版 | 不再单独训练 |
| 双塔 ID Dropout + hard negative `R3.6/R3.6R` | 单路不是最强；固定 5%/10% 双塔配额会显著拉低 Recall | 只保留已验证 R3.4/R3.6 思路作为候选源，不做增强搜索 |
| 固定配额融合双塔 | 5% 双塔配额 Recall 显著下降 1.96%，10% 配额显著下降 4.33% | 不做固定配额召回融合 |
| DIN BPR / Hybrid / ListNet | 没有超过 BCE，ListNet 与 BCE 近似，BPR/Hybrid 下降 | 不再搜索排序损失 |
| 多行为 DIN | Recall/NDCG 均值略升但不显著 | 不进入 48GB 主排期 |
| 作者/内容/时间增强 DIN | Recall 和 NDCG 没有稳定收益 | 不进入 48GB 主排期 |
| OOV 内容表示 + ID Dropout | Recall 可升但 NDCG 显著下降，Candidate OOV Recall 没改善 | 不进入 48GB 主排期 |

本轮真正保留的优化点：

1. 扩大候选池：`top_per_channel=300`、`max_candidates_per_group=500`。
2. 加强双塔与候选级主排序：提升双塔训练吞吐、LambdaRank estimators、基础 DIN capacity、训练 batch 和轻量高阶交叉特征。
3. 重搜 RankMix 权重：让 DIN / LambdaRank 在新候选池上重新分配贡献。
4. 轻量搜索 MMR lambda：只做最终展示多样性的微调。

新增交叉特征见：

```text
docs/FEATURE_ENGINEERING_GROUPS.md
```

本轮不直接引入 DCN，而是先把用户侧、物品侧、召回侧的组合关系显式构造成特征，让 LambdaRank 和 DIN 在现有链路内吸收高阶交叉信号。

## 5. 实验执行命令

以下命令假设已经在项目根目录：

```bash
cd /root/autodl-tmp/kuairec
```

### 5.1 环境检查

```bash
python -m pytest -q
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

期望：

```text
38 passed
True
RTX 4090
```

### 5.2 数据准备

```bash
python scripts/run_experiment.py --profile full_48gb_optimized --stage prepare
```

产物：

```text
data/processed/full_48gb_optimized/summary.json
```

### 5.3 三路召回 full_val

```bash
python scripts/run_experiment.py --profile full_48gb_optimized --stage recall --experiment-id itemcf_main --panel full_val
python scripts/run_experiment.py --profile full_48gb_optimized --stage recall --experiment-id content_text_category --panel full_val
python scripts/run_experiment.py --profile full_48gb_optimized --stage recall --experiment-id feature_tower_id_dropout --panel full_val
```

观察指标：

- Recall@500
- NDCG@500
- Coverage@500
- Cold Recall@500

说明：

`full_48gb_optimized` 的 `recall_k=500`，所以召回阶段会以更大的 K 评估候选上限。召回指标变动不代表最终排序一定提升，但如果 Recall@500 明显提升，说明后续精排有更大的候选空间。

不要在本轮继续运行 `R3.5`、`R3.6`、`R3.5R`、`R3.6R`。已有报告显示它们适合作为“双塔互补性”论据，但不适合作为本次提指标的主要投入方向。

### 5.4 主 RankMix full_val

先跑一组主实验：

```bash
python scripts/run_rank_mix.py \
  --profile full_48gb_optimized \
  --panel full_val \
  --cpu-threads 14 \
  --din-weight 0.6 \
  --pr3-weight 0.4 \
  --rrf-k 60 \
  --mmr-lambda 0.9
```

产物：

```text
artifacts/full_48gb_optimized/rank_mix/summary_full_val.csv
artifacts/full_48gb_optimized/rank_mix/results_full_val.json
```

### 5.5 RankMix 权重搜索

在主实验完成后，只跑少量高价值组合：

```bash
python scripts/run_rank_mix.py --profile full_48gb_optimized --panel full_val --cpu-threads 14 --din-weight 0.5 --pr3-weight 0.5 --rrf-k 60 --mmr-lambda 0.9

python scripts/run_rank_mix.py --profile full_48gb_optimized --panel full_val --cpu-threads 14 --din-weight 0.7 --pr3-weight 0.3 --rrf-k 60 --mmr-lambda 0.9

python scripts/run_rank_mix.py --profile full_48gb_optimized --panel full_val --cpu-threads 14 --din-weight 0.6 --pr3-weight 0.4 --rrf-k 30 --mmr-lambda 0.9

python scripts/run_rank_mix.py --profile full_48gb_optimized --panel full_val --cpu-threads 14 --din-weight 0.6 --pr3-weight 0.4 --rrf-k 100 --mmr-lambda 0.9
```

选择标准：

1. 主看 `NDCG@10`。
2. 次看 `Recall@500` / `NDCG@500`。
3. 再看 `Utility@10` / `Utility@500`。
4. 如果差异很小，选择更稳定、更容易解释的 `0.6/0.4, rrf_k=60`。

### 5.6 MMR 搜索

只搜索 3 组：

```bash
python scripts/run_rank_mix.py --profile full_48gb_optimized --panel full_val --cpu-threads 14 --din-weight 0.6 --pr3-weight 0.4 --rrf-k 60 --mmr-lambda 0.85

python scripts/run_rank_mix.py --profile full_48gb_optimized --panel full_val --cpu-threads 14 --din-weight 0.6 --pr3-weight 0.4 --rrf-k 60 --mmr-lambda 0.90

python scripts/run_rank_mix.py --profile full_48gb_optimized --panel full_val --cpu-threads 14 --din-weight 0.6 --pr3-weight 0.4 --rrf-k 60 --mmr-lambda 0.95
```

如果 RankMix 搜索已经选出非默认权重，则把上述命令中的 `--din-weight`、`--pr3-weight`、`--rrf-k` 替换成最佳值。

选择标准：

- `NDCG@10` 不能明显下降。
- `Recall@500` 和 `NDCG@500` 不能明显下降。
- `Coverage@500` 有提升更好。
- `Cold Recall@500` 如果下降过多，需要谨慎采用。

### 5.7 最终 full_test 冻结验收

只把 full_val 最优方案跑一次 full_test。模板：

```bash
python scripts/run_rank_mix.py \
  --profile full_48gb_optimized \
  --panel full_test \
  --cpu-threads 14 \
  --din-weight <BEST_DIN_WEIGHT> \
  --pr3-weight <BEST_PR3_WEIGHT> \
  --rrf-k <BEST_RRF_K> \
  --mmr-lambda <BEST_MMR_LAMBDA>
```

不要提前跑多组 full_test。

## 6. 实验记录表

### 6.1 full_val 选型表

| 实验 | top_per_channel | max_candidates | DIN weight | PR3 weight | RRF k | MMR lambda | Recall@10 | NDCG@10 | Utility@10 | Recall@K | NDCG@K | Coverage@K | Cold Recall@K | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_24gb_ref | 150 | 300 | 0.6 | 0.4 | 60 | 0.9 | 0.008906 | 0.886378 | 1.082357 | 0.137488 | 0.721873 | 0.518485 | 0.220177 | 旧 full_val MMR |
| full48_main | 300 | 500 | 0.6 | 0.4 | 60 | 0.9 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 主实验 |
| mix_0.5_0.5 | 300 | 500 | 0.5 | 0.5 | 60 | 0.9 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 权重搜索 |
| mix_0.7_0.3 | 300 | 500 | 0.7 | 0.3 | 60 | 0.9 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 权重搜索 |
| rrf_30 | 300 | 500 | 0.6 | 0.4 | 30 | 0.9 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | RRF 搜索 |
| rrf_100 | 300 | 500 | 0.6 | 0.4 | 100 | 0.9 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | RRF 搜索 |
| mmr_0.85 | 300 | 500 | 最优 | 最优 | 最优 | 0.85 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | MMR 搜索 |
| mmr_0.95 | 300 | 500 | 最优 | 最优 | 最优 | 0.95 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | MMR 搜索 |

### 6.2 full_test 冻结验收表

| 指标 | 当前 baseline | 新方案 | 是否提升 |
|---|---:|---:|---|
| Recall@10 | 0.009218 | 待填 | 待填 |
| NDCG@10 | 0.876574 | 待填 | 待填 |
| Utility@10 | 1.053530 | 待填 | 待填 |
| Recall@200 | 0.139656 | 待填 | 待填 |
| NDCG@200 | 0.706562 | 待填 | 待填 |
| Coverage@200 | 0.524497 | 待填 | 待填 |
| Cold Recall@200 | 0.237664 | 待填 | 待填 |

注意：

`full_48gb_optimized` 的 `recall_k=500`，所以新方案会额外产出 `Recall@500`、`NDCG@500` 等指标。为了和旧 baseline 公平对比，最终报告中需要同时保留 @200 和 @500。如果脚本默认只输出 @10 和 @500，需要在最终汇总时补一版 @200 对齐表。

## 7. 选型规则

最终选择策略时，按以下优先级：

1. `NDCG@10`：最终展示质量，优先级最高。
2. `Utility@10`：短视频业务体验，优先级很高。
3. `Recall@200` / `NDCG@200`：用于和旧 baseline 公平对比。
4. `Recall@500` / `NDCG@500`：用于判断扩候选池后的上限。
5. `Coverage@K`：多样性指标，辅助判断 MMR。
6. `Cold Recall@K`：不能大幅恶化。

如果指标冲突：

- `NDCG@10` 明显提升但 Coverage 小降，可以接受。
- Coverage 提升但 `NDCG@10` 明显下降，不采用。
- Recall@500 提升但 Recall@200 不提升，说明候选池扩大了但精排没吃到收益，需要优先调 RankMix / DIN。

## 8. 风险和降级方案

### 8.1 显存不足

如果 `batch_size=32768` OOM：

```bash
--batch-size 24576
```

如果仍然 OOM：

```bash
--batch-size 16384
```

保持 `din_history_length=150`，不要先降历史长度。

### 8.2 训练时间超时

如果前 5 小时还没完成主 RankMix full_val：

- 不启动任何双塔增强、多行为 DIN、OOVAwareDIN。
- 只跑 `0.6/0.4, rrf_k=60` 和 `0.7/0.3, rrf_k=60`。
- MMR 只保留 `lambda=0.9`。

### 8.3 指标没有提升

如果 `full_48gb_optimized` 没有超过旧 baseline：

- 不要强行替换最终 baseline。
- 把新实验写成“扩候选和长序列实验没有稳定收益”的负实验。
- 保留旧 `rankmix_lambdarank_din_mmr` 作为最终 baseline。

## 9. 最终产物

实验结束后应保存：

```text
artifacts/full_48gb_optimized/rank_mix/summary_full_val.csv
artifacts/full_48gb_optimized/rank_mix/results_full_val.json
artifacts/full_48gb_optimized/rank_mix/summary_full_test.csv
artifacts/full_48gb_optimized/rank_mix/results_full_test.json
reports/FULL_48GB_OPTIMIZATION_REPORT.md
```

最终报告必须包含：

- 机器配置。
- 实验时间。
- full_val 选型表。
- full_test 冻结验收表。
- 新旧 baseline 对比。
- 显著性检验。
- 是否替换最终服务 baseline。

## 10. 面试表达

如果新方案提升：

> 我在 48GB 显存机器上做了全量优化，主要不是盲目换模型，而是基于已有负实验删掉低收益方向，集中做候选池扩容、候选级主排序增强和 RankMix/MMR 参数重搜。最终只在 full_val 上选型，再用 full_test 做一次冻结验收，保证评测协议干净。

如果新方案没有提升：

> 我尝试过使用更大候选池、更强候选级排序和新的融合参数，但 full_val / full_test 没有稳定超过旧 baseline。因此最终没有强行替换模型，而是保留原来的 RankMix + MMR。这个负实验说明当前瓶颈可能不是单纯模型容量，而是候选正负样本定义、内容表示质量或全曝光评测下的排序目标。
