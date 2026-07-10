# KuaiRec 推荐系统实验 RUNBOOK

> **评测协议已于结果 schema v7 重构。** 从现在开始，所有模型选型与消融统一使用
> `small_matrix/full_val` 的近全曝光反馈；`small_matrix/full_test` 只用于冻结验收；
> Big Temporal 结果仅作为有曝光偏差的诊断。旧章节中的 `small_test`、以 Big
> Validation 选择冠军、以及旧候选精排结果均已废弃。完整协议与最新命令见
> `docs/FULL_EXPOSURE_EVALUATION.md`。当前 `local_8gb_large/full_val` 结果见
> `reports/FULL_EXPOSURE_LOCAL_EXPERIMENT_REPORT.md`。当前最终服务策略为
> `rankmix_lambdarank_din_mmr`，48GB boosted 冻结 `full_test` 结果为
> `Recall@10=0.009334`、`NDCG@10=0.886832`、`Recall@200=0.153195`、
> `NDCG@200=0.757705`、`Coverage@200=0.360685`。曝光偏差诊断见
> `reports/exposure_bias/EXPOSURE_BIAS_EXPERIMENT_REPORT.md`。AutoDL RTX 3090 24GB
> 全量结果见 `reports/FULL_24GB_EXPERIMENT_REPORT.md`，48GB boosted 结果见
> `reports/FULL_48GB_BOOSTED_EXPERIMENT_REPORT.md`。

## 1. 目标与当前实现范围

本 RUNBOOK 用于在 RTX 4060 Ti 8GB 上完成小规模 baseline 与消融实验，并在 RTX 3090/4090 24GB 上使用相同代码运行完整实验。

当前已实现：

- 数据下载、MD5 校验、清洗和 Parquet 化；
- 两档实验规模：`local_8gb_large`、`full_24gb`；
- 召回：时间衰减热门、ItemCF、类目内容召回、GPU 双塔、Round-Robin 多路融合；
- 召回指标：Recall、HitRate、NDCG、Coverage、平均流行度、Cold Recall；
- 精排：Logistic Regression、LightGBM LambdaRank、DeepFM、DIN、MMoE、RankMix；
- 重排：基于 RankMix 相关性分数的 MMR 多样性重排；
- 精排特征消融：基础、用户行为、去除 Item 统计、去除 User 统计、完整特征；
- 精排指标：AUC、LogLoss、GAUC、用户-天 NDCG@10；
- 严格用户-天召回评估、Big Test 与 Small Test 冻结后审计；
- 逐用户-天指标文件与 paired bootstrap 显著性对比；
- 配置驱动消融和每次运行独立 JSON 结果存储；
- 一键消融、结果汇总和基础自动化测试。

公开文档统一使用可读实验名，例如 `feature_tower_id_dropout` 和
`rankmix_lambdarank_din`。早期章节中保留的 `R3.4`、`PR3`、`DR2.din`
属于历史 artifact ID，仅用于追溯旧实验结果。

## 2. 真实数据时间块

KuaiRec 日志日期并不连续，实际包含：

- 历史统计窗口：2020-07-05 至 2020-07-12；
- 排序训练窗口：2020-08-01 至 2020-08-10；
- 验证窗口：2020-08-27 至 2020-08-31；
- 测试窗口：2020-09-01 至 2020-09-05。

用户和物品统计只使用 7 月历史块构造，排序模型使用 8 月 1–10 日训练，避免训练行标签进入自己的统计特征。

召回模型使用全部训练期正反馈。验证和测试不参与训练。

## 3. 环境初始化

在项目根目录执行：

```powershell
python --version
nvidia-smi
python -m pip install -r requirements.txt
python -m pytest -q
```

预期：

- Python 3.11；
- `torch.cuda.is_available()` 为 `True`；
- 测试全部通过。

检查 CUDA：

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. 数据准备

如果数据尚未下载：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1
```

构建本机小规模数据：

```powershell
python scripts\run_experiment.py --profile local_8gb_large --stage prepare
```

生成文件：

```text
data/processed/local_8gb_large/
├── interactions.parquet
├── users.parquet
├── items.parquet
└── summary.json
```

## 5. 推荐运行顺序

### 5.1 第一步：4060 Ti 本机正式消融

`local_8gb_large` 默认配置：

- 最多 3,000 名用户；
- 保留所选用户的全部交互，不再随机裁剪用户历史；
- 双塔维度 64、batch size 4096、训练 5 epochs；
- LightGBM 最多使用 300 万训练行；
- 召回评估最多 1,800 名用户。

运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_ablation.ps1 -Profile local_8gb_large
python scripts\summarize_results.py --profile local_8gb_large
```

正式模型比较运行三个种子：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_ablation.ps1 -Profile local_8gb_large -Seeds "2026,2027,2028"
python scripts\summarize_results.py --profile local_8gb_large
```

建议在运行时另开终端观察：

```powershell
nvidia-smi -l 2
```

如果显存不足，在 `configs/profiles.yaml` 中依次降低：

1. `two_tower_batch_size`：4096 -> 2048 -> 1024；
2. `two_tower_embedding_dim`：64 -> 32；
3. `max_users`：3000 -> 2000。

### 5.2 第二步：单独复跑候选模型

召回单模型：

```powershell
python scripts\run_experiment.py --profile local_8gb_large --stage recall --model popular
python scripts\run_experiment.py --profile local_8gb_large --stage recall --model itemcf
python scripts\run_experiment.py --profile local_8gb_large --stage recall --model content
python scripts\run_experiment.py --profile local_8gb_large --stage recall --model two_tower --seed 2027
python scripts\run_experiment.py --profile local_8gb_large --stage recall --model fusion
```

精排单模型：

```powershell
python scripts\run_experiment.py --profile local_8gb_large --stage ranking --model logistic --feature-set basic
python scripts\run_experiment.py --profile local_8gb_large --stage ranking --model lightgbm --feature-set full
python scripts\run_experiment.py --profile local_8gb_large --stage ranking --model lightgbm --feature-set no_item_stats
python scripts\run_experiment.py --profile local_8gb_large --stage ranking --model lightgbm --feature-set no_user_stats
```

对双塔和最终候选模型至少运行三个种子：

```powershell
python scripts\run_experiment.py --profile local_8gb_large --stage recall --model two_tower --seed 2026
python scripts\run_experiment.py --profile local_8gb_large --stage recall --model two_tower --seed 2027
python scripts\run_experiment.py --profile local_8gb_large --stage recall --model two_tower --seed 2028
```

### 5.3 冻结后 Test 与小矩阵审计

只在根据 Validation 冻结实验方案后运行：

```powershell
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R1.0 --panel test
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R1.0 --panel small_test
python scripts\run_experiment.py --profile local_8gb_large --stage ranking --experiment-id P1.full --panel test
python scripts\run_experiment.py --profile local_8gb_large --stage ranking --experiment-id P1.full --panel small_test
```

小矩阵不会参与训练或 Validation 调参。

### 5.4 第三步：24GB GPU 完整实验

将项目和原始数据复制到 24GB GPU 机器，安装相同依赖后运行：

```powershell
python -m pytest -q
powershell -ExecutionPolicy Bypass -File scripts\run_ablation.ps1 -Profile full_24gb
python scripts\summarize_results.py --profile full_24gb
```

如果 AutoDL 驱动只支持到 CUDA 12.8，而 `pip install -r requirements.txt`
自动装到了 `torch+cu130`，会出现 `torch.cuda.is_available() == False`。
此时改装 cu128 版本：

```bash
pip install --force-reinstall torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

双塔召回导出候选已优化为批量 embedding 矩阵 topK。24GB 卡上可先跑一次 benchmark：

```bash
python scripts/benchmark_batched_topk.py --users 7176 --items 10728 --dim 128 --k 1000 --batch-size 4096
```

AutoDL RTX 3090 24GB 实测：`7176 x 10728`、`dim=128`、`K=1000`
约 `8.09s` 完成 topK 导出。

完整数据的 ItemCF 共现构建主要受 CPU 和内存限制，不是 GPU 限制。如果内存紧张，先将：

```yaml
itemcf_history_length: 100
itemcf_neighbors: 400
```

降低为：

```yaml
itemcf_history_length: 50
itemcf_neighbors: 200
```

## 6. 如何选择最佳 baseline

### 6.1 召回

主指标使用 `Recall@K`，同时设置以下守门指标：

- ItemCF 应显著改善 Coverage，不能只复制热门；
- 内容召回重点看 `Cold Recall@K`；
- 双塔至少运行三个随机种子；
- Fusion 必须相对最佳单通道有增益，否则说明融合策略或通道互补性不足；
- 不以小样本或临时调试结果选冠军，只以 `local_8gb_large` 和最终 `full_24gb` 结果为准。

### 6.2 精排

推荐选择顺序：

1. Logistic Regression 验证数据和指标；
2. 对比 LightGBM `basic` 与 `behavior`，确认用户统计增益；
3. 对比 `full` 与 `no_item_stats`，确认物品历史统计增益；
4. 对比 `full` 与 `no_user_stats`，确认用户历史统计增益；
5. 使用 AUC、GAUC、NDCG@10 和 LogLoss 联合选择。

如果完整特征只提升 AUC、却降低 GAUC 或 NDCG，应检查模型是否只学会全局热门和视频时长。

## 7. 修复前结果状态

此前 `local_8gb_large` 首轮结果存在时间特征解析、召回 Ground Truth、候选可用性和结果存储问题，已经作废，不能用于选择 baseline 或写入简历。

修复后必须重新运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_ablation.ps1 -Profile local_8gb_large -Seeds "2026,2027,2028"
python scripts\summarize_results.py --profile local_8gb_large
```

修复后的实验具有以下保证：

- 时间特征使用本地事件时间，缺失时按 Unix 时间戳或日期回退；
- 小规模 profile 保留所选用户完整交互历史；
- Ground Truth 与 seen-item 过滤口径一致；
- 候选按首次出现日期进行用户-天过滤；
- 用户/物品统计只使用样本日期之前的行为；
- 每次实验独立保存，结果 CSV 从 run JSON 自动重建；
- Test 与小矩阵仅用于冻结后审计。

修复后的 `local_8gb_large` 三种子正式实验和冻结后审计已经完成，完整表格、显著性与模型选型见：

`reports/LOCAL_8GB_LARGE_EXPERIMENT_REPORT.md`

## 9. 显著性检验

每个实验会保存逐用户-天指标。示例：

```powershell
python scripts\compare_experiments.py --profile local_8gb_large --stage recall --experiment-a R0.1 --experiment-b R1.0 --metric recall_at_100
python scripts\compare_experiments.py --profile local_8gb_large --stage ranking --experiment-a P1.0 --experiment-b P1.full --metric ndcg_at_10
```

只有 `ci_95_low` 和 `ci_95_high` 同号时，才能认为实验 B 相比实验 A 有稳定变化。

## 10. 正式实验记录表

### 10.1 召回消融记录

| 日期 | Profile | Seed | 模型 | Recall@K | NDCG@K | HitRate@K | Coverage@K | Cold Recall@K | Avg Popularity | 耗时秒 | 结论 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| | | | | | | | | | | | |

### 10.2 精排消融记录

| 日期 | Profile | Seed | 模型 | 特征集 | AUC | GAUC | NDCG@10 | LogLoss | 训练行数 | 耗时秒 | 结论 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| | | | | | | | | | | | |

### 10.3 最终模型对比

| 阶段 | 最佳简单 baseline | 最佳复杂模型 | 主指标增益 | Coverage/GAUC 变化 | 资源成本变化 | 是否采用 |
|---|---|---|---:|---:|---:|---|
| 召回 | | | | | | |
| 精排 | | | | | | |

## 11. 常见问题

### CUDA Out Of Memory

降低 `two_tower_batch_size` 和 `two_tower_embedding_dim`。LightGBM 默认使用 CPU，不消耗大量显存。

### ItemCF 很慢或占用大量内存

降低 `max_users`、`itemcf_history_length` 和 `itemcf_neighbors`。ItemCF 的主要复杂度来自单用户历史中的物品两两共现。

### 结果 CSV 出现重复实验

每次运行都会保留独立 JSON；汇总时对同一面板、实验和种子只采用最新一次。使用：

```powershell
python scripts\summarize_results.py --profile local_8gb_large
```

查看按模型聚合后的平均指标。

### Cold Recall 为 0

Popular、ItemCF 和 ID-only 双塔无法召回训练期未出现视频，出现 0 是合理结果。内容召回通过静态类目侧信息召回新视频，是冷启动实验的主要 baseline。

## 12. 下一步扩展

完成当前实验并冻结最佳召回与 LightGBM baseline 后，再按顺序增加：

1. TF-IDF 标题内容召回；
2. 双塔用户画像与 Item 内容特征；
3. 固定候选集上的 DeepFM；
4. DIN 序列精排；
5. 完播、强兴趣、短播多任务精排；
6. 小矩阵全量静态诊断和 IPS/SNIPS 对比。

## 13. 固定配额融合实验

当前已实现平滑配额融合 `RF1/RF2/RF3`。运行 Validation 三种子消融：

```powershell
foreach ($exp in @('RF1','RF2','RF3')) {
  foreach ($seed in @(2026,2027,2028)) {
    python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id $exp --seed $seed --panel valid
  }
}
python scripts\summarize_results.py --profile local_8gb_large
```

冻结 Validation 最优的 `RF2` 后运行审计：

```powershell
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id RF2 --seed 2026 --panel test
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id RF2 --seed 2026 --panel small_test
python scripts\compare_experiments.py --profile local_8gb_large --stage recall --panel test --experiment-a R1.0 --experiment-b RF2 --metric recall_at_100
python scripts\compare_experiments.py --profile local_8gb_large --stage recall --panel small_test --experiment-a R2.0 --experiment-b RF2 --metric recall_at_100
```

| Panel | 最佳单路 | RF2 Recall@100 | 单路 Recall@100 | RF2 相对变化 | 是否采用 RF2 |
|---|---|---:|---:|---:|---|
| Validation | R1.0 ItemCF | 0.027418 | 0.025304 | +8.36% | 仅作为调参胜者 |
| Big Test | R1.0 ItemCF | 0.021190 | 0.022597 | -6.23% | 否 |
| Small Test | R2.0 Content | 0.001464 | 0.003365 | -56.48% | 否 |

结论：不要使用固定全局配额替换单路专长模型；下一步应实验按用户历史和冷启动状态动态路由。

## 14. 增强内容召回实验

新增内容实验使用 Caption、封面文字、Topic Tag 和多级类目：

```powershell
foreach ($exp in @('R2.3','R2.4','R2F1','R2F2','R2F3')) {
  foreach ($seed in @(2026,2027,2028)) {
    python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id $exp --seed $seed --panel valid
  }
}
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R2.4 --seed 2026 --panel test
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R2.4 --seed 2026 --panel small_test
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R2F3 --seed 2026 --panel test
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R2F3 --seed 2026 --panel small_test
```

| 场景 | 最佳内容召回 | Recall@100 | Cold Recall@100 | 采用策略 |
|---|---|---:|---:|---|
| Big Test 正常时间流量 | R2.4 类目 + TF-IDF | 0.017634 | 0.018340 | 采用 |
| Small Test 跨曝光流量 | R2.0 一级类目 | 0.002754 | 0.002783 | 采用 |
| 统一内容融合 | R2F3 | 0.016297 / 0.000841 | 0.016854 / 0.000627 | 不采用 |

## 15. 特征增强双塔实验

```powershell
foreach ($seed in @(2026,2027,2028)) {
  python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R3.3 --seed $seed --panel valid
}
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R3.3 --seed 2026 --panel test
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R3.3 --seed 2026 --panel small_test
```

| Panel | R3.3 Recall@100 | 对照模型 | 对照 Recall@100 | 结论 |
|---|---:|---|---:|---|
| Validation | 0.036484 | R1.0 ItemCF | 0.031266 | R3.3 显著更高 |
| Big Test | 0.015427 | R1.0 ItemCF | 0.027398 | R3.3 时间泛化失败 |
| Small Test | 0.000365 | R2.0 Content | 0.002754 | R3.3 跨曝光泛化失败 |

R3.3 不进入当前最佳 baseline。下一步应把召回来源、通道排名和用户-候选匹配特征加入候选级精排。

## 16. Test 前滚动更新实验

滚动实验使用 Train + Validation 更新模型，只允许评估 Test：

```powershell
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R1.0R --seed 2026 --panel test
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R2.4R --seed 2026 --panel test
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R3.3R --seed 2026 --panel test
```

| 模型 | Recall@100 | NDCG@100 | Cold Recall@100 | 结论 |
|---|---:|---:|---:|---|
| R1.0R 滚动 ItemCF | 0.164005 | 0.089346 | 0.000000 | 更新后主召回 |
| R2.4R 滚动内容召回 | 0.034280 | 0.015503 | 0.050912 | 更新后冷启动召回 |
| R3.3R 滚动双塔 | 0.027514 | 0.011758 | 0.003793 | 刷新有效，但仍不采用 |

滚动模型的 Ground Truth 会排除截至 Validation 结束已经看过的视频，Cold Recall 也相对更新后的已知 Item 池计算。因此不要把滚动指标与旧冻结指标直接做显著性比较。

## 17. 双塔增量覆盖与训练消融

候选重叠分析：

```powershell
python scripts\analyze_candidate_overlap.py --profile local_8gb_large --panel test --train-through valid --base R1.0R R2.4R --candidate R3.6R
```

双塔训练消融：

```powershell
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R3.4R --seed 2026 --panel test
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R3.5R --seed 2026 --panel test
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R3.6R --seed 2026 --panel test
```

| 模型 | Recall@100 | Cold Recall@100 | 增量 Oracle Recall | 决策 |
|---|---:|---:|---:|---|
| R3.3R 原始双塔 | 0.027514 | 0.003793 | +0.005528 | 对照 |
| R3.4R ID Dropout | 0.026218 | **0.009354** | +0.005807 | 冷启动更好 |
| R3.5R 难负采样 | **0.028065** | 0.001770 | +0.005305 | 单路更好 |
| R3.6R 两者组合 | 0.024428 | 0.004573 | **+0.006123** | 扩展候选源 |

候选级精排实现前，使用 5% 双塔配额的 `RFR1` 作为阶段性工程 Baseline；候选级精排完成后，RFR1 仅作为降级策略。

采用门槛：

- 相对 RFR0 的 Recall 损失不超过 2%；
- Coverage 相对提升至少 5%；
- 双塔必须提供非零增量 Oracle Recall。

RFR1 的 Recall 下降 1.96%，Coverage 提升 6.93%，Cold Recall 提升 4.02%，满足门槛。RFR2 的 Recall 下降 4.33%，不采用。

部署配置见：`configs/serving_baseline.yaml`。

## 18. 候选级精排实验

先生成 Validation 对应的双塔训练候选：

```powershell
python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id R3.6 --seed 2026 --panel valid
```

运行候选级精排：

```powershell
python scripts\run_candidate_ranking.py --profile local_8gb_large --seed 2026 --top-per-channel 150 --max-candidates-per-group 300 --estimators 500
```

| Exp | Recall@100 | NDCG@100 | Cold Recall@100 | 双塔独有候选入选率 |
|---|---:|---:|---:|---:|
| PR1 基础特征 | 0.168742 | 0.083950 | 0.048049 | 5.78% |
| PR2 召回来源/排名 | 0.089096 | 0.046958 | 0.013869 | 47.61% |
| **PR3 完整候选级精排** | **0.186139** | **0.104246** | 0.074883 | **10.55%** |
| PR.no_tower | 0.177259 | 0.098565 | **0.075256** | 0.00% |

PR3 相对移除双塔候选的版本显著提升 Recall 和 NDCG，因此双塔正式进入候选池。部署时每路召回 Top150，合并去重后由 PR3 输出 Top100。RFR1 保留为精排降级方案。

结果汇总：`artifacts/local_8gb_large/candidate_ranking/summary.csv`。

## 19. PR3 特征组消融与固定候选缓存

首次运行会重建并持久化固定候选集；后续 DeepFM、DIN 和多任务模型必须复用该缓存：

```powershell
python scripts\run_candidate_ranking.py --profile local_8gb_large --seed 2026 --top-per-channel 150 --max-candidates-per-group 300 --estimators 500 --rebuild-cache
```

关键输出：

- `artifacts/local_8gb_large/candidate_ranking/cache/valid_train_candidates.parquet`
- `artifacts/local_8gb_large/candidate_ranking/cache/test_candidates.parquet`
- `artifacts/local_8gb_large/candidate_ranking/summary.csv`
- `artifacts/local_8gb_large/candidate_ranking/daily_stability.csv`
- `artifacts/local_8gb_large/candidate_ranking/feature_importance.csv`

| Exp | Recall@100 | NDCG@100 | 结论 |
|---|---:|---:|---|
| PR3 | **0.186139** | 0.104246 | 最佳主 Recall |
| PR3.no_recall_features | 0.178306 | 0.105006 | 召回特征有效 |
| PR3.no_cross_features | 0.183978 | **0.107764** | Recall/NDCG 存在冲突 |
| PR3.no_temporal_features | 0.116366 | 0.056518 | 时间特征不可移除 |

## 20. 固定候选集深度精排

依次运行，避免同时占用显存：

```powershell
python scripts\run_deep_candidate_ranking.py --profile local_8gb_large --seed 2026 --models deepfm
python scripts\run_deep_candidate_ranking.py --profile local_8gb_large --seed 2026 --models din
python scripts\run_deep_candidate_ranking.py --profile local_8gb_large --seed 2026 --models multitask
```

`local_8gb_large` 默认使用 3 epochs、batch size 8192、embedding dim 16、DIN 历史长度 50。若显存不足，优先降低 `--batch-size`。

| Exp | Recall@100 | NDCG@100 | Cold Recall@100 | Coverage@100 | 训练秒 | 决策 |
|---|---:|---:|---:|---:|---:|---|
| DR1.deepfm | 0.104901 | 0.051783 | 0.027790 | **0.497204** | 5.89 | 不替换 PR3 |
| DR2.din | 0.125425 | 0.061885 | **0.047234** | 0.440529 | 24.52 | 序列兴趣有效 |
| **DR3.mmoe_complete** | **0.153059** | **0.082449** | 0.010509 | 0.307327 | 8.65 | 最佳深度 challenger |
| DR3.mmoe_complete_strong | 0.151248 | 0.077748 | 0.009871 | 0.303598 | 8.65 | 强兴趣权重不采用 |
| DR3.mmoe_multitask | 0.151615 | 0.077856 | 0.009116 | 0.335291 | 8.65 | 业务加权分数不采用 |
| **PR3** | **0.186139** | **0.104246** | **0.074883** | 0.313292 | 30.05 | 最终 baseline |

结果汇总：`artifacts/local_8gb_large/deep_candidate_ranking/summary.csv`。

## 21. 显著性与 24GB GPU 全量复跑

跨阶段比较 PR3 与最佳深度模型：

```powershell
python scripts\compare_experiments.py --profile local_8gb_large --stage-a deep_candidate_ranking --stage-b candidate_ranking --panel test --experiment-a DR3.mmoe_complete --experiment-b PR3 --metric recall_at_100
```

24GB GPU 上将 profile 替换为 `full_24gb`，先完成三路 Validation/Test 召回产物，再执行：

```powershell
python scripts\run_candidate_ranking.py --profile full_24gb --seed 2026 --top-per-channel 150 --max-candidates-per-group 300 --estimators 1200 --rebuild-cache
python scripts\run_deep_candidate_ranking.py --profile full_24gb --seed 2026 --models deepfm
python scripts\run_deep_candidate_ranking.py --profile full_24gb --seed 2026 --models din
python scripts\run_deep_candidate_ranking.py --profile full_24gb --seed 2026 --models multitask
```

当前 Test 已用于模型审计，禁止继续在该 Test 上选择深度融合权重。下一步 listwise/pairwise loss 或 OOF 深度分数融合必须使用新的 Validation 窗口。

AutoDL RTX 3090 24GB 全量结果：

| Panel | 阶段 | Exp | Recall@10 | NDCG@10 | Recall@200 | NDCG@200 | Cold Recall@200 | Coverage@200 | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| full_val | 召回 | R1.0 ItemCF | - | - | 0.092415 | 0.493586 | 0.000000 | 0.773670 | 协同过滤覆盖对照 |
| full_val | 召回 | R2.4 Content | - | - | 0.065363 | 0.353986 | 0.080218 | 0.429817 | 冷启动内容通道 |
| full_val | 召回 | R3.4 TwoTower | - | - | 0.129288 | 0.673765 | **0.274715** | 0.484821 | 最强单路召回 |
| full_val | 精排 | PR3 LambdaRank | 0.008681 | 0.867396 | 0.135728 | 0.712195 | 0.209376 | **0.593928** | CPU fallback |
| full_val | 精排 | DR2.din | 0.008822 | 0.880864 | 0.135536 | 0.714261 | **0.248442** | 0.408777 | 序列模型组件 |
| full_val | 融合精排 | DR4.rank_mix | **0.008896** | **0.886979** | **0.137131** | **0.721257** | 0.241866 | 0.483919 | 选型胜者 |
| full_test | 精排 | PR3 LambdaRank | 0.008970 | 0.856819 | 0.138076 | 0.697271 | 0.215410 | **0.622783** | 冻结测试 fallback |
| full_test | 精排 | DR2.din | 0.009116 | 0.868332 | 0.137707 | 0.698952 | **0.255448** | 0.429817 | 冻结测试序列模型 |
| full_test | 融合精排 | DR4.rank_mix | 0.009220 | 0.878226 | 0.139507 | 0.706465 | 0.247697 | 0.519988 | 原始 RankMix 冻结测试参考 |
| full_test | 重排 | RankMix + MMR | **0.009218** | 0.876574 | **0.139656** | **0.706562** | 0.237664 | **0.524497** | 最终服务策略 |

最终采用：`R1.0 + R2.4 + R3.4` 三路候选，`DR4.rank_mix` 作为主相关性排序，
并在最终展示前使用 MMR 做多样性重排。`DR2.din` 作为序列兴趣组件，
`PR3` 作为 CPU/tabular fallback。冻结 `full_test` 已用于验收，后续不要继续在该面板上调参。

## 22. MMR 第三层重排实验

MMR 位于最终 TopK 生成前，结构为：

```text
召回层：ItemCF + 内容召回 + 双塔召回
精排层：DIN + LambdaRank，通过 RankMix 融合相关性排序
重排层：MMR 在相关性和多样性之间做 trade-off
```

当前实现使用 `rankmix_lambdarank_din` 作为相关性分数，并用类目和作者构造相似度：

```text
MMR = lambda * relevance - (1 - lambda) * max_similarity_to_selected
similarity = 0.6 * same_category + 0.4 * same_author
```

默认 `lambda=0.9`，即主要保留 RankMix 的相关性，只轻微惩罚同类目、同作者的重复内容。运行命令：

```powershell
python scripts\run_rank_mix.py --profile local_8gb_large --panel full_val --mmr-lambda 0.9 --cpu-threads 12
```

在 24GB GPU 全量机器上：

```bash
python scripts/run_rank_mix.py --profile full_24gb --panel full_val --mmr-lambda 0.9 --cpu-threads 14
```

如果需要做 lambda 消融，只允许先在 `full_val` 上比较：

```powershell
foreach ($lambda in @(0.7, 0.8, 0.9, 0.95)) {
  python scripts\run_rank_mix.py --profile local_8gb_large --panel full_val --mmr-lambda $lambda --cpu-threads 12
}
```

实验记录表：

| Panel | Exp | Lambda | Recall@10 | NDCG@10 | Recall@K | NDCG@K | Coverage@K | Cold Recall@K | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| full_val | rankmix_lambdarank_din | - | 0.008897 | 0.886585 | 0.137736 | 0.723145 | 0.507665 | 0.233041 | 纯离线相关性指标更高 |
| full_val | rankmix_lambdarank_din_mmr | 0.9 | 0.008906 | 0.886378 | 0.137488 | 0.721873 | 0.518485 | 0.220177 | Coverage 提升，@200 相关性小幅下降；作为最终重排层采用 |
| full_test | rankmix_lambdarank_din | - | 0.009220 | 0.878226 | 0.139507 | 0.706465 | 0.519988 | 0.247697 | 原始 RankMix 冻结测试参考 |
| full_test | rankmix_lambdarank_din_mmr | 0.9 | 0.009218 | 0.876574 | 0.139656 | 0.706562 | 0.524497 | 0.237664 | 最终服务策略 |

采用门槛：

- `NDCG@10` 不应明显下降；
- `Recall@K` 损失应可解释且幅度很小；
- `Coverage@K` 或推荐内容多样性应有提升；
- 如果业务目标强调最终展示多样性，可以接受小幅 @200 损失，将 MMR 作为产品重排层；如果只追求离线相关性最高，则仍以原始 RankMix 作为对照。

本轮 AutoDL 全量 MMR 实验结论见：

```text
reports/MMR_RERANKING_EXPERIMENT_REPORT.md
```

## 23. DIN 排序目标与滚动序列实验

该实验严格在 Validation 内选型：

```powershell
python scripts\run_din_ranking_loss.py --profile local_8gb_large --seed 2026
```

脚本使用 2020-08-27 至 08-29 训练，2020-08-30 至 08-31 选型，并运行：

- 点式 BCE；
- 组内 BPR 与补足训练预算的 BPR；
- BPR + BCE 混合损失；
- 完整候选组 ListNet；
- 严格日期前滚动序列 BCE。

只有相对 BCE 的 Recall 配对 bootstrap 95% CI 下界大于 0，且 NDCG 不显著下降，challenger 才能进入 Test。当前没有 challenger 满足门槛，因此脚本引用已有 `DR2.din` Test 结果，不重新运行 Test。

| Exp | Recall@100 | NDCG@100 | 结论 |
|---|---:|---:|---|
| **DL1.din_bce** | 0.102036 | **0.069305** | 最终选择 |
| DL5.din_hybrid_budget | 0.100619 | 0.063757 | 显著退化 |
| DL7.din_listnet | 0.101927 | 0.069178 | 与 BCE 统计等价 |
| DL8.din_bce_rolling | **0.102108** | 0.069185 | Recall 均值提升不显著 |

结果目录：`artifacts/local_8gb_large/din_ranking_loss/`。

## 24. DIN 序列增强与 OOV 内容实验

按照顺序分别运行：

```powershell
python scripts\run_din_sequence_enhancements.py --profile local_8gb_large --seed 2026
python scripts\run_din_enriched_sequence.py --profile local_8gb_large --seed 2026
python scripts\run_din_oov_content.py --profile local_8gb_large --seed 2026
```

每轮只使用 2020-08-27 至 08-29 训练、08-30 至 08-31 选型；只有 Recall 显著提升且 NDCG 不显著下降才会自动进入冻结 Test。

| 轮次 | Exp | Recall@100 | NDCG@100 | Candidate OOV Recall | 是否进入 Test |
|---|---|---:|---:|---:|---|
| 对照 | DL1.din_bce | 0.102036 | **0.069305** | - | 已有冻结结果 |
| 1 | DS1.din_multibehavior | 0.102218 | 0.069429 | - | 否 |
| 2 | DS2.din_author_content_time | 0.101768 | 0.069210 | **0.051086** | 否 |
| 3 | DS3.1.din_oov_content | 0.102276 | 0.068736 | **0.051086** | 否 |
| 3 | DS3.2.din_oov_content_dropout | **0.102408** | 0.068184 | **0.051086** | 否 |

结果目录：`artifacts/local_8gb_large/din_sequence_enhancements/`。

