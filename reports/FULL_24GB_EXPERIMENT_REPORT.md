# KuaiRec 24GB 全量实验报告

## 1. 实验环境

- 机器：AutoDL RTX 3090 24GB，14 核 CPU，90GB 内存。
- CUDA / PyTorch：CUDA 可正常识别 RTX 3090。
- 实验配置：`full_24gb`。
- 数据规模：12,530,806 条交互，7,176 名用户，10,728 个视频。
- 全曝光评测面板：`full_val` 717 名用户，`full_test` 694 名用户。
- 评测协议：使用 KuaiRec 小矩阵近全曝光反馈做模型选择和冻结验收，避免传统稀疏日志评测的曝光偏差。

## 2. 工程优化

| 优化项 | 证据 | 结论 |
|---|---:|---|
| 双塔批量 TopK | `7176 users x 10728 items x dim128, K=1000` 用时约 8 秒 | 保留 |
| 双塔 checkpoint 缓存 | 复用 full exposure checkpoint 后评测只需秒级加载 | 保留 |
| 多核 LightGBM / PyTorch 线程 | `--cpu-threads` 可显式使用多核 CPU | 保留 |
| 固定候选缓存 | 候选级精排复用 parquet cache，避免重复构造候选 | 保留 |

优化后，双塔候选导出不再是主要瓶颈。首次运行耗时主要来自模型训练和候选特征构建，后续可以依赖 checkpoint 和候选缓存复用。

## 3. 召回实验结果

| Panel | 公开命名 | 历史 ID | 模型 | Recall@200 | NDCG@200 | Cold Recall@200 | Coverage@200 | 耗时 |
|---|---|---|---|---:|---:|---:|---:|---:|
| `full_val` | `itemcf_main` | `R1.0` | ItemCF | 0.092415 | 0.493586 | 0.000000 | 0.773670 | 88.54 秒 |
| `full_val` | `content_text_category` | `R2.4` | 内容召回 | 0.065363 | 0.353986 | 0.080218 | 0.429817 | 8.51 秒 |
| `full_val` | `feature_tower_id_dropout` | `R3.4` | 特征双塔 + ID Dropout | 0.129288 | 0.673765 | 0.274715 | 0.484821 | 1.36 秒，缓存命中 |
| `full_test` | `itemcf_main` | `R1.0` | ItemCF | 0.093114 | 0.480963 | 0.000000 | 0.776075 | 81.68 秒 |
| `full_test` | `content_text_category` | `R2.4` | 内容召回 | 0.065112 | 0.341876 | 0.078033 | 0.402164 | 8.32 秒 |
| `full_test` | `feature_tower_id_dropout` | `R3.4` | 特征双塔 + ID Dropout | 0.130753 | 0.656944 | 0.284699 | 0.486023 | 877.78 秒，首次写入缓存 |

召回结论：`feature_tower_id_dropout` 是 `full_val` 和冻结 `full_test` 上最强的单路召回；`itemcf_main` 提供协同过滤覆盖；`content_text_category` 负责内容相似和冷启动补充。最终候选池保留三路召回。

## 4. 精排实验结果

### 4.1 展示层 Top10 指标

| Panel | 公开命名 | 历史 ID | 精排策略 | Recall@10 | NDCG@10 | Cold Recall@10 | Coverage@10 | Utility@10 |
|---|---|---|---|---:|---:|---:|---:|---:|
| `full_val` | `lambdarank_full_features_refit` | `PR3.refit` | LambdaRank | 0.008681 | 0.867396 | 0.028306 | 0.080252 | 1.029219 |
| `full_val` | `din_sequence_ranker_refit` | `DR2.din.refit` | DIN | 0.008822 | 0.880864 | **0.038106** | 0.026150 | 1.063110 |
| `full_val` | `rankmix_lambdarank_din` | `DR4.rank_mix` | RankMix：DIN 0.6 + LambdaRank 0.4 | **0.008896** | **0.886979** | 0.036667 | 0.034265 | **1.076081** |
| `full_test` | `lambdarank_full_features_refit` | `PR3.refit` | LambdaRank | 0.008970 | 0.856819 | 0.028836 | **0.080553** | 1.011599 |
| `full_test` | `din_sequence_ranker_refit` | `DR2.din.refit` | DIN | 0.009116 | 0.868332 | **0.039023** | 0.029757 | 1.039625 |
| `full_test` | `rankmix_lambdarank_din` | `DR4.rank_mix` | RankMix：DIN 0.6 + LambdaRank 0.4 | **0.009225** | **0.878409** | 0.037353 | 0.036970 | **1.054035** |

### 4.2 候选层 Top200 指标

| Panel | 公开命名 | 历史 ID | Recall@200 | NDCG@200 | Cold Recall@200 | Coverage@200 | Utility@200 | 双塔独有候选入选率 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `full_val` | `lambdarank_full_features_refit` | `PR3.refit` | 0.135728 | 0.712195 | 0.209376 | **0.593928** | 0.709414 | 0.455474 |
| `full_val` | `din_sequence_ranker_refit` | `DR2.din.refit` | 0.135536 | 0.714261 | **0.248442** | 0.408777 | 0.712266 | **0.557936** |
| `full_val` | `rankmix_lambdarank_din` | `DR4.rank_mix` | **0.137131** | **0.721257** | 0.241866 | 0.483919 | **0.719104** | 0.526562 |
| `full_test` | `lambdarank_full_features_refit` | `PR3.refit` | 0.138076 | 0.697271 | 0.215410 | **0.622783** | 0.690321 | 0.450713 |
| `full_test` | `din_sequence_ranker_refit` | `DR2.din.refit` | 0.137707 | 0.698952 | **0.255448** | 0.429817 | 0.693588 | **0.553602** |
| `full_test` | `rankmix_lambdarank_din` | `DR4.rank_mix` | **0.139489** | **0.706355** | 0.247849 | 0.520289 | **0.700040** | 0.521009 |

精排结论：最终选择 `rankmix_lambdarank_din` 作为主策略。它不直接混合 LambdaRank 和 DIN 的原始分数，而是使用 RRF 风格的排名融合，避免不同模型分数尺度不可比。

## 5. RankMix 相对 DIN 的显著性检验

使用用户级 paired bootstrap，比较 `rankmix_lambdarank_din - din_sequence_ranker_refit`。如果 95% 置信区间不跨 0，则认为提升具有统计显著性。

| Panel | Metric | Mean Diff | 95% CI Low | 95% CI High | 结论 |
|---|---|---:|---:|---:|---|
| `full_val` | Recall@10 | 0.000074 | 0.000016 | 0.000133 | 显著提升 |
| `full_val` | NDCG@10 | 0.006114 | 0.001376 | 0.011020 | 显著提升 |
| `full_val` | Recall@200 | 0.001595 | 0.001274 | 0.001926 | 显著提升 |
| `full_val` | NDCG@200 | 0.006997 | 0.005762 | 0.008242 | 显著提升 |
| `full_test` | Recall@10 | 0.000109 | 0.000035 | 0.000182 | 显著提升 |
| `full_test` | NDCG@10 | 0.010077 | 0.004655 | 0.015363 | 显著提升 |
| `full_test` | Recall@200 | 0.001783 | 0.001437 | 0.002138 | 显著提升 |
| `full_test` | NDCG@200 | 0.007403 | 0.006107 | 0.008644 | 显著提升 |

结论：`rankmix_lambdarank_din` 相比单独 `din_sequence_ranker_refit` 的优势不是均值波动，而是在 `full_val` 与冻结 `full_test` 上都通过了配对显著性检验。

## 6. 最终 Baseline 架构

```text
召回候选：
  itemcf_main
  content_text_category
  feature_tower_id_dropout

精排：
  主策略：rankmix_lambdarank_din
    = 0.6 * RRF(din_sequence_ranker) + 0.4 * RRF(lambdarank_full_features)
  序列模型：din_sequence_ranker
  降级模型：lambdarank_full_features

评测：
  选型面板：full_val
  冻结验收面板：full_test
```

## 7. 面试讲法建议

这个项目可以按“真实推荐系统链路”来讲：

- 数据层：KuaiRec 有近全曝光小矩阵，因此可以反思传统离线评测的曝光偏差，并用全曝光面板做更可靠的模型选择。
- 召回层：ItemCF 负责协同覆盖，内容召回负责冷启动，双塔负责学习用户和内容表示，并在全量数据上成为最强单路召回。
- 精排层：LambdaRank 是强表格 baseline，DIN 引入用户历史序列兴趣，RankMix 用排名融合稳定结合二者。
- 工程层：实现了批量 TopK、模型 checkpoint、候选缓存、多核训练、全曝光验证和冻结测试，项目不只是调模型，也考虑了复现和部署。

## 8. 注意事项

`full_test` 已经作为冻结测试集使用，不应继续在这个面板上调参。后续如果继续做 MMoE、多行为 DIN 或融合权重实验，应只在新的 validation 切分或交叉验证方案中选型，再把 `full_test` 作为最终一次性验收。
