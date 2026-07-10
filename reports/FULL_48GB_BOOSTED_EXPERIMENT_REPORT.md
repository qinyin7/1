# KuaiRec 48GB Boosted 实验报告

## 1. 实验目标

本轮实验目标是在 `full_48gb_optimized` 的基础上继续提高指标，重点验证三件事：

- 召回层是否可以把增强双塔升级为主力召回。
- 精排层扩大 LightGBM / DIN 容量后，是否能提升 `@10` 和 `@200`。
- RankMix 与 MMR 在更强候选池下是否仍然需要保留 MMR。

评测继续使用 KuaiRec 的近全曝光协议：`full_val` 选型，`full_test` 只做一次冻结验收。

## 2. 配置变更

| 模块 | clean `full_48gb_optimized` | boosted `full_48gb_boosted` | 目的 |
|---|---:|---:|---|
| `top_per_channel` | 300 | 350 | 扩大单路候选池 |
| `max_candidates_per_group` | 500 | 650 | 扩大精排候选上限 |
| 双塔通道 | `feature_tower_id_dropout` | `feature_tower_dropout_hard_negative` | 加入 hard negative |
| 双塔 embedding | 128 | 192 | 增强召回表征能力 |
| 双塔 epoch | 10 | 18 | 训练更充分 |
| 候选通道权重 | 等权 | tower 1.5 / itemcf 1.0 / content 1.0 | 若双塔更强，让双塔成为候选主力 |
| LightGBM estimators | 1600 | 2200 | 增强表格精排 |
| DIN embedding | 64 | 96 | 增强序列表征 |
| DIN epoch | 8 | 12 | 训练更充分 |
| DIN history length | 150 | 200 | 使用更长历史 |

多行为 DIN 也做了实验，但最终没有进入 baseline。原因见第 5 节。

## 3. full_val 召回结果

| 召回通道 | Recall@500 | NDCG@500 | Coverage@500 | Cold Recall@500 | Utility@500 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `itemcf_main` | 0.207361 | 0.447538 | 0.794710 | 0.000000 | 0.421315 | 稳定协同过滤基线 |
| `content_text_category` | 0.167979 | 0.361530 | 0.685002 | 0.203045 | 0.350643 | 冷启动补充 |
| `feature_tower_dropout_hard_negative` | 0.322608 | 0.669889 | 0.595732 | 0.491866 | 0.682608 | 最强召回，升级为主力 |

双塔在 `full_val` 上显著超过 ItemCF：`Recall@500 +0.115247`，`NDCG@500 +0.222352`。因此本轮后续候选池采用双塔主力策略是合理的。

## 4. full_val 精排结果

| 模型 | DIN 变体 | Recall@10 | NDCG@10 | Utility@10 | Recall@200 | NDCG@200 | Utility@200 | Recall@500 | NDCG@500 | Coverage@200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lambdarank_full_features_refit` | - | 0.008862 | 0.875900 | 1.084589 | 0.149109 | 0.765186 | 0.799268 | 0.337079 | 0.705096 | 0.458671 |
| `din_sequence_ranker_refit` | basic | 0.008510 | 0.856405 | 1.002650 | 0.145731 | 0.751053 | 0.785488 | 0.334591 | 0.699547 | 0.234445 |
| `rankmix_lambdarank_din` | basic | 0.009082 | 0.899101 | 1.126220 | 0.150000 | 0.771462 | 0.807880 | 0.338252 | 0.708609 | 0.326420 |
| `rankmix_lambdarank_din_mmr` | basic | 0.009059 | 0.896911 | 1.120572 | 0.149713 | 0.770011 | 0.805481 | 0.335487 | 0.703920 | 0.327322 |

`full_val` 上如果只按 `NDCG@10` 选择，最优方案是 `rankmix_lambdarank_din`。但项目最终目标是三层推荐链路，因此保留 `rankmix_lambdarank_din_mmr` 作为最终工程 baseline：它牺牲少量 `NDCG@10`、`Utility@10`、`Recall@200`，换取更高的覆盖和更完整的重排层。

相对 clean `full_48gb_optimized` 的同名 `rankmix_lambdarank_din`：

| 指标 | boosted | clean optimized | 差值 |
|---|---:|---:|---:|
| Recall@10 | 0.009082 | 0.009052 | +0.000030 |
| NDCG@10 | 0.899101 | 0.897720 | +0.001381 |
| Utility@10 | 1.126220 | 1.122734 | +0.003487 |
| Recall@200 | 0.150000 | 0.147799 | +0.002201 |
| NDCG@200 | 0.771462 | 0.762409 | +0.009053 |
| Utility@200 | 0.807880 | 0.782918 | +0.024962 |
| Recall@500 | 0.338252 | 0.308448 | +0.029804 |
| NDCG@500 | 0.708609 | 0.662914 | +0.045695 |
| Coverage@200 | 0.326420 | 0.386234 | -0.059814 |

结论：boosted 方案显著提高了相关性和候选上限，但多样性覆盖下降。

## 5. 多行为 DIN 复盘

本轮也测试了多行为 DIN：历史序列额外加入完播、强兴趣、短播、观看比例。它在 `batch_size=65536` 和 `32768` 下发生 OOM，最终使用 `batch_size=16384` 跑通。

多行为 DIN 跑通后的主要结果：

| 模型 | DIN 变体 | NDCG@10 | Recall@200 | NDCG@200 | 结论 |
|---|---|---:|---:|---:|---|
| `rankmix_lambdarank_din` | multibehavior | 0.870425 | 0.146857 | 0.755234 | 明显低于 basic |
| `rankmix_lambdarank_din_mmr` | multibehavior | 0.870027 | 0.146112 | 0.752292 | 明显低于 basic |

判断：多行为信号本身可以讲，但当前实现不适合作为最终 baseline。可能原因是候选池已由强双塔主导，序列行为特征在当前训练目标下没有稳定转化为 top 排序收益。

## 6. full_test 冻结验收

冻结测试使用 `full_val` 固定的参数：basic DIN，`din_weight=0.6`，`pr3_weight=0.4`，`rrf_k=60`，`mmr_lambda=0.9`。报告同时保留未加 MMR 的 RankMix 作为相关性上界对照。

### 6.1 full_test 召回

| 召回通道 | Recall@500 | NDCG@500 | Coverage@500 | Cold Recall@500 | Utility@500 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `itemcf_main` | 0.207636 | 0.434727 | 0.794710 | 0.000000 | 0.404174 | 协同过滤基线 |
| `content_text_category` | 0.168024 | 0.351373 | 0.678088 | 0.201647 | 0.338967 | 冷启动补充 |
| `feature_tower_dropout_hard_negative` | 0.325918 | 0.654386 | 0.606252 | 0.496088 | 0.660865 | 冻结测试仍最强 |

双塔在 `full_test` 上继续强于 ItemCF：`Recall@500 +0.118282`，`NDCG@500 +0.219659`，说明双塔主力策略不是 `full_val` 偶然现象。

### 6.2 full_test 精排

| 模型 | Recall@10 | NDCG@10 | Utility@10 | Recall@200 | NDCG@200 | Utility@200 | Recall@500 | NDCG@500 | Coverage@200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lambdarank_full_features_refit` | 0.009108 | 0.859256 | 1.058573 | 0.152388 | 0.751423 | 0.779186 | 0.342426 | 0.691394 | 0.501653 |
| `din_sequence_ranker_refit` | 0.008845 | 0.851178 | 0.989049 | 0.149280 | 0.739653 | 0.767756 | 0.339624 | 0.686449 | 0.260896 |
| `rankmix_lambdarank_din` | 0.009381 | 0.889421 | 1.103818 | 0.153779 | 0.759932 | 0.790407 | 0.343704 | 0.695711 | 0.356477 |
| `rankmix_lambdarank_din_mmr` | 0.009334 | 0.886832 | 1.096110 | 0.153195 | 0.757705 | 0.786985 | 0.340863 | 0.691192 | 0.360685 |

冻结测试结论：`rankmix_lambdarank_din` 是相关性指标最优对照；最终 boosted 工程 baseline 选择 `rankmix_lambdarank_din_mmr`，用于体现“召回、精排、重排”的三层链路和多样性 trade-off。

## 7. 与 24GB baseline 对比

对比对象：`full_24gb` 的最终展示链路 `rankmix_lambdarank_din_mmr`。

| 指标 | 24GB RankMix+MMR | 48GB boosted RankMix+MMR | 差值 | 结论 |
|---|---:|---:|---:|---|
| Recall@10 | 0.009218 | 0.009334 | +0.000116 | 提升 |
| NDCG@10 | 0.876574 | 0.886832 | +0.010258 | 明显提升 |
| Utility@10 | 1.053530 | 1.096110 | +0.042580 | 明显提升 |
| Recall@200 | 0.139656 | 0.153195 | +0.013539 | 明显提升 |
| NDCG@200 | 0.706562 | 0.757705 | +0.051143 | 明显提升 |
| Utility@200 | 0.700544 | 0.786985 | +0.086441 | 明显提升 |
| Coverage@200 | 0.524497 | 0.360685 | -0.163812 | 下降 |

结论：48GB boosted + MMR 在相关性、召回、效用指标上全面优于 24GB MMR baseline，但覆盖率仍明显低于 24GB。MMR 在 boosted 候选池内确实提升覆盖，但强双塔召回本身更集中，导致整体覆盖不如旧 24GB。

## 8. 最终结论

本轮最终工程 baseline：

```text
召回层：ItemCF + 内容召回 + hard-negative 特征双塔
候选策略：双塔主力，tower 通道配额 1.5x
精排层：LightGBM LambdaRank + basic DIN
融合层：RRF RankMix，din_weight=0.6，pr3_weight=0.4，rrf_k=60
重排层：MMR，lambda=0.9，similarity=0.6 * same_category + 0.4 * same_author
评测协议：full_val 选型，full_test 冻结验收
```

可以写进项目最终结论：

- 增强双塔召回确实超过 ItemCF，可以作为主力候选通道。
- boosted + MMR 把 `full_test Recall@200` 从 24GB MMR 的 0.139656 提升到 0.153195。
- 最终工程 baseline `full_test NDCG@10=0.886832`，相对 24GB MMR baseline 提升 0.010258。
- MMR 进入最终工程 baseline，原因是它提供独立重排层并提升 boosted 候选池内覆盖；需要诚实说明它相对未加 MMR 的 RankMix 会牺牲少量 `NDCG@10` 和 `Utility@10`。
- 多行为 DIN 是探索实验，没有进入最终 baseline。

## 9. Artifact

关键结果文件：

- `artifacts/full_48gb_boosted/recall/results.csv`
- `artifacts/full_48gb_boosted/rank_mix/summary_full_val.csv`
- `artifacts/full_48gb_boosted/rank_mix/results_full_val.json`
- `artifacts/full_48gb_boosted/rank_mix/summary_full_test.csv`
- `artifacts/full_48gb_boosted/rank_mix/results_full_test.json`
- `artifacts/full_48gb_boosted/rank_mix/snapshots/summary_full_val_basic_0604.csv`
- `artifacts/full_48gb_boosted/rank_mix/snapshots/summary_full_test_basic_0604.csv`

运行日志：

- `logs/full_48gb_boosted_20260704_081843.log`
- `logs/full_48gb_boosted_rankmix_basic_0604_20260704_100630.log`
- `logs/full_48gb_boosted_full_test_basic_20260704_101802.log`
