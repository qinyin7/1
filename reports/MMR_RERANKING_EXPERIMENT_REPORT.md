# MMR 重排实验报告

## 实验设置

- Profile：`full_24gb`。
- 基础排序：`rankmix_lambdarank_din`。
- 重排策略：`rankmix_lambdarank_din_mmr`。
- MMR 参数：`lambda=0.9`，`similarity=0.6 * same_category + 0.4 * same_author`。
- full_val 机器：AutoDL RTX 4090 24GB。
- full_test 机器：AutoDL RTX 3090 24GB。
- 结果文件：
  - `artifacts/full_24gb/rank_mix/summary_full_val.csv`
  - `artifacts/full_24gb/rank_mix/summary_full_test.csv`
  - `reports/mmr_significance/full_val_*.json`
  - `reports/mmr_significance/full_test_*.json`

## full_val 结果

| Exp | Recall@10 | NDCG@10 | Utility@10 | Recall@200 | NDCG@200 | Coverage@200 | Cold Recall@200 | Utility@200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lambdarank_full_features_refit | 0.008591 | 0.863578 | 1.026499 | 0.135553 | 0.710369 | 0.607454 | 0.204731 | 0.708201 |
| din_sequence_ranker_refit | 0.008928 | 0.888260 | 1.086820 | 0.136306 | 0.717888 | 0.415690 | 0.242261 | 0.715683 |
| rankmix_lambdarank_din | 0.008897 | 0.886585 | 1.081381 | 0.137736 | 0.723145 | 0.507665 | 0.233041 | 0.721245 |
| rankmix_lambdarank_din_mmr | 0.008906 | 0.886378 | 1.082357 | 0.137488 | 0.721873 | 0.518485 | 0.220177 | 0.719822 |

full_val 上，MMR 提升 Coverage@200，但 Recall@200 和 NDCG@200 下降。这个结果说明 MMR 的确在做多样性重排，但会带来一定相关性 trade-off。

## full_test 结果

| Exp | Recall@10 | NDCG@10 | Utility@10 | Recall@200 | NDCG@200 | Coverage@200 | Cold Recall@200 | Utility@200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lambdarank_full_features_refit | 0.008970 | 0.856819 | 1.011599 | 0.138076 | 0.697271 | 0.622783 | 0.215410 | 0.690321 |
| din_sequence_ranker_refit | 0.009104 | 0.868664 | 1.038761 | 0.137712 | 0.699075 | 0.429817 | 0.255326 | 0.693534 |
| rankmix_lambdarank_din | 0.009220 | 0.878226 | 1.052522 | 0.139507 | 0.706465 | 0.519988 | 0.247697 | 0.700151 |
| rankmix_lambdarank_din_mmr | 0.009218 | 0.876574 | 1.053530 | 0.139656 | 0.706562 | 0.524497 | 0.237664 | 0.700544 |

full_test 上，MMR 的 Recall@200、NDCG@200、Coverage@200、Utility@10 和 Utility@200 均略高于原始 RankMix；Recall@10 和 NDCG@10 略低，Cold Recall@200 下降。

## 显著性检验

对 `rankmix_lambdarank_din_mmr` 相对 `rankmix_lambdarank_din` 做 paired bootstrap。

full_val：

| Metric | Mean Diff | 95% CI | 结论 |
|---|---:|---|---|
| Recall@10 | +0.000008 | [-0.000024, +0.000043] | 不显著 |
| NDCG@10 | -0.000206 | [-0.002248, +0.002066] | 不显著 |
| Recall@200 | -0.000249 | [-0.000403, -0.000083] | 显著下降 |
| NDCG@200 | -0.001272 | [-0.001844, -0.000700] | 显著下降 |

full_test：

| Metric | Mean Diff | 95% CI | 结论 |
|---|---:|---|---|
| Recall@10 | -0.000002 | [-0.000039, +0.000034] | 不显著 |
| NDCG@10 | -0.001653 | [-0.004158, +0.000527] | 不显著 |
| Recall@200 | +0.000149 | [-0.000011, +0.000321] | 不显著 |
| NDCG@200 | +0.000097 | [-0.000471, +0.000677] | 不显著 |

## 结论

最终采用：

```text
rankmix_lambdarank_din_mmr
```

原因是 MMR 在冻结 `full_test` 上没有显著损伤 Recall/NDCG 主指标，同时提升了 Coverage@200，并且 Utility@10 与 Utility@200 略高。它更适合作为最终展示前的第三层重排策略，而不是替代 DIN 或 LambdaRank 的精排模型。

需要诚实说明的 trade-off：

- `full_val` 上 MMR 的 @200 相关性指标显著下降。
- `full_test` 上 MMR 的 @200 均值略高，但显著性检验不显著。
- `Cold Recall@200` 在 full_val 和 full_test 都下降。
- 因此 MMR 的价值应解释为“多样性/体验层取舍”，不要说成“全指标显著优于 RankMix”。

## 面试表达

可以这样讲：

> 我在 RankMix 后面补充了 MMR 重排层，用类目和作者相似度惩罚重复内容。full_val 显示 MMR 会用少量相关性指标换 Coverage；冻结 full_test 上它没有显著损伤 Recall/NDCG，同时 Coverage@200 和 Utility 略有提升。因此最终链路采用 RankMix + MMR：RankMix 负责相关性排序，MMR 负责最终列表多样性。这是一个真实推荐系统里常见的业务 trade-off，而不是单纯追求离线单指标最高。
