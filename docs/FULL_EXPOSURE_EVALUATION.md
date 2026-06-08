# KuaiRec 全曝光优先评测协议

## 1. 决策

从结果 schema v7 开始，所有模型选型和消融实验统一使用 KuaiRec `small_matrix`
的近全曝光反馈：

- `full_val`：固定的一半 small-matrix 用户，用于模型选择和消融。
- `full_test`：另一半固定用户，只在方案冻结后验收一次。
- `big_matrix`：只用于训练，以及带曝光偏差的时间回放和线上分布诊断。

这样做合理，但不能反复使用完整 small matrix 调参。用户级固定拆分用于防止
benchmark 过拟合，并检验模型对未参与选型用户的泛化。

当前稳定用户哈希切分得到 `717 full_val / 694 full_test`。`local_8gb_large`
固定选择全部 1,411 名全曝光用户，并加入 1,589 名普通用户提供协同过滤训练信号。

## 2. 数据边界

| 环节 | 使用数据 | 是否允许影响模型选择 |
|---|---|---|
| 召回、精排训练 | `big_matrix` 训练日志 | 是 |
| 候选精排负样本 | `big_matrix` 中明确曝光且有反馈的候选 | 是 |
| 消融与超参选择 | `small_matrix/full_val` | 是 |
| 冻结验收 | `small_matrix/full_test` | 否，只运行一次 |
| 时间稳定性诊断 | `big_matrix/valid,test` | 否，只解释分布偏差 |

禁止把 big matrix 中未曝光的召回候选标记为负样本。禁止按日期过滤 small
matrix 后仍将其称为全曝光评测。

## 3. 评测口径

全曝光召回只允许推荐 small matrix 覆盖的 3,327 个视频。对每位用户，模型从
该目录生成 Top-K，然后使用真实反馈计算：

- `Precision@K`、`Recall@K`、`NDCG@K`、`HitRate@K`
- `CompleteRate@K`、`StrongRate@K`、`ShortRate@K`
- `WatchRatio@K`
- `Utility@K = complete + 0.5 * strong - 0.5 * short`
- `Coverage@K`、`ColdRecall@K`、平均流行度

每次结果必须同时保存 `matrix_density`、`evaluated_pairs`、
`invalid_recommendations` 和 `unobserved_recommendation_pairs`，用于证明评测确实
发生在近全曝光目录内。

## 4. 候选精排协议

候选精排训练仍使用 big matrix，因为 small matrix 不进入模型拟合：

- 正样本：召回候选中明确曝光且完播的物品。
- 负样本：召回候选中明确曝光但未完播的物品。
- 未曝光候选：标签未知，不参与监督训练。
- 验证：在 `full_val` 候选上使用 small matrix 真实标签。
- 测试：冻结后在 `full_test` 候选上使用 small matrix 真实标签。

旧 PR3、DeepFM、DIN 结果使用过伪负样本，不能与 schema v7 新结果直接比较。

## 5. 运行顺序

```powershell
python scripts\run_experiment.py --profile local_8gb_large --stage prepare
powershell -ExecutionPolicy Bypass -File scripts\run_ablation.ps1 -Profile local_8gb_large

foreach ($panel in @('valid', 'full_val')) {
  foreach ($exp in @('R1.0', 'R2.4', 'R3.4')) {
    python scripts\run_experiment.py --profile local_8gb_large --stage recall --experiment-id $exp --panel $panel
  }
}

python scripts\run_candidate_ranking.py --profile local_8gb_large --panel full_val --rebuild-cache
python scripts\run_deep_candidate_ranking.py --profile local_8gb_large --panel full_val --models deepfm
python scripts\run_deep_candidate_ranking.py --profile local_8gb_large --panel full_val --models din
python scripts\run_deep_candidate_ranking.py --profile local_8gb_large --panel full_val --models multitask
```

冻结方案后，先为三路召回生成 `full_test` 推荐，再将候选和深度精排命令中的
`--panel` 改为 `full_test`。不要根据 `full_test` 结果回头修改模型。

候选和深度精排结果按面板独立保存为 `summary_full_val.csv` 与
`summary_full_test.csv`，冻结测试不会覆盖验证结果。

## 6. 旧入口状态

以下脚本内部固化了 logged temporal validation，当前已禁用：

- `run_din_ranking_loss.py`
- `run_din_sequence_enhancements.py`
- `run_din_enriched_sequence.py`
- `run_din_oov_content.py`

这些实验后续应迁移为“big matrix 训练 + full_val 选型 + full_test 冻结验收”后再启用。
