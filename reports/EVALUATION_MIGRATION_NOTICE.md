# Evaluation Migration Notice

项目已迁移到结果 schema v7 的全曝光优先评测协议。

迁移前的 Big Temporal、`small_test`、PR3、DeepFM、DIN、MMoE 指标均不能继续用于
选择最佳 baseline，原因包括：

- Big Temporal 属于旧曝光策略回放，不代表全候选真实偏好。
- 旧 `small_test` 按日期过滤，破坏了 small matrix 的近全曝光性质。
- 旧候选精排将未曝光召回候选直接标记为负样本。

当前 `configs/serving_baseline.yaml` 仅保留为待重新验证的工程候选，不代表已经被
全曝光实验确认。新的模型结论必须来自：

1. `small_matrix/full_val` 消融与选型；
2. 冻结后运行一次 `small_matrix/full_test`；
3. Big Temporal 仅作为分布漂移诊断。

协议与命令见 `docs/FULL_EXPOSURE_EVALUATION.md`。
