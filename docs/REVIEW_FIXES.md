# Code Review 修复记录

| 审查问题 | 修复方式 | 验证 |
|---|---|---|
| 秒级时间戳被按毫秒解析 | 优先解析本地 `time`；缺失时使用秒级 Unix 时间转上海时区，最后按日期中午回退 | 本地验证小时匹配率 100%，覆盖 24 小时与 7 个星期值 |
| Recall Ground Truth 包含被过滤的 seen item | Ground Truth 构造时同步排除训练期 seen item | 验证 Ground Truth 与 seen 交集为 0 |
| Content Recall 包含未来视频 | Item 特征保存 `first_seen_date`，召回按用户-天过滤可用目录 | 验证 Content 候选中未来视频数为 0 |
| 空推荐用户被跳过 | 空推荐组按 Recall/HitRate/NDCG 为 0 计入 | 自动化测试覆盖 |
| 小规模 profile 随机删除用户历史 | 仅抽样用户，保留所选用户全部交互 | `preserves_complete_user_histories=true` |
| 视频时长填充错误索引对齐 | 使用 `video_id.map(duration_by_item)` | 未来视频填充样本 mismatch 为 0 |
| Item 统计验证集几乎全缺失 | 按每个样本日期仅使用历史行为构造滚动统计 | 本地验证中 Item 统计缺失率约 22.46% |
| ItemCF 多反馈消融无实际差异 | 强兴趣与短播进入反馈权重和共现权重 | R1.3 与 R1.0/R1.2 产生不同结果 |
| `experiments.yaml` 未驱动实验 | `run_suite.py` 读取启用实验；未实现模型明确 `enabled: false` | suite 可按配置运行启用的召回与精排实验 |
| 结果 CSV 被不同 schema 追加损坏 | 每次运行独立 JSON，按当前 schema 自动重建 CSV | 结果表可稳定读取，旧 schema 自动退出正式汇总 |
| 不同种子改变验证用户子集 | 训练抽样使用模型 seed，评估用户固定使用 `data_seed` | 两种子 `valid_rows` 一致 |
| 缺少显著性检验 | 保存逐用户-天指标并提供 paired bootstrap 脚本 | 对比脚本已运行 |
| 缺少 Test 与小矩阵审计 | 增加冻结后评测面板 | 评测面板已运行 |

## 当前边界

- 大矩阵没有原始请求候选集或未曝光物品反馈，精排 baseline 使用相同 Logged Exposure 用户-天组进行公平比较。
- R3.0 是明确标注的 ID-only 双塔；画像与内容增强双塔仍处于禁用状态。
- DeepFM、DIN、多任务精排、TF-IDF 内容召回仍为后续扩展，不会进入当前 suite。
