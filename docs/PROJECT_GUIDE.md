# KuaiRec 短视频推荐系统项目说明书

## 1. 引言

这个项目基于 KuaiRec 2.0 数据集，实现了一套两阶段短视频推荐系统：

```text
原始日志 -> 数据清洗与特征构造 -> 多路召回 -> 候选级精排 -> RankMix 融合 -> 全曝光评测
```

它最适合用来学习搜广推工程里的三个核心问题：

- 召回层如何同时兼顾协同过滤、内容冷启动和双塔表示学习；
- 精排层如何比较 LambdaRank、DIN 和融合策略；
- 离线评测如何利用 KuaiRec 的近全曝光数据，避免传统稀疏日志评测的曝光偏差。

项目最终不是单个 notebook，而是一套可以复现实验的工程：有配置、有脚本、有缓存、有测试、有 RUNBOOK、有显著性检验，也有适合写进简历的实验结论。

## 2. 拿到这个项目，怎么学？

### 2.1 路线一：突击备战

适合时间紧、近期要面试的情况。目标是先能讲清楚项目，而不是一上来改模型。

第一步，读 `README.md` 和本说明书，理解最终 baseline：

```text
itemcf_main + content_text_category + feature_tower_id_dropout
  -> rankmix_lambdarank_din
```

第二步，读 `docs/FULL_EXPOSURE_EVALUATION.md`，重点掌握为什么 KuaiRec 不能继续用传统“留出一部分交互当测试集”的方式评测。

第三步，读 `reports/FULL_24GB_EXPERIMENT_REPORT.md`，记住三个面试结论：

- 双塔是最强单路召回，但不能单独解决排序问题；
- DIN 在 Top10 的 NDCG 和冷启动候选利用率上优于单纯表格模型；
- RankMix 通过排名融合稳定提升 DIN 和 LambdaRank，且通过 paired bootstrap 显著性检验。

第四步，按 `docs/RUNBOOK.md` 跑 `local_8gb_large` 小规模实验，确保自己能从命令行复现项目。

### 2.2 路线二：扎实学习

适合想真正掌握搜广推工程的情况。

第 1 天：跑通数据和测试。

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python scripts\run_experiment.py --profile local_8gb_large --stage prepare
```

第 2 天：读数据分析和评测协议。

- `reports/KUAI_REC_ANALYSIS.md`：数据分布、特征处理、冷启动现象；
- `docs/FULL_EXPOSURE_EVALUATION.md`：全曝光评测、曝光偏差、冻结测试；
- `docs/ENGINEERING_DESIGN.md`：召回和精排工程设计。

第 3 天：跑小规模消融。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_ablation.ps1 -Profile local_8gb_large
python scripts\summarize_results.py --profile local_8gb_large
```

第 4 天：理解全量实验。

- 看三路召回各自解决什么问题；
- 看 DIN 为什么不是简单替代 LambdaRank；
- 看 RankMix 为什么用 RRF，而不是直接 raw score 相加；
- 看显著性检验如何证明提升不是随机波动。

第 5 天：准备面试讲法和简历表达。

重点不是背指标，而是讲清楚“为什么这么设计、实验怎么证明、失败实验说明了什么”。

## 3. 简历上怎么写？

可以写成下面这种形式：

```text
KuaiRec 短视频推荐系统：基于 1253 万条真实交互日志和近全曝光小矩阵，构建多路召回 + DIN/LambdaRank 精排 + RankMix 融合的两阶段推荐系统。
```

可展开的项目点：

- 设计近全曝光离线评测协议，将 `full_val` 用于模型选择、`full_test` 用于冻结验收，避免传统稀疏日志评测中的曝光偏差。
- 实现 ItemCF、内容召回和特征双塔三路召回，双塔使用 ID Dropout 缓解视频 ID 记忆和 OOV 问题，并通过批量 embedding 矩阵 TopK 提升推理效率。
- 构建候选级排序特征，包括召回来源、通道排名、用户统计、视频统计、作者偏好、类目偏好、内容新鲜度和冷启动标记。
- 对比 LightGBM LambdaRank、DIN 序列模型和 RankMix 融合，最终采用 `rankmix_lambdarank_din`。
- 使用用户级 paired bootstrap 显著性检验，验证 RankMix 相对 DIN 在 `Recall@10`、`NDCG@10`、`Recall@200`、`NDCG@200` 上均显著提升。

## 4. 怎么介绍给面试官？

可以按这条主线讲：

第一，KuaiRec 的特殊价值是有近全曝光小矩阵。传统推荐离线评测经常把“没出现在日志里的物品”当负样本，但这其实混淆了“用户不喜欢”和“系统没曝光”。KuaiRec 小矩阵接近全曝光，所以更适合做可靠的离线评测。

第二，召回层不是只追求最高 Recall，而是让不同通道承担不同职责：

| 公开命名 | 作用 | 适合解决的问题 |
|---|---|---|
| `itemcf_main` | 协同过滤主召回 | 用户行为相似、热门内容覆盖 |
| `content_text_category` | 类目 + 文本召回 | 新视频、冷启动、内容相似 |
| `feature_tower_id_dropout` | 双塔召回 | 用户和内容表示学习、补充额外候选 |

第三，精排层先建立强表格基线，再加入 DIN。LambdaRank 很适合候选排序和表格特征，DIN 更适合表达用户最近兴趣序列。最终不是让一个模型吃掉另一个模型，而是用 RRF 排名融合做 RankMix，降低不同模型 raw score 尺度不一致的问题。

第四，项目里有失败实验和取舍。比如 DIN 单独不一定全面超过 LambdaRank，但它在序列兴趣、冷启动候选利用和 Top10 质量上有价值；双塔单路排序不一定最好，但它能带来候选覆盖和冷启动补充。

## 5. KuaiRec 数据集是什么？

本项目使用 KuaiRec 2.0 的核心文件：

- `big_matrix.csv`：大规模真实曝光交互，适合训练召回和排序模型；
- `small_matrix.csv`：近全曝光小矩阵，适合做更可靠的离线评测；
- `user_features.csv`：用户画像；
- `item_categories.csv`、`kuairec_caption_category.csv`：视频内容和类目；
- `item_daily_features.csv`：视频每日统计特征；
- `social_network.csv`：社交关系，覆盖有限，项目中主要作为可扩展方向。

关键事实：

| 数据事实 | 工程影响 |
|---|---|
| 大矩阵是稀疏曝光日志 | 未曝光物品不能直接当负样本 |
| 小矩阵接近全曝光 | 适合作为模型选择和冻结测试 |
| 短视频有新鲜度和冷启动 | 必须报告 Cold Recall 和 item age |
| watch ratio 极长尾 | 需要截断、分桶或构造多行为标签 |
| 用户兴趣会漂移 | 需要时间切分和冻结测试 |

## 6. 推荐系统难点

强时效性与兴趣漂移：短视频兴趣变化快，7 月、8 月和 9 月数据分布并不完全一致。项目采用时间块切分，并要求 `full_test` 只做最终验收。

冷启动问题：测试期会出现训练期未见或很少见的视频。`content_text_category` 和 `feature_tower_id_dropout` 用内容特征和表示学习补充协同过滤不足。

隐式反馈噪声：播放完成、强兴趣、短播和观看比例都不是完美标签。项目里将 `watch_ratio >= 1` 作为主目标，同时保留强兴趣、短播和观看比例用于 DIN 扩展实验。

多路召回分数不可比：ItemCF、内容召回和双塔的分数尺度不同，因此候选级精排使用召回来源和通道排名特征，而 RankMix 使用 RRF 排名融合，不直接相加 raw score。

候选规模与精排压力：召回每路取 Top150，合并去重后进入精排。最终既看 Top10 展示质量，也看 Top200 候选层保留能力。

## 7. 数据集怎么划分？

项目使用两套数据角色：

| 数据面板 | 用途 |
|---|---|
| `train` | 训练召回、排序和序列模型 |
| `valid` | 构建候选级排序训练样本 |
| `full_val` | 近全曝光模型选择和消融 |
| `full_test` | 冻结测试，只用于最终验收 |
| `big_temporal_diagnostic` | 带曝光偏差的时间诊断，不作为最终选型依据 |

核心原则：

- 不用 `full_test` 调参；
- 不把未曝光物品当确定负样本；
- 候选级排序训练只使用已曝光反馈构造正负样本；
- 选型结论必须优先看近全曝光面板。

## 8. 正负样本怎么选？

召回阶段：

- 正样本主要来自完播行为；
- ItemCF 使用完播历史构造共现；
- 双塔使用用户历史和视频内容特征学习向量；
- 内容召回使用类目、标题/描述文本 TF-IDF 近邻。

精排阶段：

- 正样本：候选集中当天真实完播的视频；
- 负样本：候选集中当天已曝光但未完播的视频；
- 不把“召回到了但当天没有曝光记录”的视频直接标成负样本；
- 训练时限制每个用户-天候选数量，避免少数高活跃用户主导训练。

特征处理建议：

| 特征类型 | 处理方式 |
|---|---|
| 用户统计 | 只用预测时点之前的历史，避免标签泄漏 |
| 视频统计 | 使用滞后统计，不能用当天未来信息 |
| 类目/作者偏好 | 基于用户历史完播分布计算 affinity |
| watch ratio | 截断或分桶，避免极端值支配训练 |
| 视频 ID | 双塔中使用 ID Dropout，缓解过拟合 |
| 新视频 | 使用内容表示和冷启动标记补充 |

## 9. 评价指标怎么看？

展示层指标：

| 指标 | 含义 |
|---|---|
| `Recall@10` | Top10 是否覆盖用户真实喜欢的视频 |
| `NDCG@10` | Top10 排序质量，越靠前越重要 |
| `Utility@10` | 综合完播、强兴趣和短播惩罚 |

候选层指标：

| 指标 | 含义 |
|---|---|
| `Recall@200` | 精排前候选保留能力 |
| `NDCG@200` | 候选层排序质量 |
| `Coverage@200` | 推荐覆盖的视频多样性 |
| `Cold Recall@200` | 对冷启动视频的覆盖能力 |

为什么同时看 `@10` 和 `@200`？

`@10` 更接近用户最终看到的展示位，适合衡量推荐质量。`@200` 更像精排候选池能力，适合判断召回和粗排有没有把潜在好内容保留下来。最终线上不一定展示 200 个，而是用它评估候选层是否足够可靠。

## 10. 运行结果

### 10.1 三路召回独立效果

| Panel | 召回通道 | Recall@200 | NDCG@200 | Cold Recall@200 | 结论 |
|---|---|---:|---:|---:|---|
| `full_val` | `itemcf_main` | 0.092415 | 0.493586 | 0.000000 | 协同主召回 |
| `full_val` | `content_text_category` | 0.065363 | 0.353986 | 0.080218 | 冷启动补充 |
| `full_val` | `feature_tower_id_dropout` | 0.129288 | 0.673765 | 0.274715 | 最强单路召回 |
| `full_test` | `itemcf_main` | 0.093114 | 0.480963 | 0.000000 | 冻结测试稳定 |
| `full_test` | `content_text_category` | 0.065112 | 0.341876 | 0.078033 | 冷启动稳定 |
| `full_test` | `feature_tower_id_dropout` | 0.130753 | 0.656944 | 0.284699 | 保留为主候选通道 |

### 10.2 精排与融合效果

| Panel | 策略 | Recall@10 | NDCG@10 | Recall@200 | NDCG@200 | 结论 |
|---|---|---:|---:|---:|---:|---|
| `full_val` | `lambdarank_full_features_refit` | 0.008681 | 0.867396 | 0.135728 | 0.712195 | 强表格基线 |
| `full_val` | `din_sequence_ranker_refit` | 0.008822 | 0.880864 | 0.135536 | 0.714261 | 序列兴趣有效 |
| `full_val` | `rankmix_lambdarank_din` | 0.008896 | 0.886979 | 0.137131 | 0.721257 | 选型胜者 |
| `full_test` | `lambdarank_full_features_refit` | 0.008970 | 0.856819 | 0.138076 | 0.697271 | CPU fallback |
| `full_test` | `din_sequence_ranker_refit` | 0.009116 | 0.868332 | 0.137707 | 0.698952 | 序列模型主干 |
| `full_test` | `rankmix_lambdarank_din` | 0.009225 | 0.878409 | 0.139489 | 0.706355 | 最终 baseline |

### 10.3 显著性检验

比较对象：`rankmix_lambdarank_din - din_sequence_ranker_refit`。

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

## 11. 工程结构

```text
kuairec/
├── configs/                 # 实验配置和最终 serving baseline
├── data/                    # 原始数据和处理后 parquet，不建议提交 Git
├── docs/                    # 设计文档、RUNBOOK、评测协议和项目说明书
├── reports/                 # 数据分析与实验报告
├── scripts/                 # 可执行实验脚本
├── src/                     # 数据、召回、排序、评测和公共工具
├── tests/                   # 单元测试
└── README.md
```

关键脚本：

| 脚本 | 作用 |
|---|---|
| `scripts/run_experiment.py` | 数据准备、召回实验、基础排序实验 |
| `scripts/run_candidate_ranking.py` | 候选级 LambdaRank 消融 |
| `scripts/run_deep_candidate_ranking.py` | DeepFM、DIN、MMoE 精排 |
| `scripts/run_rank_mix.py` | DIN + LambdaRank 的 RankMix 融合 |
| `scripts/compare_experiments.py` | paired bootstrap 显著性检验 |
| `scripts/benchmark_batched_topk.py` | 双塔批量 TopK 性能测试 |

## 12. 当前项目是否完整？

从简历项目角度，已经完整：

- 有真实数据分析；
- 有工程设计；
- 有召回、精排、DIN、双塔和融合；
- 有全曝光评测协议；
- 有本机小规模和 24GB 全量实验路径；
- 有冻结测试结果；
- 有显著性检验；
- 有 RUNBOOK 和结果表格。

后续如果继续增强，优先级建议如下：

- 增加线上服务 Demo：FastAPI + 简单召回缓存 + TopK 推理接口；
- 增加 MLflow 或 SQLite 实验追踪；
- 增加特征重要性可视化；
- 增加 Dockerfile，方便在 AutoDL/4090 环境复现；
- 对 DIN 做 listwise 或 pairwise 目标的重新验证，但不要继续用 `full_test` 调参。
