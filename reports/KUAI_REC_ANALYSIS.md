# KuaiRec 数据分析与搜广推项目设计

## 1. 项目定位

KuaiRec 是来自快手短视频推荐日志的公开数据集。它最有价值的地方不是数据量本身，而是同时提供了：

- `big_matrix`：真实推荐系统曝光日志，适合研究召回、排序、长尾、时间切分和曝光偏置。
- `small_matrix`：密度接近 100% 的用户-物品反馈矩阵，适合研究无偏评估、反事实评估和 OPE。
- 用户画像、视频类目、标题文本、社交网络和视频日粒度统计等丰富侧信息。

因此，建议将简历项目定位为：

> 基于 KuaiRec 的多路召回、精排与无偏离线评估系统

这比“用某个模型预测点击率”更完整，也更能体现搜广推工程思维。

## 2. 数据版本与完整性

- 官方 Zenodo 版本发布日期：2026-01-06。
- 核心压缩包：`KuaiRec.zip`，MD5 为 `261550d472c48eff4990fb13c0e5bcf7`。
- 新增原始视频类目：`video_raw_categories_multi.csv`。
- 新增原始用户画像：`user_features_raw.csv`。
- 本项目已完成下载、MD5 校验与解压。

运行方式：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_data.ps1
python -m pip install -r requirements.txt
python scripts/analyze_kuairec.py
```

## 3. 核心数据结论

| 指标 | big_matrix | small_matrix | 含义 |
|---|---:|---:|---|
| 交互数 | 12,530,806 | 4,676,570 | 都可支持完整推荐实验 |
| 用户数 | 7,176 | 1,411 | 小矩阵用户属于大矩阵用户 |
| 视频数 | 10,728 | 3,327 | 小矩阵视频属于大矩阵视频 |
| 矩阵密度 | 16.28% | 99.62% | 小矩阵接近全观测 |
| 完播率 `watch_ratio >= 1` | 33.82% | 32.40% | 可作为主排序标签之一 |
| 强正反馈率 `watch_ratio >= 2` | 7.47% | 4.65% | 可作为高价值目标 |
| 短播率 | 18.41% | 7.76% | 大矩阵负反馈更明显 |
| `watch_ratio` 中位数（采样） | 0.723 | 0.769 | 大部分播放未完整看完 |
| `watch_ratio` P99（采样） | 4.791 | 3.656 | 标签明显右偏 |
| `watch_ratio` 最大值 | 573.46 | 571.52 | 存在循环播放/异常极值 |

关键观察：

1. 大矩阵 Top 20% 视频贡献 66.39% 交互，Top 5% 视频贡献 23.35%，存在明显流行度偏置。
2. 小矩阵 Top 20% 视频只贡献 20.09% 交互，近似均匀曝光，可作为大矩阵模型的无偏测试集。
3. 两张矩阵的日期范围均为 2020-07-05 至 2020-09-05，适合严格时间切分。
4. `watch_ratio` 极端右偏，直接做 MSE 回归会被少量循环播放样本主导。
5. 原始用户特征对大矩阵用户覆盖率为 100%，冷启动和画像建模条件较好。

![矩阵密度与完播率](figures/matrix_density_and_completion.png)

![物品流行度集中度](figures/item_popularity_concentration.png)

## 4. 数据表与适用场景

| 数据表 | 主要内容 | 推荐用途 | 主要风险 |
|---|---|---|---|
| `big_matrix.csv` | 真实曝光、播放时长、时间 | 召回/排序训练、时序验证、曝光偏置分析 | 只观察到已曝光物品的反馈 |
| `small_matrix.csv` | 近全观测用户-物品反馈 | 无偏测试集、OPE、偏置校准 | 曝光机制与线上流量不同 |
| `user_features_raw.csv` | 性别、年龄、设备、地域、安装 App | 冷启动、用户塔、分群分析 | 高基数、隐私与公平性风险 |
| `user_features.csv` | 编码后的用户特征 | DeepFM/双塔/排序模型 | 加密字段语义不可解释 |
| `item_categories.csv` | 视频多标签 ID | ItemCF、内容召回、Embedding | 列表字段需多热化 |
| `video_raw_categories_multi.csv` | 多级类目、概率、父子关系 | 层级类目建模、冷启动、多任务学习 | 同一视频多行，需聚合 |
| `kuairec_caption_category.csv` | 标题、话题、三级类目 | 文本向量召回、语义特征 | 文本清洗、缺失、超长字段 |
| `item_daily_features.csv` | 曝光、播放、点赞、评论等日统计 | 热度、趋势、质量、多目标排序 | 极易发生时间泄漏 |
| `social_network.csv` | 用户好友列表 | 社交召回、图推荐 | 仅覆盖 472 个用户，较稀疏 |

## 5. 特征工程建议

### 5.1 标签设计

建议不要只做一个标签。短视频推荐通常是多目标问题。

| 标签 | 构造方式 | 适合任务 | 处理建议 |
|---|---|---|---|
| 完播 | `watch_ratio >= 1` | 二分类排序 | 作为主任务，样本相对均衡 |
| 强兴趣 | `watch_ratio >= 2` | 高价值二分类 | 正样本稀疏，使用加权 BCE/Focal Loss |
| 有效播放 | 视频短于 7 秒则播完，否则播放超过 7 秒 | 二分类排序 | 比完播更适合长视频 |
| 短播负反馈 | `play_duration < min(3s, video_duration)` | 负反馈任务 | 可作为多任务模型的负目标 |
| 播放比例 | `watch_ratio` | 回归/序数学习 | 建议 `clip` 后做 `log1p`，不要直接拟合原值 |

推荐的连续标签处理：

```text
watch_ratio_clipped = min(watch_ratio, 5)
watch_ratio_log = log1p(watch_ratio_clipped)
```

也可以把播放比例分桶为 `<0.25`、`0.25-0.5`、`0.5-1`、`1-2`、`>=2`，做序数分类。

### 5.2 用户特征

| 特征类型 | 示例 | 推荐处理 |
|---|---|---|
| 低基数类别 | 性别、年龄段、活跃度、平台 | Embedding 或 One-hot；缺失单独成桶 |
| 高基数类别 | 手机型号、城市、下载渠道 | Hash Embedding、频次截断、低频合并为 `OTHER` |
| 数值计数 | 粉丝数、关注数、好友数、注册天数、手机价格 | `log1p`、分桶、标准化 |
| App 安装信号 | 是否安装抖音/斗鱼/虎牙等 | 多热二值特征，可表达跨产品兴趣 |
| 地域层级 | 国家区域、省、市、城市等级 | 层级 Embedding，不建议把城市当有序数值 |
| 社交关系 | 好友列表 | GraphSAGE/LightGCN 或好友兴趣聚合 |

用户画像适合：

- 双塔召回中的 User Tower。
- 冷启动用户表示。
- 用户分群和分群指标诊断。
- 排序模型中的稀疏类别特征。

注意：性别、年龄、地域等特征应做公平性分群评估，不应只追求总体指标。

### 5.3 视频与内容特征

| 特征类型 | 示例 | 推荐处理 |
|---|---|---|
| 视频 ID/作者 ID/音乐 ID | `video_id`、`author_id`、`music_id` | Embedding；低频 ID 共享 OOV |
| 视频时长 | `video_duration` | `log1p`、分桶，并与完播目标做交叉 |
| 多级类目 | root/parent/category | 分层 Embedding、父子约束、多任务预测 |
| 多标签 | `feat`、`topic_tag` | 多热池化、Attention Pooling |
| 类目置信度 | `prob` | 对类目 Embedding 加权池化 |
| 标题文本 | caption、封面文字 | TF-IDF/BM25 基线，再升级中文预训练模型向量 |
| 视频尺寸 | width、height | 构造横竖屏、宽高比 |

视频内容适合：

- 内容召回和冷启动视频召回。
- Item Tower 表示。
- 相似视频检索。
- 类目多样性重排。

`video_raw_categories_multi.csv` 存在 55 条完全重复行，聚合前应先去重。

### 5.4 上下文与序列特征

交互日志包含精确时间戳，可构造：

- 小时、星期、工作日/周末。
- 用户最近 5/20/50 个视频和类目序列。
- 用户最近一次交互距当前的时间差。
- 最近窗口内完播率、短播率、类目偏好分布。
- 作者/类目重复曝光次数和疲劳度。
- 用户长期兴趣与短期兴趣的差异。

这些特征适合 DIN、DIEN、Transformer 序列模型或简单的统计排序模型。

必须按时间构造，只能使用当前交互之前的信息。

### 5.5 日聚合统计特征

`item_daily_features.csv` 包含丰富的曝光后指标，例如曝光、播放、完播、点赞、评论、关注、分享和负反馈。

建议派生：

- `play_rate = play_cnt / show_cnt`
- `complete_rate = complete_play_cnt / play_cnt`
- `like_rate = like_cnt / play_cnt`
- `comment_rate = comment_cnt / play_cnt`
- `follow_rate = follow_cnt / play_cnt`
- `negative_feedback_rate = reduce_similar_cnt / play_cnt`
- 近 1/3/7 日趋势、同比变化、指数衰减热度

重要：预测日期 `D` 的样本只能使用 `D-1` 及以前的统计。直接连接同一天完整统计，会把未来行为泄漏给模型。对分母很小的比例应使用贝叶斯平滑：

```text
smoothed_rate = (positive_count + alpha * global_rate) / (exposure_count + alpha)
```

部分原始比率可能大于 1，例如重复播放导致 `play_cnt > show_cnt`。建模前需要明确业务定义，不应盲目截断。

## 6. 数据清洗与切分方案

推荐清洗步骤：

1. 检查并删除完全重复行；原始视频类目表已发现 55 条重复行。
2. 将列表字符串安全解析为数组，并对多标签做去重。
3. 将缺失类别填为 `UNKNOWN`，不要用众数掩盖缺失机制。
4. 对 `watch_ratio`、粉丝数、播放量等长尾数值做 `clip + log1p`。
5. 保留原始值用于分析，另建模型特征列，避免不可逆覆盖。
6. 所有统计特征都按事件时间左连接，严禁使用未来数据。

推荐时间切分：

- 训练集：2020-07-05 至 2020-08-25
- 验证集：2020-08-26 至 2020-08-31
- 测试集：2020-09-01 至 2020-09-05

不要随机切分交互。随机切分会把用户未来兴趣和物品未来热度泄漏到训练集。

## 7. 可落地的推荐系统方案

### 阶段一：可解释基线

- 热门召回：按过去 3/7 日平滑热度。
- ItemCF：基于用户交互序列计算物品相似度。
- 类目召回：按用户历史完播类目召回。
- 排序：LightGBM/XGBoost，预测完播和强兴趣。
- 指标：AUC、LogLoss、NDCG@K、Recall@K、Coverage@K。

### 阶段二：工业化多路召回与精排

- 双塔召回：用户画像 + 历史序列，对视频 ID + 类目 + 文本向量。
- 图召回：LightGCN 建模用户-视频图；社交关系作为附加边。
- 精排：DeepFM/DIN，多任务预测完播、强兴趣和短播。
- 融合：对多路候选做归一化、去重和配额控制。
- 重排：MMR 或类目配额改善多样性，降低热门挤压。

### 阶段三：KuaiRec 特色实验

使用 `big_matrix` 训练模型，再用 `small_matrix` 做近全观测评估：

1. 对比普通曝光日志测试集与全观测测试集的指标差异。
2. 分析热门模型是否在偏置测试集上被高估。
3. 尝试 IPS/SNIPS/DR 等反事实评估方法。
4. 报告准确率、覆盖率、新颖度、流行度偏置和不同用户群体指标。

这部分是项目最有研究价值、也最容易与普通推荐项目拉开差距的实验。

## 8. 指标体系

离线指标建议分四组：

| 维度 | 指标 |
|---|---|
| 排序效果 | AUC、LogLoss、GAUC |
| Top-K 效果 | Recall@K、NDCG@K、HitRate@K |
| 系统健康度 | Coverage@K、Novelty、ILD、多样性 |
| 去偏与公平 | Popularity Bias、分群 Recall、全观测集指标、IPS/SNIPS |

GAUC 应按用户交互数加权；只有正样本或只有负样本的用户不能计算用户 AUC，需要明确处理规则。

## 9. 简历表述模板

项目名称：

> KuaiRec 多路召回、精排与无偏离线评估推荐系统

可在完成对应实验后使用以下表述，数字必须替换为你的真实实验结果：

- 基于 1,253 万条真实短视频交互构建可复现推荐数据管线，采用分块聚合处理 GB 级 CSV，完成用户、物品、序列与日粒度统计特征工程，并通过时点左连接规避未来信息泄漏。
- 搭建热门、ItemCF、内容与双塔多路召回，使用多任务排序模型联合预测完播、强兴趣与短播负反馈，通过候选融合和多样性重排提升 `Recall@50`、`NDCG@10` 与目录覆盖率。
- 利用密度 99.62% 的全观测矩阵构建无偏测试集，对比普通离线评估与 IPS/SNIPS 评估，量化曝光偏置和流行度偏置对推荐指标的影响。

面试时要能回答：

- 为什么不能随机切分？
- 为什么 `item_daily_features` 容易泄漏？
- 为什么大矩阵适合训练、小矩阵适合无偏评估？
- `watch_ratio` 为什么不能直接做普通回归？
- 如何处理负采样、位置偏置、热门偏置与冷启动？
- 精度提升和覆盖率下降冲突时如何权衡？

## 10. 下一步实施顺序

1. 建立严格时间切分和统一离线评估器。
2. 完成热门召回、ItemCF、内容召回三个基线。
3. 训练 LightGBM/DeepFM 排序基线，记录完整实验表。
4. 加入双塔召回和 DIN 序列排序。
5. 用小矩阵做全观测评估与 IPS/SNIPS 对比。
6. 增加 FastAPI 推理接口、FAISS 向量索引和简单推荐演示页。

分析脚本生成的完整统计位于 `reports/summary.json`，细分统计位于 `reports/tables/`。

## 11. 数据来源

- KuaiRec GitHub：<https://github.com/chongminggao/KuaiRec>
- 官方 Zenodo：<https://zenodo.org/records/18164998>
- 论文：Gao et al., *KuaiRec: A Fully-Observed Dataset and Insights for Evaluating Recommender Systems*, CIKM 2022.
