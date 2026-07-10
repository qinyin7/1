# KuaiRec 候选级特征分组说明

本文整理候选级精排使用的特征体系，方便面试讲解、后续消融和 48GB 优化实验记录。

## 1. 总体结构

当前候选级特征分为四类：

```text
召回侧特征
用户侧特征
物品侧特征
交叉侧特征
```

这些特征同时服务于：

- LightGBM LambdaRank；
- DeepFM / DIN / MMoE 等深度精排模型；
- RankMix 融合前的候选排序。

DIN / DeepFM 中，`first_category`、`author_id_hash`、`user_id`、`video_id`、`source_mask`
作为 sparse embedding 特征，其余候选级特征作为 dense features 输入。

## 2. 召回侧特征

召回侧特征描述候选从哪些召回通道来，以及在通道内排得靠不靠前。

| 特征 | 含义 |
|---|---|
| `itemcf_present` | 是否被 ItemCF 召回 |
| `content_present` | 是否被内容召回召回 |
| `tower_present` | 是否被双塔召回召回 |
| `itemcf_rank_score` | ItemCF 通道排名分，`1 / rank` |
| `content_rank_score` | 内容召回通道排名分，`1 / rank` |
| `tower_rank_score` | 双塔通道排名分，`1 / rank` |
| `channel_count` | 候选被几个通道共同命中 |

面试表达：

> 召回侧特征让精排知道候选来源。多通道同时命中的视频通常更可靠，而单通道命中的视频可能代表探索、冷启动或语义补充。

## 3. 用户侧特征

用户侧特征描述用户历史行为强度和偏好倾向。

| 特征 | 含义 |
|---|---|
| `user_interactions` | 用户历史交互数 |
| `user_complete_rate` | 用户历史完播率 |
| `user_strong_rate` | 用户历史强兴趣率 |
| `user_short_rate` | 用户历史短播率 |
| `user_mean_watch_ratio` | 用户平均观看比例 |

面试表达：

> 用户侧特征主要刻画用户整体活跃度和反馈强度。例如高完播率用户和低完播率用户对同一候选视频的分数解释不同。

## 4. 物品侧特征

物品侧特征描述候选视频自身属性、历史表现和新鲜度。

| 特征 | 含义 |
|---|---|
| `first_category` | 视频一级类目 |
| `video_duration` | 视频时长 |
| `author_id_hash` | 作者 ID hash |
| `item_interactions` | 视频历史交互数 |
| `item_complete_rate` | 视频历史完播率 |
| `item_strong_rate` | 视频历史强兴趣率 |
| `item_short_rate` | 视频历史短播率 |
| `item_mean_watch_ratio` | 视频平均观看比例 |
| `is_cold_item` | 是否训练历史未见视频 |
| `item_age_days` | 视频距当前推荐日期的天数 |

面试表达：

> 物品侧特征既包含视频静态属性，也包含历史质量统计和新鲜度。冷启动视频不能简单依赖历史统计，因此需要内容召回、双塔召回和冷启动标记辅助判断。

## 5. 交叉侧特征

交叉侧特征描述“这个用户是否适合这个候选视频”，是本轮从截图项目中借鉴后新增的重点。

### 5.1 原有交叉偏好

| 特征 | 含义 |
|---|---|
| `category_affinity` | 用户对候选视频类目的历史偏好 |
| `author_affinity` | 用户对候选视频作者的历史偏好 |

### 5.2 新增轻量高阶交叉

| 特征 | 公式 | 作用 |
|---|---|---|
| `category_item_complete_cross` | `category_affinity * item_complete_rate` | 用户类目偏好和视频历史质量交叉 |
| `author_item_complete_cross` | `author_affinity * item_complete_rate` | 用户作者偏好和视频历史质量交叉 |
| `channel_item_complete_cross` | `channel_count * item_complete_rate` | 多通道共识和视频质量交叉 |
| `cold_content_cross` | `is_cold_item * content_present` | 冷启动视频是否由内容召回补充 |
| `tower_age_cross` | `tower_present * item_age_days` | 双塔候选与视频新鲜度交叉 |

这些特征的目的不是引入复杂新模型，而是把 DCN/DeepFM 项目里常见的“高阶交叉”思想，用更低成本的显式特征交给 LambdaRank 和 DIN 使用。

面试表达：

> 我没有直接引入更重的 DCN，而是先做轻量显式交叉特征。比如用户对某类目有偏好，同时该视频历史完播率高，这个组合比单独看类目偏好或视频完播率更有信息量。

## 6. 当前特征数量

当前 `FEATURE_COLUMNS` 共 29 个。

DIN / DeepFM dense features 去掉 `first_category` 和 `author_id_hash` 后，共 27 个。

Sparse features 为：

```text
user_id
video_id
first_category
author_id_hash
source_mask
```

其中：

```text
source_mask = itemcf_present + 2 * content_present + 4 * tower_present
```

## 7. 消融建议

如果 48GB 实验中需要验证新特征收益，优先做：

```text
lambdarank_full_features
vs
lambdarank_without_cross_features
```

观察指标：

- `NDCG@10`
- `Utility@10`
- `Recall@200`
- `NDCG@200`
- `Coverage@200`

如果新增交叉特征没有提升，不应强行保留为“有效特征”，可以在报告里作为负实验说明。
