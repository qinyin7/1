# KuaiRec 短视频推荐系统项目

这是一个基于 KuaiRec 2.0 真实短视频曝光日志构建的推荐系统工程项目，覆盖数据分析、全曝光评测、多路召回、候选级精排、DIN 序列模型、RankMix 融合、显著性检验和实验 RUNBOOK。

项目目标不是“堆模型”，而是做出一个能放进简历、能在面试中讲清楚工程取舍的推荐系统 baseline。

## 项目亮点

- 使用 KuaiRec 小矩阵的近全曝光优势重构离线评测，避免把未曝光物品误当负样本。
- 召回层包含 `itemcf_main`、`content_text_category`、`feature_tower_id_dropout` 三路候选。
- 精排层包含 `lambdarank_full_features`、`din_sequence_ranker` 和最终 `rankmix_lambdarank_din`。
- 支持 `Recall@10 / NDCG@10` 展示层指标，以及 `Recall@200 / NDCG@200` 候选层指标。
- 在 RTX 4060 Ti 8GB 上可跑小规模消融，在 RTX 3090/4090 24GB 上可跑完整实验。

## 最终 Baseline

```text
候选召回：
  itemcf_main                    # 协同过滤主召回
  content_text_category           # 类目 + 文本内容召回，补充冷启动
  feature_tower_id_dropout        # 双塔召回，补充表示学习候选

精排与融合：
  din_sequence_ranker             # DIN 序列兴趣模型
  lambdarank_full_features        # LightGBM LambdaRank 表格特征强基线
  rankmix_lambdarank_din          # 0.6 * RRF(DIN) + 0.4 * RRF(LambdaRank)
```

冻结 `full_test` 结果：

| 策略 | Recall@10 | NDCG@10 | Recall@200 | NDCG@200 | Utility@10 |
|---|---:|---:|---:|---:|---:|
| `rankmix_lambdarank_din` | 0.009225 | 0.878409 | 0.139489 | 0.706355 | 1.054035 |

## 快速开始

安装依赖并运行测试：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

准备数据：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1
python scripts\run_experiment.py --profile local_8gb_large --stage prepare
```

运行本机小规模消融：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_ablation.ps1 -Profile local_8gb_large
python scripts\summarize_results.py --profile local_8gb_large
```

运行全曝光召回与候选级精排：

```powershell
python scripts\run_experiment.py --profile full_24gb --stage recall --experiment-id itemcf_main --panel full_val
python scripts\run_experiment.py --profile full_24gb --stage recall --experiment-id content_text_category --panel full_val
python scripts\run_experiment.py --profile full_24gb --stage recall --experiment-id feature_tower_id_dropout --panel full_val
python scripts\run_rank_mix.py --profile full_24gb --panel full_val --cpu-threads 14
```

## 文档入口

- `docs/PROJECT_GUIDE.md`：项目说明书，适合学习、写简历和准备面试。
- `docs/RUNBOOK.md`：从本机小规模实验到 AutoDL 24GB 全量实验的运行手册。
- `docs/FULL_EXPOSURE_EVALUATION.md`：为什么使用近全曝光评测，以及旧评测为什么会偏。
- `docs/ENGINEERING_DESIGN.md`：召回、精排、消融实验和工程方案设计。
- `reports/KUAI_REC_ANALYSIS.md`：数据分析与特征处理建议。
- `reports/FULL_24GB_EXPERIMENT_REPORT.md`：RTX 3090 24GB 全量实验结论。

## 命名说明

仓库对外使用可读实验名，例如 `rankmix_lambdarank_din`。早期实验结果中仍可能出现 `R3.4`、`PR3`、`DR2.din` 等 legacy ID，它们只用于兼容历史 artifact。

核心映射：

| 公开命名 | 历史 ID | 含义 |
|---|---|---|
| `itemcf_main` | `R1.0` | ItemCF 主召回 |
| `content_text_category` | `R2.4` | 类目 + TF-IDF 内容召回 |
| `feature_tower_id_dropout` | `R3.4` | 特征双塔 + ID Dropout |
| `lambdarank_full_features` | `PR3` | LightGBM LambdaRank |
| `din_sequence_ranker` | `DR2.din` | DIN 序列兴趣模型 |
| `rankmix_lambdarank_din` | `DR4.rank_mix` | RRF 排名融合 |
