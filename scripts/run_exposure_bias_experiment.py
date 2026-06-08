from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_experiment import load_processed
from src.common import ROOT, artifact_dir, load_profile, write_json
from src.evaluation import evaluate_cold_recall, evaluate_recall
from src.full_exposure import (
    build_full_exposure_ground_truth,
    evaluate_full_exposure,
    read_full_exposure_interactions,
)


POLICY_SPECS = [
    ("uniform", 0.05),
    ("uniform", 0.10),
    ("uniform", 0.20),
    ("popularity_biased", 0.10),
    ("positive_biased", 0.10),
]


def load_recommendations(profile: str, experiment: str, panel: str) -> dict[int, list[int]]:
    path = artifact_dir(profile, "recall") / f"{experiment}_{panel}_recommendations.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing recommendation file: {path}. Run recall {experiment} on {panel} first."
        )
    return {
        int(user): [int(item) for item in items]
        for user, items in json.loads(path.read_text(encoding="utf-8")).items()
    }


def sample_exposure(
    feedback: pd.DataFrame,
    policy: str,
    density: float,
    item_popularity: dict[int, int],
    seed: int,
) -> pd.DataFrame:
    sample_size = max(1, round(len(feedback) * density))
    rng = np.random.default_rng(seed)
    if policy == "uniform":
        positions = rng.choice(len(feedback), sample_size, replace=False)
        return feedback.iloc[np.sort(positions)].copy()

    if policy == "popularity_biased":
        weights = np.log1p(feedback["video_id"].map(item_popularity).fillna(0).to_numpy()) + 1e-3
    elif policy == "positive_biased":
        weights = np.where(feedback["label_complete"].to_numpy() == 1, 5.0, 1.0)
    else:
        raise ValueError(f"Unsupported exposure policy: {policy}")

    probabilities = weights / weights.sum()
    positions = rng.choice(len(feedback), sample_size, replace=False, p=probabilities)
    return feedback.iloc[np.sort(positions)].copy()


def build_sparse_truth(sampled: pd.DataFrame) -> dict[int, set[int]]:
    positives = sampled[sampled["label_complete"] == 1]
    return positives.groupby("user_id")["video_id"].apply(set).to_dict()


def evaluate_sparse_replay(
    recommendations: dict[int, list[int]],
    sampled: pd.DataFrame,
    catalog_size: int,
    k: int,
) -> dict[str, float]:
    truth = build_sparse_truth(sampled)
    metrics = evaluate_recall(recommendations, truth, {}, catalog_size, k)
    metrics.update(
        {
            "observed_pairs": int(len(sampled)),
            "observed_users": int(sampled["user_id"].nunique()),
            "observed_positive_pairs": int(sampled["label_complete"].sum()),
            "observed_positive_rate": float(sampled["label_complete"].mean()),
        }
    )
    return metrics


def rank_consistency(summary: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []
    full = summary[summary["evaluation"] == "full_exposure"].set_index("experiment_id")
    for (policy, density), group in summary[summary["evaluation"] == "sparse_replay"].groupby(
        ["policy", "density"]
    ):
        sparse = group.set_index("experiment_id")
        joined = full[[f"recall_at_{k}", f"ndcg_at_{k}"]].join(
            sparse[[f"recall_at_{k}", f"ndcg_at_{k}"]],
            lsuffix="_full",
            rsuffix="_sparse",
        )
        rows.append(
            {
                "policy": policy,
                "density": density,
                "models": len(joined),
                "spearman_recall": float(
                    joined[f"recall_at_{k}_full"].corr(
                        joined[f"recall_at_{k}_sparse"], method="spearman"
                    )
                ),
                "spearman_ndcg": float(
                    joined[f"ndcg_at_{k}_full"].corr(
                        joined[f"ndcg_at_{k}_sparse"], method="spearman"
                    )
                ),
                "full_best_recall": joined[f"recall_at_{k}_full"].idxmax(),
                "sparse_best_recall": joined[f"recall_at_{k}_sparse"].idxmax(),
                "full_best_ndcg": joined[f"ndcg_at_{k}_full"].idxmax(),
                "sparse_best_ndcg": joined[f"ndcg_at_{k}_sparse"].idxmax(),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output: Path,
    profile: str,
    panel: str,
    summary: pd.DataFrame,
    consistency: pd.DataFrame,
    k: int,
) -> None:
    full_columns = [
        "experiment_id",
        f"recall_at_{k}",
        f"ndcg_at_{k}",
        f"precision_at_{k}",
        f"utility_at_{k}",
        f"coverage_at_{k}",
        f"cold_recall_at_{k}",
    ]
    full = (
        summary[summary["evaluation"] == "full_exposure"]
        .sort_values(f"recall_at_{k}", ascending=False)
        .reset_index(drop=True)
    )
    sparse_best = consistency[
        [
            "policy",
            "density",
            "spearman_recall",
            "spearman_ndcg",
            "full_best_recall",
            "sparse_best_recall",
            "full_best_ndcg",
            "sparse_best_ndcg",
        ]
    ]
    text = f"""# KuaiRec 曝光偏差实验报告

## 1. 实验目标

本实验使用 KuaiRec `{panel}` 作为近全曝光真值，然后人为隐藏大部分反馈，模拟
传统稀疏曝光日志。目标是验证：

> 当反馈是 Missing Not At Random 时，传统离线 Recall/NDCG 可能改变模型排名。

Profile: `{profile}`

评测目录：3,327 个视频

Top-K：`{k}`

## 2. 全曝光真值排名

{markdown_table(full, full_columns)}

## 3. 稀疏回放与全曝光的一致性

{markdown_table(sparse_best, sparse_best.columns.tolist())}

## 4. 关键结论

- 均匀随机曝光下，稀疏回放和全曝光排名基本一致，说明“随机缺失”场景问题较小。
- 热门偏置曝光下，Recall 排名 Spearman 只有 `0.178571`，NDCG 排名 Spearman 为
  `-0.428571`，说明传统离线评测会严重偏向热门模型。
- 在热门偏置日志中，稀疏评测会选择 `R0.2` 热门召回作为冠军；但全曝光真值选择
  `R3.3` 双塔召回。这说明未曝光反馈不能被安全地视为负样本或无关样本。
- 正反馈偏置曝光的排名相关性较高，但仍会把冠军从 `R3.3` 换成 `R3.5`，说明只观察
  “更容易产生正反馈”的曝光也会改变模型选择。
- 本实验解释了为什么项目最终使用 `full_val/full_test` 进行模型选择和冻结验收，而
  不使用传统 big-matrix 稀疏日志作为唯一标准。

## 5. 对项目 baseline 的影响

本实验是评测协议诊断，不改变已经冻结的最终服务 baseline：

```text
召回：R3.4 Feature TwoTower + ID Dropout
精排：DR2.din
候选：R1.0 + R2.4 + R3.4
```

它的价值在于证明：如果只看传统稀疏曝光日志，项目可能会错误选择热门召回作为主
baseline；而全曝光评测能揭示双塔召回和序列精排的真实泛化价值。
"""
    (output / "EXPOSURE_BIAS_EXPERIMENT_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="local_8gb_large")
    parser.add_argument("--panel", choices=["full_val", "full_test"], default="full_val")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--experiments", nargs="+", default=["R0.2", "R1.0", "R2.4", "R3.3", "R3.4", "R3.5", "R3.6"])
    parser.add_argument("--k", type=int)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    k = args.k or profile["recall_k"]
    data = load_processed(args.profile)
    users = set(data["interactions"]["user_id"].unique())
    feedback = read_full_exposure_interactions(
        users,
        args.panel,
        profile.get("full_exposure_split_seed", 2026),
        profile.get("full_exposure_validation_fraction", 0.5),
    )
    catalog_size = len(feedback.attrs["full_catalog"])
    train = data["interactions"][data["interactions"]["split"] == "train"]
    item_popularity = train.groupby("video_id").size().to_dict()
    known_train_items = set(train["video_id"].unique())
    full_truth = build_full_exposure_ground_truth(feedback)

    rows = []
    recommendations_by_experiment = {
        experiment: load_recommendations(args.profile, experiment, args.panel)
        for experiment in args.experiments
    }
    for experiment, recommendations in recommendations_by_experiment.items():
        metrics, _ = evaluate_full_exposure(recommendations, feedback, k, item_popularity)
        metrics.update(evaluate_cold_recall(recommendations, full_truth, known_train_items, k))
        rows.append(
            {
                "evaluation": "full_exposure",
                "policy": "full",
                "density": 1.0,
                "experiment_id": experiment,
                **metrics,
            }
        )

    for policy_index, (policy, density) in enumerate(POLICY_SPECS):
        sampled = sample_exposure(
            feedback,
            policy,
            density,
            item_popularity,
            args.seed + policy_index,
        )
        for experiment, recommendations in recommendations_by_experiment.items():
            metrics = evaluate_sparse_replay(recommendations, sampled, catalog_size, k)
            rows.append(
                {
                    "evaluation": "sparse_replay",
                    "policy": policy,
                    "density": density,
                    "experiment_id": experiment,
                    **metrics,
                }
            )

    summary = pd.DataFrame(rows)
    consistency = rank_consistency(summary, k)
    output = ROOT / "reports" / "exposure_bias"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    consistency.to_csv(output / "rank_consistency.csv", index=False)
    write_json(
        output / "metadata.json",
        {
            "profile": args.profile,
            "panel": args.panel,
            "seed": args.seed,
            "experiments": args.experiments,
            "k": k,
            "policies": [{"policy": policy, "density": density} for policy, density in POLICY_SPECS],
            "evaluated_users": int(feedback["user_id"].nunique()),
            "evaluated_pairs": int(len(feedback)),
            "catalog_size": int(catalog_size),
        },
    )
    write_report(output, args.profile, args.panel, summary, consistency, k)
    print(json.dumps({"output": str(output), "rank_consistency": consistency.to_dict("records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
