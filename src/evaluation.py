from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def build_ground_truth(
    interactions: pd.DataFrame,
    split: str,
    label: str = "label_complete",
    excluded_items_by_user: dict[int, set[int]] | None = None,
) -> dict[int, set[int]]:
    positive = interactions[(interactions["split"] == split) & (interactions[label] == 1)]
    ground_truth = positive.groupby("user_id")["video_id"].apply(lambda values: set(values)).to_dict()
    if excluded_items_by_user is not None:
        ground_truth = {
            user: positives - excluded_items_by_user.get(user, set())
            for user, positives in ground_truth.items()
        }
    return {user: positives for user, positives in ground_truth.items() if positives}


def build_daily_ground_truth(
    interactions: pd.DataFrame,
    split: str,
    label: str = "label_complete",
    excluded_items_by_user: dict[int, set[int]] | None = None,
) -> dict[tuple[int, int], set[int]]:
    positive = interactions[(interactions["split"] == split) & (interactions[label] == 1)]
    ground_truth = (
        positive.groupby(["user_id", "date"])["video_id"].apply(lambda values: set(values)).to_dict()
    )
    if excluded_items_by_user is not None:
        ground_truth = {
            key: positives - excluded_items_by_user.get(key[0], set())
            for key, positives in ground_truth.items()
        }
    return {key: positives for key, positives in ground_truth.items() if positives}


def filter_daily_recommendations(
    recommendations: dict[int, list[int]],
    ground_truth: dict[tuple[int, int], set[int]],
    first_seen_date: dict[int, int],
    k: int,
) -> dict[tuple[int, int], list[int]]:
    return {
        (user, date): [
            item
            for item in recommendations.get(user, [])
            if first_seen_date.get(item, 99999999) <= date
        ][:k]
        for user, date in ground_truth
    }


def evaluate_recall(
    recommendations: dict,
    ground_truth: dict,
    item_popularity: dict[int, int],
    catalog_size: int,
    k: int,
) -> dict[str, float]:
    recalls, hits, ndcgs, popularity = [], [], [], []
    recommended_items: set[int] = set()
    evaluated_users: set[int] = set()
    for key, positives in ground_truth.items():
        ranked = recommendations.get(key, [])[:k]
        if not positives:
            continue
        evaluated_users.add(key[0] if isinstance(key, tuple) else key)
        matched = [1 if item in positives else 0 for item in ranked]
        hit_count = sum(matched)
        recalls.append(hit_count / len(positives))
        hits.append(float(hit_count > 0))
        dcg = sum(value / math.log2(index + 2) for index, value in enumerate(matched))
        ideal = sum(1 / math.log2(index + 2) for index in range(min(len(positives), k)))
        ndcgs.append(dcg / ideal if ideal else 0)
        popularity.extend(math.log1p(item_popularity.get(item, 0)) for item in ranked)
        recommended_items.update(ranked)
    return {
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"hit_rate_at_{k}": float(np.mean(hits)) if hits else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"coverage_at_{k}": len(recommended_items) / max(catalog_size, 1),
        f"average_log_popularity_at_{k}": float(np.mean(popularity)) if popularity else 0.0,
        "evaluated_users": len(evaluated_users),
        "evaluated_groups": len(recalls),
    }


def recall_group_metrics(
    recommendations: dict[tuple[int, int], list[int]],
    ground_truth: dict[tuple[int, int], set[int]],
    k: int,
) -> pd.DataFrame:
    rows = []
    for (user, date), positives in ground_truth.items():
        ranked = recommendations.get((user, date), [])[:k]
        matched = np.array([item in positives for item in ranked], dtype=float)
        recall = float(matched.sum() / len(positives))
        discounts = 1 / np.log2(np.arange(2, len(matched) + 2))
        dcg = float(np.sum(matched * discounts))
        ideal_length = min(len(positives), k)
        ideal = float(np.sum(1 / np.log2(np.arange(2, ideal_length + 2))))
        rows.append(
            {
                "user_id": user,
                "date": date,
                f"recall_at_{k}": recall,
                f"ndcg_at_{k}": dcg / ideal if ideal else 0.0,
                f"hit_rate_at_{k}": float(matched.sum() > 0),
            }
        )
    return pd.DataFrame(rows)


def evaluate_cold_recall(
    recommendations: dict,
    ground_truth: dict,
    known_train_items: set[int],
    k: int,
) -> dict[str, float]:
    cold_truth = {
        key: {item for item in positives if item not in known_train_items}
        for key, positives in ground_truth.items()
    }
    cold_truth = {key: positives for key, positives in cold_truth.items() if positives}
    recalls = []
    cold_users = set()
    for key, positives in cold_truth.items():
        ranked = recommendations.get(key, [])[:k]
        cold_users.add(key[0] if isinstance(key, tuple) else key)
        recalls.append(len(set(ranked) & positives) / len(positives))
    return {
        f"cold_recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        "cold_evaluated_users": len(cold_users),
        "cold_evaluated_groups": len(recalls),
    }


def evaluate_ranking(frame: pd.DataFrame, score_column: str = "score") -> dict[str, float]:
    y_true = frame["label_complete"].to_numpy()
    scores = frame[score_column].to_numpy()
    metrics = {
        "auc": float(roc_auc_score(y_true, scores)),
        "log_loss": float(log_loss(y_true, np.clip(scores, 1e-7, 1 - 1e-7))),
    }
    user_auc, user_weight = [], []
    for _, group in frame.groupby("user_id"):
        if group["label_complete"].nunique() < 2:
            continue
        user_auc.append(roc_auc_score(group["label_complete"], group[score_column]))
        user_weight.append(len(group))
    metrics["gauc"] = float(np.average(user_auc, weights=user_weight)) if user_auc else 0.0
    ndcg_values = []
    top_k_rows = []
    for _, group in frame.groupby(["user_id", "date"]):
        selected = group.sort_values(score_column, ascending=False).head(10)
        ranked = selected["label_complete"].to_numpy()
        ideal = np.sort(group["label_complete"].to_numpy())[::-1][:10]
        discounts = 1 / np.log2(np.arange(2, len(ranked) + 2))
        dcg = float(np.sum(ranked * discounts))
        idcg = float(np.sum(ideal * discounts))
        if idcg > 0:
            ndcg_values.append(dcg / idcg)
        top_k_rows.append(selected)
    metrics["ndcg_at_10"] = float(np.mean(ndcg_values)) if ndcg_values else 0.0
    if top_k_rows:
        top_k = pd.concat(top_k_rows, ignore_index=True)
        metrics["complete_rate_at_10"] = float(top_k["label_complete"].mean())
        if "label_strong" in top_k:
            metrics["strong_rate_at_10"] = float(top_k["label_strong"].mean())
        if "label_short" in top_k:
            metrics["short_rate_at_10"] = float(top_k["label_short"].mean())
        if {"label_complete", "label_strong", "label_short"} <= set(top_k):
            utility = (
                top_k["label_complete"] + 0.5 * top_k["label_strong"] - 0.5 * top_k["label_short"]
            )
            metrics["utility_at_10"] = float(utility.mean())
    return metrics


def ranking_group_metrics(frame: pd.DataFrame, score_column: str = "score") -> pd.DataFrame:
    rows = []
    for (user, date), group in frame.groupby(["user_id", "date"]):
        ranked = group.sort_values(score_column, ascending=False).head(10)["label_complete"].to_numpy()
        ideal = np.sort(group["label_complete"].to_numpy())[::-1][:10]
        discounts = 1 / np.log2(np.arange(2, len(ranked) + 2))
        dcg = float(np.sum(ranked * discounts))
        idcg = float(np.sum(ideal * discounts))
        rows.append({"user_id": user, "date": date, "ndcg_at_10": dcg / idcg if idcg else 0.0})
    return pd.DataFrame(rows)
