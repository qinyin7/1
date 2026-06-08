from __future__ import annotations

import math
from hashlib import sha256

import numpy as np
import pandas as pd

from src.common import RAW_DATA_DIR


FULL_EXPOSURE_REFERENCE_DATE = 20200905
FULL_EXPOSURE_PANELS = ("full_val", "full_test")


def split_full_exposure_users(
    users: list[int] | set[int],
    seed: int = 2026,
    validation_fraction: float = 0.5,
) -> dict[str, set[int]]:
    """Create a deterministic user split so repeated tuning never sees full_test users."""
    ordered = sorted(set(users))
    if len(ordered) < 2:
        raise ValueError("At least two overlapping users are required for full_val/full_test.")
    scored = [
        (
            int.from_bytes(sha256(f"{seed}:{user}".encode("ascii")).digest()[:8], "big")
            / 2**64,
            user,
        )
        for user in ordered
    ]
    validation = {user for score, user in scored if score < validation_fraction}
    test = set(ordered) - validation
    if not validation:
        validation.add(min(scored)[1])
        test = set(ordered) - validation
    if not test:
        test.add(max(scored)[1])
        validation = set(ordered) - test
    return {
        "full_val": validation,
        "full_test": test,
    }


def read_full_exposure_interactions(
    selected_users: set[int] | None,
    panel: str,
    seed: int = 2026,
    validation_fraction: float = 0.5,
    reference_date: int = FULL_EXPOSURE_REFERENCE_DATE,
) -> pd.DataFrame:
    """Read the near-fully-observed matrix without dropping rows that lack event dates."""
    if panel not in FULL_EXPOSURE_PANELS:
        raise ValueError(f"Unsupported full-exposure panel: {panel}")
    parts = []
    full_catalog: set[int] = set()
    for chunk in pd.read_csv(RAW_DATA_DIR / "small_matrix.csv", chunksize=500_000):
        full_catalog.update(chunk["video_id"].dropna().astype(int).unique())
        if selected_users is not None:
            chunk = chunk[chunk["user_id"].isin(selected_users)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        raise ValueError("No small_matrix users overlap the selected profile users.")

    frame = pd.concat(parts, ignore_index=True)
    user_panels = split_full_exposure_users(
        set(frame["user_id"].unique()), seed, validation_fraction
    )
    frame = frame[frame["user_id"].isin(user_panels[panel])].copy()
    frame["source_date"] = frame["date"]
    frame["date"] = int(reference_date)
    frame["split"] = panel
    frame["label_complete"] = (frame["watch_ratio"] >= 1).astype("int8")
    frame["label_strong"] = (frame["watch_ratio"] >= 2).astype("int8")
    frame["label_short"] = (
        frame["play_duration"] < np.minimum(3000, frame["video_duration"])
    ).astype("int8")
    frame["watch_ratio_clipped"] = frame["watch_ratio"].clip(lower=0, upper=5)
    frame["utility"] = (
        frame["label_complete"] + 0.5 * frame["label_strong"] - 0.5 * frame["label_short"]
    ).astype("float32")
    frame["hour"] = np.int8(12)
    frame["weekday"] = np.int8(pd.Timestamp(str(reference_date)).weekday())
    frame.attrs["full_catalog"] = full_catalog
    return frame


def build_full_exposure_ground_truth(
    frame: pd.DataFrame,
    label: str = "label_complete",
) -> dict[int, set[int]]:
    positive = frame[frame[label] == 1]
    return positive.groupby("user_id")["video_id"].apply(set).to_dict()


def evaluate_full_exposure(
    recommendations: dict[int, list[int]],
    feedback: pd.DataFrame,
    k: int,
    item_popularity: dict[int, int] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate Top-K against observed feedback for every item in the small catalog."""
    item_popularity = item_popularity or {}
    catalog = feedback.attrs.get("full_catalog", set(feedback["video_id"].unique()))
    feedback_lookup = feedback.set_index(["user_id", "video_id"])
    rows = []
    recommended_items: set[int] = set()
    invalid_recommendations = 0
    unobserved_pairs = 0

    for user, user_feedback in feedback.groupby("user_id", sort=True):
        observed_items = set(user_feedback["video_id"])
        ranked = []
        for item in recommendations.get(int(user), []):
            if item not in catalog:
                invalid_recommendations += 1
                continue
            if item not in observed_items:
                unobserved_pairs += 1
                continue
            if item not in ranked:
                ranked.append(int(item))
            if len(ranked) == k:
                break

        positives = set(user_feedback.loc[user_feedback["label_complete"] == 1, "video_id"])
        matched = np.array([item in positives for item in ranked], dtype=float)
        discounts = 1 / np.log2(np.arange(2, len(ranked) + 2))
        ideal_length = min(len(positives), k)
        ideal = float(np.sum(1 / np.log2(np.arange(2, ideal_length + 2))))
        selected = feedback_lookup.loc[[(int(user), item) for item in ranked]] if ranked else None
        recommended_items.update(ranked)
        rows.append(
            {
                "user_id": int(user),
                "recommended_count": len(ranked),
                f"precision_at_{k}": float(matched.sum() / k),
                f"recall_at_{k}": float(matched.sum() / len(positives)) if positives else 0.0,
                f"hit_rate_at_{k}": float(matched.sum() > 0),
                f"ndcg_at_{k}": float(np.sum(matched * discounts) / ideal) if ideal else 0.0,
                f"complete_rate_at_{k}": float(selected["label_complete"].mean())
                if selected is not None
                else 0.0,
                f"strong_rate_at_{k}": float(selected["label_strong"].mean())
                if selected is not None
                else 0.0,
                f"short_rate_at_{k}": float(selected["label_short"].mean())
                if selected is not None
                else 0.0,
                f"watch_ratio_at_{k}": float(selected["watch_ratio_clipped"].mean())
                if selected is not None
                else 0.0,
                f"utility_at_{k}": float(selected["utility"].mean())
                if selected is not None
                else 0.0,
                f"average_log_popularity_at_{k}": float(
                    np.mean([math.log1p(item_popularity.get(item, 0)) for item in ranked])
                )
                if ranked
                else 0.0,
            }
        )

    group_metrics = pd.DataFrame(rows)
    metric_columns = [column for column in group_metrics if column != "user_id"]
    metrics = {column: float(group_metrics[column].mean()) for column in metric_columns}
    metrics.update(
        {
            f"coverage_at_{k}": len(recommended_items) / max(len(catalog), 1),
            "catalog_size": len(catalog),
            "evaluated_users": int(feedback["user_id"].nunique()),
            "evaluated_pairs": len(feedback),
            "matrix_density": len(feedback)
            / max(feedback["user_id"].nunique() * len(catalog), 1),
            "invalid_recommendations": invalid_recommendations,
            "unobserved_recommendation_pairs": unobserved_pairs,
        }
    )
    return metrics, group_metrics
