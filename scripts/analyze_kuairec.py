"""Reproducible exploratory analysis for the KuaiRec dataset."""

from __future__ import annotations

import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "extracted" / "KuaiRec 2.0" / "data"
RAW_DIR = ROOT / "data" / "raw"
TABLE_DIR = ROOT / "reports" / "tables"
FIGURE_DIR = ROOT / "reports" / "figures"
CHUNK_SIZE = 500_000

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


def prepare_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep", rc={"font.family": "Microsoft YaHei"})


def pct(value: float) -> float:
    return round(100 * float(value), 4)


def flatten_counter_series(parts: list[pd.Series]) -> pd.Series:
    if not parts:
        return pd.Series(dtype="int64")
    frame = pd.concat(parts, axis=1).fillna(0)
    return frame.sum(axis=1).astype("int64").sort_values(ascending=False)


def concentration(counts: pd.Series, share: float) -> float:
    n = max(1, int(np.ceil(len(counts) * share)))
    return pct(counts.iloc[:n].sum() / counts.sum())


def scan_interactions(name: str, path: Path) -> tuple[dict, pd.Series, pd.Series, pd.Series]:
    user_parts: list[pd.Series] = []
    item_parts: list[pd.Series] = []
    daily_parts: list[pd.Series] = []
    totals = Counter()
    watch_sample: list[np.ndarray] = []
    watch_sum = 0.0
    watch_count = 0
    watch_max = 0.0
    min_date = None
    max_date = None

    columns = [
        "user_id",
        "video_id",
        "play_duration",
        "video_duration",
        "date",
        "watch_ratio",
    ]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=CHUNK_SIZE):
        totals["rows"] += len(chunk)
        totals["complete"] += int((chunk["watch_ratio"] >= 1).sum())
        totals["strong_positive"] += int((chunk["watch_ratio"] >= 2).sum())
        totals["short_play"] += int((chunk["play_duration"] < np.minimum(3000, chunk["video_duration"])).sum())
        totals["invalid_duration"] += int((chunk["video_duration"] <= 0).sum())
        watch = chunk["watch_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        watch_sum += float(watch.sum())
        watch_count += len(watch)
        watch_max = max(watch_max, float(watch.max()))
        if len(watch_sample) < 24:
            watch_sample.append(watch.sample(min(25_000, len(watch)), random_state=len(watch_sample)).to_numpy())

        current_min = int(chunk["date"].min())
        current_max = int(chunk["date"].max())
        min_date = current_min if min_date is None else min(min_date, current_min)
        max_date = current_max if max_date is None else max(max_date, current_max)
        user_parts.append(chunk["user_id"].value_counts())
        item_parts.append(chunk["video_id"].value_counts())
        daily_parts.append(chunk["date"].value_counts())

    user_counts = flatten_counter_series(user_parts)
    item_counts = flatten_counter_series(item_parts)
    daily_counts = flatten_counter_series(daily_parts).sort_index()
    sample = np.concatenate(watch_sample)
    possible = len(user_counts) * len(item_counts)
    summary = {
        "interactions": int(totals["rows"]),
        "users": int(len(user_counts)),
        "items": int(len(item_counts)),
        "density_pct": pct(totals["rows"] / possible),
        "date_min": str(min_date),
        "date_max": str(max_date),
        "watch_ratio_mean": round(watch_sum / watch_count, 6),
        "watch_ratio_median_sample": round(float(np.median(sample)), 6),
        "watch_ratio_p90_sample": round(float(np.quantile(sample, 0.9)), 6),
        "watch_ratio_p99_sample": round(float(np.quantile(sample, 0.99)), 6),
        "watch_ratio_max": round(watch_max, 6),
        "complete_rate_pct": pct(totals["complete"] / totals["rows"]),
        "strong_positive_rate_pct": pct(totals["strong_positive"] / totals["rows"]),
        "short_play_rate_pct": pct(totals["short_play"] / totals["rows"]),
        "invalid_duration_rows": int(totals["invalid_duration"]),
        "interactions_per_user_mean": round(float(user_counts.mean()), 4),
        "interactions_per_user_median": round(float(user_counts.median()), 4),
        "interactions_per_item_mean": round(float(item_counts.mean()), 4),
        "interactions_per_item_median": round(float(item_counts.median()), 4),
        "top_1pct_item_interaction_share_pct": concentration(item_counts, 0.01),
        "top_5pct_item_interaction_share_pct": concentration(item_counts, 0.05),
        "top_20pct_item_interaction_share_pct": concentration(item_counts, 0.20),
    }

    user_counts.rename("interaction_count").to_csv(TABLE_DIR / f"{name}_user_activity.csv", index_label="user_id")
    item_counts.rename("interaction_count").to_csv(TABLE_DIR / f"{name}_item_popularity.csv", index_label="video_id")
    daily_counts.rename("interaction_count").to_csv(TABLE_DIR / f"{name}_daily_interactions.csv", index_label="date")
    return summary, user_counts, item_counts, daily_counts


def summarize_table(path: Path) -> dict:
    read_kwargs = {"engine": "python"} if path.name == "kuairec_caption_category.csv" else {}
    frame = pd.read_csv(path, **read_kwargs)
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "missing_cell_pct": pct(frame.isna().sum().sum() / frame.size),
    }


def analyze_side_information() -> tuple[dict, dict[str, pd.DataFrame]]:
    users = pd.read_csv(DATA_DIR / "user_features.csv")
    raw_users = pd.read_csv(RAW_DIR / "user_features_raw.csv")
    item_categories = pd.read_csv(DATA_DIR / "item_categories.csv")
    raw_categories = pd.read_csv(RAW_DIR / "video_raw_categories_multi.csv")
    captions = pd.read_csv(DATA_DIR / "kuairec_caption_category.csv", engine="python")
    social = pd.read_csv(DATA_DIR / "social_network.csv")
    daily = pd.read_csv(DATA_DIR / "item_daily_features.csv")

    degree = social["friend_list"].fillna("[]").map(lambda x: len(ast.literal_eval(x)))
    tags_per_item = item_categories["feat"].fillna("[]").map(lambda x: len(ast.literal_eval(x)))
    root_dist = raw_categories["root_name"].fillna("UNKNOWN").value_counts().rename_axis("root_name").reset_index(name="count")
    active_dist = raw_users["user_active_degree"].fillna("UNKNOWN").value_counts().rename_axis("user_active_degree").reset_index(name="count")
    age_dist = raw_users["age_range"].fillna("UNKNOWN").value_counts().rename_axis("age_range").reset_index(name="count")
    city_dist = raw_users["fre_city_level"].fillna("UNKNOWN").value_counts().rename_axis("city_level").reset_index(name="count")
    platform_dist = raw_users["platform"].fillna("UNKNOWN").value_counts().rename_axis("platform").reset_index(name="count")

    show = daily["show_cnt"].replace(0, np.nan)
    play = daily["play_cnt"].replace(0, np.nan)
    derived = pd.DataFrame(
        {
            "play_rate": daily["play_cnt"] / show,
            "complete_rate": daily["complete_play_cnt"] / play,
            "like_rate": daily["like_cnt"] / play,
            "comment_rate": daily["comment_cnt"] / play,
            "follow_rate": daily["follow_cnt"] / play,
            "share_rate": daily["share_cnt"] / play,
            "negative_feedback_rate": daily["reduce_similar_cnt"] / play,
        }
    ).replace([np.inf, -np.inf], np.nan)
    derived.describe(percentiles=[0.5, 0.9, 0.99]).T.to_csv(TABLE_DIR / "item_daily_derived_rate_summary.csv")

    distributions = {
        "raw_category_root_distribution": root_dist,
        "user_active_degree_distribution": active_dist,
        "user_age_distribution": age_dist,
        "user_city_level_distribution": city_dist,
        "user_platform_distribution": platform_dist,
    }
    for name, frame in distributions.items():
        frame.to_csv(TABLE_DIR / f"{name}.csv", index=False)

    summary = {
        "user_features": summarize_table(DATA_DIR / "user_features.csv"),
        "user_features_raw": summarize_table(RAW_DIR / "user_features_raw.csv"),
        "item_categories": summarize_table(DATA_DIR / "item_categories.csv"),
        "video_raw_categories_multi": summarize_table(RAW_DIR / "video_raw_categories_multi.csv"),
        "captions": summarize_table(DATA_DIR / "kuairec_caption_category.csv"),
        "social_network": summarize_table(DATA_DIR / "social_network.csv"),
        "item_daily_features": summarize_table(DATA_DIR / "item_daily_features.csv"),
        "unique_authors": int(daily["author_id"].nunique()),
        "ad_row_rate_pct": pct((daily["video_type"] == "AD").mean()),
        "social_users": int(social["user_id"].nunique()),
        "social_degree_mean": round(float(degree.mean()), 4),
        "social_degree_max": int(degree.max()),
        "tags_per_item_mean": round(float(tags_per_item.mean()), 4),
        "raw_category_annotations_per_item_mean": round(float(raw_categories.groupby("video_id").size().mean()), 4),
        "caption_non_null_pct": pct(captions["caption"].notna().mean()),
        "topic_tag_non_null_pct": pct(captions["topic_tag"].notna().mean()),
        "first_level_category_non_null_pct": pct(captions["first_level_category_name"].notna().mean()),
    }
    return summary, distributions


def save_figures(
    interaction_summaries: dict,
    daily_counts: dict[str, pd.Series],
    item_counts: dict[str, pd.Series],
    distributions: dict[str, pd.DataFrame],
) -> None:
    matrix_frame = pd.DataFrame(interaction_summaries).T.reset_index(names="matrix")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=matrix_frame, x="matrix", y="density_pct", ax=axes[0])
    axes[0].set_title("Interaction Matrix Density")
    axes[0].set_ylabel("Density (%)")
    sns.barplot(data=matrix_frame, x="matrix", y="complete_rate_pct", ax=axes[1])
    axes[1].set_title("Complete Watch Rate")
    axes[1].set_ylabel("Rate (%)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "matrix_density_and_completion.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, series in daily_counts.items():
        parsed = pd.to_datetime(series.index.astype("int64").astype(str), format="%Y%m%d")
        ax.plot(parsed, series.values, marker="o", label=name)
    ax.set_title("Daily Interactions")
    ax.set_ylabel("Interactions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "daily_interactions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, counts in item_counts.items():
        ranked = counts.sort_values(ascending=False).reset_index(drop=True)
        x = (np.arange(len(ranked)) + 1) / len(ranked)
        y = ranked.cumsum() / ranked.sum()
        ax.plot(x, y, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="uniform")
    ax.set_title("Item Popularity Concentration")
    ax.set_xlabel("Top fraction of items")
    ax.set_ylabel("Cumulative interaction share")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "item_popularity_concentration.png", dpi=180)
    plt.close(fig)

    root = distributions["raw_category_root_distribution"].head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=root, y="root_name", x="count", ax=ax)
    ax.set_title("Top Raw Video Root Categories")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "top_video_categories.png", dpi=180)
    plt.close(fig)

    active = distributions["user_active_degree_distribution"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=active, x="user_active_degree", y="count", ax=ax)
    ax.set_title("Raw User Activity Degree")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "user_activity_degree.png", dpi=180)
    plt.close(fig)


def main() -> None:
    prepare_output_dirs()
    interaction_summaries = {}
    user_counts = {}
    item_counts = {}
    daily_counts = {}
    for name in ("big_matrix", "small_matrix"):
        print(f"Scanning {name}...")
        summary, users, items, daily = scan_interactions(name, DATA_DIR / f"{name}.csv")
        interaction_summaries[name] = summary
        user_counts[name] = users
        item_counts[name] = items
        daily_counts[name] = daily

    side_summary, distributions = analyze_side_information()
    overlap = {
        "small_users_in_big_pct": pct(user_counts["small_matrix"].index.isin(user_counts["big_matrix"].index).mean()),
        "small_items_in_big_pct": pct(item_counts["small_matrix"].index.isin(item_counts["big_matrix"].index).mean()),
        "user_feature_coverage_big_pct": pct(
            user_counts["big_matrix"].index.isin(pd.read_csv(DATA_DIR / "user_features.csv", usecols=["user_id"])["user_id"]).mean()
        ),
    }
    summary = {
        "interactions": interaction_summaries,
        "side_information": side_summary,
        "overlap_and_coverage": overlap,
    }
    (ROOT / "reports" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(interaction_summaries).T.to_csv(TABLE_DIR / "interaction_summary.csv", index_label="matrix")
    save_figures(interaction_summaries, daily_counts, item_counts, distributions)
    print("Analysis complete. Outputs are under reports/.")


if __name__ == "__main__":
    main()
