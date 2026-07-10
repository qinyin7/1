from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_experiment import load_processed
from src.candidate_features import (
    CHANNELS,
    EXPERIMENT_FEATURES,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    PUBLIC_EXPERIMENT_FEATURES,
)
from src.common import (
    RESULT_SCHEMA_VERSION,
    artifact_dir,
    candidate_channels,
    load_profile,
    ranking_eval_ks,
    resolve_candidate_channel_limits,
    resolve_candidate_limits,
    write_json,
)
from src.data_pipeline import _aggregate_statistics
from src.evaluation import (
    build_daily_ground_truth,
    evaluate_cold_recall,
    evaluate_recall,
    recall_group_metrics,
)
from src.full_exposure import FULL_EXPOSURE_PANELS, read_full_exposure_interactions
from src.experiment_registry import canonical_experiment_id


DEFAULT_CANDIDATE_CHANNELS = [
    "itemcf_main",
    "content_text_category",
    "feature_tower_id_dropout",
]


def load_recommendations(profile: str, experiment: str, panel: str) -> dict[int, list[int]]:
    recall_dir = artifact_dir(profile, "recall")
    path = recall_dir / f"{experiment}_{panel}_recommendations.json"
    if not path.exists():
        canonical_id = canonical_experiment_id(experiment)
        path = recall_dir / f"{canonical_id}_{panel}_recommendations.json"
    return {int(user): [int(item) for item in items] for user, items in json.loads(path.read_text()).items()}


def build_affinity(history: pd.DataFrame, items: pd.DataFrame) -> tuple[dict, dict]:
    positive = history[history["label_complete"] == 1][["user_id", "video_id"]].merge(
        items[["video_id", "first_category", "author_id"]], on="video_id", how="left"
    )
    category_affinity, author_affinity = {}, {}
    for user, group in positive.groupby("user_id"):
        total = max(len(group), 1)
        category_affinity[user] = {
            key: value / total for key, value in Counter(group["first_category"]).items()
        }
        author_affinity[user] = {
            key: value / total for key, value in Counter(group["author_id"]).items()
        }
    return category_affinity, author_affinity


def build_candidate_frame(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    panel: str,
    history_splits: list[str],
    experiment_ids: list[str],
    profile_name: str,
    max_eval_users: int,
    top_per_channel: int,
    max_candidates_per_group: int,
    seed: int,
    sample_training_negatives: bool,
    history_interactions: pd.DataFrame | None = None,
    evaluation_catalog_size: int | None = None,
    channel_top_limits: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict, set[int], int]:
    history_source = history_interactions if history_interactions is not None else interactions
    history = history_source[history_source["split"].isin(history_splits)]
    seen = history.groupby("user_id")["video_id"].apply(set).to_dict()
    panel_interactions = interactions[interactions["split"] == panel]
    positives = panel_interactions[panel_interactions["label_complete"] == 1]
    ground_truth = positives.groupby(["user_id", "date"])["video_id"].apply(set).to_dict()
    ground_truth = {
        key: values - seen.get(key[0], set()) for key, values in ground_truth.items()
    }
    ground_truth = {key: values for key, values in ground_truth.items() if values}
    users = sorted({user for user, _ in ground_truth})
    if max_eval_users and len(users) > max_eval_users:
        users = (
            pd.Series(users).sample(max_eval_users, random_state=seed).sort_values().tolist()
        )
        ground_truth = {key: values for key, values in ground_truth.items() if key[0] in users}

    recommendations = [
        load_recommendations(profile_name, experiment, panel) for experiment in experiment_ids
    ]
    item_lookup = items.set_index("video_id")
    first_seen = item_lookup["first_seen_date"].to_dict()
    known_items = set(history["video_id"].unique())
    user_stats, item_stats = _aggregate_statistics(history)
    user_stats = user_stats.set_index("user_id").to_dict("index")
    item_stats = item_stats.set_index("video_id").to_dict("index")
    category_affinity, author_affinity = build_affinity(history, items)
    candidate_labels = (
        panel_interactions.groupby(["user_id", "date", "video_id"])[LABEL_COLUMNS]
        .max()
        .to_dict("index")
    )
    rng = np.random.default_rng(seed)
    rows = []
    for (user, date), positive_items in ground_truth.items():
        candidates: dict[int, dict[str, float]] = {}
        for channel, channel_result in zip(CHANNELS, recommendations):
            channel_top = (channel_top_limits or {}).get(channel, top_per_channel)
            eligible = [
                item
                for item in channel_result.get(user, [])
                if first_seen.get(item, 99999999) <= date
            ][:channel_top]
            for rank, item in enumerate(eligible, start=1):
                values = candidates.setdefault(item, {})
                values[f"{channel}_present"] = 1.0
                values[f"{channel}_rank_score"] = 1.0 / rank
        candidate_items = list(candidates)
        positive_candidates = [item for item in candidate_items if item in positive_items]
        # An unexposed recalled item has unknown feedback, not a negative label.
        negative_candidates = [
            item
            for item in candidate_items
            if item not in positive_items and (user, date, item) in candidate_labels
        ]
        if sample_training_negatives:
            negative_limit = max(0, max_candidates_per_group - len(positive_candidates))
            if len(negative_candidates) > negative_limit:
                negative_candidates = rng.choice(
                    negative_candidates, negative_limit, replace=False
                ).tolist()
        for item in positive_candidates + negative_candidates:
            item_info = item_lookup.loc[item] if item in item_lookup.index else {}
            item_category = item_info.get("first_category", -1)
            author_id = item_info.get("author_id", -1)
            row = {
                "user_id": user,
                "date": date,
                "video_id": item,
                "label": int(item in positive_items),
                **candidate_labels.get(
                    (user, date, item),
                    {label_column: 0 for label_column in LABEL_COLUMNS},
                ),
                "channel_count": sum(
                    candidates[item].get(f"{channel}_present", 0.0) for channel in CHANNELS
                ),
                "first_category": item_category,
                "video_duration": item_info.get("video_duration", 0),
                "author_id_hash": int(author_id) % 256,
                "is_cold_item": float(item not in known_items),
                "item_age_days": max(
                    0,
                    (
                        pd.to_datetime(str(date), format="%Y%m%d")
                        - pd.to_datetime(str(first_seen.get(item, date)), format="%Y%m%d")
                    ).days,
                ),
                "category_affinity": category_affinity.get(user, {}).get(item_category, 0.0),
                "author_affinity": author_affinity.get(user, {}).get(author_id, 0.0),
                **user_stats.get(user, {}),
                **item_stats.get(item, {}),
            }
            for channel in CHANNELS:
                row[f"{channel}_present"] = candidates[item].get(f"{channel}_present", 0.0)
                row[f"{channel}_rank_score"] = candidates[item].get(
                    f"{channel}_rank_score", 0.0
                )
            item_complete_rate = float(row.get("item_complete_rate", 0.0) or 0.0)
            row["category_item_complete_cross"] = (
                float(row["category_affinity"]) * item_complete_rate
            )
            row["author_item_complete_cross"] = (
                float(row["author_affinity"]) * item_complete_rate
            )
            row["channel_item_complete_cross"] = (
                float(row["channel_count"]) * item_complete_rate
            )
            row["cold_content_cross"] = (
                float(row["is_cold_item"]) * float(row["content_present"])
            )
            row["tower_age_cross"] = (
                float(row["tower_present"]) * float(row["item_age_days"])
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].astype("float32")
    frame[LABEL_COLUMNS + ["label"]] = frame[LABEL_COLUMNS + ["label"]].astype("int8")
    frame["date"] = frame["date"].astype("int32")
    eligible_catalog = evaluation_catalog_size or int(
        items["first_seen_date"]
        .le(interactions[interactions["split"].eq(panel)]["date"].max())
        .sum()
    )
    return frame, ground_truth, known_items, eligible_catalog


def load_or_build_candidate_frames(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    profile_name: str,
    max_eval_users: int,
    top_per_channel: int,
    max_candidates_per_group: int,
    seed: int,
    rebuild_cache: bool,
    evaluation_panel: str = "full_val",
) -> tuple[pd.DataFrame, pd.DataFrame, dict, set[int], int]:
    if evaluation_panel not in FULL_EXPOSURE_PANELS:
        raise ValueError("Candidate model selection must use full_val or frozen full_test.")
    profile = load_profile(profile_name)
    experiment_ids = candidate_channels(profile, DEFAULT_CANDIDATE_CHANNELS)
    channel_top_limits = resolve_candidate_channel_limits(profile, top_per_channel, CHANNELS)
    split_seed = profile.get("full_exposure_split_seed", 2026)
    validation_fraction = profile.get("full_exposure_validation_fraction", 0.5)
    cache_dir = artifact_dir(profile_name, "candidate_ranking") / "cache"
    train_path = cache_dir / "logged_valid_train_candidates.parquet"
    test_path = cache_dir / f"{evaluation_panel}_candidates.parquet"
    metadata_path = cache_dir / f"{evaluation_panel}_metadata.json"
    cache_matches = False
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cache_matches = metadata == {
            "profile": profile_name,
            "seed": seed,
            "top_per_channel": top_per_channel,
            "max_candidates_per_group": max_candidates_per_group,
            "candidate_channel_top_limits": channel_top_limits,
            "train_channels": experiment_ids,
            "evaluation_panel": evaluation_panel,
            "evaluation_channels": experiment_ids,
            "full_exposure_split_seed": split_seed,
            "full_exposure_validation_fraction": validation_fraction,
            "schema_version": 7,
        }
    if not rebuild_cache and cache_matches and train_path.exists() and test_path.exists():
        train_frame = pd.read_parquet(train_path)
        test_frame = pd.read_parquet(test_path)
    else:
        train_frame, _, _, _ = build_candidate_frame(
            interactions,
            items,
            "valid",
            ["train"],
            experiment_ids,
            profile_name,
            max_eval_users,
            top_per_channel,
            max_candidates_per_group,
            seed,
            True,
            channel_top_limits=channel_top_limits,
        )
        full_exposure = read_full_exposure_interactions(
            set(interactions["user_id"].unique()),
            evaluation_panel,
            split_seed,
            validation_fraction,
        )
        test_frame, test_truth, known_items, catalog_size = build_candidate_frame(
            full_exposure,
            items,
            evaluation_panel,
            ["train"],
            experiment_ids,
            profile_name,
            max_eval_users,
            top_per_channel,
            max_candidates_per_group,
            seed,
            False,
            history_interactions=interactions,
            evaluation_catalog_size=len(full_exposure.attrs["full_catalog"]),
            channel_top_limits=channel_top_limits,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_frame.to_parquet(train_path, index=False)
        test_frame.to_parquet(test_path, index=False)
        write_json(
            metadata_path,
            {
                "profile": profile_name,
                "seed": seed,
                "top_per_channel": top_per_channel,
                "max_candidates_per_group": max_candidates_per_group,
                "candidate_channel_top_limits": channel_top_limits,
                "train_channels": experiment_ids,
                "evaluation_panel": evaluation_panel,
                "evaluation_channels": experiment_ids,
                "full_exposure_split_seed": split_seed,
                "full_exposure_validation_fraction": validation_fraction,
                "schema_version": 7,
            },
        )
    if not rebuild_cache and cache_matches and train_path.exists() and test_path.exists():
        full_exposure = read_full_exposure_interactions(
            set(interactions["user_id"].unique()),
            evaluation_panel,
            split_seed,
            validation_fraction,
        )
        seen = (
            interactions[interactions["split"].eq("train")]
            .groupby("user_id")["video_id"]
            .apply(set)
            .to_dict()
        )
        test_truth = build_daily_ground_truth(
            full_exposure,
            evaluation_panel,
            excluded_items_by_user=seen,
        )
        evaluated_users = set(test_frame["user_id"].unique())
        test_truth = {
            key: positives for key, positives in test_truth.items() if key[0] in evaluated_users
        }
        known_items = set(interactions.loc[interactions["split"].eq("train"), "video_id"])
        catalog_size = len(full_exposure.attrs["full_catalog"])
    return train_frame, test_frame, test_truth, known_items, catalog_size


def evaluate_ranked(
    frame: pd.DataFrame,
    ground_truth: dict,
    known_items: set[int],
    catalog_size: int,
    k: int,
) -> tuple[dict, dict]:
    recommendations = {
        key: group.sort_values("score", ascending=False)["video_id"].head(k).astype(int).tolist()
        for key, group in frame.groupby(["user_id", "date"])
    }
    metrics = evaluate_recall(recommendations, ground_truth, {}, catalog_size, k)
    metrics.update(evaluate_cold_recall(recommendations, ground_truth, known_items, k))
    selected = (
        frame.sort_values(["user_id", "date", "score"], ascending=[True, True, False])
        .groupby(["user_id", "date"])
        .head(k)
    )
    if not selected.empty and {"label_complete", "label_strong", "label_short"} <= set(selected):
        metrics[f"precision_at_{k}"] = float(selected["label_complete"].sum() / (len(ground_truth) * k))
        metrics[f"complete_rate_at_{k}"] = float(selected["label_complete"].mean())
        metrics[f"strong_rate_at_{k}"] = float(selected["label_strong"].mean())
        metrics[f"short_rate_at_{k}"] = float(selected["label_short"].mean())
        utility = (
            selected["label_complete"]
            + 0.5 * selected["label_strong"]
            - 0.5 * selected["label_short"]
        )
        metrics[f"utility_at_{k}"] = float(utility.mean())
    return metrics, recommendations


def evaluate_ranked_at_ks(
    frame: pd.DataFrame,
    ground_truth: dict,
    known_items: set[int],
    catalog_size: int,
    ks: list[int],
) -> tuple[dict, dict, pd.DataFrame]:
    metrics = {}
    recommendations_by_k = {}
    group_frames = []
    for k in sorted(set(ks)):
        current_metrics, recommendations = evaluate_ranked(
            frame,
            ground_truth,
            known_items,
            catalog_size,
            k,
        )
        current_metrics.pop(f"average_log_popularity_at_{k}", None)
        metrics.update(current_metrics)
        recommendations_by_k[k] = recommendations
        group_frames.append(recall_group_metrics(recommendations, ground_truth, k))
    group_metrics = group_frames[0]
    for current in group_frames[1:]:
        group_metrics = group_metrics.merge(
            current,
            on=["user_id", "date"],
            how="outer",
        )
    return metrics, recommendations_by_k[max(recommendations_by_k)], group_metrics


def add_channel_selection_metrics(
    metrics: dict,
    evaluation: pd.DataFrame,
    ks: list[int],
    primary_k: int,
) -> None:
    sorted_evaluation = evaluation.sort_values(
        ["user_id", "date", "score"],
        ascending=[True, True, False],
    )
    for k in sorted(set(ks)):
        selected = sorted_evaluation.groupby(["user_id", "date"]).head(k)
        metrics[f"tower_selected_rate_at_{k}"] = float(selected["tower_present"].mean())
        metrics[f"tower_unique_selected_rate_at_{k}"] = float(
            (
                (selected["tower_present"] == 1)
                & (selected["itemcf_present"] == 0)
                & (selected["content_present"] == 0)
            ).mean()
        )
    metrics["tower_selected_rate"] = metrics[f"tower_selected_rate_at_{primary_k}"]
    metrics["tower_unique_selected_rate"] = metrics[f"tower_unique_selected_rate_at_{primary_k}"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="local_8gb_large")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-per-channel", type=int)
    parser.add_argument("--max-candidates-per-group", type=int)
    parser.add_argument("--estimators", type=int)
    parser.add_argument("--panel", choices=list(FULL_EXPOSURE_PANELS), default="full_val")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(PUBLIC_EXPERIMENT_FEATURES),
        choices=list(EXPERIMENT_FEATURES),
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    top_per_channel, max_candidates_per_group = resolve_candidate_limits(
        profile,
        args.top_per_channel,
        args.max_candidates_per_group,
    )
    estimators = args.estimators or profile["lightgbm_estimators"]
    data = load_processed(args.profile)
    interactions, items = data["interactions"], data["items"]

    start = time.perf_counter()
    train_frame, test_frame, test_truth, known_items, catalog_size = load_or_build_candidate_frames(
        interactions,
        items,
        args.profile,
        profile["max_eval_users"],
        top_per_channel,
        max_candidates_per_group,
        args.seed,
        args.rebuild_cache,
        args.panel,
    )
    results = {}
    output = artifact_dir(args.profile, "candidate_ranking")
    daily_rows = []
    importance_rows = []
    eval_ks = ranking_eval_ks(profile)
    for experiment_id in args.experiments:
        columns = EXPERIMENT_FEATURES[experiment_id]
        if experiment_id in {"PR.no_tower", "lambdarank_without_tower_candidates"}:
            train_experiment = train_frame[
                (train_frame["itemcf_present"] == 1) | (train_frame["content_present"] == 1)
            ]
            test_experiment = test_frame[
                (test_frame["itemcf_present"] == 1) | (test_frame["content_present"] == 1)
            ]
        else:
            train_experiment = train_frame
            test_experiment = test_frame
        model = LGBMRanker(
            objective="lambdarank",
            n_estimators=estimators,
            learning_rate=0.05,
            num_leaves=63,
            random_state=args.seed,
            n_jobs=-1,
            verbosity=-1,
        )
        train_sorted = train_experiment.sort_values(["user_id", "date"])
        groups = train_sorted.groupby(["user_id", "date"], sort=False).size().to_numpy()
        train_start = time.perf_counter()
        model.fit(train_sorted[columns], train_sorted["label"], group=groups)
        train_seconds = time.perf_counter() - train_start
        evaluation = test_experiment.copy()
        predict_start = time.perf_counter()
        evaluation["score"] = model.predict(evaluation[columns])
        prediction_seconds = time.perf_counter() - predict_start
        metrics, recommendations, group_metrics = evaluate_ranked_at_ks(
            evaluation, test_truth, known_items, catalog_size, eval_ks
        )
        add_channel_selection_metrics(metrics, evaluation, eval_ks, profile["recall_k"])
        metrics.update(
            {
                "train_rows": len(train_experiment),
                "test_rows": len(test_experiment),
                "train_positive_rate": float(train_experiment["label_complete"].mean()),
                "train_seconds": train_seconds,
                "prediction_seconds": prediction_seconds,
                "top_per_channel": top_per_channel,
                "max_candidates_per_group": max_candidates_per_group,
                "estimators": estimators,
            }
        )
        results[experiment_id] = metrics
        group_metrics.to_parquet(
            output / f"{experiment_id}_{args.panel}_{args.seed}_group_metrics.parquet",
            index=False,
        )
        metric_columns = [
            column
            for column in group_metrics.columns
            if column.startswith(("recall_at_", "ndcg_at_", "hit_rate_at_"))
        ]
        daily = group_metrics.groupby("date", as_index=False)[metric_columns].mean()
        daily["experiment_id"] = experiment_id
        daily_rows.extend(daily.to_dict("records"))
        importance_rows.extend(
            {
                "experiment_id": experiment_id,
                "feature": feature,
                "importance_gain": float(gain),
            }
            for feature, gain in zip(
                columns,
                model.booster_.feature_importance(importance_type="gain"),
            )
        )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "profile": args.profile,
        "panel": args.panel,
        "evaluation_protocol": "near_fully_observed",
        "full_exposure_split_method": "stable_user_hash_v1",
        "seconds": time.perf_counter() - start,
        "experiments": results,
    }
    write_json(output / f"results_{args.panel}.json", result)
    pd.DataFrame(
        [{"experiment_id": experiment_id, **metrics} for experiment_id, metrics in results.items()]
    ).to_csv(output / f"summary_{args.panel}.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(output / f"daily_stability_{args.panel}.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(
        output / f"feature_importance_{args.panel}.csv", index=False
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
