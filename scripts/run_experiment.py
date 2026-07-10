from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (
    ROOT,
    append_result,
    artifact_dir,
    load_profile,
    load_yaml,
    profile_dir,
    sample_complete_users,
    sample_rows,
    set_seed,
)
from src.data_pipeline import (
    FEATURE_END,
    PROCESSED_SCHEMA_VERSION,
    attach_history_statistics,
    prepare_profile,
)
from src.evaluation import (
    build_daily_ground_truth,
    evaluate_cold_recall,
    evaluate_ranking,
    evaluate_recall,
    filter_daily_recommendations,
    ranking_group_metrics,
    recall_group_metrics,
)
from src.experiment_registry import canonical_experiment_id, public_experiment_id
from src.full_exposure import (
    FULL_EXPOSURE_PANELS,
    build_full_exposure_ground_truth,
    evaluate_full_exposure,
    read_full_exposure_interactions,
)
from src.ranking_models import build_ranking_frame, train_lightgbm, train_logistic
from src.recall_models import (
    ContentRecall,
    FeatureTwoTowerRecall,
    ItemCFRecall,
    PopularRecall,
    TextContentRecall,
    TwoTowerRecall,
    quota_fuse_rankings,
)


def load_processed(profile_name: str) -> dict[str, pd.DataFrame]:
    path = profile_dir(profile_name)
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if summary.get("processed_schema_version") != PROCESSED_SCHEMA_VERSION:
        raise RuntimeError(
            f"Processed profile {profile_name!r} is stale. "
            f"Run: python scripts/run_experiment.py --profile {profile_name} --stage prepare"
        )
    return {
        name: pd.read_parquet(path / f"{name}.parquet")
        for name in ("interactions", "users", "items")
    }


def _int_set_fingerprint(values: set[int]) -> dict:
    ordered = sorted(int(value) for value in values)
    digest = hashlib.sha1(",".join(map(str, ordered)).encode("utf-8")).hexdigest()
    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        "max": ordered[-1] if ordered else None,
        "sha1": digest[:16],
    }


def _recall_model_cache_path(
    profile_name: str,
    profile: dict,
    model_name: str,
    experiment_id: str | None,
    seed: int,
    train_through: str,
    params: dict,
    eligible_items: set[int],
) -> Path:
    payload = {
        "cache_schema_version": 1,
        "processed_schema_version": PROCESSED_SCHEMA_VERSION,
        "profile": profile_name,
        "model": model_name,
        "experiment_id": experiment_id or model_name,
        "seed": seed,
        "train_through": train_through,
        "params": params,
        "eligible_items": _int_set_fingerprint(eligible_items),
        "two_tower_embedding_dim": profile["two_tower_embedding_dim"],
        "two_tower_epochs": profile["two_tower_epochs"],
        "two_tower_batch_size": profile["two_tower_batch_size"],
        "itemcf_history_length": profile["itemcf_history_length"],
    }
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    label = (experiment_id or model_name).replace(".", "_")
    return artifact_dir(profile_name, "recall") / "model_cache" / f"{label}_{key}.pt"


def run_recall(
    profile_name: str,
    profile: dict,
    model_name: str,
    seed: int,
    experiment_id: str | None = None,
    params: dict | None = None,
    panel: str = "full_val",
) -> dict:
    params = params or {}
    train_through = params.get("train_through", "train")
    if train_through == "valid":
        if panel not in {"test", "full_test"}:
            raise ValueError("train_through=valid is only valid for a frozen test panel")
    data = load_processed(profile_name)
    interactions = data["interactions"]
    if train_through == "valid":
        train = interactions[interactions["split"].isin(["train", "valid"])]
    else:
        train = interactions[interactions["split"] == "train"]
    seen = train.groupby("user_id")["video_id"].apply(set).to_dict()
    full_exposure = panel in FULL_EXPOSURE_PANELS
    if full_exposure:
        panel_interactions = read_full_exposure_interactions(
            set(interactions["user_id"].unique()),
            panel,
            profile.get("full_exposure_split_seed", 2026),
            profile.get("full_exposure_validation_fraction", 0.5),
        )
        ground_truth = build_full_exposure_ground_truth(panel_interactions)
        all_users = sorted(ground_truth)
        full_catalog = panel_interactions.attrs["full_catalog"]
    else:
        panel_interactions = interactions
        ground_truth = build_daily_ground_truth(
            panel_interactions, panel, excluded_items_by_user=seen
        )
        all_users = sorted({key[0] for key in ground_truth})
    max_eval_users = profile["max_eval_users"]
    if max_eval_users is not None and len(all_users) > max_eval_users:
        users = (
            pd.Series(all_users)
            .sample(max_eval_users, random_state=profile.get("data_seed", 2026))
            .sort_values()
            .tolist()
        )
    else:
        users = all_users
    if full_exposure:
        ground_truth = {user: positives for user, positives in ground_truth.items() if user in users}
        panel_interactions = panel_interactions[panel_interactions["user_id"].isin(users)]
        panel_interactions.attrs["full_catalog"] = full_catalog
        allowed_items = full_catalog
        eligible_items = set(data["items"]["video_id"].unique())
    else:
        ground_truth = {
            key: positives for key, positives in ground_truth.items() if key[0] in users
        }
        panel_end = int(
            panel_interactions.loc[panel_interactions["split"] == panel, "date"].max()
        )
        eligible_items = set(
            data["items"].loc[data["items"]["first_seen_date"] <= panel_end, "video_id"]
        )
        allowed_items = None
    first_seen_date = data["items"].set_index("video_id")["first_seen_date"].to_dict()
    retrieval_k = max(profile["recall_k"] * 5, 500) if full_exposure else profile["recall_k"] * 5
    start = time.perf_counter()
    model_cache_hit = None
    model_cache_path = None
    if model_name == "fusion":
        component_names = params.get("components", ["popular", "itemcf", "content", "two_tower"])
        component_weights = params.get("weights")
        recommendation_files = [
            artifact_dir(profile_name, "recall") / f"{name}_{panel}_recommendations.json"
            for name in component_names
        ]
        missing = [str(path) for path in recommendation_files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Run component recalls before fusion. Missing: {missing}")
        component_results = [
            {int(user): [int(item) for item in items] for user, items in json.loads(path.read_text()).items()}
            for path in recommendation_files
        ]
        recommendations = {}
        for user in users:
            component_rankings = [component.get(user, []) for component in component_results]
            if component_weights:
                recommendations[user] = quota_fuse_rankings(
                    component_rankings, component_weights, retrieval_k
                )
            else:
                recommendations[user] = quota_fuse_rankings(
                    component_rankings, [1.0] * len(component_rankings), retrieval_k
                )
        model = None
    elif model_name == "popular":
        model = PopularRecall(
            time_decay=params.get("time_decay", True),
            half_life_days=params.get("half_life_days", 7),
            window_days=params.get("window_days"),
        )
        model.fit(train)
    elif model_name == "itemcf":
        model = ItemCFRecall(
            params.get("history_length", profile["itemcf_history_length"]),
            profile["itemcf_neighbors"],
            time_decay=params.get("time_decay", True),
            use_iuf=params.get("iuf", True),
            feedback=params.get("feedback", "complete"),
        )
        model.fit(train)
    elif model_name == "content":
        feature_mode = params.get("features", "first_level_category")
        if feature_mode in {"text_tfidf", "category_and_tfidf"}:
            model = TextContentRecall(
                profile["itemcf_history_length"],
                category_weight=0.15 if feature_mode == "category_and_tfidf" else 0.0,
            )
        else:
            model = ContentRecall(profile["itemcf_history_length"])
        model.fit(train, data["items"], eligible_items)
    elif model_name == "two_tower":
        if params.get("features") == "profile_history_item_content":
            model = FeatureTwoTowerRecall(
                profile["two_tower_embedding_dim"],
                profile["two_tower_epochs"],
                profile["two_tower_batch_size"],
                seed,
                profile["itemcf_history_length"],
                params.get("id_dropout", 0.0),
                params.get("hard_negative_ratio", 0.0),
            )
            model_cache_path = _recall_model_cache_path(
                profile_name,
                profile,
                model_name,
                experiment_id,
                seed,
                train_through,
                params,
                eligible_items,
            )
            if model_cache_path.exists():
                model.load_checkpoint(model_cache_path)
                model_cache_hit = True
            else:
                model.fit(train, data["users"], data["items"], eligible_items)
                model.save_checkpoint(model_cache_path)
                model_cache_hit = False
        else:
            model = TwoTowerRecall(
                profile["two_tower_embedding_dim"],
                profile["two_tower_epochs"],
                profile["two_tower_batch_size"],
                seed,
            )
            model_cache_path = _recall_model_cache_path(
                profile_name,
                profile,
                model_name,
                experiment_id,
                seed,
                train_through,
                params,
                eligible_items,
            )
            if model_cache_path.exists():
                model.load_checkpoint(model_cache_path)
                model_cache_hit = True
            else:
                model.fit(train)
                model.save_checkpoint(model_cache_path)
                model_cache_hit = False
    else:
        raise ValueError(f"Unsupported recall model: {model_name}")
    if model_name != "fusion":
        recommendations = model.recommend(users, seen, retrieval_k, allowed_items=allowed_items)
    elapsed = time.perf_counter() - start
    popularity = train.groupby("video_id").size().to_dict()
    if full_exposure:
        metrics, group_metrics = evaluate_full_exposure(
            recommendations,
            panel_interactions,
            profile["recall_k"],
            popularity,
        )
        evaluated_recommendations = recommendations
    else:
        evaluated_recommendations = filter_daily_recommendations(
            recommendations, ground_truth, first_seen_date, profile["recall_k"]
        )
        metrics = evaluate_recall(
            evaluated_recommendations,
            ground_truth,
            popularity,
            len(eligible_items),
            profile["recall_k"],
        )
        group_metrics = recall_group_metrics(
            evaluated_recommendations, ground_truth, profile["recall_k"]
        )
    metrics.update(
        evaluate_cold_recall(
            evaluated_recommendations,
            ground_truth,
            set(train["video_id"].unique()),
            profile["recall_k"],
        )
    )
    metrics.update(
        {
            "profile": profile_name,
            "stage": "recall",
            "panel": panel,
            "experiment_id": experiment_id or model_name,
            "public_experiment_id": public_experiment_id(experiment_id or model_name),
            "model": model_name,
            "seed": seed,
            "seconds": elapsed,
            "train_through": train_through,
            "evaluation_protocol": "near_fully_observed"
            if full_exposure
            else "biased_logged_temporal_replay",
            "full_exposure_split_method": "stable_user_hash_v1" if full_exposure else None,
            "model_cache_hit": model_cache_hit,
            "model_cache_path": str(model_cache_path.relative_to(ROOT))
            if model_cache_path is not None
            else None,
        }
    )
    output = artifact_dir(profile_name, "recall")
    serializable = {int(user): [int(item) for item in items] for user, items in recommendations.items()}
    recommendation_key = experiment_id or model_name
    Path(output / f"{recommendation_key}_{panel}_recommendations.json").write_text(
        json.dumps(serializable), encoding="utf-8"
    )
    group_metrics.to_parquet(
        output / f"{recommendation_key}_{panel}_{seed}_group_metrics.parquet", index=False
    )
    append_result(output / "results.csv", metrics)
    return metrics


def run_ranking(
    profile_name: str,
    profile: dict,
    model_name: str,
    feature_set: str,
    seed: int,
    experiment_id: str | None = None,
    feature_groups_override: list[str] | None = None,
    panel: str = "full_val",
) -> dict:
    data = load_processed(profile_name)
    frame = build_ranking_frame(**data)
    train = sample_rows(
        frame[(frame["split"] == "train") & (frame["date"] > FEATURE_END)],
        profile["ranking_train_rows"],
        seed,
    )
    if panel in FULL_EXPOSURE_PANELS:
        full_exposure = read_full_exposure_interactions(
            set(data["interactions"]["user_id"].unique()),
            panel,
            profile.get("full_exposure_split_seed", 2026),
            profile.get("full_exposure_validation_fraction", 0.5),
        )
        full_exposure = attach_history_statistics(
            full_exposure,
            data["interactions"][data["interactions"]["split"] == "train"],
        )
        evaluation_frame = build_ranking_frame(full_exposure, data["users"], data["items"])
    else:
        evaluation_frame = frame[frame["split"] == panel]
    evaluation = sample_complete_users(
        evaluation_frame,
        profile["ranking_valid_rows"],
        profile.get("data_seed", 2026),
    )
    groups = feature_groups_override or {
        "basic": ["F0"],
        "behavior": ["F0", "F1", "F3"],
        "full": ["F0", "F1", "F2", "F3", "F5"],
        "no_item_stats": ["F0", "F1", "F2", "F3"],
        "no_user_stats": ["F0", "F2", "F5"],
    }[feature_set]
    start = time.perf_counter()
    if model_name == "logistic":
        model = train_logistic(train, groups)
    elif model_name == "lightgbm":
        model = train_lightgbm(train, groups, profile["lightgbm_estimators"], seed)
    else:
        raise ValueError(f"Unsupported ranking model: {model_name}")
    evaluation = evaluation.copy()
    evaluation["score"] = model.predict(evaluation)
    metrics = evaluate_ranking(evaluation)
    feature_set_label = experiment_id if feature_groups_override else feature_set
    metrics.update(
        {
            "profile": profile_name,
            "stage": "ranking",
            "panel": panel,
            "experiment_id": experiment_id or f"{model_name}_{feature_set}",
            "public_experiment_id": public_experiment_id(
                experiment_id or f"{model_name}_{feature_set}"
            ),
            "model": model_name,
            "feature_set": feature_set_label,
            "seed": seed,
            "train_rows": len(train),
            "valid_rows": len(evaluation),
            "seconds": time.perf_counter() - start,
            "evaluation_protocol": "near_fully_observed"
            if panel in FULL_EXPOSURE_PANELS
            else "logged_exposure_conditional",
            "full_exposure_split_method": "stable_user_hash_v1"
            if panel in FULL_EXPOSURE_PANELS
            else None,
        }
    )
    output = artifact_dir(profile_name, "ranking")
    model.save(output / f"{model_name}_{feature_set_label}.joblib")
    ranking_group_metrics(evaluation).to_parquet(
        output / f"{experiment_id or feature_set}_{panel}_{seed}_group_metrics.parquet", index=False
    )
    append_result(output / "results.csv", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="local_8gb_large",
    )
    parser.add_argument("--stage", choices=["prepare", "recall", "ranking"], required=True)
    parser.add_argument("--model", default="popular")
    parser.add_argument("--feature-set", default="full")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--panel",
        choices=["full_val", "full_test", "valid", "test"],
        default="full_val",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    profile = load_profile(args.profile)
    if args.stage == "prepare":
        result = prepare_profile(args.profile, profile, profile.get("data_seed", 2026))
    elif args.stage == "recall":
        params = {}
        experiment_id = args.experiment_id
        if experiment_id:
            experiment_id = canonical_experiment_id(experiment_id)
            experiments = load_yaml(ROOT / "configs" / "experiments.yaml")["recall_experiments"]
            experiment = next(
                item
                for item in experiments
                if item["id"] == experiment_id or item.get("public_id") == experiment_id
            )
            params = experiment.get("params", {})
            args.model = experiment["model"]
        result = run_recall(
            args.profile, profile, args.model, args.seed, experiment_id, params, args.panel
        )
    else:
        feature_groups_override = None
        experiment_id = args.experiment_id
        if experiment_id:
            experiment_id = canonical_experiment_id(experiment_id)
            experiments = load_yaml(ROOT / "configs" / "experiments.yaml")["ranking_experiments"]
            experiment = next(
                item
                for item in experiments
                if item["id"] == experiment_id or item.get("public_id") == experiment_id
            )
            args.model = "logistic" if experiment["model"] == "logistic_regression" else experiment["model"]
            feature_groups_override = experiment.get("feature_groups")
        result = run_ranking(
            args.profile,
            profile,
            args.model,
            args.feature_set,
            args.seed,
            experiment_id,
            feature_groups_override,
            args.panel,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
