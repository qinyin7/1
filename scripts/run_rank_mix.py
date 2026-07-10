from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRanker
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_candidate_ranking import (
    add_channel_selection_metrics,
    evaluate_ranked_at_ks,
    load_or_build_candidate_frames,
)
from scripts.run_experiment import load_processed
from src.candidate_features import EXPERIMENT_FEATURES
from src.common import (
    RESULT_SCHEMA_VERSION,
    artifact_dir,
    load_profile,
    ranking_eval_ks,
    resolve_candidate_limits,
    set_seed,
    write_json,
)
from src.deep_ranking_models import (
    CandidateFeatureEncoder,
    DIN,
    MultiBehaviorDIN,
    build_history_tables,
    build_history_indices,
    build_multibehavior_history_tables,
    predict_torch_ranker,
    train_torch_ranker,
)
from src.reranking import apply_mmr_rerank


def _rank_series(frame: pd.DataFrame, score_column: str) -> pd.Series:
    return frame.groupby(["user_id", "date"], sort=False)[score_column].rank(
        method="first",
        ascending=False,
    )


def _evaluate_scores(
    frame: pd.DataFrame,
    score_column: str,
    test_truth: dict,
    known_items: set[int],
    catalog_size: int,
    eval_ks: list[int],
    primary_k: int,
) -> tuple[dict, pd.DataFrame]:
    evaluation = frame[
        [
            "user_id",
            "date",
            "video_id",
            "label_complete",
            "label_strong",
            "label_short",
            "itemcf_present",
            "content_present",
            "tower_present",
        ]
    ].copy()
    evaluation["score"] = frame[score_column].to_numpy()
    metrics, _, group_metrics = evaluate_ranked_at_ks(
        evaluation,
        test_truth,
        known_items,
        catalog_size,
        eval_ks,
    )
    add_channel_selection_metrics(metrics, evaluation, eval_ks, primary_k)
    return metrics, group_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="full_24gb")
    parser.add_argument("--panel", choices=["full_val", "full_test"], default="full_val")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-per-channel", type=int)
    parser.add_argument("--max-candidates-per-group", type=int)
    parser.add_argument("--estimators", type=int)
    parser.add_argument("--din-weight", type=float, default=0.6)
    parser.add_argument("--pr3-weight", type=float, default=0.4)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--mmr-lambda", type=float, default=0.9)
    parser.add_argument("--mmr-category-weight", type=float, default=0.6)
    parser.add_argument("--mmr-author-weight", type=float, default=0.4)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--din-variant", choices=["basic", "multibehavior"])
    parser.add_argument("--cpu-threads", type=int, default=0)
    args = parser.parse_args()

    if args.cpu_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
        torch.set_num_threads(args.cpu_threads)

    set_seed(args.seed)
    profile = load_profile(args.profile)
    top_per_channel, max_candidates_per_group = resolve_candidate_limits(
        profile,
        args.top_per_channel,
        args.max_candidates_per_group,
    )
    estimators = args.estimators or profile["lightgbm_estimators"]
    eval_ks = ranking_eval_ks(profile)
    output = artifact_dir(args.profile, "rank_mix")
    start = time.perf_counter()
    data = load_processed(args.profile)
    train_frame, test_frame, test_truth, known_items, catalog_size = (
        load_or_build_candidate_frames(
            data["interactions"],
            data["items"],
            args.profile,
            profile["max_eval_users"],
            top_per_channel,
            max_candidates_per_group,
            args.seed,
            False,
            args.panel,
        )
    )

    columns = EXPERIMENT_FEATURES["lambdarank_full_features"]
    pr3_model = LGBMRanker(
        objective="lambdarank",
        n_estimators=estimators,
        learning_rate=0.05,
        num_leaves=63,
        random_state=args.seed,
        n_jobs=args.cpu_threads if args.cpu_threads > 0 else -1,
        verbosity=-1,
    )
    train_sorted = train_frame.sort_values(["user_id", "date"])
    groups = train_sorted.groupby(["user_id", "date"], sort=False).size().to_numpy()
    pr3_start = time.perf_counter()
    pr3_model.fit(train_sorted[columns], train_sorted["label"], group=groups)
    pr3_seconds = time.perf_counter() - pr3_start
    test_frame = test_frame.copy()
    test_frame["pr3_score"] = pr3_model.predict(test_frame[columns])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs or profile["deep_ranking_epochs"]
    batch_size = args.batch_size or profile["deep_ranking_batch_size"]
    embedding_dim = profile["deep_ranking_embedding_dim"]
    history_length = profile["din_history_length"]
    encoder = CandidateFeatureEncoder.fit(train_frame)
    din_variant = args.din_variant or profile.get("rankmix_din_variant", "basic")
    if din_variant == "multibehavior":
        model = MultiBehaviorDIN(encoder.cardinalities, len(encoder.dense_mean), embedding_dim)
        history_dates = sorted(
            set(train_frame["date"].unique()) | set(test_frame["date"].unique())
        )
        history_tables, history_offsets = build_multibehavior_history_tables(
            data["interactions"],
            data["items"],
            history_dates,
            encoder,
            history_length,
        )
        train_history = history_tables
        test_history = history_tables
        train_history_indices = build_history_indices(train_frame, encoder, history_offsets)
        test_history_indices = build_history_indices(test_frame, encoder, history_offsets)
    else:
        model = DIN(encoder.cardinalities, len(encoder.dense_mean), embedding_dim)
        train_history = build_history_tables(
            data["interactions"],
            data["items"],
            ["train"],
            encoder,
            history_length,
        )
        test_history = build_history_tables(
            data["interactions"],
            data["items"],
            ["train", "valid"],
            encoder,
            history_length,
        )
        train_history_indices = None
        test_history_indices = None
    train_sparse, train_dense = encoder.encode(train_frame)
    train_labels = train_frame[["label_complete"]].to_numpy(dtype=np.float32)
    din_history, din_train_seconds = train_torch_ranker(
        model,
        train_sparse,
        train_dense,
        train_labels,
        epochs,
        batch_size,
        device,
        train_history,
        train_history_indices,
    )
    del train_sparse, train_dense, train_labels
    test_sparse, test_dense = encoder.encode(test_frame)
    din_predictions, din_prediction_seconds = predict_torch_ranker(
        model,
        test_sparse,
        test_dense,
        batch_size,
        device,
        test_history,
        test_history_indices,
    )
    test_frame["din_score"] = din_predictions[:, 0]
    test_frame["pr3_rank"] = _rank_series(test_frame, "pr3_score")
    test_frame["din_rank"] = _rank_series(test_frame, "din_score")
    test_frame["rank_mix_score"] = (
        args.din_weight / (args.rrf_k + test_frame["din_rank"])
        + args.pr3_weight / (args.rrf_k + test_frame["pr3_rank"])
    )
    test_frame = apply_mmr_rerank(
        test_frame,
        score_column="rank_mix_score",
        output_column="rank_mix_mmr_score",
        lambda_relevance=args.mmr_lambda,
        category_weight=args.mmr_category_weight,
        author_weight=args.mmr_author_weight,
    )

    experiments = {}
    group_frames = []
    score_columns = {
        "lambdarank_full_features_refit": "pr3_score",
        "din_sequence_ranker_refit": "din_score",
        "rankmix_lambdarank_din": "rank_mix_score",
        "rankmix_lambdarank_din_mmr": "rank_mix_mmr_score",
    }
    for experiment_id, score_column in score_columns.items():
        metrics, group_metrics = _evaluate_scores(
            test_frame,
            score_column,
            test_truth,
            known_items,
            catalog_size,
            eval_ks,
            profile["recall_k"],
        )
        metrics.update(
            {
                "panel": args.panel,
                "train_rows": len(train_frame),
                "test_rows": len(test_frame),
                "pr3_train_seconds": pr3_seconds,
                "din_train_seconds": din_train_seconds,
                "din_prediction_seconds": din_prediction_seconds,
                "device": str(device),
                "din_variant": din_variant
                if experiment_id in {"din_sequence_ranker_refit", "rankmix_lambdarank_din", "rankmix_lambdarank_din_mmr"}
                else None,
                "top_per_channel": top_per_channel,
                "max_candidates_per_group": max_candidates_per_group,
                "estimators": estimators,
                "din_weight": args.din_weight
                if experiment_id in {"rankmix_lambdarank_din", "rankmix_lambdarank_din_mmr"}
                else None,
                "pr3_weight": args.pr3_weight
                if experiment_id in {"rankmix_lambdarank_din", "rankmix_lambdarank_din_mmr"}
                else None,
                "rrf_k": args.rrf_k
                if experiment_id in {"rankmix_lambdarank_din", "rankmix_lambdarank_din_mmr"}
                else None,
                "rerank_strategy": "mmr"
                if experiment_id == "rankmix_lambdarank_din_mmr"
                else None,
                "mmr_lambda": args.mmr_lambda
                if experiment_id == "rankmix_lambdarank_din_mmr"
                else None,
                "mmr_category_weight": args.mmr_category_weight
                if experiment_id == "rankmix_lambdarank_din_mmr"
                else None,
                "mmr_author_weight": args.mmr_author_weight
                if experiment_id == "rankmix_lambdarank_din_mmr"
                else None,
                "evaluation_protocol": "near_fully_observed",
                "full_exposure_split_method": "stable_user_hash_v1",
            }
        )
        if experiment_id == "din_sequence_ranker_refit":
            metrics["pointwise_complete_auc"] = float(
                roc_auc_score(test_frame["label_complete"], test_frame["din_score"])
            )
            metrics["pointwise_complete_log_loss"] = float(
                log_loss(
                    test_frame["label_complete"],
                    np.clip(test_frame["din_score"], 1e-7, 1 - 1e-7),
                )
            )
        experiments[experiment_id] = metrics
        group_metrics["experiment_id"] = experiment_id
        group_metrics["seed"] = args.seed
        group_frames.append(group_metrics)
        group_metrics.to_parquet(
            output / f"{experiment_id}_{args.panel}_{args.seed}_group_metrics.parquet",
            index=False,
        )

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "profile": args.profile,
        "panel": args.panel,
        "evaluation_protocol": "near_fully_observed",
        "full_exposure_split_method": "stable_user_hash_v1",
        "seconds": time.perf_counter() - start,
        "experiments": experiments,
        "din_training_history": din_history,
    }
    write_json(output / f"results_{args.panel}.json", result)
    pd.DataFrame(
        [{"experiment_id": experiment_id, **metrics} for experiment_id, metrics in experiments.items()]
    ).to_csv(output / f"summary_{args.panel}.csv", index=False)
    daily = pd.concat(group_frames, ignore_index=True)
    metric_columns = [
        column
        for column in daily.columns
        if column.startswith(("recall_at_", "ndcg_at_", "hit_rate_at_"))
    ]
    daily.groupby(["experiment_id", "date"], as_index=False)[metric_columns].mean().to_csv(
        output / f"daily_stability_{args.panel}.csv",
        index=False,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
