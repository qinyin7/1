from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_candidate_ranking import evaluate_ranked, load_or_build_candidate_frames
from scripts.run_experiment import load_processed
from src.common import artifact_dir, load_profile, set_seed, write_json
from src.deep_ranking_models import (
    CandidateFeatureEncoder,
    DIN,
    build_history_tables,
    build_history_indices,
    build_listwise_groups,
    build_pair_indices,
    build_rolling_history_tables,
    predict_torch_ranker,
    train_listwise_ranker,
    train_pairwise_ranker,
    train_torch_ranker,
)
from src.evaluation import build_daily_ground_truth, recall_group_metrics


SELECTION_TRAIN_END = 20200829
SELECTION_EVAL_START = 20200830
LOSS_VARIANTS = {
    "DL1.din_bce": {"loss": "bce", "pointwise_weight": None},
    "DL2.din_bpr": {"loss": "pairwise", "pointwise_weight": 0.0},
    "DL3.din_hybrid": {"loss": "pairwise", "pointwise_weight": 0.2},
    "DL4.din_bpr_budget": {
        "loss": "pairwise",
        "pointwise_weight": 0.0,
        "epochs": 36,
    },
    "DL5.din_hybrid_budget": {
        "loss": "pairwise",
        "pointwise_weight": 0.2,
        "epochs": 36,
    },
    "DL6.din_bpr_wide": {
        "loss": "pairwise",
        "pointwise_weight": 0.0,
        "epochs": 12,
        "negatives_per_positive": 32,
    },
    "DL7.din_listnet": {
        "loss": "listwise",
        "pointwise_weight": None,
        "epochs": 5,
    },
    "DL8.din_bce_rolling": {
        "loss": "bce",
        "pointwise_weight": None,
        "rolling_history": True,
    },
}


def paired_bootstrap_difference(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
    metric: str,
    seed: int,
    samples: int = 1000,
) -> dict:
    paired = (
        baseline.groupby("user_id")[metric]
        .mean()
        .rename("baseline")
        .to_frame()
        .join(
            challenger.groupby("user_id")[metric].mean().rename("challenger"),
            how="inner",
        )
    )
    differences = (paired["challenger"] - paired["baseline"]).to_numpy()
    rng = np.random.default_rng(seed)
    bootstrap = np.array(
        [
            rng.choice(differences, len(differences), replace=True).mean()
            for _ in range(samples)
        ]
    )
    return {
        "mean_difference": float(differences.mean()),
        "ci_95_low": float(np.quantile(bootstrap, 0.025)),
        "ci_95_high": float(np.quantile(bootstrap, 0.975)),
    }


def evaluation_context(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    frame: pd.DataFrame,
    split: str,
    minimum_date: int | None,
    history_splits: list[str],
) -> tuple[dict, set[int], int]:
    history = interactions[interactions["split"].isin(history_splits)]
    seen = history.groupby("user_id")["video_id"].apply(set).to_dict()
    truth = build_daily_ground_truth(
        interactions,
        split,
        excluded_items_by_user=seen,
    )
    if minimum_date is not None:
        truth = {key: values for key, values in truth.items() if key[1] >= minimum_date}
    evaluated_users = set(frame["user_id"].unique())
    truth = {key: values for key, values in truth.items() if key[0] in evaluated_users}
    panel_end = max(date for _, date in truth)
    catalog_size = int(items["first_seen_date"].le(panel_end).sum())
    return truth, set(history["video_id"].unique()), catalog_size


def train_variant(
    experiment_id: str,
    variant: dict,
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    train_history: tuple,
    evaluation_history: tuple,
    truth: dict,
    known_items: set[int],
    catalog_size: int,
    profile: dict,
    seed: int,
    epochs: int,
    batch_size: int,
    negatives_per_positive: int,
    device: torch.device,
    train_history_indices=None,
    evaluation_history_indices=None,
) -> tuple[dict, list[dict], pd.DataFrame, dict]:
    set_seed(seed)
    variant_epochs = int(variant.get("epochs", epochs))
    variant_negatives = int(
        variant.get("negatives_per_positive", negatives_per_positive)
    )
    encoder = CandidateFeatureEncoder.fit(train_frame)
    model = DIN(
        encoder.cardinalities,
        len(encoder.dense_mean),
        profile["deep_ranking_embedding_dim"],
    )
    train_sparse, train_dense = encoder.encode(train_frame)
    group_count = 0
    if variant["loss"] == "bce":
        labels = train_frame[["label_complete"]].to_numpy(dtype="float32")
        training_history, train_seconds = train_torch_ranker(
            model,
            train_sparse,
            train_dense,
            labels,
            variant_epochs,
            batch_size,
            device,
            train_history,
            train_history_indices,
        )
        pair_count = 0
    elif variant["loss"] == "pairwise":
        pair_frame = train_frame.reset_index(drop=True)
        positive_indices, negative_indices = build_pair_indices(
            pair_frame,
            variant_negatives,
            seed,
        )
        training_history, train_seconds = train_pairwise_ranker(
            model,
            train_sparse,
            train_dense,
            positive_indices,
            negative_indices,
            variant_epochs,
            batch_size,
            device,
            train_history,
            variant["pointwise_weight"],
            train_history_indices,
        )
        pair_count = len(positive_indices)
    else:
        listwise_frame = train_frame.reset_index(drop=True)
        group_indices, group_labels = build_listwise_groups(listwise_frame)
        group_batch_size = max(8, batch_size // group_indices.shape[1])
        training_history, train_seconds = train_listwise_ranker(
            model,
            train_sparse,
            train_dense,
            group_indices,
            group_labels,
            variant_epochs,
            group_batch_size,
            device,
            train_history,
            train_history_indices,
        )
        pair_count = 0
        group_count = len(group_indices)
    evaluation_sparse, evaluation_dense = encoder.encode(evaluation_frame)
    scores, prediction_seconds = predict_torch_ranker(
        model,
        evaluation_sparse,
        evaluation_dense,
        batch_size,
        device,
        evaluation_history,
        evaluation_history_indices,
    )
    ranked = evaluation_frame[["user_id", "date", "video_id"]].copy()
    ranked["score"] = scores[:, 0]
    metrics, recommendations = evaluate_ranked(
        ranked,
        truth,
        known_items,
        catalog_size,
        profile["recall_k"],
    )
    metrics.pop("average_log_popularity_at_100", None)
    metrics.update(
        {
            "loss": variant["loss"],
            "pointwise_weight": variant["pointwise_weight"],
            "train_rows": len(train_frame),
            "evaluation_rows": len(evaluation_frame),
            "pair_count": pair_count,
            "listwise_group_count": group_count,
            "epochs": variant_epochs,
            "batch_size": batch_size,
            "negatives_per_positive": variant_negatives if pair_count else 0,
            "train_seconds": train_seconds,
            "prediction_seconds": prediction_seconds,
            "device": str(device),
            "rolling_history": bool(variant.get("rolling_history", False)),
        }
    )
    group_metrics = recall_group_metrics(recommendations, truth, profile["recall_k"])
    checkpoint = {
        "experiment_id": experiment_id,
        "model_state": model.state_dict(),
        "encoder": encoder.state_dict(),
        "history_length": profile["din_history_length"],
        "loss": variant,
    }
    return metrics, training_history, group_metrics, checkpoint


def main() -> None:
    raise RuntimeError(
        "This legacy entrypoint uses logged temporal validation and is disabled. "
        "Use run_deep_candidate_ranking.py --panel full_val until this ablation is migrated."
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="local_8gb_large")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--negatives-per-positive", type=int, default=8)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    epochs = args.epochs or profile["deep_ranking_epochs"]
    batch_size = args.batch_size or profile["deep_ranking_batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_processed(args.profile)
    valid_frame, test_frame, test_truth, known_items, catalog_size = (
        load_or_build_candidate_frames(
            data["interactions"],
            data["items"],
            args.profile,
            profile["max_eval_users"],
            150,
            300,
            args.seed,
            False,
        )
    )
    output = artifact_dir(args.profile, "din_ranking_loss")
    selection_train = valid_frame[valid_frame["date"].le(SELECTION_TRAIN_END)].reset_index(
        drop=True
    )
    selection_eval = valid_frame[
        valid_frame["date"].ge(SELECTION_EVAL_START)
    ].reset_index(drop=True)
    selection_encoder = CandidateFeatureEncoder.fit(selection_train)
    selection_history = build_history_tables(
        data["interactions"],
        data["items"],
        ["train"],
        selection_encoder,
        profile["din_history_length"],
    )
    selection_dates = sorted(
        set(selection_train["date"].unique()) | set(selection_eval["date"].unique())
    )
    selection_rolling_history, selection_offsets = build_rolling_history_tables(
        data["interactions"],
        data["items"],
        selection_dates,
        selection_encoder,
        profile["din_history_length"],
    )
    selection_train_history_indices = build_history_indices(
        selection_train, selection_encoder, selection_offsets
    )
    selection_eval_history_indices = build_history_indices(
        selection_eval, selection_encoder, selection_offsets
    )
    selection_truth, selection_known, selection_catalog = evaluation_context(
        data["interactions"],
        data["items"],
        selection_eval,
        "valid",
        SELECTION_EVAL_START,
        ["train"],
    )

    selection_results = {}
    selection_group_metrics = {}
    for experiment_id, variant in LOSS_VARIANTS.items():
        rolling = variant.get("rolling_history", False)
        metrics, training_history, group_metrics, checkpoint = train_variant(
            experiment_id,
            variant,
            selection_train,
            selection_eval,
            selection_rolling_history if rolling else selection_history,
            selection_rolling_history if rolling else selection_history,
            selection_truth,
            selection_known,
            selection_catalog,
            profile,
            args.seed,
            epochs,
            batch_size,
            args.negatives_per_positive,
            device,
            selection_train_history_indices if rolling else None,
            selection_eval_history_indices if rolling else None,
        )
        selection_results[experiment_id] = metrics
        selection_group_metrics[experiment_id] = group_metrics
        write_json(
            output / f"{experiment_id}_selection_training_history.json",
            {"experiment_id": experiment_id, "epochs": training_history},
        )
        group_metrics.to_parquet(
            output / f"{experiment_id}_selection_{args.seed}_group_metrics.parquet",
            index=False,
        )
        torch.save(checkpoint, output / f"{experiment_id}_selection.pt")
        print(json.dumps({experiment_id: metrics}, ensure_ascii=False, indent=2))

    baseline_id = "DL1.din_bce"
    comparisons = {}
    eligible_challengers = []
    for experiment_id in LOSS_VARIANTS:
        if experiment_id == baseline_id:
            continue
        recall_comparison = paired_bootstrap_difference(
            selection_group_metrics[baseline_id],
            selection_group_metrics[experiment_id],
            f"recall_at_{profile['recall_k']}",
            args.seed,
        )
        ndcg_comparison = paired_bootstrap_difference(
            selection_group_metrics[baseline_id],
            selection_group_metrics[experiment_id],
            f"ndcg_at_{profile['recall_k']}",
            args.seed,
        )
        eligible = (
            recall_comparison["ci_95_low"] > 0
            and ndcg_comparison["ci_95_high"] >= 0
        )
        comparisons[experiment_id] = {
            "versus": baseline_id,
            "recall": recall_comparison,
            "ndcg": ndcg_comparison,
            "eligible": eligible,
        }
        if eligible:
            eligible_challengers.append(experiment_id)
    selection_summary = pd.DataFrame(
        [{"experiment_id": key, **value} for key, value in selection_results.items()]
    ).sort_values(["recall_at_100", "ndcg_at_100"], ascending=False)
    selection_summary.to_csv(output / "selection_summary.csv", index=False)
    mean_winner = str(selection_summary.iloc[0]["experiment_id"])
    winner = (
        max(
            eligible_challengers,
            key=lambda experiment_id: selection_results[experiment_id]["recall_at_100"],
        )
        if eligible_challengers
        else baseline_id
    )
    write_json(
        output / "selection_results.json",
        {
            "profile": args.profile,
            "selection_train_dates": [20200827, 20200828, 20200829],
            "selection_eval_dates": [20200830, 20200831],
            "selection_rule": "Recall CI lower bound > 0 and NDCG is not significantly worse versus DL1.din_bce.",
            "mean_winner": mean_winner,
            "winner": winner,
            "experiments": selection_results,
            "comparisons": comparisons,
        },
    )
    if args.selection_only:
        return
    if winner == baseline_id:
        existing_results = artifact_dir(args.profile, "deep_candidate_ranking") / "results.json"
        existing_metrics = json.loads(existing_results.read_text(encoding="utf-8"))[
            "experiments"
        ]["DR2.din"]
        write_json(
            output / "frozen_test_results.json",
            {
                "profile": args.profile,
                "selected_on_validation": winner,
                "experiment_id": "DR2.din",
                "metrics": existing_metrics,
                "test_rerun": False,
                "reason": "The selected model is the already-audited static BCE DIN.",
            },
        )
        return

    final_id = f"{winner}.frozen_test"
    final_encoder = CandidateFeatureEncoder.fit(valid_frame)
    final_train_history = build_history_tables(
        data["interactions"],
        data["items"],
        ["train"],
        final_encoder,
        profile["din_history_length"],
    )
    final_test_history = build_history_tables(
        data["interactions"],
        data["items"],
        ["train", "valid"],
        final_encoder,
        profile["din_history_length"],
    )
    winner_rolling = bool(LOSS_VARIANTS[winner].get("rolling_history", False))
    if winner_rolling:
        final_dates = sorted(
            set(valid_frame["date"].unique()) | set(test_frame["date"].unique())
        )
        final_rolling_history, final_offsets = build_rolling_history_tables(
            data["interactions"],
            data["items"],
            final_dates,
            final_encoder,
            profile["din_history_length"],
        )
        final_train_history = final_rolling_history
        final_test_history = final_rolling_history
        final_train_history_indices = build_history_indices(
            valid_frame, final_encoder, final_offsets
        )
        final_test_history_indices = build_history_indices(
            test_frame, final_encoder, final_offsets
        )
    else:
        final_train_history_indices = final_test_history_indices = None
    final_metrics, final_history, final_groups, final_checkpoint = train_variant(
        final_id,
        LOSS_VARIANTS[winner],
        valid_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
        final_train_history,
        final_test_history,
        test_truth,
        known_items,
        catalog_size,
        profile,
        args.seed,
        epochs,
        batch_size,
        args.negatives_per_positive,
        device,
        final_train_history_indices,
        final_test_history_indices,
    )
    write_json(
        output / "frozen_test_results.json",
        {
            "profile": args.profile,
            "selected_on_validation": winner,
            "experiment_id": final_id,
            "metrics": final_metrics,
        },
    )
    write_json(
        output / f"{final_id}_training_history.json",
        {"experiment_id": final_id, "epochs": final_history},
    )
    final_groups.to_parquet(
        output / f"{final_id}_test_{args.seed}_group_metrics.parquet",
        index=False,
    )
    torch.save(final_checkpoint, output / f"{final_id}.pt")
    print(json.dumps({final_id: final_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
