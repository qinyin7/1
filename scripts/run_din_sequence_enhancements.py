from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_candidate_ranking import evaluate_ranked, load_or_build_candidate_frames
from scripts.run_din_ranking_loss import (
    SELECTION_EVAL_START,
    SELECTION_TRAIN_END,
    evaluation_context,
    paired_bootstrap_difference,
)
from scripts.run_experiment import load_processed
from src.common import artifact_dir, load_profile, set_seed, write_json
from src.deep_ranking_models import (
    CandidateFeatureEncoder,
    MultiBehaviorDIN,
    build_history_indices,
    build_multibehavior_history_tables,
    predict_torch_ranker,
    train_torch_ranker,
)
from src.evaluation import recall_group_metrics


EXPERIMENT_ID = "DS1.din_multibehavior"
BASELINE_ID = "DL1.din_bce"


def train_and_evaluate(
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    truth: dict,
    known_items: set[int],
    catalog_size: int,
    profile: dict,
    seed: int,
    device: torch.device,
) -> tuple[dict, list[dict], pd.DataFrame, dict]:
    set_seed(seed)
    encoder = CandidateFeatureEncoder.fit(train_frame)
    dates = sorted(
        set(train_frame["date"].unique()) | set(evaluation_frame["date"].unique())
    )
    history_tables, offsets = build_multibehavior_history_tables(
        interactions,
        items,
        dates,
        encoder,
        profile["din_history_length"],
    )
    train_history_indices = build_history_indices(train_frame, encoder, offsets)
    evaluation_history_indices = build_history_indices(evaluation_frame, encoder, offsets)
    train_sparse, train_dense = encoder.encode(train_frame)
    model = MultiBehaviorDIN(
        encoder.cardinalities,
        len(encoder.dense_mean),
        profile["deep_ranking_embedding_dim"],
    )
    labels = train_frame[["label_complete"]].to_numpy(dtype="float32")
    training_history, train_seconds = train_torch_ranker(
        model,
        train_sparse,
        train_dense,
        labels,
        profile["deep_ranking_epochs"],
        profile["deep_ranking_batch_size"],
        device,
        history_tables,
        train_history_indices,
    )
    evaluation_sparse, evaluation_dense = encoder.encode(evaluation_frame)
    scores, prediction_seconds = predict_torch_ranker(
        model,
        evaluation_sparse,
        evaluation_dense,
        profile["deep_ranking_batch_size"],
        device,
        history_tables,
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
            "train_rows": len(train_frame),
            "evaluation_rows": len(evaluation_frame),
            "epochs": profile["deep_ranking_epochs"],
            "batch_size": profile["deep_ranking_batch_size"],
            "history_length": profile["din_history_length"],
            "history_features": [
                "item",
                "category",
                "complete",
                "strong",
                "short",
                "watch_ratio",
            ],
            "train_seconds": train_seconds,
            "prediction_seconds": prediction_seconds,
            "device": str(device),
        }
    )
    group_metrics = recall_group_metrics(recommendations, truth, profile["recall_k"])
    checkpoint = {
        "experiment_id": EXPERIMENT_ID,
        "model_state": model.state_dict(),
        "encoder": encoder.state_dict(),
        "history_length": profile["din_history_length"],
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
    args = parser.parse_args()

    profile = load_profile(args.profile)
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
    selection_train = valid_frame[valid_frame["date"].le(SELECTION_TRAIN_END)].reset_index(
        drop=True
    )
    selection_eval = valid_frame[
        valid_frame["date"].ge(SELECTION_EVAL_START)
    ].reset_index(drop=True)
    selection_truth, selection_known, selection_catalog = evaluation_context(
        data["interactions"],
        data["items"],
        selection_eval,
        "valid",
        SELECTION_EVAL_START,
        ["train"],
    )
    output = artifact_dir(args.profile, "din_sequence_enhancements")
    metrics, training_history, group_metrics, checkpoint = train_and_evaluate(
        selection_train,
        selection_eval,
        data["interactions"],
        data["items"],
        selection_truth,
        selection_known,
        selection_catalog,
        profile,
        args.seed,
        device,
    )
    baseline_path = (
        artifact_dir(args.profile, "din_ranking_loss")
        / f"{BASELINE_ID}_selection_{args.seed}_group_metrics.parquet"
    )
    baseline_groups = pd.read_parquet(baseline_path)
    recall_comparison = paired_bootstrap_difference(
        baseline_groups,
        group_metrics,
        f"recall_at_{profile['recall_k']}",
        args.seed,
    )
    ndcg_comparison = paired_bootstrap_difference(
        baseline_groups,
        group_metrics,
        f"ndcg_at_{profile['recall_k']}",
        args.seed,
    )
    eligible = (
        recall_comparison["ci_95_low"] > 0 and ndcg_comparison["ci_95_high"] >= 0
    )
    selection_result = {
        "profile": args.profile,
        "experiment_id": EXPERIMENT_ID,
        "baseline_id": BASELINE_ID,
        "selection_train_dates": [20200827, 20200828, 20200829],
        "selection_eval_dates": [20200830, 20200831],
        "metrics": metrics,
        "comparison": {
            "recall": recall_comparison,
            "ndcg": ndcg_comparison,
            "eligible_for_frozen_test": eligible,
        },
    }
    write_json(output / "DS1_selection_results.json", selection_result)
    write_json(
        output / "DS1_selection_training_history.json",
        {"experiment_id": EXPERIMENT_ID, "epochs": training_history},
    )
    group_metrics.to_parquet(
        output / f"{EXPERIMENT_ID}_selection_{args.seed}_group_metrics.parquet",
        index=False,
    )
    torch.save(checkpoint, output / f"{EXPERIMENT_ID}_selection.pt")
    print(json.dumps(selection_result, ensure_ascii=False, indent=2))

    if not eligible:
        write_json(
            output / "DS1_frozen_test_results.json",
            {
                "profile": args.profile,
                "experiment_id": EXPERIMENT_ID,
                "test_rerun": False,
                "reason": "Validation selection threshold was not met.",
            },
        )
        return

    final_metrics, final_history, final_groups, final_checkpoint = train_and_evaluate(
        valid_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
        data["interactions"],
        data["items"],
        test_truth,
        known_items,
        catalog_size,
        profile,
        args.seed,
        device,
    )
    write_json(
        output / "DS1_frozen_test_results.json",
        {
            "profile": args.profile,
            "experiment_id": EXPERIMENT_ID,
            "test_rerun": True,
            "metrics": final_metrics,
        },
    )
    write_json(
        output / "DS1_frozen_test_training_history.json",
        {"experiment_id": EXPERIMENT_ID, "epochs": final_history},
    )
    final_groups.to_parquet(
        output / f"{EXPERIMENT_ID}_test_{args.seed}_group_metrics.parquet", index=False
    )
    torch.save(final_checkpoint, output / f"{EXPERIMENT_ID}_frozen_test.pt")


if __name__ == "__main__":
    main()
