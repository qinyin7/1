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
    OOVAwareEnrichedDIN,
    build_content_features,
    build_enriched_history_tables,
    build_history_indices,
    encode_candidate_content,
    predict_torch_ranker,
    train_torch_ranker,
)
from src.evaluation import evaluate_cold_recall, recall_group_metrics


VARIANTS = {
    "DS3.1.din_oov_content": {"id_dropout": 0.0},
    "DS3.2.din_oov_content_dropout": {"id_dropout": 0.5},
}
FINAL_BASELINE_ID = "DL1.din_bce"
INCREMENTAL_BASELINE_ID = "DS2.din_author_content_time"


def train_and_evaluate(
    experiment_id: str,
    id_dropout: float,
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    content_item_ids,
    content_vectors,
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
    history_tables, offsets = build_enriched_history_tables(
        interactions,
        items,
        dates,
        encoder,
        profile["din_history_length"],
        content_item_ids,
        content_vectors,
    )
    train_history_indices = build_history_indices(train_frame, encoder, offsets)
    evaluation_history_indices = build_history_indices(evaluation_frame, encoder, offsets)
    train_sparse, train_dense_base = encoder.encode(train_frame)
    evaluation_sparse, evaluation_dense_base = encoder.encode(evaluation_frame)
    train_content = encode_candidate_content(train_frame, content_item_ids, content_vectors)
    evaluation_content = encode_candidate_content(
        evaluation_frame, content_item_ids, content_vectors
    )
    train_dense = np.concatenate([train_dense_base, train_content], axis=1)
    evaluation_dense = np.concatenate(
        [evaluation_dense_base, evaluation_content], axis=1
    )
    model = OOVAwareEnrichedDIN(
        encoder.cardinalities,
        train_dense.shape[1],
        profile["deep_ranking_embedding_dim"],
        content_vectors.shape[1],
        id_dropout,
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
        ranked, truth, known_items, catalog_size, profile["recall_k"]
    )
    metrics.pop("average_log_popularity_at_100", None)
    train_candidate_items = set(train_frame["video_id"].unique())
    oov_metrics = evaluate_cold_recall(
        recommendations, truth, train_candidate_items, profile["recall_k"]
    )
    metrics.update(
        {
            "candidate_oov_recall_at_100": oov_metrics["cold_recall_at_100"],
            "candidate_oov_evaluated_groups": oov_metrics["cold_evaluated_groups"],
            "candidate_oov_row_rate": float((evaluation_sparse[:, 1] == 0).mean()),
            "train_rows": len(train_frame),
            "evaluation_rows": len(evaluation_frame),
            "epochs": profile["deep_ranking_epochs"],
            "batch_size": profile["deep_ranking_batch_size"],
            "history_length": profile["din_history_length"],
            "content_dimension": content_vectors.shape[1],
            "candidate_oov_content_enabled": True,
            "id_dropout": id_dropout,
            "train_seconds": train_seconds,
            "prediction_seconds": prediction_seconds,
            "device": str(device),
        }
    )
    group_metrics = recall_group_metrics(recommendations, truth, profile["recall_k"])
    checkpoint = {
        "experiment_id": experiment_id,
        "model_state": model.state_dict(),
        "encoder": encoder.state_dict(),
        "history_length": profile["din_history_length"],
        "content_dimension": content_vectors.shape[1],
        "id_dropout": id_dropout,
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
    content_item_ids, content_vectors = build_content_features(data["items"], seed=args.seed)
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
    final_baseline_groups = pd.read_parquet(
        artifact_dir(args.profile, "din_ranking_loss")
        / f"{FINAL_BASELINE_ID}_selection_{args.seed}_group_metrics.parquet"
    )
    incremental_baseline_groups = pd.read_parquet(
        output / f"{INCREMENTAL_BASELINE_ID}_selection_{args.seed}_group_metrics.parquet"
    )
    results = {}
    group_results = {}
    checkpoints = {}
    histories = {}
    for experiment_id, variant in VARIANTS.items():
        metrics, history, groups, checkpoint = train_and_evaluate(
            experiment_id,
            variant["id_dropout"],
            selection_train,
            selection_eval,
            data["interactions"],
            data["items"],
            content_item_ids,
            content_vectors,
            selection_truth,
            selection_known,
            selection_catalog,
            profile,
            args.seed,
            device,
        )
        results[experiment_id] = metrics
        group_results[experiment_id] = groups
        checkpoints[experiment_id] = checkpoint
        histories[experiment_id] = history
        groups.to_parquet(
            output / f"{experiment_id}_selection_{args.seed}_group_metrics.parquet",
            index=False,
        )
        torch.save(checkpoint, output / f"{experiment_id}_selection.pt")

    comparisons = {}
    eligible = []
    for experiment_id, groups in group_results.items():
        comparisons[experiment_id] = {}
        for baseline_id, baseline_groups in {
            FINAL_BASELINE_ID: final_baseline_groups,
            INCREMENTAL_BASELINE_ID: incremental_baseline_groups,
        }.items():
            comparisons[experiment_id][baseline_id] = {
                "recall": paired_bootstrap_difference(
                    baseline_groups,
                    groups,
                    f"recall_at_{profile['recall_k']}",
                    args.seed,
                ),
                "ndcg": paired_bootstrap_difference(
                    baseline_groups,
                    groups,
                    f"ndcg_at_{profile['recall_k']}",
                    args.seed,
                ),
            }
        final_comparison = comparisons[experiment_id][FINAL_BASELINE_ID]
        if (
            final_comparison["recall"]["ci_95_low"] > 0
            and final_comparison["ndcg"]["ci_95_high"] >= 0
        ):
            eligible.append(experiment_id)
    winner = (
        max(eligible, key=lambda experiment_id: results[experiment_id]["recall_at_100"])
        if eligible
        else FINAL_BASELINE_ID
    )
    selection_result = {
        "profile": args.profile,
        "selection_train_dates": [20200827, 20200828, 20200829],
        "selection_eval_dates": [20200830, 20200831],
        "winner": winner,
        "experiments": results,
        "comparisons": comparisons,
    }
    write_json(output / "DS3_selection_results.json", selection_result)
    pd.DataFrame(
        [{"experiment_id": key, **value} for key, value in results.items()]
    ).to_csv(output / "DS3_selection_summary.csv", index=False)
    for experiment_id, history in histories.items():
        write_json(
            output / f"{experiment_id}_selection_training_history.json",
            {"experiment_id": experiment_id, "epochs": history},
        )
    print(json.dumps(selection_result, ensure_ascii=False, indent=2))
    if winner == FINAL_BASELINE_ID:
        write_json(
            output / "DS3_frozen_test_results.json",
            {
                "profile": args.profile,
                "selected_on_validation": winner,
                "test_rerun": False,
                "reason": "No OOV content variant passed the Validation threshold.",
            },
        )
        return

    final_metrics, final_history, final_groups, final_checkpoint = train_and_evaluate(
        winner,
        VARIANTS[winner]["id_dropout"],
        valid_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
        data["interactions"],
        data["items"],
        content_item_ids,
        content_vectors,
        test_truth,
        known_items,
        catalog_size,
        profile,
        args.seed,
        device,
    )
    write_json(
        output / "DS3_frozen_test_results.json",
        {
            "profile": args.profile,
            "selected_on_validation": winner,
            "test_rerun": True,
            "metrics": final_metrics,
        },
    )
    write_json(
        output / "DS3_frozen_test_training_history.json",
        {"experiment_id": winner, "epochs": final_history},
    )
    final_groups.to_parquet(
        output / f"{winner}_test_{args.seed}_group_metrics.parquet", index=False
    )
    torch.save(final_checkpoint, output / f"{winner}_frozen_test.pt")


if __name__ == "__main__":
    main()
