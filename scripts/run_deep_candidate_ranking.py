from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_candidate_ranking import (
    add_channel_selection_metrics,
    evaluate_ranked,
    evaluate_ranked_at_ks,
    load_or_build_candidate_frames,
)
from scripts.run_experiment import load_processed
from src.common import RESULT_SCHEMA_VERSION, artifact_dir, load_profile, set_seed, write_json
from src.deep_ranking_models import (
    TASK_COLUMNS,
    CandidateFeatureEncoder,
    DIN,
    DeepFM,
    MMoE,
    build_history_tables,
    predict_torch_ranker,
    train_torch_ranker,
)
from src.evaluation import recall_group_metrics


MODEL_IDS = {
    "deepfm": "DR1.deepfm",
    "din": "DR2.din",
    "multitask": "DR3.mmoe_multitask",
}


def save_results(
    output: Path,
    profile_name: str,
    experiment_id: str,
    seed: int,
    metrics: dict,
    training_history: list[dict],
    group_metrics: pd.DataFrame,
    panel: str,
) -> None:
    result_path = output / f"results_{panel}.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "profile": profile_name,
            "panel": panel,
            "evaluation_protocol": "near_fully_observed",
            "experiments": {},
        }
    result["experiments"][experiment_id] = metrics
    write_json(result_path, result)
    summary = pd.DataFrame(
        [
            {"experiment_id": current_id, **current_metrics}
            for current_id, current_metrics in result["experiments"].items()
        ]
    )
    summary.to_csv(output / f"summary_{panel}.csv", index=False)
    write_json(
        output / f"{experiment_id}_{panel}_training_history.json",
        {"experiment_id": experiment_id, "epochs": training_history},
    )
    group_metrics.to_parquet(
        output / f"{experiment_id}_{panel}_{seed}_group_metrics.parquet",
        index=False,
    )
    daily_frames = []
    for path in output.glob(f"*_{panel}_{seed}_group_metrics.parquet"):
        current = pd.read_parquet(path)
        metric_columns = [
            column
            for column in current.columns
            if column.startswith(("recall_at_", "ndcg_at_", "hit_rate_at_"))
        ]
        current = current.groupby("date", as_index=False)[metric_columns].mean()
        current["experiment_id"] = path.name.split(f"_{panel}_{seed}")[0]
        current["seed"] = seed
        daily_frames.append(current)
    pd.concat(daily_frames, ignore_index=True).to_csv(
        output / f"daily_stability_{panel}.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="local_8gb_large")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_IDS), required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--panel", choices=["full_val", "full_test"], default="full_val")
    args = parser.parse_args()

    set_seed(args.seed)
    profile = load_profile(args.profile)
    epochs = args.epochs or profile["deep_ranking_epochs"]
    batch_size = args.batch_size or profile["deep_ranking_batch_size"]
    embedding_dim = profile["deep_ranking_embedding_dim"]
    history_length = profile["din_history_length"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_processed(args.profile)
    train_frame, test_frame, test_truth, known_items, catalog_size = (
        load_or_build_candidate_frames(
            data["interactions"],
            data["items"],
            args.profile,
            profile["max_eval_users"],
            150,
            300,
            args.seed,
            False,
            args.panel,
        )
    )
    encoder = CandidateFeatureEncoder.fit(train_frame)
    output = artifact_dir(args.profile, "deep_candidate_ranking")
    eval_ks = [10, profile["recall_k"]]

    for model_name in args.models:
        experiment_id = MODEL_IDS[model_name]
        set_seed(args.seed)
        if model_name == "deepfm":
            model = DeepFM(encoder.cardinalities, len(encoder.dense_mean), embedding_dim)
            train_labels = train_frame[["label_complete"]].to_numpy(dtype=np.float32)
            train_history = test_history = None
        elif model_name == "din":
            model = DIN(encoder.cardinalities, len(encoder.dense_mean), embedding_dim)
            train_labels = train_frame[["label_complete"]].to_numpy(dtype=np.float32)
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
        else:
            model = MMoE(encoder.cardinalities, len(encoder.dense_mean), embedding_dim)
            train_labels = train_frame[TASK_COLUMNS].to_numpy(dtype=np.float32)
            train_history = test_history = None

        train_sparse, train_dense = encoder.encode(train_frame)
        training_history, train_seconds = train_torch_ranker(
            model,
            train_sparse,
            train_dense,
            train_labels,
            epochs,
            batch_size,
            device,
            train_history,
        )
        del train_sparse, train_dense, train_labels
        test_sparse, test_dense = encoder.encode(test_frame)
        predictions, prediction_seconds = predict_torch_ranker(
            model,
            test_sparse,
            test_dense,
            batch_size,
            device,
            test_history,
        )
        score_variants = {experiment_id: predictions[:, 0]}
        if model_name == "multitask":
            score_variants = {
                "DR3.mmoe_complete": predictions[:, 0],
                "DR3.mmoe_complete_strong": (
                    0.7 * predictions[:, 0] + 0.3 * predictions[:, 1]
                ),
                "DR3.mmoe_multitask": (
                    0.6 * predictions[:, 0]
                    + 0.3 * predictions[:, 1]
                    + 0.1 * (1 - predictions[:, 2])
                ),
            }
        torch.save(
            {
                "experiment_id": experiment_id,
                "model_state": model.state_dict(),
                "encoder": encoder.state_dict(),
                "embedding_dim": embedding_dim,
                "history_length": history_length if model_name == "din" else None,
            },
            output / f"{experiment_id}_{args.panel}.pt",
        )
        for score_id, scores in score_variants.items():
            evaluation = test_frame[
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
            evaluation["score"] = scores
            metrics, recommendations, group_metrics = evaluate_ranked_at_ks(
                evaluation,
                test_truth,
                known_items,
                catalog_size,
                eval_ks,
            )
            add_channel_selection_metrics(metrics, evaluation, eval_ks, profile["recall_k"])
            metrics.update(
                {
                    "pointwise_complete_auc": float(
                        roc_auc_score(test_frame["label_complete"], predictions[:, 0])
                    ),
                    "pointwise_complete_log_loss": float(
                        log_loss(
                            test_frame["label_complete"],
                            np.clip(predictions[:, 0], 1e-7, 1 - 1e-7),
                        )
                    ),
                    "train_rows": len(train_frame),
                    "test_rows": len(test_frame),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "embedding_dim": embedding_dim,
                    "train_seconds": train_seconds,
                    "prediction_seconds": prediction_seconds,
                    "device": str(device),
                    "panel": args.panel,
                    "evaluation_protocol": "near_fully_observed",
                    "full_exposure_split_method": "stable_user_hash_v1",
                }
            )
            if model_name == "multitask":
                for task_index, task_column in enumerate(TASK_COLUMNS):
                    metrics[f"pointwise_{task_column}_auc"] = float(
                        roc_auc_score(test_frame[task_column], predictions[:, task_index])
                    )
            save_results(
                output,
                args.profile,
                score_id,
                args.seed,
                metrics,
                training_history,
                group_metrics,
                args.panel,
            )
            print(json.dumps({score_id: metrics}, ensure_ascii=False, indent=2))
        del test_sparse, test_dense, predictions, evaluation, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
