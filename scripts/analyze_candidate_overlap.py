from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_experiment import load_processed
from src.common import artifact_dir, load_profile
from src.evaluation import build_daily_ground_truth, filter_daily_recommendations


def load_recommendations(profile: str, experiment: str, panel: str) -> dict[int, list[int]]:
    path = artifact_dir(profile, "recall") / f"{experiment}_{panel}_recommendations.json"
    return {int(user): [int(item) for item in items] for user, items in json.loads(path.read_text()).items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="local_8gb_large")
    parser.add_argument("--panel", default="test")
    parser.add_argument("--base", nargs="+", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--train-through", choices=["train", "valid"], default="train")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    data = load_processed(args.profile)
    interactions = data["interactions"]
    history_splits = ["train", "valid"] if args.train_through == "valid" else ["train"]
    history = interactions[interactions["split"].isin(history_splits)]
    seen = history.groupby("user_id")["video_id"].apply(set).to_dict()
    ground_truth = build_daily_ground_truth(interactions, args.panel, excluded_items_by_user=seen)
    users = sorted({user for user, _ in ground_truth})
    if profile["max_eval_users"] is not None and len(users) > profile["max_eval_users"]:
        rng = np.random.default_rng(profile.get("data_seed", 2026))
        users = sorted(rng.choice(users, profile["max_eval_users"], replace=False).tolist())
        ground_truth = {key: value for key, value in ground_truth.items() if key[0] in users}
    first_seen = data["items"].set_index("video_id")["first_seen_date"].to_dict()
    k = profile["recall_k"]
    base_results = [load_recommendations(args.profile, experiment, args.panel) for experiment in args.base]
    candidate_result = load_recommendations(args.profile, args.candidate, args.panel)
    daily_base = [
        filter_daily_recommendations(result, ground_truth, first_seen, k) for result in base_results
    ]
    daily_candidate = filter_daily_recommendations(candidate_result, ground_truth, first_seen, k)

    jaccards, unique_rates, base_recalls, union_recalls, unique_hit_counts = [], [], [], [], []
    for key, positives in ground_truth.items():
        base_items = set().union(*(set(result.get(key, [])) for result in daily_base))
        candidate_items = set(daily_candidate.get(key, []))
        union_items = base_items | candidate_items
        jaccards.append(len(base_items & candidate_items) / max(len(union_items), 1))
        unique_items = candidate_items - base_items
        unique_rates.append(len(unique_items) / max(len(candidate_items), 1))
        base_hits = len(base_items & positives)
        union_hits = len(union_items & positives)
        base_recalls.append(base_hits / len(positives))
        union_recalls.append(union_hits / len(positives))
        unique_hit_counts.append(union_hits - base_hits)

    result = {
        "profile": args.profile,
        "panel": args.panel,
        "train_through": args.train_through,
        "base": args.base,
        "candidate": args.candidate,
        "evaluated_groups": len(ground_truth),
        "mean_candidate_jaccard_with_base_union": float(np.mean(jaccards)),
        "mean_candidate_unique_item_rate": float(np.mean(unique_rates)),
        "base_oracle_recall_at_100": float(np.mean(base_recalls)),
        "union_oracle_recall_at_100": float(np.mean(union_recalls)),
        "incremental_oracle_recall_at_100": float(np.mean(union_recalls) - np.mean(base_recalls)),
        "groups_with_unique_positive_hit": int(np.sum(np.array(unique_hit_counts) > 0)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
