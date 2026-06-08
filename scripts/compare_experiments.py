from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    stage_choices = [
        "recall",
        "ranking",
        "candidate_ranking",
        "deep_candidate_ranking",
        "rank_mix",
        "din_ranking_loss",
        "din_sequence_enhancements",
    ]
    parser.add_argument("--stage", choices=stage_choices)
    parser.add_argument("--stage-a", choices=stage_choices)
    parser.add_argument("--stage-b", choices=stage_choices)
    parser.add_argument("--experiment-a", required=True)
    parser.add_argument("--experiment-b", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--panel", default="full_val")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output")
    args = parser.parse_args()

    stage_a = args.stage_a or args.stage
    stage_b = args.stage_b or args.stage
    if not stage_a or not stage_b:
        parser.error("Use --stage, or provide both --stage-a and --stage-b.")
    artifact_root = Path(__file__).resolve().parents[1] / "artifacts" / args.profile
    root_a = artifact_root / stage_a
    root_b = artifact_root / stage_b
    suffix = f"{args.panel}_{args.seed}_group_metrics.parquet"
    a = pd.read_parquet(root_a / f"{args.experiment_a}_{suffix}")
    b = pd.read_parquet(root_b / f"{args.experiment_b}_{suffix}")
    paired = (
        a.groupby("user_id")[args.metric]
        .mean()
        .rename("a")
        .to_frame()
        .join(b.groupby("user_id")[args.metric].mean().rename("b"), how="inner")
    )
    differences = (paired["b"] - paired["a"]).to_numpy()
    rng = np.random.default_rng(args.seed)
    bootstrap = np.array(
        [rng.choice(differences, len(differences), replace=True).mean() for _ in range(args.bootstrap_samples)]
    )
    result = {
        "stage_a": stage_a,
        "stage_b": stage_b,
        "experiment_a": args.experiment_a,
        "experiment_b": args.experiment_b,
        "metric": args.metric,
        "paired_users": len(differences),
        "mean_difference_b_minus_a": float(differences.mean()),
        "ci_95_low": float(np.quantile(bootstrap, 0.025)),
        "ci_95_high": float(np.quantile(bootstrap, 0.975)),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
