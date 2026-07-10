from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_24GB = {
    "Recall@10": 0.009218,
    "NDCG@10": 0.876574,
    "Utility@10": 1.053530,
    "Recall@200": 0.139656,
    "NDCG@200": 0.706562,
    "Coverage@200": 0.524497,
    "Utility@200": 0.700544,
}


def _metric(row: pd.Series, name: str) -> str:
    value = row.get(name)
    if pd.isna(value):
        return "-"
    return f"{float(value):.6f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    align = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(row.get(column, "-")) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, align, *rows])


def _load_recall(profile: str) -> pd.DataFrame:
    path = ROOT / "artifacts" / profile / "recall" / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "created_at_utc" in frame:
        key = ["panel", "experiment_id", "seed"]
        frame = frame.sort_values("created_at_utc").drop_duplicates(key, keep="last")
    return frame


def _load_rank_mix(profile: str, panel: str) -> pd.DataFrame:
    path = ROOT / "artifacts" / profile / "rank_mix" / f"summary_{panel}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def recall_table(profile: str, panel: str) -> str:
    frame = _load_recall(profile)
    if frame.empty:
        return f"`{profile}` recall results are not available yet."
    frame = frame[frame["panel"].eq(panel)].copy()
    if frame.empty:
        return f"`{profile}` recall results for `{panel}` are not available yet."

    metric_columns = {
        "Recall@200": "recall_at_200",
        "NDCG@200": "ndcg_at_200",
        "Recall@500": "recall_at_500",
        "NDCG@500": "ndcg_at_500",
        "Cold Recall@200": "cold_recall_at_200",
        "Cold Recall@500": "cold_recall_at_500",
        "Coverage@200": "coverage_at_200",
        "Coverage@500": "coverage_at_500",
    }
    for label, column in metric_columns.items():
        frame[label] = frame.apply(lambda row, metric=column: _metric(row, metric), axis=1)
    frame = frame.rename(columns={"public_experiment_id": "recall_id"})
    return _markdown_table(
        frame,
        ["recall_id", "model", *metric_columns.keys()],
    )


def rank_mix_table(profile: str, panel: str) -> str:
    frame = _load_rank_mix(profile, panel)
    if frame.empty:
        return f"`{profile}` rank_mix summary for `{panel}` is not available yet."

    metric_columns = {
        "Recall@10": "recall_at_10",
        "NDCG@10": "ndcg_at_10",
        "Utility@10": "utility_at_10",
        "Recall@200": "recall_at_200",
        "NDCG@200": "ndcg_at_200",
        "Recall@500": "recall_at_500",
        "NDCG@500": "ndcg_at_500",
        "Coverage@200": "coverage_at_200",
        "Coverage@500": "coverage_at_500",
        "Utility@200": "utility_at_200",
        "Utility@500": "utility_at_500",
    }
    for label, column in metric_columns.items():
        frame[label] = frame.apply(lambda row, metric=column: _metric(row, metric), axis=1)
    return _markdown_table(
        frame,
        ["experiment_id", *metric_columns.keys()],
    )


def baseline_comparison(profile: str, panel: str, experiment_id: str) -> str:
    frame = _load_rank_mix(profile, panel)
    if frame.empty:
        return f"`{profile}` rank_mix summary for `{panel}` is not available yet."
    candidates = frame[frame["experiment_id"].eq(experiment_id)]
    if candidates.empty:
        return f"`{experiment_id}` is not available in `{profile}` `{panel}` summary yet."

    row = candidates.iloc[0]
    metric_map = {
        "Recall@10": "recall_at_10",
        "NDCG@10": "ndcg_at_10",
        "Utility@10": "utility_at_10",
        "Recall@200": "recall_at_200",
        "NDCG@200": "ndcg_at_200",
        "Coverage@200": "coverage_at_200",
        "Utility@200": "utility_at_200",
    }
    rows = []
    for label, column in metric_map.items():
        current = row.get(column)
        baseline = BASELINE_24GB[label]
        if pd.isna(current):
            current_text, diff_text, verdict = "-", "-", "pending"
        else:
            current_value = float(current)
            diff = current_value - baseline
            current_text = f"{current_value:.6f}"
            diff_text = f"{diff:+.6f}"
            verdict = "up" if diff > 0 else "down" if diff < 0 else "flat"
        rows.append(
            {
                "metric": label,
                "full_24gb": f"{baseline:.6f}",
                "full_48gb_optimized": current_text,
                "diff": diff_text,
                "verdict": verdict,
            }
        )
    return _markdown_table(
        pd.DataFrame(rows),
        ["metric", "full_24gb", "full_48gb_optimized", "diff", "verdict"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="full_48gb_optimized")
    parser.add_argument("--panel", choices=["full_val", "full_test"], default="full_val")
    parser.add_argument("--experiment-id", default="rankmix_lambdarank_din_mmr")
    args = parser.parse_args()

    print(f"# {args.profile} {args.panel} summary\n")
    print("## Recall\n")
    print(recall_table(args.profile, args.panel))
    print("\n## RankMix\n")
    print(rank_mix_table(args.profile, args.panel))
    print("\n## 24GB Comparison\n")
    print(baseline_comparison(args.profile, args.panel, args.experiment_id))


if __name__ == "__main__":
    main()
