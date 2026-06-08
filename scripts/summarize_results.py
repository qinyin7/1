from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="local_8gb_large")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "artifacts" / args.profile
    for stage in ("recall", "ranking"):
        path = root / stage / "results.csv"
        if not path.exists():
            print(f"{stage}: no results")
            continue
        frame = pd.read_csv(path)
        if "schema_version" in frame:
            frame = frame[frame["schema_version"] == frame["schema_version"].max()]
        if "panel" in frame:
            frame = frame[frame["panel"].notna()]
        dedupe_key = (
            ["profile", "panel", "experiment_id", "seed"]
            if "experiment_id" in frame
            else ["profile", "panel", "model", "seed"]
        )
        frame = frame.sort_values("created_at_utc").drop_duplicates(dedupe_key, keep="last")
        key = ["panel", "experiment_id", "model"] if "experiment_id" in frame else ["panel", "model"]
        if stage == "ranking" and "feature_set" in frame:
            key.append("feature_set")
        if stage == "recall" and "train_through" in frame:
            key.append("train_through")
        numeric = [
            column
            for column in frame.select_dtypes("number").columns
            if column not in {"seed", "schema_version"}
        ]
        summary = frame.groupby(key, dropna=False)[numeric].mean().reset_index()
        seed_counts = frame.groupby(key, dropna=False)["seed"].nunique().reset_index(name="seed_count")
        summary = summary.merge(seed_counts, on=key, how="left")
        output = root / stage / "summary.csv"
        summary.to_csv(output, index=False)
        print(f"\n{stage.upper()} ({output})")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
