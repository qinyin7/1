import pandas as pd

from scripts import summarize_48gb_experiment as summary


def test_48gb_summary_tables_use_latest_runs_and_compare_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "ROOT", tmp_path)
    recall_dir = tmp_path / "artifacts" / "full_48gb_optimized" / "recall"
    rank_dir = tmp_path / "artifacts" / "full_48gb_optimized" / "rank_mix"
    recall_dir.mkdir(parents=True)
    rank_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "created_at_utc": "2026-07-03T00:00:00Z",
                "panel": "full_val",
                "experiment_id": "R1.0",
                "public_experiment_id": "itemcf_main",
                "model": "itemcf",
                "seed": 2026,
                "recall_at_200": 0.10,
                "ndcg_at_200": 0.50,
            },
            {
                "created_at_utc": "2026-07-03T01:00:00Z",
                "panel": "full_val",
                "experiment_id": "R1.0",
                "public_experiment_id": "itemcf_main",
                "model": "itemcf",
                "seed": 2026,
                "recall_at_200": 0.11,
                "ndcg_at_200": 0.51,
                "recall_at_500": 0.20,
                "ndcg_at_500": 0.60,
                "cold_recall_at_200": 0.01,
                "cold_recall_at_500": 0.02,
                "coverage_at_200": 0.30,
                "coverage_at_500": 0.40,
            },
        ]
    ).to_csv(recall_dir / "results.csv", index=False)

    pd.DataFrame(
        [
            {
                "experiment_id": "rankmix_lambdarank_din_mmr",
                "recall_at_10": 0.010000,
                "ndcg_at_10": 0.880000,
                "utility_at_10": 1.060000,
                "recall_at_200": 0.140000,
                "ndcg_at_200": 0.710000,
                "recall_at_500": 0.220000,
                "ndcg_at_500": 0.760000,
                "coverage_at_200": 0.530000,
                "coverage_at_500": 0.700000,
                "utility_at_200": 0.710000,
                "utility_at_500": 0.690000,
            }
        ]
    ).to_csv(rank_dir / "summary_full_val.csv", index=False)

    recall_output = summary.recall_table("full_48gb_optimized", "full_val")
    rank_output = summary.rank_mix_table("full_48gb_optimized", "full_val")
    comparison_output = summary.baseline_comparison(
        "full_48gb_optimized",
        "full_val",
        "rankmix_lambdarank_din_mmr",
    )

    assert "0.110000" in recall_output
    assert "0.100000" not in recall_output
    assert "Cold Recall@500" in recall_output
    assert "Utility@500" in rank_output
    assert "0.220000" in rank_output
    assert "+0.000344" in comparison_output
    assert "| Recall@200 | 0.139656 | 0.140000 | +0.000344 | up |" in comparison_output


def test_48gb_summary_reports_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "ROOT", tmp_path)

    assert "not available yet" in summary.recall_table("full_48gb_optimized", "full_val")
    assert "not available yet" in summary.rank_mix_table("full_48gb_optimized", "full_val")
