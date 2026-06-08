import pandas as pd

from src.evaluation import (
    build_daily_ground_truth,
    build_ground_truth,
    evaluate_recall,
    filter_daily_recommendations,
)
from src.full_exposure import evaluate_full_exposure, split_full_exposure_users


def test_ground_truth_keeps_only_positive_split_rows():
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "video_id": [10, 11, 12],
            "split": ["valid", "valid", "train"],
            "label_complete": [1, 0, 1],
        }
    )
    assert build_ground_truth(frame, "valid") == {1: {10}}


def test_recall_metrics_are_correct_for_simple_case():
    metrics = evaluate_recall({1: [10, 12]}, {1: {10, 11}}, {10: 2, 12: 1}, 4, 2)
    assert metrics["recall_at_2"] == 0.5
    assert metrics["hit_rate_at_2"] == 1.0
    assert metrics["coverage_at_2"] == 0.5


def test_empty_recommendations_count_as_zero():
    metrics = evaluate_recall({1: []}, {1: {10}}, {}, 4, 2)
    assert metrics["recall_at_2"] == 0.0
    assert metrics["evaluated_users"] == 1


def test_ground_truth_excludes_seen_items():
    frame = pd.DataFrame(
        {
            "user_id": [1, 1],
            "video_id": [10, 11],
            "date": [20200827, 20200827],
            "split": ["valid", "valid"],
            "label_complete": [1, 1],
        }
    )
    result = build_daily_ground_truth(frame, "valid", excluded_items_by_user={1: {10}})
    assert result == {(1, 20200827): {11}}


def test_daily_recommendations_filter_future_items():
    ground_truth = {(1, 20200827): {10}}
    filtered = filter_daily_recommendations(
        {1: [20, 10]},
        ground_truth,
        {10: 20200801, 20: 20200901},
        2,
    )
    assert filtered == {(1, 20200827): [10]}


def test_full_exposure_user_split_is_disjoint_and_reproducible():
    first = split_full_exposure_users({1, 2, 3, 4, 5, 6}, seed=2026)
    second = split_full_exposure_users({1, 2, 3, 4, 5, 6}, seed=2026)
    assert first == second
    assert first["full_val"].isdisjoint(first["full_test"])
    assert first["full_val"] | first["full_test"] == {1, 2, 3, 4, 5, 6}
    larger = split_full_exposure_users(set(range(1, 20)), seed=2026)
    assert all(
        user in larger[panel]
        for panel in ("full_val", "full_test")
        for user in first[panel]
    )


def test_full_exposure_metrics_use_observed_feedback_and_skip_unknown_pairs():
    feedback = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "video_id": [10, 11, 12],
            "label_complete": [1, 0, 1],
            "label_strong": [0, 0, 1],
            "label_short": [0, 1, 0],
            "watch_ratio_clipped": [1.0, 0.1, 2.0],
            "utility": [1.0, -0.5, 1.5],
        }
    )
    metrics, groups = evaluate_full_exposure({1: [99, 10, 11]}, feedback, 2)
    assert metrics["invalid_recommendations"] == 1
    assert metrics["precision_at_2"] == 0.5
    assert metrics["recall_at_2"] == 0.5
    assert groups.loc[0, "complete_rate_at_2"] == 0.5
