import pandas as pd
import pytest

from src.data_pipeline import FEATURE_END, TRAIN_END, VALID_END, attach_history_statistics
from scripts.run_experiment import run_recall


def test_split_boundaries_do_not_overlap():
    assert FEATURE_END < TRAIN_END < VALID_END


def test_statistics_feature_window_precedes_ranking_train_window():
    sample = pd.DataFrame({"date": [20200712, 20200801]})
    stats_rows = sample[sample["date"] <= FEATURE_END]
    ranking_rows = sample[(sample["date"] > FEATURE_END) & (sample["date"] <= TRAIN_END)]
    assert stats_rows["date"].max() < ranking_rows["date"].min()


def test_local_time_parsing_has_expected_hour_and_weekday():
    parsed = pd.to_datetime(pd.Series(["2020-07-05 13:32:39.746"]))
    assert int(parsed.dt.hour.iloc[0]) == 13
    assert int(parsed.dt.weekday.iloc[0]) == 6


def test_unix_timestamp_fallback_converts_to_shanghai_time():
    fallback = (
        pd.to_datetime(pd.Series([1593927159.746]), unit="s", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    assert int(fallback.dt.hour.iloc[0]) == 13


def test_history_statistics_use_only_prior_dates():
    history = pd.DataFrame(
        {
            "user_id": [1, 1],
            "video_id": [10, 11],
            "date": [20200101, 20200103],
            "watch_ratio": [1.0, 0.0],
            "label_complete": [1, 0],
            "label_strong": [0, 0],
            "label_short": [0, 1],
        }
    )
    target = pd.DataFrame({"user_id": [1], "video_id": [12], "date": [20200102]})
    result = attach_history_statistics(target, history)
    assert result.loc[0, "user_interactions"] == 1
    assert result.loc[0, "user_complete_rate"] == 1.0


def test_rolling_recall_is_restricted_to_test_panel():
    with pytest.raises(ValueError, match="only valid for a frozen test panel"):
        run_recall(
            "local_8gb_large",
            {"max_eval_users": 1},
            "itemcf",
            2026,
            "rolling_invalid",
            {"train_through": "valid"},
            "valid",
        )
