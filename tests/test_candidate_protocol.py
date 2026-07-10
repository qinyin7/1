import pandas as pd

import scripts.run_candidate_ranking as candidate_ranking


def test_candidate_training_does_not_label_unexposed_items_as_negatives(monkeypatch):
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "video_id": [20, 10, 11],
            "date": [20200801, 20200827, 20200827],
            "split": ["train", "valid", "valid"],
            "watch_ratio": [1.0, 1.0, 0.1],
            "label_complete": [1, 1, 0],
            "label_strong": [0, 0, 0],
            "label_short": [0, 0, 1],
        }
    )
    items = pd.DataFrame(
        {
            "video_id": [10, 11, 12, 20],
            "first_seen_date": [20200801] * 4,
            "first_category": [1] * 4,
            "author_id": [100] * 4,
            "video_duration": [1000] * 4,
        }
    )
    monkeypatch.setattr(
        candidate_ranking,
        "load_recommendations",
        lambda *args: {1: [10, 11, 12]},
    )

    frame, _, _, _ = candidate_ranking.build_candidate_frame(
        interactions,
        items,
        "valid",
        ["train"],
        ["R1.0", "R2.4", "R3.6"],
        "local_8gb_large",
        10,
        3,
        10,
        2026,
        True,
    )

    assert set(frame["video_id"]) == {10, 11}
    assert frame.set_index("video_id").loc[11, "label"] == 0


def test_candidate_frame_builds_lightweight_cross_features(monkeypatch):
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "video_id": [20, 21, 10, 11],
            "date": [20200801, 20200801, 20200827, 20200827],
            "split": ["train", "train", "valid", "valid"],
            "watch_ratio": [1.0, 0.2, 1.0, 0.1],
            "label_complete": [1, 0, 1, 0],
            "label_strong": [0, 0, 0, 0],
            "label_short": [0, 1, 0, 1],
        }
    )
    items = pd.DataFrame(
        {
            "video_id": [10, 11, 20, 21],
            "first_seen_date": [20200801] * 4,
            "first_category": [7, 8, 7, 8],
            "author_id": [100, 101, 100, 101],
            "video_duration": [1000] * 4,
        }
    )
    monkeypatch.setattr(
        candidate_ranking,
        "load_recommendations",
        lambda *args: {1: [10, 11]},
    )

    frame, _, _, _ = candidate_ranking.build_candidate_frame(
        interactions,
        items,
        "valid",
        ["train"],
        ["R1.0", "R2.4", "R3.6"],
        "local_8gb_large",
        10,
        2,
        10,
        2026,
        True,
    )
    row = frame.set_index("video_id").loc[10]

    assert row["category_item_complete_cross"] == row["category_affinity"] * row["item_complete_rate"]
    assert row["author_item_complete_cross"] == row["author_affinity"] * row["item_complete_rate"]
    assert row["channel_item_complete_cross"] == row["channel_count"] * row["item_complete_rate"]
    assert row["cold_content_cross"] == row["is_cold_item"] * row["content_present"]
    assert row["tower_age_cross"] == row["tower_present"] * row["item_age_days"]


def test_candidate_frame_can_use_larger_tower_channel_limit(monkeypatch):
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1],
            "video_id": [99, 10, 20, 30, 31],
            "date": [20200801, 20200827, 20200827, 20200827, 20200827],
            "split": ["train", "valid", "valid", "valid", "valid"],
            "watch_ratio": [1.0, 1.0, 1.0, 1.0, 1.0],
            "label_complete": [1, 1, 1, 1, 1],
            "label_strong": [0, 0, 0, 0, 0],
            "label_short": [0, 0, 0, 0, 0],
        }
    )
    items = pd.DataFrame(
        {
            "video_id": [10, 20, 30, 31, 99],
            "first_seen_date": [20200801] * 5,
            "first_category": [1] * 5,
            "author_id": [100] * 5,
            "video_duration": [1000] * 5,
        }
    )
    recommendations = {
        "itemcf_main": {1: [10]},
        "content_text_category": {1: [20]},
        "feature_tower_dropout_hard_negative": {1: [30, 31]},
    }
    monkeypatch.setattr(
        candidate_ranking,
        "load_recommendations",
        lambda _profile, experiment, _panel: recommendations[experiment],
    )

    frame, _, _, _ = candidate_ranking.build_candidate_frame(
        interactions,
        items,
        "valid",
        ["train"],
        [
            "itemcf_main",
            "content_text_category",
            "feature_tower_dropout_hard_negative",
        ],
        "local_8gb_large",
        10,
        1,
        10,
        2026,
        True,
        channel_top_limits={"itemcf": 1, "content": 1, "tower": 2},
    )

    assert set(frame["video_id"]) == {10, 20, 30, 31}
    assert frame.set_index("video_id").loc[31, "tower_present"] == 1.0
