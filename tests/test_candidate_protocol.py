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
