from __future__ import annotations


CHANNELS = ["itemcf", "content", "tower"]
LABEL_COLUMNS = ["label_complete", "label_strong", "label_short"]
RECALL_FEATURES = [
    "itemcf_present",
    "content_present",
    "tower_present",
    "itemcf_rank_score",
    "content_rank_score",
    "tower_rank_score",
    "channel_count",
]
CROSS_FEATURES = ["category_affinity", "author_affinity"]
HIGH_ORDER_CROSS_FEATURES = [
    "category_item_complete_cross",
    "author_item_complete_cross",
    "channel_item_complete_cross",
    "cold_content_cross",
    "tower_age_cross",
]
TEMPORAL_FEATURES = ["item_age_days"]
COLD_FEATURES = ["is_cold_item"]
ITEM_IDENTITY_FEATURES = ["first_category", "video_duration", "author_id_hash"]
USER_FEATURES = [
    "user_interactions",
    "user_complete_rate",
    "user_strong_rate",
    "user_short_rate",
    "user_mean_watch_ratio",
]
ITEM_STAT_FEATURES = [
    "item_interactions",
    "item_complete_rate",
    "item_strong_rate",
    "item_short_rate",
    "item_mean_watch_ratio",
]
FEATURE_COLUMNS = [
    *RECALL_FEATURES,
    *ITEM_IDENTITY_FEATURES,
    *COLD_FEATURES,
    *TEMPORAL_FEATURES,
    *USER_FEATURES,
    *ITEM_STAT_FEATURES,
    *CROSS_FEATURES,
    *HIGH_ORDER_CROSS_FEATURES,
]
FEATURE_GROUPS = {
    "recall_side": RECALL_FEATURES,
    "user_side": USER_FEATURES,
    "item_side": [*ITEM_IDENTITY_FEATURES, *ITEM_STAT_FEATURES, *COLD_FEATURES, *TEMPORAL_FEATURES],
    "cross_side": [*CROSS_FEATURES, *HIGH_ORDER_CROSS_FEATURES],
}
PUBLIC_EXPERIMENT_FEATURES = {
    "lambdarank_basic_user_features": [
        "first_category",
        "video_duration",
        "author_id_hash",
        "user_interactions",
        "user_complete_rate",
        "user_strong_rate",
        "user_short_rate",
        "user_mean_watch_ratio",
    ],
    "lambdarank_recall_features_only": RECALL_FEATURES,
    "lambdarank_full_features": FEATURE_COLUMNS,
    "lambdarank_without_recall_features": [
        column for column in FEATURE_COLUMNS if column not in RECALL_FEATURES
    ],
    "lambdarank_without_cross_features": [
        column
        for column in FEATURE_COLUMNS
        if column not in {*CROSS_FEATURES, *HIGH_ORDER_CROSS_FEATURES}
    ],
    "lambdarank_without_temporal_features": [
        column for column in FEATURE_COLUMNS if column not in TEMPORAL_FEATURES
    ],
    "lambdarank_without_cold_features": [
        column for column in FEATURE_COLUMNS if column not in COLD_FEATURES
    ],
    "lambdarank_without_tower_candidates": [
        column for column in FEATURE_COLUMNS if not column.startswith("tower_")
    ],
}

LEGACY_EXPERIMENT_FEATURES = {
    "PR1": PUBLIC_EXPERIMENT_FEATURES["lambdarank_basic_user_features"],
    "PR2": PUBLIC_EXPERIMENT_FEATURES["lambdarank_recall_features_only"],
    "PR3": PUBLIC_EXPERIMENT_FEATURES["lambdarank_full_features"],
    "PR3.no_recall_features": PUBLIC_EXPERIMENT_FEATURES[
        "lambdarank_without_recall_features"
    ],
    "PR3.no_cross_features": PUBLIC_EXPERIMENT_FEATURES[
        "lambdarank_without_cross_features"
    ],
    "PR3.no_temporal_features": PUBLIC_EXPERIMENT_FEATURES[
        "lambdarank_without_temporal_features"
    ],
    "PR3.no_cold_features": PUBLIC_EXPERIMENT_FEATURES[
        "lambdarank_without_cold_features"
    ],
    "PR.no_tower": PUBLIC_EXPERIMENT_FEATURES["lambdarank_without_tower_candidates"],
}

EXPERIMENT_FEATURES = {**PUBLIC_EXPERIMENT_FEATURES, **LEGACY_EXPERIMENT_FEATURES}
