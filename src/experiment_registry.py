from __future__ import annotations


LEGACY_TO_PUBLIC_ID = {
    "R0.2": "decayed_popular_fallback",
    "R1.0": "itemcf_main",
    "R2.0": "category_content_exploration",
    "R2.4": "content_text_category",
    "R3.4": "feature_tower_id_dropout",
    "R3.6": "feature_tower_dropout_hard_negative",
    "PR3": "lambdarank_full_features",
    "PR3.refit": "lambdarank_full_features_refit",
    "PR3.no_recall_features": "lambdarank_without_recall_features",
    "PR3.no_cross_features": "lambdarank_without_cross_features",
    "PR3.no_temporal_features": "lambdarank_without_temporal_features",
    "PR3.no_cold_features": "lambdarank_without_cold_features",
    "PR.no_tower": "lambdarank_without_tower_candidates",
    "DR1.deepfm": "deepfm_candidate_ranker",
    "DR2.din": "din_sequence_ranker",
    "DR2.din.refit": "din_sequence_ranker_refit",
    "DR3.mmoe_complete": "mmoe_complete_ranker",
    "DR3.mmoe_complete_strong": "mmoe_complete_strong_ranker",
    "DR3.mmoe_multitask": "mmoe_multitask_ranker",
    "DR4.rank_mix": "rankmix_lambdarank_din",
    "DL1.din_bce": "din_pointwise_bce",
    "DL2.din_bpr": "din_pairwise_bpr",
    "DL7.din_listnet": "din_listnet",
    "DL8.din_bce_rolling": "din_pointwise_bce_rolling",
    "DS1.din_multibehavior": "din_multibehavior_sequence",
    "DS2.din_author_content_time": "din_author_content_time_sequence",
    "DS3.1.din_oov_content": "din_oov_content",
    "DS3.2.din_oov_content_dropout": "din_oov_content_dropout",
}

PUBLIC_TO_LEGACY_ID = {public: legacy for legacy, public in LEGACY_TO_PUBLIC_ID.items()}


def canonical_experiment_id(experiment_id: str) -> str:
    """Return the historical storage id for an experiment id or public alias."""
    return PUBLIC_TO_LEGACY_ID.get(experiment_id, experiment_id)


def public_experiment_id(experiment_id: str) -> str:
    """Return the readable project-facing id for an experiment id."""
    return LEGACY_TO_PUBLIC_ID.get(experiment_id, experiment_id)


def canonical_experiment_ids(experiment_ids: list[str]) -> list[str]:
    return [canonical_experiment_id(experiment_id) for experiment_id in experiment_ids]


def public_experiment_ids(experiment_ids: list[str]) -> list[str]:
    return [public_experiment_id(experiment_id) for experiment_id in experiment_ids]
