from src.common import (
    candidate_channels,
    load_profile,
    ranking_eval_ks,
    resolve_candidate_channel_limits,
    resolve_candidate_limits,
)


def test_resolve_candidate_limits_uses_profile_defaults_when_cli_values_are_missing():
    profile = {
        "top_per_channel": 300,
        "max_candidates_per_group": 500,
    }

    top_per_channel, max_candidates_per_group = resolve_candidate_limits(
        profile,
        top_per_channel=None,
        max_candidates_per_group=None,
    )

    assert top_per_channel == 300
    assert max_candidates_per_group == 500


def test_resolve_candidate_limits_keeps_explicit_cli_values():
    profile = {
        "top_per_channel": 300,
        "max_candidates_per_group": 500,
    }

    top_per_channel, max_candidates_per_group = resolve_candidate_limits(
        profile,
        top_per_channel=150,
        max_candidates_per_group=250,
    )

    assert top_per_channel == 150
    assert max_candidates_per_group == 250


def test_ranking_eval_ks_keeps_legacy_200_comparison_for_large_profiles():
    assert ranking_eval_ks({"recall_k": 500}) == [10, 200, 500]


def test_ranking_eval_ks_does_not_add_200_for_local_100_profiles():
    assert ranking_eval_ks({"recall_k": 100}) == [10, 100]


def test_candidate_channel_limits_default_to_equal_top_per_channel():
    profile = {}

    limits = resolve_candidate_channel_limits(profile, 300, ["itemcf", "content", "tower"])

    assert limits == {"itemcf": 300, "content": 300, "tower": 300}


def test_candidate_channel_limits_can_promote_tower_channel():
    profile = {"candidate_channel_weights": {"tower": 1.5, "itemcf": 1.0}}

    limits = resolve_candidate_channel_limits(profile, 350, ["itemcf", "content", "tower"])

    assert limits == {"itemcf": 350, "content": 350, "tower": 525}


def test_candidate_channels_use_profile_override_when_present():
    profile = {
        "candidate_channels": [
            "itemcf_main",
            "content_text_category",
            "feature_tower_dropout_hard_negative",
        ]
    }

    channels = candidate_channels(profile, ["itemcf_main", "content_text_category"])

    assert channels == [
        "itemcf_main",
        "content_text_category",
        "feature_tower_dropout_hard_negative",
    ]


def test_full_48gb_boosted_enhances_recall_ranking_and_fusion_capacity():
    profile = load_profile("full_48gb_boosted")

    assert profile["candidate_channels"][-1] == "feature_tower_dropout_hard_negative"
    assert profile["candidate_channel_weights"]["tower"] > profile["candidate_channel_weights"]["itemcf"]
    assert profile["two_tower_embedding_dim"] > load_profile("full_48gb_optimized")["two_tower_embedding_dim"]
    assert profile["lightgbm_estimators"] > load_profile("full_48gb_optimized")["lightgbm_estimators"]
    assert profile["deep_ranking_embedding_dim"] > load_profile("full_48gb_optimized")["deep_ranking_embedding_dim"]
    assert profile["rankmix_din_variant"] == "multibehavior"
