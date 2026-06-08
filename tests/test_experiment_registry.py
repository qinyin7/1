from src.candidate_features import PUBLIC_EXPERIMENT_FEATURES
from src.experiment_registry import canonical_experiment_id, public_experiment_id


def test_public_experiment_ids_resolve_to_legacy_storage_ids():
    assert canonical_experiment_id("itemcf_main") == "R1.0"
    assert canonical_experiment_id("feature_tower_id_dropout") == "R3.4"
    assert canonical_experiment_id("rankmix_lambdarank_din") == "DR4.rank_mix"


def test_legacy_experiment_ids_have_readable_public_names():
    assert public_experiment_id("PR3") == "lambdarank_full_features"
    assert public_experiment_id("DR2.din") == "din_sequence_ranker"


def test_candidate_ranking_public_defaults_are_readable():
    assert "lambdarank_full_features" in PUBLIC_EXPERIMENT_FEATURES
    assert "PR3" not in PUBLIC_EXPERIMENT_FEATURES
