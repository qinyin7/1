from scripts.run_experiment import _recall_model_cache_path


def test_recall_model_cache_key_depends_on_eligible_items():
    profile = {
        "two_tower_embedding_dim": 16,
        "two_tower_epochs": 1,
        "two_tower_batch_size": 128,
        "itemcf_history_length": 20,
    }
    first = _recall_model_cache_path(
        "local_8gb_large",
        profile,
        "two_tower",
        "R3.4",
        2026,
        "train",
        {"features": "profile_history_item_content", "id_dropout": 0.5},
        {1, 2, 3},
    )
    second = _recall_model_cache_path(
        "local_8gb_large",
        profile,
        "two_tower",
        "R3.4",
        2026,
        "train",
        {"features": "profile_history_item_content", "id_dropout": 0.5},
        {1, 2, 3, 4},
    )

    assert first != second
