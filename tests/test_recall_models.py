import pandas as pd
import torch

from src.recall_models import (
    FeatureTwoTowerRecall,
    ItemCFRecall,
    TextContentRecall,
    _sample_hard_negative_indices,
    _batched_topk_recommendations,
    quota_fuse_rankings,
)


def test_multi_feedback_itemcf_builds_weights_for_every_history_user():
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "video_id": [10, 11, 10, 12],
            "timestamp": [1, 2, 1, 2],
            "label_complete": [1, 1, 1, 1],
            "label_strong": [0, 1, 0, 0],
            "label_short": [0, 0, 0, 0],
        }
    )
    model = ItemCFRecall(feedback="multi_feedback")
    model.fit(train)
    assert set(model.user_item_weight) == {1, 2}
    assert model.user_item_weight[1][11] > model.user_item_weight[1][10]


def test_quota_fuse_rankings_respects_weights_and_removes_duplicates():
    fused = quota_fuse_rankings(
        [
            list(range(100, 200)),
            [100, *range(200, 300)],
            list(range(300, 400)),
        ],
        [0.7, 0.2, 0.1],
        100,
    )

    assert len(fused) == len(set(fused)) == 100
    assert sum(item < 200 for item in fused) == 70
    assert sum(200 <= item < 300 for item in fused) == 20
    assert sum(item >= 300 for item in fused) == 10


def test_text_content_recall_can_retrieve_unseen_similar_item():
    train = pd.DataFrame(
        {
            "user_id": [1],
            "video_id": [10],
            "timestamp": [1],
            "label_complete": [1],
        }
    )
    items = pd.DataFrame(
        {
            "video_id": [10, 11, 12],
            "first_category": [1, 1, 2],
            "content_text": ["猫咪 日常 可爱", "可爱 猫咪 玩耍", "汽车 维修 教程"],
        }
    )
    model = TextContentRecall()
    model.fit(train, items, {10, 11, 12})

    assert model.recommend([1], {1: {10}}, 2)[1][0] == 11


def test_recall_can_be_restricted_to_fully_observed_catalog():
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "video_id": [10, 11, 12],
            "timestamp": [1, 2, 1],
            "label_complete": [1, 1, 1],
            "label_strong": [0, 0, 0],
            "label_short": [0, 0, 0],
        }
    )
    model = ItemCFRecall()
    model.fit(train)
    recommendations = model.recommend([1], {1: set()}, 3, allowed_items={12})
    assert recommendations == {1: [12]}


def test_batched_topk_recommendations_masks_seen_and_allowed_items():
    user_vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    item_vectors = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.0],
            [0.0, 1.0],
            [0.0, 0.8],
        ]
    )
    recommendations = _batched_topk_recommendations(
        users=[1, 2, 3],
        known_users=[1, 2],
        user_vectors=user_vectors,
        item_vectors=item_vectors,
        index_to_item=[10, 11, 12, 13],
        item_to_index={10: 0, 11: 1, 12: 2, 13: 3},
        seen={1: {10}, 2: {12}},
        k=2,
        device=torch.device("cpu"),
        allowed_items={10, 11, 12},
        batch_size=2,
    )

    assert 10 not in recommendations[1]
    assert 12 not in recommendations[2]
    assert 13 not in recommendations[1] + recommendations[2]
    assert recommendations[3] == []


def test_feature_two_tower_builds_cold_item_content_features():
    model = FeatureTwoTowerRecall(
        dimension=8,
        epochs=1,
        batch_size=2,
        seed=2026,
        id_dropout=0.5,
        hard_negative_ratio=0.5,
    )
    items = pd.DataFrame(
        {
            "video_id": [10, 11, 12],
            "first_category": [1, 1, 2],
            "author_id": [100, 101, 102],
            "video_duration": [1000, 2000, 3000],
            "content_text": ["猫咪 日常", "猫咪 玩耍", "汽车 教程"],
        }
    )

    features = model._build_item_features(items)

    assert features.shape[0] == 3
    assert features.shape[1] > 3
    assert model.id_dropout == 0.5
    assert model.hard_negative_ratio == 0.5


def test_sample_hard_negative_indices_avoids_sampling_the_positive_item():
    positive_index = pd.Series([0, 1, 2, 3]).to_numpy()
    categories = pd.Series([10, 10, 20, 30]).to_numpy()
    known_indices = pd.Series([0, 1, 2, 3]).to_numpy()
    known_by_category = {
        10: pd.Series([0, 1]).to_numpy(),
        20: pd.Series([2]).to_numpy(),
        30: pd.Series([3]).to_numpy(),
    }

    sampled = _sample_hard_negative_indices(
        positive_index=positive_index,
        categories=categories,
        known_indices=known_indices,
        known_by_category=known_by_category,
        seed=2026,
    )

    assert sampled.shape == positive_index.shape
    assert sampled[0] == 1
    assert sampled[1] == 0
    assert sampled[2] in known_indices and sampled[2] != 2
    assert sampled[3] in known_indices and sampled[3] != 3
