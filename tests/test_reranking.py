import pandas as pd

from src.reranking import apply_mmr_rerank


def test_mmr_promotes_diverse_items_when_relevance_scores_are_close():
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "date": [20200827, 20200827, 20200827, 20200827],
            "video_id": [10, 11, 12, 13],
            "rank_mix_score": [1.00, 0.99, 0.98, 0.60],
            "first_category": [3, 3, 8, 9],
            "author_id_hash": [42, 42, 7, 9],
        }
    )

    reranked = apply_mmr_rerank(
        frame,
        score_column="rank_mix_score",
        output_column="mmr_score",
        lambda_relevance=0.7,
    )

    ordered = (
        reranked.sort_values(["user_id", "date", "mmr_score"], ascending=[True, True, False])
        ["video_id"]
        .tolist()
    )
    assert ordered[:3] == [10, 12, 11]


def test_mmr_keeps_original_order_when_lambda_is_one():
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "date": [20200827, 20200827, 20200827],
            "video_id": [10, 11, 12],
            "rank_mix_score": [0.40, 0.90, 0.80],
            "first_category": [3, 3, 8],
            "author_id_hash": [42, 42, 7],
        }
    )

    reranked = apply_mmr_rerank(
        frame,
        score_column="rank_mix_score",
        output_column="mmr_score",
        lambda_relevance=1.0,
    )

    ordered = (
        reranked.sort_values(["user_id", "date", "mmr_score"], ascending=[True, True, False])
        ["video_id"]
        .tolist()
    )
    assert ordered == [11, 12, 10]
