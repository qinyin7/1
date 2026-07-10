from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_relevance(scores: pd.Series) -> np.ndarray:
    values = scores.to_numpy(dtype=np.float64)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum <= minimum:
        return np.ones(len(values), dtype=np.float64)
    return (values - minimum) / (maximum - minimum)


def _similarity_to_item(
    group: pd.DataFrame,
    selected_index: int,
    category_weight: float,
    author_weight: float,
) -> np.ndarray:
    similarity = np.zeros(len(group), dtype=np.float64)
    total_weight = 0.0
    if "first_category" in group:
        total_weight += category_weight
        categories = group["first_category"].to_numpy()
        similarity += category_weight * (categories == categories[selected_index])
    if "author_id_hash" in group:
        total_weight += author_weight
        authors = group["author_id_hash"].to_numpy()
        similarity += author_weight * (authors == authors[selected_index])
    if total_weight == 0:
        return similarity
    return similarity / total_weight


def _rerank_group(
    group: pd.DataFrame,
    score_column: str,
    output_column: str,
    lambda_relevance: float,
    category_weight: float,
    author_weight: float,
) -> pd.DataFrame:
    group = group.copy()
    relevance = _normalize_relevance(group[score_column])
    available = np.ones(len(group), dtype=bool)
    max_similarity = np.zeros(len(group), dtype=np.float64)
    selected: list[int] = []

    while available.any():
        if not selected:
            mmr_scores = relevance.copy()
        else:
            mmr_scores = (
                lambda_relevance * relevance - (1.0 - lambda_relevance) * max_similarity
            )
        mmr_scores[~available] = -np.inf
        best_index = int(np.argmax(mmr_scores))
        selected.append(best_index)
        available[best_index] = False
        max_similarity = np.maximum(
            max_similarity,
            _similarity_to_item(group, best_index, category_weight, author_weight),
        )

    rerank_score = np.zeros(len(group), dtype=np.float64)
    rerank_rank = np.zeros(len(group), dtype=np.int32)
    for rank, selected_index in enumerate(selected, start=1):
        rerank_score[selected_index] = 1.0 / rank
        rerank_rank[selected_index] = rank

    group[output_column] = rerank_score
    group[f"{output_column}_rank"] = rerank_rank
    return group


def apply_mmr_rerank(
    frame: pd.DataFrame,
    score_column: str,
    output_column: str = "mmr_score",
    lambda_relevance: float = 0.9,
    category_weight: float = 0.6,
    author_weight: float = 0.4,
) -> pd.DataFrame:
    """Apply per user-day MMR reranking and return a copy with rerank scores.

    The output score is rank-shaped (`1 / mmr_rank`) so existing evaluators can
    sort by it without knowing about the greedy MMR selection order.
    """
    if not 0.0 <= lambda_relevance <= 1.0:
        raise ValueError("lambda_relevance must be between 0 and 1.")
    if score_column not in frame:
        raise KeyError(f"Missing score column: {score_column}")
    if frame.empty:
        result = frame.copy()
        result[output_column] = []
        result[f"{output_column}_rank"] = []
        return result
    pieces = []
    for _, group in frame.groupby(["user_id", "date"], sort=False):
        pieces.append(
            _rerank_group(
                group,
                score_column,
                output_column,
                lambda_relevance,
                category_weight,
                author_weight,
            )
        )
    return pd.concat(pieces, ignore_index=True)
