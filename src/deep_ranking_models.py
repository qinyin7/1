from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.candidate_features import FEATURE_COLUMNS


SPARSE_COLUMNS = [
    "user_id",
    "video_id",
    "first_category",
    "author_id_hash",
    "source_mask",
]
DENSE_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column not in {"first_category", "author_id_hash"}
]
TASK_COLUMNS = ["label_complete", "label_strong", "label_short"]


def initialize_embeddings(embeddings: nn.ModuleList, std: float = 0.01) -> None:
    for embedding in embeddings:
        nn.init.normal_(embedding.weight, mean=0.0, std=std)
        if embedding.padding_idx is not None:
            with torch.no_grad():
                embedding.weight[embedding.padding_idx].zero_()


def source_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["itemcf_present"].to_numpy(dtype=np.int64)
        + 2 * frame["content_present"].to_numpy(dtype=np.int64)
        + 4 * frame["tower_present"].to_numpy(dtype=np.int64)
    )


@dataclass
class CandidateFeatureEncoder:
    vocabularies: dict[str, np.ndarray]
    dense_mean: np.ndarray
    dense_std: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "CandidateFeatureEncoder":
        vocabularies = {}
        for column in SPARSE_COLUMNS:
            values = source_mask(frame) if column == "source_mask" else frame[column].to_numpy()
            vocabularies[column] = np.sort(pd.unique(values))
        dense = frame[DENSE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        dense_mean = np.nanmean(dense, axis=0).astype(np.float32)
        dense_std = np.nanstd(dense, axis=0).astype(np.float32)
        dense_std[dense_std < 1e-6] = 1.0
        return cls(vocabularies, dense_mean, dense_std)

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocabularies[column]) + 1 for column in SPARSE_COLUMNS]

    def encode_values(self, column: str, values: np.ndarray) -> np.ndarray:
        codes = pd.Index(self.vocabularies[column]).get_indexer(values)
        return (codes + 1).astype(np.int64)

    def encode(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        sparse = np.empty((len(frame), len(SPARSE_COLUMNS)), dtype=np.int64)
        for index, column in enumerate(SPARSE_COLUMNS):
            values = source_mask(frame) if column == "source_mask" else frame[column].to_numpy()
            sparse[:, index] = self.encode_values(column, values)
        dense = frame[DENSE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        dense -= self.dense_mean
        dense /= self.dense_std
        np.nan_to_num(dense, copy=False)
        return sparse, dense

    def state_dict(self) -> dict:
        return {
            "vocabularies": {
                column: values.tolist() for column, values in self.vocabularies.items()
            },
            "dense_mean": self.dense_mean.tolist(),
            "dense_std": self.dense_std.tolist(),
            "sparse_columns": SPARSE_COLUMNS,
            "dense_columns": DENSE_COLUMNS,
        }


class DeepFM(nn.Module):
    requires_history = False

    def __init__(self, cardinalities: list[int], dense_dim: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, embedding_dim) for cardinality in cardinalities
        )
        self.first_order = nn.ModuleList(
            nn.Embedding(cardinality, 1) for cardinality in cardinalities
        )
        initialize_embeddings(self.embeddings)
        for embedding in self.first_order:
            nn.init.zeros_(embedding.weight)
        self.dense_linear = nn.Linear(dense_dim, 1)
        input_dim = len(cardinalities) * embedding_dim + dense_dim
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        embeddings = torch.stack(
            [embedding(sparse[:, index]) for index, embedding in enumerate(self.embeddings)],
            dim=1,
        )
        linear = sum(
            embedding(sparse[:, index])
            for index, embedding in enumerate(self.first_order)
        ) + self.dense_linear(dense)
        summed = embeddings.sum(dim=1)
        fm = 0.5 * (summed.square() - embeddings.square().sum(dim=1)).sum(
            dim=1, keepdim=True
        )
        deep_input = torch.cat([embeddings.flatten(1), dense], dim=1)
        return linear + fm + self.deep(deep_input)


class DIN(nn.Module):
    requires_history = True

    def __init__(self, cardinalities: list[int], dense_dim: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, embedding_dim, padding_idx=0)
            for cardinality in cardinalities
        )
        initialize_embeddings(self.embeddings)
        attention_dim = embedding_dim * 2
        self.attention = nn.Sequential(
            nn.Linear(attention_dim * 4, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )
        input_dim = len(cardinalities) * embedding_dim + attention_dim + dense_dim
        self.output = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        sparse: torch.Tensor,
        dense: torch.Tensor,
        history_items: torch.Tensor,
        history_categories: torch.Tensor,
    ) -> torch.Tensor:
        candidate_embeddings = [
            embedding(sparse[:, index]) for index, embedding in enumerate(self.embeddings)
        ]
        query = torch.cat([candidate_embeddings[1], candidate_embeddings[2]], dim=1)
        keys = torch.cat(
            [
                self.embeddings[1](history_items),
                self.embeddings[2](history_categories),
            ],
            dim=2,
        )
        expanded_query = query.unsqueeze(1).expand_as(keys)
        attention_input = torch.cat(
            [expanded_query, keys, expanded_query - keys, expanded_query * keys],
            dim=2,
        )
        attention_logits = self.attention(attention_input).squeeze(-1)
        mask = (history_items != 0) | (history_categories != 0)
        attention_logits = attention_logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(attention_logits, dim=1)
        weights = weights * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        interest = torch.sum(keys * weights.unsqueeze(-1), dim=1)
        model_input = torch.cat([*candidate_embeddings, interest, dense], dim=1)
        return self.output(model_input)


class MultiBehaviorDIN(nn.Module):
    requires_history = True

    def __init__(self, cardinalities: list[int], dense_dim: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, embedding_dim, padding_idx=0)
            for cardinality in cardinalities
        )
        initialize_embeddings(self.embeddings)
        self.behavior_embedding = nn.Embedding(8, embedding_dim, padding_idx=0)
        initialize_embeddings(nn.ModuleList([self.behavior_embedding]))
        self.watch_projection = nn.Linear(1, embedding_dim, bias=False)
        query_dim = embedding_dim * 2
        key_dim = embedding_dim * 4
        self.query_projection = nn.Linear(query_dim, key_dim)
        self.attention = nn.Sequential(
            nn.Linear(key_dim * 4, 96),
            nn.PReLU(),
            nn.Linear(96, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )
        input_dim = len(cardinalities) * embedding_dim + key_dim + dense_dim
        self.output = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        sparse: torch.Tensor,
        dense: torch.Tensor,
        history_items: torch.Tensor,
        history_categories: torch.Tensor,
        history_behaviors: torch.Tensor,
        history_watch_ratios: torch.Tensor,
    ) -> torch.Tensor:
        candidate_embeddings = [
            embedding(sparse[:, index]) for index, embedding in enumerate(self.embeddings)
        ]
        query = self.query_projection(
            torch.cat([candidate_embeddings[1], candidate_embeddings[2]], dim=1)
        )
        keys = torch.cat(
            [
                self.embeddings[1](history_items),
                self.embeddings[2](history_categories),
                self.behavior_embedding(history_behaviors),
                self.watch_projection(history_watch_ratios.unsqueeze(-1)),
            ],
            dim=2,
        )
        expanded_query = query.unsqueeze(1).expand_as(keys)
        attention_input = torch.cat(
            [expanded_query, keys, expanded_query - keys, expanded_query * keys],
            dim=2,
        )
        attention_logits = self.attention(attention_input).squeeze(-1)
        mask = history_behaviors.ne(0)
        attention_logits = attention_logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(attention_logits, dim=1) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        interest = torch.sum(keys * weights.unsqueeze(-1), dim=1)
        model_input = torch.cat([*candidate_embeddings, interest, dense], dim=1)
        return self.output(model_input)


class EnrichedDIN(nn.Module):
    requires_history = True

    def __init__(
        self,
        cardinalities: list[int],
        dense_dim: int,
        embedding_dim: int,
        candidate_content_lookup: np.ndarray,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, embedding_dim, padding_idx=0)
            for cardinality in cardinalities
        )
        initialize_embeddings(self.embeddings)
        self.behavior_embedding = nn.Embedding(8, embedding_dim, padding_idx=0)
        initialize_embeddings(nn.ModuleList([self.behavior_embedding]))
        content_dim = candidate_content_lookup.shape[1]
        self.register_buffer(
            "candidate_content_lookup",
            torch.from_numpy(candidate_content_lookup.astype(np.float32)),
        )
        self.content_projection = nn.Linear(content_dim, embedding_dim, bias=False)
        self.watch_projection = nn.Linear(1, embedding_dim, bias=False)
        self.time_gap_projection = nn.Linear(1, embedding_dim, bias=False)
        query_dim = embedding_dim * 4
        key_dim = embedding_dim * 7
        self.query_projection = nn.Linear(query_dim, key_dim)
        self.attention = nn.Sequential(
            nn.Linear(key_dim * 4, 128),
            nn.PReLU(),
            nn.Linear(128, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )
        input_dim = len(cardinalities) * embedding_dim + key_dim + dense_dim
        self.output = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        sparse: torch.Tensor,
        dense: torch.Tensor,
        history_items: torch.Tensor,
        history_categories: torch.Tensor,
        history_behaviors: torch.Tensor,
        history_watch_ratios: torch.Tensor,
        history_authors: torch.Tensor,
        history_content: torch.Tensor,
        history_time_gaps: torch.Tensor,
    ) -> torch.Tensor:
        candidate_embeddings = [
            embedding(sparse[:, index]) for index, embedding in enumerate(self.embeddings)
        ]
        candidate_content = self.content_projection(
            self.candidate_content_lookup[sparse[:, 1]]
        )
        query = self.query_projection(
            torch.cat(
                [
                    candidate_embeddings[1],
                    candidate_embeddings[2],
                    candidate_embeddings[3],
                    candidate_content,
                ],
                dim=1,
            )
        )
        keys = torch.cat(
            [
                self.embeddings[1](history_items),
                self.embeddings[2](history_categories),
                self.behavior_embedding(history_behaviors),
                self.watch_projection(history_watch_ratios.unsqueeze(-1)),
                self.embeddings[3](history_authors),
                self.content_projection(history_content),
                self.time_gap_projection(history_time_gaps.unsqueeze(-1)),
            ],
            dim=2,
        )
        expanded_query = query.unsqueeze(1).expand_as(keys)
        attention_input = torch.cat(
            [expanded_query, keys, expanded_query - keys, expanded_query * keys],
            dim=2,
        )
        attention_logits = self.attention(attention_input).squeeze(-1)
        mask = history_behaviors.ne(0)
        attention_logits = attention_logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(attention_logits, dim=1) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        interest = torch.sum(keys * weights.unsqueeze(-1), dim=1)
        model_input = torch.cat([*candidate_embeddings, interest, dense], dim=1)
        return self.output(model_input)


class OOVAwareEnrichedDIN(nn.Module):
    requires_history = True

    def __init__(
        self,
        cardinalities: list[int],
        dense_dim: int,
        embedding_dim: int,
        content_dim: int,
        id_dropout: float = 0.0,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.id_dropout = id_dropout
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, embedding_dim, padding_idx=0)
            for cardinality in cardinalities
        )
        initialize_embeddings(self.embeddings)
        self.behavior_embedding = nn.Embedding(8, embedding_dim, padding_idx=0)
        initialize_embeddings(nn.ModuleList([self.behavior_embedding]))
        self.content_projection = nn.Linear(content_dim, embedding_dim, bias=False)
        self.watch_projection = nn.Linear(1, embedding_dim, bias=False)
        self.time_gap_projection = nn.Linear(1, embedding_dim, bias=False)
        query_dim = embedding_dim * 4
        key_dim = embedding_dim * 7
        self.query_projection = nn.Linear(query_dim, key_dim)
        self.attention = nn.Sequential(
            nn.Linear(key_dim * 4, 128),
            nn.PReLU(),
            nn.Linear(128, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )
        input_dim = len(cardinalities) * embedding_dim + key_dim + dense_dim
        self.output = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        sparse: torch.Tensor,
        dense: torch.Tensor,
        history_items: torch.Tensor,
        history_categories: torch.Tensor,
        history_behaviors: torch.Tensor,
        history_watch_ratios: torch.Tensor,
        history_authors: torch.Tensor,
        history_content: torch.Tensor,
        history_time_gaps: torch.Tensor,
    ) -> torch.Tensor:
        candidate_embeddings = [
            embedding(sparse[:, index]) for index, embedding in enumerate(self.embeddings)
        ]
        if self.training and self.id_dropout > 0:
            keep = (
                torch.rand(
                    (len(sparse), 1),
                    device=candidate_embeddings[1].device,
                )
                >= self.id_dropout
            )
            candidate_embeddings[1] = candidate_embeddings[1] * keep
        candidate_content = self.content_projection(dense[:, -self.content_dim :])
        query = self.query_projection(
            torch.cat(
                [
                    candidate_embeddings[1],
                    candidate_embeddings[2],
                    candidate_embeddings[3],
                    candidate_content,
                ],
                dim=1,
            )
        )
        keys = torch.cat(
            [
                self.embeddings[1](history_items),
                self.embeddings[2](history_categories),
                self.behavior_embedding(history_behaviors),
                self.watch_projection(history_watch_ratios.unsqueeze(-1)),
                self.embeddings[3](history_authors),
                self.content_projection(history_content),
                self.time_gap_projection(history_time_gaps.unsqueeze(-1)),
            ],
            dim=2,
        )
        expanded_query = query.unsqueeze(1).expand_as(keys)
        attention_input = torch.cat(
            [expanded_query, keys, expanded_query - keys, expanded_query * keys],
            dim=2,
        )
        attention_logits = self.attention(attention_input).squeeze(-1)
        mask = history_behaviors.ne(0)
        attention_logits = attention_logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(attention_logits, dim=1) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        interest = torch.sum(keys * weights.unsqueeze(-1), dim=1)
        model_input = torch.cat([*candidate_embeddings, interest, dense], dim=1)
        return self.output(model_input)


class MMoE(nn.Module):
    requires_history = False

    def __init__(
        self,
        cardinalities: list[int],
        dense_dim: int,
        embedding_dim: int,
        num_experts: int = 4,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, embedding_dim) for cardinality in cardinalities
        )
        initialize_embeddings(self.embeddings)
        input_dim = len(cardinalities) * embedding_dim + dense_dim
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            for _ in range(num_experts)
        )
        self.gates = nn.ModuleList(
            nn.Linear(input_dim, num_experts) for _ in TASK_COLUMNS
        )
        self.towers = nn.ModuleList(
            nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            for _ in TASK_COLUMNS
        )

    def forward(self, sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        embeddings = [
            embedding(sparse[:, index]) for index, embedding in enumerate(self.embeddings)
        ]
        model_input = torch.cat([*embeddings, dense], dim=1)
        experts = torch.stack([expert(model_input) for expert in self.experts], dim=1)
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(model_input), dim=1).unsqueeze(-1)
            outputs.append(tower(torch.sum(experts * weights, dim=1)))
        return torch.cat(outputs, dim=1)


def build_history_tables(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    splits: list[str],
    encoder: CandidateFeatureEncoder,
    history_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    positive = interactions[
        interactions["split"].isin(splits) & interactions["label_complete"].eq(1)
    ][["user_id", "video_id", "date", "timestamp"]]
    positive = positive.sort_values(["user_id", "date", "timestamp"]).groupby(
        "user_id", sort=False
    ).tail(history_length)
    category_by_item = items.set_index("video_id")["first_category"].to_dict()
    user_cardinality = encoder.cardinalities[0]
    history_items = np.zeros((user_cardinality, history_length), dtype=np.int64)
    history_categories = np.zeros((user_cardinality, history_length), dtype=np.int64)
    for user_id, group in positive.groupby("user_id", sort=False):
        user_code = encoder.encode_values("user_id", np.array([user_id]))[0]
        if user_code == 0:
            continue
        raw_items = group["video_id"].to_numpy()
        raw_categories = np.array(
            [category_by_item.get(item, -1) for item in raw_items]
        )
        item_codes = encoder.encode_values("video_id", raw_items)
        category_codes = encoder.encode_values("first_category", raw_categories)
        length = min(history_length, len(item_codes))
        history_items[user_code, -length:] = item_codes[-length:]
        history_categories[user_code, -length:] = category_codes[-length:]
    return history_items, history_categories


def build_multibehavior_history_tables(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    target_dates: list[int],
    encoder: CandidateFeatureEncoder,
    history_length: int,
) -> tuple[tuple[np.ndarray, ...], dict[int, int]]:
    user_cardinality = encoder.cardinalities[0]
    table_shape = (len(target_dates) * user_cardinality, history_length)
    history_items = np.zeros(table_shape, dtype=np.int64)
    history_categories = np.zeros(table_shape, dtype=np.int64)
    history_behaviors = np.zeros(table_shape, dtype=np.int64)
    history_watch_ratios = np.zeros(table_shape, dtype=np.float32)
    offsets = {
        int(date): index * user_cardinality for index, date in enumerate(target_dates)
    }
    category_by_item = items.set_index("video_id")["first_category"].to_dict()
    historical_columns = [
        "user_id",
        "video_id",
        "date",
        "timestamp",
        "label_complete",
        "label_strong",
        "label_short",
        "watch_ratio",
    ]
    history = interactions[historical_columns].sort_values(
        ["user_id", "date", "timestamp"]
    )
    for target_date in target_dates:
        historical = history[history["date"].lt(target_date)].groupby(
            "user_id", sort=False
        ).tail(history_length)
        offset = offsets[int(target_date)]
        for user_id, group in historical.groupby("user_id", sort=False):
            user_code = encoder.encode_values("user_id", np.array([user_id]))[0]
            if user_code == 0:
                continue
            raw_items = group["video_id"].to_numpy()
            raw_categories = np.array(
                [category_by_item.get(item, -1) for item in raw_items]
            )
            item_codes = encoder.encode_values("video_id", raw_items)
            category_codes = encoder.encode_values("first_category", raw_categories)
            behavior_codes = (
                1
                + group["label_complete"].to_numpy(dtype=np.int64)
                + 2 * group["label_strong"].to_numpy(dtype=np.int64)
                + 4 * group["label_short"].to_numpy(dtype=np.int64)
            )
            watch_ratios = (
                group["watch_ratio"].clip(lower=0, upper=5).to_numpy(dtype=np.float32)
                / 5.0
            )
            length = min(history_length, len(item_codes))
            table_index = offset + user_code
            history_items[table_index, -length:] = item_codes[-length:]
            history_categories[table_index, -length:] = category_codes[-length:]
            history_behaviors[table_index, -length:] = behavior_codes[-length:]
            history_watch_ratios[table_index, -length:] = watch_ratios[-length:]
    return (
        history_items,
        history_categories,
        history_behaviors,
        history_watch_ratios,
    ), offsets


def build_content_features(
    items: pd.DataFrame,
    dimension: int = 16,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=2,
        max_features=4096,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(items["content_text"].fillna(""))
    dimension = min(dimension, max(1, min(tfidf.shape) - 1))
    vectors = TruncatedSVD(n_components=dimension, random_state=seed).fit_transform(tfidf)
    vectors = normalize(vectors).astype(np.float32)
    return items["video_id"].to_numpy(dtype=np.int64), vectors


def build_candidate_content_lookup(
    encoder: CandidateFeatureEncoder,
    content_item_ids: np.ndarray,
    content_vectors: np.ndarray,
) -> np.ndarray:
    lookup = np.zeros(
        (encoder.cardinalities[1], content_vectors.shape[1]), dtype=np.float32
    )
    content_index = pd.Index(content_item_ids)
    positions = content_index.get_indexer(encoder.vocabularies["video_id"])
    known = positions >= 0
    lookup[np.flatnonzero(known) + 1] = content_vectors[positions[known]]
    return lookup


def encode_candidate_content(
    frame: pd.DataFrame,
    content_item_ids: np.ndarray,
    content_vectors: np.ndarray,
) -> np.ndarray:
    positions = pd.Index(content_item_ids).get_indexer(frame["video_id"].to_numpy())
    encoded = np.zeros((len(frame), content_vectors.shape[1]), dtype=np.float32)
    known = positions >= 0
    encoded[known] = content_vectors[positions[known]]
    return encoded


def build_enriched_history_tables(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    target_dates: list[int],
    encoder: CandidateFeatureEncoder,
    history_length: int,
    content_item_ids: np.ndarray,
    content_vectors: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], dict[int, int]]:
    user_cardinality = encoder.cardinalities[0]
    table_shape = (len(target_dates) * user_cardinality, history_length)
    history_items = np.zeros(table_shape, dtype=np.int64)
    history_categories = np.zeros(table_shape, dtype=np.int64)
    history_behaviors = np.zeros(table_shape, dtype=np.int64)
    history_watch_ratios = np.zeros(table_shape, dtype=np.float32)
    history_authors = np.zeros(table_shape, dtype=np.int64)
    history_content = np.zeros((*table_shape, content_vectors.shape[1]), dtype=np.float32)
    history_time_gaps = np.zeros(table_shape, dtype=np.float32)
    offsets = {
        int(date): index * user_cardinality for index, date in enumerate(target_dates)
    }
    item_lookup = items.set_index("video_id")
    category_by_item = item_lookup["first_category"].to_dict()
    author_by_item = item_lookup["author_id"].to_dict()
    content_index = pd.Index(content_item_ids)
    historical_columns = [
        "user_id",
        "video_id",
        "date",
        "timestamp",
        "label_complete",
        "label_strong",
        "label_short",
        "watch_ratio",
    ]
    history = interactions[historical_columns].sort_values(
        ["user_id", "date", "timestamp"]
    )
    for target_date in target_dates:
        historical = history[history["date"].lt(target_date)].groupby(
            "user_id", sort=False
        ).tail(history_length)
        target_timestamp = pd.to_datetime(str(target_date), format="%Y%m%d")
        offset = offsets[int(target_date)]
        for user_id, group in historical.groupby("user_id", sort=False):
            user_code = encoder.encode_values("user_id", np.array([user_id]))[0]
            if user_code == 0:
                continue
            raw_items = group["video_id"].to_numpy()
            raw_categories = np.array(
                [category_by_item.get(item, -1) for item in raw_items]
            )
            raw_authors = np.array(
                [int(author_by_item.get(item, -1)) % 256 for item in raw_items]
            )
            item_codes = encoder.encode_values("video_id", raw_items)
            category_codes = encoder.encode_values("first_category", raw_categories)
            author_codes = encoder.encode_values("author_id_hash", raw_authors)
            behavior_codes = (
                1
                + group["label_complete"].to_numpy(dtype=np.int64)
                + 2 * group["label_strong"].to_numpy(dtype=np.int64)
                + 4 * group["label_short"].to_numpy(dtype=np.int64)
            )
            watch_ratios = (
                group["watch_ratio"].clip(lower=0, upper=5).to_numpy(dtype=np.float32)
                / 5.0
            )
            positions = content_index.get_indexer(raw_items)
            item_content = np.zeros((len(raw_items), content_vectors.shape[1]), dtype=np.float32)
            known_content = positions >= 0
            item_content[known_content] = content_vectors[positions[known_content]]
            event_dates = pd.to_datetime(group["date"].astype(str), format="%Y%m%d")
            time_gaps = np.log1p(
                (target_timestamp - event_dates).dt.days.clip(lower=0).to_numpy()
            ).astype(np.float32) / np.log1p(90)
            length = min(history_length, len(item_codes))
            table_index = offset + user_code
            history_items[table_index, -length:] = item_codes[-length:]
            history_categories[table_index, -length:] = category_codes[-length:]
            history_behaviors[table_index, -length:] = behavior_codes[-length:]
            history_watch_ratios[table_index, -length:] = watch_ratios[-length:]
            history_authors[table_index, -length:] = author_codes[-length:]
            history_content[table_index, -length:] = item_content[-length:]
            history_time_gaps[table_index, -length:] = time_gaps[-length:]
    return (
        history_items,
        history_categories,
        history_behaviors,
        history_watch_ratios,
        history_authors,
        history_content,
        history_time_gaps,
    ), offsets


def build_rolling_history_tables(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    target_dates: list[int],
    encoder: CandidateFeatureEncoder,
    history_length: int,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[int, int]]:
    user_cardinality = encoder.cardinalities[0]
    history_items = np.zeros(
        (len(target_dates) * user_cardinality, history_length), dtype=np.int64
    )
    history_categories = np.zeros_like(history_items)
    offsets = {
        int(date): index * user_cardinality for index, date in enumerate(target_dates)
    }
    category_by_item = items.set_index("video_id")["first_category"].to_dict()
    positive = interactions[interactions["label_complete"].eq(1)][
        ["user_id", "video_id", "date", "timestamp"]
    ].sort_values(["user_id", "date", "timestamp"])
    for target_date in target_dates:
        historical = positive[positive["date"].lt(target_date)].groupby(
            "user_id", sort=False
        ).tail(history_length)
        offset = offsets[int(target_date)]
        for user_id, group in historical.groupby("user_id", sort=False):
            user_code = encoder.encode_values("user_id", np.array([user_id]))[0]
            if user_code == 0:
                continue
            raw_items = group["video_id"].to_numpy()
            raw_categories = np.array(
                [category_by_item.get(item, -1) for item in raw_items]
            )
            item_codes = encoder.encode_values("video_id", raw_items)
            category_codes = encoder.encode_values("first_category", raw_categories)
            length = min(history_length, len(item_codes))
            table_index = offset + user_code
            history_items[table_index, -length:] = item_codes[-length:]
            history_categories[table_index, -length:] = category_codes[-length:]
    return (history_items, history_categories), offsets


def build_history_indices(
    frame: pd.DataFrame,
    encoder: CandidateFeatureEncoder,
    offsets: dict[int, int],
) -> np.ndarray:
    user_codes = encoder.encode_values("user_id", frame["user_id"].to_numpy())
    date_offsets = frame["date"].map(offsets)
    if date_offsets.isna().any():
        missing_dates = sorted(frame.loc[date_offsets.isna(), "date"].unique())
        raise ValueError(f"Missing rolling history offsets for dates: {missing_dates}")
    return user_codes + date_offsets.to_numpy(dtype=np.int64)


def _forward(
    model: nn.Module,
    sparse: torch.Tensor,
    dense: torch.Tensor,
    history_tables: tuple[torch.Tensor, torch.Tensor] | None,
    history_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if model.requires_history:
        if history_tables is None:
            raise ValueError("DIN requires history tables")
        history_items, history_categories, *extra_history = history_tables
        user_codes = sparse[:, 0] if history_indices is None else history_indices
        selected_history = [
            history_items[user_codes],
            history_categories[user_codes],
            *(history[user_codes] for history in extra_history),
        ]
        return model(sparse, dense, *selected_history)
    return model(sparse, dense)


def train_torch_ranker(
    model: nn.Module,
    sparse: np.ndarray,
    dense: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
    history_tables: tuple[np.ndarray, np.ndarray] | None = None,
    history_indices: np.ndarray | None = None,
) -> tuple[list[dict], float]:
    model.to(device)
    sparse_tensor = torch.from_numpy(sparse)
    dense_tensor = torch.from_numpy(dense)
    label_tensor = torch.from_numpy(labels.astype(np.float32, copy=True))
    gpu_history = None
    if history_tables is not None:
        gpu_history = tuple(torch.from_numpy(table).to(device) for table in history_tables)
    history_index_tensor = (
        torch.from_numpy(history_indices) if history_indices is not None else None
    )
    positive_rate = label_tensor.mean(dim=0)
    pos_weight = torch.sqrt((1 - positive_rate) / positive_rate.clamp_min(1e-6))
    pos_weight = pos_weight.clamp(max=25).to(device)
    task_weights = torch.ones(labels.shape[1], device=device)
    if labels.shape[1] == 3:
        task_weights = torch.tensor([0.5, 0.3, 0.2], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(sparse_tensor))
        total_loss = 0.0
        seen_rows = 0
        epoch_start = time.perf_counter()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch_sparse = sparse_tensor[indices].to(device, non_blocking=True)
            batch_dense = dense_tensor[indices].to(device, non_blocking=True)
            batch_labels = label_tensor[indices].to(device, non_blocking=True)
            batch_history_indices = (
                history_index_tensor[indices].to(device, non_blocking=True)
                if history_index_tensor is not None
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = _forward(
                    model,
                    batch_sparse,
                    batch_dense,
                    gpu_history,
                    batch_history_indices,
                )
                task_losses = F.binary_cross_entropy_with_logits(
                    logits,
                    batch_labels,
                    pos_weight=pos_weight,
                    reduction="none",
                ).mean(dim=0)
                loss = torch.sum(task_losses * task_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_rows = len(indices)
            total_loss += float(loss.detach()) * batch_rows
            seen_rows += batch_rows
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / seen_rows,
                "seconds": time.perf_counter() - epoch_start,
            }
        )
    return history, time.perf_counter() - start_time


def build_pair_indices(
    frame: pd.DataFrame,
    negatives_per_positive: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positive_indices = []
    negative_indices = []
    rank_columns = ["itemcf_rank_score", "content_rank_score", "tower_rank_score"]
    for _, group in frame.groupby(["user_id", "date"], sort=False):
        positives = group.index[group["label_complete"].eq(1)].to_numpy()
        negatives = group.index[group["label_complete"].eq(0)].to_numpy()
        if len(positives) == 0 or len(negatives) == 0:
            continue
        hard_count = max(1, negatives_per_positive // 2)
        hard_negatives = (
            group.loc[negatives, rank_columns]
            .max(axis=1)
            .nlargest(min(len(negatives), hard_count * len(positives)))
            .index.to_numpy()
        )
        for positive in positives:
            chosen_hard = rng.choice(
                hard_negatives,
                size=min(hard_count, len(hard_negatives)),
                replace=False,
            )
            random_count = negatives_per_positive - len(chosen_hard)
            chosen_random = rng.choice(
                negatives,
                size=random_count,
                replace=len(negatives) < random_count,
            )
            chosen = np.concatenate([chosen_hard, chosen_random])
            positive_indices.extend([positive] * len(chosen))
            negative_indices.extend(chosen.tolist())
    return (
        np.asarray(positive_indices, dtype=np.int64),
        np.asarray(negative_indices, dtype=np.int64),
    )


def build_listwise_groups(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    groups = []
    labels = []
    for _, group in frame.groupby(["user_id", "date"], sort=False):
        group_labels = group["label_complete"].to_numpy(dtype=np.float32)
        if group_labels.sum() == 0:
            continue
        groups.append(group.index.to_numpy(dtype=np.int64))
        labels.append(group_labels)
    max_group_size = max(len(group) for group in groups)
    group_matrix = np.full((len(groups), max_group_size), -1, dtype=np.int64)
    label_matrix = np.zeros((len(groups), max_group_size), dtype=np.float32)
    for index, (group, group_labels) in enumerate(zip(groups, labels)):
        group_matrix[index, : len(group)] = group
        label_matrix[index, : len(group)] = group_labels
    return group_matrix, label_matrix


def train_pairwise_ranker(
    model: nn.Module,
    sparse: np.ndarray,
    dense: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
    history_tables: tuple[np.ndarray, np.ndarray] | None = None,
    pointwise_weight: float = 0.0,
    history_indices: np.ndarray | None = None,
) -> tuple[list[dict], float]:
    if len(positive_indices) != len(negative_indices) or len(positive_indices) == 0:
        raise ValueError("Pairwise training requires non-empty aligned positive/negative pairs.")
    model.to(device)
    sparse_tensor = torch.from_numpy(sparse)
    dense_tensor = torch.from_numpy(dense)
    positive_tensor = torch.from_numpy(positive_indices)
    negative_tensor = torch.from_numpy(negative_indices)
    gpu_history = None
    if history_tables is not None:
        gpu_history = tuple(torch.from_numpy(table).to(device) for table in history_tables)
    history_index_tensor = (
        torch.from_numpy(history_indices) if history_indices is not None else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(positive_tensor))
        total_loss = 0.0
        total_pairwise_loss = 0.0
        total_pointwise_loss = 0.0
        seen_pairs = 0
        epoch_start = time.perf_counter()
        for start in range(0, len(permutation), batch_size):
            pair_batch = permutation[start : start + batch_size]
            positive_batch = positive_tensor[pair_batch]
            negative_batch = negative_tensor[pair_batch]
            positive_sparse = sparse_tensor[positive_batch].to(device, non_blocking=True)
            positive_dense = dense_tensor[positive_batch].to(device, non_blocking=True)
            negative_sparse = sparse_tensor[negative_batch].to(device, non_blocking=True)
            negative_dense = dense_tensor[negative_batch].to(device, non_blocking=True)
            positive_history_indices = (
                history_index_tensor[positive_batch].to(device, non_blocking=True)
                if history_index_tensor is not None
                else None
            )
            negative_history_indices = (
                history_index_tensor[negative_batch].to(device, non_blocking=True)
                if history_index_tensor is not None
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                positive_logits = _forward(
                    model,
                    positive_sparse,
                    positive_dense,
                    gpu_history,
                    positive_history_indices,
                ).squeeze(-1)
                negative_logits = _forward(
                    model,
                    negative_sparse,
                    negative_dense,
                    gpu_history,
                    negative_history_indices,
                ).squeeze(-1)
                pairwise_loss = F.softplus(-(positive_logits - negative_logits)).mean()
                pointwise_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(
                        positive_logits, torch.ones_like(positive_logits)
                    )
                    + F.binary_cross_entropy_with_logits(
                        negative_logits, torch.zeros_like(negative_logits)
                    )
                )
                loss = pairwise_loss + pointwise_weight * pointwise_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            pair_count = len(pair_batch)
            total_loss += float(loss.detach()) * pair_count
            total_pairwise_loss += float(pairwise_loss.detach()) * pair_count
            total_pointwise_loss += float(pointwise_loss.detach()) * pair_count
            seen_pairs += pair_count
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / seen_pairs,
                "pairwise_loss": total_pairwise_loss / seen_pairs,
                "pointwise_loss": total_pointwise_loss / seen_pairs,
                "seconds": time.perf_counter() - epoch_start,
            }
        )
    return history, time.perf_counter() - start_time


def train_listwise_ranker(
    model: nn.Module,
    sparse: np.ndarray,
    dense: np.ndarray,
    group_indices: np.ndarray,
    group_labels: np.ndarray,
    epochs: int,
    group_batch_size: int,
    device: torch.device,
    history_tables: tuple[np.ndarray, np.ndarray] | None = None,
    history_indices: np.ndarray | None = None,
) -> tuple[list[dict], float]:
    if len(group_indices) == 0 or group_indices.shape != group_labels.shape:
        raise ValueError("Listwise training requires aligned non-empty group matrices.")
    model.to(device)
    sparse_tensor = torch.from_numpy(sparse)
    dense_tensor = torch.from_numpy(dense)
    group_tensor = torch.from_numpy(group_indices)
    label_tensor = torch.from_numpy(group_labels)
    gpu_history = None
    if history_tables is not None:
        gpu_history = tuple(torch.from_numpy(table).to(device) for table in history_tables)
    history_index_tensor = (
        torch.from_numpy(history_indices) if history_indices is not None else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(group_tensor))
        total_loss = 0.0
        seen_groups = 0
        epoch_start = time.perf_counter()
        for start in range(0, len(permutation), group_batch_size):
            batch_groups = group_tensor[permutation[start : start + group_batch_size]]
            batch_labels = label_tensor[permutation[start : start + group_batch_size]].to(
                device
            )
            mask = batch_groups.ge(0).to(device)
            safe_indices = batch_groups.clamp_min(0).flatten()
            batch_sparse = sparse_tensor[safe_indices].to(device, non_blocking=True)
            batch_dense = dense_tensor[safe_indices].to(device, non_blocking=True)
            batch_history_indices = (
                history_index_tensor[safe_indices].to(device, non_blocking=True)
                if history_index_tensor is not None
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = _forward(
                    model,
                    batch_sparse,
                    batch_dense,
                    gpu_history,
                    batch_history_indices,
                ).reshape(batch_groups.shape)
                logits = logits.masked_fill(~mask, -1e4)
                target = batch_labels / batch_labels.sum(dim=1, keepdim=True).clamp_min(
                    1.0
                )
                loss = -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_group_count = len(batch_groups)
            total_loss += float(loss.detach()) * batch_group_count
            seen_groups += batch_group_count
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / seen_groups,
                "seconds": time.perf_counter() - epoch_start,
            }
        )
    return history, time.perf_counter() - start_time


@torch.inference_mode()
def predict_torch_ranker(
    model: nn.Module,
    sparse: np.ndarray,
    dense: np.ndarray,
    batch_size: int,
    device: torch.device,
    history_tables: tuple[np.ndarray, np.ndarray] | None = None,
    history_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    model.eval()
    sparse_tensor = torch.from_numpy(sparse)
    dense_tensor = torch.from_numpy(dense)
    gpu_history = None
    if history_tables is not None:
        gpu_history = tuple(torch.from_numpy(table).to(device) for table in history_tables)
    history_index_tensor = (
        torch.from_numpy(history_indices) if history_indices is not None else None
    )
    outputs = []
    start_time = time.perf_counter()
    for start in range(0, len(sparse_tensor), batch_size):
        batch_sparse = sparse_tensor[start : start + batch_size].to(device)
        batch_dense = dense_tensor[start : start + batch_size].to(device)
        batch_history_indices = (
            history_index_tensor[start : start + batch_size].to(device)
            if history_index_tensor is not None
            else None
        )
        logits = _forward(
            model,
            batch_sparse,
            batch_dense,
            gpu_history,
            batch_history_indices,
        )
        outputs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(outputs), time.perf_counter() - start_time
