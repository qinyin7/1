from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def quota_fuse_rankings(
    component_rankings: list[list[int]],
    component_weights: list[float],
    k: int,
) -> list[int]:
    """Merge ranked lists with smooth weighted quotas while removing duplicates."""
    if len(component_rankings) != len(component_weights):
        raise ValueError("component_rankings and component_weights must have equal lengths")
    if not component_rankings or k <= 0:
        return []
    weights = np.asarray(component_weights, dtype=float)
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("component_weights must be non-negative with a positive sum")
    weights /= weights.sum()

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_counts = np.zeros(len(component_rankings), dtype=int)
    positions = np.zeros(len(component_rankings), dtype=int)

    while len(selected) < k:
        available = [
            index
            for index, ranking in enumerate(component_rankings)
            if positions[index] < len(ranking)
        ]
        if not available:
            break
        next_position = len(selected) + 1
        component_index = max(
            available,
            key=lambda index: (weights[index] * next_position - selected_counts[index], -index),
        )
        ranking = component_rankings[component_index]
        candidate = None
        while positions[component_index] < len(ranking):
            item = ranking[positions[component_index]]
            positions[component_index] += 1
            if item not in selected_set:
                candidate = item
                break
        if candidate is None:
            continue
        selected.append(candidate)
        selected_set.add(candidate)
        selected_counts[component_index] += 1
    return selected


def _positive_history(train: pd.DataFrame, history_length: int) -> dict[int, list[int]]:
    positive = train[train["label_complete"] == 1].sort_values("timestamp")
    return positive.groupby("user_id")["video_id"].apply(lambda values: values.tail(history_length).tolist()).to_dict()


def _batched_topk_recommendations(
    users: list[int],
    known_users: list[int],
    user_vectors: torch.Tensor,
    item_vectors: torch.Tensor,
    index_to_item: list[int],
    item_to_index: dict[int, int],
    seen: dict[int, set[int]],
    k: int,
    device: torch.device,
    allowed_items: set[int] | None = None,
    batch_size: int = 1024,
) -> dict[int, list[int]]:
    """Score users against all item embeddings with batched matrix topK."""
    output = {user: [] for user in users}
    if not known_users or k <= 0 or not index_to_item:
        return output

    item_count = len(index_to_item)
    top_k = min(k, item_count)
    allowed_mask = torch.ones(item_count, dtype=torch.bool, device=device)
    if allowed_items is not None:
        allowed_mask = torch.tensor(
            [item in allowed_items for item in index_to_item],
            dtype=torch.bool,
            device=device,
        )

    item_vectors = item_vectors.to(device)
    user_vectors = user_vectors.to(device)
    for start in range(0, len(known_users), batch_size):
        end = start + batch_size
        batch_users = known_users[start:end]
        scores = user_vectors[start:end] @ item_vectors.T
        scores[:, ~allowed_mask] = -torch.inf
        for row_index, user in enumerate(batch_users):
            seen_indices = [
                item_to_index[item]
                for item in seen.get(user, set())
                if item in item_to_index
            ]
            if seen_indices:
                scores[row_index, seen_indices] = -torch.inf
        top_scores, top_indices = torch.topk(scores, top_k, dim=1)
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        for row_index, user in enumerate(batch_users):
            output[user] = [
                index_to_item[int(item_index)]
                for score, item_index in zip(top_scores[row_index], top_indices[row_index])
                if np.isfinite(score)
            ][:k]
    return output


def _load_torch_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class PopularRecall:
    def __init__(self, time_decay: bool = False, half_life_days: float = 7, window_days: int | None = None):
        self.time_decay = time_decay
        self.half_life_days = half_life_days
        self.window_days = window_days
        self.ranked_items: list[int] = []

    def fit(self, train: pd.DataFrame) -> None:
        if self.window_days is not None:
            dates = pd.to_datetime(train["date"].astype(str), format="%Y%m%d")
            cutoff = dates.max() - pd.Timedelta(days=self.window_days - 1)
            train = train[dates >= cutoff]
        weights = 1 + train["label_complete"] + train["label_strong"] - 0.5 * train["label_short"]
        if self.time_decay:
            dates = pd.to_datetime(train["date"].astype(str), format="%Y%m%d")
            age_days = (dates.max() - dates).dt.days
            weights = weights * np.exp(-age_days / self.half_life_days)
        scores = weights.groupby(train["video_id"]).sum()
        self.ranked_items = scores.sort_values(ascending=False).index.tolist()

    def recommend(
        self,
        users: list[int],
        seen: dict[int, set[int]],
        k: int,
        allowed_items: set[int] | None = None,
    ) -> dict[int, list[int]]:
        return {
            user: [
                item
                for item in self.ranked_items
                if item not in seen.get(user, set())
                and (allowed_items is None or item in allowed_items)
            ][:k]
            for user in users
        }


class ItemCFRecall:
    def __init__(
        self,
        history_length: int = 50,
        neighbors: int = 200,
        time_decay: bool = True,
        use_iuf: bool = True,
        feedback: str = "complete",
    ):
        self.history_length = history_length
        self.neighbors = neighbors
        self.time_decay = time_decay
        self.use_iuf = use_iuf
        self.feedback = feedback
        self.similarity: dict[int, list[tuple[int, float]]] = {}
        self.user_history: dict[int, list[int]] = {}
        self.user_item_weight: dict[int, dict[int, float]] = {}
        self.fallback: list[int] = []

    def fit(self, train: pd.DataFrame) -> None:
        if self.feedback == "multi_feedback":
            positive = train[train["label_complete"] == 1].copy()
            positive["feedback_weight"] = 1.0 + positive["label_strong"] - 0.5 * positive["label_short"]
            positive = positive.sort_values("timestamp")
            self.user_history = (
                positive.groupby("user_id")["video_id"]
                .apply(lambda values: values.tail(self.history_length).tolist())
                .to_dict()
            )
            self.user_item_weight = defaultdict(dict)
            for (user, item), weight in positive.groupby(["user_id", "video_id"])[
                "feedback_weight"
            ].max().items():
                self.user_item_weight[user][item] = float(weight)
        else:
            self.user_history = _positive_history(train, self.history_length)
            self.user_item_weight = {
                user: {item: 1.0 for item in history} for user, history in self.user_history.items()
            }
        item_count = Counter()
        cooccurrence: dict[int, Counter] = defaultdict(Counter)
        for user, items in self.user_history.items():
            unique_items = list(dict.fromkeys(items))
            user_weight = 1 / math.log1p(len(unique_items) + 1) if self.use_iuf else 1.0
            for item in unique_items:
                item_count[item] += 1
                for other in unique_items:
                    if item != other:
                        feedback_weight = math.sqrt(
                            self.user_item_weight[user].get(item, 1.0)
                            * self.user_item_weight[user].get(other, 1.0)
                        )
                        cooccurrence[item][other] += user_weight * feedback_weight
        for item, related in cooccurrence.items():
            scored = [
                (other, value / math.sqrt(item_count[item] * item_count[other]))
                for other, value in related.items()
            ]
            self.similarity[item] = sorted(scored, key=lambda pair: pair[1], reverse=True)[: self.neighbors]
        self.fallback = [item for item, _ in item_count.most_common()]

    def recommend(
        self,
        users: list[int],
        seen: dict[int, set[int]],
        k: int,
        allowed_items: set[int] | None = None,
    ) -> dict[int, list[int]]:
        output = {}
        for user in users:
            scores = Counter()
            history = self.user_history.get(user, [])
            for position, item in enumerate(reversed(history)):
                decay = 0.9**position if self.time_decay else 1.0
                for candidate, similarity in self.similarity.get(item, []):
                    if candidate not in seen.get(user, set()) and (
                        allowed_items is None or candidate in allowed_items
                    ):
                        scores[candidate] += (
                            decay * similarity * self.user_item_weight.get(user, {}).get(item, 1.0)
                        )
            ranked = [item for item, _ in scores.most_common(k)]
            ranked.extend(
                item
                for item in self.fallback
                if item not in seen.get(user, set())
                and item not in ranked
                and (allowed_items is None or item in allowed_items)
            )
            output[user] = ranked[:k]
        return output


class ContentRecall:
    def __init__(self, history_length: int = 50):
        self.history_length = history_length
        self.item_category: dict[int, int] = {}
        self.category_items: dict[int, list[int]] = {}
        self.user_categories: dict[int, list[int]] = {}
        self.known_items: set[int] = set()

    def fit(self, train: pd.DataFrame, items: pd.DataFrame, eligible_items: set[int]) -> None:
        self.known_items = set(train["video_id"].unique())
        items = items[items["video_id"].isin(eligible_items)]
        self.item_category = items.set_index("video_id")["first_category"].to_dict()
        popularity = train.groupby("video_id").size().to_dict()
        category_items: dict[int, list[int]] = defaultdict(list)
        # Content recall can retrieve unseen videos; known popularity only controls ordering.
        for item in sorted(self.item_category, key=lambda value: popularity.get(value, 0), reverse=True):
            category_items[self.item_category.get(item, -1)].append(item)
        self.category_items = dict(category_items)
        histories = _positive_history(train, self.history_length)
        for user, history in histories.items():
            counts = Counter(self.item_category.get(item, -1) for item in history)
            self.user_categories[user] = [category for category, _ in counts.most_common()]

    def recommend(
        self,
        users: list[int],
        seen: dict[int, set[int]],
        k: int,
        allowed_items: set[int] | None = None,
    ) -> dict[int, list[int]]:
        output = {}
        for user in users:
            ranked, cold_ranked = [], []
            for category in self.user_categories.get(user, []):
                for item in self.category_items.get(category, []):
                    if item in seen.get(user, set()) or (
                        allowed_items is not None and item not in allowed_items
                    ):
                        continue
                    target = ranked if item in self.known_items else cold_ranked
                    if item not in target:
                        target.append(item)
                    if len(ranked) + len(cold_ranked) >= k * 4:
                        break
                if len(ranked) + len(cold_ranked) >= k * 4:
                    break
            cold_quota = max(1, k // 5)
            selected_cold = cold_ranked[:cold_quota]
            selected_known = ranked[: k - cold_quota]
            merged = []
            for index in range(k):
                if index % 5 == 0 and selected_cold:
                    merged.append(selected_cold.pop(0))
                elif selected_known:
                    merged.append(selected_known.pop(0))
                elif selected_cold:
                    merged.append(selected_cold.pop(0))
            output[user] = merged
        return output


class TextContentRecall:
    def __init__(self, history_length: int = 50, category_weight: float = 0.0):
        self.history_length = history_length
        self.category_weight = category_weight
        self.item_ids: list[int] = []
        self.item_to_index: dict[int, int] = {}
        self.item_matrix = None
        self.item_categories = np.array([])
        self.user_history: dict[int, list[int]] = {}
        self.user_category_preferences: dict[int, set[int]] = {}
        self.known_items: set[int] = set()

    def fit(self, train: pd.DataFrame, items: pd.DataFrame, eligible_items: set[int]) -> None:
        self.known_items = set(train["video_id"].unique())
        eligible = (
            items[items["video_id"].isin(eligible_items)]
            .drop_duplicates("video_id")
            .sort_values("video_id")
        )
        self.item_ids = eligible["video_id"].astype(int).tolist()
        self.item_to_index = {item: index for index, item in enumerate(self.item_ids)}
        self.item_categories = eligible["first_category"].fillna(-1).to_numpy()
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=2,
            max_features=30_000,
            sublinear_tf=True,
            norm="l2",
        )
        self.item_matrix = vectorizer.fit_transform(eligible["content_text"].fillna(""))
        self.user_history = _positive_history(train, self.history_length)
        category_by_item = items.set_index("video_id")["first_category"].to_dict()
        for user, history in self.user_history.items():
            self.user_category_preferences[user] = {
                category_by_item.get(item, -1) for item in history if category_by_item.get(item, -1) != -1
            }

    def recommend(
        self,
        users: list[int],
        seen: dict[int, set[int]],
        k: int,
        allowed_items: set[int] | None = None,
    ) -> dict[int, list[int]]:
        assert self.item_matrix is not None
        output = {}
        for user in users:
            history_indices = [
                self.item_to_index[item]
                for item in self.user_history.get(user, [])
                if item in self.item_to_index
            ]
            if not history_indices:
                output[user] = []
                continue
            profile = self.item_matrix[history_indices].mean(axis=0)
            scores = np.asarray(self.item_matrix @ profile.T).ravel()
            if self.category_weight:
                preferred = self.user_category_preferences.get(user, set())
                scores += self.category_weight * np.isin(self.item_categories, list(preferred))
            order = np.argsort(-scores)
            known_ranked, cold_ranked = [], []
            for index in order:
                item = self.item_ids[index]
                if item in seen.get(user, set()) or (
                    allowed_items is not None and item not in allowed_items
                ):
                    continue
                target = known_ranked if item in self.known_items else cold_ranked
                target.append(item)
                if len(known_ranked) + len(cold_ranked) >= k * 2:
                    break
            cold_quota = max(1, k // 5)
            selected_cold = cold_ranked[:cold_quota]
            selected_known = known_ranked[: k - cold_quota]
            merged = []
            for index in range(k):
                if index % 5 == 0 and selected_cold:
                    merged.append(selected_cold.pop(0))
                elif selected_known:
                    merged.append(selected_known.pop(0))
                elif selected_cold:
                    merged.append(selected_cold.pop(0))
            output[user] = merged
        return output


class _MatrixFactorization(nn.Module):
    def __init__(self, user_count: int, item_count: int, dimension: int):
        super().__init__()
        self.user_embedding = nn.Embedding(user_count, dimension)
        self.item_embedding = nn.Embedding(item_count, dimension)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_embedding(users) * self.item_embedding(items)).sum(dim=1)


class TwoTowerRecall:
    def __init__(self, dimension: int, epochs: int, batch_size: int, seed: int):
        self.dimension, self.epochs, self.batch_size, self.seed = dimension, epochs, batch_size, seed
        self.user_to_index: dict[int, int] = {}
        self.item_to_index: dict[int, int] = {}
        self.index_to_item: list[int] = []
        self.model: _MatrixFactorization | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train: pd.DataFrame) -> None:
        positive = train[train["label_complete"] == 1][["user_id", "video_id"]].drop_duplicates()
        users = sorted(positive["user_id"].unique())
        items = sorted(train["video_id"].unique())
        self.user_to_index = {value: index for index, value in enumerate(users)}
        self.item_to_index = {value: index for index, value in enumerate(items)}
        self.index_to_item = items
        user_index = positive["user_id"].map(self.user_to_index).to_numpy()
        positive_index = positive["video_id"].map(self.item_to_index).to_numpy()
        dataset = TensorDataset(torch.tensor(user_index), torch.tensor(positive_index))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = _MatrixFactorization(len(users), len(items), self.dimension).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.model.train()
        for _ in range(self.epochs):
            for batch_users, batch_positive in loader:
                batch_users, batch_positive = batch_users.to(self.device), batch_positive.to(self.device)
                batch_negative = torch.randint(len(items), batch_positive.shape, device=self.device, generator=generator)
                loss = -torch.nn.functional.logsigmoid(
                    self.model(batch_users, batch_positive) - self.model(batch_users, batch_negative)
                ).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def save_checkpoint(self, path: Path) -> None:
        assert self.model is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "class_name": self.__class__.__name__,
                "dimension": self.dimension,
                "user_to_index": self.user_to_index,
                "item_to_index": self.item_to_index,
                "index_to_item": self.index_to_item,
                "model_state": {
                    key: value.detach().cpu() for key, value in self.model.state_dict().items()
                },
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = _load_torch_checkpoint(path, self.device)
        self.user_to_index = {int(key): int(value) for key, value in checkpoint["user_to_index"].items()}
        self.item_to_index = {int(key): int(value) for key, value in checkpoint["item_to_index"].items()}
        self.index_to_item = [int(item) for item in checkpoint["index_to_item"]]
        self.model = _MatrixFactorization(
            len(self.user_to_index), len(self.index_to_item), int(checkpoint["dimension"])
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def recommend(
        self,
        users: list[int],
        seen: dict[int, set[int]],
        k: int,
        allowed_items: set[int] | None = None,
    ) -> dict[int, list[int]]:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            known_users = [user for user in users if user in self.user_to_index]
            user_indices = torch.tensor(
                [self.user_to_index[user] for user in known_users],
                device=self.device,
                dtype=torch.long,
            )
            user_vectors = self.model.user_embedding(user_indices)
            item_vectors = self.model.item_embedding.weight.detach()
            return _batched_topk_recommendations(
                users,
                known_users,
                user_vectors,
                item_vectors,
                self.index_to_item,
                self.item_to_index,
                seen,
                k,
                self.device,
                allowed_items,
                self.batch_size,
            )


class _FeatureTwoTower(nn.Module):
    def __init__(
        self,
        user_count: int,
        item_count: int,
        user_feature_dim: int,
        item_feature_dim: int,
        dimension: int,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(user_count, dimension)
        self.item_embedding = nn.Embedding(item_count, dimension)
        self.user_feature_projection = nn.Linear(user_feature_dim, dimension)
        self.item_feature_projection = nn.Linear(item_feature_dim, dimension)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

    def user_vector(self, users: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(
            self.user_embedding(users) + self.user_feature_projection(features), dim=1
        )

    def item_vector(
        self, items: torch.Tensor, features: torch.Tensor, use_id_embedding: torch.Tensor
    ) -> torch.Tensor:
        id_vector = self.item_embedding(items) * use_id_embedding.unsqueeze(1)
        return torch.nn.functional.normalize(id_vector + self.item_feature_projection(features), dim=1)


class FeatureTwoTowerRecall:
    def __init__(
        self,
        dimension: int,
        epochs: int,
        batch_size: int,
        seed: int,
        history_length: int = 50,
        id_dropout: float = 0.0,
        hard_negative_ratio: float = 0.0,
    ):
        self.dimension, self.epochs, self.batch_size, self.seed = dimension, epochs, batch_size, seed
        self.history_length = history_length
        self.id_dropout = id_dropout
        self.hard_negative_ratio = hard_negative_ratio
        self.user_to_index: dict[int, int] = {}
        self.item_to_index: dict[int, int] = {}
        self.index_to_item: list[int] = []
        self.known_item_mask: torch.Tensor | None = None
        self.user_features: torch.Tensor | None = None
        self.item_features: torch.Tensor | None = None
        self.model: _FeatureTwoTower | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_item_features(self, items: pd.DataFrame) -> np.ndarray:
        texts = items["content_text"].fillna("")
        tfidf = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=2,
            max_features=20_000,
            sublinear_tf=True,
        ).fit_transform(texts)
        components = min(32, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        text_features = TruncatedSVD(n_components=max(1, components), random_state=self.seed).fit_transform(tfidf)
        category_features = pd.get_dummies(items["first_category"].fillna(-1), dtype=float).to_numpy()
        author_hash = pd.get_dummies(items["author_id"].fillna(-1).astype(int) % 64, dtype=float).to_numpy()
        duration = np.log1p(items["video_duration"].fillna(0).to_numpy()).reshape(-1, 1)
        duration = StandardScaler().fit_transform(duration)
        return np.hstack([text_features, category_features, author_hash, duration]).astype("float32")

    def _build_user_features(
        self, train: pd.DataFrame, users: pd.DataFrame, item_features: np.ndarray
    ) -> np.ndarray:
        histories = _positive_history(train, self.history_length)
        history_features = np.zeros((len(self.user_to_index), item_features.shape[1]), dtype="float32")
        for user, history in histories.items():
            indices = [self.item_to_index[item] for item in history if item in self.item_to_index]
            if indices:
                history_features[self.user_to_index[user]] = item_features[indices].mean(axis=0)
        profile = users.set_index("user_id").reindex(self.user_to_index)
        numeric_columns = [
            "is_lowactive_period",
            "is_live_streamer",
            "is_video_author",
            "follow_user_num",
            "fans_user_num",
            "friend_user_num",
            "register_days",
        ]
        numeric = profile[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        numeric = StandardScaler().fit_transform(numeric).astype("float32")
        active = pd.get_dummies(profile["user_active_degree"].fillna("UNKNOWN"), dtype=float).to_numpy(
            dtype="float32"
        )
        return np.hstack([history_features, numeric, active]).astype("float32")

    def fit(
        self,
        train: pd.DataFrame,
        users: pd.DataFrame,
        items: pd.DataFrame,
        eligible_items: set[int],
    ) -> None:
        positive = train[train["label_complete"] == 1][["user_id", "video_id"]].drop_duplicates()
        user_ids = sorted(positive["user_id"].unique())
        eligible = (
            items[items["video_id"].isin(eligible_items)]
            .drop_duplicates("video_id")
            .sort_values("video_id")
            .reset_index(drop=True)
        )
        self.index_to_item = eligible["video_id"].astype(int).tolist()
        self.user_to_index = {value: index for index, value in enumerate(user_ids)}
        self.item_to_index = {value: index for index, value in enumerate(self.index_to_item)}
        positive = positive[
            positive["user_id"].isin(self.user_to_index) & positive["video_id"].isin(self.item_to_index)
        ]
        item_features = self._build_item_features(eligible)
        user_features = self._build_user_features(train, users, item_features)
        known_items = set(train["video_id"].unique())
        known_mask = np.array([item in known_items for item in self.index_to_item], dtype="float32")
        known_indices = np.flatnonzero(known_mask)
        categories = eligible["first_category"].fillna(-1).to_numpy()
        known_by_category = {
            category: np.flatnonzero((categories == category) & (known_mask == 1))
            for category in np.unique(categories)
        }

        user_index = positive["user_id"].map(self.user_to_index).to_numpy()
        positive_index = positive["video_id"].map(self.item_to_index).to_numpy()
        rng = np.random.default_rng(self.seed)
        hard_negative_index = np.empty_like(positive_index)
        for row, item_index in enumerate(positive_index):
            candidates = known_by_category.get(categories[item_index], known_indices)
            hard_negative_index[row] = rng.choice(candidates if len(candidates) > 1 else known_indices)
            if hard_negative_index[row] == item_index and len(candidates) > 1:
                hard_negative_index[row] = rng.choice(candidates[candidates != item_index])
        dataset = TensorDataset(
            torch.tensor(user_index),
            torch.tensor(positive_index),
            torch.tensor(hard_negative_index),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.user_features = torch.tensor(user_features, device=self.device)
        self.item_features = torch.tensor(item_features, device=self.device)
        self.known_item_mask = torch.tensor(known_mask, device=self.device)
        known_indices_tensor = torch.tensor(known_indices, device=self.device)
        self.model = _FeatureTwoTower(
            len(user_ids), len(self.index_to_item), user_features.shape[1], item_features.shape[1], self.dimension
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.model.train()
        for _ in range(self.epochs):
            for batch_users, batch_positive, batch_hard_negative in loader:
                batch_users = batch_users.to(self.device)
                batch_positive = batch_positive.to(self.device)
                batch_hard_negative = batch_hard_negative.to(self.device)
                random_positions = torch.randint(
                    len(known_indices_tensor), batch_positive.shape, device=self.device, generator=generator
                )
                batch_negative = known_indices_tensor[random_positions]
                if self.hard_negative_ratio:
                    use_hard = torch.rand(
                        batch_positive.shape, device=self.device, generator=generator
                    ) < self.hard_negative_ratio
                    batch_negative = torch.where(use_hard, batch_hard_negative, batch_negative)
                user_vectors = self.model.user_vector(batch_users, self.user_features[batch_users])
                positive_id_mask = self.known_item_mask[batch_positive]
                negative_id_mask = self.known_item_mask[batch_negative]
                if self.id_dropout:
                    positive_id_mask = positive_id_mask * (
                        torch.rand(batch_positive.shape, device=self.device, generator=generator)
                        >= self.id_dropout
                    )
                    negative_id_mask = negative_id_mask * (
                        torch.rand(batch_negative.shape, device=self.device, generator=generator)
                        >= self.id_dropout
                    )
                positive_vectors = self.model.item_vector(
                    batch_positive, self.item_features[batch_positive], positive_id_mask
                )
                negative_vectors = self.model.item_vector(
                    batch_negative, self.item_features[batch_negative], negative_id_mask
                )
                loss = -torch.nn.functional.logsigmoid(
                    (user_vectors * positive_vectors).sum(dim=1)
                    - (user_vectors * negative_vectors).sum(dim=1)
                ).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def save_checkpoint(self, path: Path) -> None:
        assert self.model is not None
        assert self.user_features is not None and self.item_features is not None
        assert self.known_item_mask is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "class_name": self.__class__.__name__,
                "dimension": self.dimension,
                "history_length": self.history_length,
                "id_dropout": self.id_dropout,
                "hard_negative_ratio": self.hard_negative_ratio,
                "user_to_index": self.user_to_index,
                "item_to_index": self.item_to_index,
                "index_to_item": self.index_to_item,
                "user_features": self.user_features.detach().cpu(),
                "item_features": self.item_features.detach().cpu(),
                "known_item_mask": self.known_item_mask.detach().cpu(),
                "model_state": {
                    key: value.detach().cpu() for key, value in self.model.state_dict().items()
                },
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = _load_torch_checkpoint(path, self.device)
        self.user_to_index = {int(key): int(value) for key, value in checkpoint["user_to_index"].items()}
        self.item_to_index = {int(key): int(value) for key, value in checkpoint["item_to_index"].items()}
        self.index_to_item = [int(item) for item in checkpoint["index_to_item"]]
        self.user_features = checkpoint["user_features"].to(self.device)
        self.item_features = checkpoint["item_features"].to(self.device)
        self.known_item_mask = checkpoint["known_item_mask"].to(self.device)
        self.model = _FeatureTwoTower(
            len(self.user_to_index),
            len(self.index_to_item),
            self.user_features.shape[1],
            self.item_features.shape[1],
            int(checkpoint["dimension"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def recommend(
        self,
        users: list[int],
        seen: dict[int, set[int]],
        k: int,
        allowed_items: set[int] | None = None,
    ) -> dict[int, list[int]]:
        assert self.model is not None
        assert self.user_features is not None and self.item_features is not None
        assert self.known_item_mask is not None
        self.model.eval()
        with torch.no_grad():
            item_indices = torch.arange(len(self.index_to_item), device=self.device)
            item_vectors = self.model.item_vector(item_indices, self.item_features, self.known_item_mask)
            known_users = [user for user in users if user in self.user_to_index]
            user_indices = torch.tensor(
                [self.user_to_index[user] for user in known_users],
                device=self.device,
                dtype=torch.long,
            )
            user_vectors = self.model.user_vector(user_indices, self.user_features[user_indices])
            return _batched_topk_recommendations(
                users,
                known_users,
                user_vectors,
                item_vectors,
                self.index_to_item,
                self.item_to_index,
                seen,
                k,
                self.device,
                allowed_items,
                self.batch_size,
            )
