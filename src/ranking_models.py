from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_GROUPS = {
    "F0": ["hour", "weekday", "video_duration"],
    "F1": ["user_interactions", "user_complete_rate", "user_strong_rate", "user_short_rate"],
    "F2": ["author_id", "first_category"],
    "F3": ["user_mean_watch_ratio"],
    "F4": [],
    "F5": ["item_interactions", "item_complete_rate", "item_strong_rate", "item_short_rate", "item_mean_watch_ratio"],
    "F6": [],
}


def build_ranking_frame(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    frame = interactions.merge(
        items[["video_id", "author_id", "first_category", "first_seen_date"]],
        on="video_id",
        how="left",
    )
    return frame


def feature_columns(groups: list[str]) -> list[str]:
    return list(dict.fromkeys(column for group in groups for column in FEATURE_GROUPS[group]))


@dataclass
class RankingModel:
    name: str
    model: object
    columns: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(frame[self.columns])[:, 1]

    def save(self, path) -> None:
        joblib.dump(self, path)


def train_logistic(train: pd.DataFrame, groups: list[str]) -> RankingModel:
    columns = feature_columns(groups)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=300, class_weight="balanced")),
        ]
    )
    model.fit(train[columns], train["label_complete"])
    return RankingModel("logistic_regression", model, columns)


def train_lightgbm(train: pd.DataFrame, groups: list[str], estimators: int, seed: int) -> RankingModel:
    columns = feature_columns(groups)
    model = LGBMClassifier(
        n_estimators=estimators,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train[columns], train["label_complete"])
    return RankingModel("lightgbm", model, columns)
