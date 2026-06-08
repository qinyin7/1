from __future__ import annotations

import ast
from functools import lru_cache

import numpy as np
import pandas as pd

from src.common import RAW_DATA_DIR, profile_dir, write_json


TRAIN_END = 20200810
VALID_END = 20200831
FEATURE_END = 20200712
PROCESSED_SCHEMA_VERSION = 4


@lru_cache(maxsize=1)
def _read_small_matrix_users() -> frozenset[int]:
    users: set[int] = set()
    for chunk in pd.read_csv(
        RAW_DATA_DIR / "small_matrix.csv",
        usecols=["user_id"],
        chunksize=500_000,
    ):
        users.update(chunk["user_id"].unique())
    return frozenset(users)


def _select_users(
    max_users: int | None,
    seed: int,
    full_exposure_user_fraction: float = 0.5,
) -> set[int] | None:
    if max_users is None:
        return None
    all_users = pd.read_csv(RAW_DATA_DIR / "user_features.csv", usecols=["user_id"])["user_id"]
    small_users = _read_small_matrix_users()
    remaining = all_users[~all_users.isin(small_users)]
    target_small_users = min(
        len(small_users),
        max_users,
        max(2, round(max_users * full_exposure_user_fraction)),
    )
    selected_small = set(
        pd.Series(sorted(small_users))
        .sample(target_small_users, random_state=seed)
        .tolist()
    )
    additional = remaining.sample(
        min(max_users - len(selected_small), len(remaining)),
        random_state=seed,
    )
    return selected_small | set(additional.tolist())


def read_interactions(selected_users: set[int] | None, filename: str = "big_matrix.csv") -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(RAW_DATA_DIR / filename, chunksize=500_000):
        if selected_users is not None:
            chunk = chunk[chunk["user_id"].isin(selected_users)]
        if not chunk.empty:
            parts.append(chunk)
    frame = pd.concat(parts, ignore_index=True)
    frame = frame[frame["date"].notna()].copy()
    frame["date"] = frame["date"].astype("int64")
    frame["split"] = np.select(
        [frame["date"] <= TRAIN_END, frame["date"] <= VALID_END],
        ["train", "valid"],
        default="test",
    )
    frame["label_complete"] = (frame["watch_ratio"] >= 1).astype("int8")
    frame["label_strong"] = (frame["watch_ratio"] >= 2).astype("int8")
    frame["label_short"] = (
        frame["play_duration"] < np.minimum(3000, frame["video_duration"])
    ).astype("int8")
    local_time = pd.to_datetime(frame["time"], errors="coerce")
    if local_time.isna().any():
        timestamp_fallback = (
            pd.to_datetime(frame.loc[local_time.isna(), "timestamp"], unit="s", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
        )
        local_time.loc[local_time.isna()] = timestamp_fallback
    if local_time.isna().any():
        date_fallback = pd.to_datetime(
            frame.loc[local_time.isna(), "date"].astype(str), format="%Y%m%d"
        ) + pd.Timedelta(hours=12)
        local_time.loc[local_time.isna()] = date_fallback
    frame["hour"] = local_time.dt.hour.astype("int8")
    frame["weekday"] = local_time.dt.weekday.astype("int8")
    return frame


def _parse_first_category(value: str) -> int:
    values = ast.literal_eval(value) if isinstance(value, str) else []
    return int(values[0]) if values else -1


def _build_static_features(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv(RAW_DATA_DIR / "user_features.csv")
    categories = pd.read_csv(RAW_DATA_DIR / "item_categories.csv")
    categories["first_category"] = categories["feat"].map(_parse_first_category)
    content = pd.read_csv(
        RAW_DATA_DIR / "kuairec_caption_category.csv",
        engine="python",
        on_bad_lines="skip",
        usecols=[
            "video_id",
            "manual_cover_text",
            "caption",
            "topic_tag",
            "first_level_category_name",
            "second_level_category_name",
            "third_level_category_name",
        ],
    )
    content["video_id"] = pd.to_numeric(content["video_id"], errors="coerce")
    content = content.dropna(subset=["video_id"]).copy()
    content["video_id"] = content["video_id"].astype("int64")
    text_columns = [
        "manual_cover_text",
        "caption",
        "topic_tag",
        "first_level_category_name",
        "second_level_category_name",
        "third_level_category_name",
    ]
    content["content_text"] = (
        content[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace("UNKNOWN", "", regex=False)
    )
    daily_items = pd.read_csv(
        RAW_DATA_DIR / "item_daily_features.csv",
        usecols=["video_id", "date", "author_id", "video_type", "video_duration"],
    )
    first_seen = daily_items.groupby("video_id", as_index=False)["date"].min().rename(
        columns={"date": "first_seen_date"}
    )
    latest_items = (
        daily_items.query("date <= @TRAIN_END")
        .sort_values("date")
        .drop_duplicates("video_id", keep="last")
        .merge(first_seen, on="video_id", how="right")
    )
    items = categories.merge(latest_items, on="video_id", how="left").merge(
        content[["video_id", "content_text"]], on="video_id", how="left"
    )
    items["content_text"] = items["content_text"].fillna("")
    items["video_type"] = items["video_type"].fillna("UNKNOWN")
    items["author_id"] = items["author_id"].fillna(-1).astype("int64")
    duration_by_item = interactions.groupby("video_id")["video_duration"].median()
    items["video_duration"] = items["video_duration"].fillna(items["video_id"].map(duration_by_item))
    return users, items


def _aggregate_statistics(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = history.assign(watch_ratio_clipped=history["watch_ratio"].clip(upper=5))
    user_stats = history.groupby("user_id").agg(
        user_interactions=("video_id", "size"),
        user_complete_rate=("label_complete", "mean"),
        user_strong_rate=("label_strong", "mean"),
        user_short_rate=("label_short", "mean"),
        user_mean_watch_ratio=("watch_ratio_clipped", "mean"),
    )
    item_stats = history.groupby("video_id").agg(
        item_interactions=("user_id", "size"),
        item_complete_rate=("label_complete", "mean"),
        item_strong_rate=("label_strong", "mean"),
        item_short_rate=("label_short", "mean"),
        item_mean_watch_ratio=("watch_ratio_clipped", "mean"),
    )
    return user_stats.reset_index(), item_stats.reset_index()


def _build_point_in_time_statistics(interactions: pd.DataFrame) -> pd.DataFrame:
    enriched_parts = []
    for date in sorted(interactions["date"].unique()):
        current = interactions[interactions["date"] == date].copy()
        history = interactions[interactions["date"] < date]
        if history.empty:
            enriched_parts.append(current)
            continue
        user_stats, item_stats = _aggregate_statistics(history)
        current = current.merge(user_stats, on="user_id", how="left")
        current = current.merge(item_stats, on="video_id", how="left")
        enriched_parts.append(current)
    return pd.concat(enriched_parts, ignore_index=True)


def attach_history_statistics(target: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    enriched_parts = []
    statistic_columns = [
        column
        for column in target.columns
        if column != "user_id" and (column.startswith("user_") or column.startswith("item_"))
    ]
    target = target.drop(columns=statistic_columns, errors="ignore")
    for date in sorted(target["date"].unique()):
        current = target[target["date"] == date].copy()
        historical = history[history["date"] < date]
        user_stats, item_stats = _aggregate_statistics(historical)
        current = current.merge(user_stats, on="user_id", how="left")
        current = current.merge(item_stats, on="video_id", how="left")
        enriched_parts.append(current)
    return pd.concat(enriched_parts, ignore_index=True)


def prepare_profile(profile_name: str, profile: dict, seed: int = 2026) -> dict:
    output = profile_dir(profile_name)
    selected_users = _select_users(
        profile["max_users"],
        seed,
        profile.get("full_exposure_user_fraction", 0.5),
    )
    interactions = read_interactions(selected_users)
    users, items = _build_static_features(interactions)
    interactions = _build_point_in_time_statistics(interactions)

    interactions.to_parquet(output / "interactions.parquet", index=False)
    users.to_parquet(output / "users.parquet", index=False)
    items.to_parquet(output / "items.parquet", index=False)

    summary = {
        "processed_schema_version": PROCESSED_SCHEMA_VERSION,
        "profile": profile_name,
        "rows": int(len(interactions)),
        "users": int(interactions["user_id"].nunique()),
        "items": int(interactions["video_id"].nunique()),
        "full_exposure_users_in_profile": int(
            read_full_exposure_user_count(set(interactions["user_id"].unique()))
        ),
        "split_rows": {key: int(value) for key, value in interactions["split"].value_counts().items()},
        "complete_rate": float(interactions["label_complete"].mean()),
        "seed": seed,
        "preserves_complete_user_histories": True,
        "interaction_columns": sorted(interactions.columns.tolist()),
        "item_columns": sorted(items.columns.tolist()),
    }
    write_json(output / "summary.json", summary)
    return summary


def read_full_exposure_user_count(selected_users: set[int]) -> int:
    return len(_read_small_matrix_users() & selected_users)
