from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "raw" / "extracted" / "KuaiRec 2.0" / "data"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
RESULT_SCHEMA_VERSION = 7


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_profile(profile_name: str) -> dict[str, Any]:
    profiles = load_yaml(ROOT / "configs" / "profiles.yaml")
    if profile_name not in profiles:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(profiles)}")
    return profiles[profile_name]


def resolve_candidate_limits(
    profile: dict[str, Any],
    top_per_channel: int | None,
    max_candidates_per_group: int | None,
) -> tuple[int, int]:
    """Resolve candidate-pool sizes with CLI values taking precedence."""
    resolved_top = (
        top_per_channel
        if top_per_channel is not None
        else int(profile.get("top_per_channel", 150))
    )
    resolved_max = (
        max_candidates_per_group
        if max_candidates_per_group is not None
        else int(profile.get("max_candidates_per_group", 300))
    )
    return resolved_top, resolved_max


def resolve_candidate_channel_limits(
    profile: dict[str, Any],
    top_per_channel: int,
    channels: list[str],
) -> dict[str, int]:
    """Resolve per-channel candidate limits from optional profile weights."""
    weights = profile.get("candidate_channel_weights") or {}
    return {
        channel: max(1, int(round(top_per_channel * float(weights.get(channel, 1.0)))))
        for channel in channels
    }


def candidate_channels(profile: dict[str, Any], default_channels: list[str]) -> list[str]:
    """Return profile-specific candidate channels while preserving old defaults."""
    return list(profile.get("candidate_channels") or default_channels)


def ranking_eval_ks(profile: dict[str, Any]) -> list[int]:
    """Return ranking cutoffs while preserving @200 comparability for large runs."""
    primary_k = int(profile["recall_k"])
    eval_ks = {10, primary_k}
    if primary_k >= 200:
        eval_ks.add(200)
    return sorted(eval_ks)


def profile_dir(profile_name: str) -> Path:
    path = PROCESSED_DIR / profile_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_dir(profile_name: str, stage: str) -> Path:
    path = ARTIFACT_DIR / profile_name / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sample_rows(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame
    return frame.sample(max_rows, random_state=seed).sort_index()


def sample_complete_users(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame
    user_count = frame["user_id"].nunique()
    sampled_user_count = max(1, int(user_count * max_rows / len(frame)))
    sampled_users = (
        frame["user_id"].drop_duplicates().sample(sampled_user_count, random_state=seed).tolist()
    )
    return frame[frame["user_id"].isin(sampled_users)]


def append_result(path: Path, row: dict[str, Any]) -> None:
    run_dir = path.parent / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **row,
    }
    write_json(run_dir / f"{result['run_id']}.json", result)
    rows = [
        row
        for run_file in sorted(run_dir.glob("*.json"))
        if (row := json.loads(run_file.read_text(encoding="utf-8"))).get("schema_version")
        == RESULT_SCHEMA_VERSION
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
