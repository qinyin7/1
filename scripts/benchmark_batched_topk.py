from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recall_models import _batched_topk_recommendations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=7176)
    parser.add_argument("--items", type=int, default=10728)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    users = list(range(args.users))
    user_vectors = torch.randn(args.users, args.dim, device=device)
    item_vectors = torch.randn(args.items, args.dim, device=device)
    index_to_item = list(range(args.items))
    item_to_index = {item: item for item in index_to_item}
    seen = {user: set(range(user % 30)) for user in users}

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    recommendations = _batched_topk_recommendations(
        users,
        users,
        user_vectors,
        item_vectors,
        index_to_item,
        item_to_index,
        seen,
        args.k,
        device,
        batch_size=args.batch_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_len = sum(len(items) for items in recommendations.values()) / len(recommendations)
    print(
        {
            "device": str(device),
            "users": args.users,
            "items": args.items,
            "dim": args.dim,
            "k": args.k,
            "batch_size": args.batch_size,
            "seconds": round(elapsed, 4),
            "avg_recommendations": round(avg_len, 2),
        }
    )


if __name__ == "__main__":
    main()
