from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_experiment import run_ranking, run_recall
from src.common import ROOT, load_profile, load_yaml, set_seed
from src.data_pipeline import prepare_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["local_8gb_large", "full_24gb"],
        default="local_8gb_large",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    args = parser.parse_args()
    profile = load_profile(args.profile)
    config = load_yaml(ROOT / "configs" / "experiments.yaml")
    prepare_profile(args.profile, profile, profile["data_seed"])

    for seed in args.seeds:
        set_seed(seed)
        for experiment in config["recall_experiments"]:
            if experiment.get("enabled", True):
                run_recall(
                    args.profile,
                    profile,
                    experiment["model"],
                    seed,
                    experiment["id"],
                    experiment.get("params", {}),
                    "full_val",
                )

        for experiment in config["ranking_experiments"]:
            if not experiment.get("enabled", True):
                continue
            model_name = "logistic" if experiment["model"] == "logistic_regression" else experiment["model"]
            run_ranking(
                args.profile,
                profile,
                model_name,
                experiment["id"],
                seed,
                experiment["id"],
                experiment.get("feature_groups"),
                "full_val",
            )


if __name__ == "__main__":
    main()
