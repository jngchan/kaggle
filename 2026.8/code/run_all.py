"""Run selected model experiments and generate the best CV-based submission."""

from __future__ import annotations

import argparse

from feature import SUPPORTED_MISSING_STRATEGIES
from models import SUPPORTED_MODELS, SUPPORTED_PRESETS
from select_and_submit import main as select_and_submit
from train import train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=SUPPORTED_MODELS, default=["lightgbm", "catboost"]
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--preset", choices=SUPPORTED_PRESETS, default="baseline")
    parser.add_argument(
        "--missing-strategy", choices=SUPPORTED_MISSING_STRATEGIES, default="native"
    )
    args = parser.parse_args()

    run_names = []
    for model_name in args.models:
        suffixes = []
        if args.missing_strategy != "native":
            suffixes.append(args.missing_strategy)
        if args.preset != "baseline":
            suffixes.append(args.preset)
        run_name = "_".join([model_name, *suffixes])
        train_model(
            model_name,
            args.folds,
            args.seed,
            args.sample_fraction,
            args.missing_strategy,
            args.preset,
            run_name,
        )
        run_names.append(run_name)
    select_and_submit(run_names)
