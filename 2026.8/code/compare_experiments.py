"""Create paired fold and missing-segment comparisons for two experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"


def main(baseline: str, candidate: str) -> None:
    baseline_folds = pd.read_csv(OUTPUT_DIR / f"{baseline}_fold_metrics.csv")
    candidate_folds = pd.read_csv(OUTPUT_DIR / f"{candidate}_fold_metrics.csv")
    paired = baseline_folds[["fold", "auc"]].merge(
        candidate_folds[["fold", "auc"]],
        on="fold",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    paired["auc_delta"] = paired["auc_candidate"] - paired["auc_baseline"]
    paired.to_csv(
        OUTPUT_DIR / f"{candidate}_vs_{baseline}_fold_comparison.csv", index=False
    )

    baseline_segment_path = OUTPUT_DIR / f"{baseline}_segment_metrics.csv"
    candidate_segment_path = OUTPUT_DIR / f"{candidate}_segment_metrics.csv"
    segments = None
    if baseline_segment_path.exists() and candidate_segment_path.exists():
        baseline_segments = pd.read_csv(baseline_segment_path)
        candidate_segments = pd.read_csv(candidate_segment_path)
        segments = baseline_segments[["segment", "rows", "auc"]].merge(
            candidate_segments[["segment", "rows", "auc"]],
            on="segment",
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        if not (segments["rows_baseline"] == segments["rows_candidate"]).all():
            raise ValueError(
                "Segment row counts differ; experiments are not directly comparable."
            )
        segments["auc_delta"] = segments["auc_candidate"] - segments["auc_baseline"]
        segments.sort_values("auc_delta", ascending=False).to_csv(
            OUTPUT_DIR / f"{candidate}_vs_{baseline}_segment_comparison.csv",
            index=False,
        )

    summary = {
        "baseline": baseline,
        "candidate": candidate,
        "fold_delta_mean": float(paired["auc_delta"].mean()),
        "fold_delta_std": float(paired["auc_delta"].std()),
        "fold_wins": int((paired["auc_delta"] > 0).sum()),
        "fold_count": len(paired),
    }
    with (
        OUTPUT_DIR / f"{candidate}_vs_{baseline}_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
    if segments is not None:
        print(segments.sort_values("auc_delta", ascending=False).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    main(args.baseline, args.candidate)
