"""Compare completed experiments, test blends, and create submission.csv."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"


def rank_average(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def main(models: list[str]):
    target = None
    oof_ids = None
    oof_folds = None
    rows = []
    predictions = {}

    for name in models:
        oof_path = OUTPUT_DIR / f"{name}_oof.npy"
        test_path = OUTPUT_DIR / f"{name}_test.npy"
        target_path = OUTPUT_DIR / f"{name}_oof_target.npy"
        ids_path = OUTPUT_DIR / f"{name}_oof_ids.npy"
        folds_path = OUTPUT_DIR / f"{name}_oof_folds.npy"
        if not all(path.exists() for path in (oof_path, test_path, target_path, ids_path)):
            print(f"Skipping {name}: training outputs are incomplete.")
            continue
        model_target = np.load(target_path)
        model_ids = np.load(ids_path)
        model_folds = np.load(folds_path) if folds_path.exists() else None
        if target is not None and not np.array_equal(target, model_target):
            raise ValueError("OOF targets differ. Use identical folds/sample settings for blending.")
        if oof_ids is not None and not np.array_equal(oof_ids, model_ids):
            raise ValueError("OOF row IDs differ. Use identical seed/sample settings for blending.")
        if model_folds is not None:
            if oof_folds is not None and not np.array_equal(oof_folds, model_folds):
                raise ValueError("OOF folds differ. Only blend experiments using one registry.")
            oof_folds = model_folds
        target = model_target
        oof_ids = model_ids
        oof = np.load(oof_path)
        test = np.load(test_path)
        score = roc_auc_score(target, oof)
        predictions[name] = (oof, test)
        rows.append({"candidate": name, "type": "single", "oof_auc": score})

    if not predictions:
        raise FileNotFoundError("No completed model outputs found in output/.")

    if len(predictions) >= 2:
        names = list(predictions)
        blend_oof = np.mean([rank_average(predictions[n][0]) for n in names], axis=0)
        blend_test = np.mean([rank_average(predictions[n][1]) for n in names], axis=0)
        blend_name = "rank_blend_" + "_".join(names)
        predictions[blend_name] = (blend_oof, blend_test)
        rows.append(
            {
                "candidate": blend_name,
                "type": "equal_rank_blend",
                "oof_auc": roc_auc_score(target, blend_oof),
            }
        )
        for first, second in combinations(names, 2):
            best = None
            for first_weight in np.arange(0.05, 1.0, 0.05):
                blend_oof = (
                    first_weight * predictions[first][0]
                    + (1.0 - first_weight) * predictions[second][0]
                )
                score = roc_auc_score(target, blend_oof)
                if best is None or score > best[0]:
                    best = (score, float(first_weight), blend_oof)
            score, first_weight, blend_oof = best
            second_weight = 1.0 - first_weight
            blend_test = (
                first_weight * predictions[first][1]
                + second_weight * predictions[second][1]
            )
            blend_name = (
                f"weighted_{first_weight:.2f}_{first}_{second_weight:.2f}_{second}"
            )
            predictions[blend_name] = (blend_oof, blend_test)
            rows.append(
                {
                    "candidate": blend_name,
                    "type": "oof_weighted_blend",
                    "oof_auc": score,
                }
            )

    comparison = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    winner = comparison.iloc[0]["candidate"]
    test_prediction = np.clip(predictions[winner][1], 0.0, 1.0)
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    test_ids = pd.read_csv(ROOT / "data" / "test.csv", usecols=["id"])["id"]
    if len(sample) != len(test_prediction):
        raise ValueError("Prediction length does not match sample_submission.csv.")
    if sample.columns.tolist() != ["id", "addicted_label"]:
        raise ValueError("sample_submission.csv has unexpected columns or column order.")
    if not sample["id"].equals(test_ids):
        raise ValueError("Test IDs and sample submission IDs are not aligned.")
    sample["addicted_label"] = test_prediction
    sample.to_csv(ROOT / "submission.csv", index=False)

    selection = {
        "selected_candidate": winner,
        "selected_oof_auc": float(comparison.iloc[0]["oof_auc"]),
        "submission_rows": len(sample),
        "prediction_min": float(test_prediction.min()),
        "prediction_max": float(test_prediction.max()),
    }
    with (OUTPUT_DIR / "selection.json").open("w", encoding="utf-8") as file:
        json.dump(selection, file, indent=2)
    print(comparison.to_string(index=False))
    print(f"Created {ROOT / 'submission.csv'} using {winner}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["lightgbm", "xgboost", "catboost"]
    )
    main(parser.parse_args().models)
