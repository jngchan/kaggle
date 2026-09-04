"""Cross-validated training entry point for a single model."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from feature import (
    SUPPORTED_MISSING_STRATEGIES,
    build_features,
    encode_fold,
    transform_fold,
)
from models import (
    SUPPORTED_MODELS,
    SUPPORTED_PRESETS,
    make_model,
    preset_overrides,
    tuning_guide,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


def _write_segment_metrics(
    features: pd.DataFrame, target: pd.Series, prediction: np.ndarray, run_name: str
) -> None:
    missing_columns = [
        column
        for column in features.columns
        if column.endswith("__missing")
        and column.removesuffix("__missing") in features.columns
    ]
    missing_count = features[missing_columns].sum(axis=1)
    rows = []

    def add_segment(segment: str, mask: pd.Series) -> None:
        segment_target = target.loc[mask]
        if len(segment_target) == 0 or segment_target.nunique() < 2:
            return
        rows.append(
            {
                "segment": segment,
                "rows": int(mask.sum()),
                "positive_rate": float(segment_target.mean()),
                "auc": float(roc_auc_score(segment_target, prediction[mask.to_numpy()])),
            }
        )

    for count in range(4):
        add_segment(f"missing_count_{count}", missing_count == count)
    add_segment("missing_count_4plus", missing_count >= 4)
    for column in missing_columns:
        add_segment(column, features[column].astype(bool))
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / f"{run_name}_segment_metrics.csv", index=False
    )


def _fit_fold(model_name, model, x_train, y_train, x_valid, y_valid, cat_columns):
    if model_name == "lightgbm":
        from lightgbm import early_stopping, log_evaluation

        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            callbacks=[early_stopping(150), log_evaluation(200)],
        )
    elif model_name == "xgboost":
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=200)
    else:
        model.fit(
            x_train,
            y_train,
            eval_set=(x_valid, y_valid),
            cat_features=cat_columns,
            early_stopping_rounds=150,
            use_best_model=True,
        )


def train_model(
    model_name: str,
    folds: int = 5,
    seed: int = 2026,
    sample_fraction: float = 1.0,
    missing_strategy: str = "native",
    preset: str = "baseline",
    experiment_name: str | None = None,
) -> dict:
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1].")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if experiment_name:
        run_name = experiment_name
    else:
        suffixes = []
        if missing_strategy != "native":
            suffixes.append(missing_strategy)
        if preset != "baseline":
            suffixes.append(preset)
        run_name = "_".join([model_name, *suffixes])
    bundle = build_features(DATA_DIR / "train.csv", DATA_DIR / "test.csv")
    x = bundle.train
    x_test = bundle.test
    y = bundle.target
    train_ids = bundle.train_ids
    analysis_features = bundle.train

    if sample_fraction < 1.0:
        sample_index, _ = train_test_split(
            np.arange(len(x)),
            train_size=sample_fraction,
            random_state=seed,
            stratify=y,
        )
        x = x.iloc[sample_index].reset_index(drop=True)
        y = y.iloc[sample_index].reset_index(drop=True)
        train_ids = train_ids.iloc[sample_index].reset_index(drop=True)
        analysis_features = analysis_features.iloc[sample_index].reset_index(drop=True)

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(x, y))
    fold_assignment = np.full(len(x), -1, dtype="int8")
    for fold_number, (_, valid_index) in enumerate(splits):
        fold_assignment[valid_index] = fold_number
    if (fold_assignment < 0).any():
        raise RuntimeError("Fold registry contains unassigned rows.")
    sample_tag = "full" if sample_fraction == 1.0 else f"sample_{len(x)}"
    registry_path = OUTPUT_DIR / (
        f"fold_registry_{folds}fold_seed{seed}_{sample_tag}.csv"
    )
    registry = pd.DataFrame({"id": train_ids, "fold": fold_assignment})
    if registry_path.exists():
        existing_registry = pd.read_csv(registry_path)
        if (
            existing_registry.columns.tolist() != registry.columns.tolist()
            or not np.array_equal(existing_registry["id"].to_numpy(), registry["id"].to_numpy())
            or not np.array_equal(
                existing_registry["fold"].to_numpy(), registry["fold"].to_numpy()
            )
        ):
            raise ValueError(f"Existing fold registry does not match: {registry_path}")
    else:
        registry.to_csv(registry_path, index=False)

    oof = np.zeros(len(x), dtype="float32")
    test_prediction = np.zeros(len(x_test), dtype="float64")
    fold_rows = []
    importances = []
    feature_counts = []
    started = time.time()

    for fold, (train_index, valid_index) in enumerate(splits, start=1):
        fold_train, fold_valid, fold_test = transform_fold(
            x.iloc[train_index].reset_index(drop=True),
            x.iloc[valid_index].reset_index(drop=True),
            x_test.reset_index(drop=True),
            strategy=missing_strategy,
            random_state=seed + fold,
        )
        x_train, x_valid, x_test_fold, cat_columns = encode_fold(
            fold_train,
            fold_valid,
            fold_test,
            bundle.categorical_columns,
            model_name,
        )
        feature_counts.append(x_train.shape[1])
        model = make_model(
            model_name, seed + fold, overrides=preset_overrides(model_name, preset)
        )
        _fit_fold(
            model_name,
            model,
            x_train,
            y.iloc[train_index],
            x_valid,
            y.iloc[valid_index],
            cat_columns,
        )
        valid_prediction = model.predict_proba(x_valid)[:, 1]
        oof[valid_index] = valid_prediction
        test_prediction += model.predict_proba(x_test_fold)[:, 1] / folds
        fold_auc = roc_auc_score(y.iloc[valid_index], valid_prediction)
        best_iteration = getattr(model, "best_iteration_", None)
        if best_iteration is None and hasattr(model, "get_best_iteration"):
            best_iteration = model.get_best_iteration()
        fold_rows.append(
            {"fold": fold, "auc": fold_auc, "best_iteration": best_iteration}
        )
        if hasattr(model, "feature_importances_"):
            importances.append(
                pd.DataFrame(
                    {
                        "feature": x_train.columns,
                        "importance": model.feature_importances_,
                        "fold": fold,
                    }
                )
            )
        print(f"{model_name} fold {fold}/{folds}: AUC={fold_auc:.7f}")
        del fold_train, fold_valid, fold_test, x_train, x_valid, x_test_fold, model
        gc.collect()

    overall_auc = roc_auc_score(y, oof)
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(OUTPUT_DIR / f"{run_name}_fold_metrics.csv", index=False)
    np.save(OUTPUT_DIR / f"{run_name}_oof.npy", oof)
    np.save(OUTPUT_DIR / f"{run_name}_test.npy", test_prediction.astype("float32"))
    np.save(OUTPUT_DIR / f"{run_name}_oof_target.npy", y.to_numpy(dtype="int8"))
    np.save(OUTPUT_DIR / f"{run_name}_oof_ids.npy", train_ids.to_numpy())
    np.save(OUTPUT_DIR / f"{run_name}_oof_folds.npy", fold_assignment)
    _write_segment_metrics(analysis_features, y, oof, run_name)

    if importances:
        importance_frame = pd.concat(importances, ignore_index=True)
        summary = (
            importance_frame.groupby("feature")["importance"]
            .agg(mean="mean", std="std")
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        summary.to_csv(OUTPUT_DIR / f"{run_name}_feature_importance.csv", index=False)

    result = {
        "model": model_name,
        "experiment_name": run_name,
        "oof_auc": overall_auc,
        "fold_auc_mean": float(fold_frame["auc"].mean()),
        "fold_auc_std": float(fold_frame["auc"].std()),
        "elapsed_minutes": (time.time() - started) / 60,
        "folds": folds,
        "seed": seed,
        "sample_fraction": sample_fraction,
        "train_rows": len(x),
        "feature_count_min": min(feature_counts),
        "feature_count_max": max(feature_counts),
        "missing_strategy": missing_strategy,
        "preset": preset,
        "fold_registry": registry_path.name,
    }
    with (OUTPUT_DIR / f"{run_name}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=True, indent=2)
    with (OUTPUT_DIR / "tuning_guide.json").open("w", encoding="utf-8") as file:
        json.dump(tuning_guide(), file, ensure_ascii=True, indent=2)
    feature_report = {
        "full_train_rows": len(bundle.train),
        "test_rows": len(bundle.test),
        "positive_rate": float(bundle.target.mean()),
        "engineered_feature_count": bundle.train.shape[1],
        "features": bundle.train.columns.tolist(),
        "missing_rates": {
            column: float(rate)
            for column, rate in bundle.train.isna().mean().items()
            if rate > 0
        },
    }
    with (OUTPUT_DIR / "feature_report.json").open("w", encoding="utf-8") as file:
        json.dump(feature_report, file, ensure_ascii=True, indent=2)
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="lightgbm")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--preset", choices=SUPPORTED_PRESETS, default="baseline")
    parser.add_argument(
        "--missing-strategy", choices=SUPPORTED_MISSING_STRATEGIES, default="native"
    )
    parser.add_argument("--experiment-name")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Use e.g. 0.2 for a quick server smoke test.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        args.model,
        args.folds,
        args.seed,
        args.sample_fraction,
        args.missing_strategy,
        args.preset,
        args.experiment_name,
    )
