"""Kaggle V9: exact-value target encoding and generator-lattice features.

This standalone script uses only competition-provided train.csv, test.csv and
sample_submission.csv. It writes diagnostics to /kaggle/working/output and the
selected prediction to /kaggle/working/submission.csv.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5
SMOOTHING = 10.0
TARGET = "addicted_label"
ID_COLUMN = "id"
OUTPUT_DIR = Path("/kaggle/working/output")
SUBMISSION_PATH = Path("/kaggle/working/submission.csv")

CATEGORICAL_COLUMNS = ["gender", "stress_level", "academic_work_impact"]
NUMERIC_COLUMNS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]
ORIGINAL_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
FRACTIONAL_COLUMNS = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "weekend_screen_time",
]
RAW_MODEL_NAMES = [
    "xgb_te",
    "xgb_lattice_seed42",
    "xgb_lattice_seed2024",
    "xgb_lattice_seed7",
    "xgb_lattice_low_lr",
    "lgb_te",
]
STACK_MODEL_NAMES = [
    "xgb_te",
    "xgb_lattice_seed_avg",
    "xgb_lattice_low_lr",
    "lgb_te",
]
LATTICE_SEEDS = [42, 2024, 7]
MIN_STACK_GAIN = 0.00005
CONSERVATIVE_STACK_WEIGHT = 0.50
MIN_STACK_POSITIVE_FOLDS = 4


def locate_input_files() -> tuple[Path, Path, Path]:
    input_root = Path("/kaggle/input")
    for train_path in sorted(input_root.rglob("train.csv")):
        test_path = train_path.parent / "test.csv"
        sample_path = train_path.parent / "sample_submission.csv"
        if not (test_path.exists() and sample_path.exists()):
            continue
        train_columns = set(pd.read_csv(train_path, nrows=0).columns)
        test_columns = set(pd.read_csv(test_path, nrows=0).columns)
        required_train = {ID_COLUMN, TARGET, *ORIGINAL_COLUMNS}
        if required_train.issubset(train_columns) and (
            required_train - {TARGET}
        ).issubset(test_columns):
            print(f"Using competition data from: {train_path.parent}", flush=True)
            return train_path, test_path, sample_path
    raise FileNotFoundError("Competition CSV files were not found under /kaggle/input.")


def make_xgb_classifier(seed: int, device: str) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=4000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        tree_method="hist",
        device=device,
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=seed,
        n_jobs=-1,
    )



def make_low_lr_xgb_classifier(seed: int, device: str) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=12000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        tree_method="hist",
        device=device,
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=300,
        random_state=seed,
        n_jobs=-1,
    )

def make_xgb_imputer(seed: int, device: str) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        tree_method="hist",
        device=device,
        enable_categorical=True,
        random_state=seed,
        n_jobs=-1,
    )


def transductive_impute(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit feature-only regressors on competition train+test and retain raw NaNs."""
    train_size = len(raw_train)
    full = pd.concat(
        [raw_train[ORIGINAL_COLUMNS], raw_test[ORIGINAL_COLUMNS]],
        ignore_index=True,
    )
    predictors = full.copy()
    for column in CATEGORICAL_COLUMNS:
        predictors[column] = predictors[column].astype("category")

    imputed = full[NUMERIC_COLUMNS].copy()
    rows = []
    for target_column in NUMERIC_COLUMNS:
        started = time.time()
        observed = predictors[target_column].notna().to_numpy()
        feature_columns = [
            column for column in ORIGINAL_COLUMNS if column != target_column
        ]
        model = make_xgb_imputer(SEED, device)
        model.fit(
            predictors.loc[observed, feature_columns],
            predictors.loc[observed, target_column],
            verbose=False,
        )
        missing = ~observed
        if missing.any():
            imputed.loc[missing, target_column] = model.predict(
                predictors.loc[missing, feature_columns]
            )
        rows.append(
            {
                "feature": target_column,
                "observed_rows": int(observed.sum()),
                "imputed_rows": int(missing.sum()),
                "missing_rate": float(missing.mean()),
                "elapsed_seconds": time.time() - started,
            }
        )
        print(
            f"imputed {target_column}: missing={missing.mean():.4f}, "
            f"seconds={rows[-1]['elapsed_seconds']:.1f}",
            flush=True,
        )
        del model
        gc.collect()

    return (
        imputed.iloc[:train_size].reset_index(drop=True),
        imputed.iloc[train_size:].reset_index(drop=True),
        pd.DataFrame(rows),
    )


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def build_augmented_features(
    imputed: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """Add generator composition features while preserving original NaN columns."""
    result = imputed.copy()
    daily = result["daily_screen_time_hours"]
    social = result["social_media_hours"]
    gaming = result["gaming_hours"]
    work = result["work_study_hours"]
    sleep = result["sleep_hours"]
    weekend = result["weekend_screen_time"]
    notifications = result["notifications_per_day"]
    opens = result["app_opens_per_day"]
    parts = social + gaming + work

    result["resid"] = daily - parts
    result["leisure"] = daily - work
    result["social_frac"] = safe_divide(social, daily)
    result["gaming_frac"] = safe_divide(gaming, daily)
    result["work_frac"] = safe_divide(work, daily)
    result["leisure_frac"] = safe_divide(daily - work, daily)
    result["resid_frac"] = safe_divide(daily - parts, daily)
    result["weekend_ratio"] = safe_divide(weekend, daily)
    result["week_total"] = 5.0 * daily + 2.0 * weekend
    result["awake_screen_frac"] = safe_divide(daily, 24.0 - sleep)
    result["free_time"] = 24.0 - sleep - daily - work
    result["notifications_per_open"] = safe_divide(notifications, opens)
    result["minutes_per_open"] = safe_divide(daily * 60.0, opens)

    for column in CATEGORICAL_COLUMNS:
        result[column] = raw[column].astype("category").array
    for column in ORIGINAL_COLUMNS:
        result[f"na_{column}"] = raw[column].isna().astype("int8").to_numpy()
    for column in NUMERIC_COLUMNS:
        result[f"rawnan_{column}"] = raw[column].to_numpy()

    numeric = result.select_dtypes(exclude=["category", "object"]).columns
    result[numeric] = result[numeric].astype("float32")
    return result.replace([np.inf, -np.inf], np.nan)


def lattice_features(raw: pd.DataFrame) -> pd.DataFrame:
    result = {}
    for column in FRACTIONAL_COLUMNS:
        values = raw[column].to_numpy(dtype="float64")
        result[f"frac_{column}"] = values - np.floor(values)
        result[f"first_decimal_{column}"] = np.floor(values * 10.0) % 10.0
    return pd.DataFrame(result, dtype="float32")


def exact_levels(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: raw[column].astype(str).to_numpy()
            for column in ORIGINAL_COLUMNS
        }
    )


def fit_encoding_maps(
    levels: pd.DataFrame,
    target: np.ndarray,
) -> tuple[dict[str, tuple[pd.Series, pd.Series]], float]:
    global_mean = float(np.mean(target))
    mappings = {}
    for column in ORIGINAL_COLUMNS:
        grouped = (
            pd.DataFrame({"level": levels[column].to_numpy(), "target": target})
            .groupby("level", sort=False)["target"]
            .agg(["count", "mean"])
        )
        target_map = (
            (
                grouped["count"] * grouped["mean"]
                + SMOOTHING * global_mean
            )
            / (grouped["count"] + SMOOTHING)
        ).astype("float32")
        frequency_map = grouped["count"].astype("float32")
        mappings[column] = (target_map, frequency_map)
    return mappings, global_mean


def apply_encoding_maps(
    levels: pd.DataFrame,
    mappings: dict[str, tuple[pd.Series, pd.Series]],
    global_mean: float,
) -> pd.DataFrame:
    output = {}
    for column in ORIGINAL_COLUMNS:
        target_map, frequency_map = mappings[column]
        output[f"te_{column}"] = (
            levels[column].map(target_map).fillna(global_mean).to_numpy("float32")
        )
        output[f"fq_{column}"] = (
            levels[column].map(frequency_map).fillna(0.0).to_numpy("float32")
        )
    return pd.DataFrame(output)


def build_nested_encodings(
    train_levels: pd.DataFrame,
    test_levels: pd.DataFrame,
    target: np.ndarray,
    fit_index: np.ndarray,
    valid_index: np.ndarray,
    outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_levels = train_levels.iloc[fit_index].reset_index(drop=True)
    outer_target = target[fit_index]
    column_order = [
        *(f"te_{column}" for column in ORIGINAL_COLUMNS),
        *(f"fq_{column}" for column in ORIGINAL_COLUMNS),
    ]
    fit_values = np.zeros(
        (len(fit_index), len(column_order)),
        dtype="float32",
    )
    encoding_coverage = np.zeros(len(fit_index), dtype="int8")
    inner = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=0,
    )
    for inner_fit, inner_valid in inner.split(outer_levels, outer_target):
        mappings, global_mean = fit_encoding_maps(
            outer_levels.iloc[inner_fit],
            outer_target[inner_fit],
        )
        encoded = apply_encoding_maps(
            outer_levels.iloc[inner_valid].reset_index(drop=True),
            mappings,
            global_mean,
        )
        fit_values[inner_valid] = encoded[column_order].to_numpy()
        encoding_coverage[inner_valid] += 1

    if not np.all(encoding_coverage == 1):
        raise RuntimeError("Inner target encodings are not exactly cross-fitted.")

    mappings, global_mean = fit_encoding_maps(outer_levels, outer_target)
    valid_encoded = apply_encoding_maps(
        train_levels.iloc[valid_index].reset_index(drop=True),
        mappings,
        global_mean,
    )[column_order]
    test_encoded = apply_encoding_maps(
        test_levels.reset_index(drop=True),
        mappings,
        global_mean,
    )[column_order]
    return (
        pd.DataFrame(fit_values, columns=column_order),
        valid_encoded.reset_index(drop=True),
        test_encoded.reset_index(drop=True),
    )


def logit(prediction: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(prediction, dtype="float64"), 1e-15, 1 - 1e-15)
    return np.clip(np.log(clipped / (1.0 - clipped)), -30.0, 30.0)


def segment_metrics(
    raw: pd.DataFrame,
    target: np.ndarray,
    prediction: np.ndarray,
    model_name: str,
) -> list[dict]:
    missing_count = raw[ORIGINAL_COLUMNS].isna().sum(axis=1).to_numpy()
    rows = []
    for name, mask in [
        ("missing_count_0", missing_count == 0),
        ("missing_count_1", missing_count == 1),
        ("missing_count_2", missing_count == 2),
        ("missing_count_3", missing_count == 3),
        ("missing_count_4plus", missing_count >= 4),
    ]:
        if mask.sum() and np.unique(target[mask]).size == 2:
            rows.append(
                {
                    "model": model_name,
                    "segment": name,
                    "rows": int(mask.sum()),
                    "auc": float(roc_auc_score(target[mask], prediction[mask])),
                }
            )
    return rows


def main() -> None:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path, test_path, sample_path = locate_input_files()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)
    target = train[TARGET].to_numpy(dtype="int8")
    train_ids = train[ID_COLUMN].copy()
    test_ids = test[ID_COLUMN].copy()

    # V9 logs proved that the apparent CUDA probe silently fell back to CPU.
    device = "cpu"
    print(
        f"train={len(train):,}, test={len(test):,}, device={device}, "
        f"outer={N_FOLDS}, inner={INNER_FOLDS}",
        flush=True,
    )

    accounting_columns = [
        "daily_screen_time_hours",
        "social_media_hours",
        "gaming_hours",
        "work_study_hours",
    ]
    accounting = train.dropna(subset=accounting_columns)
    accounting_residual = (
        accounting["daily_screen_time_hours"]
        - accounting[
            ["social_media_hours", "gaming_hours", "work_study_hours"]
        ].sum(axis=1)
    )
    generator_audit = {
        "complete_accounting_rows": int(len(accounting)),
        "minimum_accounting_residual": float(accounting_residual.min()),
        "negative_residual_violations": int((accounting_residual < -1e-9).sum()),
    }
    with (OUTPUT_DIR / "v10_generator_audit.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(generator_audit, file, indent=2)

    train_imputed, test_imputed, imputation_report = transductive_impute(
        train,
        test,
        device,
    )
    if not np.isfinite(train_imputed.to_numpy()).all() or not np.isfinite(
        test_imputed.to_numpy()
    ).all():
        raise ValueError("Feature-only imputation produced non-finite values.")
    imputation_report.to_csv(
        OUTPUT_DIR / "v10_imputation_report.csv", index=False
    )

    train_augmented = build_augmented_features(train_imputed, train)
    test_augmented = build_augmented_features(test_imputed, test)
    train_lattice = lattice_features(train)
    test_lattice = lattice_features(test)
    train_levels = exact_levels(train)
    test_levels = exact_levels(test)

    cardinality = []
    for column in ORIGINAL_COLUMNS:
        level_count = int(train_levels[column].nunique())
        cardinality.append(
            {
                "feature": column,
                "levels": level_count,
                "rows_per_level": len(train) / level_count,
                "smoothing": SMOOTHING,
            }
        )
    pd.DataFrame(cardinality).to_csv(
        OUTPUT_DIR / "v10_encoding_cardinality.csv", index=False
    )

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )
    folds = list(splitter.split(train_augmented, target))
    fold_registry = np.full(len(train), -1, dtype="int8")
    oof = {
        name: np.zeros(len(train), dtype="float32")
        for name in RAW_MODEL_NAMES
    }
    test_predictions = {
        name: np.zeros(len(test), dtype="float64")
        for name in RAW_MODEL_NAMES
    }
    fold_rows = []
    importance_rows = []

    for fold, (fit_index, valid_index) in enumerate(folds, start=1):
        if np.intersect1d(fit_index, valid_index).size:
            raise RuntimeError("Outer training and validation rows overlap.")
        fold_registry[valid_index] = fold - 1
        print(
            f"\n===== fold {fold}/{N_FOLDS}: nested exact-value encoding =====",
            flush=True,
        )
        encoded_fit, encoded_valid, encoded_test = build_nested_encodings(
            train_levels,
            test_levels,
            target,
            fit_index,
            valid_index,
            fold,
        )

        augmented_fit = train_augmented.iloc[fit_index].reset_index(drop=True)
        augmented_valid = train_augmented.iloc[valid_index].reset_index(drop=True)
        augmented_test = test_augmented.reset_index(drop=True)
        te_fit = pd.concat([augmented_fit, encoded_fit], axis=1)
        te_valid = pd.concat([augmented_valid, encoded_valid], axis=1)
        te_test = pd.concat([augmented_test, encoded_test], axis=1)
        lattice_fit = pd.concat(
            [te_fit, train_lattice.iloc[fit_index].reset_index(drop=True)],
            axis=1,
        )
        lattice_valid = pd.concat(
            [te_valid, train_lattice.iloc[valid_index].reset_index(drop=True)],
            axis=1,
        )
        lattice_test = pd.concat(
            [te_test, test_lattice.reset_index(drop=True)],
            axis=1,
        )
        fit_target = target[fit_index]
        valid_target = target[valid_index]

        print(f"===== fold {fold}/{N_FOLDS}: xgb_te =====", flush=True)
        te_model = make_xgb_classifier(SEED, device)
        te_model.fit(
            te_fit,
            fit_target,
            eval_set=[(te_valid, valid_target)],
            verbose=500,
        )
        te_valid_prediction = te_model.predict_proba(te_valid)[:, 1]
        oof["xgb_te"][valid_index] = te_valid_prediction
        test_predictions["xgb_te"] += (
            te_model.predict_proba(te_test)[:, 1] / N_FOLDS
        )
        te_auc = float(roc_auc_score(valid_target, te_valid_prediction))
        fold_rows.append(
            {
                "fold": fold,
                "model": "xgb_te",
                "auc": te_auc,
                "best_iteration": int(te_model.best_iteration),
                "seed": SEED,
            }
        )
        importance_rows.extend(
            {
                "fold": fold,
                "model": "xgb_te",
                "feature": feature,
                "importance": importance,
            }
            for feature, importance in zip(
                te_fit.columns, te_model.feature_importances_
            )
        )
        print(
            f"xgb_te: auc={te_auc:.7f}, best_iteration={te_model.best_iteration}",
            flush=True,
        )
        del te_model
        gc.collect()

        for lattice_seed in LATTICE_SEEDS:
            name = f"xgb_lattice_seed{lattice_seed}"
            print(f"===== fold {fold}/{N_FOLDS}: {name} =====", flush=True)
            lattice_model = make_xgb_classifier(lattice_seed, device)
            lattice_model.fit(
                lattice_fit,
                fit_target,
                eval_set=[(lattice_valid, valid_target)],
                verbose=500,
            )
            valid_prediction = lattice_model.predict_proba(lattice_valid)[:, 1]
            oof[name][valid_index] = valid_prediction
            test_predictions[name] += (
                lattice_model.predict_proba(lattice_test)[:, 1] / N_FOLDS
            )
            auc = float(roc_auc_score(valid_target, valid_prediction))
            fold_rows.append(
                {
                    "fold": fold,
                    "model": name,
                    "auc": auc,
                    "best_iteration": int(lattice_model.best_iteration),
                    "seed": lattice_seed,
                }
            )
            importance_rows.extend(
                {
                    "fold": fold,
                    "model": name,
                    "feature": feature,
                    "importance": importance,
                }
                for feature, importance in zip(
                    lattice_fit.columns,
                    lattice_model.feature_importances_,
                )
            )
            print(
                f"{name}: auc={auc:.7f}, "
                f"best_iteration={lattice_model.best_iteration}",
                flush=True,
            )
            del lattice_model
            gc.collect()

        print(
            f"===== fold {fold}/{N_FOLDS}: xgb_lattice_low_lr =====",
            flush=True,
        )
        low_lr_model = make_low_lr_xgb_classifier(SEED, device)
        low_lr_model.fit(
            lattice_fit,
            fit_target,
            eval_set=[(lattice_valid, valid_target)],
            verbose=500,
        )
        low_lr_valid_prediction = low_lr_model.predict_proba(lattice_valid)[:, 1]
        oof["xgb_lattice_low_lr"][valid_index] = low_lr_valid_prediction
        test_predictions["xgb_lattice_low_lr"] += (
            low_lr_model.predict_proba(lattice_test)[:, 1] / N_FOLDS
        )
        low_lr_auc = float(
            roc_auc_score(valid_target, low_lr_valid_prediction)
        )
        fold_rows.append(
            {
                "fold": fold,
                "model": "xgb_lattice_low_lr",
                "auc": low_lr_auc,
                "best_iteration": int(low_lr_model.best_iteration),
                "seed": SEED,
            }
        )
        importance_rows.extend(
            {
                "fold": fold,
                "model": "xgb_lattice_low_lr",
                "feature": feature,
                "importance": importance,
            }
            for feature, importance in zip(
                lattice_fit.columns,
                low_lr_model.feature_importances_,
            )
        )
        print(
            f"xgb_lattice_low_lr: auc={low_lr_auc:.7f}, "
            f"best_iteration={low_lr_model.best_iteration}",
            flush=True,
        )
        del low_lr_model
        gc.collect()

        print(f"===== fold {fold}/{N_FOLDS}: lgb_te =====", flush=True)
        lgb_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=4000,
            learning_rate=0.05,
            num_leaves=63,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            min_child_samples=100,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
        lgb_model.fit(
            te_fit,
            fit_target,
            eval_set=[(te_valid, valid_target)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        lgb_valid_prediction = lgb_model.predict_proba(te_valid)[:, 1]
        oof["lgb_te"][valid_index] = lgb_valid_prediction
        test_predictions["lgb_te"] += (
            lgb_model.predict_proba(te_test)[:, 1] / N_FOLDS
        )
        lgb_auc = float(roc_auc_score(valid_target, lgb_valid_prediction))
        fold_rows.append(
            {
                "fold": fold,
                "model": "lgb_te",
                "auc": lgb_auc,
                "best_iteration": int(lgb_model.best_iteration_),
                "seed": SEED,
            }
        )
        importance_rows.extend(
            {
                "fold": fold,
                "model": "lgb_te",
                "feature": feature,
                "importance": importance,
            }
            for feature, importance in zip(
                te_fit.columns, lgb_model.feature_importances_
            )
        )
        print(
            f"lgb_te: auc={lgb_auc:.7f}, "
            f"best_iteration={lgb_model.best_iteration_}",
            flush=True,
        )

        del (
            encoded_fit,
            encoded_valid,
            encoded_test,
            augmented_fit,
            augmented_valid,
            augmented_test,
            te_fit,
            te_valid,
            te_test,
            lattice_fit,
            lattice_valid,
            lattice_test,
            lgb_model,
        )
        gc.collect()

    oof["xgb_lattice_seed_avg"] = np.mean(
        np.column_stack(
            [oof[f"xgb_lattice_seed{seed}"] for seed in LATTICE_SEEDS]
        ),
        axis=1,
    ).astype("float32")
    test_predictions["xgb_lattice_seed_avg"] = np.mean(
        np.column_stack(
            [
                test_predictions[f"xgb_lattice_seed{seed}"]
                for seed in LATTICE_SEEDS
            ]
        ),
        axis=1,
    )

    stack_model_auc = {
        name: float(roc_auc_score(target, oof[name]))
        for name in STACK_MODEL_NAMES
    }
    raw_model_auc = {
        name: float(roc_auc_score(target, oof[name]))
        for name in RAW_MODEL_NAMES
    }
    best_single_name = max(stack_model_auc, key=stack_model_auc.get)
    best_single_auc = stack_model_auc[best_single_name]

    oof_logits = np.column_stack(
        [logit(oof[name]) for name in STACK_MODEL_NAMES]
    )
    test_logits = np.column_stack(
        [logit(test_predictions[name]) for name in STACK_MODEL_NAMES]
    )
    stack_oof = np.zeros(len(train), dtype="float64")
    stack_fold_rows = []
    for fold, (fit_index, valid_index) in enumerate(folds, start=1):
        meta = LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
        )
        meta.fit(oof_logits[fit_index], target[fit_index])
        stack_oof[valid_index] = meta.predict_proba(
            oof_logits[valid_index]
        )[:, 1]
        stack_auc = float(
            roc_auc_score(target[valid_index], stack_oof[valid_index])
        )
        best_fold_auc = max(
            roc_auc_score(target[valid_index], oof[name][valid_index])
            for name in STACK_MODEL_NAMES
        )
        stack_fold_rows.append(
            {
                "fold": fold,
                "model": "logit_stack",
                "auc": stack_auc,
                "best_single_fold_auc": best_fold_auc,
                "delta_vs_best_single_fold": stack_auc - best_fold_auc,
            }
        )

    stack_auc = float(roc_auc_score(target, stack_oof))
    stack_positive_folds = int(
        sum(row["delta_vs_best_single_fold"] > 0 for row in stack_fold_rows)
    )
    stack_accepted = bool(
        stack_auc >= best_single_auc + MIN_STACK_GAIN
        and stack_positive_folds >= MIN_STACK_POSITIVE_FOLDS
    )
    final_meta = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
    ).fit(oof_logits, target)
    stack_test = final_meta.predict_proba(test_logits)[:, 1]

    seed_average_oof = oof["xgb_lattice_seed_avg"]
    seed_average_test = test_predictions["xgb_lattice_seed_avg"]
    conservative_oof = (
        CONSERVATIVE_STACK_WEIGHT * stack_oof
        + (1.0 - CONSERVATIVE_STACK_WEIGHT) * seed_average_oof
    )
    conservative_test = (
        CONSERVATIVE_STACK_WEIGHT * stack_test
        + (1.0 - CONSERVATIVE_STACK_WEIGHT) * seed_average_test
    )
    conservative_auc = float(roc_auc_score(target, conservative_oof))

    if stack_accepted:
        selected_name = "logit_stack"
        selected_oof = stack_oof
        selected_test = stack_test
        selection_reason = "cross-fitted stack passed fixed stability guardrails"
    else:
        candidate_auc = {
            **stack_model_auc,
            "conservative_blend": conservative_auc,
        }
        selected_name = max(candidate_auc, key=candidate_auc.get)
        if selected_name == "conservative_blend":
            selected_oof = conservative_oof
            selected_test = conservative_test
        else:
            selected_oof = oof[selected_name]
            selected_test = test_predictions[selected_name]
        selection_reason = "stack rejected; selected strongest complete-OOF candidate"

    candidate_predictions = {
        "selected": selected_test,
        "aggressive_stack": stack_test,
        "conservative": conservative_test,
        "seed_average": seed_average_test,
        "best_single": test_predictions[best_single_name],
    }
    for name, prediction in candidate_predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"{name} contains non-finite test predictions.")

    if sample_submission.columns.tolist() != [ID_COLUMN, TARGET]:
        raise ValueError("Unexpected sample_submission.csv columns.")
    if not sample_submission[ID_COLUMN].equals(test_ids):
        raise ValueError("sample submission IDs do not align with test.csv.")

    output_names = {
        "selected": SUBMISSION_PATH,
        "aggressive_stack": OUTPUT_DIR / "submission_v10_aggressive_stack.csv",
        "conservative": OUTPUT_DIR / "submission_v10_conservative.csv",
        "seed_average": OUTPUT_DIR / "submission_v10_seed_average.csv",
        "best_single": OUTPUT_DIR / "submission_v10_best_single.csv",
    }
    for name, path in output_names.items():
        submission = sample_submission.copy()
        submission[TARGET] = np.clip(candidate_predictions[name], 0.0, 1.0)
        submission.to_csv(path, index=False)

    fold_frame = pd.concat(
        [pd.DataFrame(fold_rows), pd.DataFrame(stack_fold_rows)],
        ignore_index=True,
        sort=False,
    )
    fold_frame.to_csv(OUTPUT_DIR / "v10_fold_metrics.csv", index=False)

    model_rows = [
        {"model": name, "oof_auc": auc, "kind": "raw"}
        for name, auc in raw_model_auc.items()
    ]
    model_rows += [
        {
            "model": "xgb_lattice_seed_avg",
            "oof_auc": stack_model_auc["xgb_lattice_seed_avg"],
            "kind": "aggregate",
        },
        {"model": "logit_stack", "oof_auc": stack_auc, "kind": "stack"},
        {
            "model": "conservative_blend",
            "oof_auc": conservative_auc,
            "kind": "risk_candidate",
        },
    ]
    pd.DataFrame(model_rows).sort_values("oof_auc", ascending=False).to_csv(
        OUTPUT_DIR / "v10_model_metrics.csv", index=False
    )

    oof_frame = pd.DataFrame(
        {
            ID_COLUMN: train_ids,
            "fold": fold_registry,
            TARGET: target,
            **{
                f"{name}_prediction": oof[name]
                for name in STACK_MODEL_NAMES
            },
            "logit_stack_prediction": stack_oof,
            "conservative_prediction": conservative_oof,
            "selected_prediction": selected_oof,
        }
    )
    test_frame = pd.DataFrame(
        {
            ID_COLUMN: test_ids,
            **{
                f"{name}_prediction": test_predictions[name]
                for name in STACK_MODEL_NAMES
            },
            "logit_stack_prediction": stack_test,
            "conservative_prediction": conservative_test,
            "selected_prediction": selected_test,
        }
    )
    oof_frame.to_csv(OUTPUT_DIR / "v10_oof.csv", index=False)
    test_frame.to_csv(OUTPUT_DIR / "v10_test_predictions.csv", index=False)

    pd.DataFrame(
        {
            "model": STACK_MODEL_NAMES,
            "coefficient": final_meta.coef_[0],
            "single_oof_auc": [
                stack_model_auc[name] for name in STACK_MODEL_NAMES
            ],
        }
    ).sort_values("coefficient", ascending=False).to_csv(
        OUTPUT_DIR / "v10_stack_coefficients.csv", index=False
    )

    importance = pd.DataFrame(importance_rows)
    importance.groupby(["model", "feature"], as_index=False)["importance"].agg(
        mean="mean",
        std="std",
    ).sort_values(["model", "mean"], ascending=[True, False]).to_csv(
        OUTPUT_DIR / "v10_feature_importance.csv", index=False
    )

    prediction_view = pd.DataFrame(
        {name: oof[name] for name in STACK_MODEL_NAMES}
    )
    prediction_view.corr().to_csv(
        OUTPUT_DIR / "v10_prediction_correlation.csv"
    )
    prediction_view.rsub(target, axis=0).corr().to_csv(
        OUTPUT_DIR / "v10_residual_correlation.csv"
    )

    segments = []
    for name in STACK_MODEL_NAMES:
        segments += segment_metrics(train, target, oof[name], name)
    segments += segment_metrics(train, target, stack_oof, "logit_stack")
    segments += segment_metrics(
        train, target, conservative_oof, "conservative_blend"
    )
    segments += segment_metrics(train, target, selected_oof, "selected")
    pd.DataFrame(segments).to_csv(
        OUTPUT_DIR / "v10_segment_metrics.csv", index=False
    )

    metrics = {
        "version": "v10_final",
        "competition_data_only": True,
        "preprocessing_audit": {
            "imputation_uses_target": False,
            "imputation_scope": "competition_train_plus_test_features",
            "target_encoding_outer_fold_isolated": True,
            "target_encoding_inner_cross_fitted": True,
            "public_leaderboard_used_for_selection": False,
        },
        "device": device,
        "raw_model_oof_auc": raw_model_auc,
        "stack_model_oof_auc": stack_model_auc,
        "best_single_model": best_single_name,
        "best_single_oof_auc": best_single_auc,
        "seed_average_oof_auc": stack_model_auc["xgb_lattice_seed_avg"],
        "low_lr_gain_vs_seed42": (
            raw_model_auc["xgb_lattice_low_lr"]
            - raw_model_auc["xgb_lattice_seed42"]
        ),
        "seed_average_gain_vs_seed42": (
            stack_model_auc["xgb_lattice_seed_avg"]
            - raw_model_auc["xgb_lattice_seed42"]
        ),
        "logit_stack_oof_auc": stack_auc,
        "logit_stack_gain_vs_best_single": stack_auc - best_single_auc,
        "logit_stack_positive_folds": stack_positive_folds,
        "stack_accepted": stack_accepted,
        "conservative_oof_auc": conservative_auc,
        "selected_model": selected_name,
        "selected_oof_auc": float(roc_auc_score(target, selected_oof)),
        "selected_test_mean": float(np.mean(selected_test)),
        "train_target_mean": float(np.mean(target)),
        "selection_reason": selection_reason,
        "risk_candidates": {
            name: str(path) for name, path in output_names.items()
        },
        "outer_folds": N_FOLDS,
        "inner_folds": INNER_FOLDS,
        "target_encoding_smoothing": SMOOTHING,
        "lattice_seeds": LATTICE_SEEDS,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "submission_path": str(SUBMISSION_PATH),
    }
    with (OUTPUT_DIR / "v10_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print("\nV10 complete", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
