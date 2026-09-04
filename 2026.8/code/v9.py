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
from catboost import CatBoostClassifier, CatBoostError
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
MODEL_NAMES = ["xgb_aug", "xgb_te", "xgb_lattice", "lgb_te", "cat_aug"]
MIN_STACK_GAIN = 0.00005
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


def detect_xgb_device() -> str:
    try:
        probe_frame = pd.DataFrame(
            {
                "number": np.zeros(8, dtype="float32"),
                "category": pd.Series(["a", "b"] * 4, dtype="category"),
            }
        )
        probe = xgb.XGBClassifier(
            n_estimators=1,
            tree_method="hist",
            device="cuda",
            enable_categorical=True,
            eval_metric="auc",
        )
        probe.fit(probe_frame, np.array([0, 1] * 4))
        print("XGBoost CUDA probe succeeded.", flush=True)
        return "cuda"
    except Exception as error:
        print(f"XGBoost CUDA unavailable; using CPU: {error}", flush=True)
        return "cpu"


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
    for offset, target_column in enumerate(NUMERIC_COLUMNS):
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


def prepare_catboost_frames(
    fit: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int]]:
    frames = [fit.copy(), valid.copy(), test.copy()]
    categorical = [
        column
        for column in fit.columns
        if isinstance(fit[column].dtype, pd.CategoricalDtype)
    ]
    for frame in frames:
        for column in categorical:
            frame[column] = (
                frame[column].astype(object).fillna("__missing__").astype(str)
            )
    indices = [frames[0].columns.get_loc(column) for column in categorical]
    return frames[0], frames[1], frames[2], indices


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
    device = detect_xgb_device()

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
    with (OUTPUT_DIR / "v9_generator_audit.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(generator_audit, file, indent=2)

    digit_rows = []
    for column in FRACTIONAL_COLUMNS:
        values = train[column].to_numpy(dtype="float64")
        digits = np.floor(values * 10.0) % 10.0
        observed = np.isfinite(digits)
        table = (
            pd.DataFrame(
                {"digit": digits[observed].astype("int8"), "target": target[observed]}
            )
            .groupby("digit")["target"]
            .agg(["mean", "size"])
            .reset_index()
        )
        for row in table.itertuples(index=False):
            digit_rows.append(
                {
                    "feature": column,
                    "first_decimal_digit": int(row.digit),
                    "target_rate": float(row.mean),
                    "rows": int(row.size),
                }
            )
    pd.DataFrame(digit_rows).to_csv(
        OUTPUT_DIR / "v9_decimal_lattice_audit.csv", index=False
    )

    train_imputed, test_imputed, imputation_report = transductive_impute(
        train,
        test,
        device,
    )
    if not np.isfinite(train_imputed.to_numpy()).all() or not np.isfinite(
        test_imputed.to_numpy()
    ).all():
        raise ValueError("Feature-only imputation produced non-finite values.")
    train_augmented = build_augmented_features(train_imputed, train)
    test_augmented = build_augmented_features(test_imputed, test)
    train_lattice = lattice_features(train)
    test_lattice = lattice_features(test)
    train_levels = exact_levels(train)
    test_levels = exact_levels(test)

    cardinality = []
    for column in ORIGINAL_COLUMNS:
        levels = int(train_levels[column].nunique())
        cardinality.append(
            {
                "feature": column,
                "levels": levels,
                "rows_per_level": len(train) / levels,
                "smoothing": SMOOTHING,
            }
        )
    pd.DataFrame(cardinality).to_csv(
        OUTPUT_DIR / "v9_encoding_cardinality.csv",
        index=False,
    )
    imputation_report.to_csv(OUTPUT_DIR / "v9_imputation_report.csv", index=False)

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )
    folds = list(splitter.split(train_augmented, target))
    fold_registry = np.full(len(train), -1, dtype="int8")
    oof = {
        name: np.zeros(len(train), dtype="float32")
        for name in MODEL_NAMES
    }
    test_predictions = {
        name: np.zeros(len(test), dtype="float64")
        for name in MODEL_NAMES
    }
    fold_rows = []
    importance_rows = []

    for fold, (fit_index, valid_index) in enumerate(folds, start=1):
        fold_registry[valid_index] = fold - 1
        print(f"\n===== fold {fold}/{N_FOLDS}: nested exact-value encoding =====", flush=True)
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

        xgb_jobs = [
            ("xgb_aug", augmented_fit, augmented_valid, augmented_test),
            ("xgb_te", te_fit, te_valid, te_test),
            ("xgb_lattice", lattice_fit, lattice_valid, lattice_test),
        ]
        for name, fit_x, valid_x, test_x in xgb_jobs:
            print(f"===== fold {fold}/{N_FOLDS}: {name} =====", flush=True)
            model = make_xgb_classifier(SEED, device)
            model.fit(
                fit_x,
                fit_target,
                eval_set=[(valid_x, valid_target)],
                verbose=False,
            )
            valid_prediction = model.predict_proba(valid_x)[:, 1]
            oof[name][valid_index] = valid_prediction
            test_predictions[name] += (
                model.predict_proba(test_x)[:, 1] / N_FOLDS
            )
            auc = float(roc_auc_score(valid_target, valid_prediction))
            fold_rows.append(
                {
                    "fold": fold,
                    "model": name,
                    "auc": auc,
                    "best_iteration": int(model.best_iteration),
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
                    fit_x.columns,
                    model.feature_importances_,
                )
            )
            print(
                f"{name}: auc={auc:.7f}, best_iteration={model.best_iteration}",
                flush=True,
            )
            del model
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
                te_fit.columns,
                lgb_model.feature_importances_,
            )
        )
        print(
            f"lgb_te: auc={lgb_auc:.7f}, best_iteration={lgb_model.best_iteration_}",
            flush=True,
        )

        print(f"===== fold {fold}/{N_FOLDS}: cat_aug =====", flush=True)
        cat_fit, cat_valid, cat_test, cat_indices = prepare_catboost_frames(
            augmented_fit,
            augmented_valid,
            augmented_test,
        )
        cat_parameters = dict(
            iterations=3000,
            learning_rate=0.06,
            depth=6,
            eval_metric="AUC",
            verbose=0,
            random_seed=SEED,
            early_stopping_rounds=100,
            allow_writing_files=False,
        )
        if device == "cuda":
            cat_parameters["task_type"] = "GPU"
        try:
            cat_model = CatBoostClassifier(**cat_parameters)
            cat_model.fit(
                cat_fit,
                fit_target,
                eval_set=(cat_valid, valid_target),
                cat_features=cat_indices,
                verbose=False,
            )
        except CatBoostError as error:
            if cat_parameters.get("task_type") != "GPU":
                raise
            print(f"CatBoost GPU failed; retrying CPU: {error}", flush=True)
            cat_parameters.pop("task_type", None)
            cat_model = CatBoostClassifier(**cat_parameters)
            cat_model.fit(
                cat_fit,
                fit_target,
                eval_set=(cat_valid, valid_target),
                cat_features=cat_indices,
                verbose=False,
            )
        cat_valid_prediction = cat_model.predict_proba(cat_valid)[:, 1]
        oof["cat_aug"][valid_index] = cat_valid_prediction
        test_predictions["cat_aug"] += (
            cat_model.predict_proba(cat_test)[:, 1] / N_FOLDS
        )
        cat_auc = float(roc_auc_score(valid_target, cat_valid_prediction))
        fold_rows.append(
            {
                "fold": fold,
                "model": "cat_aug",
                "auc": cat_auc,
                "best_iteration": int(cat_model.get_best_iteration()),
            }
        )
        importance_rows.extend(
            {
                "fold": fold,
                "model": "cat_aug",
                "feature": feature,
                "importance": importance,
            }
            for feature, importance in zip(
                cat_fit.columns,
                cat_model.feature_importances_,
            )
        )
        print(
            f"cat_aug: auc={cat_auc:.7f}, "
            f"best_iteration={cat_model.get_best_iteration()}",
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
            cat_model,
            cat_fit,
            cat_valid,
            cat_test,
        )
        gc.collect()

    model_auc = {
        name: float(roc_auc_score(target, prediction))
        for name, prediction in oof.items()
    }
    best_single_name = max(model_auc, key=model_auc.get)
    best_single_auc = model_auc[best_single_name]

    oof_logits = np.column_stack([logit(oof[name]) for name in MODEL_NAMES])
    test_logits = np.column_stack(
        [logit(test_predictions[name]) for name in MODEL_NAMES]
    )
    meta_oof = np.zeros(len(train), dtype="float64")
    stack_fold_rows = []
    for fold, (fit_index, valid_index) in enumerate(folds, start=1):
        meta = LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
        )
        meta.fit(oof_logits[fit_index], target[fit_index])
        meta_oof[valid_index] = meta.predict_proba(oof_logits[valid_index])[:, 1]
        stack_auc = float(
            roc_auc_score(target[valid_index], meta_oof[valid_index])
        )
        best_fold_auc = max(
            roc_auc_score(target[valid_index], oof[name][valid_index])
            for name in MODEL_NAMES
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

    stack_auc = float(roc_auc_score(target, meta_oof))
    stack_positive_folds = sum(
        row["delta_vs_best_single_fold"] > 0
        for row in stack_fold_rows
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
    if stack_accepted:
        selected_name = "logit_stack"
        selected_oof = meta_oof
        selected_test = stack_test
        selection_reason = "cross-fitted logit stack passed fixed guardrails"
    else:
        selected_name = best_single_name
        selected_oof = oof[best_single_name]
        selected_test = test_predictions[best_single_name]
        selection_reason = "fallback to strongest complete-OOF single model"

    fold_frame = pd.concat(
        [pd.DataFrame(fold_rows), pd.DataFrame(stack_fold_rows)],
        ignore_index=True,
        sort=False,
    )
    fold_frame.to_csv(OUTPUT_DIR / "v9_fold_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"model": name, "oof_auc": model_auc[name]}
            for name in MODEL_NAMES
        ]
        + [{"model": "logit_stack", "oof_auc": stack_auc}]
    ).sort_values("oof_auc", ascending=False).to_csv(
        OUTPUT_DIR / "v9_model_metrics.csv",
        index=False,
    )

    oof_frame = pd.DataFrame(
        {
            ID_COLUMN: train_ids,
            "fold": fold_registry,
            TARGET: target,
            **{f"{name}_prediction": oof[name] for name in MODEL_NAMES},
            "logit_stack_prediction": meta_oof,
            "selected_prediction": selected_oof,
        }
    )
    test_frame = pd.DataFrame(
        {
            ID_COLUMN: test_ids,
            **{
                f"{name}_prediction": test_predictions[name]
                for name in MODEL_NAMES
            },
            "logit_stack_prediction": stack_test,
            "selected_prediction": selected_test,
        }
    )
    oof_frame.to_csv(OUTPUT_DIR / "v9_oof.csv", index=False)
    test_frame.to_csv(OUTPUT_DIR / "v9_test_predictions.csv", index=False)

    coefficient_frame = pd.DataFrame(
        {
            "model": MODEL_NAMES,
            "coefficient": final_meta.coef_[0],
            "single_oof_auc": [model_auc[name] for name in MODEL_NAMES],
        }
    ).sort_values("coefficient", ascending=False)
    coefficient_frame.to_csv(
        OUTPUT_DIR / "v9_stack_coefficients.csv",
        index=False,
    )

    importance = pd.DataFrame(importance_rows)
    importance.groupby(["model", "feature"], as_index=False)["importance"].agg(
        mean="mean",
        std="std",
    ).sort_values(["model", "mean"], ascending=[True, False]).to_csv(
        OUTPUT_DIR / "v9_feature_importance.csv",
        index=False,
    )

    prediction_frame = pd.DataFrame(
        {name: oof[name] for name in MODEL_NAMES}
    )
    prediction_frame.corr().to_csv(
        OUTPUT_DIR / "v9_prediction_correlation.csv"
    )
    prediction_frame.rsub(target, axis=0).corr().to_csv(
        OUTPUT_DIR / "v9_residual_correlation.csv"
    )
    segments = []
    for name in MODEL_NAMES:
        segments += segment_metrics(train, target, oof[name], name)
    segments += segment_metrics(train, target, meta_oof, "logit_stack")
    segments += segment_metrics(train, target, selected_oof, "selected")
    pd.DataFrame(segments).to_csv(
        OUTPUT_DIR / "v9_segment_metrics.csv",
        index=False,
    )

    if sample_submission.columns.tolist() != [ID_COLUMN, TARGET]:
        raise ValueError("Unexpected sample_submission.csv columns.")
    if not sample_submission[ID_COLUMN].equals(test_ids):
        raise ValueError("sample submission IDs do not align with test.csv.")
    if not np.isfinite(selected_test).all():
        raise ValueError("Selected test prediction contains non-finite values.")
    sample_submission[TARGET] = np.clip(selected_test, 0.0, 1.0)
    sample_submission.to_csv(SUBMISSION_PATH, index=False)

    metrics = {
        "version": "v9",
        "competition_data_only": True,
        "device": device,
        "model_oof_auc": model_auc,
        "xgb_te_gain_vs_aug": model_auc["xgb_te"] - model_auc["xgb_aug"],
        "lattice_gain_vs_te": (
            model_auc["xgb_lattice"] - model_auc["xgb_te"]
        ),
        "best_single_model": best_single_name,
        "best_single_oof_auc": best_single_auc,
        "logit_stack_oof_auc": stack_auc,
        "logit_stack_gain": stack_auc - best_single_auc,
        "logit_stack_positive_folds": int(stack_positive_folds),
        "stack_accepted": stack_accepted,
        "selected_model": selected_name,
        "selected_oof_auc": float(roc_auc_score(target, selected_oof)),
        "selected_test_mean": float(np.mean(selected_test)),
        "train_target_mean": float(np.mean(target)),
        "selection_reason": selection_reason,
        "outer_folds": N_FOLDS,
        "inner_folds": INNER_FOLDS,
        "target_encoding_smoothing": SMOOTHING,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "submission_path": str(SUBMISSION_PATH),
    }
    with (OUTPUT_DIR / "v9_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print("\nV9 complete", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
