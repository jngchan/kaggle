"""Kaggle V7: test missingness-robust training against the fixed V3 baseline.

Designed to run as a single script in the kaggle/python environment. Attach the
competition data, then execute this file. All reusable artifacts are written to
/kaggle/working/output and the final submission to /kaggle/working/submission.csv.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


SEED = 2026
N_FOLDS = 5
NATIVE_ESTIMATORS = 3000
SEMANTIC_ESTIMATORS = 4500
NATIVE_EARLY_STOPPING = 150
SEMANTIC_EARLY_STOPPING = 200
XGB_ESTIMATORS = 3500
XGB_EARLY_STOPPING = 200
XGB_DEVICE = "cpu"
AUGMENT_FRACTION = 0.40
AUGMENT_WEIGHT = 0.35
MIN_OOF_GAIN = 0.00015
MIN_WEIGHTED_GAIN = 0.00015
MIN_POSITIVE_FOLDS = 4
MIN_HEAVY_MISSING_GAIN = 0.00010
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
V3_WEIGHTS = {"native": 0.10, "semantic": 0.25, "xgboost": 0.65}
REGRESSION_IMPUTATION_TARGETS = [
    "daily_screen_time_hours",
    "weekend_screen_time",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
]
VALUE_BOUNDS = {
    "age": (18.0, 35.0),
    "daily_screen_time_hours": (0.5, 15.0),
    "social_media_hours": (0.0, 8.0),
    "gaming_hours": (0.0, 4.0),
    "work_study_hours": (0.0, 6.0),
    "sleep_hours": (4.5, 9.0),
    "notifications_per_day": (20.0, 250.0),
    "app_opens_per_day": (15.0, 180.0),
    "weekend_screen_time": (0.5, 18.0),
}


def locate_input_files() -> tuple[Path, Path, Path]:
    """Find one attached Kaggle dataset containing all three competition files."""
    input_root = Path("/kaggle/input")
    for train_path in sorted(input_root.rglob("train.csv")):
        test_path = train_path.parent / "test.csv"
        sample_path = train_path.parent / "sample_submission.csv"
        if not (test_path.exists() and sample_path.exists()):
            continue
        train_columns = pd.read_csv(train_path, nrows=0).columns
        test_columns = pd.read_csv(test_path, nrows=0).columns
        required_train = {ID_COLUMN, TARGET, *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS}
        required_test = required_train.difference({TARGET})
        if required_train.issubset(train_columns) and required_test.issubset(test_columns):
            print(f"Using competition data from: {train_path.parent}", flush=True)
            return train_path, test_path, sample_path
    discovered = [str(path) for path in input_root.rglob("*.csv")]
    raise FileNotFoundError(
        "Could not find train.csv, test.csv and sample_submission.csv in one "
        f"/kaggle/input directory. Discovered CSV files: {discovered[:20]}"
    )


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_row_features(frame: pd.DataFrame, add_missing_flags: bool = True) -> pd.DataFrame:
    """Create deterministic row-level features without fitting across rows."""
    result = frame.copy()
    original_columns = [column for column in result if column != ID_COLUMN]
    if add_missing_flags:
        for column in original_columns:
            result[f"{column}__missing"] = result[column].isna().astype("int8")

    result["weekend_weekday_screen_diff"] = (
        result["weekend_screen_time"] - result["daily_screen_time_hours"]
    )
    result["weekend_weekday_screen_ratio"] = safe_divide(
        result["weekend_screen_time"], result["daily_screen_time_hours"]
    )
    result["social_screen_share"] = safe_divide(
        result["social_media_hours"], result["daily_screen_time_hours"]
    )
    result["gaming_screen_share"] = safe_divide(
        result["gaming_hours"], result["daily_screen_time_hours"]
    )
    result["productive_screen_share"] = safe_divide(
        result["work_study_hours"], result["daily_screen_time_hours"]
    )
    result["entertainment_hours"] = result["social_media_hours"] + result["gaming_hours"]
    result["entertainment_screen_share"] = safe_divide(
        result["entertainment_hours"], result["daily_screen_time_hours"]
    )
    result["screen_sleep_ratio"] = safe_divide(
        result["daily_screen_time_hours"], result["sleep_hours"]
    )
    result["weekend_screen_sleep_ratio"] = safe_divide(
        result["weekend_screen_time"], result["sleep_hours"]
    )
    result["notifications_per_screen_hour"] = safe_divide(
        result["notifications_per_day"], result["daily_screen_time_hours"]
    )
    result["opens_per_screen_hour"] = safe_divide(
        result["app_opens_per_day"], result["daily_screen_time_hours"]
    )
    result["notifications_per_open"] = safe_divide(
        result["notifications_per_day"], result["app_opens_per_day"]
    )
    result["digital_intensity"] = (
        result["daily_screen_time_hours"]
        + result["weekend_screen_time"]
        + result["social_media_hours"]
        + result["gaming_hours"]
    )
    result["recovery_balance"] = result["sleep_hours"] - result["daily_screen_time_hours"]
    return result.replace([np.inf, -np.inf], np.nan)


def build_base_features(raw: pd.DataFrame) -> pd.DataFrame:
    features = add_row_features(raw).drop(columns=ID_COLUMN)
    for column in CATEGORICAL_COLUMNS:
        features[column] = features[column].fillna("Missing").astype(str)
    numeric = features.select_dtypes(exclude="object").columns
    features[numeric] = features[numeric].astype("float32")
    return features


class SemanticFoldImputer:
    """Field-aware imputer fitted only on an outer training fold."""

    def __init__(self, random_state: int):
        self.random_state = random_state
        self.medians: pd.Series | None = None
        self.predictor_columns: list[str] = []
        self.regressors: dict[str, tuple[list[str], lgb.LGBMRegressor]] = {}
        self.group_tables: dict[str, tuple[list[str], pd.DataFrame]] = {}

    def _context(self, frame: pd.DataFrame) -> pd.DataFrame:
        context = pd.DataFrame(index=frame.index)
        context["gender"] = frame["gender"].fillna("Missing").astype(str)
        context["stress_level"] = frame["stress_level"].fillna("Missing").astype(str)
        age = frame["age"].fillna(self.medians["age"])
        daily = frame["daily_screen_time_hours"].fillna(
            self.medians["daily_screen_time_hours"]
        )
        notifications = frame["notifications_per_day"].fillna(
            self.medians["notifications_per_day"]
        )
        opens = frame["app_opens_per_day"].fillna(self.medians["app_opens_per_day"])
        context["age_band"] = pd.cut(
            age, [-np.inf, 21, 25, 29, 33, np.inf], labels=False
        ).astype("int8")
        context["screen_band"] = pd.cut(
            daily, [-np.inf, 3, 6, 9, 12, np.inf], labels=False
        ).astype("int8")
        context["notification_band"] = pd.cut(
            notifications, [-np.inf, 60, 110, 160, 210, np.inf], labels=False
        ).astype("int8")
        context["opens_band"] = pd.cut(
            opens, [-np.inf, 50, 90, 130, 160, np.inf], labels=False
        ).astype("int8")
        return context

    def _predictors(self, frame: pd.DataFrame, fitting: bool = False) -> pd.DataFrame:
        predictors = frame[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
        for column in NUMERIC_COLUMNS:
            predictors[f"{column}__missing"] = predictors[column].isna().astype("int8")
            predictors[column] = predictors[column].fillna(self.medians[column])
        predictors = pd.get_dummies(
            predictors, columns=CATEGORICAL_COLUMNS, dtype="int8"
        ).astype("float32")
        if fitting:
            self.predictor_columns = predictors.columns.tolist()
            return predictors
        return predictors.reindex(columns=self.predictor_columns, fill_value=0)

    def _fit_group(self, frame, context, target, keys):
        values = context[keys].copy()
        values[target] = frame[target]
        table = values.dropna(subset=[target]).groupby(keys, as_index=False)[target].median()
        self.group_tables[target] = (keys, table)

    def fit(self, frame: pd.DataFrame) -> "SemanticFoldImputer":
        self.medians = frame[NUMERIC_COLUMNS].median()
        predictors = self._predictors(frame, fitting=True)
        rng = np.random.default_rng(self.random_state)

        for offset, target_column in enumerate(REGRESSION_IMPUTATION_TARGETS):
            print(f"  fitting imputer: {target_column}", flush=True)
            observed = np.flatnonzero(frame[target_column].notna().to_numpy())
            if len(observed) > 400_000:
                observed = rng.choice(observed, size=400_000, replace=False)
            columns = [
                column
                for column in self.predictor_columns
                if column not in (target_column, f"{target_column}__missing")
            ]
            model = lgb.LGBMRegressor(
                objective="regression_l1",
                n_estimators=350,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=100,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=self.random_state + offset,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(predictors.iloc[observed][columns], frame.iloc[observed][target_column])
            self.regressors[target_column] = (columns, model)

        context = self._context(frame)
        self._fit_group(frame, context, "age", ["gender"])
        self._fit_group(frame, context, "sleep_hours", ["age_band", "stress_level"])
        self._fit_group(
            frame, context, "notifications_per_day", ["screen_band", "opens_band"]
        )
        self._fit_group(
            frame, context, "app_opens_per_day", ["screen_band", "notification_band"]
        )
        return self

    def _group_prediction(self, frame: pd.DataFrame, target: str) -> pd.Series:
        keys, table = self.group_tables[target]
        context = self._context(frame)
        prediction = context[keys].reset_index(drop=True).merge(
            table, on=keys, how="left", sort=False
        )[target]
        prediction.index = frame.index
        return prediction.fillna(self.medians[target])

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        imputed = frame[NUMERIC_COLUMNS].astype("float64").copy()
        predictors = self._predictors(frame)
        for target_column, (columns, model) in self.regressors.items():
            missing = imputed[target_column].isna()
            if missing.any():
                imputed.loc[missing, target_column] = model.predict(
                    predictors.loc[missing, columns]
                )
        for target_column in self.group_tables:
            missing = imputed[target_column].isna()
            if missing.any():
                imputed.loc[missing, target_column] = self._group_prediction(
                    frame, target_column
                ).loc[missing]
        imputed = imputed.fillna(self.medians)
        for column, (lower, upper) in VALUE_BOUNDS.items():
            imputed[column] = imputed[column].clip(lower, upper)
        imputed["notifications_per_day"] = imputed["notifications_per_day"].round()
        imputed["app_opens_per_day"] = imputed["app_opens_per_day"].round()

        companions = (
            add_row_features(imputed, add_missing_flags=False)
            .astype("float32")
            .add_suffix("__semantic_imputed")
        )
        return pd.concat(
            [frame.reset_index(drop=True), companions.reset_index(drop=True)], axis=1
        )


def encode_from_fold_vocabulary(fit, valid, test):
    fit_encoded = pd.get_dummies(fit, columns=CATEGORICAL_COLUMNS, dtype="int8")
    valid_encoded = pd.get_dummies(
        valid, columns=CATEGORICAL_COLUMNS, dtype="int8"
    ).reindex(columns=fit_encoded.columns, fill_value=0)
    test_encoded = pd.get_dummies(
        test, columns=CATEGORICAL_COLUMNS, dtype="int8"
    ).reindex(columns=fit_encoded.columns, fill_value=0)
    return fit_encoded, valid_encoded, test_encoded


def make_classifier(seed: int, n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_classifier(model, fit_x, fit_y, valid_x, valid_y, early_stopping_rounds):
    model.fit(
        fit_x,
        fit_y,
        eval_set=[(valid_x, valid_y)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(200),
        ],
    )


def detect_xgb_device() -> str:
    """Keep V3 reproduction independent of Kaggle GPU availability."""
    return XGB_DEVICE


def make_xgb_classifier(seed: int, device: str) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=XGB_ESTIMATORS,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        tree_method="hist",
        device=device,
        early_stopping_rounds=XGB_EARLY_STOPPING,
        random_state=seed,
        n_jobs=-1,
    )


def missing_mask(frame: pd.DataFrame) -> np.ndarray:
    """Return missingness for the 12 competition fields, including categories."""
    return frame[ORIGINAL_COLUMNS].isna().to_numpy(dtype=bool)


def make_drift_weights(raw_train: pd.DataFrame, raw_test: pd.DataFrame):
    """Reweight train missing patterns toward the unlabeled test distribution."""
    powers = 1 << np.arange(len(ORIGINAL_COLUMNS), dtype="int16")
    train_code = missing_mask(raw_train).astype("int16") @ powers
    test_code = missing_mask(raw_test).astype("int16") @ powers
    patterns = np.union1d(train_code, test_code)
    train_counts = pd.Series(train_code).value_counts().reindex(patterns, fill_value=0)
    test_counts = pd.Series(test_code).value_counts().reindex(patterns, fill_value=0)
    smoothing = 20.0
    train_rate = (train_counts + smoothing) / (
        len(train_code) + smoothing * len(patterns)
    )
    test_rate = (test_counts + smoothing) / (
        len(test_code) + smoothing * len(patterns)
    )
    ratio = (test_rate / train_rate).clip(0.25, 4.0)
    weights = pd.Series(train_code).map(ratio).to_numpy(dtype="float64")
    return weights / weights.mean()


def build_augmented_rows(
    fold_fit: pd.DataFrame,
    fit_y: pd.Series,
    test_masks: np.ndarray,
    seed: int,
):
    """Create lower-weight rows with exact test-like masks inside one outer fold.

    A source row is only used when all of its existing missing fields are also
    missing in the sampled target pattern. The operation therefore only hides
    observed values; it never fabricates values or uses validation labels.
    """
    rng = np.random.default_rng(seed)
    row_count = int(round(len(fold_fit) * AUGMENT_FRACTION))
    target_masks = test_masks[rng.integers(0, len(test_masks), size=row_count)]
    source_masks = fold_fit[
        [f"{column}__missing" for column in ORIGINAL_COLUMNS]
    ].to_numpy(dtype=bool)
    source_index = rng.integers(0, len(fold_fit), size=row_count)

    for _ in range(20):
        incompatible = np.any(source_masks[source_index] & ~target_masks, axis=1)
        if not incompatible.any():
            break
        source_index[incompatible] = rng.integers(
            0, len(fold_fit), size=int(incompatible.sum())
        )

    incompatible = np.any(source_masks[source_index] & ~target_masks, axis=1)
    if incompatible.any():
        complete = np.flatnonzero(~source_masks.any(axis=1))
        if not len(complete):
            raise RuntimeError("No complete rows available for mask augmentation.")
        source_index[incompatible] = rng.choice(
            complete, size=int(incompatible.sum()), replace=True
        )

    raw_augmented = fold_fit.iloc[source_index][ORIGINAL_COLUMNS].reset_index(drop=True)
    raw_augmented[CATEGORICAL_COLUMNS] = raw_augmented[
        CATEGORICAL_COLUMNS
    ].replace("Missing", np.nan)
    for column_index, column in enumerate(ORIGINAL_COLUMNS):
        mask = target_masks[:, column_index]
        if column in CATEGORICAL_COLUMNS:
            raw_augmented.loc[mask, column] = np.nan
        else:
            raw_augmented.loc[mask, column] = np.nan
    augmented = add_row_features(raw_augmented)
    for column in CATEGORICAL_COLUMNS:
        augmented[column] = augmented[column].fillna("Missing").astype(str)
    actual_masks = augmented[
        [f"{column}__missing" for column in ORIGINAL_COLUMNS]
    ].to_numpy(dtype=bool)
    if not np.array_equal(actual_masks, target_masks):
        raise RuntimeError("Augmented rows do not match their sampled test masks.")
    numeric = augmented.select_dtypes(exclude="object").columns
    augmented[numeric] = augmented[numeric].astype("float32")
    labels = fit_y.iloc[source_index].reset_index(drop=True)
    return augmented, labels, target_masks


def weighted_auc(target, prediction, weights) -> float:
    return float(roc_auc_score(target, prediction, sample_weight=weights))


def segment_metrics(base_features, target, prediction, model_name):
    missing_columns = [
        column
        for column in base_features
        if column.endswith("__missing")
        and column.removesuffix("__missing") in base_features
    ]
    missing_count = base_features[missing_columns].sum(axis=1)
    rows = []

    def add(name, mask):
        subset = target.loc[mask]
        if len(subset) and subset.nunique() == 2:
            rows.append(
                {
                    "model": model_name,
                    "segment": name,
                    "rows": int(mask.sum()),
                    "auc": float(roc_auc_score(subset, prediction[mask.to_numpy()])),
                }
            )

    for count in range(4):
        add(f"missing_count_{count}", missing_count == count)
    add("missing_count_4plus", missing_count >= 4)
    for column in missing_columns:
        add(column, base_features[column].astype(bool))
    return rows


def main() -> None:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path, test_path, sample_path = locate_input_files()
    raw_train = pd.read_csv(train_path)
    raw_test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)
    target = raw_train.pop(TARGET).astype("int8")
    train_ids = raw_train[ID_COLUMN].copy()
    test_ids = raw_test[ID_COLUMN].copy()
    test_masks = missing_mask(raw_test)
    drift_weights = make_drift_weights(raw_train, raw_test)
    base_train = build_base_features(raw_train)
    base_test = build_base_features(raw_test)
    xgb_device = detect_xgb_device()
    print(
        f"train={len(base_train):,}, test={len(base_test):,}, "
        f"base_features={base_train.shape[1]}, xgb_device={xgb_device}",
        flush=True,
    )

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(base_train, target))
    fold_registry = np.full(len(base_train), -1, dtype="int8")
    native_oof = np.zeros(len(base_train), dtype="float32")
    semantic_oof = np.zeros(len(base_train), dtype="float32")
    xgb_oof = np.zeros(len(base_train), dtype="float32")
    augmented_oof = np.zeros(len(base_train), dtype="float32")
    native_test = np.zeros(len(base_test), dtype="float64")
    semantic_test = np.zeros(len(base_test), dtype="float64")
    xgb_test = np.zeros(len(base_test), dtype="float64")
    augmented_test = np.zeros(len(base_test), dtype="float64")
    fold_rows = []
    importance_rows = []
    augmentation_rows = []

    for fold, (fit_index, valid_index) in enumerate(splits, start=1):
        fold_registry[valid_index] = fold - 1
        fold_fit = base_train.iloc[fit_index].reset_index(drop=True)
        fold_valid = base_train.iloc[valid_index].reset_index(drop=True)

        print(f"\n===== fold {fold}/{N_FOLDS}: native model =====", flush=True)
        native_fit, native_valid, native_test_x = encode_from_fold_vocabulary(
            fold_fit, fold_valid, base_test
        )
        native_model = make_classifier(SEED + fold, NATIVE_ESTIMATORS)
        fit_classifier(
            native_model,
            native_fit,
            target.iloc[fit_index],
            native_valid,
            target.iloc[valid_index],
            NATIVE_EARLY_STOPPING,
        )
        native_valid_prediction = native_model.predict_proba(native_valid)[:, 1]
        native_oof[valid_index] = native_valid_prediction
        native_test += native_model.predict_proba(native_test_x)[:, 1] / N_FOLDS
        native_auc = roc_auc_score(target.iloc[valid_index], native_valid_prediction)

        print(f"===== fold {fold}/{N_FOLDS}: semantic imputation =====", flush=True)
        imputer = SemanticFoldImputer(SEED + fold).fit(fold_fit)
        semantic_fit = imputer.transform(fold_fit)
        semantic_valid = imputer.transform(fold_valid)
        semantic_test_frame = imputer.transform(base_test)
        semantic_fit, semantic_valid, semantic_test_x = encode_from_fold_vocabulary(
            semantic_fit, semantic_valid, semantic_test_frame
        )

        print(f"===== fold {fold}/{N_FOLDS}: semantic model =====", flush=True)
        semantic_model = make_classifier(SEED + fold, SEMANTIC_ESTIMATORS)
        fit_classifier(
            semantic_model,
            semantic_fit,
            target.iloc[fit_index],
            semantic_valid,
            target.iloc[valid_index],
            SEMANTIC_EARLY_STOPPING,
        )
        semantic_valid_prediction = semantic_model.predict_proba(semantic_valid)[:, 1]
        semantic_oof[valid_index] = semantic_valid_prediction
        semantic_test += semantic_model.predict_proba(semantic_test_x)[:, 1] / N_FOLDS
        semantic_auc = roc_auc_score(target.iloc[valid_index], semantic_valid_prediction)

        print(f"===== fold {fold}/{N_FOLDS}: semantic XGBoost =====", flush=True)
        xgb_model = make_xgb_classifier(SEED + fold, xgb_device)
        xgb_model.fit(
            semantic_fit,
            target.iloc[fit_index],
            eval_set=[(semantic_valid, target.iloc[valid_index])],
            verbose=200,
        )
        xgb_valid_prediction = xgb_model.predict_proba(semantic_valid)[:, 1]
        xgb_oof[valid_index] = xgb_valid_prediction
        xgb_test += xgb_model.predict_proba(semantic_test_x)[:, 1] / N_FOLDS
        xgb_auc = roc_auc_score(target.iloc[valid_index], xgb_valid_prediction)

        print(f"===== fold {fold}/{N_FOLDS}: missing-mask augmentation =====", flush=True)
        augmented_base, augmented_y, sampled_masks = build_augmented_rows(
            fold_fit,
            target.iloc[fit_index].reset_index(drop=True),
            test_masks,
            SEED + 1000 + fold,
        )
        for column_index, column in enumerate(ORIGINAL_COLUMNS):
            augmentation_rows.append(
                {
                    "fold": fold,
                    "field": column,
                    "outer_fit_missing_rate": float(
                        fold_fit[f"{column}__missing"].mean()
                    ),
                    "test_missing_rate": float(test_masks[:, column_index].mean()),
                    "augmented_missing_rate": float(
                        sampled_masks[:, column_index].mean()
                    ),
                }
            )
        augmented_semantic = imputer.transform(augmented_base)
        augmented_semantic = pd.get_dummies(
            augmented_semantic, columns=CATEGORICAL_COLUMNS, dtype="int8"
        ).reindex(columns=semantic_fit.columns, fill_value=0)
        robust_fit = pd.concat(
            [semantic_fit, augmented_semantic], ignore_index=True, copy=False
        )
        robust_y = pd.concat(
            [target.iloc[fit_index].reset_index(drop=True), augmented_y],
            ignore_index=True,
        )
        robust_weight = np.concatenate(
            [
                np.ones(len(semantic_fit), dtype="float32"),
                np.full(len(augmented_semantic), AUGMENT_WEIGHT, dtype="float32"),
            ]
        )
        # Match the baseline seed so the paired delta isolates augmentation.
        robust_model = make_xgb_classifier(SEED + fold, xgb_device)
        robust_model.fit(
            robust_fit,
            robust_y,
            sample_weight=robust_weight,
            eval_set=[(semantic_valid, target.iloc[valid_index])],
            verbose=200,
        )
        robust_valid_prediction = robust_model.predict_proba(semantic_valid)[:, 1]
        augmented_oof[valid_index] = robust_valid_prediction
        augmented_test += robust_model.predict_proba(semantic_test_x)[:, 1] / N_FOLDS
        robust_auc = roc_auc_score(
            target.iloc[valid_index], robust_valid_prediction
        )
        robust_weighted_auc = weighted_auc(
            target.iloc[valid_index],
            robust_valid_prediction,
            drift_weights[valid_index],
        )
        baseline_weighted_auc = weighted_auc(
            target.iloc[valid_index],
            xgb_valid_prediction,
            drift_weights[valid_index],
        )
        fold_rows.extend(
            [
                {
                    "fold": fold,
                    "model": "native",
                    "auc": native_auc,
                    "best_iteration": native_model.best_iteration_,
                },
                {
                    "fold": fold,
                    "model": "semantic",
                    "auc": semantic_auc,
                    "best_iteration": semantic_model.best_iteration_,
                },
                {
                    "fold": fold,
                    "model": "xgboost",
                    "auc": xgb_auc,
                    "drift_weighted_auc": baseline_weighted_auc,
                    "best_iteration": xgb_model.best_iteration,
                },
                {
                    "fold": fold,
                    "model": "missing_robust_xgboost",
                    "auc": robust_auc,
                    "drift_weighted_auc": robust_weighted_auc,
                    "best_iteration": robust_model.best_iteration,
                    "augmented_rows": len(augmented_semantic),
                    "augmentation_weight": AUGMENT_WEIGHT,
                    "sampled_missing_rate": float(sampled_masks.mean()),
                },
            ]
        )
        for model_name, model, columns in (
            ("native", native_model, native_fit.columns),
            ("semantic", semantic_model, semantic_fit.columns),
            ("xgboost", xgb_model, semantic_fit.columns),
            ("missing_robust_xgboost", robust_model, semantic_fit.columns),
        ):
            importance_rows.extend(
                {
                    "fold": fold,
                    "model": model_name,
                    "feature": feature,
                    "importance": importance,
                }
                for feature, importance in zip(columns, model.feature_importances_)
            )
        print(
            f"fold {fold}: native_auc={native_auc:.7f}, "
            f"semantic_auc={semantic_auc:.7f}, xgb_auc={xgb_auc:.7f}, "
            f"robust_auc={robust_auc:.7f}, delta={robust_auc-xgb_auc:+.7f}",
            flush=True,
        )
        del (
            native_fit,
            native_valid,
            native_test_x,
            semantic_fit,
            semantic_valid,
            semantic_test_x,
            semantic_test_frame,
            native_model,
            semantic_model,
            xgb_model,
            robust_model,
            augmented_base,
            augmented_semantic,
            augmented_y,
            sampled_masks,
            robust_fit,
            robust_y,
            robust_weight,
            imputer,
        )
        gc.collect()

    native_auc = roc_auc_score(target, native_oof)
    semantic_auc = roc_auc_score(target, semantic_oof)
    xgb_auc = roc_auc_score(target, xgb_oof)
    augmented_auc = roc_auc_score(target, augmented_oof)
    xgb_weighted_auc = weighted_auc(target, xgb_oof, drift_weights)
    augmented_weighted_auc = weighted_auc(target, augmented_oof, drift_weights)
    v3_oof = (
        V3_WEIGHTS["native"] * native_oof
        + V3_WEIGHTS["semantic"] * semantic_oof
        + V3_WEIGHTS["xgboost"] * xgb_oof
    )
    v3_test = (
        V3_WEIGHTS["native"] * native_test
        + V3_WEIGHTS["semantic"] * semantic_test
        + V3_WEIGHTS["xgboost"] * xgb_test
    )
    v3_auc = float(roc_auc_score(target, v3_oof))
    v3_weighted_auc = weighted_auc(target, v3_oof, drift_weights)

    missing_columns = [f"{column}__missing" for column in ORIGINAL_COLUMNS]
    missing_count = base_train[missing_columns].sum(axis=1).to_numpy()
    heavy_mask = missing_count >= 2
    xgb_heavy_auc = float(roc_auc_score(target[heavy_mask], xgb_oof[heavy_mask]))
    augmented_heavy_auc = float(
        roc_auc_score(target[heavy_mask], augmented_oof[heavy_mask])
    )
    fold_frame = pd.DataFrame(fold_rows)
    baseline_folds = fold_frame[fold_frame["model"] == "xgboost"].set_index("fold")
    robust_folds = fold_frame[
        fold_frame["model"] == "missing_robust_xgboost"
    ].set_index("fold")
    positive_folds = int((robust_folds["auc"] > baseline_folds["auc"]).sum())
    model_accepted = bool(
        augmented_auc >= xgb_auc + MIN_OOF_GAIN
        and augmented_weighted_auc >= xgb_weighted_auc + MIN_WEIGHTED_GAIN
        and positive_folds >= MIN_POSITIVE_FOLDS
        and augmented_heavy_auc >= xgb_heavy_auc + MIN_HEAVY_MISSING_GAIN
    )

    blend_candidates = []
    for robust_weight in (0.10, 0.20, 0.30):
        prediction = (1.0 - robust_weight) * v3_oof + robust_weight * augmented_oof
        ordinary = float(roc_auc_score(target, prediction))
        pressure = weighted_auc(target, prediction, drift_weights)
        blend_candidates.append(
            {
                "robust_weight": robust_weight,
                "oof_auc": ordinary,
                "drift_weighted_auc": pressure,
                "robust_score": 0.5 * (ordinary + pressure),
            }
        )
    blend_frame = pd.DataFrame(blend_candidates).sort_values(
        ["robust_score", "robust_weight"], ascending=[False, True]
    )
    winner = blend_frame.iloc[0]
    blend_accepted = bool(
        model_accepted
        and winner["oof_auc"] >= v3_auc + MIN_OOF_GAIN
        and winner["drift_weighted_auc"] >= v3_weighted_auc + MIN_WEIGHTED_GAIN
    )
    if blend_accepted:
        robust_weight = float(winner["robust_weight"])
        blend_oof = (1.0 - robust_weight) * v3_oof + robust_weight * augmented_oof
        blend_test = (1.0 - robust_weight) * v3_test + robust_weight * augmented_test
        selection_reason = "missing-robust model passed all fixed CV guardrails"
    else:
        robust_weight = 0.0
        blend_oof = v3_oof
        blend_test = v3_test
        selection_reason = "strict CV fallback to fixed V3 blend"

    oof_frame = pd.DataFrame(
        {
            ID_COLUMN: train_ids,
            "fold": fold_registry,
            TARGET: target,
            "native_prediction": native_oof,
            "semantic_prediction": semantic_oof,
            "xgb_prediction": xgb_oof,
            "missing_robust_prediction": augmented_oof,
            "drift_weight": drift_weights,
            "blend_prediction": blend_oof,
        }
    )
    test_prediction_frame = pd.DataFrame(
        {
            ID_COLUMN: test_ids,
            "native_prediction": native_test,
            "semantic_prediction": semantic_test,
            "xgb_prediction": xgb_test,
            "missing_robust_prediction": augmented_test,
            "blend_prediction": blend_test,
        }
    )
    oof_frame.to_csv(OUTPUT_DIR / "v7_oof.csv", index=False)
    test_prediction_frame.to_csv(OUTPUT_DIR / "v7_test_predictions.csv", index=False)
    fold_frame.to_csv(OUTPUT_DIR / "v7_fold_metrics.csv", index=False)
    pd.DataFrame(augmentation_rows).to_csv(
        OUTPUT_DIR / "v7_augmentation_report.csv", index=False
    )
    blend_frame.to_csv(OUTPUT_DIR / "v7_blend_search.csv", index=False)
    importance = pd.DataFrame(importance_rows)
    importance.groupby(["model", "feature"], as_index=False)["importance"].agg(
        mean="mean", std="std"
    ).sort_values(["model", "mean"], ascending=[True, False]).to_csv(
        OUTPUT_DIR / "v7_feature_importance.csv", index=False
    )
    segments = segment_metrics(base_train, target, native_oof, "native")
    segments += segment_metrics(base_train, target, semantic_oof, "semantic")
    segments += segment_metrics(base_train, target, xgb_oof, "xgboost")
    segments += segment_metrics(
        base_train, target, augmented_oof, "missing_robust_xgboost"
    )
    segments += segment_metrics(base_train, target, v3_oof, "fixed_v3_blend")
    segments += segment_metrics(base_train, target, blend_oof, "selected_blend")
    pd.DataFrame(segments).to_csv(OUTPUT_DIR / "v7_segment_metrics.csv", index=False)

    if sample_submission.columns.tolist() != [ID_COLUMN, TARGET]:
        raise ValueError("Unexpected sample_submission.csv columns.")
    if not sample_submission[ID_COLUMN].equals(test_ids):
        raise ValueError("sample_submission IDs are not aligned with test.csv.")
    sample_submission[TARGET] = np.clip(blend_test, 0.0, 1.0)
    if not np.isfinite(sample_submission[TARGET]).all():
        raise ValueError("Submission contains non-finite predictions.")
    sample_submission.to_csv(SUBMISSION_PATH, index=False)

    metrics = {
        "version": "v7",
        "native_oof_auc": float(native_auc),
        "semantic_oof_auc": float(semantic_auc),
        "xgb_oof_auc": float(xgb_auc),
        "missing_robust_xgb_oof_auc": float(augmented_auc),
        "missing_robust_xgb_gain": float(augmented_auc - xgb_auc),
        "xgb_drift_weighted_auc": float(xgb_weighted_auc),
        "missing_robust_xgb_drift_weighted_auc": float(augmented_weighted_auc),
        "missing_robust_xgb_drift_gain": float(
            augmented_weighted_auc - xgb_weighted_auc
        ),
        "xgb_heavy_missing_auc": xgb_heavy_auc,
        "missing_robust_xgb_heavy_missing_auc": augmented_heavy_auc,
        "missing_robust_xgb_heavy_missing_gain": float(
            augmented_heavy_auc - xgb_heavy_auc
        ),
        "missing_robust_positive_folds": positive_folds,
        "model_accepted": model_accepted,
        "fixed_v3_oof_auc": v3_auc,
        "fixed_v3_drift_weighted_auc": v3_weighted_auc,
        "selected_oof_auc": float(roc_auc_score(target, blend_oof)),
        "selected_drift_weighted_auc": weighted_auc(
            target, blend_oof, drift_weights
        ),
        "selected_robust_weight": robust_weight,
        "blend_accepted": blend_accepted,
        "selection_reason": selection_reason,
        "v3_weights": V3_WEIGHTS,
        "augmentation_fraction": AUGMENT_FRACTION,
        "augmentation_weight": AUGMENT_WEIGHT,
        "guardrails": {
            "min_oof_gain": MIN_OOF_GAIN,
            "min_weighted_gain": MIN_WEIGHTED_GAIN,
            "min_positive_folds": MIN_POSITIVE_FOLDS,
            "min_heavy_missing_gain": MIN_HEAVY_MISSING_GAIN,
        },
        "folds": N_FOLDS,
        "seed": SEED,
        "native_estimators": NATIVE_ESTIMATORS,
        "semantic_estimators": SEMANTIC_ESTIMATORS,
        "native_early_stopping": NATIVE_EARLY_STOPPING,
        "semantic_early_stopping": SEMANTIC_EARLY_STOPPING,
        "xgb_estimators": XGB_ESTIMATORS,
        "xgb_early_stopping": XGB_EARLY_STOPPING,
        "xgb_device": xgb_device,
        "elapsed_minutes": (time.time() - started) / 60,
        "submission_path": str(SUBMISSION_PATH),
    }
    with (OUTPUT_DIR / "v7_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print("\nV7 complete", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
