"""Kaggle V8: cross-fitted spline main effects plus XGBoost residual trees.

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
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer


SEED = 2026
N_FOLDS = 3
XGB_ESTIMATORS = 3500
XGB_EARLY_STOPPING = 200
XGB_DEVICE = "cpu"
INNER_FOLDS = 3
MIN_OOF_GAIN = 0.00020
MIN_WEIGHTED_GAIN = 0.00020
MIN_POSITIVE_FOLDS = 3
SPLINE_KNOTS = 10
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


def make_residual_xgb_classifier(seed: int, device: str) -> xgb.XGBClassifier:
    """Use shallower trees after smooth main effects have been removed."""
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=15,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_alpha=0.1,
        reg_lambda=2.0,
        tree_method="hist",
        device=device,
        early_stopping_rounds=XGB_EARLY_STOPPING,
        random_state=seed,
        n_jobs=-1,
    )


def missing_mask(frame: pd.DataFrame) -> np.ndarray:
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
    ratios = (test_rate / train_rate).clip(0.25, 4.0)
    weights = pd.Series(train_code).map(ratios).to_numpy(dtype="float64")
    return weights / weights.mean()


def weighted_auc(target, prediction, weights) -> float:
    return float(roc_auc_score(target, prediction, sample_weight=weights))


class SplineAdditiveModel:
    """Sparse GAM-like logistic model for smooth main effects and context flags."""

    def __init__(self, seed: int):
        self.seed = seed
        self.spline_columns = [
            f"{column}__semantic_imputed" for column in NUMERIC_COLUMNS
        ]
        self.context_columns: list[str] = []
        self.transformer = SplineTransformer(
            n_knots=SPLINE_KNOTS,
            degree=3,
            knots="quantile",
            include_bias=False,
            sparse_output=True,
        )
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=2e-6,
            max_iter=150,
            tol=1e-5,
            average=True,
            random_state=seed,
            n_jobs=-1,
        )

    def _context(self, frame: pd.DataFrame):
        values = frame.reindex(columns=self.context_columns, fill_value=0)
        return sparse.csr_matrix(values.to_numpy(dtype="float32"))

    def fit(self, frame: pd.DataFrame, target: pd.Series):
        self.context_columns = [
            column
            for column in frame.columns
            if column.endswith("__missing")
            or any(column.startswith(f"{cat}_") for cat in CATEGORICAL_COLUMNS)
        ]
        spline_values = self.transformer.fit_transform(frame[self.spline_columns])
        design = sparse.hstack(
            [spline_values, self._context(frame)], format="csr", dtype="float32"
        )
        self.model.fit(design, target)
        return self

    def margin(self, frame: pd.DataFrame) -> np.ndarray:
        spline_values = self.transformer.transform(frame[self.spline_columns])
        design = sparse.hstack(
            [spline_values, self._context(frame)], format="csr", dtype="float32"
        )
        return np.clip(self.model.decision_function(design), -12.0, 12.0).astype(
            "float32"
        )

    def coefficients(self, outer_fold: int) -> pd.DataFrame:
        spline_names = self.transformer.get_feature_names_out(self.spline_columns)
        names = list(spline_names) + self.context_columns
        return pd.DataFrame(
            {
                "outer_fold": outer_fold,
                "feature": names,
                "coefficient": self.model.coef_[0],
            }
        )


def cross_fitted_additive_margins(
    fit: pd.DataFrame,
    fit_y: pd.Series,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    outer_fold: int,
):
    """Produce leakage-safe training margins and outer-fold predictions."""
    inner = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=SEED + 100 * outer_fold,
    )
    fit_margin = np.zeros(len(fit), dtype="float32")
    for inner_fold, (inner_fit, inner_valid) in enumerate(
        inner.split(fit, fit_y), start=1
    ):
        model = SplineAdditiveModel(
            SEED + 1000 * outer_fold + inner_fold
        ).fit(fit.iloc[inner_fit], fit_y.iloc[inner_fit])
        fit_margin[inner_valid] = model.margin(fit.iloc[inner_valid])
        del model
        gc.collect()

    full_model = SplineAdditiveModel(SEED + 2000 + outer_fold).fit(fit, fit_y)
    valid_margin = full_model.margin(valid)
    test_margin = full_model.margin(test)
    coefficients = full_model.coefficients(outer_fold)
    return fit_margin, valid_margin, test_margin, coefficients


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
    drift_weights = make_drift_weights(raw_train, raw_test)
    base_train = build_base_features(raw_train)
    base_test = build_base_features(raw_test)
    xgb_device = detect_xgb_device()

    pd.DataFrame(
        {
            "feature": NUMERIC_COLUMNS,
            "train_unique_nonmissing": [
                int(raw_train[column].nunique(dropna=True))
                for column in NUMERIC_COLUMNS
            ],
            "test_unique_nonmissing": [
                int(raw_test[column].nunique(dropna=True))
                for column in NUMERIC_COLUMNS
            ],
        }
    ).to_csv(OUTPUT_DIR / "v8_numeric_cardinality.csv", index=False)
    print(
        f"train={len(base_train):,}, test={len(base_test):,}, "
        f"screening_folds={N_FOLDS}, inner_folds={INNER_FOLDS}, "
        f"xgb_device={xgb_device}",
        flush=True,
    )

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(base_train, target))
    fold_registry = np.full(len(base_train), -1, dtype="int8")
    baseline_oof = np.zeros(len(base_train), dtype="float32")
    additive_oof = np.zeros(len(base_train), dtype="float32")
    residual_oof = np.zeros(len(base_train), dtype="float32")
    baseline_test = np.zeros(len(base_test), dtype="float64")
    additive_test = np.zeros(len(base_test), dtype="float64")
    residual_test = np.zeros(len(base_test), dtype="float64")
    fold_rows = []
    importance_rows = []
    coefficient_frames = []

    for fold, (fit_index, valid_index) in enumerate(splits, start=1):
        fold_registry[valid_index] = fold - 1
        fold_fit = base_train.iloc[fit_index].reset_index(drop=True)
        fold_valid = base_train.iloc[valid_index].reset_index(drop=True)
        fit_y = target.iloc[fit_index].reset_index(drop=True)
        valid_y = target.iloc[valid_index].reset_index(drop=True)
        valid_drift = drift_weights[valid_index]

        print(f"\n===== fold {fold}/{N_FOLDS}: semantic imputation =====", flush=True)
        imputer = SemanticFoldImputer(SEED + fold).fit(fold_fit)
        semantic_fit_frame = imputer.transform(fold_fit)
        semantic_valid_frame = imputer.transform(fold_valid)
        semantic_test_frame = imputer.transform(base_test)
        semantic_fit, semantic_valid, semantic_test = encode_from_fold_vocabulary(
            semantic_fit_frame, semantic_valid_frame, semantic_test_frame
        )

        print(f"===== fold {fold}/{N_FOLDS}: baseline XGBoost =====", flush=True)
        baseline_model = make_xgb_classifier(SEED + fold, xgb_device)
        baseline_model.fit(
            semantic_fit,
            fit_y,
            eval_set=[(semantic_valid, valid_y)],
            verbose=200,
        )
        baseline_valid_prediction = baseline_model.predict_proba(semantic_valid)[:, 1]
        baseline_oof[valid_index] = baseline_valid_prediction
        baseline_test += baseline_model.predict_proba(semantic_test)[:, 1] / N_FOLDS
        baseline_auc = float(roc_auc_score(valid_y, baseline_valid_prediction))
        baseline_weighted_auc = weighted_auc(
            valid_y, baseline_valid_prediction, valid_drift
        )

        print(
            f"===== fold {fold}/{N_FOLDS}: cross-fitted spline main effects =====",
            flush=True,
        )
        (
            fit_margin,
            valid_margin,
            test_margin,
            coefficients,
        ) = cross_fitted_additive_margins(
            semantic_fit,
            fit_y,
            semantic_valid,
            semantic_test,
            fold,
        )
        coefficient_frames.append(coefficients)
        additive_valid_prediction = 1.0 / (1.0 + np.exp(-valid_margin))
        additive_test_prediction = 1.0 / (1.0 + np.exp(-test_margin))
        additive_fit_prediction = 1.0 / (1.0 + np.exp(-fit_margin))
        additive_oof[valid_index] = additive_valid_prediction
        additive_test += additive_test_prediction / N_FOLDS
        additive_auc = float(roc_auc_score(valid_y, additive_valid_prediction))

        print(f"===== fold {fold}/{N_FOLDS}: residual XGBoost =====", flush=True)
        residual_model = make_residual_xgb_classifier(SEED + fold, xgb_device)
        residual_model.fit(
            semantic_fit,
            fit_y,
            base_margin=fit_margin,
            eval_set=[(semantic_valid, valid_y)],
            base_margin_eval_set=[valid_margin],
            verbose=200,
        )
        residual_valid_prediction = residual_model.predict_proba(
            semantic_valid, base_margin=valid_margin
        )[:, 1]
        residual_oof[valid_index] = residual_valid_prediction
        residual_test += residual_model.predict_proba(
            semantic_test, base_margin=test_margin
        )[:, 1] / N_FOLDS
        residual_auc = float(roc_auc_score(valid_y, residual_valid_prediction))
        residual_weighted_auc = weighted_auc(
            valid_y, residual_valid_prediction, valid_drift
        )

        fold_rows.extend(
            [
                {
                    "fold": fold,
                    "model": "baseline_xgboost",
                    "auc": baseline_auc,
                    "drift_weighted_auc": baseline_weighted_auc,
                    "best_iteration": baseline_model.best_iteration,
                },
                {
                    "fold": fold,
                    "model": "spline_additive",
                    "auc": additive_auc,
                    "drift_weighted_auc": weighted_auc(
                        valid_y, additive_valid_prediction, valid_drift
                    ),
                    "inner_oof_auc": float(
                        roc_auc_score(fit_y, additive_fit_prediction)
                    ),
                    "spline_knots": SPLINE_KNOTS,
                },
                {
                    "fold": fold,
                    "model": "spline_residual_xgboost",
                    "auc": residual_auc,
                    "drift_weighted_auc": residual_weighted_auc,
                    "best_iteration": residual_model.best_iteration,
                },
            ]
        )
        for model_name, model in (
            ("baseline_xgboost", baseline_model),
            ("spline_residual_xgboost", residual_model),
        ):
            importance_rows.extend(
                {
                    "fold": fold,
                    "model": model_name,
                    "feature": feature,
                    "importance": importance,
                }
                for feature, importance in zip(
                    semantic_fit.columns, model.feature_importances_
                )
            )

        print(
            f"fold {fold}: baseline={baseline_auc:.7f}, "
            f"additive={additive_auc:.7f}, residual={residual_auc:.7f}, "
            f"delta={residual_auc-baseline_auc:+.7f}",
            flush=True,
        )
        del (
            fold_fit,
            fold_valid,
            semantic_fit_frame,
            semantic_valid_frame,
            semantic_test_frame,
            semantic_fit,
            semantic_valid,
            semantic_test,
            baseline_model,
            residual_model,
            imputer,
            fit_margin,
            valid_margin,
            test_margin,
            coefficients,
        )
        gc.collect()

    baseline_auc = float(roc_auc_score(target, baseline_oof))
    additive_auc = float(roc_auc_score(target, additive_oof))
    residual_auc = float(roc_auc_score(target, residual_oof))
    baseline_weighted_auc = weighted_auc(target, baseline_oof, drift_weights)
    residual_weighted_auc = weighted_auc(target, residual_oof, drift_weights)
    fold_frame = pd.DataFrame(fold_rows)
    baseline_folds = fold_frame[
        fold_frame["model"] == "baseline_xgboost"
    ].set_index("fold")
    residual_folds = fold_frame[
        fold_frame["model"] == "spline_residual_xgboost"
    ].set_index("fold")
    positive_folds = int((residual_folds["auc"] > baseline_folds["auc"]).sum())
    model_accepted = bool(
        residual_auc >= baseline_auc + MIN_OOF_GAIN
        and residual_weighted_auc >= baseline_weighted_auc + MIN_WEIGHTED_GAIN
        and positive_folds >= MIN_POSITIVE_FOLDS
    )

    blend_candidates = []
    for residual_weight in (0.25, 0.50, 0.75, 1.00):
        prediction = (
            (1.0 - residual_weight) * baseline_oof
            + residual_weight * residual_oof
        )
        ordinary = float(roc_auc_score(target, prediction))
        pressure = weighted_auc(target, prediction, drift_weights)
        blend_candidates.append(
            {
                "residual_weight": residual_weight,
                "oof_auc": ordinary,
                "drift_weighted_auc": pressure,
                "robust_score": 0.5 * (ordinary + pressure),
            }
        )
    blend_frame = pd.DataFrame(blend_candidates).sort_values(
        ["robust_score", "residual_weight"], ascending=[False, True]
    )
    winner = blend_frame.iloc[0]
    blend_accepted = bool(
        model_accepted
        and winner["oof_auc"] >= baseline_auc + MIN_OOF_GAIN
        and winner["drift_weighted_auc"]
        >= baseline_weighted_auc + MIN_WEIGHTED_GAIN
    )
    if blend_accepted:
        residual_weight = float(winner["residual_weight"])
        selected_oof = (
            (1.0 - residual_weight) * baseline_oof
            + residual_weight * residual_oof
        )
        selected_test = (
            (1.0 - residual_weight) * baseline_test
            + residual_weight * residual_test
        )
        selection_reason = "spline-residual model passed 3-fold screening guardrails"
    else:
        residual_weight = 0.0
        selected_oof = baseline_oof
        selected_test = baseline_test
        selection_reason = "screening fallback to 3-fold semantic XGBoost baseline"

    oof_frame = pd.DataFrame(
        {
            ID_COLUMN: train_ids,
            "fold": fold_registry,
            TARGET: target,
            "baseline_prediction": baseline_oof,
            "additive_prediction": additive_oof,
            "residual_prediction": residual_oof,
            "drift_weight": drift_weights,
            "selected_prediction": selected_oof,
        }
    )
    test_frame = pd.DataFrame(
        {
            ID_COLUMN: test_ids,
            "baseline_prediction": baseline_test,
            "additive_prediction": additive_test,
            "residual_prediction": residual_test,
            "selected_prediction": selected_test,
        }
    )
    oof_frame.to_csv(OUTPUT_DIR / "v8_oof.csv", index=False)
    test_frame.to_csv(OUTPUT_DIR / "v8_test_predictions.csv", index=False)
    fold_frame.to_csv(OUTPUT_DIR / "v8_fold_metrics.csv", index=False)
    blend_frame.to_csv(OUTPUT_DIR / "v8_blend_search.csv", index=False)
    pd.concat(coefficient_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "v8_spline_coefficients.csv", index=False
    )
    importance = pd.DataFrame(importance_rows)
    importance.groupby(["model", "feature"], as_index=False)["importance"].agg(
        mean="mean", std="std"
    ).sort_values(["model", "mean"], ascending=[True, False]).to_csv(
        OUTPUT_DIR / "v8_feature_importance.csv", index=False
    )

    prediction_columns = [
        "baseline_prediction",
        "additive_prediction",
        "residual_prediction",
    ]
    prediction_view = oof_frame[prediction_columns]
    prediction_view.corr().to_csv(OUTPUT_DIR / "v8_prediction_correlation.csv")
    prediction_view.rsub(target.to_numpy(), axis=0).corr().to_csv(
        OUTPUT_DIR / "v8_residual_correlation.csv"
    )
    segments = segment_metrics(base_train, target, baseline_oof, "baseline_xgboost")
    segments += segment_metrics(base_train, target, additive_oof, "spline_additive")
    segments += segment_metrics(
        base_train, target, residual_oof, "spline_residual_xgboost"
    )
    segments += segment_metrics(base_train, target, selected_oof, "selected")
    pd.DataFrame(segments).to_csv(OUTPUT_DIR / "v8_segment_metrics.csv", index=False)

    if sample_submission.columns.tolist() != [ID_COLUMN, TARGET]:
        raise ValueError("Unexpected sample_submission.csv columns.")
    if not sample_submission[ID_COLUMN].equals(test_ids):
        raise ValueError("sample_submission IDs are not aligned with test.csv.")
    sample_submission[TARGET] = np.clip(selected_test, 0.0, 1.0)
    if not np.isfinite(sample_submission[TARGET]).all():
        raise ValueError("Submission contains non-finite predictions.")
    sample_submission.to_csv(SUBMISSION_PATH, index=False)

    metrics = {
        "version": "v8_structural_screen",
        "baseline_oof_auc": baseline_auc,
        "spline_additive_oof_auc": additive_auc,
        "spline_residual_oof_auc": residual_auc,
        "spline_residual_gain": residual_auc - baseline_auc,
        "baseline_drift_weighted_auc": baseline_weighted_auc,
        "spline_residual_drift_weighted_auc": residual_weighted_auc,
        "spline_residual_drift_gain": (
            residual_weighted_auc - baseline_weighted_auc
        ),
        "positive_folds": positive_folds,
        "model_accepted": model_accepted,
        "selected_oof_auc": float(roc_auc_score(target, selected_oof)),
        "selected_drift_weighted_auc": weighted_auc(
            target, selected_oof, drift_weights
        ),
        "selected_residual_weight": residual_weight,
        "blend_accepted": blend_accepted,
        "selection_reason": selection_reason,
        "screening_only": True,
        "v3_preserved_separately": True,
        "outer_folds": N_FOLDS,
        "inner_folds": INNER_FOLDS,
        "spline_knots": SPLINE_KNOTS,
        "guardrails": {
            "min_oof_gain": MIN_OOF_GAIN,
            "min_weighted_gain": MIN_WEIGHTED_GAIN,
            "min_positive_folds": MIN_POSITIVE_FOLDS,
        },
        "seed": SEED,
        "xgb_device": xgb_device,
        "elapsed_minutes": (time.time() - started) / 60,
        "submission_path": str(SUBMISSION_PATH),
    }
    with (OUTPUT_DIR / "v8_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print("\nV8 structural screening complete", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
