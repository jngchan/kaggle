"""Kaggle V6: nested cross-fitted risk-table features.

This standalone Kaggle script uses only the competition-provided CSV files.
It rebuilds the V3 ensemble, trains one
additional XGBoost model with supervised risk features, and only uses that
model when fixed-CV guardrails are satisfied. Artifacts are written to
/kaggle/working/output and the selected submission to /kaggle/working/submission.csv.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


SEED = 2026
N_FOLDS = 5
NATIVE_ESTIMATORS = 3000
SEMANTIC_ESTIMATORS = 4500
XGB_ESTIMATORS = 3500
NATIVE_EARLY_STOPPING = 150
SEMANTIC_EARLY_STOPPING = 200
XGB_EARLY_STOPPING = 200
MIN_FEATURE_AUC_GAIN = 0.0001
MIN_POSITIVE_FOLDS = 4
MIN_BLEND_ROBUST_GAIN = 0.00005
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
MODEL_NAMES = ["native", "semantic", "xgb_baseline", "xgb_risk", "risk_gam"]
V3_WEIGHTS = {
    "native": 0.10,
    "semantic": 0.25,
    "xgb_baseline": 0.65,
    "xgb_risk": 0.0,
    "risk_gam": 0.0,
}


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
        if required_train.issubset(train_columns) and required_train.difference(
            {TARGET}
        ).issubset(test_columns):
            print(f"Using competition data from: {train_path.parent}", flush=True)
            return train_path, test_path, sample_path
    raise FileNotFoundError("Competition train/test/sample CSV files were not found.")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_row_features(frame: pd.DataFrame, add_missing_flags: bool = True) -> pd.DataFrame:
    """Reproduce the V3 feature set exactly."""
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
    result["entertainment_hours"] = (
        result["social_media_hours"] + result["gaming_hours"]
    )
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
    result["recovery_balance"] = (
        result["sleep_hours"] - result["daily_screen_time_hours"]
    )
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
        predictors = frame[ORIGINAL_COLUMNS].copy()
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
        table = (
            values.dropna(subset=[target])
            .groupby(keys, as_index=False)[target]
            .median()
        )
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
            model.fit(
                predictors.iloc[observed][columns],
                frame.iloc[observed][target_column],
            )
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
        prediction = self._context(frame)[keys].reset_index(drop=True).merge(
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


class CrossFittedRiskEncoder:
    """Learn smoothed one- and two-dimensional risk tables without leakage."""

    RISK_NUMERIC = [
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
    PAIRS = [
        ("daily_screen_time_hours", "social_media_hours"),
        ("daily_screen_time_hours", "weekend_screen_time"),
        ("social_media_hours", "gaming_hours"),
        ("daily_screen_time_hours", "sleep_hours"),
        ("notifications_per_day", "app_opens_per_day"),
        ("daily_screen_time_hours", "work_study_hours"),
    ]

    def __init__(self, seed: int, inner_folds: int = 5):
        self.seed = seed
        self.inner_folds = inner_folds
        self.prior = 0.5
        self.edges = {}
        self.mappings = {}
        self.feature_names = []

    @staticmethod
    def _source_name(column: str) -> str:
        return f"{column}__semantic_imputed"

    def _fit_edges(self, frame: pd.DataFrame) -> None:
        quantiles = np.linspace(0.0, 1.0, 65)
        for column in self.RISK_NUMERIC:
            values = frame[self._source_name(column)].to_numpy(dtype="float64")
            edges = np.unique(np.quantile(values, quantiles))
            if len(edges) < 3:
                edges = np.array([-np.inf, np.inf])
            else:
                edges[0] = -np.inf
                edges[-1] = np.inf
            self.edges[column] = edges

    def _bin(self, frame: pd.DataFrame, column: str) -> np.ndarray:
        values = frame[self._source_name(column)].to_numpy(dtype="float64")
        return np.searchsorted(self.edges[column][1:-1], values, side="right").astype(
            "int16"
        )

    def _keys(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        bins = {column: self._bin(frame, column) for column in self.RISK_NUMERIC}
        keys = {f"te_{column}": values for column, values in bins.items()}
        for left, right in self.PAIRS:
            left_coarse = bins[left] // 4
            right_coarse = bins[right] // 4
            keys[f"te_{left}__x__{right}"] = (
                left_coarse.astype("int32") * 100 + right_coarse.astype("int32")
            )
        for column in CATEGORICAL_COLUMNS:
            keys[f"te_{column}"] = frame[column].fillna("Missing").astype(str).to_numpy()
        keys["te_stress__x__impact"] = (
            frame["stress_level"].fillna("Missing").astype(str)
            + "|"
            + frame["academic_work_impact"].fillna("Missing").astype(str)
        ).to_numpy()
        return keys

    @staticmethod
    def _fit_mapping(keys, target, prior, alpha):
        stats = pd.DataFrame({"key": keys, "target": np.asarray(target)}).groupby(
            "key", sort=False
        )["target"].agg(["sum", "count"])
        return (stats["sum"] + alpha * prior) / (stats["count"] + alpha)

    @staticmethod
    def _apply_mapping(keys, mapping, prior):
        return pd.Series(keys).map(mapping).fillna(prior).to_numpy(dtype="float32")

    @staticmethod
    def _alpha(feature_name: str) -> float:
        return 200.0 if "__x__" in feature_name else 80.0

    def fit_transform(self, frame: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        self.prior = float(np.mean(target))
        self._fit_edges(frame)
        keys = self._keys(frame)
        self.feature_names = list(keys)
        encoded = pd.DataFrame(index=np.arange(len(frame)))
        splitter = StratifiedKFold(
            n_splits=self.inner_folds,
            shuffle=True,
            random_state=self.seed,
        )
        target_array = np.asarray(target)
        splits = list(splitter.split(np.zeros(len(frame)), target_array))
        for feature_name, feature_keys in keys.items():
            values = np.zeros(len(frame), dtype="float32")
            alpha = self._alpha(feature_name)
            for fit_index, valid_index in splits:
                inner_prior = float(target_array[fit_index].mean())
                mapping = self._fit_mapping(
                    feature_keys[fit_index],
                    target_array[fit_index],
                    inner_prior,
                    alpha,
                )
                values[valid_index] = self._apply_mapping(
                    feature_keys[valid_index], mapping, inner_prior
                )
            encoded[feature_name] = values
            self.mappings[feature_name] = self._fit_mapping(
                feature_keys, target_array, self.prior, alpha
            )
        return encoded

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        keys = self._keys(frame)
        return pd.DataFrame(
            {
                feature_name: self._apply_mapping(
                    keys[feature_name], self.mappings[feature_name], self.prior
                )
                for feature_name in self.feature_names
            }
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


def make_lgb_classifier(seed: int, n_estimators: int) -> lgb.LGBMClassifier:
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


def fit_lgb(model, fit_x, fit_y, valid_x, valid_y, early_stopping_rounds):
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


def missing_pattern(frame: pd.DataFrame) -> pd.Series:
    mask = frame[ORIGINAL_COLUMNS].isna().to_numpy(dtype="int16")
    powers = (1 << np.arange(len(ORIGINAL_COLUMNS), dtype="int16"))
    return pd.Series(mask @ powers, index=frame.index, dtype="int16")


def make_drift_weights(raw_train: pd.DataFrame, raw_test: pd.DataFrame):
    """Match OOF evaluation to test missing-pattern frequencies without labels."""
    train_pattern = missing_pattern(raw_train)
    test_pattern = missing_pattern(raw_test)
    patterns = np.union1d(train_pattern.unique(), test_pattern.unique())
    train_counts = train_pattern.value_counts().reindex(patterns, fill_value=0)
    test_counts = test_pattern.value_counts().reindex(patterns, fill_value=0)
    smoothing = 20.0
    train_rate = (train_counts + smoothing) / (
        len(train_pattern) + smoothing * len(patterns)
    )
    test_rate = (test_counts + smoothing) / (
        len(test_pattern) + smoothing * len(patterns)
    )
    ratio = (test_rate / train_rate).clip(0.25, 4.0)
    weights = train_pattern.map(ratio).astype("float64")
    weights /= weights.mean()
    report = pd.DataFrame(
        {
            "pattern": patterns,
            "missing_columns": [
                "+".join(
                    column
                    for bit, column in enumerate(ORIGINAL_COLUMNS)
                    if int(pattern) & (1 << bit)
                )
                or "none"
                for pattern in patterns
            ],
            "train_rows": train_counts.to_numpy(),
            "test_rows": test_counts.to_numpy(),
            "train_rate": train_rate.to_numpy(),
            "test_rate": test_rate.to_numpy(),
            "raw_ratio": (test_rate / train_rate).to_numpy(),
            "clipped_ratio": ratio.to_numpy(),
        }
    ).sort_values("test_rows", ascending=False)
    return weights.to_numpy(), report


def auc_pair(target, prediction, weights):
    return (
        float(roc_auc_score(target, prediction)),
        float(roc_auc_score(target, prediction, sample_weight=weights)),
    )


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype="float64")


def combine(predictions: dict[str, np.ndarray], weights: dict[str, float]):
    return sum(weights[name] * predictions[name] for name in MODEL_NAMES)


def segment_metrics(base_features, target, prediction, model_name, drift_weights):
    missing_columns = [
        column
        for column in base_features
        if column.endswith("__missing")
        and column.removesuffix("__missing") in base_features
    ]
    missing_count = base_features[missing_columns].sum(axis=1)
    rows = []

    def add(name, mask):
        mask_array = mask.to_numpy()
        subset = target.loc[mask]
        if len(subset) and subset.nunique() == 2:
            ordinary, weighted = auc_pair(
                subset,
                prediction[mask_array],
                drift_weights[mask_array],
            )
            rows.append(
                {
                    "model": model_name,
                    "segment": name,
                    "rows": int(mask.sum()),
                    "auc": ordinary,
                    "drift_weighted_auc": weighted,
                }
            )

    for count in range(4):
        add(f"missing_count_{count}", missing_count == count)
    add("missing_count_4plus", missing_count >= 4)
    for column in missing_columns:
        add(column, base_features[column].astype(bool))
    return rows


def weight_compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for value in range(total + 1):
        for remainder in weight_compositions(total - value, parts - 1):
            yield (value, *remainder)


def search_blends(target, oof, test, drift_weights, allow_risk):
    """Use a coarse grid and a balanced standard/drift-weighted objective."""
    oof_views = {
        "raw": oof,
        "rank": {name: percentile_rank(values) for name, values in oof.items()},
    }
    test_views = {
        "raw": test,
        "rank": {name: percentile_rank(values) for name, values in test.items()},
    }
    rows = []
    # A coarse grid is deliberate: fine OOF weight fitting is unlikely to
    # transfer reliably to the private leaderboard.
    step_units = 10
    for blend_type in ("raw", "rank"):
        for units in weight_compositions(step_units, len(MODEL_NAMES)):
            weights = {
                name: units[index] / step_units
                for index, name in enumerate(MODEL_NAMES)
            }
            if not allow_risk and (weights["xgb_risk"] + weights["risk_gam"] > 0):
                continue
            prediction = combine(oof_views[blend_type], weights)
            ordinary, weighted = auc_pair(target, prediction, drift_weights)
            rows.append(
                {
                    "blend_type": blend_type,
                    **{f"{name}_weight": weights[name] for name in MODEL_NAMES},
                    "oof_auc": ordinary,
                    "drift_weighted_oof_auc": weighted,
                    "robust_auc": 0.5 * (ordinary + weighted),
                }
            )
        if allow_risk:
            for risk_weight in (0.05, 0.10, 0.15, 0.20):
                for risk_mix in ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5)):
                    weights = {
                        name: V3_WEIGHTS[name] * (1.0 - risk_weight)
                        for name in MODEL_NAMES
                    }
                    weights["xgb_risk"] += risk_weight * risk_mix[0]
                    weights["risk_gam"] += risk_weight * risk_mix[1]
                    prediction = combine(oof_views[blend_type], weights)
                    ordinary, weighted = auc_pair(target, prediction, drift_weights)
                    rows.append(
                        {
                            "blend_type": blend_type,
                            **{f"{name}_weight": weights[name] for name in MODEL_NAMES},
                            "oof_auc": ordinary,
                            "drift_weighted_oof_auc": weighted,
                            "robust_auc": 0.5 * (ordinary + weighted),
                        }
                    )
    frame = pd.DataFrame(rows).sort_values(
        ["robust_auc", "oof_auc"], ascending=False
    )
    winner = frame.iloc[0]
    winner_weights = {
        name: float(winner[f"{name}_weight"]) for name in MODEL_NAMES
    }
    winner_test = combine(test_views[str(winner["blend_type"])], winner_weights)
    return frame, winner_test


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
    drift_weights, drift_report = make_drift_weights(raw_train, raw_test)
    drift_report.to_csv(OUTPUT_DIR / "v6_missing_drift_report.csv", index=False)
    field_drift_rows = []
    for column in ORIGINAL_COLUMNS:
        row = {
            "feature": column,
            "train_missing_rate": float(raw_train[column].isna().mean()),
            "test_missing_rate": float(raw_test[column].isna().mean()),
        }
        row["missing_rate_delta"] = (
            row["test_missing_rate"] - row["train_missing_rate"]
        )
        if column in NUMERIC_COLUMNS:
            row["train_nonmissing_mean"] = float(raw_train[column].mean())
            row["test_nonmissing_mean"] = float(raw_test[column].mean())
        field_drift_rows.append(row)
    pd.DataFrame(field_drift_rows).to_csv(
        OUTPUT_DIR / "v6_field_drift_report.csv", index=False
    )
    base_train = build_base_features(raw_train)
    base_test = build_base_features(raw_test)
    xgb_device = "cpu"
    print(
        f"train={len(base_train):,}, test={len(base_test):,}, "
        f"base_features={base_train.shape[1]}, xgb_device={xgb_device}",
        flush=True,
    )

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(base_train, target))
    fold_registry = np.full(len(base_train), -1, dtype="int8")
    oof = {name: np.zeros(len(base_train), dtype="float32") for name in MODEL_NAMES}
    test_predictions = {
        name: np.zeros(len(base_test), dtype="float64") for name in MODEL_NAMES
    }
    fold_rows = []
    importance_rows = []
    gam_coefficient_rows = []

    for fold, (fit_index, valid_index) in enumerate(splits, start=1):
        fold_registry[valid_index] = fold - 1
        fold_fit = base_train.iloc[fit_index].reset_index(drop=True)
        fold_valid = base_train.iloc[valid_index].reset_index(drop=True)
        fit_y = target.iloc[fit_index]
        valid_y = target.iloc[valid_index]
        valid_drift_weights = drift_weights[valid_index]

        print(f"\n===== fold {fold}/{N_FOLDS}: native LightGBM =====", flush=True)
        native_fit, native_valid, native_test_x = encode_from_fold_vocabulary(
            fold_fit, fold_valid, base_test
        )
        native_model = make_lgb_classifier(SEED + fold, NATIVE_ESTIMATORS)
        fit_lgb(
            native_model,
            native_fit,
            fit_y,
            native_valid,
            valid_y,
            NATIVE_EARLY_STOPPING,
        )

        print(f"===== fold {fold}/{N_FOLDS}: semantic imputation =====", flush=True)
        imputer = SemanticFoldImputer(SEED + fold).fit(fold_fit)
        semantic_fit_frame = imputer.transform(fold_fit)
        semantic_valid_frame = imputer.transform(fold_valid)
        semantic_test_frame = imputer.transform(base_test)
        semantic_fit, semantic_valid, semantic_test_x = encode_from_fold_vocabulary(
            semantic_fit_frame, semantic_valid_frame, semantic_test_frame
        )

        print(f"===== fold {fold}/{N_FOLDS}: semantic LightGBM =====", flush=True)
        semantic_model = make_lgb_classifier(SEED + fold, SEMANTIC_ESTIMATORS)
        fit_lgb(
            semantic_model,
            semantic_fit,
            fit_y,
            semantic_valid,
            valid_y,
            SEMANTIC_EARLY_STOPPING,
        )

        print(f"===== fold {fold}/{N_FOLDS}: baseline XGBoost =====", flush=True)
        baseline_model = make_xgb_classifier(SEED + fold, xgb_device)
        baseline_model.fit(
            semantic_fit,
            fit_y,
            eval_set=[(semantic_valid, valid_y)],
            verbose=200,
        )

        print(f"===== fold {fold}/{N_FOLDS}: cross-fitted risk tables =====", flush=True)
        risk_encoder = CrossFittedRiskEncoder(SEED + 500 + fold)
        risk_fit = risk_encoder.fit_transform(
            semantic_fit_frame, fit_y.reset_index(drop=True)
        )
        risk_valid = risk_encoder.transform(semantic_valid_frame)
        risk_test = risk_encoder.transform(semantic_test_frame)
        for name, frame in (
            ("fit", risk_fit),
            ("valid", risk_valid),
            ("test", risk_test),
        ):
            if not np.isfinite(frame.to_numpy()).all():
                raise ValueError(f"Risk features contain non-finite values in {name}.")

        print(f"===== fold {fold}/{N_FOLDS}: risk GAM =====", flush=True)
        risk_gam = LogisticRegression(
            C=0.2,
            max_iter=500,
            solver="lbfgs",
            random_state=SEED + fold,
        )
        risk_gam.fit(risk_fit, fit_y)
        gam_valid_prediction = risk_gam.predict_proba(risk_valid)[:, 1]
        gam_test_prediction = risk_gam.predict_proba(risk_test)[:, 1]
        gam_coefficient_rows.extend(
            {
                "fold": fold,
                "feature": feature,
                "coefficient": coefficient,
            }
            for feature, coefficient in zip(risk_fit.columns, risk_gam.coef_[0])
        )

        risk_xgb_fit = semantic_fit.copy()
        risk_xgb_valid = semantic_valid.copy()
        risk_xgb_test = semantic_test_x.copy()
        for column in risk_fit.columns:
            risk_xgb_fit[column] = risk_fit[column].to_numpy()
            risk_xgb_valid[column] = risk_valid[column].to_numpy()
            risk_xgb_test[column] = risk_test[column].to_numpy()
        print(f"===== fold {fold}/{N_FOLDS}: risk-feature XGBoost =====", flush=True)
        risk_model = make_xgb_classifier(SEED + fold, xgb_device)
        risk_model.fit(
            risk_xgb_fit,
            fit_y,
            eval_set=[(risk_xgb_valid, valid_y)],
            verbose=200,
        )

        models = {
            "native": (native_model, native_valid, native_test_x),
            "semantic": (semantic_model, semantic_valid, semantic_test_x),
            "xgb_baseline": (baseline_model, semantic_valid, semantic_test_x),
            "xgb_risk": (risk_model, risk_xgb_valid, risk_xgb_test),
        }
        for model_name, (model, valid_x, test_x) in models.items():
            prediction = model.predict_proba(valid_x)[:, 1]
            oof[model_name][valid_index] = prediction
            test_predictions[model_name] += (
                model.predict_proba(test_x)[:, 1] / N_FOLDS
            )
            ordinary, weighted = auc_pair(valid_y, prediction, valid_drift_weights)
            best_iteration = (
                model.best_iteration_
                if model_name in ("native", "semantic")
                else model.best_iteration
            )
            fold_rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "auc": ordinary,
                    "drift_weighted_auc": weighted,
                    "best_iteration": best_iteration,
                }
            )
            for feature, importance in zip(valid_x.columns, model.feature_importances_):
                importance_rows.append(
                    {
                        "fold": fold,
                        "model": model_name,
                        "feature": feature,
                        "importance": importance,
                    }
                )
        oof["risk_gam"][valid_index] = gam_valid_prediction
        test_predictions["risk_gam"] += gam_test_prediction / N_FOLDS
        gam_auc, gam_weighted_auc = auc_pair(
            valid_y, gam_valid_prediction, valid_drift_weights
        )
        fold_rows.append(
            {
                "fold": fold,
                "model": "risk_gam",
                "auc": gam_auc,
                "drift_weighted_auc": gam_weighted_auc,
                "best_iteration": int(risk_gam.n_iter_[0]),
            }
        )
        risk_auc = fold_rows[-1]["auc"]
        baseline_auc = next(
            row["auc"]
            for row in reversed(fold_rows)
            if row["fold"] == fold and row["model"] == "xgb_baseline"
        )
        risk_xgb_auc = next(
            row["auc"]
            for row in reversed(fold_rows)
            if row["fold"] == fold and row["model"] == "xgb_risk"
        )
        print(
            f"fold {fold}: baseline_xgb={baseline_auc:.7f}, "
            f"risk_xgb={risk_xgb_auc:.7f}, delta={risk_xgb_auc-baseline_auc:+.7f}, "
            f"risk_gam={risk_auc:.7f}",
            flush=True,
        )
        del (
            native_fit,
            native_valid,
            native_test_x,
            semantic_fit_frame,
            semantic_valid_frame,
            semantic_test_frame,
            semantic_fit,
            semantic_valid,
            semantic_test_x,
            risk_fit,
            risk_valid,
            risk_test,
            risk_xgb_fit,
            risk_xgb_valid,
            risk_xgb_test,
            native_model,
            semantic_model,
            baseline_model,
            risk_model,
            risk_gam,
            risk_encoder,
            imputer,
        )
        gc.collect()

    model_metrics = {}
    for model_name in MODEL_NAMES:
        ordinary, weighted = auc_pair(target, oof[model_name], drift_weights)
        model_metrics[model_name] = {
            "oof_auc": ordinary,
            "drift_weighted_oof_auc": weighted,
        }
    fold_frame = pd.DataFrame(fold_rows)
    baseline_folds = fold_frame[fold_frame["model"] == "xgb_baseline"].set_index("fold")
    risk_folds = fold_frame[fold_frame["model"] == "xgb_risk"].set_index("fold")
    ablation = pd.DataFrame(
        {
            "fold": baseline_folds.index,
            "baseline_auc": baseline_folds["auc"],
            "risk_auc": risk_folds["auc"],
            "auc_delta": risk_folds["auc"] - baseline_folds["auc"],
            "baseline_drift_weighted_auc": baseline_folds["drift_weighted_auc"],
            "risk_drift_weighted_auc": risk_folds["drift_weighted_auc"],
            "drift_weighted_auc_delta": (
                risk_folds["drift_weighted_auc"]
                - baseline_folds["drift_weighted_auc"]
            ),
        }
    ).reset_index(drop=True)
    overall_gain = (
        model_metrics["xgb_risk"]["oof_auc"]
        - model_metrics["xgb_baseline"]["oof_auc"]
    )
    weighted_gain = (
        model_metrics["xgb_risk"]["drift_weighted_oof_auc"]
        - model_metrics["xgb_baseline"]["drift_weighted_oof_auc"]
    )
    positive_folds = int((ablation["auc_delta"] > 0).sum())
    feature_accepted = bool(
        overall_gain >= MIN_FEATURE_AUC_GAIN
        and weighted_gain >= 0.0
        and positive_folds >= MIN_POSITIVE_FOLDS
    )

    v3_oof = combine(oof, V3_WEIGHTS)
    v3_test = combine(test_predictions, V3_WEIGHTS)
    v3_auc, v3_weighted_auc = auc_pair(target, v3_oof, drift_weights)
    v3_robust_auc = 0.5 * (v3_auc + v3_weighted_auc)
    blend_frame, candidate_test = search_blends(
        target, oof, test_predictions, drift_weights, True
    )
    winner = blend_frame.iloc[0]
    blend_accepted = bool(
        (winner["xgb_risk_weight"] + winner["risk_gam_weight"] > 0.0)
        and winner["robust_auc"] >= v3_robust_auc + MIN_BLEND_ROBUST_GAIN
        and winner["oof_auc"] >= v3_auc + 0.00005
        and winner["drift_weighted_oof_auc"] >= v3_weighted_auc + 0.00005
    )
    if blend_accepted:
        final_oof_view = (
            oof
            if winner["blend_type"] == "raw"
            else {name: percentile_rank(values) for name, values in oof.items()}
        )
        final_weights = {
            name: float(winner[f"{name}_weight"]) for name in MODEL_NAMES
        }
        final_oof = combine(final_oof_view, final_weights)
        final_test = candidate_test
        selection_reason = "cross-fitted risk features passed all CV guardrails"
    else:
        final_oof = v3_oof
        final_test = v3_test
        final_weights = V3_WEIGHTS.copy()
        selection_reason = "CV guardrail fallback to fixed V3 blend"
    final_auc, final_weighted_auc = auc_pair(target, final_oof, drift_weights)

    fold_frame.to_csv(OUTPUT_DIR / "v6_fold_metrics.csv", index=False)
    ablation.to_csv(OUTPUT_DIR / "v6_risk_ablation.csv", index=False)
    blend_frame.to_csv(OUTPUT_DIR / "v6_blend_search.csv", index=False)
    importance = pd.DataFrame(importance_rows)
    importance.groupby(["model", "feature"], as_index=False)["importance"].agg(
        mean="mean", std="std"
    ).sort_values(["model", "mean"], ascending=[True, False]).to_csv(
        OUTPUT_DIR / "v6_feature_importance.csv", index=False
    )
    gam_coefficients = pd.DataFrame(gam_coefficient_rows)
    gam_coefficients.groupby("feature", as_index=False)["coefficient"].agg(
        mean="mean", std="std"
    ).sort_values("mean", key=lambda values: values.abs(), ascending=False).to_csv(
        OUTPUT_DIR / "v6_risk_gam_coefficients.csv", index=False
    )
    segments = []
    for model_name in MODEL_NAMES:
        segments += segment_metrics(
            base_train, target, oof[model_name], model_name, drift_weights
        )
    segments += segment_metrics(
        base_train, target, final_oof, "selected_blend", drift_weights
    )
    pd.DataFrame(segments).to_csv(
        OUTPUT_DIR / "v6_segment_metrics.csv", index=False
    )
    prediction_frame = pd.DataFrame(oof)
    prediction_frame.corr().to_csv(OUTPUT_DIR / "v6_prediction_correlation.csv")
    prediction_frame.apply(lambda values: target - values).corr().to_csv(
        OUTPUT_DIR / "v6_residual_correlation.csv"
    )
    pd.DataFrame(
        {
            ID_COLUMN: train_ids,
            "fold": fold_registry,
            TARGET: target,
            "drift_weight": drift_weights,
            **{f"{name}_prediction": values for name, values in oof.items()},
            "v3_fixed_prediction": v3_oof,
            "selected_prediction": final_oof,
        }
    ).to_csv(OUTPUT_DIR / "v6_oof.csv", index=False)
    pd.DataFrame(
        {
            ID_COLUMN: test_ids,
            **{
                f"{name}_prediction": values
                for name, values in test_predictions.items()
            },
            "v3_fixed_prediction": v3_test,
            "selected_prediction": final_test,
        }
    ).to_csv(OUTPUT_DIR / "v6_test_predictions.csv", index=False)

    if sample_submission.columns.tolist() != [ID_COLUMN, TARGET]:
        raise ValueError("Unexpected sample_submission.csv columns.")
    if not sample_submission[ID_COLUMN].equals(test_ids):
        raise ValueError("sample_submission IDs are not aligned with test.csv.")
    if not np.isfinite(final_test).all():
        raise ValueError("Selected test predictions contain NaN or infinity.")
    sample_submission[TARGET] = np.clip(final_test, 0.0, 1.0)
    sample_submission.to_csv(SUBMISSION_PATH, index=False)

    metrics = {
        "version": "v6",
        "models": model_metrics,
        "risk_vs_baseline_oof_gain": float(overall_gain),
        "risk_vs_baseline_drift_weighted_gain": float(weighted_gain),
        "risk_positive_folds": positive_folds,
        "feature_accepted": feature_accepted,
        "v3_fixed_oof_auc": float(v3_auc),
        "v3_fixed_drift_weighted_oof_auc": float(v3_weighted_auc),
        "selected_oof_auc": float(final_auc),
        "selected_drift_weighted_oof_auc": float(final_weighted_auc),
        "selected_weights": final_weights,
        "selected_blend_type": (
            str(winner["blend_type"]) if blend_accepted else "raw"
        ),
        "blend_accepted": blend_accepted,
        "selection_reason": selection_reason,
        "folds": N_FOLDS,
        "seed": SEED,
        "xgb_device": xgb_device,
        "elapsed_minutes": (time.time() - started) / 60,
        "submission_path": str(SUBMISSION_PATH),
    }
    with (OUTPUT_DIR / "v6_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print("\nV6 complete", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
