"""Shared deterministic features and fold-safe missing-value transformations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET = "addicted_label"
ID_COLUMN = "id"
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
SUPPORTED_MISSING_STRATEGIES = ("native", "median", "semantic")

# These fields have enough observed relationship to justify conditional regression.
REGRESSION_IMPUTATION_TARGETS = [
    "daily_screen_time_hours",
    "weekend_screen_time",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
]

# Observed competition ranges. Clipping prevents an imputer from creating impossible values.
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


@dataclass
class FeatureBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    target: pd.Series
    train_ids: pd.Series
    test_ids: pd.Series
    categorical_columns: list[str]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _add_features(frame: pd.DataFrame, add_missing_flags: bool = True) -> pd.DataFrame:
    """Add deterministic features without learning from labels or other rows."""
    result = frame.copy()
    original_features = [c for c in result.columns if c not in (ID_COLUMN, TARGET)]

    if add_missing_flags:
        for column in original_features:
            result[f"{column}__missing"] = result[column].isna().astype("int8")

    result["weekend_weekday_screen_diff"] = (
        result["weekend_screen_time"] - result["daily_screen_time_hours"]
    )
    result["weekend_weekday_screen_ratio"] = _safe_divide(
        result["weekend_screen_time"], result["daily_screen_time_hours"]
    )
    result["social_screen_share"] = _safe_divide(
        result["social_media_hours"], result["daily_screen_time_hours"]
    )
    result["gaming_screen_share"] = _safe_divide(
        result["gaming_hours"], result["daily_screen_time_hours"]
    )
    result["productive_screen_share"] = _safe_divide(
        result["work_study_hours"], result["daily_screen_time_hours"]
    )
    result["entertainment_hours"] = result["social_media_hours"] + result["gaming_hours"]
    result["entertainment_screen_share"] = _safe_divide(
        result["entertainment_hours"], result["daily_screen_time_hours"]
    )
    result["screen_sleep_ratio"] = _safe_divide(
        result["daily_screen_time_hours"], result["sleep_hours"]
    )
    result["weekend_screen_sleep_ratio"] = _safe_divide(
        result["weekend_screen_time"], result["sleep_hours"]
    )
    result["notifications_per_screen_hour"] = _safe_divide(
        result["notifications_per_day"], result["daily_screen_time_hours"]
    )
    result["opens_per_screen_hour"] = _safe_divide(
        result["app_opens_per_day"], result["daily_screen_time_hours"]
    )
    result["notifications_per_open"] = _safe_divide(
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


def build_features(train_path: str, test_path: str) -> FeatureBundle:
    """Load data and create the target-free, row-local baseline feature set."""
    raw_train = pd.read_csv(train_path)
    raw_test = pd.read_csv(test_path)

    missing = {TARGET, ID_COLUMN}.difference(raw_train.columns)
    if missing:
        raise ValueError(f"train data is missing required columns: {sorted(missing)}")
    if ID_COLUMN not in raw_test.columns:
        raise ValueError(f"test data is missing required column: {ID_COLUMN}")

    target = raw_train.pop(TARGET).astype("int8")
    train_ids = raw_train[ID_COLUMN].copy()
    test_ids = raw_test[ID_COLUMN].copy()
    train = _add_features(raw_train).drop(columns=ID_COLUMN)
    test = _add_features(raw_test).drop(columns=ID_COLUMN)

    for column in CATEGORICAL_COLUMNS:
        train[column] = train[column].fillna("Missing").astype(str)
        test[column] = test[column].fillna("Missing").astype(str)

    numeric_columns = train.select_dtypes(exclude="object").columns
    train[numeric_columns] = train[numeric_columns].astype("float32")
    test[numeric_columns] = test[numeric_columns].astype("float32")
    return FeatureBundle(train, test, target, train_ids, test_ids, CATEGORICAL_COLUMNS)


class FoldImputer:
    """Fit missing-value transformations on one outer training fold only."""

    def __init__(self, strategy: str, random_state: int = 2026):
        if strategy not in SUPPORTED_MISSING_STRATEGIES:
            raise ValueError(
                f"Unknown missing strategy '{strategy}'; choose {SUPPORTED_MISSING_STRATEGIES}."
            )
        self.strategy = strategy
        self.random_state = random_state
        self.medians: pd.Series | None = None
        self.predictor_columns: list[str] = []
        self.regressors: dict[str, object] = {}
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
            age, bins=[-np.inf, 21, 25, 29, 33, np.inf], labels=False
        ).astype("int8")
        context["screen_band"] = pd.cut(
            daily, bins=[-np.inf, 3, 6, 9, 12, np.inf], labels=False
        ).astype("int8")
        context["notification_band"] = pd.cut(
            notifications, bins=[-np.inf, 60, 110, 160, 210, np.inf], labels=False
        ).astype("int8")
        context["opens_band"] = pd.cut(
            opens, bins=[-np.inf, 50, 90, 130, 160, np.inf], labels=False
        ).astype("int8")
        return context

    def _predictor_frame(self, frame: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        predictors = frame[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
        for column in NUMERIC_COLUMNS:
            predictors[f"{column}__missing"] = predictors[column].isna().astype("int8")
            predictors[column] = predictors[column].fillna(self.medians[column])
        predictors = pd.get_dummies(
            predictors, columns=CATEGORICAL_COLUMNS, dummy_na=False, dtype="int8"
        ).astype("float32")
        if fit:
            self.predictor_columns = predictors.columns.tolist()
            return predictors
        return predictors.reindex(columns=self.predictor_columns, fill_value=0)

    def _fit_group_table(
        self, frame: pd.DataFrame, context: pd.DataFrame, target: str, keys: list[str]
    ) -> None:
        grouped = context[keys].copy()
        grouped[target] = frame[target]
        table = grouped.dropna(subset=[target]).groupby(keys, as_index=False)[target].median()
        self.group_tables[target] = (keys, table)

    def fit(self, frame: pd.DataFrame) -> "FoldImputer":
        self.medians = frame[NUMERIC_COLUMNS].median()
        if self.strategy != "semantic":
            return self

        from lightgbm import LGBMRegressor

        predictors = self._predictor_frame(frame, fit=True)
        rng = np.random.default_rng(self.random_state)
        for offset, target_column in enumerate(REGRESSION_IMPUTATION_TARGETS):
            observed_index = np.flatnonzero(frame[target_column].notna().to_numpy())
            if len(observed_index) > 400_000:
                observed_index = rng.choice(observed_index, size=400_000, replace=False)
            feature_columns = [
                column
                for column in self.predictor_columns
                if column not in (target_column, f"{target_column}__missing")
            ]
            regressor = LGBMRegressor(
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
            regressor.fit(
                predictors.iloc[observed_index][feature_columns],
                frame.iloc[observed_index][target_column],
            )
            self.regressors[target_column] = (feature_columns, regressor)

        context = self._context(frame)
        self._fit_group_table(frame, context, "age", ["gender"])
        self._fit_group_table(
            frame, context, "sleep_hours", ["age_band", "stress_level"]
        )
        self._fit_group_table(
            frame,
            context,
            "notifications_per_day",
            ["screen_band", "opens_band"],
        )
        self._fit_group_table(
            frame,
            context,
            "app_opens_per_day",
            ["screen_band", "notification_band"],
        )
        return self

    def _group_values(self, frame: pd.DataFrame, target: str) -> pd.Series:
        keys, table = self.group_tables[target]
        context = self._context(frame)
        mapped = context[keys].reset_index(drop=True).merge(
            table, on=keys, how="left", sort=False
        )[target]
        mapped.index = frame.index
        return mapped.fillna(self.medians[target])

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.medians is None:
            raise RuntimeError("FoldImputer.fit must be called before transform.")
        if self.strategy == "native":
            return frame.copy()

        # Recent pandas versions reject float64 predictions assigned into float32 blocks.
        imputed = frame[NUMERIC_COLUMNS].astype("float64").copy()
        if self.strategy == "median":
            imputed = imputed.fillna(self.medians)
        else:
            predictors = self._predictor_frame(frame)
            for target_column, (feature_columns, regressor) in self.regressors.items():
                missing = imputed[target_column].isna()
                if missing.any():
                    imputed.loc[missing, target_column] = regressor.predict(
                        predictors.loc[missing, feature_columns]
                    )
            for target_column in self.group_tables:
                missing = imputed[target_column].isna()
                if missing.any():
                    imputed.loc[missing, target_column] = self._group_values(
                        frame, target_column
                    ).loc[missing]
            imputed = imputed.fillna(self.medians)

        for column, (lower, upper) in VALUE_BOUNDS.items():
            imputed[column] = imputed[column].clip(lower, upper)
        imputed["notifications_per_day"] = imputed["notifications_per_day"].round()
        imputed["app_opens_per_day"] = imputed["app_opens_per_day"].round()

        companions = (
            _add_features(imputed, add_missing_flags=False)
            .astype("float32")
            .add_suffix(f"__{self.strategy}_imputed")
        )
        return pd.concat(
            [frame.reset_index(drop=True), companions.reset_index(drop=True)], axis=1
        )


def transform_fold(
    fit_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    strategy: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit preprocessing on an outer training fold and transform all three splits."""
    imputer = FoldImputer(strategy=strategy, random_state=random_state).fit(fit_frame)
    return (
        imputer.transform(fit_frame),
        imputer.transform(valid_frame),
        imputer.transform(test_frame),
    )


def encode_fold(
    fit_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    categorical_columns: list[str],
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Encode categories using only the outer training fold vocabulary."""
    if model_name == "catboost":
        outputs = []
        for frame in (fit_frame, valid_frame, test_frame):
            encoded = frame.copy()
            for column in categorical_columns:
                encoded[column] = encoded[column].fillna("Missing").astype(str)
            outputs.append(encoded)
        return outputs[0], outputs[1], outputs[2], categorical_columns

    fit_encoded = pd.get_dummies(
        fit_frame, columns=categorical_columns, dummy_na=False, dtype="int8"
    )
    valid_encoded = pd.get_dummies(
        valid_frame, columns=categorical_columns, dummy_na=False, dtype="int8"
    ).reindex(columns=fit_encoded.columns, fill_value=0)
    test_encoded = pd.get_dummies(
        test_frame, columns=categorical_columns, dummy_na=False, dtype="int8"
    ).reindex(columns=fit_encoded.columns, fill_value=0)
    return fit_encoded, valid_encoded, test_encoded, []
