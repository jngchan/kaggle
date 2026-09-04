"""Kaggle V2: extend only the semantic LightGBM convergence horizon.

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


SEED = 2026
N_FOLDS = 5
NATIVE_ESTIMATORS = 3000
SEMANTIC_ESTIMATORS = 4500
NATIVE_EARLY_STOPPING = 150
SEMANTIC_EARLY_STOPPING = 200
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
    base_train = build_base_features(raw_train)
    base_test = build_base_features(raw_test)
    print(
        f"train={len(base_train):,}, test={len(base_test):,}, "
        f"base_features={base_train.shape[1]}",
        flush=True,
    )

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(base_train, target))
    fold_registry = np.full(len(base_train), -1, dtype="int8")
    native_oof = np.zeros(len(base_train), dtype="float32")
    semantic_oof = np.zeros(len(base_train), dtype="float32")
    native_test = np.zeros(len(base_test), dtype="float64")
    semantic_test = np.zeros(len(base_test), dtype="float64")
    fold_rows = []
    importance_rows = []

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
            ]
        )
        for model_name, model, columns in (
            ("native", native_model, native_fit.columns),
            ("semantic", semantic_model, semantic_fit.columns),
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
            f"semantic_auc={semantic_auc:.7f}",
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
            imputer,
        )
        gc.collect()

    native_auc = roc_auc_score(target, native_oof)
    semantic_auc = roc_auc_score(target, semantic_oof)
    blend_candidates = []
    for native_weight in np.arange(0.0, 1.0001, 0.05):
        semantic_weight = 1.0 - native_weight
        prediction = native_weight * native_oof + semantic_weight * semantic_oof
        blend_candidates.append(
            {
                "native_weight": float(native_weight),
                "semantic_weight": float(semantic_weight),
                "oof_auc": float(roc_auc_score(target, prediction)),
            }
        )
    blend_frame = pd.DataFrame(blend_candidates).sort_values("oof_auc", ascending=False)
    winner = blend_frame.iloc[0]
    native_weight = float(winner["native_weight"])
    semantic_weight = float(winner["semantic_weight"])
    blend_oof = native_weight * native_oof + semantic_weight * semantic_oof
    blend_test = native_weight * native_test + semantic_weight * semantic_test

    oof_frame = pd.DataFrame(
        {
            ID_COLUMN: train_ids,
            "fold": fold_registry,
            TARGET: target,
            "native_prediction": native_oof,
            "semantic_prediction": semantic_oof,
            "blend_prediction": blend_oof,
        }
    )
    test_prediction_frame = pd.DataFrame(
        {
            ID_COLUMN: test_ids,
            "native_prediction": native_test,
            "semantic_prediction": semantic_test,
            "blend_prediction": blend_test,
        }
    )
    oof_frame.to_csv(OUTPUT_DIR / "v2_oof.csv", index=False)
    test_prediction_frame.to_csv(OUTPUT_DIR / "v2_test_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "v2_fold_metrics.csv", index=False)
    blend_frame.to_csv(OUTPUT_DIR / "v2_blend_search.csv", index=False)
    importance = pd.DataFrame(importance_rows)
    importance.groupby(["model", "feature"], as_index=False)["importance"].agg(
        mean="mean", std="std"
    ).sort_values(["model", "mean"], ascending=[True, False]).to_csv(
        OUTPUT_DIR / "v2_feature_importance.csv", index=False
    )
    segments = segment_metrics(base_train, target, native_oof, "native")
    segments += segment_metrics(base_train, target, semantic_oof, "semantic")
    segments += segment_metrics(base_train, target, blend_oof, "blend")
    pd.DataFrame(segments).to_csv(OUTPUT_DIR / "v2_segment_metrics.csv", index=False)

    if sample_submission.columns.tolist() != [ID_COLUMN, TARGET]:
        raise ValueError("Unexpected sample_submission.csv columns.")
    if not sample_submission[ID_COLUMN].equals(test_ids):
        raise ValueError("sample_submission IDs are not aligned with test.csv.")
    sample_submission[TARGET] = np.clip(blend_test, 0.0, 1.0)
    sample_submission.to_csv(SUBMISSION_PATH, index=False)

    metrics = {
        "version": "v2",
        "native_oof_auc": float(native_auc),
        "semantic_oof_auc": float(semantic_auc),
        "blend_oof_auc": float(winner["oof_auc"]),
        "native_weight": native_weight,
        "semantic_weight": semantic_weight,
        "folds": N_FOLDS,
        "seed": SEED,
        "native_estimators": NATIVE_ESTIMATORS,
        "semantic_estimators": SEMANTIC_ESTIMATORS,
        "native_early_stopping": NATIVE_EARLY_STOPPING,
        "semantic_early_stopping": SEMANTIC_EARLY_STOPPING,
        "elapsed_minutes": (time.time() - started) / 60,
        "submission_path": str(SUBMISSION_PATH),
    }
    with (OUTPUT_DIR / "v2_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print("\nV2 complete", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
