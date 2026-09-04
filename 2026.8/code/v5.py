"""Kaggle V5: domain audit and masked-augmentation neural diversity.

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
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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
MLP_MAX_EPOCHS = 25
MLP_PATIENCE = 4
MLP_BATCH_SIZE = 4096
MASKED_AUGMENTATION_ROWS = 180_000
MIN_DEEP_BLEND_ROBUST_GAIN = 0.0001
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
ORIGINAL_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
MODEL_NAMES = ["native", "semantic", "xgboost", "mlp_plain", "mlp_masked"]
V3_WEIGHTS = {
    "native": 0.10,
    "semantic": 0.25,
    "xgboost": 0.65,
    "mlp_plain": 0.0,
    "mlp_masked": 0.0,
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


class TabularMLP(nn.Module):
    """Small smooth model intended to complement tree-based rankings."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, values):
        return self.network(values).squeeze(1)


def set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_mlp_scaler(frame: pd.DataFrame):
    mean = frame.mean(axis=0).fillna(0.0)
    std = frame.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return mean, std


def transform_mlp(frame: pd.DataFrame, mean, std) -> np.ndarray:
    transformed = (frame.fillna(mean) - mean) / std
    return transformed.clip(-10.0, 10.0).to_numpy(dtype="float32")


def predict_mlp(model, values, device) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)),
        batch_size=MLP_BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device, non_blocking=True))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(predictions).astype("float32")


def fit_tabular_mlp(
    train_x,
    train_y,
    valid_x,
    valid_y,
    test_x,
    seed,
    label,
):
    set_torch_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMLP(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate=1e-3, weight_decay=1e-4
    )
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_y.astype("float32")),
        ),
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    valid_y_array = np.asarray(valid_y)
    for epoch in range(1, MLP_MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(batch_x)
            seen += len(batch_x)
        valid_prediction = predict_mlp(model, valid_x, device)
        valid_auc = roc_auc_score(valid_y_array, valid_prediction)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / seen,
                "valid_auc": float(valid_auc),
            }
        )
        print(
            f"  {label} epoch={epoch:02d} loss={running_loss/seen:.6f} "
            f"valid_auc={valid_auc:.7f}",
            flush=True,
        )
        if valid_auc > best_auc + 1e-7:
            best_auc = float(valid_auc)
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= MLP_PATIENCE:
                break
    model.load_state_dict(best_state)
    valid_prediction = predict_mlp(model, valid_x, device)
    test_prediction = predict_mlp(model, test_x, device)
    del model, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return valid_prediction, test_prediction, best_auc, best_epoch, history


def make_masked_augmentation(raw_fit, raw_test, fit_y, seed):
    """Mask complete fit rows using test patterns; labels never leave the fit fold."""
    complete = raw_fit[ORIGINAL_COLUMNS].notna().all(axis=1).to_numpy()
    candidates = np.flatnonzero(complete)
    if not len(candidates):
        raise ValueError("No complete rows are available for masked augmentation.")
    rng = np.random.default_rng(seed)
    sample_size = min(MASKED_AUGMENTATION_ROWS, len(candidates))
    selected = rng.choice(candidates, size=sample_size, replace=False)
    test_masks = raw_test[ORIGINAL_COLUMNS].isna().to_numpy(dtype=bool)
    test_masks = test_masks[test_masks.any(axis=1)]
    sampled_masks = test_masks[
        rng.integers(0, len(test_masks), size=sample_size)
    ]
    augmented = raw_fit.iloc[selected].copy().reset_index(drop=True)
    for column_index, column in enumerate(ORIGINAL_COLUMNS):
        augmented.loc[sampled_masks[:, column_index], column] = np.nan
    augmented_y = np.asarray(fit_y)[selected].astype("float32")
    return augmented, augmented_y


def adversarial_validation(base_train, base_test, output_dir):
    """Return OOF train-domain scores and density-ratio pressure weights."""
    combined = pd.concat([base_train, base_test], ignore_index=True)
    combined = pd.get_dummies(
        combined, columns=CATEGORICAL_COLUMNS, dtype="int8"
    )
    domain_target = np.concatenate(
        [np.zeros(len(base_train), dtype="int8"), np.ones(len(base_test), dtype="int8")]
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 99)
    domain_oof = np.zeros(len(combined), dtype="float32")
    fold_rows = []
    importance_rows = []
    for fold, (fit_index, valid_index) in enumerate(
        splitter.split(combined, domain_target), start=1
    ):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=200,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            random_state=SEED + 99 + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            combined.iloc[fit_index],
            domain_target[fit_index],
            eval_set=[(combined.iloc[valid_index], domain_target[valid_index])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        prediction = model.predict_proba(combined.iloc[valid_index])[:, 1]
        domain_oof[valid_index] = prediction
        fold_rows.append(
            {
                "fold": fold,
                "auc": roc_auc_score(domain_target[valid_index], prediction),
                "best_iteration": model.best_iteration_,
            }
        )
        importance_rows.extend(
            {
                "fold": fold,
                "feature": feature,
                "importance": importance,
            }
            for feature, importance in zip(combined.columns, model.feature_importances_)
        )
    domain_auc = float(roc_auc_score(domain_target, domain_oof))
    train_domain_score = domain_oof[: len(base_train)].astype("float64")
    prior_adjustment = len(base_test) / len(base_train)
    odds = train_domain_score / np.clip(1.0 - train_domain_score, 1e-4, None)
    weights = (odds / prior_adjustment).clip(0.25, 4.0)
    weights /= weights.mean()
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "v5_domain_fold_metrics.csv", index=False
    )
    importance = pd.DataFrame(importance_rows)
    importance.groupby("feature", as_index=False)["importance"].agg(
        mean="mean", std="std"
    ).sort_values("mean", ascending=False).to_csv(
        output_dir / "v5_domain_feature_importance.csv", index=False
    )
    del combined, domain_target, domain_oof, model, importance
    gc.collect()
    return train_domain_score, weights, domain_auc


def auc_pair(target, prediction, weights):
    return (
        float(roc_auc_score(target, prediction)),
        float(roc_auc_score(target, prediction, sample_weight=weights)),
    )


def percentile_rank(values):
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def combine_predictions(predictions, weights):
    return sum(weights[name] * predictions[name] for name in MODEL_NAMES)


def weight_compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for value in range(total + 1):
        for remainder in weight_compositions(total - value, parts - 1):
            yield (value, *remainder)


def search_robust_blend(target, oof, test, pressure_weights):
    steps = 10
    oof_views = {
        "raw": oof,
        "rank": {name: percentile_rank(values) for name, values in oof.items()},
    }
    test_views = {
        "raw": test,
        "rank": {name: percentile_rank(values) for name, values in test.items()},
    }
    rows = []
    for blend_type in ("raw", "rank"):
        for units in weight_compositions(steps, len(MODEL_NAMES)):
            weights = {
                name: units[index] / steps
                for index, name in enumerate(MODEL_NAMES)
            }
            prediction = combine_predictions(oof_views[blend_type], weights)
            ordinary, pressure = auc_pair(target, prediction, pressure_weights)
            rows.append(
                {
                    "blend_type": blend_type,
                    **{f"{name}_weight": weights[name] for name in MODEL_NAMES},
                    "oof_auc": ordinary,
                    "test_like_oof_auc": pressure,
                    "robust_auc": 0.5 * (ordinary + pressure),
                }
            )
        # A few predeclared low-weight deep blends test diversity without a
        # dense continuous weight search.
        for deep_weight in (0.05, 0.10, 0.15, 0.20):
            for deep_mix in ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5)):
                weights = {
                    name: V3_WEIGHTS[name] * (1.0 - deep_weight)
                    for name in MODEL_NAMES
                }
                weights["mlp_plain"] += deep_weight * deep_mix[0]
                weights["mlp_masked"] += deep_weight * deep_mix[1]
                prediction = combine_predictions(oof_views[blend_type], weights)
                ordinary, pressure = auc_pair(target, prediction, pressure_weights)
                rows.append(
                    {
                        "blend_type": blend_type,
                        **{f"{name}_weight": weights[name] for name in MODEL_NAMES},
                        "oof_auc": ordinary,
                        "test_like_oof_auc": pressure,
                        "robust_auc": 0.5 * (ordinary + pressure),
                    }
                )
    frame = pd.DataFrame(rows).sort_values(
        ["robust_auc", "oof_auc"], ascending=False
    )
    winner = frame.iloc[0]
    weights = {
        name: float(winner[f"{name}_weight"]) for name in MODEL_NAMES
    }
    test_prediction = combine_predictions(
        test_views[str(winner["blend_type"])], weights
    )
    return frame, test_prediction


def segment_metrics(base_features, target, prediction, model_name, pressure_weights):
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
            ordinary, pressure = auc_pair(
                subset,
                prediction[mask.to_numpy()],
                pressure_weights[mask.to_numpy()],
            )
            rows.append(
                {
                    "model": model_name,
                    "segment": name,
                    "rows": int(mask.sum()),
                    "auc": ordinary,
                    "test_like_auc": pressure,
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
    # V3 was trained with CPU hist. Keep it fixed even when a GPU is attached
    # for PyTorch, otherwise the fallback is no longer the same baseline.
    xgb_device = "cpu"
    mlp_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running adversarial validation...", flush=True)
    domain_score, pressure_weights, domain_auc = adversarial_validation(
        base_train, base_test, OUTPUT_DIR
    )
    drift_rows = []
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
        drift_rows.append(row)
    pd.DataFrame(drift_rows).to_csv(
        OUTPUT_DIR / "v5_field_drift_report.csv", index=False
    )
    print(
        f"train={len(base_train):,}, test={len(base_test):,}, "
        f"base_features={base_train.shape[1]}, xgb_device={xgb_device}, "
        f"mlp_device={mlp_device}, domain_auc={domain_auc:.6f}",
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
    mlp_history_rows = []

    for fold, (fit_index, valid_index) in enumerate(splits, start=1):
        fold_registry[valid_index] = fold - 1
        fold_fit = base_train.iloc[fit_index].reset_index(drop=True)
        fold_valid = base_train.iloc[valid_index].reset_index(drop=True)
        fit_y = target.iloc[fit_index]
        valid_y = target.iloc[valid_index]
        valid_pressure_weights = pressure_weights[valid_index]

        print(f"\n===== fold {fold}/{N_FOLDS}: native model =====", flush=True)
        native_fit, native_valid, native_test_x = encode_from_fold_vocabulary(
            fold_fit, fold_valid, base_test
        )
        native_model = make_classifier(SEED + fold, NATIVE_ESTIMATORS)
        fit_classifier(
            native_model,
            native_fit,
            fit_y,
            native_valid,
            valid_y,
            NATIVE_EARLY_STOPPING,
        )
        native_valid_prediction = native_model.predict_proba(native_valid)[:, 1]
        oof["native"][valid_index] = native_valid_prediction
        test_predictions["native"] += (
            native_model.predict_proba(native_test_x)[:, 1] / N_FOLDS
        )
        native_auc, native_pressure_auc = auc_pair(
            valid_y, native_valid_prediction, valid_pressure_weights
        )

        print(f"===== fold {fold}/{N_FOLDS}: semantic imputation =====", flush=True)
        imputer = SemanticFoldImputer(SEED + fold).fit(fold_fit)
        semantic_fit_frame = imputer.transform(fold_fit)
        semantic_valid_frame = imputer.transform(fold_valid)
        semantic_test_frame = imputer.transform(base_test)
        semantic_fit, semantic_valid, semantic_test_x = encode_from_fold_vocabulary(
            semantic_fit_frame, semantic_valid_frame, semantic_test_frame
        )

        print(f"===== fold {fold}/{N_FOLDS}: semantic model =====", flush=True)
        semantic_model = make_classifier(SEED + fold, SEMANTIC_ESTIMATORS)
        fit_classifier(
            semantic_model,
            semantic_fit,
            fit_y,
            semantic_valid,
            valid_y,
            SEMANTIC_EARLY_STOPPING,
        )
        semantic_valid_prediction = semantic_model.predict_proba(semantic_valid)[:, 1]
        oof["semantic"][valid_index] = semantic_valid_prediction
        test_predictions["semantic"] += (
            semantic_model.predict_proba(semantic_test_x)[:, 1] / N_FOLDS
        )
        semantic_auc, semantic_pressure_auc = auc_pair(
            valid_y, semantic_valid_prediction, valid_pressure_weights
        )

        print(f"===== fold {fold}/{N_FOLDS}: semantic XGBoost =====", flush=True)
        xgb_model = make_xgb_classifier(SEED + fold, xgb_device)
        xgb_model.fit(
            semantic_fit,
            fit_y,
            eval_set=[(semantic_valid, valid_y)],
            verbose=200,
        )
        xgb_valid_prediction = xgb_model.predict_proba(semantic_valid)[:, 1]
        oof["xgboost"][valid_index] = xgb_valid_prediction
        test_predictions["xgboost"] += (
            xgb_model.predict_proba(semantic_test_x)[:, 1] / N_FOLDS
        )
        xgb_auc, xgb_pressure_auc = auc_pair(
            valid_y, xgb_valid_prediction, valid_pressure_weights
        )
        native_best_iteration = native_model.best_iteration_
        semantic_best_iteration = semantic_model.best_iteration_
        xgb_best_iteration = xgb_model.best_iteration
        for model_name, model, columns in (
            ("native", native_model, native_fit.columns),
            ("semantic", semantic_model, semantic_fit.columns),
            ("xgboost", xgb_model, semantic_fit.columns),
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
        del native_model, semantic_model, xgb_model
        gc.collect()

        print(f"===== fold {fold}/{N_FOLDS}: MLP preparation =====", flush=True)
        mlp_mean, mlp_std = fit_mlp_scaler(semantic_fit)
        mlp_fit_x = transform_mlp(semantic_fit, mlp_mean, mlp_std)
        mlp_valid_x = transform_mlp(semantic_valid, mlp_mean, mlp_std)
        mlp_test_x = transform_mlp(semantic_test_x, mlp_mean, mlp_std)
        fit_y_array = fit_y.to_numpy(dtype="float32")

        print(f"===== fold {fold}/{N_FOLDS}: plain MLP =====", flush=True)
        (
            plain_valid_prediction,
            plain_test_prediction,
            plain_auc,
            plain_epoch,
            plain_history,
        ) = fit_tabular_mlp(
            mlp_fit_x,
            fit_y_array,
            mlp_valid_x,
            valid_y,
            mlp_test_x,
            SEED + 100 + fold,
            "plain_mlp",
        )
        oof["mlp_plain"][valid_index] = plain_valid_prediction
        test_predictions["mlp_plain"] += plain_test_prediction / N_FOLDS
        plain_auc, plain_pressure_auc = auc_pair(
            valid_y, plain_valid_prediction, valid_pressure_weights
        )
        mlp_history_rows.extend(
            {"fold": fold, "model": "mlp_plain", **row}
            for row in plain_history
        )

        print(f"===== fold {fold}/{N_FOLDS}: masked augmentation =====", flush=True)
        raw_fit = raw_train.iloc[fit_index].reset_index(drop=True)
        augmented_raw, augmented_y = make_masked_augmentation(
            raw_fit, raw_test, fit_y, SEED + 200 + fold
        )
        augmented_base = build_base_features(augmented_raw)
        augmented_semantic_frame = imputer.transform(augmented_base)
        augmented_semantic = pd.get_dummies(
            augmented_semantic_frame,
            columns=CATEGORICAL_COLUMNS,
            dtype="int8",
        ).reindex(columns=semantic_fit.columns, fill_value=0)
        augmented_x = transform_mlp(augmented_semantic, mlp_mean, mlp_std)
        masked_fit_x = np.concatenate([mlp_fit_x, augmented_x], axis=0)
        masked_fit_y = np.concatenate([fit_y_array, augmented_y], axis=0)
        (
            masked_valid_prediction,
            masked_test_prediction,
            masked_auc,
            masked_epoch,
            masked_history,
        ) = fit_tabular_mlp(
            masked_fit_x,
            masked_fit_y,
            mlp_valid_x,
            valid_y,
            mlp_test_x,
            SEED + 100 + fold,
            "masked_mlp",
        )
        oof["mlp_masked"][valid_index] = masked_valid_prediction
        test_predictions["mlp_masked"] += masked_test_prediction / N_FOLDS
        masked_auc, masked_pressure_auc = auc_pair(
            valid_y, masked_valid_prediction, valid_pressure_weights
        )
        mlp_history_rows.extend(
            {"fold": fold, "model": "mlp_masked", **row}
            for row in masked_history
        )
        fold_rows.extend(
            [
                {
                    "fold": fold,
                    "model": "native",
                    "auc": native_auc,
                    "test_like_auc": native_pressure_auc,
                    "best_iteration": native_best_iteration,
                },
                {
                    "fold": fold,
                    "model": "semantic",
                    "auc": semantic_auc,
                    "test_like_auc": semantic_pressure_auc,
                    "best_iteration": semantic_best_iteration,
                },
                {
                    "fold": fold,
                    "model": "xgboost",
                    "auc": xgb_auc,
                    "test_like_auc": xgb_pressure_auc,
                    "best_iteration": xgb_best_iteration,
                },
                {
                    "fold": fold,
                    "model": "mlp_plain",
                    "auc": plain_auc,
                    "test_like_auc": plain_pressure_auc,
                    "best_iteration": plain_epoch,
                },
                {
                    "fold": fold,
                    "model": "mlp_masked",
                    "auc": masked_auc,
                    "test_like_auc": masked_pressure_auc,
                    "best_iteration": masked_epoch,
                },
            ]
        )
        print(
            f"fold {fold}: native_auc={native_auc:.7f}, "
            f"semantic_auc={semantic_auc:.7f}, xgb_auc={xgb_auc:.7f}, "
            f"plain_mlp={plain_auc:.7f}, masked_mlp={masked_auc:.7f}",
            flush=True,
        )
        del (
            native_fit,
            native_valid,
            native_test_x,
            semantic_fit_frame,
            semantic_valid_frame,
            semantic_fit,
            semantic_valid,
            semantic_test_x,
            semantic_test_frame,
            imputer,
            mlp_fit_x,
            mlp_valid_x,
            mlp_test_x,
            augmented_raw,
            augmented_base,
            augmented_semantic_frame,
            augmented_semantic,
            augmented_x,
            masked_fit_x,
            masked_fit_y,
            raw_fit,
            augmented_y,
            fit_y_array,
            mlp_mean,
            mlp_std,
        )
        gc.collect()

    model_metrics = {}
    for model_name in MODEL_NAMES:
        ordinary, pressure = auc_pair(target, oof[model_name], pressure_weights)
        model_metrics[model_name] = {
            "oof_auc": ordinary,
            "test_like_oof_auc": pressure,
        }
    v3_oof = combine_predictions(oof, V3_WEIGHTS)
    v3_test = combine_predictions(test_predictions, V3_WEIGHTS)
    v3_auc, v3_pressure_auc = auc_pair(target, v3_oof, pressure_weights)
    v3_robust_auc = 0.5 * (v3_auc + v3_pressure_auc)
    blend_frame, candidate_test = search_robust_blend(
        target, oof, test_predictions, pressure_weights
    )
    winner = blend_frame.iloc[0]
    candidate_weights = {
        name: float(winner[f"{name}_weight"]) for name in MODEL_NAMES
    }
    deep_weight = candidate_weights["mlp_plain"] + candidate_weights["mlp_masked"]
    blend_accepted = bool(
        deep_weight > 0.0
        and winner["robust_auc"] >= v3_robust_auc + MIN_DEEP_BLEND_ROBUST_GAIN
        and winner["oof_auc"] >= v3_auc + 0.00005
        and winner["test_like_oof_auc"] >= v3_pressure_auc + 0.00005
    )
    if blend_accepted:
        selected_view = (
            oof
            if winner["blend_type"] == "raw"
            else {name: percentile_rank(values) for name, values in oof.items()}
        )
        selected_oof = combine_predictions(selected_view, candidate_weights)
        selected_test = candidate_test
        selected_weights = candidate_weights
        selected_blend_type = str(winner["blend_type"])
        selection_reason = "deep diversity passed ordinary and test-like OOF guardrails"
    else:
        selected_oof = v3_oof
        selected_test = v3_test
        selected_weights = V3_WEIGHTS.copy()
        selected_blend_type = "raw"
        selection_reason = "guardrail fallback to fixed V3 blend"
    selected_auc, selected_pressure_auc = auc_pair(
        target, selected_oof, pressure_weights
    )

    oof_frame = pd.DataFrame(
        {
            ID_COLUMN: train_ids,
            "fold": fold_registry,
            TARGET: target,
            "domain_score": domain_score,
            "test_like_weight": pressure_weights,
            **{f"{name}_prediction": values for name, values in oof.items()},
            "v3_fixed_prediction": v3_oof,
            "selected_prediction": selected_oof,
        }
    )
    test_prediction_frame = pd.DataFrame(
        {
            ID_COLUMN: test_ids,
            **{
                f"{name}_prediction": values
                for name, values in test_predictions.items()
            },
            "v3_fixed_prediction": v3_test,
            "selected_prediction": selected_test,
        }
    )
    oof_frame.to_csv(OUTPUT_DIR / "v5_oof.csv", index=False)
    test_prediction_frame.to_csv(OUTPUT_DIR / "v5_test_predictions.csv", index=False)
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(OUTPUT_DIR / "v5_fold_metrics.csv", index=False)
    pd.DataFrame(mlp_history_rows).to_csv(
        OUTPUT_DIR / "v5_mlp_training_history.csv", index=False
    )
    plain_folds = fold_frame.query("model == 'mlp_plain'").set_index("fold")
    masked_folds = fold_frame.query("model == 'mlp_masked'").set_index("fold")
    mlp_ablation = pd.DataFrame(
        {
            "fold": plain_folds.index,
            "plain_auc": plain_folds["auc"],
            "masked_auc": masked_folds["auc"],
            "masked_auc_delta": masked_folds["auc"] - plain_folds["auc"],
            "plain_test_like_auc": plain_folds["test_like_auc"],
            "masked_test_like_auc": masked_folds["test_like_auc"],
            "masked_test_like_auc_delta": (
                masked_folds["test_like_auc"] - plain_folds["test_like_auc"]
            ),
        }
    ).reset_index(drop=True)
    mlp_ablation.to_csv(OUTPUT_DIR / "v5_mlp_ablation.csv", index=False)
    blend_frame.to_csv(OUTPUT_DIR / "v5_blend_search.csv", index=False)
    importance = pd.DataFrame(importance_rows)
    importance.groupby(["model", "feature"], as_index=False)["importance"].agg(
        mean="mean", std="std"
    ).sort_values(["model", "mean"], ascending=[True, False]).to_csv(
        OUTPUT_DIR / "v5_feature_importance.csv", index=False
    )
    segments = []
    for model_name in MODEL_NAMES:
        segments += segment_metrics(
            base_train,
            target,
            oof[model_name],
            model_name,
            pressure_weights,
        )
    segments += segment_metrics(
        base_train,
        target,
        selected_oof,
        "selected_blend",
        pressure_weights,
    )
    pd.DataFrame(segments).to_csv(
        OUTPUT_DIR / "v5_segment_metrics.csv", index=False
    )

    prediction_frame = pd.DataFrame(oof)
    residual_frame = prediction_frame.apply(lambda values: target - values)
    prediction_frame.corr().to_csv(OUTPUT_DIR / "v5_prediction_correlation.csv")
    residual_frame.corr().to_csv(OUTPUT_DIR / "v5_residual_correlation.csv")
    domain_bins = pd.qcut(domain_score, q=10, labels=False, duplicates="drop")
    domain_rows = []
    for domain_bin in np.unique(domain_bins):
        mask = domain_bins == domain_bin
        for model_name in [*MODEL_NAMES, "v3_fixed", "selected"]:
            prediction = (
                oof[model_name]
                if model_name in oof
                else v3_oof
                if model_name == "v3_fixed"
                else selected_oof
            )
            domain_rows.append(
                {
                    "domain_bin": int(domain_bin),
                    "model": model_name,
                    "rows": int(mask.sum()),
                    "mean_domain_score": float(domain_score[mask].mean()),
                    "auc": float(roc_auc_score(target[mask], prediction[mask])),
                }
            )
    pd.DataFrame(domain_rows).to_csv(
        OUTPUT_DIR / "v5_domain_segment_metrics.csv", index=False
    )

    clipped = np.clip(selected_oof, 1e-7, 1.0 - 1e-7)
    per_row_logloss = -(
        target.to_numpy() * np.log(clipped)
        + (1 - target.to_numpy()) * np.log(1.0 - clipped)
    )
    error_frame = raw_train.copy()
    error_frame[TARGET] = target
    error_frame["selected_prediction"] = selected_oof
    error_frame["v3_prediction"] = v3_oof
    error_frame["domain_score"] = domain_score
    error_frame["prediction_disagreement"] = prediction_frame.std(axis=1).to_numpy()
    error_frame["logloss"] = per_row_logloss
    error_frame.nlargest(10_000, "logloss").to_csv(
        OUTPUT_DIR / "v5_high_confidence_errors.csv", index=False
    )

    if sample_submission.columns.tolist() != [ID_COLUMN, TARGET]:
        raise ValueError("Unexpected sample_submission.csv columns.")
    if not sample_submission[ID_COLUMN].equals(test_ids):
        raise ValueError("sample_submission IDs are not aligned with test.csv.")
    if not np.isfinite(selected_test).all():
        raise ValueError("Selected test predictions contain NaN or infinity.")
    sample_submission[TARGET] = np.clip(selected_test, 0.0, 1.0)
    sample_submission.to_csv(SUBMISSION_PATH, index=False)

    metrics = {
        "version": "v5",
        "domain_auc": domain_auc,
        "models": model_metrics,
        "v3_fixed_oof_auc": float(v3_auc),
        "v3_fixed_test_like_oof_auc": float(v3_pressure_auc),
        "candidate_robust_auc": float(winner["robust_auc"]),
        "selected_oof_auc": float(selected_auc),
        "selected_test_like_oof_auc": float(selected_pressure_auc),
        "selected_weights": selected_weights,
        "selected_blend_type": selected_blend_type,
        "blend_accepted": blend_accepted,
        "selection_reason": selection_reason,
        "masked_mlp_oof_gain_vs_plain": float(
            model_metrics["mlp_masked"]["oof_auc"]
            - model_metrics["mlp_plain"]["oof_auc"]
        ),
        "masked_mlp_positive_folds": int((mlp_ablation["masked_auc_delta"] > 0).sum()),
        "folds": N_FOLDS,
        "seed": SEED,
        "native_estimators": NATIVE_ESTIMATORS,
        "semantic_estimators": SEMANTIC_ESTIMATORS,
        "native_early_stopping": NATIVE_EARLY_STOPPING,
        "semantic_early_stopping": SEMANTIC_EARLY_STOPPING,
        "xgb_estimators": XGB_ESTIMATORS,
        "xgb_early_stopping": XGB_EARLY_STOPPING,
        "xgb_device": xgb_device,
        "mlp_device": mlp_device,
        "elapsed_minutes": (time.time() - started) / 60,
        "submission_path": str(SUBMISSION_PATH),
    }
    with (OUTPUT_DIR / "v5_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print("\nV5 complete", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
