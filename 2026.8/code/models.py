"""Model factory and competition-oriented defaults."""

from __future__ import annotations

from typing import Any


SUPPORTED_MODELS = ("lightgbm", "xgboost", "catboost")
SUPPORTED_PRESETS = ("baseline", "high_capacity")


def preset_overrides(name: str, preset: str) -> dict[str, Any]:
    if preset == "baseline":
        return {}
    if preset != "high_capacity":
        raise ValueError(f"Unknown preset '{preset}'. Choose from {SUPPORTED_PRESETS}.")
    if name == "lightgbm":
        return {
            "n_estimators": 6000,
            "learning_rate": 0.025,
            "num_leaves": 63,
            "min_child_samples": 100,
            "reg_lambda": 2.0,
        }
    if name == "xgboost":
        return {
            "n_estimators": 4500,
            "learning_rate": 0.025,
            "max_depth": 9,
            "min_child_weight": 15,
        }
    return {
        "iterations": 4500,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 7.0,
    }


def make_model(name: str, random_state: int, overrides: dict[str, Any] | None = None):
    overrides = overrides or {}

    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        params = {
            "objective": "binary",
            "n_estimators": 3000,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 100,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }
        params.update(overrides)
        return LGBMClassifier(**params)

    if name == "xgboost":
        from xgboost import XGBClassifier

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 2500,
            "learning_rate": 0.035,
            "max_depth": 7,
            "min_child_weight": 10,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
            "early_stopping_rounds": 150,
        }
        params.update(overrides)
        return XGBClassifier(**params)

    if name == "catboost":
        from catboost import CatBoostClassifier

        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": 2500,
            "learning_rate": 0.04,
            "depth": 7,
            "l2_leaf_reg": 5.0,
            "random_seed": random_state,
            "thread_count": -1,
            "verbose": 200,
            "allow_writing_files": False,
        }
        params.update(overrides)
        return CatBoostClassifier(**params)

    raise ValueError(f"Unknown model '{name}'. Choose from {SUPPORTED_MODELS}.")


def tuning_guide() -> dict[str, dict[str, list[Any]]]:
    """Small search spaces chosen to be useful without exploding server cost."""
    return {
        "lightgbm": {
            "learning_rate": [0.02, 0.035, 0.05],
            "num_leaves": [15, 31, 63],
            "min_child_samples": [50, 100, 200],
            "feature_fraction": [0.75, 0.9, 1.0],
            "reg_lambda": [0.5, 1.0, 3.0],
        },
        "xgboost": {
            "learning_rate": [0.02, 0.035, 0.05],
            "max_depth": [5, 7, 9],
            "min_child_weight": [5, 10, 20],
            "subsample": [0.75, 0.9, 1.0],
            "reg_lambda": [1.0, 2.0, 5.0],
        },
        "catboost": {
            "learning_rate": [0.025, 0.04, 0.06],
            "depth": [6, 7, 8],
            "l2_leaf_reg": [3.0, 5.0, 10.0],
            "random_strength": [0.5, 1.0, 2.0],
        },
    }
