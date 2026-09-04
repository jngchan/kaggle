# Competition playbook

Use this reference while designing data handling, validation, features, models, and ensembles.

## Data audit and validation

Inspect metadata before loading full tables unnecessarily: headers, sample rows, shapes, dtypes, missing counts/rates, unique counts, numeric ranges/quantiles, category frequencies, constants, duplicates, target distribution, and train/test schema or distribution differences.

Choose splits from the prediction setting:

- IID binary or multiclass rows: stratified K-fold.
- IID regression: K-fold; stratify binned targets only when justified.
- repeated users, patients, devices, households, products, or source rows: group-aware folds.
- forecasting or temporal drift: chronological or rolling splits, with a gap when information can cross the boundary.
- spatial data: geographic blocks or groups.
- severe class imbalance: stratify when valid and keep validation in its natural distribution.

Save one fold registry and reuse it across comparable experiments. If the data lacks a stable unique ID, save an explicit row key plus a data fingerprint. Never infer robustness from different random folds without accounting for split variance.

Mirror the official metric exactly. For balanced accuracy or F1, learn thresholds, class-prior corrections, or decision rules using OOF predictions only. For ROC AUC, submit ranking-quality scores and do not tune a classification threshold. For log loss, preserve calibrated probabilities. For RMSE/MAE, verify target transforms and inverse transforms. For multiclass probability metrics, preserve class-column order.

## Missing values

Select a strategy column by column, using the feature's meaning, dtype, missing rate, and likely missingness mechanism:

- numeric measurements: retain native missing values for capable tree models; otherwise learn a fold median or robust statistic and usually add a missing indicator
- skewed counts or nonnegative amounts: consider a robust fill plus indicator; do not create impossible negative sentinels unless the model and meaning support them
- categorical fields: use an explicit missing level for native categorical models or fold-fitted encoders; keep unknown test levels distinct when useful
- ordered categories: preserve the order only when documented or semantically defensible
- dates/times: parse components and represent missingness; never replace an unknown date with a value that invents chronology
- IDs/codes: determine whether they are keys, categories, ordered generators, groups, or leakage candidates before imputing or modeling them

Learn every imputer, vocabulary, scaler, dimensionality reduction, feature selector, or target statistic on the training part of each fold. Unsupervised train+test fitting is transductive: use it only when competition rules allow it, label it explicitly, and compare against a train-only baseline.

Treat missingness as potential signal but test it. Useful diagnostics include metric by missing-count bucket, per-field-missing subsets, and train/test missing-rate deltas. Preserve a raw/native feature alongside an imputed companion when the model can benefit from both and the added complexity is justified.

## Semantic feature engineering

For every feature family, write a compact hypothesis record:

| Field | Meaning |
|---|---|
| name/formula | deterministic definition |
| rationale | real process or constraint represented |
| availability | computable identically at prediction time |
| leakage risk | target, time, group, aggregate, or external-data concerns |
| expected slice | rows or subgroups where it should help |

Prefer small, coherent families such as:

- totals and residual balances among related components
- rates with a meaningful exposure denominator
- weekday/weekend or before/after differences
- utilization, intensity, recovery, or concentration measures
- monotonic transforms for heavy tails
- a few defensible categorical interactions
- counts or flags for data quality and missingness

Guard division by zero and implausible ranges. Preserve source columns unless the experiment is a deliberate ablation. Do not manufacture interactions merely because two columns correlate with the target; this double-uses sampling noise and can magnify spurious structure.

Supervised encodings must be nested or cross-fitted. For an outer validation fold, build target statistics using only the outer training rows; if training-row encodings are needed, cross-fit them again inside that outer training set. Use smoothing and a global-prior fallback for rare or unseen levels.

## Leakage and shift checklist

Check for:

- fields created after the event being predicted
- target-derived aggregates or encodings reaching validation rows
- preprocessors fitted before splitting
- duplicate or near-duplicate entities split across folds
- group, time, filename, row-order, or ID artifacts
- generated/synthetic data formulas that invite memorization
- external data or submissions whose provenance violates rules or cannot be reproduced
- repeated use of the public leaderboard as a model-selection set

Compare train/test distributions, missingness, unseen categories, time ranges, and entity overlap. Adversarial validation can locate shift, but high separability is a diagnostic rather than proof of leakage. Reconsider validation when the features distinguishing train from test also drive the target model.

## Model and experiment ladder

Use the smallest ladder that answers the current question:

1. constant/class-prior sanity check and a cheap end-to-end baseline
2. strong mixed-table baseline such as CatBoost, LightGBM, or XGBoost
3. missing-strategy or semantic-feature ablation on identical folds
4. a different representation, objective, model family, or metric-aligned decision layer
5. seed averaging or blending only when prediction errors are genuinely diverse

Change one primary axis per version. A longer training horizon followed by one capacity/regularization test is reasonable; repeated grids around negligible gains are not. After one or two parameter-only rounds, require evidence such as underfitting, unstable leaves, premature early stopping, or consistent fold gains before tuning that family again.

For each candidate compare overall OOF score, per-fold deltas, fold spread, subgroup/segment deltas, runtime, and prediction behavior. An improvement smaller than normal fold noise is weak evidence. Confirm important changes across folds or seeds before making them the new foundation.

## Ensembling and post-processing

Blend only aligned OOF and test predictions with the same row IDs and fold registry. Save the transformation applied to each prediction. Prefer simple averages or a low-dimensional nonnegative blend, and evaluate diversity using prediction/residual correlations plus fold-level gains.

Use rank averaging only for ranking metrics. Use probability averaging for probability metrics unless calibration evidence supports another method. Fit stacking, blend weights, calibration, thresholds, and class-prior corrections on cross-fitted OOF evidence; avoid a large search that overfits OOF.

Keep a single-model fallback whenever the ensemble gain is marginal or unstable.
