---
name: kaggle-tabular-competition
description: "Build and iteratively improve standalone, server-run solutions for Kaggle-style tabular competitions. Use when a project has competition instructions plus train/test/sample-submission data and the user wants data auditing, semantic feature engineering, leakage-safe validation, versioned Python scripts, returned-run analysis, model changes, ensembling, or a validated submission; especially when Codex writes code locally but the user runs it on a remote server."
---

# Kaggle Tabular Competition

Optimize the reliability of the experiment loop, not just a single local score. In this workflow Codex inspects files and writes code locally; the user normally runs each version on a server and returns the score, log, and artifacts. Do not run model training locally unless the user explicitly asks.

## Discover the project contract

Identify the competition root: the directory containing `Readme.md` or equivalent instructions, `data/`, `code/`, and `output/`. Read the instructions before designing a model, then inspect `data/train.csv`, `data/test.csv`, and `data/sample_submission.csv`.

Establish and report:

- prediction unit, target, ID/key, task type, exact metric, and required submission semantics
- train/test shapes and schema differences
- dtypes, missingness, cardinality, constants, duplicates, target balance/distribution, suspicious IDs or time/group structure
- validation constraints, server dependencies, runtime or memory limits, and the current highest script version

Infer submission columns, order, and output type from the sample submission and instructions rather than convention. If the evidence conflicts, stop and surface the conflict.

Read [references/competition-playbook.md](references/competition-playbook.md) before proposing preprocessing, features, validation, models, or ensembles. Read [references/server-iteration-loop.md](references/server-iteration-loop.md) whenever creating a script or reviewing returned server results.

## Design the next experiment

State one primary hypothesis for every version. Compare it against the strongest trustworthy baseline on the same fold registry. A version may contain the full runnable pipeline, but the experimental change should be coherent enough to attribute any gain or loss.

Choose missing-value handling from both column meaning and dtype. Preserve raw values when the model can handle missingness; add indicators or imputed companion features only when they express a plausible missing-data mechanism. Fit all learned imputers inside each training fold. Never use the target for imputation.

Build features from domain meaning first: quantities, rates, balances, time spans, bounded ratios, counts, and interactions that correspond to a plausible process. Record the formula and rationale. Correlation and feature importance are diagnostics, not sufficient reasons to create a feature. Avoid broad polynomial crosses or target-guided feature mining that can amplify noise.

Choose models to match the data and metric. Start with a complete, leakage-safe baseline; then test meaningfully different feature representations, objectives, categorical handling, or model families. After one or two focused parameter experiments on the same model family, stop parameter-only iteration unless results show a clear unresolved capacity or regularization issue. Prefer a structural feature change, a different model family, a metric-aligned decision layer, or genuinely diverse blending.

## Write a standalone server version

Create a new `code/vN.py`, where `N` is one higher than the greatest existing numeric version. Never overwrite a prior version. Each version must be self-contained: it may use installed packages but must not import sibling project modules or require a notebook state.

The script must:

- locate the three competition CSVs robustly in an attached server dataset or the project `data/` directory
- fail early on schema or submission-contract mismatches
- reproduce folds from a fixed seed and save fold assignments by stable ID
- fit learned preprocessing and supervised encodings strictly within folds
- calculate the exact competition metric on OOF predictions
- train and predict end to end with deterministic seeds where supported
- create the final `submission.csv` in the run/project root, never only inside `output/`
- write useful diagnostics under `output/` using names prefixed with the version
- print a compact completion summary containing metric, fold spread, selected candidate, paths, runtime, and prediction range/class counts

Static checks are allowed locally, but leave full training and prediction to the user's server unless asked otherwise.

## Review returned results

When the user adds logs, scores, submissions, or `output/` files, inspect them before writing the next version. Separate:

1. execution correctness and artifact completeness
2. local OOF performance and stability
3. public leaderboard movement
4. evidence from fold, subgroup, missingness, residual, calibration, or prediction-diversity diagnostics

Do not chase a leaderboard change that lacks validation support. Keep failed experiments in the history and explain what they ruled out. If artifacts are incomplete or folds differ, repair comparability before optimizing.

When third-party code appears in `code/`, audit its assumptions, validation, preprocessing boundaries, data dependencies, and leakage risk. Extract useful architecture or method ideas into a new standalone version; do not blindly copy code, trust a reported score, use test-label-like overrides, or make an external submission an unexplained dependency.

End each iteration with a short decision record: evidence reviewed, hypothesis, single main change, leakage controls, expected diagnostic outputs, exact script to upload, and what result would cause the next direction to be kept, revised, or abandoned.

## Submission gate

Construct from `sample_submission.csv` and preserve its row and column order. Check row count, IDs, duplicates, missing or non-finite predictions, label vocabulary, probability bounds, and multiclass row sums when relevant. Use `scripts/validate_submission.py` when practical. Do not claim a runnable version is successful until the user returns a completed run; distinguish static readiness from server-verified results.
