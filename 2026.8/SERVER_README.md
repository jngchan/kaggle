# Server training workflow

Use Python 3.10 or newer. From the project root:

```bash
python3 -m pip install -r requirements.txt

# Fast smoke test (20% data, 3 folds). This also validates submission generation.
python3 code/run_all.py --models lightgbm catboost --folds 3 --sample-fraction 0.2

# Full experiment. The default compares LightGBM, CatBoost and their rank blend.
python3 code/run_all.py --models lightgbm catboost --folds 5

# Recommended next experiment after the completed native/median ablation.
# Semantic v3 removes aggregate missing counts and fixes pandas dtype assignment.
python3 code/train.py --model lightgbm --missing-strategy semantic \
  --experiment-name lightgbm_semantic_v3 --folds 5

# Produce paired fold and missing-segment deltas.
python3 code/compare_experiments.py \
  --baseline lightgbm --candidate lightgbm_semantic_v3

# Experiment C: isolate the value of larger/deeper trees after feature experiments.
python3 code/train.py --model lightgbm --preset high_capacity \
  --experiment-name lightgbm_high_capacity --folds 5

# Combine semantic imputation and capacity only if an isolated change improves OOF.
python3 code/train.py --model lightgbm --missing-strategy semantic \
  --preset high_capacity --experiment-name lightgbm_semantic_high_capacity --folds 5

# Compare existing candidates and regenerate submission.csv from the OOF winner.
python3 code/select_and_submit.py --models lightgbm lightgbm_median_v2 \
  lightgbm_semantic_v3 \
  lightgbm_high_capacity lightgbm_semantic_high_capacity
```

To run models separately (useful on different servers or GPUs):

```bash
python3 code/train.py --model lightgbm --folds 5
python3 code/train.py --model xgboost --folds 5
python3 code/train.py --model catboost --folds 5
python3 code/select_and_submit.py --models lightgbm xgboost catboost
```

`submission.csv` is created in the project root. It contains the test IDs and
probabilities in the exact format required by `Readme.md`.

The `output/` directory contains per-fold scores, overall OOF scores, compact OOF/test
predictions, feature importance, model comparison, selected-model metadata, and a
small tuning search space. Each experiment also writes segment AUCs by missing-value
count and field, plus the exact fold registry. `native` keeps NaNs for the tree,
`median` adds fold-median companions, and `semantic` adds field-aware companions
without replacing raw values. Tune one parameter family at a time after checking the
importance and fold variance; preserve the same folds/seed when comparing models.

For CatBoost GPU training, edit its defaults in `code/models.py` and add
`"task_type": "GPU"`. For XGBoost GPU training, add `"device": "cuda"`.
