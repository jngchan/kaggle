# Standalone server iteration loop

Use this reference whenever creating `code/vN.py` or reviewing files returned from the server.

## Version contract

Find numeric files matching `code/v*.py` and create the next unused number. Do not edit an earlier version to represent a new experiment. Put the hypothesis and primary delta in the module docstring.

Each `vN.py` must contain everything needed for that experiment: constants, input discovery, feature construction, fold-safe preprocessing, model definitions, validation, artifact writing, submission construction, and `main()`. It may import installed libraries but not `feature.py`, `models.py`, another version, or hidden notebook state. This duplication is intentional because the user uploads versions independently.

Make paths portable:

- when the project structure exists, treat the directory above `code/` as the project/run root and read `<root>/data`
- on Kaggle, search attached directories under `/kaggle/input` for one directory containing compatible `train.csv`, `test.csv`, and `sample_submission.csv`
- use `/kaggle/working` as the run root when that is the writable execution root
- on another server, allow the script location or a clearly named environment/config override to define the root

Print the resolved input and output paths. Never silently choose the first `train.csv`; confirm required headers and compatible sibling files.

The final artifact must be `<run-root>/submission.csv`, where the run root corresponds to the competition directory such as `2026.8` or the server's writable working root. Additional artifacts belong in `<run-root>/output/`.

## Minimum script safeguards

Before training:

- verify train contains the target and test does not
- verify feature schemas and sample-submission columns/order
- verify target values and metric assumptions
- check ID uniqueness and sample/test ID alignment
- set seeds and create the output directory

During validation:

- create and save one fold registry by stable ID
- allocate exactly one OOF prediction per training row
- fit all learned transforms inside the fold
- use early stopping only on the current validation fold
- average test predictions across folds
- assert complete, finite OOF/test predictions and correct shapes

Before completion:

- build from a copy of the sample submission
- preserve sample ID and row order
- validate probability bounds, label vocabulary, or multiclass row sums as applicable
- write `submission.csv` last, after all assertions pass
- print `RUN_COMPLETE` only after the final file and essential diagnostics exist

## Artifact contract

Prefix files with the version, for example `v11_metrics.json`. Prefer portable CSV/JSON for human review; compact NumPy files are acceptable in addition when data is large.

Required unless the task makes an item inapplicable:

- `vN_metrics.json`: version, hypothesis, primary change, metric name, overall OOF score, fold mean/std, seeds, rows/features, runtime, model/config summary, selected candidate, prediction summary, and paths
- `vN_fold_metrics.csv`: fold, score, row count, best iteration when available
- `vN_oof.csv` or equivalent: ID, target, fold, and prediction(s)
- `vN_test_predictions.csv` or equivalent: ID and candidate prediction(s)
- `vN_data_audit.json`: shapes, schema, missingness, target distribution, train/test differences, and leakage/shift notes
- `vN_feature_manifest.csv`: feature, source columns, formula/family, semantic rationale, and leakage status
- `vN_feature_importance.csv` when the model exposes a defensible importance measure
- `vN_segment_metrics.csv` for important classes, missingness buckets, time/groups, or other risk slices

For multiclass classification, also save per-class recall and a confusion matrix. For ensembles, save component scores, weights, OOF/test prediction correlations, and the acceptance rule. Include enough information to compare versions without rerunning them.

## Reviewing returned files

Inventory the returned log, leaderboard score, `submission.csv`, and version-prefixed artifacts. Confirm the run completed and that metrics refer to the script the user says was run. Then compare only like with like:

- same data and fold registry
- same official metric and prediction semantics
- paired fold deltas rather than only means
- gains or regressions in relevant segments
- runtime/memory cost and failure warnings
- OOF versus public leaderboard direction and plausible distribution shift
- prediction/residual correlation for ensemble candidates

Classify the result:

- **keep**: credible OOF gain, stable folds/slices, no leakage, reasonable cost
- **diagnostic only**: useful insight but marginal or unstable score
- **reject**: regression, leakage, invalid comparison, or unjustified complexity
- **rerun**: incomplete artifacts, schema/path failure, or corrupted predictions

Choose the next experiment from evidence. Parameter-only work is limited to one or two focused rounds per model family unless diagnostics specifically justify more. Otherwise change representation, missing strategy, feature hypothesis, objective, model structure/family, or a low-dimensional diverse ensemble.

If the user returns only a leaderboard score, use it as weak evidence and request or ensure that the next version exports the missing diagnostics. Do not reverse-engineer test labels through repeated submissions.

## Third-party code intake

When a new notebook or script appears in `code/`, inspect:

- what data it reads, including external or pseudo-labeled data
- its validation split, metric, and OOF completeness
- preprocessing fit boundaries and target-encoding implementation
- feature ideas and the real-world hypothesis behind them
- model architecture/objective and post-processing
- output format, hard-coded paths, package assumptions, and resource needs
- manual ID overrides, submission blending, or other non-reproducible behavior

Create a new local version that isolates the useful method against the current baseline. Cite the source filename in the hypothesis record. Do not make the new version depend on the third-party notebook or treat its public score as proof.
