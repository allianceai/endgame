"""Guidance for agents running ML experiments through the Endgame MCP server (served by the guide tool)."""

from __future__ import annotations

WORKFLOW = """\
# Running an ML experiment with Endgame

Endgame has 31 modules (list_modules). Use the ones the problem needs, not only a GBDT.
Work in this order and report every step's evidence:

1. Frame: what one row is, what the target is, what will be known at prediction time.
   Pick validation that matches the data: time-ordered rows -> train_model(time_ordered=True);
   repeated entities (players, patients, sites) -> grouped folds (train_model / compare_models with
   group_column=...); a fixed future test set -> split_data or a held-out file.
2. Inspect and guard: inspect_data, check_data_quality (leakage, IDs, constants). For a separate
   test/production table, check train/test drift (endgame.validation.AdversarialValidator via run_python).
3. Baseline: train_model with "linear" and "lgbm". Everything later must beat these on the same folds.
4. Features (engineer_features): aggregate long/child tables per entity (stats, signal features such as
   entropy or fractal dimension for sensor series), join tables, within-group normalisation (z-score or
   rank within position/site/season), ratios and interactions, lags/rolling for time series, fold-safe
   target encoding for high-cardinality categoricals. Domain features usually beat model choice.
5. Select (select_features): with many features or few rows, compare a filter (mrmr, mutual_info), an
   importance method (boruta, null_importance, permutation) and stability or knockoff selection. Select on
   training rows only (split_data first, apply_to the held-out set) or the CV score is optimistic.
6. Models (recommend_models, then compare_models): compare families on the same folds: GBDTs, tabular
   foundation models (Kumo-Tabular, LimiX, TabPFN, TabICL, ... often best under ~10k rows), neural nets,
   interpretable models (EBM), rotation forests. compare_models stores out-of-fold predictions.
7. Ensemble (ensemble): hill climbing or stacking over the compared models' out-of-fold predictions.
   Keep it only if it beats the best single model.
8. Calibrate and quantify: conformal intervals/sets (endgame.calibration), bootstrap CIs and paired model
   comparisons (endgame.utils.bootstrap_ci, paired_bootstrap_diff), decision curves where costs matter.
9. Explain and check: explain_model, partial dependence, fairness by group (endgame.fairness) when people
   are scored.
10. Report honestly: the validation scheme, every comparison tried (not just the winner), uncertainty,
    and nulls. export_script writes a script that reruns the same folds and prints the same metrics;
    save_model keeps the fitted model.

Anything without a dedicated tool: describe_api("endgame.module.Name") for its signature, then
transform_data (any transformer), train_model(model_name="endgame.module.ClassName") (any estimator) or
run_python (any code, with session datasets and models).
Topics: guide(topic=...) with validation, features, selection, models, ensembling, small_data,
time_series, beyond_tabular, code.
"""

TOPICS = {
    "workflow": WORKFLOW,
    "validation": """\
# Validation
- Shuffled K-fold only when rows are independent. Same entity in many rows -> grouped folds:
  train_model / compare_models(group_column=...) (RepeatedStratifiedGroupKFold via run_python for repeats). Time order -> train_model(time_ordered=True), or
  PurgedTimeSeriesSplit / CombinatorialPurgedKFold with an embargo when labels overlap in time.
  Panels (entity x time) -> PurgedPanelSplit.
- NestedCV when you tune and need an unbiased score.
- AdversarialValidator: can a classifier tell train from test? If AUC >> 0.5, the CV score will not
  transfer; AdversarialKFold builds folds that look like the test set.
- check_data_quality before training; endgame.utils.batch_leakage_check for site/batch effects.
- Small data: repeat CV with different seeds and report the spread, not one number.
""",
    "features": """\
# Feature engineering (engineer_features; operations run in order on one dataset)
- aggregate: per-entity statistics of a long table (e.g. 10 Hz tracking frames per player):
  {"type": "aggregate", "source": "<long ds>", "by": ["player_id"], "within": ["attempt_id"],
   "columns": ["speed", "accel"], "aggs": ["mean", "max", "q90", "sample_entropy", "higuchi_fd"],
   "filter": "drill == 'shuttle'", "order_by": "time", "prefix": "shuttle_"}.
  Signal aggregations (endgame.signal: entropy, fractal dimension, Hjorth, RMS, line length, zero crossings,
  dominant frequency) describe one continuous recording: "within" computes them per recording (attempt,
  session) and averages per entity; without it, an entity's recordings are joined end to end.
  Each column x aggregation is one feature: start with a few that answer the question, add more only if
  they earn their place (with ~500 rows, thousands of features is noise).
- join: {"type": "join", "other": "<ds>", "on": ["player_id"], "how": "left"}
- group_normalize: z-score / percentile rank / difference from group mean within a group (position, site,
  season): {"type": "group_normalize", "by": "position", "columns": [...], "method": "zscore"}
- formula: {"type": "formula", "name": "bmi", "expr": "weight / height ** 2"}
- interactions: products/ratios of numeric columns (endgame.preprocessing.InteractionFeatures)
- lags / rolling: per-entity lags and rolling stats for time-ordered rows (LagFeatures, RollingFeatures)
- target_encode (fold-safe SafeTargetEncoder), frequency_encode, datetime parts, ranks
- auto_aggregate: AutoAggregator group statistics of every numeric column by a key
For transformers without an operation: transform_data(transformer="endgame.module.ClassName").
""",
    "selection": """\
# Feature selection (select_features)
Methods: mrmr, mutual_info, f_test, chi2, relieff, correlation (drop redundant), variance, rfe,
sequential, genetic, boruta, permutation, shap, null_importance, tree_importance, stability, knockoff.
- Few rows, many features: mrmr or relieff, then stability (bootstrap frequency) to keep what repeats.
- Tree models available: boruta or null_importance (tests features against shuffled copies).
- Need a false-discovery guarantee: knockoff (fdr=0.1).
- Select on training rows only; pass apply_to=[held-out ids] so the same columns are kept there.
- Report how many features each method kept and whether the CV score moved beyond noise.
""",
    "models": """\
# Choosing models
recommend_models ranks what to try for the table's size, task, time budget and GPU, and lists models
that need a package or licence. compare_models trains a list on the same folds.
- GBDTs (lgbm, catboost, xgb): strong default on medium/large tables.
- Tabular foundation models: pretrained in-context learners; strongest under ~10-50k rows. TabArena
  (Oct 2026) Elo: kumo_tabular 1979, limix2 1971, tabpfn_35 1882, tabfm 1800, causilo 1799, mitra_v2 1780,
  exaone_tabular 1763 vs tuned LightGBM 1381. On a CUDA machine they train in a separate process so GPU
  memory is freed between models.
- Neural (realmlp, tabm, ft_transformer): diversity for ensembles on larger tables.
- Interpretable (ebm, linear, mars, rulefit, c50): when the decision must be explained, or as a check.
- Any estimator class in any module: train_model(model_name="endgame.models.trees.RotationForestClassifier").
""",
    "ensembling": """\
# Ensembling (ensemble)
Needs models trained by train_model/compare_models on the same dataset with the same (shuffled) folds;
their out-of-fold predictions are combined.
- hill_climbing: greedy forward selection with replacement (Caruana); robust default.
- stacking: a ridge/logistic meta-model on out-of-fold predictions.
- optimized: weights found by Optuna; mean / rank_average: equal weights.
Keep the ensemble only if it beats the best single model by more than the fold-to-fold spread.
""",
    "small_data": """\
# Small tables (hundreds to a few thousand rows)
- Foundation models (TabPFN, TabICL, Kumo-Tabular, LimiX) usually beat GBDTs here; try them first.
- Few, well-motivated features beat many: domain aggregates, then mrmr/stability selection.
- Repeated or grouped CV, bootstrap CIs (endgame.utils.bootstrap_ci), paired comparisons
  (paired_bootstrap_diff) instead of a single split.
- Correct for the number of comparisons you ran; say which results would not survive it.
""",
    "time_series": """\
# Time series
- Forecasting a series: forecast tool (statistical) or endgame.timeseries (N-BEATS, TFT, PatchTST).
- Tabular rows in time order: lags/rolling features (engineer_features), train_model(time_ordered=True).
- Classifying whole series: MiniRocket/Hydra (endgame.timeseries) via transform_data or train_model.
- Sensor signals per entity: aggregate with signal features (entropy, spectra, fractal dimension).
""",
    "beyond_tabular": """\
# Text, images, audio, survival, ranking
No dedicated tools yet; use run_python with:
- endgame.nlp (TransformerClassifier, DAPT, pseudo-labelling), endgame.vision (VisionBackbone, TTA),
  endgame.audio (SpectrogramTransformer, SEDModel), endgame.automl.MultiModalPredictor;
- endgame.survival (CoxPH, random survival forests, concordance_index);
- endgame.ranking (GroupedRanker for scoring within groups).
describe_api shows each class's parameters.
""",
    "code": """\
# run_python
Runs Python in this server's session. Available names: eg (endgame), np, pd, pl (polars), and
- dataset(id) -> a copy of a session dataset as a pandas DataFrame
- add_dataset(df, name, target=None) -> new dataset id usable by every tool
- model(id) -> a trained model's estimator
- add_model(estimator, dataset_id, name, oof_predictions=None, metrics=None) -> model id (fit it first)
- session: the raw session (datasets, models)
Variables persist between calls. Print what you want to see; the value of a final expression is
returned too. Long output is truncated.
Datasets made by tools or add_dataset are saved as parquet (their "path" in endgame://session/state) and keep
their ids if the server restarts; datasets read with load_data are not saved: load them again.
""",
}
