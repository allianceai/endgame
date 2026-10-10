# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `endgame.signal.sample_entropy` and `approximate_entropy` counted template matches with a Python double loop: 7 s on 1,500 points, hours for per-player tracking series. They use a KD-tree now (7 ms and 17 ms on 1,500 points) with identical results.

### Added
- MCP `engineer_features` aggregate takes `within` (e.g. `["event_id"]`): signal features are computed per continuous recording and averaged per entity, instead of over an entity's recordings joined end to end. Its help names a few quantiles (`q10`, `q90`) instead of a `q05..q95` range an agent read as all nineteen, and a timeout says what to cut.

## [1.3.0] - 2026-10-09

### Added
- MCP server: agents can reach all 31 modules and are told how to run a full experiment, not just a GBDT.
  - Guidance: server instructions name every module and the workflow; `guide(topic)` (workflow, validation, features, selection, models, ensembling, small data, time series, beyond tabular, code); resources `endgame://catalog/modules` and `endgame://guide/workflow`; examples rewritten around full experiments.
  - Discovery: `list_modules` (every module, its members, search across modules) and `describe_api` (signature and docstring of any class or function).
  - `engineer_features`: aggregate a long table per entity with statistics and `eg.signal` features (sample/permutation/spectral entropy, fractal dimension, Hurst, Hjorth, dominant frequency, ...), join tables, within-group normalisation, formulas, interactions, lags/rolling, out-of-fold target encoding, frequency encoding, datetime parts, ranks, AutoAggregator.
  - `select_features`: 18 `eg.feature_selection` methods (mrmr, boruta, stability, knockoff, null importance, SHAP, ...), with `apply_to` for held-out sets.
  - `transform_data`: any transformer by class path (Endgame or sklearn), including resamplers and `apply_to`.
  - `compare_models`: several models on the same folds, ranked, each kept with out-of-fold predictions; failures are reported per model. `ensemble`: hill climbing, stacking (nested-CV score), Optuna-optimised weights, mean or rank average over those out-of-fold predictions; the result predicts like any model.
  - `run_python`: Python in the session with `eg` and the session's datasets and models (annotated destructive; `ENDGAME_MCP_ALLOW_CODE=0` removes it).
  - `train_model` takes any estimator class path and `group_column` (grouped folds); classifiers keep out-of-fold probabilities (multiclass too); `load_data(columns=...)` reads a subset of a large file.
- `recommend_models` (and `endgame.automl.model_registry.recommend_portfolio`) recommends tabular foundation models ranked by TabArena Elo on tables up to 50k rows (without a GPU only the faster ones), more families with a bigger budget, a linear baseline, and lists models that need a package or licence. Before, a 510-row table got only GBDTs, a GP, linear and ELM. AutoML's `high_quality`, `best_quality` and `all` pools include Kumo-Tabular, TabPFN-3.5, TabICL and Causilo, skipped above 50k rows or without the TabPFN licence.
- `tabpfn_licence_available(model_version)` asks Prior Labs' licence server whether the token accepted that model's licence (a token can cover TabPFN-2.5 but not 3.5).
- `KaggleClient.notebook_status` and the MCP tool `kaggle_notebook_status`: whether a pushed notebook's Kaggle run is queued, running, complete or failed, its failure message and log, and optionally its output files.

### Fixed
- MCP `preprocess` target encoding used each category's mean target over all rows, which leaks the target (a category seen once was encoded as its own label); it is now out-of-fold (`SafeTargetEncoder`).
- MCP training no longer uses an integer id column (`player_id`, `nfl_id`, `ID`; unique per row) as a feature; like unique text ids it is listed in `dropped_columns`.

## [1.2.1] - 2026-10-09

### Fixed
- `eg.quick.classify`, `regress` and `compare` pick TabPFN on their own only when its weights can load without a licence prompt: tabpfn installed and a Prior Labs token available (`TABPFN_TOKEN` or a cached one). In 1.2.0, with tabpfn >= 7 installed and no licence accepted, `quick.classify` on small numeric data raised `TabPFNLicenseError` (or opened a browser login) instead of training another model. A TabPFN model you ask for by name still raises the licence error, whose message says how to accept. New: `endgame.models.tabular.tabpfn.tabpfn_licence_available()`.
- Tests: the Kaggle client tests run without the kaggle package, and the TabPFN 2.5 tests that download weights skip when tabpfn is installed without a licence token.

## [1.2.0] - 2026-10-09

### Fixed
- `KaggleClient` works with kaggle >= 1.7 (kagglesdk): `list_competitions`/`get_competition` crashed on the wrapped list response, and competition refs, file sizes, entry status, submissions, leaderboard and dataset fields read camelCase names that no longer exist (so `user_has_entered` was always False and sizes 0). `Competition` finds data files in sub-folders, as newer competitions ship them, and downloads hard-link out of the kagglehub cache instead of copying (a second copy of a multi-GB dataset).
- MCP `train_model(time_ordered=True)` failed on every classification task ("a mix of binary and unknown targets"): fold predictions went into an object array. They now keep their dtype, and binary tasks also report out-of-fold ROC AUC on the scored (later) rows.
- `TabPFNv2Classifier`/`TabPFNv2Regressor` pin the v2 checkpoint under `tabpfn >= 8`; they passed no checkpoint, so tabpfn 9 gave them its default TabPFN-3.5 (licence-gated, so they fell back to kNN).
- TabPFN wrappers re-raise a licence the user has not accepted (`TabPFNLicenseError`, e.g. TabPFN-3.5 before accepting at ux.priorlabs.ai) instead of falling back to kNN; the MCP server sets `TABPFN_NO_BROWSER=1` (unless set) so that licence flow raises instead of opening a browser and reading an API key from stdin, the protocol stream. The MCP registry entry for TabFM no longer claims native NaN support, so the server imputes for it.
- `TabDPTClassifier`/`TabDPTRegressor` never ran TabDPT: they passed `n_estimators`/`random_state` to a constructor no tabdpt release accepts, and the TypeError dropped them to the kNN approximation with only a warning. Ensembling and seeding now go to tabdpt's predict (`n_ensembles`, `seed`; classification through `ensemble_predict_proba`) and flash attention is enabled only on Ampere or newer GPUs. Works with tabdpt 1.1 to 1.3; `tabdpt>=1.3.1` is TabArena's TabDPT-1.3.
- `TabPFN25Classifier`/`TabPFN25Regressor` pin the requested checkpoint under `tabpfn >= 8` (whose constructor dropped `model_version` and defaults to TabPFN-3); the checkpoint used is recorded as `model_path_`. Previously the `model_version` argument was silently ignored on tabpfn 8.x.
- Stack passthrough imputation now reuses training statistics; single-patient and batch predictions agree. Unavailable/single-class blocks use an explicit class-prior fallback, with stable patient indices and strict feature schemas.
- ComBat excludes globally constant features from empirical-Bayes priors, rejects confounded/degenerate designs, and bounds convergence. One-active-feature and failed-prior cases require an explicit plain-adjustment policy. Adjustment reports expose raw passthrough and skipped blocks.
- Normative reference markers reject missing/ambiguous membership and are optional at inference; rank, residual degrees of freedom, scale and extrapolation are checked.
- Decision curves reject broadcasting-prone probability shapes and invalid labels/thresholds; optional target prevalence reweights net benefit. DeLong validates binary samples and handles identical predictions without fabricated variance.
- PLS-DA remaps original-label class weights and checks rank and feature identity. Block transforms handle empty training features and partial missingness explicitly.
- CI compatibility: constrain MCP to its supported v1 API, update MLflow tests/default local storage to SQLite, support old/new scikit-learn LassoCV parameters, and fix pandas string-label encoding and repository lint. Non-neural interpretable models no longer eagerly import torch models.

### Changed
- `bootstrap_ci` and `paired_bootstrap_diff` default to ordinary resampling; classification stratification is opt-in. Whole-patient `groups` and explicit `strata` are supported. Paired difference intervals use paired bootstrap draws; the p-value now uses paired score randomization with a plus-one correction and an explicit exchangeability assumption.
- Stacking and scanner diagnostics accept grouped/custom/temporal folds. Initial chronological history is excluded from meta-training. Named block metadata supports fold-local base preprocessing; scanner diagnostics clone preprocessing and impute inside folds, with optional multiple probes.
- MLflow's default local URI is `sqlite:///mlruns.db`; existing filesystem stores are not automatically migrated.

### Added
- Kaggle notebooks: `KaggleClient.list_notebooks` (a competition's public notebooks, hottest first), `read_notebook` (source as markdown plus code, outputs dropped) and `push_notebook` (upload and run a notebook on Kaggle, private by default) and `competition_pages` (the overview, evaluation, timeline, data and rules text, which the public API does not serve; uses the endpoint kaggle.com's own pages call). The MCP server exposes Kaggle as `kaggle_competition`, `kaggle_download`, `kaggle_notebooks`, `kaggle_read_notebook` and `kaggle_push_notebook`.
- MCP server: on a CUDA machine, foundation models train and predict out of process. `train_model` cross-validates them in a child process, and the stored model fits in a single worker process on first use; training anything or using another foundation model stops that worker, freeing its GPU memory. A session can train any number of them on a small GPU (previously the sixth or seventh ran out of memory); going back to an earlier model costs a refit.
- TabArena top-30 models (Oct 2026), all optional imports with install hints, each checked against the real package: `KumoTabularClassifier`/`KumoTabularRegressor` (NVIDIA Kumo-Tabular, TabArena #1; `size="large" | "medium" | "small"`), `LimiX2Classifier`/`LimiX2Regressor`, `MitraClassifier` (Mitra-v2), `iLTMClassifier`/`iLTMRegressor` and `SAPRPTClassifier`/`SAPRPTRegressor` (SAP-RPT-OSS) in `endgame.models.tabular`; the packages' own sklearn estimators `CausiloClassifier`/`CausiloRegressor` and `TabLDMEnhancedClassifier`/`TabLDMEnhancedRegressor` (Xiaomi-TabLDM) re-exported there; `endgame.models.boosters` re-exports ChimeraBoost and CTBoost. `TabPFN25Classifier(model_version="3.5" | "3.5-fast")` selects TabPFN-3.5 (tabpfn >= 9). All of them, plus EXAONE-Tabular, TabFM and TabICL, are in the AutoML model registry, so the MCP server's `list_models`/`train_model` reach them: `kumo_tabular`, `limix2`, `tabpfn_35`, `causilo`, `mitra_v2`, `exaone_tabular`, `tabfm`, `tabldm`, `tabicl`, `iltm`, `sap_rpt`, `chimeraboost`, `ctboost`.
- Tabular foundation-model wrappers `EXAONETabularClassifier` (LG AI Research EXAONE-Tabular 1.0), `TabFMClassifier` (Google TabFM 1.0) and `TabICLClassifier` (TabICL v2), all optional imports with install hints; `TabPFN25Classifier(model_version="3")` selects TabPFN-3.
- Mandatory scientific CI matrix (Python 3.10–3.12), pinned scientific dependencies, neuroCombat/LightGBM reference checks, and scientific regression tests with numerical runtime warnings treated as errors.
- [Imaging validation guide](docs/guides/imaging_validation.md) covering new patients, new sites, repeated visits and group inference, including remaining cohort-specific validation gates.
- `preprocessing.BlockwiseHarmonizer`: one train-fitted ComBat per feature block with the block's own batch column (multi-modal data where each modality has its own scanner); rows lacking a block or its batch pass through, batch levels below `min_batch_n` are left un-adjusted.
- `preprocessing.MissingBlockIndicator`: per-block availability indicator plus mean imputation, so early-fusion models see missingness instead of an imputed modality.
- `utils.paired_bootstrap_diff` and `utils.delong_test`: paired comparison of two models' metrics on aligned subjects; inference requires the stated independence/exchangeability assumptions.
- `utils.decision_curve`: decision-curve analysis (net benefit vs treat-all / treat-none).
- `utils.batch_leakage_check`: cross-validated one-vs-rest AUROC of the features for predicting the batch, before and after harmonisation.
- `models.BlockStackingClassifier`: late-fusion stacking over named feature blocks (one base model per modality, out-of-fold block probabilities, logistic meta-model, optional raw passthrough block, missing-block fallback).
- `preprocessing.NormativeDeviation`: covariate-adjusted deviation (W/z) scores against a reference group, fitted in `fit` only (normative modelling as a pipeline step).
- `models.PLSDAClassifier`: PLS-DA with a logistic head, latent-score `transform`, per-feature importances.
- `utils.bootstrap_ci`: percentile-bootstrap confidence intervals with explicit classification stratification and patient-cluster resampling.
- `preprocessing.ComBatHarmonizer`: ComBat batch harmonization with a fit/transform split, empirical Bayes, biological covariates and explicit unseen-batch handling. Reference parity is tested against `neuroCombat`; validation requires correct fold ownership and cohort-specific checks.

## [1.0.0] - 2026-02-22

First stable release of Endgame.

### Changed
- Version bump from 0.7.0-alpha to 1.0.0
- Updated development status to Production/Stable

### Fixed
- `eg.timeseries`, `eg.signal`, `eg.automl`, `eg.dimensionality_reduction`, `eg.feature_selection` now accessible via top-level lazy loading
- Updated `__all__` to include all public modules

## [0.7.0-alpha] - 2026-02-19

Initial public release of Endgame.

### Added
- **100+ models** with unified sklearn-compatible API: GBDTs (LightGBM, XGBoost, CatBoost), deep tabular (FT-Transformer, SAINT, NODE, TabPFN, NAM, GANDALF), custom trees (Rotation Forest, C5.0/Cubist, Oblique, Evolutionary), rules (RuleFit, FURIA), Bayesian classifiers (TAN, KDB, ESKDB), kernel methods, probabilistic models (NGBoost, BART), and interpretable models (EBM, MARS)
- **Polars-powered preprocessing**: SafeTargetEncoder, AutoAggregator, 18+ resampling methods (SMOTE family, ADASYN, geometric, generative), MICE/MissForest imputation, ConfidentLearning noise detection
- **Ensemble methods**: SuperLearner (NNLS), HillClimbingEnsemble, StackingEnsemble, BlendingEnsemble, ThresholdOptimizer, knowledge distillation
- **Calibration**: Conformal prediction (classification + regression), CQR, Venn-ABERS, Temperature/Platt/Beta/Isotonic scaling
- **Validation**: AdversarialValidator, PurgedTimeSeriesSplit, StratifiedGroupKFold, CPCV
- **42 interactive visualizations**: ROC, PR, calibration, PDP/ICE, waterfall, confusion matrix, Sankey, network diagrams, tree visualizer, and more — all self-contained HTML with no CDN dependencies
- **Signal processing**: filtering (Butterworth, FIR, notch), spectral analysis (FFT, Welch, multitaper), wavelets (CWT, DWT), entropy measures, complexity measures, spatial filtering (CSP)
- **Time series**: statistical forecasters (AutoARIMA, AutoETS, MSTL), neural forecasters (N-BEATS, TFT, PatchTST), ROCKET/Hydra classification
- **Explainability**: SHAP, LIME, PDP, feature interactions, counterfactual explanations
- **Fairness**: demographic parity, equalized odds, bias mitigation (Reweighing, Exponentiated Gradient, Calibrated EqOdds)
- **AutoML**: TabularPredictor with preset system (best_quality, high_quality, good_quality, medium_quality, fast, interpretable) with evolutionary/genetic search strategy
- **Anomaly detection**: Isolation Forest, Extended IF, LOF, GritBot, PyOD wrapper (39+ algorithms)
- **Persistence**: model save/load, ONNX export, ModelServer for inference
- **MCP server**: 20 tools + 6 resources for LLM-powered ML pipelines with consistent categorical encoding, timeout protection, and structured error handling
- **Benchmark suite**: OpenML integration, meta-learning, synthetic data generation
- **Example notebooks**: 6 notebooks covering quickstart, interpretable models, AutoML, ensembles, MCP, and signal/timeseries

### Dependencies
- Core: numpy>=1.24, polars>=0.20, scikit-learn>=1.3, optuna>=3.4, scipy>=1.10, networkx>=3.0
- Optional groups: `tabular`, `vision`, `nlp`, `audio`, `benchmark`, `calibration`, `explain`, `fairness`, `deployment`, `mcp`
