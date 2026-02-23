# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
