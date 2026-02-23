# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Endgame is a comprehensive machine learning toolkit providing 300+ estimators, transformers, and visualizers across tabular, time series, signal processing, CV, NLP, audio, and multimodal domains. It unifies state-of-the-art and classical methods under a consistent scikit-learn-compatible API for researchers and practitioners.

**Import Convention:** `import endgame as eg`

**Version:** 1.0.0 (see ROADMAP.md for detailed implementation status)

**Core Philosophy:** Classic-to-SOTA under one API. Sklearn-native, Polars-powered, interpretability-first. For research, production, and competition.

## Architecture

The library has the following core modules following the competition workflow:

```
eg.validation              → CV strategies, adversarial validation, drift detection
eg.preprocessing           → Feature engineering (Polars-based), encoding, class balancing
eg.models                  → 100+ models: GBDTs, trees, rules, Bayesian, neural, kernel, baselines
eg.ensemble                → Hill climbing, stacking, blending, threshold optimization
eg.calibration             → Conformal prediction, probability calibration, Venn-ABERS
eg.tune                    → Optuna integration with competition-specific search spaces
eg.explain                 → SHAP, LIME, PDP, feature interactions, counterfactuals
eg.fairness                → Demographic parity, equalized odds, bias mitigation
eg.anomaly                 → Isolation Forest, LOF, GritBot, PyOD integration
eg.semi_supervised         → Self-training for classification and regression
eg.benchmark               → Systematic evaluation, meta-learning, learning curves
eg.quick                   → One-line model training and comparison
eg.vision                  → timm backbones, TTA, WBF, augmentation pipelines
eg.nlp                     → Transformers, DAPT, pseudo-labeling, LLM utilities
eg.audio                   → Spectrogram conversion, SED models, audio augmentation
eg.timeseries              → Forecasting (statistical + neural), ROCKET classification
eg.signal                  → Filtering, spectral analysis, wavelets, entropy, complexity
eg.kaggle                  → Competition management, submissions, project scaffolding
eg.persistence             → Model save/load, ONNX export, model serving
eg.utils                   → Metrics, submission helpers, Sharpe ratio analysis
eg.automl                  → Intelligent AutoML framework (TabularPredictor, multi-modal)
eg.clustering              → 16 clustering algorithms with auto-selection
eg.dimensionality_reduction → PCA, UMAP, t-SNE, TriMAP, PHATE, VAE
eg.feature_selection       → 16+ methods: filter, wrapper, importance-based, advanced
eg.visualization           → 42 interactive HTML chart types
eg.tracking                → Experiment logging (MLflow, console)
eg.mcp                     → MCP server for LLM-powered ML pipelines
```

**Dependency Flow:** validation → preprocessing → models/vision/nlp/audio/signal/timeseries → ensemble → calibration

## Key Design Decisions

1. **Sklearn Interface**: All estimators implement `fit`/`predict`/`transform` for pipeline compatibility
2. **Polars-First**: Tabular preprocessing converts to `pl.LazyFrame` internally (accepts pandas/numpy input)
3. **Configuration Presets**: Competition-winning hyperparameters as defaults (e.g., `preset='endgame'`)
4. **Explicit Over Implicit**: No magic - every technique requires explicit invocation
5. **Lazy Loading**: Heavy modules (models, vision, nlp, audio, benchmark, kaggle, quick, visualization, persistence, explain, tracking, timeseries, signal, automl, dimensionality_reduction, feature_selection) loaded on demand

## Directory Structure

```
endgame/
├── core/                       # Base classes, Polars ops, config, types
├── validation/                 # AdversarialValidator, CV splitters (CPCV, StratifiedGroupKFold)
├── preprocessing/              # Encoders, aggregation, feature selection, DAE, imbalanced learning
├── models/
│   ├── wrappers.py             # Unified GBDT interface (LightGBM/XGBoost/CatBoost)
│   ├── trees/                  # Rotation Forest, C5.0/Cubist, Oblique, Quantile, Evolutionary
│   ├── rules/                  # RuleFit, FURIA
│   ├── bayesian/               # TAN, KDB, ESKDB, EBMC, AutoSLE, NeuralKDB
│   ├── tabular/                # FT-Transformer, SAINT, NODE, TabPFN, NAM, GANDALF, TabularResNet
│   ├── neural/                 # MLP, EmbeddingMLP, TabNet
│   ├── kernel/                 # GP, SVM
│   ├── baselines/              # ELM, NaiveBayes, LDA/QDA/RDA, KNN, Linear
│   ├── probabilistic/          # BART, NGBoost
│   ├── ordinal/                # Ordinal regression (mord wrappers)
│   ├── symbolic/               # PySR symbolic regression
│   ├── subgroup/               # PRIM bump hunting
│   └── ebm.py                  # Explainable Boosting Machines
├── ensemble/                   # Hill climbing, stacking, blending, threshold optimization
├── calibration/                # Conformal prediction, scaling methods, Venn-ABERS
├── anomaly/                    # Isolation Forest, Extended IF, LOF, GritBot, PyOD
├── semi_supervised/            # Self-training classifier/regressor
├── tune/                       # Optuna optimizer with preset search spaces
├── benchmark/                  # Suite loading, meta-learning, learning curves, synthetic data
├── quick/                      # One-line API (classify, regress, compare)
├── vision/                     # Backbones, TTA, WBF, segmentation
├── nlp/                        # Transformers, translation, DAPT, LLM utilities
├── audio/                      # Spectrograms, PCEN, SED
├── timeseries/                 # Forecasting (baselines, statsforecast, Darts), ROCKET classification
├── signal/                     # Filtering, spectral, wavelets, entropy, complexity, spatial, connectivity
├── explain/                    # SHAP, LIME, PDP, interactions, counterfactuals
├── fairness/                   # Fairness metrics, bias mitigation, reports
├── persistence/                # Model save/load, ONNX export, model serving
├── kaggle/                     # Competition client, project scaffolding
├── automl/                     # Intelligent AutoML framework (TabularPredictor, multi-modal)
├── clustering/                 # 16 clustering algorithms with auto-selection
├── dimensionality_reduction/   # PCA, UMAP, TriMAP, PHATE, VAE
├── feature_selection/          # 16+ methods: filter, wrapper, importance-based, advanced
├── visualization/              # 42 interactive HTML chart types
├── tracking/                   # Experiment logging (MLflow, console)
├── mcp/                        # MCP server for LLM-powered ML pipelines
└── utils/                      # Metrics, submission, reproducibility, Sharpe analysis
```

## Key Implemented Classes

### Validation
- `AdversarialValidator`: Detects train/test drift via classifier AUC
- `PurgedTimeSeriesSplit`: Time series CV with purging & embargo
- `StratifiedGroupKFold`, `MultilabelStratifiedKFold`, `AdversarialKFold`
- `CombinatorialPurgedKFold`: CPCV for financial backtesting

### Preprocessing
- `SafeTargetEncoder`: M-estimate smoothing with inner-fold encoding
- `AutoAggregator`: "Magic feature" group aggregations
- `SMOTEResampler`, `ADASYNResampler`, `AutoBalancer`: Full imbalanced learning suite (18 samplers)

### Models (100+ total)
- **GBDTs**: `LGBMWrapper`, `XGBWrapper`, `CatBoostWrapper` with presets
- **Custom Trees**: `RotationForest`, `C50Classifier`, `ObliqueRandomForest`, `QuantileRegressorForest`, `EvolutionaryTree`
- **Rules**: `RuleFitClassifier`, `FURIAClassifier` (fuzzy rules)
- **Bayesian**: `TANClassifier`, `KDBClassifier`, `ESKDBClassifier`, `AutoSLE`
- **Deep Tabular**: `FTTransformer`, `SAINT`, `NODE`, `TabPFN`, `NAM`, `GANDALF`, `TabularResNet`, `TabTransformer`
- **Probabilistic**: `NGBoostClassifier`, `BARTClassifier`
- **Kernel**: `GPClassifier`, `SVMClassifier`
- **Baselines**: `ELMClassifier`, `NaiveBayesClassifier`, `LDA/QDA/RDA`, `KNNClassifier`, `LinearClassifier`
- **Interpretable**: `EBMClassifier`, `MARSClassifier`, `SymbolicRegressor`
- **Subgroup**: `PRIMClassifier` (bump hunting)

### Calibration
- `ConformalClassifier`, `ConformalRegressor`: Prediction sets/intervals with coverage guarantees
- `ConformizedQuantileRegressor`: CQR for adaptive intervals
- `VennABERS`: Distribution-free probability calibration
- `TemperatureScaling`, `PlattScaling`, `BetaCalibration`, `IsotonicCalibration`

### Ensemble
- `HillClimbingEnsemble`: Forward selection optimizing arbitrary metrics
- `StackingEnsemble`, `BlendingEnsemble`, `OptimizedBlender`, `RankAverageBlender`
- `ThresholdOptimizer`: Classification threshold optimization

### Explainability
- `explain()`: One-line model explanation with auto-detection
- `SHAPExplainer`: Tree/linear/kernel/deep SHAP auto-selection
- `LIMEExplainer`: LIME explanations for any model
- `PartialDependence`: 1D and 2D partial dependence plots
- `FeatureInteraction`: H-statistic interaction detection
- `CounterfactualExplainer`: DiCE counterfactual explanations

### Fairness
- `demographic_parity`, `equalized_odds`, `disparate_impact`, `calibration_by_group`
- `ReweighingPreprocessor`: Sample weight adjustment for fairness
- `ExponentiatedGradient`: Fairness-constrained training (fairlearn)
- `CalibratedEqOdds`: Post-processing threshold adjustment per group
- `FairnessReport`: HTML report generation

### Persistence & Deployment
- `save`/`load`: Model serialization with metadata
- `export_onnx()`: ONNX export with auto-backend detection (skl2onnx, hummingbird, torch)
- `ModelServer`: ONNX Runtime inference server

### Anomaly Detection
- `IsolationForestDetector`, `ExtendedIsolationForest`, `LocalOutlierFactorDetector`
- `GritBotDetector`: Rule-based with interpretable context
- `PyODDetector`: Wrapper for 39+ PyOD algorithms

### Time Series
- **Baselines**: `NaiveForecaster`, `SeasonalNaiveForecaster`, `ThetaForecaster`
- **Statistical** (statsforecast): `AutoARIMAForecaster`, `AutoETSForecaster`, `MSTLForecaster`
- **Neural** (Darts): `NBEATSForecaster`, `TFTForecaster`, `PatchTSTForecaster`
- **ROCKET**: `MiniRocketClassifier`, `HydraClassifier` (time series classification)

### Signal Processing
- **Filtering**: `ButterworthFilter`, `FIRFilter`, `SavgolFilter`, `NotchFilter`, `FilterBank`
- **Spectral**: `FFTTransformer`, `WelchPSD`, `MultitaperPSD`, `BandPowerExtractor`
- **Wavelets**: `CWTTransformer`, `DWTTransformer`, `WaveletPacketTransformer`
- **Entropy**: `PermutationEntropy`, `SampleEntropy`, `ApproximateEntropy`
- **Complexity**: `HiguchiFD`, `HurstExponent`, `DFA`, `LempelZivComplexity`
- **Spatial**: `CSP`, `TangentSpace`, `FilterBankCSP`

## Dependencies

Core: numpy, polars, scikit-learn, optuna, scipy, networkx
Tabular: xgboost, lightgbm, catboost, pytorch, ngboost
Vision: timm, albumentations, segmentation-models-pytorch
NLP: transformers, tokenizers, bitsandbytes
Audio: librosa, torchaudio
Time Series: statsforecast, darts, tsfresh, sktime
Signal: scipy, pywavelets (optional)
Benchmark: openml
Anomaly: pyod (optional)

## Common Commands

```bash
# Run tests
pytest tests/ -v

# Run specific test module
pytest tests/test_models.py -v

# Run tests with coverage
pytest tests/ --cov=endgame

# Type checking
mypy endgame/

# Install in development mode
pip install -e .
```

## Code Style Guidelines

1. **Sklearn-compatible API**: All estimators must implement `fit`, `predict`, `predict_proba` (classifiers), `transform` (transformers)
2. **Support `sample_weight`** where applicable
3. **Implement `feature_importances_`** property where meaningful
4. **Include preset hyperparameters**: Use `preset='competition'` pattern for competition defaults
5. **Comprehensive docstrings**: Include examples in Google style
6. **Unit tests**: Cover edge cases, write tests in `tests/test_<module>.py`
7. **Polars-first**: Use `pl.LazyFrame` for tabular transformations

## Notes on Optional Dependencies

Some modules require optional dependencies. Import patterns handle missing packages gracefully:

```python
# Julia-based models (TabPFN, Symbolic) - may conflict with PyTorch
# Import directly from submodule:
from endgame.models.tabular.tabpfn import TabPFNClassifier
from endgame.models.symbolic import SymbolicRegressor

# GANDALF requires pytorch-tabular
from endgame.models.tabular import GANDALFClassifier  # lazy loaded

# PyOD algorithms
from endgame.anomaly import PyODDetector
detector = PyODDetector(algorithm='ECOD')
```
