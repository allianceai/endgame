<p align="center">
  <h1 align="center">Endgame</h1>
  <p align="center">
    <strong>A unified framework for tabular, time-series, and multimodal machine learning</strong>
  </p>
  <p align="center">
    Competition-grade models &middot; Research architectures &middot; Production guardrails &middot; Agent-ready via MCP
  </p>
  <p align="center">
    <a href="https://pypi.org/project/endgame-ml/"><img src="https://img.shields.io/pypi/v/endgame-ml.svg" alt="PyPI"></a>
    <a href="https://github.com/allianceai/endgame/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &middot;
    <a href="#what-you-get">What You Get</a> &middot;
    <a href="#installation">Installation</a> &middot;
    <a href="#modules">Modules</a> &middot;
    <a href="https://endgame.readthedocs.io">Documentation</a>
  </p>
</p>

---

Endgame began as a personal research and production toolkit --- a way to unify competition-grade modeling, modern tabular architectures, and deployable pipelines under one coherent API.

Most ML work today means gluing together dozens of libraries with incompatible APIs. Endgame eliminates that. Every component --- from a LightGBM wrapper to a wavelet packet transformer to a Venn-ABERS calibrator --- implements `fit`, `predict`, and `transform`. You can drop any of them into a scikit-learn pipeline, and you can drop any scikit-learn component into an Endgame ensemble.

Endgame doesn't replace scikit-learn. It extends it --- with the models, calibration methods, and production tooling that sklearn doesn't ship.

```python
import endgame as eg
```

## About

Endgame is built and maintained by [Cameron Hamilton](https://github.com/allianceai) --- an ML researcher and engineer focused on tabular learning, AutoML systems, and production-grade modeling infrastructure.

This framework grew out of years of research and production work spanning financial ML, healthcare modeling, and competition pipelines --- systems where interpretability, calibration, and deployment constraints matter as much as raw accuracy. It powers internal tooling at [Alliance AI](https://github.com/allianceai) and serves as the modeling backbone for [FitPilot](https://github.com/allianceai/fitpilot), a nutrition intelligence engine.

I built Endgame because I wanted something that followed scikit-learn's syntax but included *everything* I actually reach for --- SOTA deep tabular models, competition-winning ensemble techniques, unusual methods like fuzzy rule learners and Bayesian network classifiers, and rigorous tools like conformal prediction and Venn-ABERS calibration. Several novel methods developed alongside Endgame are currently under peer review and will be integrated upon publication. Research modules currently under review are temporarily withheld from the public release and will be added following publication.

## Who This Is For

- **ML researchers** building new tabular models or running large-scale benchmarks
- **Engineers** shipping production systems that need interpretability, calibration, or deployment constraints
- **Competitors and data scientists** who want competition-winning defaults without the glue code
- **Teams needing auditable models** --- 30+ glass-box estimators with the same `fit`/`predict` API

## Quick Start

### Train and evaluate a model

```python
import endgame as eg
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load data
credit = fetch_openml(data_id=31, as_frame=False, parser="auto")
X_train, X_test, y_train, y_test = train_test_split(
    credit.data, credit.target, test_size=0.3, random_state=42
)

# Train with competition-winning defaults
model = eg.models.LGBMWrapper(preset="endgame")
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

### AutoML with guardrails and deployment constraints

```python
from endgame.automl import TabularPredictor, DeploymentConstraints

predictor = TabularPredictor(
    label="target",
    presets="good_quality",
    guardrails_strict=True,  # Abort on target leakage or critical data issues
    constraints=DeploymentConstraints(max_predict_latency_ms=10.0),
)
predictor.fit(train_df, time_limit=3600)

predictions = predictor.predict(test_df)
report = predictor.report()       # Full performance report
explanations = predictor.explain() # SHAP feature importances
```

### Generate a full evaluation report

```python
from endgame.visualization import ClassificationReport

report = ClassificationReport(
    model, X_test, y_test,
    feature_names=credit.feature_names,
    model_name="LightGBM",
    dataset_name="German Credit",
)
report.save("evaluation_report.html")
```

Self-contained HTML with metrics, confusion matrix, ROC curve, precision-recall curve, calibration plot, feature importances, and prediction histograms.

### Build an optimized ensemble

```python
from endgame.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sl = SuperLearner(
    base_estimators=[
        ("lgbm", eg.models.LGBMWrapper(preset="endgame")),
        ("xgb", eg.models.XGBWrapper(preset="endgame")),
        ("rf", RandomForestClassifier(n_estimators=200)),
        ("lr", LogisticRegression(max_iter=1000)),
    ],
    meta_learner="nnls",  # non-negative least squares (convex combination)
    cv=5,
)
sl.fit(X_train, y_train)
print(f"Super Learner weights: {dict(zip([n for n,_ in sl.base_estimators], sl.coef_))}")
```

### Interpretable models for regulated domains

```python
from endgame.models import EBMClassifier
from endgame.models.interpretable import (
    GOSDTClassifier,      # Provably optimal sparse decision trees
    CORELSClassifier,     # Certifiably optimal rule lists
    SLIMClassifier,       # Integer risk scorecards
    GAMClassifier,        # Smooth shape functions
)

# Glass-box boosting with automatic interactions
ebm = EBMClassifier(interactions=10)
ebm.fit(X_train, y_train)
print(f"Accuracy: {ebm.score(X_test, y_test):.4f}")
ebm.explain_global()  # Per-feature shape functions
ebm.explain_local(X_test[:1])  # Per-prediction breakdown

# Hand-scoreable risk scorecard (no computer needed)
slim = SLIMClassifier(max_coef=5)
slim.fit(X_train, y_train)
print(slim.get_scorecard())
# age > 50:    +2
# income < 30k: +3
# employed:    -1
# intercept:   -2
# Score >= 3 -> high risk
```

### Agent-ready: build ML pipelines with natural language

Endgame is one of the first ML frameworks designed to be driven by LLM agents. It ships a [Model Context Protocol](docs/guides/mcp_server.md) (MCP) server that lets any LLM host autonomously build, evaluate, and explain production-grade models:

```
You: Load the German Credit dataset and build me the best classifier you can.

LLM -> load_data(source="openml:credit-g", target_column="class")
    -> recommend_models(dataset_id="ds_a1b2c3d4", time_budget="medium")
    -> train_model(dataset_id="ds_a1b2c3d4", model_name="lgbm")
    -> train_model(dataset_id="ds_a1b2c3d4", model_name="xgb")
    -> evaluate_model(model_id="model_e5f6g7h8")
    -> create_report(model_id="model_e5f6g7h8")
    -> export_script(model_id="model_e5f6g7h8")
```

Setup takes one line in your project:

```json
// .mcp.json
{
  "mcpServers": {
    "endgame": {
      "command": "python",
      "args": ["-m", "endgame.mcp"]
    }
  }
}
```

See the [MCP Server Guide](docs/guides/mcp_server.md) for full documentation.

## What You Get

- **100+ supervised models.** GBDTs, deep tabular architectures (FT-Transformer, SAINT, TabPFN v2.5, NODE, GRANDE, TabR), custom trees (Rotation Forest, C5.0, Oblique), Bayesian classifiers, rule learners, kernel methods, and interpretable models --- plus dedicated modules for time series, signal processing, vision, NLP, and audio.

- **30+ interpretable models.** Explainable Boosting Machines, GAMs (pyGAM, GAMI-Net, NODE-GAM, NAM), optimal rule lists (CORELS), optimal sparse trees (GOSDT), integer risk scorecards (SLIM, FasterRisk), symbolic regression (PySR), and fuzzy rule learners (FURIA) --- all with the same `fit`/`predict` API.

- **Depth, not just numbers.** Conformal prediction with finite-sample coverage guarantees. Venn-ABERS calibration. 8 cross-validation strategies including combinatorial purged CV for finance. Adversarial validation for drift detection. 42 self-contained interactive HTML visualizations.

- **Ensembles that win.** Super Learner with NNLS-optimal weighting. Bayesian Model Averaging. Negative Correlation Learning. Cascade ensembles with early exit. Snapshot ensembles. Hill climbing. Optuna-optimized blending. 22 ensemble methods total.

- **Full AutoML.** 15-stage pipeline: profiling, quality guardrails, preprocessing, feature engineering, augmentation, model selection, training, constraint checking, HPO, ensembling, threshold optimization, calibration, post-training, and explainability. Time-budgeted with dynamic reallocation. Includes TabPFN v2.5 as a default model for datasets under 50K samples.

- **Agent-ready architecture.** One of the first ML libraries designed to be driven by LLM agents. Ships a [Model Context Protocol](docs/guides/mcp_server.md) (MCP) server with 20 tools and 6 resources, allowing AI agents (Claude Code, Claude Desktop, VS Code Copilot) to autonomously build, evaluate, and explain production-grade models.

- **No lock-in.** Every estimator is a scikit-learn estimator. Your existing code works. Your existing pipelines work. Endgame adds to your toolkit --- it doesn't replace it.

## Installation

```bash
# Core (numpy, polars, scikit-learn, optuna)
pip install endgame-ml

# With gradient boosting + deep tabular models
pip install endgame-ml[tabular]

# Full installation (all domains)
pip install endgame-ml[all]
```

<details>
<summary>Optional dependency groups</summary>

| Group | Includes |
|---|---|
| `tabular` | XGBoost, LightGBM, CatBoost, PyTorch, EBM, TabPFN |
| `vision` | timm, albumentations, segmentation-models-pytorch |
| `nlp` | Transformers, tokenizers, bitsandbytes |
| `audio` | librosa, torchaudio |
| `benchmark` | OpenML, pymfe |
| `calibration` | scipy |
| `explain` | SHAP, LIME, DiCE |
| `fairness` | fairlearn |
| `deployment` | ONNX, ONNX Runtime, skl2onnx, hummingbird-ml |
| `tracking` | MLflow experiment tracking |
| `mcp` | MCP server for LLM integration |
| `all` | Everything above |
| `dev` | pytest, ruff, mypy |

</details>

## Modules

Endgame is organized into 26 modules following the ML workflow:

### Core Modeling

| Module | Description | Key Classes |
|---|---|---|
| `eg.models` | 100+ estimators: GBDTs, deep tabular, trees, rules, Bayesian, kernel, neural, baselines | `LGBMWrapper`, `FTTransformerClassifier`, `RotationForestClassifier`, `TANClassifier` |
| `eg.models.interpretable` | 30+ glass-box models: EBM, GAM, GAMI-Net, NODE-GAM, NAM, CORELS, GOSDT, SLIM, FasterRisk, PySR | `EBMClassifier`, `GAMClassifier`, `GOSDTClassifier`, `CORELSClassifier`, `SLIMClassifier` |
| `eg.ensemble` | 22 ensemble methods: voting, bagging, boosting, stacking, Super Learner, BMA, NCL, cascades | `SuperLearner`, `VotingClassifier`, `AdaBoostClassifier`, `CascadeEnsemble` |
| `eg.calibration` | Conformal prediction, Venn-ABERS, temperature/Platt/beta/isotonic scaling | `ConformalClassifier`, `VennABERS`, `TemperatureScaling` |
| `eg.tune` | Optuna integration with domain-specific search spaces | `OptunaOptimizer` |

### Data & Features

| Module | Description | Key Classes |
|---|---|---|
| `eg.preprocessing` | 45+ transformers: encoding, imputation, aggregation, imbalanced learning (25 samplers), noise detection | `SafeTargetEncoder`, `AutoAggregator`, `MissForestImputer`, `SMOTEResampler` |
| `eg.validation` | 8 CV strategies, adversarial validation, drift detection | `PurgedTimeSeriesSplit`, `CombinatorialPurgedKFold`, `AdversarialValidator` |
| `eg.anomaly` | Isolation Forest, LOF, GritBot, PyOD (39+ algorithms) | `IsolationForestDetector`, `GritBotDetector`, `PyODDetector` |
| `eg.semi_supervised` | Self-training for classification and regression | `SelfTrainingClassifier`, `SelfTrainingRegressor` |

### Domain-Specific

| Module | Description | Key Classes |
|---|---|---|
| `eg.timeseries` | 31 classes: statistical + neural forecasting, ROCKET/HYDRA classification | `AutoARIMAForecaster`, `NBEATSForecaster`, `MiniRocketClassifier` |
| `eg.signal` | 41 transforms: filtering, spectral analysis, wavelets, entropy, complexity, spatial | `ButterworthFilter`, `WelchPSD`, `CSP`, `PermutationEntropy` |
| `eg.vision` | timm backbones, TTA, WBF, segmentation, augmentation pipelines | `VisionBackbone`, `WeightedBoxesFusion` |
| `eg.nlp` | Transformers, DAPT, pseudo-labeling, back-translation, LLM utilities | `TransformerClassifier`, `DomainAdaptivePretrainer` |
| `eg.audio` | Spectrograms, PCEN, sound event detection, audio augmentation | `SEDModel`, `SpectrogramTransformer` |

### Analysis & Infrastructure

| Module | Description |
|---|---|
| `eg.automl` | Full AutoML: 15-stage pipeline with quality guardrails, HPO, constraint checking, explainability, feedback loop, and performance reports |
| `eg.visualization` | 42 interactive chart types + model reports, all self-contained HTML |
| `eg.tracking` | Experiment tracking: MLflow, console logger, abstract interface |
| `eg.mcp` | MCP server: 20 tools + 6 resources for LLM-powered ML pipelines |
| `eg.explain` | SHAP, LIME, PDP, feature interactions, counterfactuals |
| `eg.fairness` | Fairness metrics, bias mitigation, HTML reports |
| `eg.benchmark` | OpenML suite loading, meta-learning, learning curves |
| `eg.quick` | One-line model training and comparison |
| `eg.persistence` | Model save/load, ONNX export, model serving |
| `eg.kaggle` | Competition management, submissions, project scaffolding |

## Design Principles

1. **Sklearn interface everywhere.** Every estimator implements `fit`/`predict`/`transform`. No proprietary APIs to learn. Works inside `Pipeline`, `GridSearchCV`, `cross_val_score`.

2. **Polars-first preprocessing.** Tabular transformations use `pl.LazyFrame` internally for speed and memory efficiency, while accepting pandas/numpy input transparently.

3. **Explicit over implicit.** No magic. Every technique requires explicit invocation. You control the pipeline, the features, the ensemble, the calibration.

4. **Depth over convenience.** Conformal prediction sets, calibration curves, decision rules, feature importances --- these are first-class citizens, not afterthoughts.

5. **Production-aware defaults.** Quality guardrails catch target leakage and data drift before training starts. Deployment constraints enforce latency and model size limits. Calibration ensures predicted probabilities mean something.

6. **Self-contained outputs.** Every visualization generates a standalone HTML file with all CSS and JavaScript inlined. No CDN dependencies, no network required. Open it anywhere, share it with anyone.

## How Endgame Extends the Ecosystem

Endgame is fully scikit-learn compatible --- it adds to your toolkit rather than replacing it. Here's what it brings beyond existing frameworks:

| Capability | Endgame | scikit-learn | AutoGluon | PyCaret |
|---|---|---|---|---|
| Sklearn-compatible API | Yes | Yes | Partial | Partial |
| Deep tabular models | 15+ (FT-Transformer, SAINT, TabPFN, ...) | --- | 5+ | --- |
| Conformal prediction | Classification + regression | --- | --- | --- |
| Signal processing | 41 transforms | --- | --- | --- |
| Time series classification (ROCKET) | Yes | --- | --- | --- |
| Interactive visualizations | 42 self-contained HTML types | --- | Moderate | Limited |
| Ensemble optimization (Super Learner, BMA) | Yes | Basic | Yes | Basic |
| Imbalanced learning | 25 samplers | --- | --- | Yes |
| Interpretable models (EBM, GAM, CORELS, GOSDT, SLIM) | 30+ | --- | Partial | --- |
| Custom tree algorithms (Rotation Forest, C5.0) | Yes | --- | --- | --- |
| Production guardrails (leakage, drift, constraints) | Yes | --- | --- | --- |
| Experiment tracking (MLflow) | Yes | --- | --- | Yes |
| AutoML with deployment constraints | Yes | --- | --- | --- |
| LLM agent integration (MCP) | 20 tools + 6 resources | --- | --- | --- |

## Roadmap

Endgame is under active development. Upcoming work includes:

- Integration of novel research methods (currently under peer review) for tabular learning and data augmentation
- Expanded AutoML search strategies and meta-learning
- Additional domain-specific predictors (multimodal, geospatial)
- Documentation site and tutorials

See [ROADMAP.md](ROADMAP.md) for the full implementation status.

<details>
<summary>Project structure</summary>

```
endgame/
├── core/              # Base classes, Polars ops, config, types
├── models/            # 100+ estimators across 12 submodules
│   ├── wrappers.py    #   GBDT wrappers (LightGBM, XGBoost, CatBoost)
│   ├── tabular/       #   Deep tabular (FT-Transformer, SAINT, NODE, ...)
│   ├── trees/         #   Custom trees (Rotation Forest, C5.0, Oblique)
│   ├── bayesian/      #   Bayesian classifiers (TAN, KDB, ESKDB)
│   ├── rules/         #   Rule learners (RuleFit, FURIA)
│   ├── interpretable/ #   EBM, GAM, GAMI-Net, NODE-GAM, CORELS, GOSDT, SLIM, FasterRisk
│   ├── neural/        #   MLP, EmbeddingMLP, TabNet
│   ├── kernel/        #   GP, SVM
│   ├── baselines/     #   ELM, NaiveBayes, LDA/QDA, KNN, Linear
│   └── probabilistic/ #   BART, NGBoost
├── ensemble/          # 22 ensemble methods
├── calibration/       # Conformal prediction, probability calibration
├── preprocessing/     # 45+ transformers
├── validation/        # 8 CV strategies, adversarial validation
├── visualization/     # 42 chart types + model reports
├── timeseries/        # Forecasting + classification
├── signal/            # 41 signal processing transforms
├── automl/            # Full AutoML system
├── tracking/          # Experiment tracking (MLflow, console)
├── anomaly/           # Anomaly detection
├── vision/            # Computer vision
├── nlp/               # Natural language processing
├── audio/             # Audio processing
├── benchmark/         # Benchmarking + meta-learning
├── semi_supervised/   # Self-training
├── tune/              # Hyperparameter optimization
├── quick/             # One-line API
├── mcp/               # MCP server for LLM integration
├── explain/           # SHAP, LIME, PDP, counterfactuals
├── fairness/          # Fairness metrics, bias mitigation
├── persistence/       # Save/load, ONNX export, model serving
├── kaggle/            # Competition management
└── utils/             # Metrics, reproducibility
```

</details>

<details>
<summary>Dependencies</summary>

**Core** (always required): numpy, polars, scikit-learn, optuna, scipy

**Optional** (installed per domain):
- **Tabular**: xgboost, lightgbm, catboost, pytorch, interpret
- **Vision**: timm, albumentations, segmentation-models-pytorch
- **NLP**: transformers, tokenizers, bitsandbytes
- **Audio**: librosa, torchaudio
- **Time Series**: statsforecast, darts, tsfresh
- **Signal**: pywavelets
- **Benchmark**: openml, pymfe
- **Anomaly**: pyod
- **Explain**: shap, lime, dice-ml
- **Fairness**: fairlearn
- **Deployment**: onnx, onnxruntime, skl2onnx, hummingbird-ml
- **Tracking**: mlflow
- **MCP**: mcp (for LLM server integration)

</details>

<details>
<summary>Development</summary>

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_ensemble.py -v

# Type checking
mypy endgame/

# Linting
ruff check endgame/
```

</details>

## Commercial Support

Endgame is built and maintained by [Cameron Hamilton](https://github.com/allianceai), ML engineer at Alliance AI. For consulting on production ML pipelines, auditable AI in finance and healthcare, or custom integration work, reach out directly via [GitHub](https://github.com/allianceai) or [LinkedIn](https://linkedin.com/in/cameron-r-hamilton).

## Citation

If you use Endgame in your research, please cite:

```bibtex
@software{endgame2026,
  title     = {Endgame: A Unified Framework for Machine Learning},
  author    = {Hamilton, Cameron},
  year      = {2026},
  url       = {https://github.com/allianceai/endgame},
  version   = {0.7.0-alpha},
  license   = {Apache-2.0},
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Development of Endgame is ongoing and public. Feedback, contributions, and research collaboration are welcome.

---

<p align="center">
  <em>Classic to SOTA. Research to production to competition. Go deeper.</em>
</p>
