# Models Guide

Endgame provides 100+ estimators organized into families. All models follow the
scikit-learn interface: `fit`, `predict`, and `predict_proba` (classifiers) or
`transform` (transformers). Every estimator is pipeline-compatible and accepts
`sample_weight` where applicable.

## Model Family Overview

| Family | Key Classes | Best For |
|--------|-------------|----------|
| **GBDTs** | `LGBMWrapper`, `XGBWrapper`, `CatBoostWrapper` | General tabular, competitions |
| **Deep Tabular** | `FTTransformerClassifier`, `SAINTClassifier`, `NODEClassifier`, `TabPFNClassifier`, `NAMClassifier`, `GANDALFClassifier`, `TabularResNetClassifier` | Large datasets, categorical embeddings |
| **Custom Trees** | `RotationForestClassifier`, `C50Classifier`, `ObliqueRandomForestClassifier`, `QuantileRegressorForest`, `EvolutionaryTreeClassifier` | Structured data, diverse ensembles |
| **Rules** | `RuleFitClassifier`, `FURIAClassifier` | Interpretable rule extraction |
| **Bayesian** | `TANClassifier`, `KDBClassifier`, `ESKDBClassifier` | Probabilistic, small data |
| **Kernel** | `GPClassifier`, `SVMClassifier` | Small to medium datasets |
| **Interpretable** | `EBMClassifier`, `MARSClassifier`, `SymbolicRegressor` | Regulatory compliance, auditability |
| **Neural** | `ELMClassifier`, `EmbeddingMLPClassifier`, `TabNetClassifier` | Custom architectures, entity embeddings |
| **Probabilistic** | `NGBoostClassifier`, `BARTClassifier` | Uncertainty quantification |
| **Baselines** | `NaiveBayesClassifier`, `LDAClassifier`, `QDAClassifier`, `RDAClassifier`, `KNNClassifier`, `LinearClassifier` | Benchmarking, ensemble diversity |

---

## Preset System

The `preset` parameter loads competition-winning hyperparameter configurations.
Three presets are available across all GBDT wrappers:

- `'endgame'` — competition-tuned defaults (low learning rate, many trees, early
  stopping). This is the default.
- `'fast'` — higher learning rate, fewer trees. Useful for rapid iteration.
- `'overfit'` — aggressively deep trees, no regularization. Use only for
  ensembling experiments.
- `'custom'` — no preset applied; pass all hyperparameters explicitly.

```python
from endgame.models import LGBMWrapper

# Competition-ready defaults
model = LGBMWrapper(preset='endgame')
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# Fast iteration during feature engineering
quick_model = LGBMWrapper(preset='fast')
quick_model.fit(X_train, y_train)

# Override specific parameters within a preset
model = LGBMWrapper(preset='endgame', num_leaves=63, min_child_samples=50)
```

The `'endgame'` preset sets `learning_rate=0.01`, `n_estimators=10000`, and
relies on early stopping to find the optimal number of rounds. Always pass a
validation set when using this preset.

---

## GBDTs

Gradient boosted decision trees are the default choice for tabular competitions.
All three wrappers share the same interface via `GBDTWrapper`.

```python
from endgame.models import LGBMWrapper, XGBWrapper, CatBoostWrapper

# LightGBM — fastest training, best default performance
lgbm = LGBMWrapper(preset='endgame')
lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
proba = lgbm.predict_proba(X_test)

# XGBoost — strong GPU support, wider ecosystem integration
xgb = XGBWrapper(preset='endgame', use_gpu=True)
xgb.fit(X_train, y_train)

# CatBoost — native categorical feature handling, often best out of the box
catboost = CatBoostWrapper(preset='endgame', categorical_features=['city', 'product'])
catboost.fit(X_train, y_train)
```

Feature importances are available via `model.feature_importances_` after fitting.

---

## Deep Tabular Models

Deep learning models for tabular data. These require PyTorch and are imported
from `endgame.models.tabular`. They tend to shine on datasets with many
categorical features or when pre-trained representations are available.

### FT-Transformer

Feature Tokenizer + Transformer. Strong general-purpose deep tabular model.

```python
from endgame.models.tabular import FTTransformerClassifier

ft = FTTransformerClassifier(
    d_token=192,
    n_blocks=3,
    attention_dropout=0.2,
    n_epochs=100,
    batch_size=512,
)
ft.fit(X_train, y_train)
proba = ft.predict_proba(X_test)
```

### SAINT

Self-Attention and Intersample Attention Transformer. Captures both feature-level
and sample-level interactions.

```python
from endgame.models.tabular import SAINTClassifier

saint = SAINTClassifier(depth=6, heads=8, n_epochs=50)
saint.fit(X_train, y_train)
```

### NODE

Neural Oblivious Decision Ensembles. Differentiable tree structure — fast and
competitive with GBDTs on structured data.

```python
from endgame.models.tabular import NODEClassifier

node = NODEClassifier(num_trees=2048, tree_depth=6, n_epochs=50)
node.fit(X_train, y_train)
```

### NAM

Neural Additive Models. Each feature is modeled by an independent neural network,
enabling per-feature shape functions with neural expressiveness.

```python
from endgame.models.tabular import NAMClassifier

nam = NAMClassifier(hidden_units=[64, 64], n_epochs=100)
nam.fit(X_train, y_train)
# Access per-feature shape functions
contributions = nam.feature_contributions(X_test)
```

### GANDALF

Gated Adaptive Network for Deep Automated Learning of Features. Requires the
`pytorch-tabular` package and should be imported directly.

```python
from endgame.models.tabular.gandalf import GANDALFClassifier

gandalf = GANDALFClassifier(gflu_stages=6, n_epochs=100)
gandalf.fit(X_train, y_train)
```

### TabularResNet

Residual network architecture adapted for tabular data. Straightforward and
reliable with normalization and skip connections.

```python
from endgame.models.tabular import TabularResNetClassifier

resnet = TabularResNetClassifier(
    hidden_dim=256,
    n_layers=4,
    dropout=0.1,
    n_epochs=100,
)
resnet.fit(X_train, y_train)
```

---

## Custom Trees

### Rotation Forest

Applies PCA rotations to random feature subsets before building decision trees.
Increases diversity substantially over standard random forests.

```python
from endgame.models import RotationForestClassifier

rf = RotationForestClassifier(n_estimators=100, n_features_per_subset=3)
rf.fit(X_train, y_train)
```

### C5.0

The classic C5.0 decision tree algorithm.
Includes rule extraction, pruning, and boosting.

```python
from endgame.models import C50Classifier

c50 = C50Classifier(n_trials=10, pruning=True)
c50.fit(X_train, y_train)
rules = c50.get_rules()  # Human-readable rule set
```

### Oblique Random Forest

Uses linear combinations of features at each split, rather than axis-aligned
splits. Captures diagonal decision boundaries.

```python
from endgame.models import ObliqueRandomForestClassifier

orf = ObliqueRandomForestClassifier(n_estimators=100, max_depth=10)
orf.fit(X_train, y_train)
```

### Quantile Regressor Forest

Provides prediction intervals via quantile regression. Each leaf stores the full
empirical distribution of training targets.

```python
from endgame.models import QuantileRegressorForest

qrf = QuantileRegressorForest(n_estimators=200)
qrf.fit(X_train, y_train)
lower, median, upper = qrf.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])
```

### Evolutionary Tree

Optimizes tree structure via evolutionary algorithms rather than greedy splitting.
Finds globally better splits at the cost of training time.

```python
from endgame.models.trees.evtree import EvolutionaryTreeClassifier

evt = EvolutionaryTreeClassifier(population_size=100, n_generations=50)
evt.fit(X_train, y_train)
```

---

## Rule-Based Models

### RuleFit

Extracts linear rules from an ensemble of trees, then fits a sparse linear model
over those rules. The result is a human-readable list of weighted conditions.

```python
from endgame.models import RuleFitClassifier

rulefit = RuleFitClassifier(tree_size=4, max_rules=2000)
rulefit.fit(X_train, y_train)
rules_df = rulefit.get_rules()
print(rules_df[rules_df['importance'] > 0.01])
```

### FURIA

Fuzzy Unordered Rule Induction Algorithm. Produces fuzzy rule sets that handle
overlapping class regions gracefully.

```python
from endgame.models import FURIAClassifier

furia = FURIAClassifier(n_rules=20)
furia.fit(X_train, y_train)
rule_list = furia.rules_  # List of FuzzyRule objects
```

---

## Bayesian Network Classifiers

Bayesian classifiers are well-suited for small datasets where probabilistic
structure is meaningful and calibrated probabilities are important.

### TAN (Tree Augmented Naive Bayes)

Extends Naive Bayes by allowing each feature to have one additional parent
(a single dependency tree over features).

```python
from endgame.models import TANClassifier

tan = TANClassifier()
tan.fit(X_train, y_train)
proba = tan.predict_proba(X_test)
```

### KDB (k-Dependence Bayesian)

Generalizes TAN by allowing each feature to depend on up to `k` other features.
Higher `k` captures more complex dependencies at the cost of data requirements.

```python
from endgame.models import KDBClassifier

kdb = KDBClassifier(k=2)
kdb.fit(X_train, y_train)
```

### ESKDB (Ensemble Smoothed KDB)

Ensemble of KDB classifiers with Laplace smoothing and random structure
perturbation for improved accuracy.

```python
from endgame.models import ESKDBClassifier

eskdb = ESKDBClassifier(k=2, n_estimators=50)
eskdb.fit(X_train, y_train)
```

---

## Kernel Methods

### Gaussian Process Classifier

Provides well-calibrated probabilistic predictions with uncertainty estimates.
Exact GP scales as O(n^3), so use on datasets below ~5,000 samples.

```python
from endgame.models import GPClassifier
from sklearn.gaussian_process.kernels import RBF, Matern

gp = GPClassifier(kernel=Matern(nu=2.5), n_restarts_optimizer=5)
gp.fit(X_train, y_train)
proba = gp.predict_proba(X_test)  # Well-calibrated probabilities
```

### SVM Classifier

Support Vector Machine with kernel selection. Competitive on medium-sized
datasets with fewer than ~50,000 samples.

```python
from endgame.models import SVMClassifier

svm = SVMClassifier(kernel='rbf', C=10.0, probability=True)
svm.fit(X_train, y_train)
```

---

## Interpretable Models

These models are suitable for regulated industries where predictions must be
auditable or explained to non-technical stakeholders.

### EBM (Explainable Boosting Machine)

EBMs are generalized additive models trained with gradient boosting. They achieve
near-GBDT accuracy while remaining fully interpretable via shape functions for
each feature and pairwise interaction.

```python
from endgame.models import EBMClassifier

ebm = EBMClassifier(interactions=15, max_bins=256)
ebm.fit(X_train, y_train)

# Inspect global explanation
ebm.explain_global()

# Local explanation for a single prediction
ebm.explain_local(X_test[:5])
```

EBMs support both classification and regression via `EBMRegressor`.

### MARS (Multivariate Adaptive Regression Splines)

Fits piecewise linear splines with automatic knot selection. Produces explicit
mathematical expressions for each prediction.

```python
from endgame.models import MARSClassifier

mars = MARSClassifier(max_degree=2, max_terms=20)
mars.fit(X_train, y_train)
print(mars.summary())  # Equation with hinge functions
```

### Symbolic Regression

Discovers explicit mathematical formulas via genetic programming. Best for
scientific applications where the functional form matters.

```python
from endgame.models import SymbolicRegressor

sr = SymbolicRegressor(
    population_size=1000,
    generations=20,
    function_set=['add', 'mul', 'sqrt', 'log'],
)
sr.fit(X_train, y_train)
print(sr.best_program_)  # e.g., "0.42 * x1 + sqrt(x2) - 1.7"
```

---

## Neural Models

### ELM (Extreme Learning Machine)

Single hidden-layer network where input weights are randomly assigned and only
the output layer is trained. Extremely fast, useful as a cheap ensemble member.

```python
from endgame.models import ELMClassifier

elm = ELMClassifier(n_hidden=1000, activation='relu')
elm.fit(X_train, y_train)
```

### Embedding MLP

MLP with learned entity embeddings for categorical features. Effective when
categorical cardinality is high (cities, products, user IDs).

```python
from endgame.models.neural import EmbeddingMLPClassifier

mlp = EmbeddingMLPClassifier(
    cat_features=['city', 'product'],
    hidden_layers=[256, 128, 64],
    dropout=0.3,
    n_epochs=100,
)
mlp.fit(X_train, y_train)
```

### TabNet

Attention-based neural network using sequential attention to select features at
each decision step. Provides built-in feature importance.

```python
from endgame.models.neural import TabNetClassifier

tabnet = TabNetClassifier(n_steps=5, gamma=1.5, n_epochs=100)
tabnet.fit(X_train, y_train)
importances = tabnet.feature_importances_
```

---

## Probabilistic Models

### NGBoost

Natural Gradient Boosting outputs full probability distributions rather than
point estimates. Use when calibrated uncertainty is required.

```python
from endgame.models import NGBoostClassifier

ngb = NGBoostClassifier(n_estimators=500, learning_rate=0.01)
ngb.fit(X_train, y_train)

# Returns probability distributions, not just point estimates
distributions = ngb.pred_dist(X_test)
proba = ngb.predict_proba(X_test)
```

### BART (Bayesian Additive Regression Trees)

Fully Bayesian nonparametric model providing posterior distributions over
predictions. Requires `pymc` and `pymc-bart`.

```python
from endgame.models import BARTClassifier

bart = BARTClassifier(m=50, n_samples=1000, tune=500)
bart.fit(X_train, y_train)
proba = bart.predict_proba(X_test)
credible_intervals = bart.predict_interval(X_test, hdi_prob=0.94)
```

---

## Foundation Models

### TabPFN

TabPFN is a prior-fitted network trained on millions of synthetic tabular
datasets. It performs in-context learning — no gradient-based training is
needed at inference time.

```python
from endgame.models.tabular import TabPFNClassifier

# No training loop — model uses the dataset as context directly
tabpfn = TabPFNClassifier(n_ensemble_configurations=32)
tabpfn.fit(X_train, y_train)  # Stores context, no gradient updates
proba = tabpfn.predict_proba(X_test)
```

TabPFN works best on datasets with fewer than 10,000 samples and fewer than
100 features. For larger datasets, use TabPFNv2 or TabPFN25:

```python
from endgame.models.tabular import TabPFNv2Classifier, TabPFN25Classifier

# v2 — extended context window, improved accuracy
tabpfn_v2 = TabPFNv2Classifier()
tabpfn_v2.fit(X_train, y_train)
```

Because TabPFN has large optional dependencies, import it directly from the
submodule rather than from `endgame.models`.

---

## Baseline Models

Lightweight models useful for ensemble diversity and benchmarking.

```python
from endgame.models import (
    NaiveBayesClassifier,
    LDAClassifier,
    QDAClassifier,
    RDAClassifier,
    KNNClassifier,
    LinearClassifier,
)

# Linear discriminant analysis — fast, good baseline for linearly separable data
lda = LDAClassifier(solver='svd')
lda.fit(X_train, y_train)

# Regularized discriminant analysis — blend of LDA and QDA
rda = RDAClassifier(alpha=0.5)
rda.fit(X_train, y_train)

# KNN — strong baseline, no training required
knn = KNNClassifier(n_neighbors=15, weights='distance')
knn.fit(X_train, y_train)
```

---

## Model Selection Guidance

Use the following heuristics as a starting point:

| Situation | Recommended approach |
|-----------|---------------------|
| Small dataset (< 1,000 samples) | `TANClassifier`, `ESKDBClassifier`, `TabPFNClassifier`, `GPClassifier` |
| Medium dataset (1K–100K samples) | `LGBMWrapper` or `XGBWrapper` with `preset='endgame'` |
| Large dataset (> 100K samples) | `LGBMWrapper`, `FTTransformerClassifier`, `TabularResNetClassifier` |
| High-cardinality categoricals | `CatBoostWrapper`, `EmbeddingMLPClassifier`, `SAINTClassifier` |
| Interpretability required | `EBMClassifier`, `RuleFitClassifier`, `MARSClassifier` |
| Regulatory compliance | `EBMClassifier`, `SymbolicRegressor`, `C50Classifier` (with `get_rules()`) |
| Calibrated uncertainty | `NGBoostClassifier`, `BARTClassifier`, `GPClassifier` |
| No training time budget | `TabPFNClassifier` (in-context learning), `ELMClassifier` |
| Ensembling diversity | Mix families: GBDT + rotation forest + ELM + KNN |
| Time series classification | See `eg.timeseries` (`MiniRocketClassifier`, `HydraClassifier`) |

A practical workflow for competitions:

1. Start with `LGBMWrapper(preset='endgame')` as your baseline.
2. Run `eg.benchmark` or `eg.quick.compare()` to survey model families.
3. Build a diverse set of out-of-fold predictions from multiple families.
4. Use `eg.ensemble.HillClimbingEnsemble` or `eg.ensemble.StackingEnsemble`
   to combine them.
5. Calibrate probabilities with `eg.calibration` if log-loss is the metric.

---

## See Also

- [API Reference: models](../api/models) — complete parameter documentation
- [Ensemble Guide](ensembles.md) — combining multiple models
- [Calibration Guide](calibration.md) — probability calibration and conformal prediction
- [Explainability Guide](explainability.md) — SHAP, LIME, and partial dependence
- [Tuning Guide](automl.md) — Optuna integration with preset search spaces
