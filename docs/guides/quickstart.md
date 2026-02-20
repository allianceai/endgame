# Quickstart

This guide walks through the core Endgame workflow from data loading to model
evaluation. After completing it you will have seen the main API entry points and
know where to look for deeper documentation on each topic.

## Import Convention

```python
import endgame as eg
```

Heavy sub-modules (models, vision, nlp, audio, benchmark, kaggle, quick) are
lazy-loaded, so the import is fast even though the library is large.

---

## End-to-End Example

The example below uses scikit-learn's breast cancer dataset and covers the
complete workflow: split, train, evaluate, and visualize.

```python
import endgame as eg
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names.tolist()
class_names = load_breast_cancer().target_names.tolist()

# Split: train / calibration / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train with competition-winning defaults
model = eg.models.LGBMWrapper(preset="endgame")
model.fit(X_train, y_train)

# Evaluate
proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print(f"Test ROC-AUC: {auc:.4f}")

# Visualize feature importances
from endgame.visualization import BarChartVisualizer

bar = BarChartVisualizer.from_importances(model, feature_names=feature_names)
bar.save("feature_importances.html")
```

All Endgame estimators implement the standard scikit-learn interface:
`fit`, `predict`, `predict_proba` (classifiers), and `transform`
(transformers). They drop into any sklearn `Pipeline` without modification.

---

## Model Families

### GBDTs

`LGBMWrapper`, `XGBWrapper`, and `CatBoostWrapper` expose a unified interface
with competition-tuned hyperparameter presets.

```python
import endgame as eg

# LightGBM with competition defaults
lgbm = eg.models.LGBMWrapper(preset="endgame")
lgbm.fit(X_train, y_train)
proba_lgbm = lgbm.predict_proba(X_test)

# XGBoost
xgb = eg.models.XGBWrapper(preset="endgame")
xgb.fit(X_train, y_train)

# CatBoost (handles categoricals natively)
cat = eg.models.CatBoostWrapper(preset="endgame")
cat.fit(X_train, y_train)
```

Available presets: `"fast"`, `"endgame"` (competition defaults).

See [models guide](models.md) for the full list of supported parameters.

### Deep Tabular Models

Deep tabular models are imported directly from their submodule to keep the
top-level namespace lean.

```python
from endgame.models.tabular import FTTransformerClassifier

ft = FTTransformerClassifier(
    n_blocks=3,
    d_token=192,
    n_heads=8,
    n_epochs=100,
)
ft.fit(X_train, y_train)
proba_ft = ft.predict_proba(X_test)
```

Other deep tabular classifiers follow the same import pattern:

```python
from endgame.models.tabular import SAINTClassifier, NODEClassifier, GANDALFClassifier
```

### Interpretable Models

Endgame first-class supports interpretable models that match or approach GBDT
accuracy while remaining explainable.

```python
from endgame.models import EBMClassifier

ebm = EBMClassifier()
ebm.fit(X_train, y_train)
proba_ebm = ebm.predict_proba(X_test)

# EBM exposes per-feature contributions
print(ebm.feature_importances_)
```

Other interpretable options:

```python
from endgame.models.rules import RuleFitClassifier
from endgame.models.symbolic import SymbolicRegressor  # PySR backend
from endgame.models.baselines import LinearClassifier
```

---

## Quick API

The Quick API provides one-line training and multi-model comparison. It runs
stratified cross-validation internally and returns out-of-fold predictions
alongside a fitted model.

### Single Model

```python
import endgame as eg

result = eg.quick.classify(X, y, preset="default", metric="roc_auc")
print(result)  # QuickResult(cv_score=0.9912, metric='roc_auc')

# Access the fitted model and OOF predictions
model = result.model
oof_preds = result.oof_predictions
importances = result.feature_importances
```

Available presets:

| Preset | Models included | CV folds | Typical runtime |
|---|---|---|---|
| `"fast"` | LightGBM, Linear | 3 | ~1 min |
| `"default"` | LightGBM, XGBoost, CatBoost, Linear | 5 | ~5 min |
| `"competition"` | GBDT trio + KNN + ELM + RotationForest | 5 | ~30 min |
| `"interpretable"` | Linear, EBM, NAM | 5 | ~5 min |

### Model Comparison

```python
comparison = eg.quick.compare(X, y, task="classification", preset="default")
print(comparison)
# ComparisonResult:
#   1. lgbm:     0.9934
#   2. xgb:      0.9921
#   3. catboost: 0.9908
#   4. linear:   0.9743

best_model = comparison.best_model
leaderboard = comparison.leaderboard  # List[Dict[str, Any]]
```

See [quick API reference](../api/quick.rst) for regression support
(`eg.quick.regress`) and additional options.

---

## Ensemble Methods

Once you have several models trained, combine them with the ensemble module.

### Super Learner

The Super Learner finds the optimal convex combination of base models using
cross-validated out-of-fold predictions (NNLS by default). It is asymptotically
at least as good as the single best base learner.

```python
from endgame.ensemble import SuperLearner
from endgame.models import EBMClassifier
from endgame.models.baselines import LinearClassifier

sl = SuperLearner(
    base_estimators=[
        ("lgbm", eg.models.LGBMWrapper(preset="endgame")),
        ("ebm", EBMClassifier()),
        ("lr", LinearClassifier()),
    ],
    meta_learner="nnls",  # non-negative least squares
    cv=5,
)
sl.fit(X_train, y_train)
proba_sl = sl.predict_proba(X_test)
```

### Hill Climbing Ensemble

Forward-selection ensemble that greedily adds models while a metric improves.

```python
from endgame.ensemble import HillClimbingEnsemble

hc = HillClimbingEnsemble(metric="roc_auc", n_iterations=20)
hc.fit(oof_preds_list, y_train)   # list of OOF prediction arrays
final_proba = hc.predict(test_preds_list)
```

See [ensemble guide](../api/ensemble.rst) for `StackingEnsemble`,
`BlendingEnsemble`, `RankAverageBlender`, and `ThresholdOptimizer`.

---

## Conformal Prediction

Conformal prediction wraps any classifier and produces prediction *sets* with
a guaranteed coverage probability. At `alpha=0.1`, the true label is contained
in the returned set for at least 90% of new examples (under exchangeability).

```python
from endgame.calibration import ConformalClassifier
import endgame as eg

# Split off a calibration set (separate from test)
X_tr, X_temp, y_tr, y_temp = train_test_split(X_train, y_train, test_size=0.3)
X_cal, X_val, y_cal, y_val = train_test_split(X_temp, y_temp, test_size=0.5)

base_model = eg.models.LGBMWrapper(preset="endgame")

cp = ConformalClassifier(base_model, method="aps", alpha=0.1)
cp.fit(X_tr, y_tr, X_cal, y_cal)   # calibrate on held-out calibration set

# Returns a list of sets — each set contains the plausible class labels
prediction_sets = cp.predict(X_val)
print(prediction_sets[:5])
# [{0}, {1}, {0, 1}, {1}, {0}]

# Verify empirical coverage
coverage = cp.coverage_score(X_val, y_val)
print(f"Empirical coverage: {coverage:.3f}")  # >= 0.90
```

Passing `X_cal=None` causes `ConformalClassifier` to automatically split a
`cal_size` fraction from the training data for calibration.

See [calibration guide](../api/calibration.rst) for `ConformalRegressor`,
`ConformizedQuantileRegressor`, and `VennABERS`.

---

## Adversarial Validation

Adversarial validation detects distribution shift between train and test data
before you ever submit a model. A high AUC means a classifier can easily
distinguish the two splits — indicating that your CV will not correlate with
the leaderboard.

```python
import endgame as eg

av = eg.validation.AdversarialValidator(threshold=0.6)
result = av.check_drift(X_train, X_test)

print(f"Drift AUC: {result.auc_score:.3f}")
print(f"Severity:  {result.drift_severity}")   # 'none', 'mild', or 'severe'
print(f"Top drifting features: {result.drifted_features[:5]}")

if result.drift_severity == "severe":
    # Drop the most drifting features
    drop_cols = result.drifted_features[:5]
    print(f"Consider dropping: {drop_cols}")
```

The default classifier is LightGBM when available, RandomForest otherwise.
You can supply any sklearn-compatible classifier via the `estimator` parameter.

See [validation guide](../api/validation.rst) for `PurgedTimeSeriesSplit`,
`StratifiedGroupKFold`, and `CombinatorialPurgedKFold`.

---

## Interactive Visualization

All Endgame visualizations produce self-contained HTML files with no external
CDN dependencies. Open them in any browser.

### Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier
from endgame.visualization import TreeVisualizer

clf = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
viz = TreeVisualizer(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    title="Breast Cancer Decision Tree",
)
viz.save("tree.html")   # click nodes to expand/collapse
```

### ROC Curve

```python
from endgame.visualization import ROCCurveVisualizer

roc = ROCCurveVisualizer.from_predictions(y_test, proba_lgbm, label="LightGBM")
roc.add_predictions(y_test, proba_ft, label="FT-Transformer")
roc.save("roc_curves.html")
```

### Classification Report

The `ClassificationReport` bundles confusion matrix, ROC/PR curves, calibration
plot, and lift chart into a single interactive HTML page.

```python
from endgame.visualization import ClassificationReport

report = ClassificationReport(
    y_true=y_test,
    y_proba=proba_lgbm,
    feature_names=feature_names,
    class_names=class_names,
)
report.save("classification_report.html")
```

See the [visualization guide](visualization.md) for the complete chart
catalogue (42 chart types including PDP, waterfall / SHAP, parallel
coordinates, and calibration plots).

---

## Next Steps

| Topic | Guide |
|---|---|
| Preprocessing (encoding, feature engineering, balancing) | [preprocessing guide](preprocessing.md) |
| Full model catalogue (100+ estimators) | [models guide](models.md) |
| Hyperparameter tuning with Optuna | [tune API](../api/tune) |
| SHAP, LIME, counterfactuals | [explain API](../api/explain) |
| Fairness metrics and mitigation | [fairness API](../api/fairness) |
| Anomaly detection | [anomaly API](../api/anomaly) |
| Time series forecasting and classification | [timeseries API](../api/timeseries) |
| Signal processing | [signal API](../api/signal) |
| Visualization catalogue | [visualization guide](visualization.md) |
| MCP server (AI assistant integration) | [MCP server guide](mcp_server.md) |
