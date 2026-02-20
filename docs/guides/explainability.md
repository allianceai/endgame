# Explainability Guide

Endgame provides a unified explainability layer that works with any
sklearn-compatible model. Techniques range from one-line auto-explanation to
detailed SHAP, LIME, partial dependence, interaction detection, and
counterfactual generation.

**Import convention:** `import endgame as eg`

---

## The `explain()` One-Liner

`explain()` is the fastest entry point. It inspects the model type, selects the
most appropriate explanation method automatically, and returns a summary plot
plus a `pandas.DataFrame` of feature importances.

```python
from endgame.explain import explain

# Works with tree models, linear models, neural nets, or black-box estimators
report = explain(model, X_test)

# report.importances  — DataFrame, one row per feature
# report.plot()       — displays a bar chart of global importances
```

Auto-detection rules:

| Model type | Method selected |
|---|---|
| Tree-based (GBDT, RF, etc.) | TreeSHAP |
| Linear / GLM | LinearSHAP |
| Neural network (PyTorch) | DeepSHAP |
| Any other estimator | KernelSHAP (slower, model-agnostic) |

Pass `method='lime'` or `method='shap'` to override auto-detection.

---

## SHAP Explanations

`SHAPExplainer` wraps the four SHAP backends behind a single class. It selects
the correct backend at `fit` time based on the model type.

```python
from endgame.explain import SHAPExplainer

explainer = SHAPExplainer(model)
explainer.fit(X_train)           # background dataset for KernelSHAP / DeepSHAP

# Global explanation — mean absolute SHAP value per feature
global_imp = explainer.global_importance(X_test)
explainer.plot_bar(X_test)       # bar chart of global importances
explainer.plot_beeswarm(X_test)  # beeswarm (violin-style) summary plot

# Local explanation — SHAP values for a single prediction
shap_values = explainer.explain(X_test)   # shape: (n_samples, n_features)
explainer.plot_waterfall(X_test[0])       # waterfall for the first sample
explainer.plot_force(X_test[0])           # force plot
```

### Interaction values

```python
# Pairwise SHAP interaction values (TreeSHAP only)
interactions = explainer.interaction_values(X_test)  # (n, n_features, n_features)
explainer.plot_interaction_heatmap(X_test)
```

---

## LIME Explanations

`LIMEExplainer` fits a local linear model around each prediction using
perturbed samples. It is model-agnostic and works with any `predict_proba`
or `predict` method.

```python
from endgame.explain import LIMEExplainer

lime = LIMEExplainer(model, mode='classification')  # 'classification' | 'regression'
lime.fit(X_train)

# Explain one instance
explanation = lime.explain(X_test[0], num_features=10)
explanation.show_in_notebook()       # interactive HTML widget
print(explanation.as_dataframe())    # feature weights as DataFrame

# Explain a batch and aggregate
batch_exp = lime.explain_batch(X_test[:50], num_features=10)
lime.plot_summary(batch_exp)
```

---

## Partial Dependence

`PartialDependence` computes marginal effect of one or two features while
averaging over all other features. Use this to understand the shape of a
feature's relationship with the target.

```python
from endgame.explain import PartialDependence

pdp = PartialDependence(model)
pdp.fit(X_train)

# 1D partial dependence
pdp.plot(X_test, feature='age')

# 2D interaction surface (two features simultaneously)
pdp.plot_2d(X_test, features=['age', 'income'])

# Individual Conditional Expectation (ICE) curves
pdp.plot_ice(X_test, feature='age', n_samples=50, center=True)

# Return raw values without plotting
grid, avg, individual = pdp.compute(X_test, feature='age')
```

---

## Feature Interactions

`FeatureInteraction` measures pairwise and higher-order interactions using
Friedman's H-statistic. High H-statistic values indicate that the joint effect
of two features cannot be decomposed into separate individual effects.

```python
from endgame.explain import FeatureInteraction

fi = FeatureInteraction(model)
fi.fit(X_train)

# Overall interaction strength for every feature (one number per feature)
h_overall = fi.overall_strength(X_test)

# Pairwise H-statistic matrix
h_matrix = fi.pairwise(X_test)          # DataFrame, (n_features, n_features)
fi.plot_heatmap(X_test)                  # colour-coded interaction matrix

# Top-k interacting pairs
top_pairs = fi.top_pairs(X_test, k=5)
print(top_pairs)
# feature_a  feature_b  h_statistic
# age        income       0.34
# ...
```

---

## Counterfactual Explanations

`CounterfactualExplainer` generates the smallest-change perturbation of an
input that flips the model prediction to a desired outcome class. The
implementation is based on DiCE (Diverse Counterfactual Explanations).

```python
from endgame.explain import CounterfactualExplainer

cf = CounterfactualExplainer(
    model,
    feature_names=X_train.columns.tolist(),
    categorical_features=['gender', 'education'],
)
cf.fit(X_train)

# Generate counterfactuals for one instance
counterfactuals = cf.explain(
    X_test[0],
    desired_class=1,         # the outcome we want
    n_counterfactuals=5,     # number of diverse alternatives
    proximity_weight=0.5,    # penalise large changes
    diversity_weight=1.0,    # encourage variety
)
print(counterfactuals.as_dataframe())

# Feature ranges can be constrained to realistic values
cf_constrained = CounterfactualExplainer(
    model,
    feature_names=X_train.columns.tolist(),
    feature_ranges={'age': (18, 80), 'income': (0, 500_000)},
    immutable_features=['gender'],
)
cf_constrained.fit(X_train)
result = cf_constrained.explain(X_test[0], desired_class=1)
```

---

## Comparing Multiple Explanation Methods

Use `explain()` with `method='all'` to run every applicable method and return
a side-by-side comparison:

```python
from endgame.explain import explain

report = explain(model, X_test, method='all', n_samples=100)
report.compare()    # prints agreement metrics across methods
report.plot_all()   # grid of summary plots
```

---

## Saving Explanation Reports

```python
report = explain(model, X_test)
report.save_html('explanation_report.html')   # standalone HTML with plots
report.save_json('explanation_report.json')   # feature importances as JSON
```

---

## API Reference

Full parameter documentation for `explain`, `SHAPExplainer`, `LIMEExplainer`,
`PartialDependence`, `FeatureInteraction`, and `CounterfactualExplainer` is
available in the auto-generated API reference at `docs/api/explain.rst` or by
calling `help(ClassName)` at the Python prompt.
