"""Explainability module: unified API for SHAP, LIME, PDP, interactions, and counterfactuals.

Provides a consistent interface for model explanation methods with automatic
method selection based on model type.

Example
-------
>>> import endgame as eg
>>> from endgame.explain import explain, SHAPExplainer, LIMEExplainer
>>>
>>> # One-line explanation with auto-detection
>>> explanation = explain(model, X_test)
>>> explanation.plot(kind='bar')
>>> df = explanation.to_dataframe()
>>>
>>> # Explicit SHAP
>>> shap_exp = SHAPExplainer(model)
>>> explanation = shap_exp.explain(X_test)
>>>
>>> # Explicit LIME (local)
>>> lime_exp = LIMEExplainer(model)
>>> explanation = lime_exp.explain(X_test[:1])
>>>
>>> # Partial dependence
>>> from endgame.explain import PartialDependence
>>> pdp = PartialDependence(model)
>>> explanation = pdp.explain(X_train, features=[0, 1, 2])
>>>
>>> # Feature interactions
>>> from endgame.explain import FeatureInteraction
>>> fi = FeatureInteraction(model)
>>> explanation = fi.explain(X_train)
>>>
>>> # Counterfactuals
>>> from endgame.explain import CounterfactualExplainer
>>> cf = CounterfactualExplainer(model, X_train)
>>> explanation = cf.explain(X_test[:1], desired_class=1)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator

from endgame.explain._base import BaseExplainer, Explanation
from endgame.explain.counterfactual import CounterfactualExplainer
from endgame.explain.interaction import FeatureInteraction
from endgame.explain.lime_explainer import LIMEExplainer
from endgame.explain.pdp import PartialDependence
from endgame.explain.shap_explainer import SHAPExplainer


def explain(
    model: BaseEstimator,
    X: np.ndarray,
    *,
    method: Literal["auto", "shap", "lime", "pdp"] = "auto",
    feature_names: list[str] | None = None,
    random_state: int | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> Explanation:
    """One-line model explanation with automatic method selection.

    Selects the most appropriate explanation method based on the model
    type and data characteristics:

    - **Tree-based models** (LightGBM, XGBoost, CatBoost, RandomForest,
      etc.) -> ``SHAPExplainer`` with ``TreeExplainer``.
    - **Linear models** (LogisticRegression, Ridge, Lasso, etc.) ->
      ``SHAPExplainer`` with ``LinearExplainer``.
    - **Other models** with ``n_samples <= 500`` ->
      ``SHAPExplainer`` with ``KernelExplainer``.
    - **Other models** with ``n_samples > 500`` ->
      ``SHAPExplainer`` with ``KernelExplainer`` (subsampled).

    Parameters
    ----------
    model : sklearn-compatible estimator
        A fitted model.
    X : array-like of shape (n_samples, n_features)
        Data to explain.
    method : str, default='auto'
        Explanation method:
        - ``'auto'``: Auto-select (see above).
        - ``'shap'``: Force SHAP.
        - ``'lime'``: Force LIME.
        - ``'pdp'``: Partial dependence.
    feature_names : list of str, optional
        Feature names.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Verbose output.
    **kwargs
        Forwarded to the underlying explainer's ``explain()`` method.

    Returns
    -------
    Explanation
        The computed explanation.

    Examples
    --------
    >>> from endgame.explain import explain
    >>> explanation = explain(model, X_test)
    >>> explanation.plot(kind='bar')
    >>> print(explanation.top_features(5))
    >>> df = explanation.to_dataframe()
    """
    if method == "auto" or method == "shap":
        # For SHAP, auto-detect the best explainer type and set
        # reasonable max_samples for large datasets.
        shap_kwargs: dict[str, Any] = {
            "feature_names": feature_names,
            "random_state": random_state,
            "verbose": verbose,
        }

        X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
        n_samples = X_arr.shape[0] if X_arr.ndim >= 2 else 1

        # Auto-set max_samples for expensive explainer types to keep
        # runtime manageable.
        from endgame.explain.shap_explainer import _detect_explainer_type
        detected = _detect_explainer_type(model)
        if detected == "kernel" and n_samples > 500:
            shap_kwargs.setdefault("max_samples", 500)

        explainer = SHAPExplainer(model, **shap_kwargs)
        return explainer.explain(X, **kwargs)

    elif method == "lime":
        explainer = LIMEExplainer(
            model,
            feature_names=feature_names,
            random_state=random_state,
            verbose=verbose,
        )
        return explainer.explain(X, **kwargs)

    elif method == "pdp":
        pdp = PartialDependence(
            model,
            feature_names=feature_names,
            random_state=random_state,
            verbose=verbose,
        )
        return pdp.explain(X, **kwargs)

    else:
        raise ValueError(
            f"Unknown explanation method '{method}'. "
            "Supported: 'auto', 'shap', 'lime', 'pdp'."
        )


__all__ = [
    # Convenience function
    "explain",
    # Core
    "Explanation",
    "BaseExplainer",
    # Explainers
    "SHAPExplainer",
    "LIMEExplainer",
    "PartialDependence",
    "FeatureInteraction",
    "CounterfactualExplainer",
]
