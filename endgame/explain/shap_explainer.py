"""SHAP-based model explanations with automatic explainer selection.

Wraps the ``shap`` library and auto-detects the most efficient explainer
(Tree, Linear, Deep, Kernel) based on the model type.

Example
-------
>>> from endgame.explain import SHAPExplainer
>>> explainer = SHAPExplainer(model)
>>> explanation = explainer.explain(X_test)
>>> explanation.plot(kind='bar')
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator

from endgame.explain._base import BaseExplainer, Explanation

# Model-name substrings used for auto-detection (shared with
# endgame.feature_selection.importance.shap_importance).
_TREE_MODEL_NAMES = frozenset([
    "lgbm", "xgb", "catboost", "randomforest", "gradientboosting",
    "extratrees", "decisiontree", "histgradientboosting",
    "rotationforest", "isolationforest",
])

_LINEAR_MODEL_NAMES = frozenset([
    "linear", "logistic", "ridge", "lasso", "elasticnet", "sgd",
])

_DEEP_MODEL_NAMES = frozenset([
    "fttransformer", "saint", "node", "tabnet", "nam", "gandalf",
    "talr", "tabularresnet", "tabtransformer", "embeddingmlp",
])


def _check_shap_installed() -> None:
    """Raise ImportError if the shap package is unavailable."""
    try:
        import shap  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'shap' package is required for SHAPExplainer. "
            "Install with: pip install endgame-ml[explain]"
        )


def _detect_explainer_type(model: Any) -> str:
    """Auto-detect the best SHAP explainer type for *model*.

    Parameters
    ----------
    model : estimator
        A fitted model.

    Returns
    -------
    str
        One of ``'tree'``, ``'linear'``, ``'deep'``, ``'kernel'``.
    """
    name = type(model).__name__.lower()

    # Unwrap common wrappers.
    inner = getattr(model, "model_", None) or getattr(model, "estimator_", None)
    inner_name = type(inner).__name__.lower() if inner is not None else ""

    for candidate in (name, inner_name):
        if any(t in candidate for t in _TREE_MODEL_NAMES):
            return "tree"
        if any(t in candidate for t in _LINEAR_MODEL_NAMES):
            return "linear"
        if any(t in candidate for t in _DEEP_MODEL_NAMES):
            return "deep"

    # Pipeline: inspect the last step.
    if hasattr(model, "steps"):
        last_step = model.steps[-1][1]
        return _detect_explainer_type(last_step)

    return "kernel"


class SHAPExplainer(BaseExplainer):
    """SHAP-based explainer with automatic backend selection.

    Supports ``TreeExplainer``, ``LinearExplainer``, ``DeepExplainer``,
    and ``KernelExplainer``.  By default the most efficient backend is
    chosen automatically based on the model type.

    Parameters
    ----------
    model : sklearn-compatible estimator
        A fitted model.
    explainer_type : str, default='auto'
        SHAP explainer backend:
        - ``'auto'``: Auto-detect from model type.
        - ``'tree'``: ``shap.TreeExplainer`` (tree-based models).
        - ``'linear'``: ``shap.LinearExplainer`` (linear models).
        - ``'deep'``: ``shap.DeepExplainer`` (neural networks).
        - ``'kernel'``: ``shap.KernelExplainer`` (model-agnostic).
    background_samples : int, default=100
        Number of background samples for Kernel / Linear / Deep explainers.
    max_samples : int, optional
        If set, subsample *X* to at most this many rows before computing
        SHAP values (useful for large datasets with KernelExplainer).
    check_additivity : bool, default=False
        Whether to verify the SHAP additivity property.
    feature_names : list of str, optional
        Feature names.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Verbose output.

    Examples
    --------
    >>> from endgame.explain import SHAPExplainer
    >>> explainer = SHAPExplainer(model, explainer_type='auto')
    >>> explanation = explainer.explain(X_test)
    >>> explanation.plot(kind='beeswarm')
    >>> print(explanation.top_features(5))
    """

    def __init__(
        self,
        model: BaseEstimator,
        explainer_type: Literal["auto", "tree", "linear", "deep", "kernel"] = "auto",
        background_samples: int = 100,
        max_samples: int | None = None,
        check_additivity: bool = False,
        feature_names: list[str] | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            model=model,
            feature_names=feature_names,
            random_state=random_state,
            verbose=verbose,
        )
        self.explainer_type = explainer_type
        self.background_samples = background_samples
        self.max_samples = max_samples
        self.check_additivity = check_additivity

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_type(self) -> str:
        """Resolve the effective explainer type string."""
        if self.explainer_type == "auto":
            detected = _detect_explainer_type(self.model)
            self._log(f"Auto-detected SHAP explainer type: {detected}")
            return detected
        return self.explainer_type

    def _build_explainer(self, X: np.ndarray, explainer_type: str) -> Any:
        """Construct the underlying ``shap.Explainer`` instance.

        Parameters
        ----------
        X : np.ndarray
            Data used to derive background summaries when needed.
        explainer_type : str
            One of ``'tree'``, ``'linear'``, ``'deep'``, ``'kernel'``.

        Returns
        -------
        shap explainer instance
        """
        import shap

        rng = np.random.RandomState(self.random_state)
        n_bg = min(self.background_samples, len(X))

        if explainer_type == "tree":
            # Unwrap endgame wrappers so SHAP sees the native model
            model = getattr(self.model, "model_", self.model)
            return shap.TreeExplainer(model)

        elif explainer_type == "linear":
            idx = rng.choice(len(X), size=n_bg, replace=False)
            return shap.LinearExplainer(self.model, X[idx])

        elif explainer_type == "deep":
            idx = rng.choice(len(X), size=n_bg, replace=False)
            return shap.DeepExplainer(self.model, X[idx])

        elif explainer_type == "kernel":
            idx = rng.choice(len(X), size=n_bg, replace=False)
            predict_fn = (
                self.model.predict_proba
                if hasattr(self.model, "predict_proba")
                else self.model.predict
            )
            return shap.KernelExplainer(predict_fn, X[idx])

        raise ValueError(
            f"Unknown explainer_type '{explainer_type}'. "
            "Expected one of: 'tree', 'linear', 'deep', 'kernel'."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        X: np.ndarray,
        *,
        check_additivity: bool | None = None,
    ) -> Explanation:
        """Compute SHAP values for *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to explain.
        check_additivity : bool, optional
            Override the instance-level ``check_additivity`` setting.

        Returns
        -------
        Explanation
            An :class:`Explanation` containing sample-level SHAP values
            of shape ``(n_samples, n_features)``.
        """
        _check_shap_installed()

        X = self._to_numpy(X)
        names = self._resolve_feature_names(X)
        check_add = check_additivity if check_additivity is not None else self.check_additivity

        # Optional subsampling for expensive explainers.
        if self.max_samples is not None and len(X) > self.max_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X), size=self.max_samples, replace=False)
            X_explain = X[idx]
        else:
            X_explain = X

        explainer_type = self._resolve_type()
        explainer = self._build_explainer(X, explainer_type)

        self._log(f"Computing SHAP values for {X_explain.shape[0]} samples ...")

        # Only TreeExplainer supports check_additivity.
        shap_kwargs: dict[str, Any] = {}
        if explainer_type == "tree":
            shap_kwargs["check_additivity"] = check_add

        shap_values = explainer.shap_values(X_explain, **shap_kwargs)

        # Multi-class handling: collapse the class dimension.
        # Older shap versions return a list of (n_samples, n_features) arrays.
        # Newer shap versions return (n_samples, n_features, n_classes).
        if isinstance(shap_values, list):
            shap_values = np.mean(np.abs(np.array(shap_values)), axis=0)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # (n_samples, n_features, n_classes) -> mean |values| over classes.
            shap_values = np.mean(np.abs(shap_values), axis=2)

        # Extract base value.
        base_value = getattr(explainer, "expected_value", None)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = np.asarray(base_value)
            if base_value.ndim > 0 and len(base_value) > 1:
                # Multi-class: take mean across classes.
                base_value = float(np.mean(base_value))
            else:
                base_value = float(base_value.flat[0])

        return Explanation(
            values=np.asarray(shap_values),
            base_value=base_value,
            feature_names=names,
            method="shap",
            metadata={
                "explainer_type": explainer_type,
                "n_samples": X_explain.shape[0],
                "n_features": X_explain.shape[1],
                "check_additivity": check_add,
                "background_samples": self.background_samples,
            },
        )

    def explain_interaction(self, X: np.ndarray) -> Explanation:
        """Compute SHAP interaction values (tree models only).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to explain.

        Returns
        -------
        Explanation
            An :class:`Explanation` with ``values`` of shape
            ``(n_samples, n_features, n_features)``.

        Raises
        ------
        ValueError
            If the model is not tree-based.
        """
        _check_shap_installed()
        import shap

        X = self._to_numpy(X)
        names = self._resolve_feature_names(X)

        explainer_type = self._resolve_type()
        if explainer_type != "tree":
            raise ValueError(
                "SHAP interaction values are only supported for tree-based models. "
                f"Detected explainer type: '{explainer_type}'."
            )

        model = getattr(self.model, "model_", self.model)
        explainer = shap.TreeExplainer(model)
        interaction_values = explainer.shap_interaction_values(X)

        # Multi-class handling: collapse the class dimension.
        # Older shap: list of (n_samples, n_features, n_features).
        # Newer shap: (n_samples, n_features, n_features, n_classes).
        if isinstance(interaction_values, list):
            interaction_values = np.mean(
                np.abs(np.array(interaction_values)), axis=0
            )
        elif isinstance(interaction_values, np.ndarray) and interaction_values.ndim == 4:
            interaction_values = np.mean(np.abs(interaction_values), axis=3)

        return Explanation(
            values=np.asarray(interaction_values),
            base_value=getattr(explainer, "expected_value", None),
            feature_names=names,
            method="shap_interaction",
            metadata={"n_samples": X.shape[0], "n_features": X.shape[1]},
        )
