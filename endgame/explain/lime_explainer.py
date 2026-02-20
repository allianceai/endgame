"""LIME-based model explanations.

Wraps the ``lime`` library to provide local interpretable explanations
for individual predictions.

Example
-------
>>> from endgame.explain import LIMEExplainer
>>> explainer = LIMEExplainer(model, feature_names=feature_names)
>>> explanation = explainer.explain(X_test[:5])
>>> explanation.plot(kind='bar')
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator

from endgame.explain._base import BaseExplainer, Explanation


def _check_lime_installed() -> None:
    """Raise ImportError if the lime package is unavailable."""
    try:
        import lime  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'lime' package is required for LIMEExplainer. "
            "Install it with: pip install lime"
        )


class LIMEExplainer(BaseExplainer):
    """LIME (Local Interpretable Model-agnostic Explanations) wrapper.

    Generates local explanations by fitting an interpretable model in
    the neighbourhood of each instance.

    Parameters
    ----------
    model : sklearn-compatible estimator
        A fitted model with ``predict`` (regression) or
        ``predict_proba`` (classification).
    mode : str, default='auto'
        Explanation mode:
        - ``'auto'``: Infer from the model (uses ``predict_proba`` when
          available).
        - ``'classification'``: Force classification mode.
        - ``'regression'``: Force regression mode.
    num_features : int, default=10
        Maximum number of features in each local explanation.
    num_samples : int, default=5000
        Size of the neighbourhood sample used by LIME.
    kernel_width : float, optional
        Width of the exponential kernel.  If ``None``, LIME uses its
        default (``sqrt(n_features) * 0.75``).
    discretize_continuous : bool, default=True
        Whether to discretize continuous features (LIME default).
    feature_names : list of str, optional
        Feature names.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Verbose output.

    Examples
    --------
    >>> from endgame.explain import LIMEExplainer
    >>> explainer = LIMEExplainer(model, mode='classification')
    >>> explanation = explainer.explain(X_test[:10])
    >>> explanation.plot(kind='bar')
    """

    def __init__(
        self,
        model: BaseEstimator,
        mode: Literal["auto", "classification", "regression"] = "auto",
        num_features: int = 10,
        num_samples: int = 5000,
        kernel_width: float | None = None,
        discretize_continuous: bool = True,
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
        self.mode = mode
        self.num_features = num_features
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.discretize_continuous = discretize_continuous

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_mode(self) -> str:
        """Determine classification vs regression mode."""
        if self.mode != "auto":
            return self.mode

        if hasattr(self.model, "predict_proba"):
            return "classification"
        return "regression"

    def _build_lime_explainer(self, X: np.ndarray, mode: str) -> Any:
        """Construct the LIME ``LimeTabularExplainer``.

        Parameters
        ----------
        X : np.ndarray
            Training / background data for LIME.
        mode : str
            ``'classification'`` or ``'regression'``.

        Returns
        -------
        lime.lime_tabular.LimeTabularExplainer
        """
        from lime.lime_tabular import LimeTabularExplainer

        names = self._resolve_feature_names(X)

        kwargs: dict[str, Any] = {
            "training_data": X,
            "feature_names": names,
            "mode": mode,
            "discretize_continuous": self.discretize_continuous,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }
        if self.kernel_width is not None:
            kwargs["kernel_width"] = self.kernel_width

        return LimeTabularExplainer(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        X: np.ndarray,
        *,
        labels: tuple | None = None,
        top_labels: int | None = None,
    ) -> Explanation:
        """Generate LIME explanations for the rows of *X*.

        When *X* has a single row, the explanation values have shape
        ``(n_features,)``.  For multiple rows, the values have shape
        ``(n_samples, n_features)`` where each row is the local weight
        vector for that instance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Instances to explain.
        labels : tuple of int, optional
            Class labels to explain (classification only).  Defaults to
            the predicted class.
        top_labels : int, optional
            Number of top predicted classes to explain.

        Returns
        -------
        Explanation
            An :class:`Explanation` with local attribution values.
        """
        _check_lime_installed()

        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        names = self._resolve_feature_names(X)
        mode = self._resolve_mode()
        lime_exp = self._build_lime_explainer(X, mode)

        predict_fn = (
            self.model.predict_proba
            if mode == "classification" and hasattr(self.model, "predict_proba")
            else self.model.predict
        )

        n_samples, n_features = X.shape
        all_weights = np.zeros((n_samples, n_features))

        self._log(f"Computing LIME explanations for {n_samples} samples ...")

        for i in range(n_samples):
            explain_kwargs: dict[str, Any] = {
                "data_row": X[i],
                "predict_fn": predict_fn,
                "num_features": min(self.num_features, n_features),
                "num_samples": self.num_samples,
            }
            if mode == "classification":
                if labels is not None:
                    explain_kwargs["labels"] = labels
                if top_labels is not None:
                    explain_kwargs["top_labels"] = top_labels

            exp = lime_exp.explain_instance(**explain_kwargs)

            # Collect the feature weights into a dense array.
            if mode == "classification":
                # Use the first label's weights.
                label_key = (
                    labels[0]
                    if labels is not None
                    else exp.available_labels()[0]
                )
                weight_map = dict(exp.as_list(label=label_key))
            else:
                weight_map = dict(exp.as_list())

            for fname, weight in weight_map.items():
                # LIME returns discretized feature descriptions; map back
                # to feature index by matching the original name prefix.
                for idx, orig_name in enumerate(names):
                    if fname.startswith(orig_name) or orig_name in fname:
                        all_weights[i, idx] = weight
                        break

        # Squeeze single-instance results.
        values = all_weights[0] if n_samples == 1 else all_weights

        return Explanation(
            values=values,
            base_value=None,
            feature_names=names,
            method="lime",
            metadata={
                "mode": mode,
                "num_features": self.num_features,
                "num_samples": self.num_samples,
                "n_instances": n_samples,
            },
        )
