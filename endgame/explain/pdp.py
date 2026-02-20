"""Partial Dependence Plots (1-D and 2-D).

Computes the marginal effect of one or two features on the model's
predictions, using sklearn's ``partial_dependence`` under the hood.

Example
-------
>>> from endgame.explain import PartialDependence
>>> pdp = PartialDependence(model, feature_names=feature_names)
>>> explanation = pdp.explain(X_train, features=[0, 1])
>>> explanation.plot()
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator

from endgame.explain._base import BaseExplainer, Explanation


class PartialDependence(BaseExplainer):
    """Partial Dependence computation for 1-D and 2-D feature effects.

    Uses :func:`sklearn.inspection.partial_dependence` when available,
    with a brute-force fallback for models that do not expose the
    ``predict`` contract sklearn expects.

    Parameters
    ----------
    model : sklearn-compatible estimator
        A fitted model with ``predict`` (regression) or
        ``predict_proba`` (classification).
    grid_resolution : int, default=50
        Number of evenly-spaced grid points along each feature axis.
    percentiles : tuple of float, default=(0.05, 0.95)
        Lower and upper percentile bounds of the grid.
    kind : str, default='average'
        ``'average'`` for the marginal expectation (classic PDP), or
        ``'individual'`` for Individual Conditional Expectation (ICE).
    feature_names : list of str, optional
        Feature names.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Verbose output.

    Examples
    --------
    >>> pdp = PartialDependence(model)
    >>> # 1-D partial dependence for feature 0
    >>> exp1d = pdp.explain(X_train, features=[0])
    >>> exp1d.plot()
    >>>
    >>> # 2-D partial dependence for feature pair (0, 1)
    >>> exp2d = pdp.explain(X_train, features=[(0, 1)])
    >>> exp2d.plot()
    """

    def __init__(
        self,
        model: BaseEstimator,
        grid_resolution: int = 50,
        percentiles: tuple[float, float] = (0.05, 0.95),
        kind: Literal["average", "individual"] = "average",
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
        self.grid_resolution = grid_resolution
        self.percentiles = percentiles
        self.kind = kind

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        X: np.ndarray,
        *,
        features: list[int | tuple[int, int]] | None = None,
        target_class: int | None = None,
    ) -> Explanation:
        """Compute partial dependence for the requested features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data used to marginalise over.
        features : list of int or list of (int, int), optional
            Feature indices to compute PDP for.  Each element is either a
            single int (1-D PDP) or a pair of ints (2-D PDP).  If
            ``None``, computes 1-D PDP for all features.
        target_class : int, optional
            For classifiers, the index of the class to explain.  Defaults
            to class 1 for binary classification.

        Returns
        -------
        Explanation
            An :class:`Explanation` whose ``values`` and ``metadata``
            contain PDP grids for each requested feature (set).
        """
        from sklearn.inspection import partial_dependence

        X = self._to_numpy(X)
        names = self._resolve_feature_names(X)

        if features is None:
            features = list(range(X.shape[1]))

        response_method = "auto"
        if hasattr(self.model, "predict_proba"):
            response_method = "predict_proba"

        results: dict[str, Any] = {}
        global_importance = np.zeros(X.shape[1])

        self._log(f"Computing partial dependence for {len(features)} feature(s) ...")

        for feat in features:
            # sklearn expects a list for the features argument.
            feat_tuple = feat if isinstance(feat, (tuple, list)) else [feat]

            pd_result = partial_dependence(
                self.model,
                X,
                features=feat_tuple,
                grid_resolution=self.grid_resolution,
                percentiles=self.percentiles,
                kind=self.kind,
                response_method=response_method,
            )

            # pd_result is a Bunch with 'average'/'individual' and 'grid_values'.
            if self.kind == "average":
                pdp_values = pd_result["average"]
            else:
                pdp_values = pd_result["individual"]

            grid_values = pd_result["grid_values"]

            # For classifiers with multiple classes, select target class.
            if pdp_values.ndim >= 2 and target_class is not None:
                pdp_values = pdp_values[target_class]
            elif pdp_values.ndim >= 2:
                # Default: class 1 for binary, or first class for multiclass.
                pdp_values = pdp_values[0]

            key = str(feat)
            results[key] = {
                "pdp_values": pdp_values,
                "grid_values": [np.asarray(g) for g in grid_values],
                "feature": feat,
            }

            # For 1-D features, measure importance as range of PDP.
            if isinstance(feat, int):
                pdp_range = float(np.ptp(pdp_values))
                global_importance[feat] = pdp_range

        return Explanation(
            values=global_importance,
            base_value=None,
            feature_names=names,
            method="pdp",
            metadata={
                "pdp_results": results,
                "kind": self.kind,
                "grid_resolution": self.grid_resolution,
                "percentiles": self.percentiles,
            },
        )

    def plot_feature(
        self,
        explanation: Explanation,
        feature: int | tuple[int, int],
        *,
        ax: Any | None = None,
        show: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot partial dependence for a single feature (or pair).

        Parameters
        ----------
        explanation : Explanation
            Result from :meth:`explain`.
        feature : int or (int, int)
            Feature index (or pair for 2-D).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.  If ``None``, a new figure is created.
        show : bool, default=True
            Whether to call ``plt.show()``.
        **kwargs
            Forwarded to matplotlib plotting functions.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        key = str(feature)
        pdp_data = explanation.metadata["pdp_results"].get(key)
        if pdp_data is None:
            raise ValueError(
                f"Feature {feature} not found in explanation. "
                f"Available: {list(explanation.metadata['pdp_results'].keys())}"
            )

        names = explanation.feature_names or []
        pdp_values = pdp_data["pdp_values"]
        grid_values = pdp_data["grid_values"]

        if isinstance(feature, (tuple, list)) and len(feature) == 2:
            # 2-D contour plot.
            fig, ax_ = plt.subplots(figsize=kwargs.pop("figsize", (8, 6))) if ax is None else (ax.figure, ax)
            XX, YY = np.meshgrid(grid_values[0], grid_values[1])
            ZZ = pdp_values.reshape(XX.shape) if pdp_values.ndim == 1 else pdp_values
            cs = ax_.contourf(XX, YY, ZZ, levels=20, cmap=kwargs.pop("cmap", "viridis"), **kwargs)
            fig.colorbar(cs, ax=ax_)
            f0_name = names[feature[0]] if feature[0] < len(names) else f"Feature {feature[0]}"
            f1_name = names[feature[1]] if feature[1] < len(names) else f"Feature {feature[1]}"
            ax_.set_xlabel(f0_name)
            ax_.set_ylabel(f1_name)
            ax_.set_title(f"2-D Partial Dependence: {f0_name} vs {f1_name}")
        else:
            # 1-D line plot.
            fig, ax_ = plt.subplots(figsize=kwargs.pop("figsize", (8, 5))) if ax is None else (ax.figure, ax)
            feat_idx = feature if isinstance(feature, int) else feature[0]
            f_name = names[feat_idx] if feat_idx < len(names) else f"Feature {feat_idx}"
            ax_.plot(grid_values[0], pdp_values.ravel(), linewidth=2, **kwargs)
            ax_.set_xlabel(f_name)
            ax_.set_ylabel("Partial Dependence")
            ax_.set_title(f"PDP: {f_name}")

        plt.tight_layout()
        if show:
            plt.show()
        return fig
