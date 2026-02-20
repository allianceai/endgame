"""Feature interaction detection via Friedman's H-statistic.

Measures the strength of pairwise (and optionally higher-order) feature
interactions using the decomposition of partial dependence.

Reference
---------
Friedman, J.H. & Popescu, B.E. (2008). "Predictive Learning via Rule
Ensembles." *Annals of Applied Statistics*, 2(3), 916-954.

Example
-------
>>> from endgame.explain import FeatureInteraction
>>> fi = FeatureInteraction(model, feature_names=feature_names)
>>> explanation = fi.explain(X_train, features=[(0, 1), (0, 2), (1, 2)])
>>> explanation.plot(kind='bar')
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from endgame.explain._base import BaseExplainer, Explanation


class FeatureInteraction(BaseExplainer):
    """Feature interaction strength via the H-statistic.

    The H-statistic measures the proportion of variance in the joint
    partial dependence that cannot be explained by the sum of the
    individual partial dependences.  A value of 0 means no interaction;
    values near 1 indicate strong interaction.

    Parameters
    ----------
    model : sklearn-compatible estimator
        A fitted model.
    grid_resolution : int, default=25
        Number of grid points per feature for PDP estimation.
        Kept lower than PDP defaults for speed since we evaluate many
        pairs.
    percentiles : tuple of float, default=(0.05, 0.95)
        Feature range bounds.
    feature_names : list of str, optional
        Feature names.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Verbose output.

    Examples
    --------
    >>> fi = FeatureInteraction(model)
    >>> explanation = fi.explain(X_train, features=[(0, 1), (2, 3)])
    >>> explanation.to_dataframe()
    """

    def __init__(
        self,
        model: BaseEstimator,
        grid_resolution: int = 25,
        percentiles: tuple[float, float] = (0.05, 0.95),
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

    # ------------------------------------------------------------------
    # H-statistic computation
    # ------------------------------------------------------------------

    def _partial_dependence_1d(
        self,
        X: np.ndarray,
        feature: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1-D partial dependence for a single feature.

        Returns
        -------
        tuple of (grid_values, pdp_values)
        """
        from sklearn.inspection import partial_dependence

        response_method = (
            "predict_proba"
            if hasattr(self.model, "predict_proba")
            else "auto"
        )

        result = partial_dependence(
            self.model,
            X,
            features=[feature],
            grid_resolution=self.grid_resolution,
            percentiles=self.percentiles,
            kind="average",
            response_method=response_method,
        )

        pdp_values = result["average"]
        # Handle multi-class: take first class.
        if pdp_values.ndim >= 2:
            pdp_values = pdp_values[0]

        return np.asarray(result["grid_values"][0]), pdp_values.ravel()

    def _partial_dependence_2d(
        self,
        X: np.ndarray,
        feature_pair: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 2-D partial dependence for a pair of features.

        Returns
        -------
        tuple of (grid_0, grid_1, pdp_values_2d)
        """
        from sklearn.inspection import partial_dependence

        response_method = (
            "predict_proba"
            if hasattr(self.model, "predict_proba")
            else "auto"
        )

        result = partial_dependence(
            self.model,
            X,
            features=list(feature_pair),
            grid_resolution=self.grid_resolution,
            percentiles=self.percentiles,
            kind="average",
            response_method=response_method,
        )

        pdp_values = result["average"]
        if pdp_values.ndim >= 3:
            pdp_values = pdp_values[0]

        grid_0 = np.asarray(result["grid_values"][0])
        grid_1 = np.asarray(result["grid_values"][1])

        return grid_0, grid_1, pdp_values

    def _h_statistic(
        self,
        X: np.ndarray,
        feature_i: int,
        feature_j: int,
    ) -> float:
        """Compute the pairwise H-statistic for features *i* and *j*.

        H^2_{ij} = sum( PDP_{ij}(x_i, x_j) - PDP_i(x_i) - PDP_j(x_j) )^2
                    / sum( PDP_{ij}(x_i, x_j)^2 )

        Parameters
        ----------
        X : np.ndarray
            Data for PDP computation.
        feature_i, feature_j : int
            Feature indices.

        Returns
        -------
        float
            H-statistic in [0, 1].
        """
        # 1-D PDPs (centred).
        grid_i, pdp_i = self._partial_dependence_1d(X, feature_i)
        grid_j, pdp_j = self._partial_dependence_1d(X, feature_j)

        # Centre the 1-D PDPs (subtract their mean).
        pdp_i_centered = pdp_i - np.mean(pdp_i)
        pdp_j_centered = pdp_j - np.mean(pdp_j)

        # 2-D PDP.
        _, _, pdp_ij = self._partial_dependence_2d(X, (feature_i, feature_j))

        # Centre the 2-D PDP.
        pdp_ij_centered = pdp_ij - np.mean(pdp_ij)

        # Construct the additive component on the joint grid.
        # pdp_ij has shape (len(grid_i), len(grid_j)).
        additive = pdp_i_centered[:, np.newaxis] + pdp_j_centered[np.newaxis, :]

        # Interaction residual.
        residual = pdp_ij_centered - additive

        numerator = np.sum(residual ** 2)
        denominator = np.sum(pdp_ij_centered ** 2)

        if denominator < 1e-12:
            return 0.0

        h_stat = float(numerator / denominator)
        # Clip to [0, 1] for numerical stability.
        return float(np.clip(h_stat, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        X: np.ndarray,
        *,
        features: list[tuple[int, int]] | None = None,
        top_k: int = 10,
    ) -> Explanation:
        """Compute pairwise H-statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data used for PDP computation.
        features : list of (int, int), optional
            Feature pairs to evaluate.  If ``None``, evaluates the
            ``top_k`` most important features (by variance of 1-D PDP)
            crossed with each other.
        top_k : int, default=10
            When *features* is ``None``, use the top-*k* features to
            form pairs.

        Returns
        -------
        Explanation
            An :class:`Explanation` with H-statistic values and metadata
            containing per-pair results.
        """
        X = self._to_numpy(X)
        names = self._resolve_feature_names(X)
        n_features = X.shape[1]

        if features is None:
            # Rank features by 1-D PDP variance, then take top-k pairs.
            variances = []
            for f in range(n_features):
                _, pdp_vals = self._partial_dependence_1d(X, f)
                variances.append(np.var(pdp_vals))

            top_indices = np.argsort(variances)[::-1][:min(top_k, n_features)]
            features = [
                (int(top_indices[i]), int(top_indices[j]))
                for i in range(len(top_indices))
                for j in range(i + 1, len(top_indices))
            ]

        self._log(f"Computing H-statistics for {len(features)} feature pairs ...")

        pair_results: dict[str, float] = {}
        interaction_matrix = np.zeros((n_features, n_features))

        for fi, fj in features:
            h = self._h_statistic(X, fi, fj)
            pair_name = f"{names[fi]} x {names[fj]}"
            pair_results[pair_name] = h
            interaction_matrix[fi, fj] = h
            interaction_matrix[fj, fi] = h
            self._log(f"  H({names[fi]}, {names[fj]}) = {h:.4f}")

        # Build a 1-D importance vector: max interaction per feature.
        feature_interaction_importance = np.max(interaction_matrix, axis=1)

        return Explanation(
            values=feature_interaction_importance,
            base_value=None,
            feature_names=names,
            method="h_statistic",
            metadata={
                "pair_results": pair_results,
                "interaction_matrix": interaction_matrix,
                "features_evaluated": features,
            },
        )

    def plot_interaction_matrix(
        self,
        explanation: Explanation,
        *,
        ax: Any | None = None,
        show: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot the pairwise interaction matrix as a heatmap.

        Parameters
        ----------
        explanation : Explanation
            Result from :meth:`explain`.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        show : bool, default=True
            Whether to call ``plt.show()``.
        **kwargs
            Forwarded to ``matplotlib.pyplot.imshow``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        matrix = explanation.metadata["interaction_matrix"]
        names = explanation.feature_names or [
            f"F{i}" for i in range(matrix.shape[0])
        ]

        # Only show features that appear in evaluated pairs.
        evaluated = explanation.metadata["features_evaluated"]
        involved = sorted(set(f for pair in evaluated for f in pair))
        sub_matrix = matrix[np.ix_(involved, involved)]
        sub_names = [names[i] for i in involved]

        fig, ax_ = (
            plt.subplots(figsize=kwargs.pop("figsize", (8, 7)))
            if ax is None
            else (ax.figure, ax)
        )

        im = ax_.imshow(
            sub_matrix,
            cmap=kwargs.pop("cmap", "YlOrRd"),
            aspect="auto",
            vmin=0,
            vmax=max(0.01, float(np.max(sub_matrix))),
            **kwargs,
        )
        fig.colorbar(im, ax=ax_, label="H-statistic")

        ax_.set_xticks(range(len(sub_names)))
        ax_.set_yticks(range(len(sub_names)))
        ax_.set_xticklabels(sub_names, rotation=45, ha="right")
        ax_.set_yticklabels(sub_names)
        ax_.set_title("Feature Interaction (H-statistic)")

        plt.tight_layout()
        if show:
            plt.show()
        return fig
