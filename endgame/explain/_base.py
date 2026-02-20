"""Base classes for the explainability module.

Provides the ``Explanation`` dataclass for storing and visualizing explanation
results, and ``BaseExplainer`` as the abstract base class for all explainers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.base import BaseEstimator

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class Explanation:
    """Container for feature-attribution explanations.

    Holds SHAP values, LIME weights, partial dependence grids, or any
    other per-feature attribution output. Provides convenience methods
    for plotting and DataFrame export.

    Attributes
    ----------
    values : np.ndarray
        Attribution values.  Shape depends on the method:
        - Global: ``(n_features,)`` (mean absolute attributions).
        - Local:  ``(n_samples, n_features)`` or ``(n_features,)`` for a
          single instance.
        - PDP:    ``(n_grid_points,)`` or ``(n_grid_1, n_grid_2)`` for 2-D.
    base_value : float or np.ndarray or None
        Expected value / base prediction the attributions are relative to.
    feature_names : list of str or None
        Names corresponding to the feature axis of *values*.
    method : str
        Name of the explanation method (e.g. ``'shap'``, ``'lime'``).
    metadata : dict
        Arbitrary extra information (explainer params, timings, etc.).

    Examples
    --------
    >>> explanation.plot(kind='bar')
    >>> df = explanation.to_dataframe()
    """

    values: np.ndarray
    base_value: float | np.ndarray | None = None
    feature_names: list[str] | None = None
    method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        kind: Literal[
            "bar", "beeswarm", "waterfall", "heatmap", "force",
        ] = "bar",
        max_display: int = 20,
        show: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot the explanation.

        Parameters
        ----------
        kind : str, default='bar'
            Plot type.  Supported values depend on the explanation method:
            - ``'bar'``: Horizontal bar chart of mean |attributions| (all methods).
            - ``'beeswarm'``: SHAP beeswarm summary plot (SHAP only).
            - ``'waterfall'``: Waterfall plot for a single prediction (SHAP only).
            - ``'heatmap'``: SHAP heatmap of sample-level attributions.
            - ``'force'``: SHAP force plot for a single prediction.
        max_display : int, default=20
            Maximum number of features to display.
        show : bool, default=True
            Whether to call ``matplotlib.pyplot.show()``.
        **kwargs
            Forwarded to the underlying plotting function.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure object when *show* is False.
        """
        import matplotlib.pyplot as plt

        vals = np.asarray(self.values)
        names = self.feature_names

        # Collapse to 1-D for bar plots when we have sample-level values.
        if kind == "bar":
            if vals.ndim == 2:
                importance = np.mean(np.abs(vals), axis=0)
            else:
                importance = np.abs(vals)

            n_features = len(importance)
            if names is None:
                names = [f"Feature {i}" for i in range(n_features)]

            order = np.argsort(importance)[::-1][:max_display]
            ordered_names = [names[int(i)] for i in order]
            ordered_vals = importance[order]

            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 0.4 * max_display + 1)))
            y_pos = np.arange(len(ordered_names))
            ax.barh(y_pos, ordered_vals, align="center", color=kwargs.pop("color", "#1f77b4"))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(ordered_names)
            ax.invert_yaxis()
            ax.set_xlabel("Mean |Attribution|")
            ax.set_title(f"Feature Attributions ({self.method})")
            plt.tight_layout()

            if show:
                plt.show()
            return fig

        # SHAP-native plots (delegate to the shap library).
        if kind in ("beeswarm", "waterfall", "heatmap", "force"):
            try:
                import shap as shap_lib
            except ImportError:
                raise ImportError(
                    f"Plot kind='{kind}' requires the 'shap' package. "
                    "Install it with: pip install shap"
                )

            shap_explanation = shap_lib.Explanation(
                values=vals,
                base_values=self.base_value,
                feature_names=names,
            )

            if kind == "beeswarm":
                shap_lib.plots.beeswarm(shap_explanation, max_display=max_display, show=show, **kwargs)
            elif kind == "waterfall":
                if vals.ndim == 2:
                    shap_explanation = shap_explanation[0]
                shap_lib.plots.waterfall(shap_explanation, max_display=max_display, show=show, **kwargs)
            elif kind == "heatmap":
                shap_lib.plots.heatmap(shap_explanation, max_display=max_display, show=show, **kwargs)
            elif kind == "force":
                if vals.ndim == 2:
                    shap_explanation = shap_explanation[0]
                shap_lib.plots.force(shap_explanation, **kwargs)

            return None

        raise ValueError(
            f"Unknown plot kind '{kind}'. "
            "Supported: 'bar', 'beeswarm', 'waterfall', 'heatmap', 'force'."
        )

    # ------------------------------------------------------------------
    # DataFrame export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert explanation to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names as the index and attribution
            values as columns.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pandas"
            )

        vals = np.asarray(self.values)
        n_features = vals.shape[-1] if vals.ndim >= 2 else vals.shape[0]
        names = self.feature_names or [f"Feature {i}" for i in range(n_features)]

        if vals.ndim == 1:
            return pd.DataFrame({"attribution": vals}, index=names)

        # Sample-level attributions: summary with mean |attribution|.
        return pd.DataFrame(
            {"mean_abs_attribution": np.mean(np.abs(vals), axis=0)},
            index=names,
        ).sort_values("mean_abs_attribution", ascending=False)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def top_features(self, n: int = 10) -> list[str]:
        """Return the top-*n* most important feature names.

        Parameters
        ----------
        n : int, default=10
            Number of top features to return.

        Returns
        -------
        list of str
            Feature names ordered by descending mean |attribution|.
        """
        vals = np.asarray(self.values)
        if vals.ndim == 2:
            importance = np.mean(np.abs(vals), axis=0)
        else:
            importance = np.abs(vals)

        n_features = len(importance)
        names = self.feature_names or [f"Feature {i}" for i in range(n_features)]
        order = np.argsort(importance)[::-1][:n]
        return [names[int(i)] for i in order]

    def __repr__(self) -> str:
        shape = np.asarray(self.values).shape
        return (
            f"Explanation(method='{self.method}', shape={shape}, "
            f"n_features={shape[-1] if len(shape) >= 1 else 0})"
        )


class BaseExplainer(ABC):
    """Abstract base class for all Endgame explainers.

    Subclasses must implement :meth:`explain` which returns an
    :class:`Explanation` object.

    Parameters
    ----------
    model : sklearn-compatible estimator
        A fitted model to explain.
    feature_names : list of str, optional
        Feature names.  If ``None``, generic names are generated.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose logging.
    """

    def __init__(
        self,
        model: BaseEstimator,
        feature_names: list[str] | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.model = model
        self.feature_names = feature_names
        self.random_state = random_state
        self.verbose = verbose

    @abstractmethod
    def explain(self, X: np.ndarray, **kwargs: Any) -> Explanation:
        """Generate an explanation for the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to explain.
        **kwargs
            Method-specific arguments.

        Returns
        -------
        Explanation
            The computed explanation.
        """

    def _resolve_feature_names(self, X: np.ndarray) -> list[str]:
        """Return feature names, generating defaults if necessary.

        Parameters
        ----------
        X : np.ndarray
            Input array used to infer the number of features.

        Returns
        -------
        list of str
            Feature names of length ``X.shape[1]``.
        """
        if self.feature_names is not None:
            return list(self.feature_names)

        # Try to pull names from the data itself (pandas / polars).
        if hasattr(X, "columns"):
            return list(X.columns)

        n_features = X.shape[1] if X.ndim >= 2 else X.shape[0]
        return [f"Feature {i}" for i in range(n_features)]

    def _to_numpy(self, X: Any) -> np.ndarray:
        """Convert input to a numpy array.

        Parameters
        ----------
        X : array-like
            Input data (numpy, pandas, or polars).

        Returns
        -------
        np.ndarray
            Numpy array.
        """
        if isinstance(X, np.ndarray):
            return X

        try:
            import pandas as pd
            if isinstance(X, (pd.DataFrame, pd.Series)):
                return X.values
        except ImportError:
            pass

        try:
            import polars as pl
            if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
                if isinstance(X, pl.LazyFrame):
                    X = X.collect()
                return X.to_numpy()
        except ImportError:
            pass

        return np.asarray(X)

    def _log(self, message: str) -> None:
        """Print a message when verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {self.__class__.__name__}: {message}")

    def __repr__(self) -> str:
        model_name = type(self.model).__name__
        return f"{self.__class__.__name__}(model={model_name})"
